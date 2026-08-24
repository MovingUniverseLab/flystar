import pdb
import copy
import warnings
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from astropy.time import Time
from astropy.stats import sigma_clip
from astropy.table import Table, Column
from pandas.api.types import is_string_dtype
from collections.abc import Iterable
from flystar import motion_model

class StarTable(Table):
    def __init__(self, *args, ref_list=0, copy=True, **kwargs):
        """
        A StarTable is an astropy.Table with stars matched from multiple starlists.

        Required table columns (input as keywords)
        ------------------------------------------
        name : 1D numpy.array with shape = N_stars
            List of unique names for each of the stars in the table.

        x : 2D numpy.array with shape = (N_stars, N_lists)
            Positions of N_stars in each of N_lists in the x dimension.

        y : 2D numpy.array with shape = (N_stars, N_lists)
            Positions of N_stars in each of N_lists in the y dimension.

        m : 2D numpy.array with shape = (N_stars, N_lists)
            Magnitudes of N_stars in each of N_lists.

        Optional table columns (input as keywords)
        ------------------------------------------
        motion_model : 1D numpy.array with shape = N_stars
            string indicating motion model type for each star

        xe : 2D numpy.array with shape = (N_stars, N_lists)
            Position uncertainties of N_stars in each of N_lists in the x dimension.

        ye : 2D numpy.array with shape = (N_stars, N_lists)
            Position uncertainties of N_stars in each of N_lists in the y dimension.

        me : 2D numpy.array with shape = (N_stars, N_lists)
            Magnitude uncertainties of N_stars in each of N_lists.

        ep_name : 2D numpy.array with shape = (N_stars, N_lists)
            Names in each epoch for each of N_stars in each of N_lists. This is
            useful for tracking purposes.

        corr : 2D numpy.array with shape = (N_stars, N_lists)
            Fitting correlation for each of N_stars in each of N_lists.

        Optional table meta data
        -------------------------
        list_names : list of strings
            List of names, one for each of the starlists.

        list_times : list of integers or floats
            List of times/dates for each starlist, as decimal years in the
            UTC scale (e.g. 2025.0 for the start of 2025) -- i.e. an
            observation timestamp expressed as a fraction of a year. Models
            that need a uniform timescale convert internally; Parallax, for
            instance, converts UTC -> TDB before evaluating the ephemeris.

        ref_list : int
            Specify which list is the reference list (if any).

        copy : bool, optional
            If True (default), the table makes its own independent copy of
            every input array -- safe if the caller might mutate their
            arrays afterward. If False, arrays that are already a
            compatible ndarray are used directly without copying (they're
            still converted/copied if genuinely necessary, e.g. from a
            list or an incompatible dtype) -- only pass False when you
            know the caller won't touch these arrays again (e.g. they were
            just freshly built and not stored anywhere else), since the
            table's data would otherwise alias and mutating one would
            silently mutate the other.

        Examples
        --------------------------

        t = startables.StarTable(name=name, x=x, y=y, m=m)

        # Access the data:
        print(t)
        print(t['name'][0:10])  # print the first 10 star names
        print(t['x'][0:10, 0])  # print x from the first epoch/list/column for the first 10 stars
        """

        # Check if the required arguments are present
        arg_req = ('name', 'x', 'y', 'm')
        found_all_required = all(arg in kwargs for arg in arg_req)

        if not found_all_required:
            if len(args) > 1: # If there are no arguments, it's because the
                          # StarTable is being created as a copy. If there is
                          # only one, it's likely to be a copy with only some
                          # columns selected
                err_msg = "The StarTable class requires arguments: " + str(arg_req)
                warnings.warn(err_msg, UserWarning)
            Table.__init__(self, *args, copy=copy, **kwargs)
        else:
            # If we have errors, we need them in both dimensions.
            if ('xe' in kwargs) ^ ('ye' in kwargs):
                raise TypeError("The StarTable class requires both 'xe' and" +
                                " 'ye' arguments")
            # np.array(..., copy=True) (the default) always copies; np.asarray
            # only converts/copies when actually necessary (e.g. a list, or an
            # incompatible dtype) -- pass a caller-owned, already-correct
            # ndarray straight through with copy=False.
            array_convert = np.array if copy else np.asarray
            kwargs['name'] = array_convert(kwargs['name'])
            kwargs['x'] = array_convert(kwargs['x'])
            kwargs['y'] = array_convert(kwargs['y'])
            kwargs['m'] = array_convert(kwargs['m'])
            if ('xe' in kwargs) and ('ye' in kwargs):
                kwargs['xe'] = array_convert(kwargs['xe'])
                kwargs['ye'] = array_convert(kwargs['ye'])
            if 'me' in kwargs:
                kwargs['me'] = array_convert(kwargs['me'])

            # Figure out the shape
            n_stars = kwargs['x'].shape[0]
            n_lists = kwargs['x'].shape[1]

            # Check if the type and size of the arguments are correct.
            # Name checking: type and shape
            if len(kwargs['name']) != n_stars:
                err_msg += f"The 'name' argument length should be {n_stars}, but got {len(kwargs['name'])}."
                raise TypeError(err_msg)

            # Check all the 2D arrays.
            arg_tab = ('x', 'y', 'm', 'xe', 'ye', 'me', 'name_in_list')

            for arg_test in arg_tab:
                if arg_test in kwargs:
                    if not isinstance(kwargs[arg_test], np.ndarray):
                        err_msg = f"The '{arg_test}' argument has to be a numpy array, not {type(kwargs[arg_test])}"
                        raise TypeError(err_msg)

                    if kwargs[arg_test].shape != (n_stars, n_lists):
                        err_msg = f"The '{arg_test}' argument has to have shape = ({n_stars}, {n_lists}), but got {kwargs[arg_test].shape}"
                        raise TypeError(err_msg)

            # Check that the reference list is specified.
            if ref_list not in range(n_lists):
                err_msg = f"The 'ref_list' argument has to be an integer between 0 and {n_lists-1}"
                raise IndexError(err_msg)

            # We have to have special handling of meta-data (i.e. info that has
            # dimensions of n_lists).
            meta_tab = ('list_times', 'list_names')
            meta_type = ((float, int), str)
            for mtab, mtype in zip(meta_tab, meta_type):
                if mtab in kwargs:
                    kwargs[mtab] = list(kwargs[mtab])  # Convert to list, as astropy.Table doesn't like numpy arrays in meta-data.
                    if len(kwargs[mtab]) != n_lists:
                        raise ValueError(f"The '{mtab}' argument has to have length = {n_lists}")
                    if not all(isinstance(tt, mtype) for tt in kwargs[mtab]):
                        raise TypeError(f"The '{mtab}' argument has to be a list of {str(mtype)}.")

            #####
            # Create the startable
            #####
            # Pull the special meta-data args out of kwargs first, so the
            # column-building loop below doesn't see them.
            meta_updates = {}
            for meta_arg in meta_tab:
                if meta_arg in kwargs:
                    meta_updates[meta_arg] = kwargs.pop(meta_arg)
                elif meta_arg.upper() in kwargs:
                    meta_updates[meta_arg] = kwargs.pop(meta_arg.upper())

            # Build every column's (name, data) pair upfront and construct
            # the whole table in a single call, instead of constructing the
            # 4 required columns and then add_column()-ing the rest one at
            # a time. add_column() is dramatically more expensive per call
            # than passing every column to the constructor together
            # (confirmed empirically: ~1.2s and +4.6GB for ~29 columns
            # built via a loop of add_column() calls at ~1.4M rows, vs
            # ~0.001s and ~0GB for the exact same columns passed to the
            # constructor at once) -- almost certainly because add_column()
            # re-validates/re-indexes the whole table on every single call.
            all_col_names = ['name', 'x', 'y', 'm']
            all_col_data = [kwargs['name'], kwargs['x'], kwargs['y'], kwargs['m']]
            for arg in kwargs:
                if arg in ('name', 'x', 'y', 'm'):
                    continue
                data = kwargs[arg]
                if arg in ('name_in_list', 'motion_model_input', 'motion_model_used'):
                    width = 'U30' if arg == 'name_in_list' else 'U20'
                    data = np.asarray(data).astype(width, copy=copy)
                all_col_names.append(arg)
                all_col_data.append(data)

            super().__init__(tuple(all_col_data), names=tuple(all_col_names), copy=copy)
            self['name'] = self['name'].astype('U30')
            self.meta = {'n_stars': n_stars, 'n_lists': n_lists, 'ref_list': ref_list}
            self.meta.update(meta_updates)
            #if 'motion_model_input' not in kwargs:
            #    self['motion_model_input'] = np.repeat(self.default_motion_model, len(self['name']))

        return

    def add_starlist(self, warn_missing_meta=True, **kwargs):
        """
        Add data from a new list to an existing StarTable.
        Note, you can pass in the data via a StarList object or
        via a series of keywords with a 1D array on each.
        In either case, the number of stars must already match
        the existing number of stars in the StarTable.

        Examples
        --------
        Pass in data via a StarList object::

            print(t['x'].shape)
            t.add_starlist(starlist=my_list)
            print(t['x'].shape)   # Should be 1 column larger than before.

        Pass in data via keywords and 1D arrays::

            t.add_starlist(x=x_new, y=y_new, m=m_new)

        Parameters
        ----------
        warn_missing_meta : bool, optional
            Whether to warn when a per-list meta value (e.g. list_times)
            already tracked by this table isn't supplied by this call. Set
            to False when the caller knows that value will be set some other
            way (e.g. rebuilt in full immediately afterward), so the warning
            would just be noise about a value that was never meant to be
            given here. By default True.
        """
        # Check if we are dealing with a StarList object or a
        # set of arguments with individual arrays.
        if 'starlist' in kwargs:
            self._add_list_data_from_starlist(kwargs['starlist'])
        else:
            self._add_list_data_from_keywords(warn_missing_meta=warn_missing_meta, **kwargs)

        return

    def _add_list_data_from_starlist(self, starlist):
        # Loop through the 2D columns and add the new data to each.
        # If there is no input data for a particular column, then fill it with
        # zeros and mask it.
        for col_name in self.colnames:
            if len(self[col_name].data.shape) == 2:      # Find the 2D columns
                # Make a new 2D array with +1 extra column. Copy over the old data.
                # This is much faster than hstack or concatenate according to:
                # https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-an-numpy-array
                old_data = self[col_name].data
                old_type = self[col_name].info.dtype
                new_data = np.empty((old_data.shape[0], old_data.shape[1] + 1), dtype=old_type)
                new_data[:, :-1] = old_data

                # Save the new data array (with both old and new data in it) to the table.
                self[col_name] = new_data

                if (col_name in starlist.colnames):            # Add data if it was input
                    self[col_name][:, -1] = starlist[col_name]
                else:                               # Add junk data it if wasn't input
                    self._set_invalid_list_values(col_name, -1)

        # Special case for list_times: Update 't' column in startable
        if ('list_time' in starlist.meta):
            if 't' not in self.colnames:
                self.add_column(Column(data=np.full((len(self), 1), starlist.meta['list_time']), name='t'))
            else:
                old_data = self['t'].data
                old_type = self['t'].info.dtype
                new_data = np.empty((old_data.shape[0], old_data.shape[1] + 1), dtype=old_type)
                new_data[:, :-1] = old_data
                self['t'] = new_data
                self['t'][:, -1] = starlist.meta['list_time']

        ##########
        # Update the table meta-data. Remember that entries are lists not numpy arrays.
        ##########
        # Get the meta keynames in the startable and the input starlist
        tab_meta_keys = list(self.meta.keys())
        lis_meta_keys = list(starlist.meta.keys())
        # append 's' to the end to pluralize the input starlist.
        lis_meta_keys_plural = [lis_meta_key + 's' for lis_meta_key in lis_meta_keys]

        for kk in range(len(tab_meta_keys)):
            tab_key = tab_meta_keys[kk]

            # Meta table entries with a size that matches the n_lists size are the ones
            # that need a new value. We have to add something... whatever was passed in or None
            if isinstance(self.meta[tab_key], Iterable) and (len(self.meta[tab_key]) == self.meta['n_lists']) and (not isinstance(self.meta[tab_key], str)):
                # If we find the key in the starlists' meta argument, then add the new values.
                # Otherwise, add "None".
                self.meta[tab_key] = list(self.meta[tab_key])  # Convert to list, as astropy.Table doesn't like numpy arrays in meta-data.
                idx = lis_meta_keys_plural.index(tab_key) if tab_key in lis_meta_keys_plural else None
                if idx is not None:
                    lis_key = lis_meta_keys[idx]
                    self.meta[tab_key].append(starlist.meta[lis_key])
                else:
                    self._append_invalid_meta_values(tab_key)

        # Update the n_lists meta keyword.
        self.meta['n_lists'] += 1

        return


    def _add_list_data_from_keywords(self, warn_missing_meta=True, **kwargs):
        # # Check if the required arguments are present
        # arg_req = ('x', 'y', 'm')

        # for arg_test in arg_req:
        #     if arg_test not in kwargs:
        #         err_msg = "Added lists require a '{0:s}' argument"
        #         raise TypeError(err_msg.format(arg_test))

        # # If we have errors, we need them in both dimensions.
        # if ('xe' in kwargs) ^ ('ye' in kwargs):
        #     raise TypeError("Added lists with errors require both 'xe' and" +
        #                     " 'ye' arguments")

        # Loop through the 2D columns and add the new data to each.
        # If there is no input data for a particular column, then fill it with
        # zeros and mask it.
        for col_name in self.colnames:
            if (np.ndim(self[col_name].data) == 2) and (col_name not in ['detect', 'n_detect']):      # Find the 2D columns
                # Make a new 2D array with +1 extra column. Copy over the old data.
                # This is much faster than hstack or concatenate according to:
                # https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-an-numpy-array
                old_data = self[col_name].data
                old_type = self[col_name].info.dtype
                new_data = np.empty((old_data.shape[0], old_data.shape[1] + 1), dtype=old_type)
                new_data[:, :-1] = old_data

                # Save the new data array (with both old and new data in it) to the table.
                self[col_name] = new_data

                if (col_name in kwargs):            # Add data if it was input
                    self[col_name][:, -1] = kwargs[col_name]
                else:                               # Add junk data it if wasn't input
                    self._set_invalid_list_values(col_name, -1)


        # Update the table meta-data. Remember that entries are lists not numpy arrays.
        for key in self.meta.keys():
            # Meta table entries with a size that matches the n_lists size are the ones
            # that need a new value. We have to add something... whatever was passed in or None
            if isinstance(self.meta[key], Iterable) and (len(self.meta[key]) == self.meta['n_lists']) and (not isinstance(self.meta[key], str)):
                # If we find the key is the passed in meta argument, then add the new values.
                # Otherwise, add "None".
                self.meta[key] = list(self.meta[key])  # Convert to list, as astropy.Table doesn't like numpy arrays in meta-data.
                if 'meta' in kwargs:
                    new_meta_keys = kwargs['meta'].keys()
                    if key in new_meta_keys:
                        self.meta[key].append(kwargs['meta'][key])
                    else:
                        self._append_invalid_meta_values(key, warn=warn_missing_meta)
                else:
                    self._append_invalid_meta_values(key, warn=warn_missing_meta)

        # Update the n_lists meta keyword.
        self.meta['n_lists'] += 1

        return

    @staticmethod
    def _invalid_float_value(col_name):
        """
        The "no data" placeholder for a float column: np.inf for uncertainty
        columns (xe, ye, me, or anything ending in '_err'), np.nan for
        everything else (x, y, m, t, ...). Matches the convention already
        used by add_rows_for_new_stars() for brand-new rows -- without this,
        the exact same "never detected in this list" situation ends up as
        NaN or inf depending only on whether the row or the column existed
        first, not on what the data actually means.
        """
        if col_name in ('xe', 'ye', 'me') or col_name.endswith('_err'):
            return np.inf
        return np.nan

    def _set_invalid_list_values(self, col_name, col_idx):
        """
        Set the contents of the specified column (in the 2D column objects)
        to an invalide value depending on the data type.
        """
        if col_name == 'n_detect_list':
            # Unlike other int columns, 0 (not -1) is the correct "no data"
            # value here -- it lets n_detect (the aggregate) be computed as
            # a direct sum(n_detect_list, axis=1) instead of a masked sum
            # against x/y.
            self[col_name][:, col_idx] = 0
        elif np.issubdtype(self[col_name].info.dtype, np.integer):
            self[col_name][:, col_idx] = -1
        elif np.issubdtype(self[col_name].info.dtype, np.floating):
            self[col_name][:, col_idx] = self._invalid_float_value(col_name)
        else:
            self[col_name][:, col_idx] = None

        return

    def _set_invalid_star_values(self, col_name, row_idx):
        """
        Set the contents of the specified rows (in the 2D column objects)
        to an invalide value depending on the data type.
        """
        if col_name == 'n_detect_list':
            self[col_name][row_idx] = 0
        elif np.issubdtype(self[col_name].info.dtype, np.integer):
            self[col_name][row_idx] = -1
        elif np.issubdtype(self[col_name].info.dtype, np.floating):
            self[col_name][row_idx] = self._invalid_float_value(col_name)
        else:
            self[col_name][row_idx] = None

        return

    def _append_invalid_meta_values(self, key, warn=True):
        """
        For an existing meta keyword that is a list (already known),
        add an invalid value depending on the type.
        """
        if issubclass(type(self.meta[key][0]), np.integer):
            self.meta[key].append(-1)
        elif issubclass(type(self.meta[key][0]), np.floating):
            self.meta[key].append(np.nan)
        elif issubclass(type(self.meta[key][0]), str):
            self.meta[key].append('')
        else:
            self.meta[key].append(None)

        if warn:
            err_msg = "StarTable.add_starlist(): Missing meta keyword: {0:s}".format(key)
            warnings.warn(err_msg, UserWarning)

        return


    def get_starlist(self, list_index):
        """
        Return a StarList object for the specified list_index or epoch.

        Parameters
        ----------
        list_index : int
            The index of the list to fetch and return as a StarList object.
        """
        from flystar.starlists import StarList

        # Get the required arrays first.
        col_req_dict = {'name': None, 'x': None, 'y': None, 'm': None}
        col_req_names = col_req_dict.keys()

        for col_name in col_req_names:
            if len(self[col_name].data.shape) == 2:      # Find the 2D columns
                col_req_dict[col_name] = self[col_name][:, list_index]
            else:
                col_req_dict[col_name] = self[col_name]

        starlist = StarList(**col_req_dict)

        for col_name in self.colnames:
            if col_name in col_req_names:
                pass

            if len(self[col_name].data.shape) == 2:      # Find the 2D columns
                starlist[col_name] = self[col_name][:, list_index]
            else:
                starlist[col_name] = self[col_name]

        return starlist


    def combine_lists_xym(self, weighted_xy=True, weighted_m=True, mask_lists=None, sigma=3, select_stars=None,
                          absolute_sigma=True):
        """
        For x, y and m columns in the table, collapse along the lists
        direction. For 'x', 'y' this means calculating the average position with
        outlier rejection. Optionally, weight by the 'xe' and 'ye' individual
        uncertainties. Optionally, use sigma clipping.
        "mask_lists" is a list with the indices of starlists that are
        excluded from the combination.
        Also, count the number of times a star is found in starlists.

        select_stars : array-like of bool or int, optional
            If given, only (re)compute x0/y0/m0 (and errors) for these star
            rows; see combine_lists() for details. By default None (compute
            for all rows, same as before).
        absolute_sigma : bool, optional
            See combine_lists() for details. By default True.
        """

        # Combine by position
        if weighted_xy:
            weights_colx = 'xe'
            weights_coly = 'ye'
        else:
            weights_colx = None
            weights_coly = None

        if weighted_m:
            weights_colm = 'me'
        else:
            weights_colm = None

        self.combine_lists('x', weights_col=weights_colx, mask_lists=mask_lists, sigma=sigma, select_stars=select_stars,
                           absolute_sigma=absolute_sigma)
        self.combine_lists('y', weights_col=weights_coly, mask_lists=mask_lists, sigma=sigma, select_stars=select_stars,
                           absolute_sigma=absolute_sigma)
        self.combine_lists('m', weights_col=weights_colm, mask_lists=mask_lists, sigma=sigma, ismag=True, select_stars=select_stars,
                           absolute_sigma=absolute_sigma)

        return

    def combine_lists(self, col_name_in, weights_col=None, mask_val=None,
                      mask_lists=None, meta_add=True, ismag=False, sigma=3,
                      select_stars=None, absolute_sigma=True):
        """
        For the specified column (col_name_in), collapse along the starlists
        direction and calculated the average value, with outlier rejection.
        Optionally, weight by a specified column (weights_col). Optionally,
        use sigma clipping. The final values are stored in a new column named
        <col_name_in>0 -- the mean (with outlier rejection)
        <col_name_in>0e -- the std (with outlier rejection)

        Masking of NaN values is also performed.

        "mask_lists" is a list with the indices of starlists that are
        excluded from the combination.

        A flag can be stored in the metadata to record if the average was
        weighted or not.

        select_stars : array-like of bool or int, optional
            If given, only (re)compute the averaged columns for these star
            rows; every other row is left untouched. Useful when most rows
            already hold a valid average from a previous call and only a
            subset of rows (e.g. newly matched/added stars) actually need
            recomputing -- avoids redoing work for the whole (potentially
            very large) table every time. Ignored (falls back to computing
            for all rows) if the <col_name_in>0/<col_name_in>0_err columns
            don't exist yet, since there's nothing to selectively update on
            a first pass. By default None (compute for all rows).
        absolute_sigma : bool, optional
            Same convention as scipy.optimize.curve_fit and every
            MotionModel.run_fit. The reported uncertainty is always an
            uncertainty OF THE MEAN, never the scatter of the points, in all
            four cases below. With S = sum((x - xbar)**2), w = 1/sigma**2,
            chi2 = sum(w * (x - xbar)**2) and dof = n_valid - 1 (one parameter,
            the mean, estimated from the data):

            ======================  =========================  ============================
            branch                  absolute_sigma=True        absolute_sigma=False
            ======================  =========================  ============================
            weighted (weights_col)  ``sqrt(1/sum(w))``         ``sqrt(1/sum(w)*chi2/dof)``
            unweighted              ``sqrt(S/(n_valid*dof))``  ``sqrt(S/(n_valid*dof))``
            ======================  =========================  ============================

            True trusts the per-point input errors and propagates them; False
            rescales by the epochs' own disagreement, which is more honest when
            the input errors are systematically underestimated. absolute_sigma
            does not reach the unweighted branch: it chooses between
            propagating the input errors and rescaling by the observed scatter,
            and that branch runs precisely when there are no input errors to
            propagate, leaving only one thing to compute. That single entry is
            exactly the weighted False entry with every weight equal to 1.

            dof <= 0 (0 or 1 valid epochs) carries no residual information, so
            the uncertainty is inf there in both branches, as in every
            MotionModel.run_fit. By default True.
        """
        col_name_avg = col_name_in + '0'
        col_name_std = col_name_in + '0_err'
        if (select_stars is not None) and (col_name_avg not in self.colnames):
            select_stars = None

        if mask_lists is not None:
            # Extract list of indices that we want to keep (i.e. not mask)
            mask_lists = np.atleast_1d(mask_lists)
            assert mask_lists.dtype == int, "mask_lists needs to be a list of integers."
            list_indices = np.array([i for i in np.arange(self[col_name_in].data.shape[1]) if i not in mask_lists])
        else:
            # Use all indices. A plain slice (rather than an arange array) keeps
            # the col_data[:, list_indices] indexing below a view instead of a
            # forced fancy-index copy -- np.array()/masked_invalid() further down
            # already makes the one copy that's actually needed.
            list_indices = slice(None)

        if select_stars is not None:
            col_data = self[col_name_in].data[select_stars]
        else:
            col_data = self[col_name_in].data
        val_2d = np.array(col_data[:, list_indices], dtype=float)

        if ismag:
            # Convert to flux.
            val_2d = 10**(-0.4 * val_2d)

        # `valid` tracks, elementwise, whether a value is usable at all --
        # this replaces numpy.ma's masking, but as a plain boolean array so
        # the arithmetic below can use ordinary (fast) numpy ops instead of
        # numpy.ma's much slower generic dispatch for every operator.
        valid = np.isfinite(val_2d)

        # Mask a user-specified invalid value too.
        if mask_val:
            valid &= ~np.isclose(val_2d, mask_val, rtol=1e-05, atol=1e-08)

        # Figure out which ones are outliers. sigma_clip already treats NaN
        # (and, via the mask below, our own invalid entries) as excluded, and
        # returns a masked array -- pull its mask into `valid` and its data
        # into a plain array immediately, rather than keep operating on the
        # masked array itself for every subsequent step.
        if sigma:
            # Pass a masked (not NaN-filled) array in: sigma_clip treats an
            # explicit mask as "already known invalid" silently, whereas raw
            # NaNs trigger an "invalid values...automatically clipped"
            # warning that the original implementation never produced.
            val_2d_for_clip = np.ma.masked_array(val_2d, mask=~valid, copy=False)
            clipped = sigma_clip(val_2d_for_clip, sigma=sigma, maxiters=5, axis=1)
            valid &= ~np.ma.getmaskarray(clipped)
            val_2d_clip = np.where(valid, clipped.data, 0.0)
        else:
            val_2d_clip = np.where(valid, val_2d, 0.0)

        # Decide if we are going to have weights (before we do the expensive sigma clipping routine).
        if weights_col in self.colnames:
            if select_stars is not None:
                weights_data = self[weights_col].data[select_stars]
            else:
                weights_data = self[weights_col].data
            err_2d = np.array(weights_data[:, list_indices], dtype=float)

            if ismag:
                # Convert to flux error
                err_2d = 0.4 * np.log(10) * val_2d * err_2d

            # Inverse variance weights minimize the propagated uncertainty.
            # `err_2d` here is never faked/patched -- it's exactly what was
            # measured, so the `wgt_2d`/`wgt_sum` derived from it below are
            # an honest record of how much real uncertainty information we
            # actually have for each star. weight_from_sigma safely zeroes
            # out any epoch where the value isn't valid (post-clipping) or
            # the error itself is invalid/zero/overflow-inducing, rather
            # than letting a bad error corrupt the weighted sum.
            wgt_2d = motion_model.weight_from_sigma(err_2d, valid)

            # Honest weight sum, built only from real, known uncertainties.
            # The reported std below is derived directly from this, so a
            # star whose every epoch lacks a usable error naturally ends up
            # with wgt_sum == 0 -> std = sqrt(1/0) == inf via ordinary
            # division -- there's no separate flag to remember to apply
            # afterward, and no way for a fabricated finite error to reach std.
            wgt_sum = wgt_2d.sum(axis=1)
            n_valid = valid.sum(axis=1)
            has_data = n_valid > 0

            with np.errstate(divide='ignore', invalid='ignore'):
                avg = (val_2d_clip * wgt_2d).sum(axis=1) / wgt_sum
                # Equivalent of avg = np.average(val_2d_clip, weights=wgt_2d, axis=1)
                std = np.sqrt(1. / wgt_sum)  # Error propagation for weighted mean
                if not absolute_sigma:
                    # Rescale by sqrt(reduced chi2) -- scipy's own
                    # absolute_sigma=False convention (pcov *= chi2/dof),
                    # same formula every MotionModel.run_fit uses. chi2 here
                    # is the weighted sum of squared residuals against the
                    # weighted mean (the same quantity align.py separately
                    # computes as chi2_x/chi2_y later in fit(), though not
                    # reusable from here: that happens afterward in the same
                    # call, using the raw, non-sigma-clipped data, so it
                    # isn't numerically identical when sigma clipping is
                    # active). dof = n_valid - 1 (the weighted mean is
                    # effectively a 1-parameter fit); dof <= 0 has no
                    # residual information to estimate scatter from, so it's
                    # forced to inf exactly like every run_fit's own dof_pos
                    # check, rather than the 0/0 = nan (or, at dof=0 with a
                    # perfect single-point "fit", a false 0) this would
                    # otherwise silently produce.
                    dof = n_valid - 1
                    dof_pos = dof > 0
                    chi2 = (wgt_2d * (val_2d_clip - avg[:, np.newaxis])**2).sum(axis=1)
                    reduced_chi2 = np.where(dof_pos, chi2 / np.where(dof_pos, dof, 1), 1.0)
                    std = np.where(dof_pos, std * np.sqrt(reduced_chi2), np.inf)

            # A star whose every epoch has an invalid raw uncertainty (e.g.
            # missing/invalid me/xe/ye everywhere) but at least one valid
            # value still gets an average -- a plain mean of its valid
            # epoch(s), same as the unweighted branch below would give --
            # instead of discarding a real measurement as nan just because
            # we don't know how to weight it. std is forced to inf here
            # regardless of absolute_sigma (True already gives inf naturally
            # via 1/0; False would otherwise give 0/0 = nan), so this can't
            # accidentally fabricate a finite reported error.
            no_usable_err = (wgt_sum == 0) & has_data
            if no_usable_err.any():
                avg[no_usable_err] = val_2d_clip[no_usable_err].sum(axis=1) / n_valid[no_usable_err]
                std[no_usable_err] = np.inf

            avg[~has_data] = np.nan

            if meta_add:
                self.meta[col_name_in + '0'] = 'weighted'
        else:
            # Calculate the (unweighted) mean and uncertainty
            n_valid = valid.sum(axis=1)
            has_data = n_valid > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                avg = val_2d_clip.sum(axis=1) / n_valid
            avg[~has_data] = np.nan
            # Uncertainty OF THE MEAN, not the scatter of the points. This
            # used to return sqrt(sum(dev**2)/n_valid) -- the population RMS of
            # the individual values -- which is larger than the error on their
            # average by sqrt(n_valid - 1), and described a different quantity
            # from the weighted branch above, though both land in the same
            # <col>0_err column and both feed align's transformation weights.
            deviations = np.where(valid, val_2d_clip - avg[:, np.newaxis], 0.0)
            # sqrt(S / (n_valid * dof)) with dof = n_valid - 1: the standard
            # error of the mean. Identical to the weighted branch above with
            # every weight set to 1 -- sqrt(1/sum(w)) * sqrt(chi2/dof)
            # collapses to exactly this -- so both branches now report the same
            # quantity. absolute_sigma does not enter: it selects between
            # propagating the input errors and rescaling by the observed
            # scatter, and with no input errors to propagate there is only one
            # thing this branch can compute.
            #
            # dof is substituted with 1 inside the denominator rather than
            # dividing by zero and repairing the result afterwards: at
            # n_valid <= 1 the honest answer is 0/0, and the std == 0 guard
            # further down tests for zero, so a nan would slip past it.
            dof = n_valid - 1
            dof_pos = dof > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                std = np.sqrt((deviations**2).sum(axis=1)
                              / (n_valid * np.where(dof_pos, dof, 1)))
            std = np.where(dof_pos, std, np.inf)

            if meta_add:
                self.meta[col_name_in + '0'] = 'not_weighted'

        std_invalid = (~has_data) | (std == 0.)  # Mask out zero uncertainties

        # Save off our new AVG and STD into columns with shape (N_stars)
        # (col_name_avg/col_name_std were resolved at the top of this function).
        if ismag:
            with np.errstate(divide='ignore', invalid='ignore'):
                std = 2.5 / np.log(10) * std / avg  # Error propagation
                avg = -2.5 * np.log10(avg)

        # Fill invalid entries with nan (avg) or inf (std)
        std[std_invalid] = np.inf

        if select_stars is not None:
            # Columns must already exist -- only the selected rows are updated,
            # everything else is left exactly as it was.
            self[col_name_avg][select_stars] = avg
            self[col_name_std][select_stars] = std
        elif col_name_avg in self.colnames:
            self[col_name_avg] = avg
            self[col_name_std] = std
        else:
            self.add_column(Column(data=avg, name=col_name_avg))
            self.add_column(Column(data=std, name=col_name_std))

        return

    def detections(self, weight_col=None):
        """
        Find where stars are detected.

        weight_col : str, optional
            If given and present in this table's columns, sum this per-list
            column directly instead of counting each valid (x, y) as 1. Its
            "no detection this epoch" cells must already be 0 (not -1/nan)
            for this to give the right total. Used to inherit a per-list
            'n_detect_list' column from starlists that are themselves the
            output of a previous, lower-level align pass, so n_detect
            reflects the total number of raw detections a star represents.
            By default None (plain count).
        """
        if (weight_col is not None) and (weight_col in self.colnames):
            n_detect = np.sum(self[weight_col], axis=1)
        else:
            valid = np.isfinite(self['x']) & np.isfinite(self['y'])
            n_detect = np.sum(valid, axis=1)

        if 'n_detect' in self.colnames:
            self['n_detect'] = n_detect
        else:
            self.add_column(Column(data=n_detect, name='n_detect'))

        return

    def fit_motion_models(
            self,
            motion_models=None,
            fixed_params_dict=None,
            weighting='var',
            absolute_sigma=True,
            select_stars=None,
            keep_existing=True,
            bootstrap=0,
            seed=None,
            mask_value=None,
            mask_lists=None,
            fill_value=np.nan,
            art_star=False,
            processes=1,
            chunksize=None,
            mp_star_threshold=100_000,
            verbose=True
    ):
        """Fit velocity for star table

        Parameters
        ----------
        motion_models : list of MotionModel or str, optional
            Motion models to use, by default Empty, Fixed and Linear.
            Empty and Fixed models are always added automatically for stars with n_fit = 0 or 1.
            The behavior is as follows:
            1. If 'motion_model_input' column is NOT in table:

                - Use the most complex model that has enough parameters to fit the data (n_fit >= n_params).
                - If multiple models are supplied, prioritize the model with the most parameters to fit.
                - If multiple models have the same number of parameters, raise AssertionError: not sure which to use.

            2. If 'motion_model_input' column IS in table:

                - Use the model specified in the 'motion_model_input' column.
                - If not enough data points to fit the specified model, use the most complex model in any 'motion_model_input' column that has enough parameters to fit the data (n_fit >= n_params) among the provided motion_models and 'motion_model_input'.

            The actual used motion model is stored in the 'motion_model_used' column. The default motion_models are [Empty, Fixed, Linear].
        fixed_params_dict : dict, optional
            Dictionary of fixed parameters for motion models, e.g., {'t0': 0., 'ra': np.array([...]), 'dec': np.array([...])}.
            - Scalar values are used for all stars, array values should have length = N_stars.
            - t0 is automatically calculated as np.average(t, weights=1/np.hypot(xe, ye)) if not provided.
            - The keys should match the fixed parameter names in the motion models. See MotionModel class for details, by default None
        weighting : str, optional
            Uncertainty weighting, 'std' for weight=1/xe(ye) or 'var' for weight=1/xe(ye)**2, by default 'var'
        absolute_sigma : bool, optional
            Same convention as scipy.optimize.curve_fit. If True (default),
            the reported parameter errors are propagated from the input
            xe/ye as given. If False, they are rescaled by
            sqrt(chi2/dof), so only the relative magnitudes of xe/ye
            matter and the errors instead reflect the epochs' own scatter.

            Special case -- no uncertainty information available (the table
            has no xe/ye columns, or a given star's xe/ye are invalid in
            every epoch): the fit falls back to a substituted unit error
            (sigma=1) so a position can still be measured. With
            absolute_sigma=True the resulting error would be purely a
            function of that fabricated sigma (e.g. exactly 1/sqrt(N_valid)
            for Fixed), so it is reported as np.inf instead -- unknown
            rather than a finite number that only looks like a real
            uncertainty. With absolute_sigma=False the sqrt(chi2/dof)
            rescaling cancels the fabricated sigma back out, leaving the
            epochs' genuine empirical scatter, which is kept.
            By default True.
        select_stars : list of int, optional
            Indices of stars to fit, by default None (fit all stars)
        keep_existing : bool, optional
            Keep existing motion model results in the table, or set them to fill_value and Inf for stars not in select_stars, by default True
        bootstrap : int, optional
            Number of bootstrap samples for uncertainty resampling, by default 0
        seed : int, optional
            Random seed for bootstrap resampling, by default None
        mask_value : float, optional
            Values to mask in data, by default None
        mask_lists : list of int, optional
            Indices of lists to mask/exclude from fitting, by default None
        fill_value : float, optional
            Fill value when there is not enough data points to fit, by default np.nan
        art_star : bool, optional
            Artifical star table or observed star table. If artificial stars, Use the output coordinates for fitting motion models (x[..., 1], y[..., 1])
        processes : int, optional
            Number of processes to use for parallel processing, maximum os.cpu_count(), by default 1 (no multiprocessing)
        chunksize : int, optional
            Chunk size for multiprocessing, by default None (auto)
        mp_star_threshold : int, optional
            Minimum number of stars needing the per-star fitting path before a
            multiprocessing Pool is spun up, even if processes > 1 was
            requested. A star needs that path only when bootstrap > 0 (bootstrap
            resampling isn't vectorized across stars). Below this
            threshold, fitting runs serially in the calling process instead --
            spinning up a Pool has real fixed overhead (worker startup,
            pickling the shared data arrays to each worker) that a small
            per-star workload doesn't recoup. Measured break-even was between
            20,000 and 100,000 stars on a 10-core machine, so 100,000
            (default) is a conservative choice. By default 100_000.
        verbose : bool, optional
            Print verbose messages or not, by default True


        Raises
        ------
        ValueError
            If weighting is not 'var' or 'std'.
        KeyError
            If time values are not found in the table or meta.
        KeyError
            If required columns 'x' and 'y' are missing in the table.
        """
        ###########################
        ####### Check Params ######
        ###########################
        if weighting not in ['var', 'std']:
            raise ValueError(f"fit_motion_models: Weighting must either be 'var' or 'std', not {weighting}!")

        if ('t' not in self.colnames) and ('list_times' not in self.meta):
            raise KeyError("fit_motion_models: Failed to access time values. No 't' column in table, no 'list_times' in meta.")

        # Check if we have the required columns
        if not all([_ in self.colnames for _ in ['x', 'y']]):
            raise KeyError(f"fit_motion_models: Missing required columns in the table: {', '.join(['x', 'y'])}!")

        # Make a copy of fixed_params_dict to avoid modifying the original one outside the function
        fixed_params_dict = copy.deepcopy(fixed_params_dict)

        # Check fixed_params_dict is a dict
        if fixed_params_dict is not None:
            if not isinstance(fixed_params_dict, dict):
                raise ValueError("fit_motion_models: fixed_params_dict must be a dictionary!")

        if select_stars is not None:
            select_idx = np.asarray(select_stars)
            if select_idx.dtype == bool:
                select_idx = np.flatnonzero(select_idx)
            else:
                select_idx = np.asarray(select_idx, dtype=int)
            if len(select_idx) == 0:
                return
        else:
            select_idx = None

        N_stars = len(self)
        if (select_idx is not None) and (len(select_idx) < N_stars):
            # Everything below this point -- the masked-array data prep,
            # n_fit/motion-model classification, and per-star fixed-params
            # dict construction -- costs O(N_stars) every single call,
            # regardless of how few stars select_stars actually asks to
            # fit. For a mosaic that's re-fit once per starlist (this
            # function called repeatedly as the table keeps growing), that
            # made the redundant, unselected majority of the table get
            # copied and reprocessed on every single call -- for many
            # starlists and a large final table, this dwarfs the actual
            # fitting cost. Slice down to just the selected rows (fancy/
            # boolean indexing always copies in numpy, so this bounds cost
            # to len(select_stars), not N_stars), run this same function
            # unmodified on that much smaller table, then scatter its
            # results back into self at the selected positions. (If
            # select_stars covers the whole table there's nothing to save
            # by slicing -- that would just pay a full-table copy for no
            # benefit -- so fall through to the normal path below instead.)
            sub_fixed_params_dict = {
                k: (v[select_idx] if (np.ndim(v) > 0 and len(v) == N_stars) else v)
                for k, v in (fixed_params_dict or {}).items()
            }

            sub_table = self[select_idx]
            orig_meta_keys = set(self.meta.keys())
            sub_table.fit_motion_models(
                motion_models=motion_models, fixed_params_dict=sub_fixed_params_dict,
                weighting=weighting, absolute_sigma=absolute_sigma,
                select_stars=None, keep_existing=keep_existing,
                bootstrap=bootstrap, seed=seed, mask_value=mask_value, mask_lists=mask_lists,
                fill_value=fill_value, art_star=art_star, processes=processes,
                chunksize=chunksize, mp_star_threshold=mp_star_threshold, verbose=verbose
            )

            for col_name in sub_table.colnames:
                if col_name not in self.colnames:
                    default = np.inf if (col_name.endswith('_err')) else fill_value
                    dtype = sub_table[col_name].dtype
                    if dtype.kind in 'US':
                        default = ''
                    elif dtype.kind == 'i':
                        default = -1
                    elif dtype.kind == 'b':
                        default = False
                    self.add_column(Column(data=np.full(N_stars, default, dtype=dtype), name=col_name))
                self[col_name][select_idx] = sub_table[col_name]

            # Only propagate meta keys fit_motion_models itself newly added
            # (e.g. n_bootstrap, or a scalar-valued fixed param) -- not
            # table-size-specific ones the smaller sub_table happens to
            # carry (n_stars, ref_list, list_times, ...).
            for key, value in sub_table.meta.items():
                if key not in orig_meta_keys:
                    self.meta[key] = value

            return

        all_mm_map = motion_model.motion_model_map()
        # Setting the default to None to avoid mutable default argument issue
        # See https://stackoverflow.com/questions/15189245/assigning-class-variable-as-default-value-to-class-method-argument
        if motion_models is None:
            # Linear by default
            motion_models = [motion_model.Linear]
        motion_models = motion_model.organize_motion_models(motion_models)
        mm_names = [mm.name for mm in motion_models]

        # Construct motion models if motion_model_input column exists
        if 'motion_model_input' in self.colnames:
            input_mm_names = np.unique(self['motion_model_input'])
            assert all([name in all_mm_map.keys() for name in input_mm_names]), \
                f"fit_motion_models: Unknown motion model name(s) in 'motion_model_input' column. Available motion models are: {', '.join(all_mm_map.keys())}."
            for mm_name in input_mm_names:
                if mm_name not in mm_names:
                    motion_models.append(all_mm_map[mm_name])

            # Sort motion models by required epochs
            motion_models = sorted(motion_models, key=lambda mm: mm.n_params)

        input_mm_map = {mm.name: mm for mm in motion_models}

        mm_n_params = np.sort([mm.n_params for mm in motion_models])
        if 'motion_model_input' not in self.colnames:
            # If motion_model_input column is not provided, assert that motion model n_params are unique and sorted
            # Otherwise the fitter does not know which motion model to use based on n_obs
            assert len(mm_n_params) == len(set(mm_n_params)), \
                f"fit_motion_models: Provided motion model n_params are not unique! Motion Models are: {[_.name for _ in motion_models]}" + '\n' + "Cannot decide which motion model to use based on n_obs. Please provide unique motion_models or a 'motion_model_input' column."


        ###########################
        ####### Prepare Data ######
        ###########################
        # Prepare data for fitting
        N_stars = len(self)
        if art_star:
            x = self['x'].data[..., 1]
            y = self['y'].data[..., 1]
        else:
            x = self['x'].data
            y = self['y'].data

        xe = self['xe'].data if 'xe' in self.colnames else None
        ye = self['ye'].data if 'ye' in self.colnames else None
        with_xe_ye = (xe is not None) and (ye is not None)

        N_times = x.shape[1]
        if mask_lists is not None:
            list_indices = np.array([i for i in range(N_times) if i not in mask_lists])
        else:
            # A plain slice (rather than an arange array) keeps x[:, list_indices]
            # etc. below a view instead of a forced fancy-index copy -- the
            # explicit copy=True/deepcopy calls further down already make the one
            # copy that's actually needed. At full-table scale this was making two
            # full (N_stars, N_times) copies of x, y, xe, ye, and t where one would do.
            list_indices = slice(None)

        x_data = np.ma.masked_invalid(x[:, list_indices], copy=True)
        y_data = np.ma.masked_invalid(y[:, list_indices], copy=True)
        xe_data = np.ma.masked_invalid(xe[:, list_indices], copy=True) if with_xe_ye else None
        ye_data = np.ma.masked_invalid(ye[:, list_indices], copy=True) if with_xe_ye else None

        # Mask out close to 0 values to avoid infinite weights
        if with_xe_ye:
            xe_data.mask[np.isclose(xe_data, 0)] = True
            ye_data.mask[np.isclose(ye_data, 0)] = True

        # If all of xe and ye is masked for a star, effectively no uncertainties provided, fill with 1.
        # Note that this automatically turn the mask to False for these stars
        if with_xe_ye:
            fill_with_one = np.all(xe_data.mask, axis=1) & np.all(ye_data.mask, axis=1)
            xe_data[fill_with_one] = 1.
            ye_data[fill_with_one] = 1.
            # motion_model.fit()'s batch path only checks isfinite(x)/isfinite(y)
            # for validity -- weight_from_sigma zeroes any weight that comes out
            # non-finite, but a merely tiny (not exactly zero/inf/nan) xe/ye from
            # the np.isclose masking above wouldn't trip that check on its own.
            # Make the epochs that are *still* masked at this point (i.e. not
            # rescued by fill_with_one above) genuinely nan, so they're reliably
            # zero-weighted regardless of how close to zero they were.
            xe_data.data[xe_data.mask] = np.nan
            ye_data.data[ye_data.mask] = np.nan

        # Ensure data is 2D for consistent indexing, even if we have only one list/epoch (shape (N_stars, 1) instead of (N_stars,))
        if np.ndim(x_data) == 1:
            x_data = x_data[:, np.newaxis]
        if np.ndim(y_data) == 1:
            y_data = y_data[:, np.newaxis]
        if with_xe_ye:
            if np.ndim(xe_data) == 1:
                xe_data = xe_data[:, np.newaxis]
            if np.ndim(ye_data) == 1:
                ye_data = ye_data[:, np.newaxis]

        # t_data: 2d array with shape (N_stars, N_epochs)
        # t0: 1d array with shape (N_stars,)
        if 't' in self.colnames:
            t_data = copy.deepcopy(self['t'].data[:, list_indices])
        else:
            t_data = copy.deepcopy(np.array(self.meta['list_times']))[list_indices]
            t_data = np.broadcast_to(t_data, x_data.shape)

        fixed_params_dict = {} if fixed_params_dict is None else fixed_params_dict
        # Add default t0 if not provided in fixed_params_dict
        if 't0' not in fixed_params_dict:
            weights = 1. / np.hypot(xe_data, ye_data) if with_xe_ye else None
            # t_data must be masked (not just weights) and np.ma.average (not
            # plain np.average) must be used here: for the fill_with_one
            # stars above (no usable xe/ye anywhere at all), the substitute
            # weight is uniform/unmasked, but t can still be genuinely
            # invalid in undetected epochs, or weights can be masked (e.g.
            # only some epochs have usable xe/ye) while t_data itself is
            # plain. Plain np.average's weight-sum denominator doesn't
            # respect either mask in that case, silently producing NaN.
            fixed_params_dict['t0'] = np.ma.average(np.ma.masked_invalid(t_data), axis=1, weights=weights).filled(np.nan)
        else:
            if np.ndim(fixed_params_dict['t0']) == 0:
                fixed_params_dict['t0'] = np.full(N_stars, fixed_params_dict['t0'])

        t0 = fixed_params_dict['t0']

        # Apply mask_value if provided
        if mask_value:
            x_data = np.ma.masked_values(x_data, mask_value)
            y_data = np.ma.masked_values(y_data, mask_value)
            if with_xe_ye:
                xe_data = np.ma.masked_values(xe_data, mask_value)
                ye_data = np.ma.masked_values(ye_data, mask_value)
            # motion_model.fit()'s batch path derives validity as
            # isfinite(x) & isfinite(y) directly from the data (no
            # separate mask array is passed) -- unlike genuine "no
            # detection" gaps, mask_value cells aren't already nan, so
            # make them so here (once, only when mask_value is actually
            # used) rather than leaving a finite-but-meaningless value
            # (e.g. mask_value itself) silently treated as real. xe/ye
            # need the same treatment: fit()'s weight calculation already
            # zeroes out any weight that comes out non-finite, so a nan
            # xe/ye is enough to exclude that epoch even though only x/y
            # feed the isfinite() mask itself.
            x_data.data[x_data.mask] = np.nan
            y_data.data[y_data.mask] = np.nan
            if with_xe_ye:
                xe_data.data[xe_data.mask] = np.nan
                ye_data.data[ye_data.mask] = np.nan

        # Calculate mask array
        valid_xy = ~ (x_data.mask | y_data.mask)
        if with_xe_ye:
            valid_xy &= ~ (xe_data.mask | ye_data.mask)

        # Calculate n_fit: unmasked x y values
        # This will be used to determine which motion model to use for each star.
        # Note that we don't require unique times here
        # as scipy.curve_fit and Linear algebra can fit non-unique times.
        # self['n_fit'] = np.sum(valid_xy, axis=1)

        # Calculate n_fit: unique times & unmasked x y values.
        # Vectorized equivalent of len(set(t_data[i][valid_xy[i]])) per star:
        # push each star's invalid entries to +inf (so they sort last and
        # never affect the count), sort, then count 1 (for the first valid
        # entry, if any) plus the number of adjacent sorted valid entries
        # that differ -- mathematically identical to counting unique values,
        # but as whole-array numpy ops instead of a per-star Python loop
        # building a set() object for each of potentially millions of stars.
        N_epochs = t_data.shape[1]
        t_for_sort = np.where(valid_xy, t_data, np.inf)
        t_sorted = np.sort(t_for_sort, axis=1)
        n_valid_per_star = valid_xy.sum(axis=1)
        if N_epochs > 1:
            with np.errstate(invalid='ignore'):
                diffs_differ = np.diff(t_sorted, axis=1) != 0
            col_idx = np.arange(N_epochs - 1)
            diff_counts_valid = col_idx[np.newaxis, :] < (n_valid_per_star[:, np.newaxis] - 1)
            n_unique_extra = (diffs_differ & diff_counts_valid).sum(axis=1)
        else:
            n_unique_extra = np.zeros(N_stars, dtype=int)
        n_fit = np.where(n_valid_per_star > 0, 1 + n_unique_extra, 0)
        self['n_fit'] = n_fit


        ###########################
        ####### Determine MM ######
        ###########################
        if 'motion_model_input' in self.colnames:
            # Determine which motion model to use based on motion_model_input column
            # If n_fit < n_params for the input motion model, use the most complicated motion model with n_fit >= n_params
            required_params = np.array([all_mm_map[mm_name].n_params for mm_name in self['motion_model_input']])
            reassign_mm = n_fit < required_params

            mm_digitized = np.digitize(
                x=n_fit[reassign_mm],
                bins=mm_n_params
            ) - 1  # Convert to 0-based index

            # Assign motion models to stars
            self['motion_model_used'] = self['motion_model_input']
            self['motion_model_used'][reassign_mm] = np.array([motion_models[d].name for d in mm_digitized], dtype='U20')

        else:
            # If motion_model_input column is not provided, use the most complicated model in motion_models with n_fit >= n_params.
            mm_digitized = np.digitize(
                x=n_fit,
                bins=mm_n_params
            ) - 1  # Convert to 0-based index

            # Assign motion models to stars
            self['motion_model_used'] = np.array([motion_models[d].name for d in mm_digitized], dtype='U20')

        ############################
        # Prepare Fixed Parameters #
        ############################
        # If required fixed params in self.meta or columns, but not provided in fixed_params_dict, add them to fixed_params_dict
        motion_model_used = [all_mm_map[name] for name in np.unique(self['motion_model_used'])]
        raise_key_error = False
        missing_params = []
        for mm in motion_model_used:
            # Check required fixed parameters
            for param in mm.required_fixed_param_names:
                # Check in the order of fixed_params_dict -> self.meta -> self columns
                if param not in fixed_params_dict:
                    # If not provided in fixed_params_dict, it must be in table columns
                    if param in self.colnames:
                        fixed_params_dict[param] = self[param].data
                    elif param in self.meta:
                        # Check if the parameter is in self.meta
                        fixed_params_dict[param] = self.meta[param]
                    else:
                        raise_key_error = True
                        missing_params.append(f"'{param}'")

            # Check optional fixed parameters
            # Set to default value if not provided in fixed_params_dict or in self
            for param, value in mm.optional_fixed_params.items():
                if param not in fixed_params_dict:
                    # If param is not provided in fixed_params_dict
                    if param in self.colnames:
                        # Set to column value if column exists
                        fixed_params_dict[param] = self[param].data
                    elif param in self.meta:
                        # Check if the parameter is in self.meta
                        fixed_params_dict[param] = self.meta[param]
                    else:
                        # Set to default value if neither in columns nor provided in fixed_params_dict
                        fixed_params_dict[param] = value
                        self.meta[param] = value

        if raise_key_error:
            raise KeyError(f"fit_motion_models: Missing required fixed parameter(s) for the motion models used: {', '.join(missing_params)}! Please provide them in fixed_params_dict, or as columns in the table, or as table metadata.")


        # Prepare fixed_params_dict for each star
        # This avoids checking types and slicing inside the fitting loop
        # Identify array parameters (length N_stars) and scalar parameters
        array_params = {k: v for k, v in fixed_params_dict.items() if np.ndim(v) > 0 and len(v) == N_stars}
        scalar_params = {k: v for k, v in fixed_params_dict.items() if k not in array_params}

        # Convert any masked-array fixed params (e.g. the default t0, which
        # comes out of np.average() as a masked array whenever xe/ye are
        # masked) to plain arrays before the per-star dict construction
        # below -- indexing a MaskedArray once per star goes through numpy.ma's
        # much slower generic machinery vs. plain ndarray indexing.
        array_params = {k: (np.ma.filled(v, np.nan) if np.ma.isMaskedArray(v) else v) for k, v in array_params.items()}

        # fixed_params_stars (one dict per star) is only actually needed by
        # the per-star/multiprocessing fitting path below (bootstrap > 0) --
        # building it here for all N_stars unconditionally meant allocating a Python dict (plus
        # boxed scalar values) per star even for the (often large) fraction
        # handled entirely by the batched Fixed-model path, which never even
        # looks at it. It's built lazily further down, once we know which
        # stars actually need it (same idea as unmasked_idx below).


        ############################
        ####### Prepare Table ######
        ############################
        # Fill table with all possible motion model parameter names as new columns.
        new_col_list = motion_model.motion_model_param_names(motion_model_used, with_errors=True, with_fixed=False)
        new_col_list += ['chi2_x', 'chi2_y', 'n_params']

        if 't0' not in new_col_list:
            new_col_list.append('t0')

        # Add new columns if they do not exist
        for col in new_col_list:
            if col in self.colnames:
                # Keep old data if the column already exists
                if keep_existing:
                    continue
                else:
                    self.remove_column(col)

            if col.endswith('_err'):
                self.add_column(
                    Column(data=np.full(N_stars, np.inf, dtype=float), name=col),
                    rename_duplicate=True
                )
            else:
                self.add_column(
                    Column(data=np.full(N_stars, fill_value, dtype=float), name=col),
                    rename_duplicate=True
                )

        # Add fixed parameter meta if scalar, column if array.
        fixed_param_names = []
        for mm in motion_model_used:
            for param in mm.fixed_param_names:
                if param not in fixed_param_names:
                    fixed_param_names.append(param)
        # Remove t0 from fixed_param_names as it will be saved during fitting
        if 't0' in fixed_param_names:
            fixed_param_names.remove('t0')


        for param in fixed_param_names:
            # Equivalent to np.array([fps[param] for fps in fixed_params_stars])
            # from the (no-longer-built-eagerly) per-star dicts, without ever
            # materializing them: every param here came from array_params or
            # scalar_params above, so it's already exactly this column, or a
            # single value to be broadcast to one.
            if param in array_params:
                coldata = np.asarray(array_params[param])
            else:
                coldata = np.full(N_stars, scalar_params[param])

            if param in self.colnames:
                existing = self[param]

                # Skip if identical
                same = (
                    np.array_equal(existing, coldata)
                    if is_string_dtype(existing)
                    else np.allclose(existing, coldata, equal_nan=True)
                )

                if same:
                    continue

            # Different (or column does not yet exist)
            if len(np.unique(coldata)) == 1:
                # If param is the same for all stars, save it as meta
                self.meta[param] = coldata[0]
            else:
                self.add_column(
                    Column(data=coldata, name=f"{param}_mm"),
                    rename_duplicate=True,
                )

        # Add a column to keep track of the number of points used in a fit and number of bootstrap used.
        self.meta['n_bootstrap'] = bootstrap

        # A star whose motion_model_used just changed to a simpler model
        # (e.g. Linear -> Fixed, because it now matches fewer epochs than
        # it used to) would otherwise keep whatever vx/vy (or other params
        # its old, more complex model had) its previous fit wrote --
        # nothing rewrites those columns for this star since they aren't
        # in its new model's fit_param_names. Reset any such leftover
        # param to fill_value/inf for every star, based on its current
        # motion_model_used, before the fitting loop below fills in the
        # correct values for the params that DO belong to its model.
        # Check against every motion model that could ever exist, not just
        # ones assigned to a star this round -- a param column can still
        # exist from an earlier call (e.g. 'vx' from a prior Linear fit)
        # even if no star is currently classified as that model.
        all_possible_params = set()
        for mm in all_mm_map.values():
            all_possible_params.update(mm.fit_param_names)
        for param_name in all_possible_params:
            if param_name not in self.colnames:
                continue
            models_with_this_param = [mm.name for mm in all_mm_map.values() if param_name in mm.fit_param_names]
            belongs = np.isin(self['motion_model_used'], models_with_this_param)
            self[param_name][~belongs] = fill_value
            err_name = param_name + '_err'
            if err_name in self.colnames:
                self[err_name][~belongs] = np.inf


        ###########################
        ######### FITTING #########
        ###########################
        unique_motion_models, unique_inv_indices = np.unique(self['motion_model_used'], return_inverse=True)
        if select_stars is not None:
            select_stars = np.asarray(select_stars)
            if select_stars.dtype == bool:
                select_stars = np.flatnonzero(select_stars)
            else:
                select_stars = np.asarray(select_stars, dtype=int)
            indices_by_motion_model = {key: np.intersect1d(select_stars, np.flatnonzero(unique_inv_indices == k)) for k, key in enumerate(unique_motion_models)}
        else:
            indices_by_motion_model = {key: np.flatnonzero(unique_inv_indices == k) for k, key in enumerate(unique_motion_models)}

        # Unmasked indices for each star -- but only for stars in groups that
        # actually need the generic per-star path below. Every group goes
        # through the batch fit() path unless bootstrap > 0 forces the
        # per-star fallback (bootstrap resampling isn't vectorized across
        # stars), so this is only ever non-empty in that case. Left as None
        # for stars that don't need it; those entries are never looked up.
        non_batch_star_idxs = list(indices_by_motion_model.values()) if bootstrap > 0 else []
        if non_batch_star_idxs:
            non_batch_star_idxs = np.concatenate(non_batch_star_idxs)
            unmasked_idx = [None] * N_stars
            fixed_params_stars = [None] * N_stars
            for i in non_batch_star_idxs:
                unmasked_idx[i] = np.flatnonzero(valid_xy[i])
                fixed_params_stars[i] = {**scalar_params, **{k: v[i] for k, v in array_params.items()}}
        else:
            unmasked_idx = None
            fixed_params_stars = None

        # Plain (non-masked) views of the per-star arrays for the per-star
        # extraction below. x_data/y_data/xe_data/ye_data need to stay
        # numpy.ma arrays up to this point because valid_xy (and thus
        # unmasked_idx) is derived from their masks -- but once we have
        # unmasked_idx, indexing with it only ever touches already-known-
        # valid entries, so the mask itself is no longer needed and plain
        # ndarray indexing (via .data, a zero-copy view) is far cheaper than
        # numpy.ma's per-element indexing machinery. Doing this extraction
        # with the masked arrays directly was previously the single largest
        # cost in this function for large tables (confirmed by profiling:
        # tens of millions of numpy.ma.core.__getitem__ calls).
        t_data_arr = np.asarray(t_data)
        x_data_arr = x_data.data if np.ma.isMaskedArray(x_data) else np.asarray(x_data)
        y_data_arr = y_data.data if np.ma.isMaskedArray(y_data) else np.asarray(y_data)
        xe_data_arr = (xe_data.data if np.ma.isMaskedArray(xe_data) else np.asarray(xe_data)) if with_xe_ye else None
        ye_data_arr = (ye_data.data if np.ma.isMaskedArray(ye_data) else np.asarray(ye_data)) if with_xe_ye else None

        # If multiprocessing, spawn ONE pool for the whole function (not one
        # per motion-model group below), and hand each worker the shared
        # per-star data arrays exactly once via the initializer. Each task
        # then only needs to cross the process boundary with a star index +
        # its small fixed_params_dict, and does its own (ragged -- stars
        # have different numbers of valid epochs) data extraction locally,
        # instead of the parent process pre-extracting a t_stars/x_stars/...
        # slice for every single star up front and pickling all of it per task.
        # Only actually pay for a multiprocessing Pool when there's enough
        # per-star work to recoup its fixed cost (worker startup, pickling
        # the shared data arrays to each worker) -- below mp_star_threshold,
        # run serially in this process even if processes > 1 was requested.
        pool = None
        if processes > 1 and unmasked_idx is not None and len(non_batch_star_idxs) >= mp_star_threshold:
            pool = Pool(
                processes,
                initializer=_fit_motion_models_init,
                initargs=(t_data_arr, x_data_arr, y_data_arr, xe_data_arr, ye_data_arr,
                          unmasked_idx, input_mm_map, weighting, absolute_sigma,
                          fill_value, bootstrap, seed, verbose)
            )

        try:
            # For each motion model
            for unique_motion_model, unique_index in indices_by_motion_model.items():
                # Create motion model instance
                motion_model_instance = input_mm_map[unique_motion_model]()
                param_names = motion_model_instance.fit_param_names
                # Initialize arrays to store results
                n_stars_this_model = len(unique_index)
                n_params = len(param_names)

                params_array = np.full((n_stars_this_model, n_params), fill_value, dtype=float)
                param_errs_array = np.full((n_stars_this_model, n_params), np.inf, dtype=float)
                chi2_x_array = np.full(n_stars_this_model, np.nan, dtype=float)
                chi2_y_array = np.full(n_stars_this_model, np.nan, dtype=float)

                # For each star
                if len(unique_index) > 0:
                    if bootstrap == 0:
                        # Every model's fit() accepts a 2D batch (many stars
                        # packed into one rectangular array) as well as a
                        # single star, and fits the whole subgroup in one
                        # vectorized pass instead of star-by-star. This matters
                        # even when the table isn't ALL Fixed (the align.py-level
                        # shortcut to combine_lists_xym only fires then): a large
                        # fraction of stars in a growing mosaic are often still
                        # Fixed regardless of what other stars need, and that
                        # fraction only shrinks as more epochs get added -- so
                        # without this, the (often huge) Fixed subset would
                        # keep paying the per-star loop/multiprocessing cost.
                        # Bootstrap resampling isn't vectorized here, so that
                        # case still falls through to the per-star path below.
                        if verbose:
                            print(f"Fitting {unique_motion_model} motion model: vectorized batch fit for {n_stars_this_model} star(s)")
                        n_epochs = t_data_arr.shape[1]
                        xe_batch = xe_data_arr[unique_index] if with_xe_ye else np.ones((n_stars_this_model, n_epochs))
                        ye_batch = ye_data_arr[unique_index] if with_xe_ye else np.ones((n_stars_this_model, n_epochs))
                        # Same {scalar params} + {array params sliced to this
                        # group} construction as fixed_params_stars above, but
                        # kept batched (not exploded into one dict per star)
                        # since fit() takes it once for the whole group.
                        fixed_params_batch = {
                            **scalar_params,
                            **{k: v[unique_index] for k, v in array_params.items()}
                        }
                        # No mask is passed -- fit()'s batch path derives
                        # validity from nan in x/y directly (x_data_arr/
                        # y_data_arr already have real nan at every invalid
                        # cell, including mask_value/near-zero-error cells,
                        # which were explicitly nan-ed above for exactly this).
                        params_array, param_errs_array, chi2_x_array, chi2_y_array = motion_model_instance.fit(
                            t_data_arr[unique_index], x_data_arr[unique_index], y_data_arr[unique_index],
                            xe_batch, ye_batch,
                            fixed_params_dict=fixed_params_batch,
                            weighting=weighting, absolute_sigma=absolute_sigma, fill_value=fill_value, verbose=verbose
                        )

                    elif pool is not None:
                        # Use multiprocessing to fit stars in parallel
                        arguments = [(i_star, unique_motion_model, fixed_params_stars[i_star]) for i_star in unique_index]

                        results = pool.starmap(
                            _fit_motion_models_worker,
                            tqdm(
                                arguments,
                                desc=f"Fitting {unique_motion_model} motion model with {processes} processes",
                                disable=not verbose
                            ),
                            chunksize=chunksize
                        )

                        for idx, (params, param_errs, chi2_x, chi2_y) in enumerate(results):
                            params_array[idx] = params
                            param_errs_array[idx] = param_errs
                            chi2_x_array[idx] = chi2_x
                            chi2_y_array[idx] = chi2_y

                    else:
                        # Prepare data as lists of arrays for faster access during fitting
                        t_stars = [t_data_arr[i][unmasked_idx[i]] for i in unique_index]
                        x_stars = [x_data_arr[i][unmasked_idx[i]] for i in unique_index]
                        y_stars = [y_data_arr[i][unmasked_idx[i]] for i in unique_index]
                        xe_stars = [xe_data_arr[i][unmasked_idx[i]] for i in unique_index] if with_xe_ye else [np.ones_like(x_star) for x_star in x_stars]
                        ye_stars = [ye_data_arr[i][unmasked_idx[i]] for i in unique_index] if with_xe_ye else [np.ones_like(y_star) for y_star in y_stars]

                        # Expensive for loop! Prepare everything beforehand to speed up.
                        for idx, i_star in enumerate(tqdm(unique_index, disable=not verbose, desc=f"Fitting {unique_motion_model} motion model")):
                            # Fit the star
                            params, param_errs, chi2_x, chi2_y = motion_model_instance.fit(
                                t=t_stars[idx],
                                x=x_stars[idx],
                                y=y_stars[idx],
                                xe=xe_stars[idx],
                                ye=ye_stars[idx],
                                fixed_params_dict=fixed_params_stars[i_star],
                                weighting=weighting,
                                absolute_sigma=absolute_sigma,
                                fill_value=fill_value,
                                return_chi2=True,
                                bootstrap=bootstrap,
                                seed=seed,
                                verbose=verbose
                            )
                            params_array[idx] = params
                            param_errs_array[idx] = param_errs
                            chi2_x_array[idx] = chi2_x
                            chi2_y_array[idx] = chi2_y

                # These stars were fit with a substituted unit error (sigma=1)
                # rather than a real measurement uncertainty -- either the
                # table has no xe/ye columns at all (xe_batch/xe_stars above
                # fall back to np.ones), or this particular star's own xe/ye
                # were invalid in every epoch (fill_with_one). Both cases are
                # the same situation: no uncertainty information exists.
                #
                # With absolute_sigma=True the reported error is then purely
                # propagated from that fabricated sigma -- e.g. exactly
                # 1/sqrt(N_valid) for Fixed (0.577... for 3 epochs) -- a
                # finite number that looks like a real uncertainty in the
                # data's units but carries no measurement information at all,
                # so report inf instead. With absolute_sigma=False the errors
                # are rescaled by sqrt(chi2/dof), which cancels the fabricated
                # sigma back out and leaves the epochs' own empirical scatter:
                # a genuine, correctly-scaled uncertainty, so that one is kept.
                if absolute_sigma:
                    if with_xe_ye:
                        no_real_err = fill_with_one[unique_index]
                    else:
                        no_real_err = np.ones(len(unique_index), dtype=bool)
                    if no_real_err.any():
                        param_errs_array[no_real_err] = np.inf

                # Store results back to the table
                for j, param_name in enumerate(param_names):
                    self[param_name][unique_index] = params_array[:, j]
                    self[param_name + '_err'][unique_index] = param_errs_array[:, j]
                self['chi2_x'][unique_index] = chi2_x_array
                self['chi2_y'][unique_index] = chi2_y_array
                self['t0'][unique_index] = t0[unique_index]
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        # Update n_params regardless of selections
        for mm in motion_model_used:
            self['n_params'][self['motion_model_used'] == mm.name] = mm.n_params
        return

    def infer_positions(self, times, fixed_params_dict=None, fill_value=np.nan):
        """Infer star positions at given times using fitted motion models.

        Parameters
        ----------
        times : scalar or array_like
            Times at which to predict positions. The SHAPE decides whether the
            times are shared across stars or are per-star -- nothing is
            inferred from ``len(times)`` matching ``N_stars``:

            ======================  =========================================
            ``times``               meaning
            ======================  =========================================
            scalar                  one time, every star
            ``(N_times,)``          one shared grid, every star -- always,
                                    even when ``N_times == N_stars``
            ``(1, N_times)``        the same, written explicitly
            ``(N_stars, N_times)``  each star has its own times
            ======================  =========================================

            For one time per star, pass a column vector
            ``times[:, np.newaxis]`` of shape ``(N_stars, 1)`` -- not a bare
            1D array. Note that propagating the whole table to a single new
            epoch does NOT need that: pass the scalar epoch and a per-star
            ``t0`` (table column or fixed_params_dict entry), and each star's
            ``dt = t - t0[star]`` already differs. Any other shape raises
            ``ValueError`` rather than being guessed at. See
            :func:`~flystar.motion_model.broadcast_times`.
        fixed_params_dict : None or dict, optional
            Dictionary of fixed parameters to use for prediction.
            If not provided, will try to look for fixed parameters in the meta data then in table columns.
            If fixed params are found in both the table and the fixed_params_dict, the values in the table will be used and the fixed_params_dict values will be ignored,
            by default None
        fill_value : float, optional
            Value to use for missing data, by default np.nan

        Returns
        -------
        x, y, xe, ye : ndarray
            Arrays of predicted x, y positions and their uncertainties xe, ye, with shape (N_stars, N_times) or (N_stars,) if N_times=1, or (N_times,) if N_stars=1, or scalar.
        """
        # Which model moves each star is decided here rather than taken from a
        # caller-supplied array: it is an internal detail, not something the
        # user should have to compute. A 'motion_model_input' column is honored
        # as an explicit per-star request, falling back -- per star -- to the
        # most complex model that star's own parameters actually support, both
        # when the column is absent and when a request cannot be evaluated
        # (e.g. Acceleration requested but ax is nan for too few epochs).
        #
        # Deliberately NOT 'motion_model_used': that records which model was
        # FIT, and a reference star imported from an external catalog can carry
        # vx/vy/t0 that were never fit here and must still move with Linear.
        # For a star that was fit, the two agree anyway -- the parameters
        # outside its own model are nan, so they cannot be selected.
        motion_model_used = motion_model.determine_motion_models(
            self, motion_models=None, fixed_params_dict=fixed_params_dict
        )[0]

        N_stars = len(self)
        # Normalize to an explicit (N_stars, N_times) grid up front, so the
        # shared-grid vs per-star distinction is settled by times' shape
        # rather than re-derived (differently) inside each motion model.
        # See motion_model.broadcast_times for the accepted shapes.
        times_grid = motion_model.broadcast_times(times, N_stars, caller='infer_positions')
        N_times = times_grid.shape[1]

        x_pred = np.full((N_stars, N_times), fill_value, dtype=float)
        y_pred = np.full((N_stars, N_times), fill_value, dtype=float)

        # Only calculate xe ye if columns exist in table, otherwise fill with np.inf
        if 'x0_err' in self.colnames and 'y0_err' in self.colnames:
            # 'x0_err' and 'y0_err' are the common uncertainty params for all motion models
            with_xe_ye = True
            xe_pred = np.full((N_stars, N_times), np.inf, dtype=float)
            ye_pred = np.full((N_stars, N_times), np.inf, dtype=float)
        else:
            with_xe_ye = False
            xe_pred = np.full((N_stars, N_times), np.inf, dtype=float)
            ye_pred = np.full((N_stars, N_times), np.inf, dtype=float)

        # Calculate the dictionary of {motion_model: indices of stars with this motion model} for faster access during prediction
        unique_motion_models, unique_inv_indices = np.unique(motion_model_used, return_inverse=True)
        indices_by_motion_model = {key: np.flatnonzero(unique_inv_indices == k) for k, key in enumerate(unique_motion_models)}

        mm_map = motion_model.motion_model_map()
        # Prepare fit_params, fixed_params, fit_param_errs for each star
        for unique_motion_model, unique_index in indices_by_motion_model.items():
            # Create motion model instance
            motion_model_instance = mm_map[unique_motion_model]()
            # Prepare parameters for prediction
            fit_params = np.array([
                self[param_name][unique_index] for param_name in motion_model_instance.fit_param_names
            ]).T # shape (N_stars_this_model, N_params)

            fit_param_errs = np.array([
                self[param_name + '_err'][unique_index] for param_name in motion_model_instance.fit_param_names
            ]).T if with_xe_ye else None # shape (N_stars_this_model, N_params)

            # Construct fixed_params: Look for fixed_params_dict -> table columns -> meta data -> default value
            fixed_params = fixed_params_dict.copy() if fixed_params_dict is not None else {}
            for param in motion_model_instance.required_fixed_param_names:
                if param not in fixed_params:
                    # If required fixed param not provided, find it in the table columns or meta data
                    if param in self.colnames:
                        fixed_params[param] = self[param][unique_index]
                    elif param in self.meta:
                        fixed_params[param] = self.meta[param]
                    else:
                        raise KeyError(f"infer_positions: Required fixed parameter '{param}' not found for motion model '{unique_motion_model}'. Please provide it in fixed_params_dict, or add it as a column in the table, or add it to the meta data.")
                else:
                    fixed_params[param] = fixed_params_dict[param]

            for param, default_value in motion_model_instance.optional_fixed_params.items():
                if param not in fixed_params:
                    # If optional fixed param not provided, find it in the table columns or meta data, otherwise use default value
                    if param in self.colnames:
                        if param == 'obsLocation':
                            # Special case for obsLocation: no vectorization implemented yet, use the value from the first star
                            assert np.unique(self[param][unique_index]).size == 1, \
                                f"infer_positions: obsLocation fixed parameter has different values ({np.unique(self[param][unique_index])}) for different stars. Vectorized handling not implemented yet."
                        fixed_params[param] = self[param][unique_index]
                    elif param in self.meta:
                        fixed_params[param] = self.meta[param]
                    else:
                        fixed_params[param] = default_value
                else:
                    fixed_params[param] = fixed_params_dict[param]


            # Predict positions
            # shape = (N_stars_this_model, N_times) or (N_stars_this_model,) if N_times=1 or (N_times,) if N_stars_this_model=1 or scalar
            # Hand this model's stars their own rows of the time grid, so a
            # per-star grid stays aligned with the per-star fit_params /
            # fixed_params sliced by unique_index just above.
            times_this_model = times_grid[unique_index]
            if with_xe_ye:
                x, y, xe, ye = motion_model_instance.model(
                    times_this_model, fit_params, fit_param_errs, fixed_params
                )
            else:
                x, y = motion_model_instance.model(
                    times_this_model, fit_params, fixed_params=fixed_params
                )

            if N_stars==1 and N_times > 1:
                # Reshape (N_times,) to (1, N_times)
                x = x[np.newaxis, :]
                y = y[np.newaxis, :]
                if with_xe_ye:
                    xe = xe[np.newaxis, :]
                    ye = ye[np.newaxis, :]
            elif N_times==1 and N_stars > 1:
                # Reshape (N_stars,) to (N_stars, 1)
                x = x[:, np.newaxis]
                y = y[:, np.newaxis]
                if with_xe_ye:
                    xe = xe[:, np.newaxis]
                    ye = ye[:, np.newaxis]

            x_pred[unique_index] = x
            y_pred[unique_index] = y
            if with_xe_ye:
                xe_pred[unique_index] = xe
                ye_pred[unique_index] = ye

        if N_stars==1 or N_times==1:
            # Reshape back to 1D array or scalar
            x_pred = x_pred.flatten()
            y_pred = y_pred.flatten()
            if with_xe_ye:
                xe_pred = xe_pred.flatten()
                ye_pred = ye_pred.flatten()

        xe_pred = xe_pred if with_xe_ye else np.full_like(x_pred, np.inf)
        ye_pred = ye_pred if with_xe_ye else np.full_like(y_pred, np.inf)
        return x_pred, y_pred, xe_pred, ye_pred


    # New function, to use in align
    def get_star_positions_at_time(self, t, motion_model_dict, allow_alt_models=True):
        """ Get current x,y positions of each star according to its motion_model
        """
        # Start with empty arrays so we can fill them in batches
        N_stars = len(self)
        if hasattr(t, "__len__"):
            x = np.full((N_stars,len(t)), np.nan, dtype=float)
            y = np.full((N_stars,len(t)), np.nan, dtype=float)
            xe = np.full((N_stars,len(t)), np.nan, dtype=float)
            ye = np.full((N_stars,len(t)), np.nan, dtype=float)
        else:
            x = np.full(N_stars, np.nan, dtype=float)
            y = np.full(N_stars, np.nan, dtype=float)
            xe = np.full(N_stars, np.nan, dtype=float)
            ye = np.full(N_stars, np.nan, dtype=float)

        # TODO: probably worth some additional testing here
        # Check which motion models we need
        # use complex_mms to collect models besides Fixed and Linear
        unique_mms = np.unique(self['motion_model_input']).tolist()
        # Calculate current position in batches by motion model
        for mm in unique_mms:
            try:
                # Identify stars with this model & get class
                idx = np.where(self['motion_model_input']==mm)[0]
                mod = motion_model_dict[mm]
                # Set up parameters
                param_dict = {}
                for par in mod.fit_param_names + mod.fixed_param_names + [pm+'_err' for pm in mod.fit_param_names]:
                    param_dict[par] = self[par][idx]
                x[idx],y[idx],xe[idx],ye[idx] = mod.get_batch_pos_at_time(t,**param_dict)
            except:
                pass
        if np.isnan(x).any() and allow_alt_models:
            re_calc = np.where(np.isnan(x))[0]
            unique_mms = np.unique(self['motion_model_used'][re_calc]).tolist()
            # Calculate current position in batches by motion model
            for mm in unique_mms:
                # Identify stars with this model & get class
                idx_0 = np.where(self['motion_model_used']==mm)[0]
                idx = np.intersect1d(re_calc, idx_0)
                mod = motion_model_dict[mm]
                # Set up parameters
                param_dict = {}
                for par in motion_model.get_one_motion_model_param_names(mm,with_errors=True,with_fixed=True):
                    param_dict[par] = self[par][idx]
                x[idx],y[idx],xe[idx],ye[idx] = mod.get_batch_pos_at_time(t,**param_dict)

        return x, y, xe, ye



    def shift_reference_frame(self, delta_vx=0.0, delta_vy=0.0, delta_pi=0.0, fixed_params_dict=None):
        """
        After completing an alignment, shift from your relative reference frame to
        the absolute frame using either Gaia or a Galactic model. This modified the
        motion model fit parameters as well as the time series astrometry, assuming
        zero error on the shift values.

        Parameters
        ----------
        delta_vx : float, optional
            velocity shift in x-direction (as/yr)
        delta_vy : float, optional
            velocity shift in y-direction (as/yr)
        delta_pi : float, optional
            parallax shift (as)
        fixed_params_dict : None or dict, optional
            Dictionary of fixed parameters to use for prediction: ra, dec, obsLocation, specifically in this case
        """
        if delta_vx==0.0 and delta_vy==0.0 and delta_pi==0.0:
            print("No shifts input, reference frame unchanged.")
            print("Specify delta_vx, delta_vy, and/or delta_pi to perform a reference frame shift.")
            return
        self['vx'] += delta_vx
        self['x'] += delta_vx*(self['t']-self['t0'][:, np.newaxis])
        self['vy'] += delta_vy
        self['y'] += delta_vy*(self['t']-self['t0'][:, np.newaxis])
        if delta_pi!=0.0:
            fixed_params_dict = {} if fixed_params_dict is None else fixed_params_dict
            if 'ra' not in fixed_params_dict or 'dec' not in fixed_params_dict:
                raise KeyError("shift_reference_frame: 'ra' and 'dec' must be provided in fixed_params_dict for parallax shift.")
            from .motion_model import Parallax
            ra = fixed_params_dict['ra']
            dec = fixed_params_dict['dec']
            pa = fixed_params_dict.get('pa', 0.0)
            obsLocation = fixed_params_dict.get('obsLocation', 'earth')
            t_all = self['t'][np.where(~np.any(np.isnan(self['t']), axis=1))[0][0]]
            # Observation epochs come from UTC timestamps, so convert UTC -> TDB
            # properly rather than relabelling the number. parallax_in_direction
            # declares its input TDB without converting, so handing it a bare UTC
            # MJD shifts every epoch by TDB-UTC (69.184 s as of 2026).
            #
            # UTC conversion consults the leap-second table, and ERFA warns
            # ("dubious year") for epochs more than ~5 years past the table in the
            # installed pyerfa -- around 2028 for 2.0.1.5. Only future epochs are
            # affected; any real observation sits inside the table.
            t_mjd = Time(t_all, format='decimalyear', scale='utc').tdb.mjd
            pvec = Parallax().calc_parallax_vector(t_mjd, ra=ra, dec=dec, pa=pa, obsLocation=obsLocation)
            self['pi'] += delta_pi
            self['x'] += delta_pi*pvec[:, 0, :] # Shape (N_stars, N_times)
            self['y'] += delta_pi*pvec[:, 1, :] # Shape (N_stars, N_times)
        return

def shift_reference_frame(table, delta_vx=0.0, delta_vy=0.0, delta_pi=0.0, fixed_params_dict=None):
    """
    After completing an alignment, shift from your relative reference frame to
    the absolute frame using either Gaia or a Galactic model. This modified the
    motion model fit parameters as well as the time series astrometry, assuming
    zero error on the shift values.

    Parameters
    ----------
    delta_vx : float, optional
        velocity shift in x-direction (as/yr)
    delta_vy : float, optional
        velocity shift in y-direction (as/yr)
    delta_pi : float, optional
        parallax shift (as)
    """
    if delta_vx==0.0 and delta_vy==0.0 and delta_pi==0.0:
        print("No shifts input, reference frame unchanged.")
        print("Specify delta_vx, delta_vy, and/or delta_pi to perform a reference frame shift.")
        return
    table['vx'] += delta_vx
    table['x'] += delta_vx*(table['t']-table['t0'][:, np.newaxis])
    table['vy'] += delta_vy
    table['y'] += delta_vy*(table['t']-table['t0'][:, np.newaxis])
    if delta_pi!=0.0:
        from .motion_model import Parallax
        fixed_params_dict = {} if fixed_params_dict is None else fixed_params_dict
        if 'ra' not in fixed_params_dict or 'dec' not in fixed_params_dict:
            raise KeyError("shift_reference_frame: 'ra' and 'dec' must be provided in fixed_params_dict for parallax shift.")
        ra = fixed_params_dict['ra']
        dec = fixed_params_dict['dec']
        pa = fixed_params_dict.get('pa', 0.0)
        obsLocation = fixed_params_dict.get('obsLocation', 'earth')
        t_all = table['t'][np.where(~np.any(np.isnan(table['t']), axis=1))[0][0]]
        t_mjd = Time(t_all, format='decimalyear', scale='utc').tdb.mjd
        pvec = Parallax().calc_parallax_vector(t_mjd, ra=ra, dec=dec, pa=pa, obsLocation=obsLocation)
        table['pi'] += delta_pi
        table['x'] += delta_pi*pvec[:, 0, :] # Shape (N_stars, N_times)
        table['y'] += delta_pi*pvec[:, 1, :] # Shape (N_stars, N_times)
    return table


# Per-worker state for the fit_motion_models() process pool, set once by
# _fit_motion_models_init() when each worker starts. Using a Pool initializer
# instead of passing this data with every task means the (potentially large)
# shared arrays cross the process boundary once per worker, not once per star.
_fmm_worker_state = {}


def _fit_motion_models_init(t_data, x_data, y_data, xe_data, ye_data, unmasked_idx,
                             input_mm_map, weighting, absolute_sigma,
                             fill_value, bootstrap, seed, verbose):
    """
    Pool initializer for fit_motion_models(). Stashes the per-star data
    arrays (shared, read-only across all stars/tasks) as module-level state
    in each worker process, so individual tasks only need to send a star
    index and its small fixed_params_dict -- not a freshly-extracted slice
    of every array -- to get fit.
    """
    _fmm_worker_state.update(
        t_data=t_data, x_data=x_data, y_data=y_data, xe_data=xe_data, ye_data=ye_data,
        unmasked_idx=unmasked_idx, input_mm_map=input_mm_map, weighting=weighting,
        absolute_sigma=absolute_sigma,
        fill_value=fill_value, bootstrap=bootstrap, seed=seed, verbose=verbose,
    )


def _fit_motion_models_worker(i_star, motion_model_name, fixed_params_dict):
    """
    Pool worker for fit_motion_models(). Slices out this one star's own
    (ragged -- stars have different numbers of valid epochs) data from the
    shared arrays stashed by _fit_motion_models_init(), then fits it.
    """
    s = _fmm_worker_state
    idx = s['unmasked_idx'][i_star]
    t = np.array(s['t_data'][i_star][idx])
    x = np.array(s['x_data'][i_star][idx])
    y = np.array(s['y_data'][i_star][idx])
    if s['xe_data'] is not None:
        xe = np.array(s['xe_data'][i_star][idx])
        ye = np.array(s['ye_data'][i_star][idx])
    else:
        xe = np.ones_like(x)
        ye = np.ones_like(y)

    motion_model_instance = s['input_mm_map'][motion_model_name]()
    return motion_model_instance.fit(
        t=t, x=x, y=y, xe=xe, ye=ye,
        fixed_params_dict=fixed_params_dict,
        weighting=s['weighting'],
        absolute_sigma=s['absolute_sigma'],
        fill_value=s['fill_value'],
        return_chi2=True,
        bootstrap=s['bootstrap'],
        seed=s['seed'],
        verbose=s['verbose'],
    )