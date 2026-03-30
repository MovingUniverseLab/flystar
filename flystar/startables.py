from astropy.table import Table, Column, MaskedColumn, hstack
from astropy.stats import sigma_clipping
from astropy.time import Time
from scipy.optimize import curve_fit
from tqdm import tqdm
import numpy as np
import warnings
import collections
import pdb
import time
import copy
from flystar import motion_model
import pandas as pd
from flystar.motion_model import Empty, Fixed, Linear
from pandas.api.types import is_string_dtype

class StarTable(Table):
    def __init__(self, *args, ref_list=0, **kwargs):
        """
        A StarTable is an astropy.Table with stars matched from multiple starlists.

        Required table columns (input as keywords):
        -------------------------
        name : 1D numpy.array with shape = N_stars
            List of unique names for each of the stars in the table.

        x : 2D numpy.array with shape = (N_stars, N_lists)
            Positions of N_stars in each of N_lists in the x dimension.

        y : 2D numpy.array with shape = (N_stars, N_lists)
            Positions of N_stars in each of N_lists in the y dimension.

        m : 2D numpy.array with shape = (N_stars, N_lists)
            Magnitudes of N_stars in each of N_lists.

        Optional table columns (input as keywords):
        -------------------------
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
            List of times/dates for each starlist.

        ref_list : int
            Specify which list is the reference list (if any).

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
            Table.__init__(self, *args, **kwargs)
        else:
            # If we have errors, we need them in both dimensions.
            if ('xe' in kwargs) ^ ('ye' in kwargs):
                raise TypeError("The StarTable class requires both 'xe' and" +
                                " 'ye' arguments")

            # Figure out the shape
            n_stars = kwargs['x'].shape[0]
            n_lists = kwargs['x'].shape[1]

            # Check if the type and size of the arguments are correct.
            # Name checking: type and shape
            if (not isinstance(kwargs['name'], np.ndarray)) or (len(kwargs['name']) != n_stars):
                err_msg = "The '{0:s}' argument has to be a numpy array "
                err_msg += "with length = {1:d}"
                raise TypeError(err_msg.format('name', n_stars))

            # Check all the 2D arrays.
            arg_tab = ('x', 'y', 'm', 'xe', 'ye', 'me', 'name_in_list')

            for arg_test in arg_tab:
                if arg_test in kwargs:
                    if not isinstance(kwargs[arg_test], np.ndarray):
                        err_msg = "The '{0:s}' argument has to be a numpy array"
                        raise TypeError(err_msg.format(arg_test))

                    if kwargs[arg_test].shape != (n_stars, n_lists):
                        err_msg = "The '{0:s}' argument has to have shape = ({1:d}, {2:d})"
                        raise TypeError(err_msg.format(arg_test, n_stars, n_lists))

            # Check that the reference list is specified.
            if ref_list not in range(n_lists):
                err_msg = "The 'ref_list' argument has to be an integer between 0 and {0:d}"
                raise IndexError(err_msg.format(n_lists-1))

            # We have to have special handling of meta-data (i.e. info that has
            # dimensions of n_lists).
            meta_tab = ('list_times', 'list_names')
            meta_tab = ('list_times', 'list_names')
            meta_type = ((float, int), str)
            for mm in range(len(meta_tab)):
                meta_test = meta_tab[mm]
                meta_type_test = meta_type[mm]

                if meta_test in kwargs:
                    if len(kwargs[meta_test]) != n_lists:
                        err_msg = "The '{0:s}' argument has to have length = {1:d}"
                        raise ValueError(err_msg.format(meta_test, n_lists))

                    if not all(isinstance(tt, meta_type_test) for tt in kwargs[meta_test]):
                        err_msg = "The '{0:s}' argument has to be a list of {1:s}."
                        raise TypeError(err_msg.format(meta_test, str(meta_type_test)))

            #####
            # Create the startable
            #####
            Table.__init__(self, (kwargs['name'], kwargs['x'], kwargs['y'], kwargs['m']),
                           names=('name', 'x', 'y', 'm'))
            self['name'] = self['name'].astype('U20')
            self.meta = {'n_stars': n_stars, 'n_lists': n_lists, 'ref_list': ref_list}

            for meta_arg in meta_tab:
                if meta_arg in kwargs:
                    self.meta[meta_arg] = kwargs[meta_arg]
                    del kwargs[meta_arg]
                elif meta_arg.upper() in kwargs:
                    self.meta[meta_arg] = kwargs[meta_arg.upper()]
                    del kwargs[meta_arg]

            for arg in kwargs:
                if arg in ['name', 'x', 'y', 'm', 'list_times', 'list_names']:
                    continue
                else:
                    self.add_column(Column(data=kwargs[arg], name=arg))
                    if arg == 'name_in_list':
                        self['name_in_list'] = self['name_in_list'].astype('U20')
                    if arg == 'motion_model_input':
                        self['motion_model_input'] = self['motion_model_input'].astype('U20')
                    if arg == 'motion_model_used':
                        self['motion_model_used'] = self['motion_model_used'].astype('U20')
            #if 'motion_model_input' not in kwargs:
            #    self['motion_model_input'] = np.repeat(self.default_motion_model, len(self['name']))

        return

    def add_starlist(self, **kwargs):
        """
        Add data from a new list to an existing StarTable.
        Note, you can pass in the data via a StarList object or
        via a series of keywords with a 1D array on each.
        In either case, the number of stars must already match
        the existing number of stars in the StarTable.

        Example 1: Pass in data via StarList object.
        ----------
        print(t['x'].shape)
        t.add_starlist(starlist=my_list)
        print(t['x'].shape)   # Should be 1 column larger than before.

        Example 2: Pass in data via keywords and 1D arrays.
        t.add_starlist(x=x_new, y=y_new, m=m_new)

        """
        # Check if we are dealing with a StarList object or a
        # set of arguments with individual arrays.
        if 'starlist' in kwargs:
            self._add_list_data_from_starlist(kwargs['starlist'])
        else:
            self._add_list_data_from_keywords(**kwargs)

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
            if isinstance(self.meta[tab_key], collections.abc.Iterable) and (len(self.meta[tab_key]) == self.meta['n_lists']) and (not isinstance(self.meta[tab_key], str)):

                # If we find the key in the starlists' meta argument, then add the new values.
                # Otherwise, add "None".
                idx = lis_meta_keys_plural.index(tab_key) if tab_key in lis_meta_keys_plural else None
                if idx is not None:
                    lis_key = lis_meta_keys[idx]
                    self.meta[tab_key] = np.append(self.meta[tab_key], [starlist.meta[lis_key]])
                else:
                    self._append_invalid_meta_values(tab_key)

        # Update the n_lists meta keyword.
        self.meta['n_lists'] += 1

        return


    def _add_list_data_from_keywords(self, **kwargs):
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
            if (len(self[col_name].data.shape) == 2) and (col_name not in ['detect', 'n_detect']):      # Find the 2D columns
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
            if isinstance(self.meta[key], collections.abc.Iterable) and (len(self.meta[key]) == self.meta['n_lists']) and (not isinstance(self.meta[key], str)):
                # If we find the key is the passed in meta argument, then add the new values.
                # Otherwise, add "None".
                if 'meta' in kwargs:
                    new_meta_keys = kwargs['meta'].keys()
                    if key in new_meta_keys:
                        self.meta[key] = np.append(self.meta[key], [kwargs['meta'][key]])
                    else:
                        self._append_invalid_meta_values(key)
                else:
                    self._append_invalid_meta_values(key)

        # Update the n_lists meta keyword.
        self.meta['n_lists'] += 1

        return

    def _set_invalid_list_values(self, col_name, col_idx):
        """
        Set the contents of the specified column (in the 2D column objects)
        to an invalide value depending on the data type.
        """
        if np.issubdtype(self[col_name].info.dtype, np.integer):
            self[col_name][:, col_idx] = -1
        elif np.issubdtype(self[col_name].info.dtype, np.floating):
            self[col_name][:, col_idx] = np.nan
        else:
            self[col_name][:, col_idx] = None

        return

    def _set_invalid_star_values(self, col_name, row_idx):
        """
        Set the contents of the specified rows (in the 2D column objects)
        to an invalide value depending on the data type.
        """
        if np.issubdtype(self[col_name].info.dtype, np.integer):
            self[col_name][row_idx] = -1
        elif np.issubdtype(self[col_name].info.dtype, np.floating):
            self[col_name][row_idx] = np.nan
        else:
            self[col_name][row_idx] = None

        return

    def _append_invalid_meta_values(self, key):
        """
        For an existing meta keyword that is a list (already known),
        add an invalid value depending on the type.
        """
        if issubclass(type(self.meta[key][0]), np.integer):
            self.meta[key] = np.append(self.meta[key], [-1])
        elif issubclass(type(self.meta[key][0]), np.floating):
            self.meta[key] = np.append(self.meta[key], [np.nan])
        elif issubclass(type(self.meta[key][0]), str):
            self.meta[key] = np.append(self.meta[key], [''])
        else:
            self.meta[key] = np.append(self.meta[key], [None])

        # Print a warning message:
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


    def combine_lists_xym(self, weighted_xy=True, weighted_m=True, mask_lists=False, sigma=3):
        """
        For x, y and m columns in the table, collapse along the lists
        direction. For 'x', 'y' this means calculating the average position with
        outlier rejection. Optionally, weight by the 'xe' and 'ye' individual
        uncertainties. Optionally, use sigma clipping.
        "mask_lists" is a list with the indices of starlists that are
        excluded from the combination.
        Also, count the number of times a star is found in starlists.
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

        self.combine_lists('x', weights_col=weights_colx, mask_lists=mask_lists, sigma=sigma)
        self.combine_lists('y', weights_col=weights_coly, mask_lists=mask_lists, sigma=sigma)
        self.combine_lists('m', weights_col=weights_colm, mask_lists=mask_lists, sigma=sigma, ismag=True)

        return

    def combine_lists(self, col_name_in, weights_col=None, mask_val=None,
                      mask_lists=False, meta_add=True, ismag=False, sigma=3):
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
        """
        # Get the array we are going to combine.  Make a copy so we don't mod it.
        val_2d = copy.deepcopy( self[col_name_in].data )

        if ismag:
            # Convert to flux.
            val_2d = 10**(-val_2d / 2.5)
        # Make a mask of invalid (NaN) values and a user-specified invalid value.
        val_2d = np.ma.masked_invalid(val_2d)
        if mask_val:
            val_2d = np.ma.masked_values(val_2d, mask_val)

        if mask_lists is not False:
            # Remove a list
            if isinstance(mask_lists, list):
                if all(isinstance(item, int) for item in mask_lists):
                    val_2d.mask[:, mask_lists] = True

            # Throw a warning if mask_lists is not a list
            if not isinstance(mask_lists, list):
                raise RuntimeError('mask_lists needs to be a list.')

        # Decide if we are going to have weights (before we
        # do the expensive sigma clipping routine). Note that
        # if we have only 1 column to average, then we can't do weighting.
        if (weights_col and weights_col in self.colnames) and (val_2d.shape[1] > 1):
            err_2d = self[weights_col].data

            if ismag:
                # Convert to flux error
                err_2d = err_2d * val_2d * np.log(10) / 2.5

            np.seterr(divide='ignore')
            wgt_2d = np.ma.masked_invalid(1.0 / err_2d**2)
            np.seterr(divide='warn')

            if meta_add:
                self.meta[col_name_in + '0'] = 'weighted'
        else:
            wgt_2d = None
            if meta_add:
                self.meta[col_name_in + '0'] = 'not_weighted'

        # Figure out which ones are outliers. Returns a masked array.
        if sigma:
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            val_2d_clip = sigma_clipping.sigma_clip(val_2d, sigma=sigma, maxiters=5, axis=1)
            warnings.filterwarnings('default', category=RuntimeWarning)
        else:
            val_2d_clip = val_2d

            # Calculate the (weighted) mean and standard deviation along
        # the N_lists direction (axis=1).
        if wgt_2d is not None:
            avg = np.ma.average(val_2d_clip, weights=wgt_2d, axis=1)
            std = np.sqrt(np.ma.average((val_2d_clip.T - avg).T**2, weights=wgt_2d, axis=1))
        else:
            avg = np.ma.mean(val_2d_clip, axis=1)
            std = np.ma.std(val_2d_clip, axis=1)
        # To Do: bring the previous uncertainties of stars that are detected
        # in only one input frame.
        if (weights_col and weights_col in self.colnames) and (val_2d.shape[1] > 1):
            mask_for_singles = ((~np.isnan(val_2d_clip)).sum(axis=1)==1)
            std[mask_for_singles]=np.nanmean(err_2d[mask_for_singles], axis=1)

        # Save off our new AVG and STD into new columns with shape (N_stars).
        col_name_avg = col_name_in + '0'
        col_name_std = col_name_in + '0_err'

        if ismag:
            std = (2.5 / np.log(10)) * std / avg
            avg = -2.5 * np.ma.log10(avg)
        if col_name_avg in self.colnames:
            self[col_name_avg] = avg.data
            self[col_name_std] = std.data
        else:
            self.add_column(Column(data=avg.data, name=col_name_avg))
            self.add_column(Column(data=std.data, name=col_name_std))

        return

    def detections(self):
        """
        Find where stars are detected.
        # """
        n_detect = np.sum(~np.isnan(self['x']), axis=1)

        if 'n_detect' in self.colnames:
            self['n_detect'] = n_detect
        else:
            self.add_column(Column(n_detect), name='n_detect')

        return

    def fit_motion_model(
            self,
            motion_models=None,
            fixed_params_dict=None,
            weighting='var',
            use_scipy=False,
            absolute_sigma=True,
            select_stars=None,
            bootstrap=0,
            verbose=True,
            mask_value=None,
            mask_lists=None,
            fill_value=np.nan
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
        use_scipy : bool, optional
            Use scipy.optimize.curve_fit or algebraic solution (for Linear model only), by default False
        absolute_sigma : bool, optional
            Use absolute sigma or not, see scipy curve_fit for details, by default True
        select_stars : list of int, optional
            Indices of stars to fit, by default None (fit all stars)
        bootstrap : int, optional
            Number of bootstrap for uncertainty resampling, by default 0
        verbose : bool, optional
            Print verbose messages or not, by default True
        mask_value : float, optional
            Values to mask in data, by default None
        mask_lists : list of int, optional
            Indices of lists to mask/exclude from fitting, by default None
        fill_value : float, optional
            Fill value when there is not enough data points to fit, by default np.nan

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
            raise ValueError(f"fit_motion_model: Weighting must either be 'var' or 'std', not {weighting}!")

        if ('t' not in self.colnames) and ('list_times' not in self.meta):
            raise KeyError("fit_motion_model: Failed to access time values. No 't' column in table, no 'list_times' in meta.")

        # Check if we have the required columns
        if not all([_ in self.colnames for _ in ['x', 'y']]):
            raise KeyError(f"fit_motion_model: Missing required columns in the table: {', '.join(['x', 'y'])}!")

        # Make a copy of fixed_params_dict to avoid modifying the original one outside the function
        fixed_params_dict = copy.deepcopy(fixed_params_dict)

        # Check fixed_params_dict is a dict
        if fixed_params_dict is not None:
            if not isinstance(fixed_params_dict, dict):
                raise ValueError("fit_motion_model: fixed_params_dict must be a dictionary!")

        # Convert motion_models to MotionModel objects if they are strings:
        if motion_models is None:
            # Setting the default to None to avoid mutable default argument issue
            # See https://stackoverflow.com/questions/15189245/assigning-class-variable-as-default-value-to-class-method-argument
            motion_models = [Empty, Fixed, Linear]
        all_mm_map = motion_model.motion_model_map()
        if all(isinstance(mm, str) for mm in motion_models):
            mm_names = motion_models
            motion_models = [all_mm_map[mm] for mm in motion_models]
        else:
            mm_names = [mm.name for mm in motion_models]

        # Always add Empty and Fixed in motion models
        if 'Fixed' not in mm_names:
            motion_models.insert(0, Fixed)
        if 'Empty' not in mm_names:
            motion_models.insert(0, Empty)
        mm_names = [mm.name for mm in motion_models]

        # Construct motion models if motion_model_input column exists
        if 'motion_model_input' in self.colnames:
            input_mm_names = np.unique(self['motion_model_input'])
            assert all([name in all_mm_map.keys() for name in input_mm_names]), \
                f"fit_motion_model: Unknown motion model name(s) in 'motion_model_input' column. Available motion models are: {', '.join(all_mm_map.keys())}."
            for mm_name in input_mm_names:
                if mm_name not in mm_names:
                    motion_models.append(all_mm_map[mm_name])

            # Sort motion models by n_params
            motion_models = sorted(motion_models, key=lambda mm: mm.n_params)

        input_mm_map = {mm.name: mm for mm in motion_models}

        mm_n_params = np.sort([mm.n_params for mm in motion_models])
        if 'motion_model_input' not in self.colnames:
            # If motion_model_input column is not provided, assert that motion model n_params are unique and sorted
            # Otherwise the fitter does not know which motion model to use based on n_obs
            assert len(mm_n_params) == len(set(mm_n_params)), \
                f"fit_motion_model: Provided motion model n_params are not unique! Motion Models are: {[_.name for _ in motion_models]} Cannot decide which motion model to use based on n_obs. Please provide unique motion_models or a 'motion_model_input' column."


        ###########################
        ####### Prepare Data ######
        ###########################
        # Prepare data for fitting
        N_stars = len(self)
        x_data = np.ma.masked_invalid(self['x'].data, copy=True)
        y_data = np.ma.masked_invalid(self['y'].data, copy=True)
        xe_data = np.ma.masked_invalid(self['xe'].data, copy=True) if 'xe' in self.colnames else None
        ye_data = np.ma.masked_invalid(self['ye'].data, copy=True) if 'ye' in self.colnames else None
        # Mask out close to 0 values
        if xe_data is not None:
            xe_data.mask[np.isclose(xe_data, 0)] = True
        if ye_data is not None:
            ye_data.mask[np.isclose(ye_data, 0)] = True

        # If all of xe and ye is masked for a star, effectively no uncertainties provided, fill with 1.
        # Note that this automatically turn the mask to False for these stars
        if (xe_data is not None) and (ye_data is not None):
            fill_with_one = np.all(xe_data.mask, axis=1) & np.all(ye_data.mask, axis=1)
            xe_data[fill_with_one] = 1.
            ye_data[fill_with_one] = 1.

        # Ensure data is 2D for consistent indexing, even if we have only one list/epoch (shape (N_stars, 1) instead of (N_stars,))
        if np.ndim(x_data) == 1:
            x_data = x_data[:, np.newaxis]
        if np.ndim(y_data) == 1:
            y_data = y_data[:, np.newaxis]
        if np.ndim(xe_data) == 1:
            xe_data = xe_data[:, np.newaxis]
        if np.ndim(ye_data) == 1:
            ye_data = ye_data[:, np.newaxis]

        if mask_lists is not None:
            x_data.mask[:, mask_lists] = True
            y_data.mask[:, mask_lists] = True
            xe_data.mask[:, mask_lists] = True
            ye_data.mask[:, mask_lists] = True

        # t_data: 2d array with shape (N_stars, N_epochs)
        # t0: 1d array with shape (N_stars,)
        if 't' in self.colnames:
            t_data = copy.deepcopy(self['t'].data)
        else:
            t_data = copy.deepcopy(np.array(self.meta['list_times']))
            t_data = np.broadcast_to(t_data, x_data.shape)

        # Add default t0 if not provided in fixed_params_dict
        if fixed_params_dict is None:
            weights = 1/np.hypot(xe_data, ye_data) if (xe_data is not None) and (ye_data is not None) else None
            fixed_params_dict = {'t0': np.average(t_data, axis=1, weights=weights)}
        elif 't0' not in fixed_params_dict:
            weights = 1/np.hypot(xe_data, ye_data) if (xe_data is not None) and (ye_data is not None) else None
            fixed_params_dict['t0'] = np.average(t_data, axis=1, weights=weights)
        else:
            if np.ndim(fixed_params_dict['t0']) == 0:
                fixed_params_dict['t0'] = np.full(N_stars, fixed_params_dict['t0'])

        t0 = fixed_params_dict['t0']

        # Prepare fixed_params_dict for each star
        # This avoids checking types and slicing inside the fitting loop
        fixed_params_stars = [{} for _ in range(N_stars)]
        # Identify array parameters (length N_stars) and scalar parameters
        array_params = {k: v for k, v in fixed_params_dict.items() if np.ndim(v) > 0 and len(v) == N_stars}
        scalar_params = {k: v for k, v in fixed_params_dict.items() if k not in array_params}

        # Construct list of dicts for each star
        # Using list comprehension for speed
        fixed_params_stars = [
            {**scalar_params, **{k: v[i] for k, v in array_params.items()}}
            for i in range(N_stars)
        ]

        # Apply mask_value if provided
        if mask_value:
            x_data = np.ma.masked_values(x_data, mask_value)
            y_data = np.ma.masked_values(y_data, mask_value)
            if xe_data is not None:
                xe_data = np.ma.masked_values(xe_data, mask_value)
            if ye_data is not None:
                ye_data = np.ma.masked_values(ye_data, mask_value)


        # Calculate mask array
        xy_mask = ~ (x_data.mask | y_data.mask)
        if (xe_data is not None) and (ye_data is not None):
            xy_mask = xy_mask & (~ (xe_data.mask | ye_data.mask))

        # Calculate n_fit: unmasked x y values
        # This will be used to determine which motion model to use for each star. 
        # Note that we don't require unique times here
        # as scipy.curve_fit and Linear algebra can fit non-unique times.
        # self['n_fit'] = np.sum(xy_mask, axis=1)

        # Calculate n_fit: unique times & unmasked x y values
        self['n_fit'] = np.array([
            len(set(t_data[i][xy_mask[i]]))
            for i in range(N_stars)
        ])

        ###########################
        ####### Determine MM ######
        ###########################
        n_fit = np.array(self['n_fit'])
        if 'motion_model_input' in self.colnames:
            # Determine which motion model to use based on motion_model_input column
            # If n_fit < required n_params for the input motion model, use the most complicated motion model with n_fit >= n_params
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

        # Add default obsLocation if not provided in fixed_params_dict
        mm_used = np.unique(self['motion_model_used'].name)
        if 'Parallax' in mm_used and 'obsLocation' not in fixed_params_dict:
            fixed_params_dict['obsLocation'] = 'earth'

        ############################
        ####### Prepare Table ######
        ############################
        # Fill table with all possible motion model parameter names as new columns.
        motion_model_used = [all_mm_map[name] for name in np.unique(self['motion_model_used'])]
        new_col_list = motion_model.motion_model_param_names(motion_model_used, with_errors=True, with_fixed=False)
        new_col_list += ['chi2_x', 'chi2_y', 'n_params']

        if 't0' not in new_col_list:
            new_col_list.append('t0')

        # Add new columns if they do not exist
        for col in new_col_list:
            if col in self.colnames:
                # Keep old data if the column already exists
                continue
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

        # Add fixed parameter columns if they do not exist
        fixed_param_names = []
        for mm in motion_model_used:
            for param in mm.fixed_param_names:
                if param not in fixed_param_names:
                    fixed_param_names.append(param)
        # Remove t0 from fixed_param_names as it will be saved during fitting
        if 't0' in fixed_param_names:
            fixed_param_names.remove('t0')

        # Add fixed parameter columns
        for param in fixed_param_names:
            coldata = np.array([fixed_params_stars[i][param] for i in range(N_stars)])
            if param in self.colnames:
                if is_string_dtype(self[param]):
                    if np.array_equal(self[param], coldata):
                        # Same data, skip
                        continue
                # If the column already exists, check if the data are the same
                elif np.allclose(self[param], coldata, equal_nan=True):
                    # Same data, skip
                    continue
                else:
                    # Different data, add with _mm suffix to avoid name conflict
                    colname = param + '_mm'
            else:
                colname = param

            self.add_column(Column(data=coldata, name=colname), rename_duplicate=True)


        # Add a column to keep track of the number of points used in a fit and number of bootstrap used.
        self.meta['n_bootstrap'] = bootstrap


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

        # Unmasked indices for each star:
        unmasked_idx = [np.flatnonzero(xy_mask[i]) for i in range(N_stars)]

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

            # Prepare data as lists of arrays for faster access during fitting
            t_stars = [np.array(t_data[i][unmasked_idx[i]]) for i in unique_index]
            x_stars = [np.array(x_data[i][unmasked_idx[i]]) for i in unique_index]
            y_stars = [np.array(y_data[i][unmasked_idx[i]]) for i in unique_index]
            xe_stars = [np.array(xe_data[i][unmasked_idx[i]]) for i in unique_index] if xe_data is not None else [None]*n_stars_this_model
            ye_stars = [np.array(ye_data[i][unmasked_idx[i]]) for i in unique_index] if ye_data is not None else [None]*n_stars_this_model

            # For each star
            # Expensive for loop! Prepare everything beforehand to speed up.
            if len(unique_index) > 0:
                for idx, i_star in enumerate(tqdm(unique_index, disable=not verbose, desc=f"Fitting motion model {unique_motion_model}")):
                    # Fit the star
                    params, param_errs, chi2_x, chi2_y = motion_model_instance.fit(
                        t=t_stars[idx],
                        x=x_stars[idx],
                        y=y_stars[idx],
                        xe=xe_stars[idx],
                        ye=ye_stars[idx],
                        fixed_params_dict=fixed_params_stars[i_star],
                        weighting=weighting,
                        use_scipy=use_scipy,
                        absolute_sigma=absolute_sigma,
                        bootstrap=bootstrap,
                        fill_value=fill_value,
                        return_chi2=True,
                        verbose=verbose
                    )
                    params_array[idx] = params
                    param_errs_array[idx] = param_errs
                    chi2_x_array[idx] = chi2_x
                    chi2_y_array[idx] = chi2_y

            # Store results back to the table
            for j, param_name in enumerate(param_names):
                self[param_name][unique_index] = params_array[:, j]
                self[param_name + '_err'][unique_index] = param_errs_array[:, j]
            self['chi2_x'][unique_index] = chi2_x_array
            self['chi2_y'][unique_index] = chi2_y_array
            self['n_params'][unique_index] = motion_model_instance.n_params
            self['t0'][unique_index] = t0[unique_index]
        return

    def infer_positions(self, times, fixed_params_dict=None, fill_value=np.nan):
        """Infer star positions at given times using fitted motion models.

        Parameters
        ----------
        times : array_like
            Times at which to predict positions. Scalar, or (N_times,) array, or (N_stars, N_times) array.
        fixed_params_dict : None or dict, optional
            Dictionary of fixed parameters to use for prediction.
            If not provided, will try to look for fixed parameters in the table columns.
            If fixed params are found in both the table and the fixed_params_dict, the values in the table will be used and the fixed_params_dict values will be ignored,
            by default None
        fill_value : float, optional
            Value to use for missing data, by default np.nan

        Returns
        -------
        x, y, xe, ye : ndarray
            Arrays of predicted x, y positions and their uncertainties xe, ye, with shape (N_stars, N_times) or (N_stars,) if N_times=1, or (N_times,) if N_stars=1, or scalar.
        """
        assert 'motion_model_used' in self.colnames, \
            "infer_positions: 'motion_model_used' column not found in the table. Please run fit_motion_model() first."

        N_stars = len(self)
        times = np.atleast_1d(times)
        N_times = len(times)

        if (N_stars > 1) and (N_times > 1):
            x_pred = np.full((N_stars, N_times), fill_value, dtype=float)
            y_pred = np.full((N_stars, N_times), fill_value, dtype=float)
            xe_pred = np.full((N_stars, N_times), np.inf, dtype=float)
            ye_pred = np.full((N_stars, N_times), np.inf, dtype=float)
        elif N_stars==1:
            x_pred = np.full(N_times, fill_value, dtype=float)
            y_pred = np.full(N_times, fill_value, dtype=float)
            xe_pred = np.full(N_times, np.inf, dtype=float)
            ye_pred = np.full(N_times, np.inf, dtype=float)
        else:
            x_pred = np.full(N_stars, fill_value, dtype=float)
            y_pred = np.full(N_stars, fill_value, dtype=float)
            xe_pred = np.full(N_stars, np.inf, dtype=float)
            ye_pred = np.full(N_stars, np.inf, dtype=float)

        # Calculate the dictionary of {motion_model: indices of stars with this motion model} for faster access during prediction
        unique_motion_models, unique_inv_indices = np.unique(self['motion_model_used'], return_inverse=True)
        indices_by_motion_model = {key: np.flatnonzero(unique_inv_indices == k) for k, key in enumerate(unique_motion_models)}

        # Prepare fit_params, fixed_params, fit_param_errs for each star
        for unique_motion_model, unique_index in indices_by_motion_model.items():
            # Create motion model instance
            motion_model_instance = motion_model.motion_model_map()[unique_motion_model]()
            # Prepare parameters for prediction
            fit_params = np.array([
                self[param_name][unique_index] for param_name in motion_model_instance.fit_param_names
            ]).T # shape (N_stars_this_model, N_params)

            fit_param_errs = np.array([
                self[param_name + '_err'][unique_index] for param_name in motion_model_instance.fit_param_names
            ]).T # shape (N_stars_this_model, N_params)

            fixed_params = {}
            for param_name in motion_model_instance.fixed_param_names:
                col_name = copy.deepcopy(param_name)
                # If column not in table, check if it's provided in fixed_params_dict. If not, raise error. If provided, use the value from fixed_params_dict for all stars.
                if (col_name not in self.colnames) and (f'{col_name}_mm' not in self.colnames):
                    if col_name in fixed_params_dict:
                        fixed_params[param_name] = fixed_params_dict[col_name]
                        continue
                    else:
                        raise KeyError(f"infer_positions: Fixed parameter '{param_name}' not found in table columns or fixed_params_dict. Please provide the value for this parameter in fixed_params_dict or add a column named '{param_name}' to the table.")

                # If original table has column and fit_motion_model added the column with _mm suffix, use the _mm column for prediction.
                if param_name + '_mm' in self.colnames:
                    col_name = param_name + '_mm'
                fixed_params[param_name] = self[col_name][unique_index]

                if (param_name == 'obsLocation'):
                    assert np.unique(fixed_params[param_name]).size == 1, \
                        "infer_positions: obsLocation fixed parameter has different values for different stars. Vectorized handling not implemented yet."
                    fixed_params[param_name] = fixed_params[param_name][0]

            # Predict positions
            x, y, xe, ye = motion_model_instance.model(
                times, fit_params, fit_param_errs, fixed_params
            )
            x_pred[unique_index] = x
            y_pred[unique_index] = y
            xe_pred[unique_index] = xe
            ye_pred[unique_index] = ye

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



    def shift_reference_frame(self, delta_vx=0.0, delta_vy=0.0, delta_pi=0.0,
                                motion_model_dict={}):
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
        motion_model_dict = motion_model.validate_motion_model_dict(motion_model_dict, self, None)
        if delta_vx==0.0 and delta_vy==0.0 and delta_pi==0.0:
            print("No shifts input, reference frame unchanged.")
            print("Specify delta_vx, delta_vy, and/or delta_pi to perform a reference frame shift.")
            return
        self['vx'] += delta_vx
        self['x'] += delta_vx*(self['t']-self['t0'][:, np.newaxis])
        self['vy'] += delta_vy
        self['y'] += delta_vy*(self['t']-self['t0'][:, np.newaxis])
        if delta_pi!=0.0:
            t_all = self['t'][np.where(~np.any(np.isnan(self['t']), axis=1))[0][0]]
            t_mjd = Time(t_all, format='decimalyear', scale='utc').mjd
            pvec = motion_model_dict['Parallax'].get_parallax_vector(t_mjd)
            self['pi'] += delta_pi
            self['x'] += delta_pi*pvec[0]
            self['y'] += delta_pi*pvec[1]
        return

def shift_reference_frame(table, delta_vx=0.0, delta_vy=0.0, delta_pi=0.0,
                            motion_model_dict={}):
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
    motion_model_dict = motion_model.validate_motion_model_dict(motion_model_dict, table, None)
    if delta_vx==0.0 and delta_vy==0.0 and delta_pi==0.0:
        print("No shifts input, reference frame unchanged.")
        print("Specify delta_vx, delta_vy, and/or delta_pi to perform a reference frame shift.")
        return
    table['vx'] += delta_vx
    table['x'] += delta_vx*(table['t']-table['t0'][:, np.newaxis])
    table['vy'] += delta_vy
    table['y'] += delta_vy*(table['t']-table['t0'][:, np.newaxis])
    if delta_pi!=0.0:
        t_all = table['t'][np.where(~np.any(np.isnan(table['t']), axis=1))[0][0]]
        t_mjd = Time(t_all, format='decimalyear', scale='utc').mjd
        pvec = motion_model_dict['Parallax'].get_parallax_vector(t_mjd)
        table['pi'] += delta_pi
        table['x'] += delta_pi*pvec[0]
        table['y'] += delta_pi*pvec[1]
    return table
