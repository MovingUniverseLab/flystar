from astropy.table import Table, Column, hstack
from astropy.stats import sigma_clipping
from scipy.optimize import curve_fit
from flystar.fit_velocity import linear_fit, calc_chi2, linear, fit_velocity
from tqdm import tqdm
import numpy as np
import warnings
import collections
import pdb
import time
import copy


class StarTable(Table):
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
    def __init__(self, *args, ref_list=0, **kwargs):
        """
        """
        
        # Check if the required arguments are present
        arg_req = ('name', 'x', 'y', 'm')
        
        found_all_required = True
        for arg_test in arg_req:
            if arg_test not in kwargs:
                found_all_required = False

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
            meta_tab = ('LIST_TIMES', 'LIST_NAMES')
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

            for arg in kwargs:
                if arg in ['name', 'x', 'y', 'm']:
                    continue
                else:
                    self.add_column(Column(data=kwargs[arg], name=arg))
                    if arg == 'name_in_list':
                        self['name_in_list'] = self['name_in_list'].astype('U20')

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
            if isinstance(self.meta[tab_key], collections.abc.Iterable) and (len(self.meta[tab_key]) == self.meta['n_lists']):

                # If we find the key in the starlists' meta argument, then add the new values.
                # Otherwise, add "None".
                idx = np.where(lis_meta_keys_plural == tab_key)[0]
                if len(idx) > 0:
                    lis_key = lis_meta_keys[idx[0]]
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
            if isinstance(self.meta[key], collections.abc.Iterable) and (len(self.meta[key]) == self.meta['n_lists']):
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
        col_name_std = col_name_in + '0e'

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

    
    def fit_velocities(self, weighting='var', use_scipy=True, absolute_sigma=True, bootstrap=0, fixed_t0=False, verbose=False,
                       mask_val=None, mask_lists=False, show_progress=True):
        """Fit velocities for all stars in the table and add to the columns 'vx', 'vxe', 'vy', 'vye', 'x0', 'x0e', 'y0', 'y0e'.

        Parameters
        ----------
        weighting : str, optional
            Weight by variance 'var' or standard deviation 'std', by default 'var'
        use_scipy : bool, optional
            Use scipy.curve_fit (recommended for large number of epochs, but may return inf or nan) or analytic fitting from flystar.fit_velocity.linear_fit (recommended for a few epochs), by default True
        absolute_sigma : bool, optional
            Absolute sigma or not. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html for details, by default True
        bootstrap : int, optional
            Calculate uncertain using bootstraping or not, by default 0
        fixed_t0 : bool or array-like, optional
            Fix the t0 in dt = time - t0 if user provides an array with the same length of the table, or automatically calculate t0 = np.average(time, weights=1/np.hypot(xe, ye)) if False, by default False
        verbose : bool, optional
            Output verbose information or not, by default False
        mask_val : float, optional
            Value that needs to be masked in the data, e.g. -100000, by default None
        mask_lists : list, optional
            Columns that needs to be masked, by default False
        show_progress : bool, optional
            Show progress bar or not, by default True

        Raises
        ------
        ValueError
            If weighting is neither 'var' or 'std'
        KeyError
            If there's not time information in the table
        """
        if weighting not in ['var', 'std']:
            raise ValueError(f"fit_velocities: Weighting must either be 'var' or 'std', not {weighting}!")
        
        if ('t' not in self.colnames) and ('LIST_TIMES' not in self.meta):
            raise KeyError("fit_velocities: Failed to access time values. No 't' column in table, no 'LIST_TIMES' in meta.")
        
        # Check if we have the required columns
        if not all([_ in self.colnames for _ in ['x', 'y']]):
            raise KeyError(f"fit_velocities: Missing required columns in the table: {', '.join(['x', 'y'])}!")
        
        N_stars = len(self)

        if verbose:
            start_time = time.time()
            msg = 'Starting startable.fit_velocities for {0:d} stars with n={1:d} bootstrap'
            print(msg.format(N_stars, bootstrap))

        # Clean/remove up old arrays.
        if 'x0' in self.colnames: self.remove_column('x0')
        if 'vx' in self.colnames: self.remove_column('vx')
        if 'y0' in self.colnames: self.remove_column('y0')
        if 'vy' in self.colnames: self.remove_column('vy')
        if 'x0e' in self.colnames: self.remove_column('x0e')
        if 'vxe' in self.colnames: self.remove_column('vxe')
        if 'y0e' in self.colnames: self.remove_column('y0e')
        if 'vye' in self.colnames: self.remove_column('vye')
        if 'chi2_vx' in self.colnames: self.remove_column('chi2_vx')
        if 'chi2_vy' in self.colnames: self.remove_column('chi2_vy')
        if 't0' in self.colnames: self.remove_column('t0')
        if 'n_vfit' in self.colnames: self.remove_column('n_vfit')
        
        # Define output arrays for the best-fit parameters.
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'x0'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vx'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'y0'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vy'))
        
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'x0e'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vxe'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'y0e'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'vye'))
        
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'chi2_vx'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 'chi2_vy'))
        
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 't0'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=int), name = 'n_vfit'))

        self.meta['N_VFIT_BOOTSTRAP'] = bootstrap

        # (FIXME: Do we need to catch the case where there's a single *unmasked* epoch?)
        # Catch the case when there is only a single epoch. Just return 0 velocity
        # and the same input position for the x0/y0.
        if len(self['x'].shape) == 1:
            self['x0'] = self['x']
            self['y0'] = self['y']
            if 't' in self.colnames:
                self['t0'] = self['t']
            else:
                self['t0'] = self.meta['LIST_TIMES'][0]
            if 'xe' in self.colnames:
                self['x0e'] = self['xe']
                self['y0e'] = self['ye']
            self['n_vfit'] = 1

            return
        
        if self['x'].shape[1] == 1:
            self['x0'] = self['x'][:,0]
            self['y0'] = self['y'][:,0]

            if 't' in self.colnames:
                self['t0'] = self['t'][:, 0]
            else:
                self['t0'] = self.meta['LIST_TIMES'][0]

            if 'xe' in self.colnames:
                self['x0e'] = self['xe'][:,0]
                self['y0e'] = self['ye'][:,0]

            self['n_vfit'] = 1

            return

        # STARS LOOP through the stars and work on them 1 at a time.
        # This is slow; but robust.
        if show_progress:
            for ss in tqdm(range(N_stars)):
                self.fit_velocity_for_star(ss, weighting=weighting, use_scipy=use_scipy, absolute_sigma=absolute_sigma, bootstrap=bootstrap, fixed_t0=fixed_t0,
                                    mask_val=mask_val, mask_lists=mask_lists)
        else:
            for ss in range(N_stars):
                self.fit_velocity_for_star(ss, weighting=weighting, use_scipy=use_scipy, absolute_sigma=absolute_sigma, bootstrap=bootstrap, fixed_t0=fixed_t0,
                                        mask_val=mask_val, mask_lists=mask_lists, )
        if verbose:
            stop_time = time.time()
            print('startable.fit_velocities runtime = {0:.0f} s for {1:d} stars'.format(stop_time - start_time, N_stars))
        
        return

    def fit_velocity_for_star(self, ss, weighting='var', use_scipy=True, absolute_sigma=True, bootstrap=False, fixed_t0=False,
                              mask_val=None, mask_lists=False):

        # Make a mask of invalid (NaN) values and a user-specified invalid value.
        x = np.ma.masked_invalid(self['x'][ss, :].data)
        y = np.ma.masked_invalid(self['y'][ss, :].data)
        if mask_val:
            x = np.ma.masked_values(x, mask_val)
            y = np.ma.masked_values(y, mask_val)
            # If no mask, convert x.mask to list
            if not np.ma.is_masked(x):
                x.mask = np.zeros_like(x.data, dtype=bool)
            if not np.ma.is_masked(y):
                y.mask = np.zeros_like(y.data, dtype=bool)
                
        
        if mask_lists is not False:
            # Remove a list
            if isinstance(mask_lists, list):
                if all(isinstance(item, int) for item in mask_lists):
                    x.mask[mask_lists] = True
                    y.mask[mask_lists] = True
                
            # Throw a warning if mask_lists is not a list
            if not isinstance(mask_lists, list):
                raise RuntimeError('mask_lists needs to be a list.')

        if 'xe' in self.colnames:
            # Make a mask of invalid (NaN) values and a user-specified invalid value.
            xe = np.ma.masked_invalid(self['xe'][ss, :].data)
            ye = np.ma.masked_invalid(self['ye'][ss, :].data)

            # Catch the case where we have positions but no errors for
            # some of the entries... we need to "fill in" reasonable
            # weights for these... just use the average weights over
            # all the other epochs.
            pos_no_err = np.where((np.isfinite(x) & np.isfinite(y)) &
                                  (np.isfinite(xe) == False) & (np.isfinite(ye) == False))[0]
            pos_with_err = np.where((np.isfinite(x) & np.isfinite(y)) &
                                  (np.isfinite(xe) & np.isfinite(ye)))[0]

            if len(pos_with_err) > 0:
                xe[pos_no_err] = xe[pos_with_err].mean()
                ye[pos_no_err] = ye[pos_with_err].mean()
            else:
                xe[pos_no_err] = 1.0
                ye[pos_no_err] = 1.0
        else:
            N_epochs = len(x)
            xe = np.ones(N_epochs, dtype=float)
            ye = np.ones(N_epochs, dtype=float)
            xe = np.ma.masked_invalid(xe)
            ye = np.ma.masked_invalid(xe)

        if mask_val:
            xe = np.ma.masked_values(xe, mask_val)
            ye = np.ma.masked_values(ye, mask_val)
            # If no mask, convert xe.mask to list
            if not np.ma.is_masked(xe):
                xe.mask = np.zeros_like(xe.data, dtype=bool)
            if not np.ma.is_masked(ye):
                ye.mask = np.zeros_like(ye.data, dtype=bool)
            
        if mask_lists is not False:
            # Remove a list
            if isinstance(mask_lists, list):
                if all(isinstance(item, int) for item in mask_lists):
                    xe.mask[mask_lists] = True
                    ye.mask[mask_lists] = True
                    
            # Throw a warning if mask_lists is not a list
            if not isinstance(mask_lists, list):
                raise RuntimeError('mask_lists needs to be a list.')    

        # Make a mask of invalid (NaN) values and a user-specified invalid value.
        if 't' in self.colnames:
            t = np.ma.masked_invalid(self['t'][ss, :].data)
        else:
            t = np.ma.masked_invalid(self.meta['LIST_TIMES'])

        if mask_val:
            t = np.ma.masked_values(t, mask_val)
            if not np.ma.is_masked(t):
                t.mask = np.zeros_like(t.data, dtype=bool)
            
        if mask_lists is not False:
            # Remove a list
            if isinstance(mask_lists, list):
                if all(isinstance(item, int) for item in mask_lists):
                    t.mask[mask_lists] = True
                    
            # Throw a warning if mask_lists is not a list
            if not isinstance(mask_lists, list):
                raise RuntimeError('mask_lists needs to be a list.')    
        
        # For inconsistent masks, mask the star if any of the values are masked.
        new_mask = np.logical_or.reduce((t.mask, x.mask, y.mask, xe.mask, ye.mask))
        # Figure out where we have detections (as indicated by error columns)
        good = np.where((xe != 0) & (ye != 0) &
                        np.isfinite(xe) & np.isfinite(ye) &
                        np.isfinite(x) & np.isfinite(y) & ~new_mask)[0]

        N_good = len(good)

        # Catch the case where there is NO good data. 
        if N_good == 0:
            return

        # Everything below has N_good >= 1
        x = x[good]
        y = y[good]
        t = t[good]
        xe = xe[good]
        ye = ye[good]

        # slope, intercept
        p0x = np.array([0., x.mean()])
        p0y = np.array([0., y.mean()])
        
        # Unless t0 is fixed, calculate the t0 for the stars.
        if fixed_t0 is False:
            t_weight = 1.0 / np.hypot(xe, ye)
            t0 = np.average(t, weights=t_weight)
        else:
            t0 = fixed_t0[ss]
        dt = t - t0

        self['t0'][ss] = t0
        self['n_vfit'][ss] = N_good

        # Catch the case where all the times are identical
        if (dt == dt[0]).all():
            if weighting == 'var':
                wgt_x = (1.0/xe)**2
                wgt_y = (1.0/ye)**2
            elif weighting == 'std':
                wgt_x = 1./np.abs(xe)
                wgt_y = 1./np.abs(ye)

            self['x0'][ss] = np.average(x, weights=wgt_x)
            self['y0'][ss] = np.average(y, weights=wgt_y)
            self['x0e'][ss] = np.sqrt(np.average((x - self['x0'][ss])**2, weights=wgt_x))
            self['y0e'][ss] = np.sqrt(np.average((y - self['y0'][ss])**2, weights=wgt_x))
            
            self['vx'][ss] = 0.0
            self['vy'][ss] = 0.0
            self['vxe'][ss] = 0.0
            self['vye'][ss] = 0.0

            return

        # Catch the case where we have enough measurements to actually
        # fit a velocity!
        if N_good > 2:
            if weighting == 'var':
                sigma_x = xe
                sigma_y = ye
            elif weighting == 'std':
                sigma_x = np.abs(xe)**0.5
                sigma_y = np.abs(ye)**0.5
            
            if use_scipy:
                vx_opt, vx_cov = curve_fit(linear, dt, x, p0=p0x, sigma=sigma_x, absolute_sigma=absolute_sigma)
                vy_opt, vy_cov = curve_fit(linear, dt, y, p0=p0y, sigma=sigma_y, absolute_sigma=absolute_sigma)
                vx = vx_opt[0]
                x0 = vx_opt[1]
                vy = vy_opt[0]
                y0 = vy_opt[1]
                chi2_vx = calc_chi2(dt, x, sigma_x, *vx_opt)
                chi2_vy = calc_chi2(dt, y, sigma_y, *vy_opt)
            
            else:
                result_vx = linear_fit(dt, x, sigma_x, absolute_sigma=absolute_sigma)
                result_vy = linear_fit(dt, y, sigma_y, absolute_sigma=absolute_sigma)
                vx = result_vx['slope']
                x0 = result_vx['intercept']
                vy = result_vy['slope']
                y0 = result_vy['intercept']
                chi2_vx = result_vx['chi2']
                chi2_vy = result_vy['chi2']
            
            self['vx'][ss] = vx
            self['x0'][ss] = x0
            self['vy'][ss] = vy
            self['y0'][ss] = y0
            self['chi2_vx'][ss] = chi2_vx
            self['chi2_vy'][ss] = chi2_vy
            
            # Run the bootstrap
            if bootstrap > 0:
                edx = np.arange(N_good, dtype=int)

                vx_b = np.zeros(bootstrap, dtype=float)
                x0_b = np.zeros(bootstrap, dtype=float)
                vy_b = np.zeros(bootstrap, dtype=float)
                y0_b = np.zeros(bootstrap, dtype=float)
            
                for bb in range(bootstrap):
                    bdx = np.random.choice(edx, N_good)
                    if weighting == 'var':
                        sigma_x_b = xe[bdx]
                        sigma_y_b = ye[bdx]
                    elif weighting == 'std':
                        sigma_x_b = xe[bdx]**0.5
                        sigma_y_b = ye[bdx]**0.5
                    
                    if use_scipy:
                        vx_opt_b, vx_cov_b = curve_fit(linear, dt[bdx], x[bdx], p0=vx_opt, sigma=sigma_x_b,
                                                        absolute_sigma=absolute_sigma)
                        vy_opt_b, vy_cov_b = curve_fit(linear, dt[bdx], y[bdx], p0=vy_opt, sigma=sigma_y_b,
                                                        absolute_sigma=absolute_sigma)
                        vx_b[bb] = vx_opt_b[0]
                        x0_b[bb] = vx_opt_b[1]
                        vy_b[bb] = vy_opt_b[0]
                        y0_b[bb] = vy_opt_b[1]
                        
                    else:
                        result_vx_b = linear_fit(dt[bdx], x[bdx], sigma=sigma_x_b, absolute_sigma=absolute_sigma)
                        result_vy_b = linear_fit(dt[bdx], y[bdx], sigma=sigma_y_b, absolute_sigma=absolute_sigma)
                        vx_b[bb] = result_vx_b['slope']
                        x0_b[bb] = result_vx_b['intercept']
                        vy_b[bb] = result_vy_b['slope']
                        y0_b[bb] = result_vy_b['intercept']
                
                # Save the errors from the bootstrap
                self['vxe'][ss] = vx_b.std()
                self['x0e'][ss] = x0_b.std()
                self['vye'][ss] = vy_b.std()
                self['y0e'][ss] = y0_b.std()
                
            else:
                if use_scipy:
                    vxe, x0e = np.sqrt(vx_cov.diagonal())
                    vye, y0e = np.sqrt(vy_cov.diagonal())
                else:
                    vxe = result_vx['e_slope']
                    x0e = result_vx['e_intercept']
                    vye = result_vy['e_slope']
                    y0e = result_vy['e_intercept']
                    
                self['vxe'][ss] = vxe
                self['x0e'][ss] = x0e
                self['vye'][ss] = vye
                self['y0e'][ss] = y0e

        elif N_good == 2:
            # Not enough epochs to fit a velocity.            
            dx = np.diff(x)[0]
            dy = np.diff(y)[0]
            dt_diff = np.diff(dt)[0]
            
            if weighting == 'var':
                sigma_x = 1./xe**2
                sigma_y = 1./ye**2
            elif weighting == 'std':
                sigma_x = 1./np.abs(xe)
                sigma_y = 1./np.abs(ye)
            
            self['x0'][ss] = np.average(x, weights=sigma_x)
            self['y0'][ss] = np.average(y, weights=sigma_y)
            self['x0e'][ss] = np.abs(dx) / 2**0.5
            self['y0e'][ss] = np.abs(dy) / 2**0.5
            self['vx'][ss] = dx / dt_diff
            self['vy'][ss] = dy / dt_diff
            self['vxe'][ss] = 0.0
            self['vye'][ss] = 0.0
            self['chi2_vx'][ss] = calc_chi2(dt, x, sigma_x, self['vx'][ss], self['x0'][ss])
            self['chi2_vy'][ss] = calc_chi2(dt, y, sigma_y, self['vy'][ss], self['y0'][ss])
            
        else:
            # N_good == 1 case
            self['n_vfit'][ss] = 1
            self['x0'][ss] = x
            self['y0'][ss] = y
            
            if 'xe' in self.colnames:
                self['x0e'] = xe
                self['y0e'] = ye

        return
    
    
    def fit_velocities_all_detected(self, weighting='var', use_scipy=False, absolute_sigma=False, epoch_cols='all', mask_val=None, art_star=False, return_result=False):
        """Fit velocities for stars detected in all epochs specified by epoch_cols. 
        Criterion: xe/ye error > 0 and finite, x/y not masked.

        Parameters
        ----------
        weighting : str, optional
            Variance weighting('var') or standard deviation weighting ('std'), by default 'var'
        use_scipy : bool, optional
            Use scipy.curve_fit or flystar.fit_velocity.fit_velocity, by default False
        absolute_sigma : bool, optional
            Absolute sigma or rescaled sigma, by default False
        epoch_cols : str or list of intergers, optional
            List of epoch column indices used for fitting velocity, by default 'all'
        mask_val : float, optional
            Values in x, y to be masked
        art_star : bool, optional
            Artificial star or observation star catalog. If artificial star, use 'det' column to select stars detected in all epochs, by default False
        return_result : bool, optional
            Return the velocity results or not, by default False
        
        Returns
        -------
        vel_result : astropy Table
            Astropy Table with velocity results
        """
        
        N_stars = len(self)
        
        if epoch_cols == 'all':
            epoch_cols = np.arange(np.shape(self['x'])[1])
            
        # Artificial Star
        if art_star:
            detected_in_all_epochs = np.all(self['det'][:, epoch_cols], axis=1)
        
        # Observation Star
        else:
            valid_xe = np.all(self['xe'][:, epoch_cols]!=0, axis=1) & np.all(np.isfinite(self['xe'][:, epoch_cols]), axis=1)
            valid_ye = np.all(self['ye'][:, epoch_cols]!=0, axis=1) & np.all(np.isfinite(self['ye'][:, epoch_cols]), axis=1)
            
            if mask_val:
                x = np.ma.masked_values(self['x'][:, epoch_cols], mask_val)
                y = np.ma.masked_values(self['y'][:, epoch_cols], mask_val)
                
                # If no mask, convert x.mask to list
                if not np.ma.is_masked(x):
                    x.mask = np.zeros_like(self['x'][:, epoch_cols].data, dtype=bool)
                if not np.ma.is_masked(y):
                    y.mask = np.zeros_like(self['y'][:, epoch_cols].data, dtype=bool)
                
                valid_x = ~np.any(x.mask, axis=1)
                valid_y = ~np.any(y.mask, axis=1)
                detected_in_all_epochs = np.logical_and.reduce((
                    valid_x, valid_y, valid_xe, valid_ye
                ))
            else:
                detected_in_all_epochs = np.logical_and(valid_xe, valid_ye)
        
        
        # Fit velocities        
        vel_result = fit_velocity(self[detected_in_all_epochs], weighting=weighting, use_scipy=use_scipy, absolute_sigma=absolute_sigma, epoch_cols=epoch_cols, art_star=art_star)
        vel_result = Table.from_pandas(vel_result)
        
        
        # Add n_vfit
        n_vfit = len(epoch_cols)
        vel_result['n_vfit'] = n_vfit
        
        # Clean/remove up old arrays.
        columns = [*vel_result.keys(), 'n_vfit']
        for column in columns:
            if column in self.colnames: self.remove_column(column)
        
        # Update self
        for column in columns:
            column_array = np.ma.zeros(N_stars)
            column_array[detected_in_all_epochs] = vel_result[column]
            column_array[~detected_in_all_epochs] = np.nan
            column_array.mask = ~detected_in_all_epochs
            self[column] = column_array
        
        if return_result:
            return vel_result
        else:
            return