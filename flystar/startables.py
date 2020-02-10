from astropy.table import Table, Column
from astropy.stats import sigma_clipping
from scipy.optimize import curve_fit
import numpy as np
import warnings
import collections
import pdb
import time


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
            if isinstance(self.meta[tab_key], collections.Iterable) and (len(self.meta[tab_key]) == self.meta['n_lists']):

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
            if isinstance(self.meta[key], collections.Iterable) and (len(self.meta[key]) == self.meta['n_lists']):
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
        "mask_lists" is an array with the indeces of lists that excluded from
        the combination.
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
        
        "mask_lists" is an array with the indices of lists that excluded from
        the combination.
        
        A flag can be stored in the metadata to record if the average was
        weighted or not.
        """
        # Get the array we are going to combine. 
        val_2d = self[col_name_in].data

        if ismag:
            # Convert to flux.
            val_2d = 10**(-val_2d / 2.5)

        # Make a mask of invalid (NaN) values and a user-specified invalid value.
        val_2d = np.ma.masked_invalid(val_2d)
        if mask_val:
            val_2d = np.ma.masked_values(val_2d, mask_val)
        
        # Remove a list
        if isinstance(mask_lists, list):
            if all(isinstance(item, int) for item in mask_lists):
                val_2d.mask[:, mask_lists] = True

        # Dedicde if we are going to have weights (before we
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

    
    def fit_velocities(self, bootstrap=0, verbose=False):
        """
        Fit velocities for all stars in the table. 
        """
        if ('t' not in self.colnames) and ('list_times' not in self.meta):
            raise RuntimeError('fit_velocities: Failed to time values.')

        N_stars, N_epochs = self['x'].shape

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
        
        self.add_column(Column(data = np.zeros(N_stars, dtype=float), name = 't0'))
        self.add_column(Column(data = np.zeros(N_stars, dtype=int), name = 'n_vfit'))

        self.meta['n_vfit_bootstrap'] = bootstrap

        # Catch the case when there is only a single epoch. Just return 0 velocity
        # and the same input position for the x0/y0.
        if self['x'].shape[1] == 1:
            self['x0'] = self['x'][:,0]
            self['y0'] = self['y'][:,0]

            if 't' in self.colnames:
                self['t0'] = self['t'][:, 0]
            else:
                self['t0'] = self.meta['list_times'][0]

            if 'xe' in self.colnames:
                self['x0e'] = self['xe'][:,0]
                self['y0e'] = self['ye'][:,0]

            self['n_vfit'] = 1

            return

        # STARS LOOP through the stars and work on them 1 at a time.
        # This is slow; but robust.
        for ss in range(N_stars):
            self.fit_velocity_for_star(ss, bootstrap=bootstrap)

        if verbose:
            stop_time = time.time()
            print('startable.fit_velocities runtime = {0:.0f} s for {1:d} stars'.format(stop_time - start_time, N_stars))
        
        return

    def fit_velocity_for_star(self, ss, bootstrap=False):
        def poly_model(time, *params):
            pos = np.polynomial.polynomial.polyval(time, params)
            return pos
        
        x = self['x'][ss, :].copy().data
        y = self['y'][ss, :].copy().data

        if 'xe' in self.colnames:
            xe = self['xe'][ss, :].copy().data
            ye = self['ye'][ss, :].copy().data

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

        if 't' in self.colnames:
            t = self['t'][ss, :].copy().data
        else:
            t = self.meta['list_times']

        # Figure out where we have detections (as indicated by error columns)
        good = np.where((xe != 0) & (ye != 0) &
                        np.isfinite(xe) & np.isfinite(ye) &
                        np.isfinite(x) & np.isfinite(y))[0]

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

        # np.polynomial ordering
        p0x = np.array([x.mean(), 0.0])
        p0y = np.array([y.mean(), 0.0])
        
        # Calculate the t0 for all the stars.
        t_weight = 1.0 / np.hypot(xe, ye)
        t0 = np.average(t, weights=t_weight)
        dt = t - t0

        self['t0'][ss] = t0
        self['n_vfit'][ss] = N_good

        # Catch the case where all the times are identical
        if (dt == dt[0]).all():
            wgt_x = (1.0/xe)**2
            wgt_y = (1.0/ye)**2
                
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
            vx_opt, vx_cov = curve_fit(poly_model, dt, x, p0=p0x, sigma=xe)
            vy_opt, vy_cov = curve_fit(poly_model, dt, y, p0=p0y, sigma=ye)

            self['x0'][ss] = vx_opt[0]
            self['vx'][ss] = vx_opt[1]
            self['y0'][ss] = vy_opt[0]
            self['vy'][ss] = vy_opt[1]

            # Run the bootstrap
            if bootstrap > 0:
                edx = np.arange(N_good, dtype=int)

                fit_x0_b = np.zeros(bootstrap, dtype=float)
                fit_vx_b = np.zeros(bootstrap, dtype=float)
                fit_y0_b = np.zeros(bootstrap, dtype=float)
                fit_vy_b = np.zeros(bootstrap, dtype=float)
            
                for bb in range(bootstrap):
                    bdx = np.random.choice(edx, N_good)

                    vx_opt_b, vx_cov_b = curve_fit(poly_model, dt[bdx], x[bdx], p0=vx_opt, sigma=xe[bdx])
                    vy_opt_b, vy_cov_b = curve_fit(poly_model, dt[bdx], y[bdx], p0=vy_opt, sigma=ye[bdx])

                    fit_x0_b[bb] = vx_opt_b[0]
                    fit_vx_b[bb] = vx_opt_b[1]
                    fit_y0_b[bb] = vy_opt_b[0]
                    fit_vy_b[bb] = vy_opt_b[1]

                # Save the errors from the bootstrap
                self['x0e'][ss] = fit_x0_b.std()
                self['vxe'][ss] = fit_vx_b.std()
                self['y0e'][ss] = fit_y0_b.std()
                self['vye'][ss] = fit_vy_b.std()
            else:
                vx_err = np.sqrt(vx_cov.diagonal())
                vy_err = np.sqrt(vy_cov.diagonal())

                self['x0e'][ss] = vx_err[0]
                self['vxe'][ss] = vx_err[1]
                self['y0e'][ss] = vy_err[0]
                self['vye'][ss] = vy_err[1]

        elif N_good == 2:
            # Note nough epochs to fit a velocity.
            self['x0'][ss] = np.average(x, weights=1.0/xe**2)
            self['y0'][ss] = np.average(y, weights=1.0/ye)
            
            dx = np.diff(x)[0]
            dy = np.diff(y)[0]
            dt_diff = np.diff(dt)[0]
            
            self['x0e'][ss] = np.abs(dx) / 2**0.5
            self['y0e'][ss] = np.abs(dy) / 2**0.5
            self['vx'][ss] = dx / dt_diff
            self['vy'][ss] = dy / dt_diff
            self['vxe'][ss] = 0.0
            self['vye'][ss] = 0.0
            
        else:
            # N_good == 1 case
            self['n_vfit'][ss] = 1
            self['x0'][ss] = x[good[0]]
            self['y0'][ss] = y[good[0]]
            
            if 'xe' in self.colnames:
                self['x0e'] = xe[good[0]]
                self['y0e'] = ye[good[0]]

        return

        
        

        
        
        
