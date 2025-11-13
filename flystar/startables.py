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
    
    def fit_velocities_new(
            self, 
            motion_models=['Empty', 'Fixed', 'Linear'],
            weighting='var', 
            use_scipy=False, 
            absolute_sigma=True, 
            bootstrap=0,
            fixed_t0=False, 
            verbose=True, 
            mask_value=None, 
            fill_value=np.nan,
            show_progress=True
    ):
        """Fit velocity for star table

        Parameters
        ----------
        motion_models : list, optional
            Motion models name to use. 
            If multiple models are supplied, prioritize the model with the most parameters to fit. 
            If multiple models have the same number of parameters, raise AssertionError: not sure which to use.
            When not enough data points, use the model with just enough parameters to fit, by default ['Empty, 'Fixed', 'Linear']
        weighting : str, optional
            Uncertainty weighting, 'std' for weight=1/xe(ye) or 'var' for weight=1/xe(ye)**2, by default 'var'
        use_scipy : bool, optional
            Use scipy.optimize.curve_fit or algebraic solution (for Linear model only), by default False
        absolute_sigma : bool, optional
            Use absolute sigma or not, see scipy curve_fit for details, by default True
        bootstrap : int, optional
            Number of bootstrap for uncertainty resampling, by default 0
        fixed_t0 : bool or float, optional
            If provided, use the fixed t0. Otherwise, use average t weighted by 1/np.hypot(xe, ye), by default False
        verbose : bool, optional
            Print verbose messages or not, by default True
        mask_value : float, optional
            Values to mask in data, by default None
        fill_value : float, optional
            Fill value when there is not enough data points to fit, by default np.nan
        show_progress : bool, optional
            Show progress bar or not, by default True

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
            raise ValueError(f"fit_velocities: Weighting must either be 'var' or 'std', not {weighting}!")

        if ('t' not in self.colnames) and ('LIST_TIMES' not in self.meta):
            raise KeyError("fit_velocities: Failed to access time values. No 't' column in table, no 'LIST_TIMES' in meta.")

        # Check if we have the required columns
        if not all([_ in self.colnames for _ in ['x', 'y']]):
            raise KeyError(f"fit_velocities: Missing required columns in the table: {', '.join(['x', 'y'])}!")

        # Convert motion models from strings to classes
        motion_model_map = motion_model.motion_model_map()
        if 'Empty' not in motion_models:
            motion_models.insert(0, 'Empty')  # Ensure Empty model is always included
        motion_models = [motion_model_map[mm] for mm in motion_models]

        ###########################
        ####### Prepare Data ######
        ###########################
        # Prepare data for fitting
        N_stars = len(self)
        x_data = np.ma.masked_invalid(self['x'].data, copy=True)
        y_data = np.ma.masked_invalid(self['y'].data, copy=True)
        xe_data = np.ma.masked_invalid(self['xe'].data, copy=True) if 'xe' in self.colnames else None
        ye_data = np.ma.masked_invalid(self['ye'].data, copy=True) if 'ye' in self.colnames else None
        # t_data: 2d array with shape (N_stars, N_epochs)
        # t0: 1d array with shape (N_stars,)
        if 't' in self.colnames:
            t_data = copy.deepcopy(self['t'].data)
            t0 = np.average(t_data, axis=1, weights=1/np.hypot(xe_data, ye_data)) if not fixed_t0 else np.ones(N_stars)*fixed_t0
        else:
            t_data = copy.deepcopy(np.array(self.meta['LIST_TIMES']))
            t_data = np.broadcast_to(t_data, x_data.shape)
            t0 = np.average(t_data, axis=1, weights=1/np.hypot(xe_data, ye_data)) if not fixed_t0 else np.ones(N_stars)*fixed_t0
        if mask_value:
            x_data = np.ma.masked_values(x_data, mask_value)
            y_data = np.ma.masked_values(y_data, mask_value)
            if xe_data is not None:
                xe_data = np.ma.masked_values(xe_data, mask_value)
            if ye_data is not None:
                ye_data = np.ma.masked_values(ye_data, mask_value)

        # Calculate mask array
        xy_mask = (~x_data.mask) & (~y_data.mask)
        self['n_obs'] = xy_mask.sum(axis=1)

        # Convert to lists of arrays for faster access during fitting
        t_stars = [np.array(t_data[i][xy_mask[i]]) for i in range(N_stars)]
        x_stars = [np.array(x_data[i][xy_mask[i]]) for i in range(N_stars)]
        y_stars = [np.array(y_data[i][xy_mask[i]]) for i in range(N_stars)]
        xe_stars = [np.array(xe_data[i][xy_mask[i]]) if xe_data is not None else None for i in range(N_stars)]
        ye_stars = [np.array(ye_data[i][xy_mask[i]]) if ye_data is not None else None for i in range(N_stars)]


        ###########################
        ####### Determine MM ######
        ###########################
        mm_n_params = np.sort([mm.n_params for mm in motion_models])
        # Assert that motion model n_params are unique and sorted
        assert len(mm_n_params) == len(set(mm_n_params)), "fit_velocities: Provided motion model n_params are not unique! Cannot decide which motion model to use based on n_obs."

        # Select motion model based on n_obs
        mm_digitized = np.digitize(
            x=self['n_obs'],
            bins=mm_n_params
        ) - 1  # -1 to convert to 0-based index
        self['motion_model'] = np.array([motion_models[d].__name__ for d in mm_digitized])

        # Fill table with all possible motion model parameter names as new columns.
        new_col_list = motion_model.get_list_motion_model_param_names(motion_models, with_errors=True)
        new_col_list += ['chi2_x', 'chi2_y', 'n_params']
        if 't0' not in new_col_list:
            new_col_list.append('t0')

        # Replace old columns if they exist
        for col in new_col_list:
            if col.endswith('_err'):
                self.add_column(
                    Column(data=np.full(N_stars, np.inf, dtype=float), name=col),
                    rename_duplicate=True
                )
            else:
                self.add_column(
                    Column(data=np.full(N_stars, np.nan, dtype=float), name=col),
                    rename_duplicate=True
                )

        # Add a column to keep track of the number of points used in a fit and number of bootstrap used.
        self['n_bootstrap'] = bootstrap

        ###########################
        ######### FITTING #########
        ###########################
        unique_motion_models, unique_inv_indices = np.unique(self['motion_model'], return_inverse=True)
        indices_by_motion_model = {key: np.flatnonzero(unique_inv_indices == k) for k, key in enumerate(unique_motion_models)}
    
        for unique_motion_model, unique_index in indices_by_motion_model.items():
            # Create motion model instance
            motion_model_instance = motion_model_map[unique_motion_model]()
            # Initialize arrays to store results
            n_stars_this_model = len(unique_index)
            n_params = len(motion_model_instance.fitter_param_names)

            params_array = np.full((n_stars_this_model, n_params), fill_value, dtype=float)
            param_errs_array = np.full((n_stars_this_model, n_params), np.inf, dtype=float)
            chi2_x_array = np.full(n_stars_this_model, np.nan, dtype=float)
            chi2_y_array = np.full(n_stars_this_model, np.nan, dtype=float)

            for idx, i_star in enumerate(tqdm(unique_index, disable=not show_progress, desc=f"Fitting motion model {unique_motion_model}")):
                # Fit the star
                params, param_errs, chi2_x, chi2_y = motion_model_instance.fit_motion_model(
                    t=t_stars[i_star],
                    x=x_stars[i_star],
                    y=y_stars[i_star],
                    xe=xe_stars[i_star],
                    ye=ye_stars[i_star],
                    t0=t0[i_star],
                    weighting=weighting,
                    use_scipy=use_scipy,
                    absolute_sigma=absolute_sigma,
                    bootstrap=bootstrap,
                    fill_value=fill_value,
                    verbose=verbose
                )
                # Store results to arrays
                params_array[idx] = params
                param_errs_array[idx] = param_errs
                chi2_x_array[idx] = chi2_x
                chi2_y_array[idx] = chi2_y

            # Store results back to the table
            param_names = motion_model_instance.fitter_param_names
            for j, param_name in enumerate(param_names):
                self[param_name][unique_index] = params_array[:, j]
                self[param_name + '_err'][unique_index] = param_errs_array[:, j]
            self['chi2_x'][unique_index] = chi2_x_array
            self['chi2_y'][unique_index] = chi2_y_array
            self['n_params'][unique_index] = n_params
            self['t0'][unique_index] = t0[unique_index]
        return

    def fit_velocities(self, weighting='var', use_scipy=True, absolute_sigma=True, bootstrap=0,
                       fixed_t0=False, verbose=False, mask_val=None, mask_lists=False, show_progress=True,
                       default_motion_model='Linear', reassign_motion_model=False, select_stars=None, motion_model_dict={}):
        """Fit velocities for all stars in the table and add to the columns 'vx', 'vxe', 'vy', 'vye', 'x0', 'x0e', 'y0', 'y0e'.

        Parameters
        ----------
        weighting : str, optional
            Weight by variance 'var' or standard deviation 'std', by default 'var'
        bootstrap : int, optional
            Calculate uncertainty using bootstraping or not, by default 0
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

        # Set all to default_motion_model if none assigned already.
        # Reset motion_model_used to the inputs for now -> will change as fits run
        if ('motion_model_input' not in self.colnames) or reassign_motion_model:
            self['motion_model_input'] = default_motion_model
        self['motion_model_used'] = self['motion_model_input']
        
        motion_model_dict = motion_model.validate_motion_model_dict(motion_model_dict, self, default_motion_model)
            
        #
        # Fill table with all possible motion model parameter names as new
        # columns. Make everything empty for now.
        #
        all_motion_models = np.unique(self['motion_model_input'].tolist() + ['Fixed']+[default_motion_model]).tolist()
        new_col_list = motion_model.get_list_motion_model_param_names(all_motion_models, with_errors=True)
        # Append goodness of fit metrics and t0.
        new_col_list += ['chi2_x', 'chi2_y', 'n_params']
        if 't0' not in new_col_list:
            new_col_list.append('t0')

        # Define output arrays for the best-fit parameters.
        for col in new_col_list:
            # Clean/remove up old arrays.
            if col in self.colnames: self.remove_column(col)
            # Add column #TODO: is this good for filling???
            self.add_column(Column(data = np.full(N_stars, np.nan, dtype=float), name = col))

        # Add a column to keep track of the number of points used in a fit.
        self['n_fit'] = 0
        
        # Preserve the number of bootstraps that will be run (if any).
        self.meta['n_fit_bootstrap'] = bootstrap
        
        # (FIXME: Do we need to catch the case where there's a single *unmasked* epoch?)
        # Catch the case when there is only a single epoch. Just return 0 velocity
        # and the same input position for the x0/y0.
        if len(self['x'].shape) == 1:
            self['motion_model_used'] = 'Fixed'
            self['x0'] = self['x']
            self['y0'] = self['y']
            if 't' in self.colnames:
                self['t0'] = self['t']
            else:
                self['t0'] = self.meta['LIST_TIMES'][0]
            if 'xe' in self.colnames:
                self['x0_err'] = self['xe']
                self['y0_err'] = self['ye']
            self['n_fit'] = 1
            self['n_params'] = 1
            return
        
        if (self['x'].shape[1] == 1):
            self['motion_model_used'] = 'Fixed'
            self['x0'] = self['x'][:,0]
            self['y0'] = self['y'][:,0]
            if 't' in self.colnames:
                self['t0'] = self['t'][:, 0]
            else:
                self['t0'] = self.meta['LIST_TIMES'][0]
            if 'xe' in self.colnames:
                self['x0_err'] = self['xe'][:,0]
                self['y0_err'] = self['ye'][:,0]
            self['n_fit'] = 1
            self['n_params'] = 1
            return
            
        # Only fit selected stars, if list given
        fit_star_idxs = range(N_stars)
        if select_stars is not None:
            fit_star_idxs = select_stars
        # STARS LOOP through the stars and work on them 1 at a time.
        # This is slow; but robust.
        if show_progress:
            for ss in tqdm(fit_star_idxs):
                self.fit_velocity_for_star(ss, motion_model_dict, weighting=weighting, bootstrap=bootstrap,
                                           use_scipy=use_scipy, absolute_sigma=absolute_sigma,
                                           fixed_t0=fixed_t0, default_motion_model=default_motion_model,
                                           mask_val=mask_val, mask_lists=mask_lists)
        else:
            for ss in fit_star_idxs:
                self.fit_velocity_for_star(ss, motion_model_dict, weighting=weighting, bootstrap=bootstrap,
                                           use_scipy=use_scipy, absolute_sigma=absolute_sigma,
                                           fixed_t0=fixed_t0, default_motion_model=default_motion_model,
                                           mask_val=mask_val, mask_lists=mask_lists)
        if verbose:
            stop_time = time.time()
            print('startable.fit_velocities runtime = {0:.0f} s for {1:d} stars'.format(stop_time - start_time, N_stars))
        
        return

    def fit_velocity_for_star(self, ss, motion_model_dict, weighting='var', use_scipy=True, absolute_sigma=True,
                              bootstrap=False, fixed_t0=False, mask_val=None, mask_lists=False,
                              default_motion_model='Linear'):
        # TODO: "weighting" is not used
        # 
        # Make a mask of invalid (NaN) values and a user-specified invalid value.
        #
        
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
        #
        # Assign the appropriate positional errors
        #
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

        #
        # Make a mask of invalid (NaN) values and a user-specified invalid value.
        #
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
        
        #
        # Figure out where we have detections (as indicated by error columns)
        #
        good = np.where((xe != 0) & (ye != 0) &
                        np.isfinite(xe) & np.isfinite(ye) &
                        np.isfinite(x) & np.isfinite(y) & ~new_mask)[0]

        N_good = len(good)

        # Catch the case where there is NO good data. 
        if N_good == 0:
            #self['motion_model_used'][ss] = 'None'
            self['n_fit'][ss] = N_good
            self['n_params'][ss] = 0
            return

        # Everything below has N_good >= 1
        x = x[good]
        y = y[good]
        t = t[good]
        xe = xe[good]
        ye = ye[good]

        #
        # Unless t0 is fixed, calculate the t0 for the stars.
        #
        if fixed_t0 is False:
            t_weight = 1.0 / np.hypot(xe, ye)
            t0 = np.average(t, weights=t_weight)
        elif fixed_t0 is True:
            t0 = self.t0
        else:
            t0 = fixed_t0[ss]
        self['t0'][ss] = t0
        self['n_fit'][ss] = N_good

        #
        # Decide which motion_model to fit.
        #
        motion_model_use = self['motion_model_input'][ss]
        # Go to default model if not enough points for assigned but enough for default
        # TODO: think about whether we want other fallbacks besides the singular default and Fixed
        if (N_good < motion_model_dict[motion_model_use].n_pts_req) and \
            (N_good >= motion_model_dict[default_motion_model].n_pts_req):
            motion_model_use = default_motion_model
        # If not enough points for either, go to a fixed model
        elif (N_good < motion_model_dict[motion_model_use].n_pts_req) and \
            (N_good < motion_model_dict[default_motion_model].n_pts_req):
            motion_model_use = 'Fixed'
        # If the points do not cover multiple times, go to a fixed model
        if (t == t[0]).all():
            motion_model_use = 'Fixed'

        self['motion_model_used'][ss] = motion_model_use

#        # Get the motion model object.
#        modClass = motion_model_dict[motion_model_use]
#
#        # Load up any prior information on parameters for this model.
#        param_dict = {}
#        for par in modClass.fitter_param_names+modClass.fixed_param_names:
#            if ~np.isnan(self[par][ss]):
#                param_dict[par] = self[par][ss]

        # Model object
        mod = motion_model_dict[motion_model_use]
        fixed_params = [self[par][ss] for par in mod.fixed_param_names]

        # Fit for the best parameters
        params, param_errs, chi2_x, chi2_y = mod.fit_motion_model(t, x, y, xe, ye, t0, bootstrap=bootstrap,
                                        weighting=weighting, use_scipy=use_scipy, absolute_sigma=absolute_sigma)
        # chi2_x,chi2_y = mod.get_chi2(params,fixed_params, t,x,y,xe,ye)
        self['chi2_x'][ss]=chi2_x
        self['chi2_y'][ss]=chi2_y
        self['n_params'][ss] = mod.n_params

        # Save parameters and errors to table.
        for pp in range(len(mod.fitter_param_names)):
            par = mod.fitter_param_names[pp]
            par_err = par + '_err'
            self[par][ss] = params[pp]
            self[par_err][ss] = param_errs[pp]
            
        return
        
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
                for par in mod.fitter_param_names + mod.fixed_param_names + [pm+'_err' for pm in mod.fitter_param_names]:
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

        return x,y,xe,ye
                

    def fit_velocities_all_detected(self, motion_model_to_fit, weighting='var', use_scipy=True, absolute_sigma=True, times=None,
                    select_stars=None, epoch_cols='all', mask_val=None, art_star=False, return_result=False):
        """Fit velocities for stars detected in all epochs specified by epoch_cols. 
        Criterion: xe/ye error > 0 and finite, x/y not masked.

        Parameters
        ----------
        motion_model_to_fit : MotionModel
            Motion model object to use for fitting all stars
        weighting : str, optional
            Variance weighting('var') or standard deviation weighting ('std'), by default 'var'
        select_idx : array-like, optional
            Indices of stars to select for fitting, by default None (fit all detected stars)
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
        if select_stars is None:
            select_stars = np.arange(N_stars)
        else:
            select_stars = np.asarray(select_stars)

        if epoch_cols == 'all':
            epoch_cols = np.arange(np.shape(self['x'])[1])
        
        # Artificial Star
        if art_star:
            detected_in_all_epochs = np.all(self['det'][select_stars, :][:, epoch_cols], axis=1)

        # Observation Star
        else:
            valid_xe = np.all(self['xe'][select_stars, :][:, epoch_cols]!=0, axis=1) & np.all(np.isfinite(self['xe'][select_stars, :][:, epoch_cols]), axis=1)
            valid_ye = np.all(self['ye'][select_stars, :][:, epoch_cols]!=0, axis=1) & np.all(np.isfinite(self['ye'][select_stars, :][:, epoch_cols]), axis=1)

            if mask_val:
                x = np.ma.masked_values(self['x'][select_stars, :][:, epoch_cols], mask_val, shrink=False)
                y = np.ma.masked_values(self['y'][select_stars, :][:, epoch_cols], mask_val, shrink=False)
                valid_x = ~np.any(x.mask, axis=1)
                valid_y = ~np.any(y.mask, axis=1)
                detected_in_all_epochs = np.logical_and.reduce((
                    valid_x, valid_y, valid_xe, valid_ye))
            else:
                detected_in_all_epochs = np.logical_and(valid_xe, valid_ye)

        N = len(self['x'][select_stars, :])
        fit_params = motion_model_to_fit.fitter_param_names
        param_data = {p: np.zeros(N) for p in fit_params}
        param_data.update({p+'_err': np.zeros(N) for p in fit_params})
        param_data.update({p: np.zeros(N) for p in motion_model_to_fit.fixed_param_names})
        param_data['chi2_x'] = np.zeros(N)
        param_data['chi2_y'] = np.zeros(N)

        if times is None:
            if 'YEARS' in self.meta:
                times = np.array(self.meta['YEARS'])[epoch_cols]
            elif 't' in self.colnames:
                times = self['t'][0, epoch_cols]
            else:
                raise ValueError("No valid time column found.")
        
        if not art_star:
            x_arr = self['x'][select_stars, :][:, epoch_cols]
            y_arr = self['y'][select_stars, :][:, epoch_cols]
        else:
            x_arr = self['x'][select_stars, :][:, epoch_cols, 1]
            y_arr = self['y'][select_stars, :][:, epoch_cols, 1]

        xe_arr = self['xe'][select_stars, :][:, epoch_cols]
        ye_arr = self['ye'][select_stars, :][:, epoch_cols]
        
        # Only fit for >1 epochs, otherwise all velocities will be 0
        if len(epoch_cols) > 1:
            # For each star
            for i in tqdm(range(N)):
                x = x_arr[i]
                y = y_arr[i]
                xe = xe_arr[i]
                ye = ye_arr[i]
                t0 = np.average(times, weights=1. / np.hypot(xe, ye))

                # Run fit and record results
                params, param_errs = motion_model_to_fit.fit_motion_model(
                    times, x, y, xe, ye, t0, weighting=weighting,
                    use_scipy=use_scipy, absolute_sigma=absolute_sigma
                )
                if 't0' in motion_model_to_fit.fixed_param_names:
                    param_data['t0'][i] = t0
                for j, param in enumerate(fit_params):
                    param_data[param][i] = params[j]
                    param_data[f'{param}_err'][i] = param_errs[j]
                chi2x, chi2y = motion_model_to_fit.get_chi2(params, [t0], times, x, y, xe, ye)
                param_data['chi2_x'][i] = chi2x
                param_data['chi2_y'][i] = chi2y

        vel_result = Table.from_pandas(pd.DataFrame(param_data))

        # Add n_vfit
        n_fit = len(epoch_cols)
        vel_result['n_fit'] = n_fit

        # Clean/remove up old arrays.
        columns = [*vel_result.keys(), 'n_fit']
        for column in columns:
            if column in self.colnames: self.remove_column(column)

        # Update self
        for column in columns:
            column_array = MaskedColumn(np.ma.zeros(N_stars), dtype=float, name=column)
            column_array[select_stars] = vel_result[column]
            column_array[select_stars][~detected_in_all_epochs] = np.nan
            column_array.mask[select_stars] = ~detected_in_all_epochs
            # Mask unselected indices
            column_array.mask[~np.isin(np.arange(N_stars), select_stars)] = True
            self[column] = column_array

        if return_result:
            return vel_result
        else:
            return

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
