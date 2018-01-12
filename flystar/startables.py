from astropy.table import Table, Column
from astropy.stats import sigma_clipping
import numpy as np
import warnings
import collections
import pdb


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
            self.meta = {'n_stars': n_stars, 'n_lists': n_lists, 'ref_list': ref_list}

            for meta_arg in meta_tab:
                if meta_arg in kwargs:
                    self.meta[meta_arg] = kwargs[meta_arg]

            for arg in kwargs:
                if arg in ['name', 'x', 'y', 'm']:
                    continue
                else:
                    self.add_column(Column(data=kwargs[arg], name=arg))

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
                    new_data[:, -1] = starlist[col_name]
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
                
                if (col_name in kwargs):            # Add data if it was input
                    self[col_name][:, -1] = kwargs[col_name]
                else:                               # Add junk data it if wasn't input
                    self._set_invalid_list_values(col_name, -1)
                    

        # Update the table meta-data. Remember that entries are lists not numpy arrays.
        for key in self.meta.keys():
            # Meta table entries with a size that matches the n_lists size are the ones
            # that need a new value. We have to add something... whatever was passed in or None
            if (isinstance(self.meta[key], collections.Iterable) and
                    (len(self.meta[key]) == self.meta['n_lists'])):

                # If we find the key in the passed in meta argument, then add the new values.
                # Otherwise, add "None".
                new_meta_keys = kwargs['meta'].keys()
                if key in new_meta_keys:
                    self.meta[key] = np.append(self.meta[key], [kwargs['meta'][key]])
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
        if np.issubclass_(self[col_name].info.dtype, np.integer):
            self[col_name][:, col_idx] = -1
        elif np.issubclass_(self[col_name].info.dtype, np.floating):
            self[col_name][:, col_idx] = np.nan
        else:
            self[col_name][:, col_idx] = None
        
        return

    def _set_invalid_star_values(self, col_name, row_idx):
        """
        Set the contents of the specified rows (in the 2D column objects)
        to an invalide value depending on the data type.
        """
        if np.issubclass_(self[col_name].info.dtype, np.integer):
            self[col_name][row_idx] = -1
        elif np.issubclass_(self[col_name].info.dtype, np.floating):
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


    def combine_lists_xym(self, weighted_xy=True, weighted_m=True, sigma=3):
        """
        For x, y and m columns in the table, collapse along the lists
        direction. For 'x', 'y' this means calculating the median position with
        outlier rejection. Optionally, weight by the 'xe' and 'ye' individual
        uncertainties. Optionally, use sigma clipping.
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
            
        self.combine_lists('x', weights_col=weights_colx, sigma=sigma)
        self.combine_lists('y', weights_col=weights_coly, sigma=sigma)
        self.combine_lists('m', weights_col=weights_colm, sigma=sigma, ismag=True)
            
        return

    def combine_lists(self, col_name_in, weights_col=None, mask_val=None,
                      meta_add=True, ismag=False, sigma=3):
        """
        For the specified column (col_name_in), collapse along the starlists
        direction and calculated the median value, with outlier rejection.
        Optionally, weight by a specified column (weights_col). Optionally,
        use sigma clipping. The final values are stored in a new column named
        <col_name_in>_avg -- the mean (with outlier rejection)
        <col_name_in>_std -- the std (with outlier rejection)

        Masking of NaN values is also performed.
        
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

        # Dedicde if we are going to have weights (before we
        # do the expensive sigma clipping routine). Note that
        # if we have only 1 column to average, then we can't do weighting. 
        if (weights_col and weights_col in self.colnames) and (val_2d.shape[1] > 1):
            np.seterr(divide='ignore')
            wgt_2d = np.ma.masked_invalid(1.0 / self[weights_col]**2)

            if ismag:
                wgt_2d = wgt_2d * 1.0  # FIX ME 
                
            np.seterr(divide='warn')
            if meta_add:
                self.meta[col_name_in + '_avg'] = 'weighted'
        else:
            wgt_2d = None
            if meta_add:
                self.meta[col_name_in + '_avg'] = 'not_weighted'

        # Figure out which ones are outliers. Returns a masked array.
        if sigma:
            val_2d_clip = sigma_clipping.sigma_clip(val_2d, sigma=3, iters=5, axis=1)
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

        # Save off our new AVG and STD into new columns with shape (N_stars).
        col_name_avg = col_name_in + '_avg'
        col_name_std = col_name_in + '_std'

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

