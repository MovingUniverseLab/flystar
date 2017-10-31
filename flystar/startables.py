from astropy.table import Table, Column
from astropy.stats import sigma_clipping
import numpy as np
import warnings
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
    def __init__(self, ref_list=0, **kwargs):
        """
        """
        
        # Check if the required arguments are present
        arg_req = ('name', 'x', 'y', 'm')
        
        found_all_required = True
        for arg_test in arg_req:
            if arg_test not in kwargs:
                found_all_required = False

        if not found_all_required:
            err_msg = "The StarList class requires a '{0:s}' argument"
            warnings.warn(err_msg, UserWarning)
            Table.__init__(self, **kwargs)
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
            arg_tab = ('x', 'y', 'm', 'xe', 'ye', 'me', 'ep_name', 'corr')

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

            for arg in arg_tab:
                if arg in ['name', 'x', 'y', 'm']:
                    continue
                if arg in kwargs:
                    self.add_column(Column(data=kwargs[arg], name=arg))

        return
    
    def add_list(self, **kwargs):
        """
        Add data from a new list to an existing StarTable. 
        Note, you can pass in the data via a StarList object or
        via a series of keywords with a 1D array on each. 
        In either case, the number of stars must already match
        the existing number of stars in the StarTable.

        Example 1: Pass in data via StarList object.
        ----------
        print(t['x'].shape)
        t.add_list(starlist=my_list)
        print(t['x'].shape)   # Should be 1 column larger than before.

        Example 2: Pass in data via keywords and 1D arrays.
        t.add_list(x=x_new, y=y_new, m=m_new)

        """
        # Check if we are dealing with a StarList object or a
        # set of arguments with individual arrays.
        if 'starlist' in kwargs:
            self._add_list_data_from_starlist(list)
        else:
            self._add_list_data_from_keywords(**kwargs)

        return

    def _add_list_data_from_keywords(self, **kwargs):
        # Check if the required arguments are present
        arg_req = ('x', 'y', 'm')
        
        for arg_test in arg_req:
            if arg_test not in kwargs:
                err_msg = "Added lists require a '{0:s}' argument"
                raise TypeError(err_msg.format(arg_test))
            
        # If we have errors, we need them in both dimensions.
        if ('xe' in kwargs) ^ ('ye' in kwargs):
            raise TypeError("Added lists with errors require both 'xe' and" +
                            " 'ye' arguments")

        # Loop through the 2D columns and add the new data to each.
        # If there is no input data for a particular column, then fill it with
        # zeros and mask it.
        for col_name in self.colnames:
            
            if len(self[col_name].shape) == 2:      # Find the 2D columns
                if (col_name in kwargs):            # Add data if it was input
                    self[col_name].append(kwargs[col_name], axis=1)
                else:                               # Add junk data it if wasn't input
                    new_type = self[col_name].data.dtype

                    if issubclass(self[col_name].info.dtype, np.integer):
                        new_data = np.empty(len(self), dtype=new_type)
                        new_data.fill()
                    elif issubclass(self[col_name].info.dtype, np.floating):
                        new_data = np.empty(len(self), dtype=new_type)
                        new_data.fill(np.nan)
                        
                    self[col_name].append(new_data, axis=1)
                
        return
    
    
    def combine_lists_xym(self, weighted=True):
        """
        For all columns in the table that have dimensions (N_stars, N_lists),
        collapse along the lists direction. For 'x', 'y' this means calculating
        the median position with outlier rejection weighted by the 'xe' and 'ye'
        individual uncertainties, if they exist. For 'm', convert to flux first
        and do the same.
        """
        return

    def combine_lists(self, col_name_in, weights_col=None, mask_val=None):
        """
        For the specified column (col_name_in), collapse along the starlists
        direction and calculated the median value, with outlier rejection.
        Optionally, weight by a specified column (weights_col). The final
        values are stored in a new column named
        <col_name_in>_avg -- the mean (with outlier rejection)
        <col_name_in>_std -- the std (with outlier rejection)

        Masking of NaN values is also performed.
        """
        # Get the array we are going to combine. Also make a mask
        # of invalid (NaN) values and a user-specified invalid value.
        val_2d = self[col_name_in].data
        val_2d = np.ma.masked_values(val_2d, mask_val)

        # Dedicde if we are going to have weights (before we
        # do the expensive sigma clipping routine.
        if weights_col:
            wgt_2d = np.ma.masked_invalid(1.0 / self[weights_col]**2)

        # Figure out which ones are outliers. Returns a masked array.
        val_2d_clip = sigma_clipping.sigma_clip(val_2d, sigma=3, iters=5, axis=1)

        # Calculate the (weighted) mean and standard deviation along
        # the N_lists direction (axis=1).
        if weights_col:
            avg = np.ma.average(val_2d_clip, weights=wgt_2d, axis=1)
            std = np.sqrt(np.ma.average((val_2d_clip.T - avg).T**2, weights=wgt_2d, axis=1))
        else:
            avg = np.ma.mean(val_2d_clip, axis=1)
            std = np.ma.std(val_2d_clip, axis=1)

        # Save off our new AVG and STD into new columns with shape (N_stars).
        col_name_avg = col_name_in + '_avg'
        col_name_std = col_name_in + '_std'

        if col_name_avg in self.colnames:
            self[col_name_avg] = avg.data
            self[col_name_std] = std.data
        else:
            self.add_column(Column(data=avg.data, name=col_name_avg))
            self.add_column(Column(data=std.data, name=col_name_std))
        
        return
