from astropy.table import Table, Column
import numpy as np
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

    Optional table meta data
    -------------------------
    list_names : list of strings
        List of names, one for each of the starlists.

    list_times : list of floats
        List of times/dates for each starlist.

    ref_list : int
        Specify which list is the reference list (if any). 
    
    """
    def __init__(self, ref_list=0, **kwargs):
        """
        """
        
        # Check if the required arguments are present
        arg_req = ('name', 'x', 'y', 'm')
        
        for arg_test in arg_req:
            if arg_test not in kwargs:
                err_msg = "The StarTable class requires a '{0:s}' argument"
                raise TypeError(err_msg.format(arg_test))

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
        arg_tab = ('x', 'y', 'm', 'xe', 'ye', 'me', 'ep_name')
        
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
        meta_type = (float, str)
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
