from astropy.table import Table, Column


class StarTable(Table):
    """
    A startable is an astropy.Table with stars matched from multiple starlists. 
    I will describe the arguments. 
    """
    
    def __init__(self, ref=0, time_lists=list(), name_lists=list(), **kwargs):
        """
        """
        
        # Check if the required arguments are present
        arg_req = ('x', 'y', 'm')
        
        for arg_test in arg_req:
            if arg_test not in kwargs:
                raise TypeError("The StarTable class requires a '%s' argument"
                                % arg_test)

        # TO DO: I thought we didn't require these? 
        if ('xe' in kwargs) ^ ('ye' in kwargs):
            raise TypeError("The StarTable class requires both 'xe' and" +
                            " 'ye' arguments")
        
        # Check if the type and size of the arguments are correct
        arg_tab = ('x', 'y', 'm', 'xe', 'ye', 'me', 'name')
        n_stars = len(kwargs['x'])
        n_lists = len(kwargs['x'].colnames)
        
        for arg_test in arg_tab:
            
            if arg_test in kwargs:
                # TO DO: Why does this have to be a Table? Don't we want a 2D array or a
                # Column object? A table has lots of other stuff too... so I don't think
                # that would work.
                if not isinstance(kwargs[arg_test], Table):
                    raise TypeError("The '%s' table has to be an astropy table"
                                    % arg_test)
            
                if len(kwargs[arg_test]) != n_stars:
                    raise TypeError("The '%s' table has to have " +
                                    str(n_stars) + " stars" % arg_test)
            
                if len(kwargs[arg_test].colnames) != n_lists:
                    raise TypeError("The '%s' table has to have " +
                                    str(n_lists) + " lists" % arg_test)
        
        if ref not in range(n_lists):
            raise IndexError("The 'ref' argument has to be an integer" +
                             "between 0 and " + str(n_lists - 1))

        if not all(isinstance(tt, (int, float)) for tt in time_lists):
            raise TypeError("The 'time_lists' argument has to be a list of" +
                            " numbers")
        
        if not all(isinstance(nn, str) for nn in name_lists):
            raise TypeError("The 'name_lists' argument has to be a list of" +
                            " strings")
        
        if (len(time_lists) != 0) & (len(time_lists) != n_lists):
            raise ValueError("The 'time_lists' argument has to have " +
                             str(n_lists) + " lists")
        
        if (len(name_lists) != 0) & (len(name_lists) != n_lists):
            raise ValueError("The 'name_lists' argument has to have " +
                             str(n_lists) + " lists")
        
        # Create the startable
        Table.__init__(self, (kwargs['x'], kwargs['y'], kwargs['m']),
                       names=('x', 'y', 'm'))
        self.meta = {'n_stars': n_stars, 'n_lists': n_lists, 'ref': ref,
                     'time_lists': time_lists, 'name_lists': name_lists}
        
        if ('xe' in kwargs) & ('ye' in kwargs):
            self.add_columns([Column(data=kwargs['xe'], name='xe'),
                              Column(data=kwargs['ye'], name='ye')])
        
        if 'me' in kwargs:
            self.add_column(Column(data=kwargs['me'], name='me'))
        
        if 'name' in kwargs:
            self.add_column(Column(data=kwargs['name'], name='name'))

        # Assign a random new name.
        uname = ["star_{:06d}".format(item) for item in range(1, (n_stars + 1))]
        self.add_column(Column(data=uname, name='uname'))
