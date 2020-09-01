import numpy as np
from astropy.table import Table, Column
import astropy.table
import warnings
import pdb

try:
    set
except NameError:
    from sets import Set as set

def restrict_by_name(table1, table2):
    """
    find stars with same name in two table.

    Input
    ------
    table1: astropy.table
        contains name,m,x,y,xe,ye,vx,vy,vxe,vye,t0

    table2: astropy.table
        ontains name,m,x,y,xe,ye


    Output
    ------
    -Array with indicies of named stars in table1
    -Array with indicies of named stars from table2
    -Number of named stars
    """

    name1 = table1['name']
    name2 = table2['name']
    
    Name = np.intersect1d(name1, name2)
    # trim out stars begin with 'star'
    idx = []
    for i in range(len(Name)):
        name = Name[i]
        if name[0:4] != 'star':
            idx.append(i)
    Name = Name[idx]

    idx1 = np.array([list(name1).index(i) for i in Name])
    idx2 = np.array([list(name2).index(i) for i in Name])

    return idx1, idx2, len(idx1)

def restrict_by_area(table1, area, exclude=False):
    """
    Restrict starlist to those within a specific area. Returns the
    indicies of the stars in table1 that fulfill this criteria. Area and table1
    positions must be in consistent units in order for this to work.

    Parameters:
    ----------
    table1: astropy table
         Starlist to be restricted. Must have standard column headers
         (i.e. x, y)

    area: 2x2 array [[x1, x2], [y1, y2]]
        X and Y coordinate range to restric the stars too. Only stars with
        coordinates with x1 < X < x2 and y1 < Y < y2 will be allowed. We
        assume the area is given in same units as the table1 positions.
        i.e., x1 = xmin, x2 = xmax; y1 = ymin, y2 = ymax

    exclude: boolean (default=False)
        If true, *exclude* the stars that fall within the given area. If false,
        then only return stars that fall within the given area
        
    Output:
    ------
    array of indicies corresponding to stars which are within the designated
    area.
    """
    # Extract star coordinates
    xpos = table1['x']
    ypos = table1['y']
    
    # Extract desired coordinate ranges
    x_range = area[0]
    y_range = area[1]

    # Apply restriction
    if not exclude:
        good = np.where( (xpos > x_range[0]) & (xpos < x_range[1]) &
                      (ypos > y_range[0]) & (ypos < y_range[1]) )

    else:
        good = np.where( ( (xpos < x_range[0]) | (xpos > x_range[1]) ) &
                         ( (ypos < y_range[0]) | (ypos > y_range[1]) ) )
        
    return good[0]

def restrict_by_use(label_mat, starlist_mat, idx_label, idx_starlist):
    """
    Restrict matching starlist to only those where the label_mat['use'] > 2.
    This behaves the same way as the -restrict flag in java align.

    Return the indicies corresponding to the stars that fulfill this condition.

    Parameters:
    -----------
    label_mat: astropy table
         Label.dat table containing the matched stars. Must have standard
         column headers. Rows are assumed to match the starlist.

    starlist_mat: astropy table
         Reference table containing the matched stars. Must have standard column
         headers. Rows are assumed to match the label.dat file

    idx_label: array of indicies
        Indicies of the matched stars in the label catalog

    idx_starlist: array of indicies
        Indicies of the matched stars in the starlist.
        
    Output:
    -------
    idx_label_f: array of indicies in the label catalog that fulfill the restrict
    condition

    idx_starlist_f: array of indicies in the starlist that fulfill the restrict
    condition
    
    
    label_trim: astropy table
         label table with only use > 2 stars

    starlist_trim: astropy table
         reference table with only stars that correspond to use > 2 stars
         in the label_mat table.
    
    """
    print( 'Restrict option activated')

    # Among label.dat matched stars, determine which are allowed by -restrict
    idx_restrict = np.where(label_mat['use'] !='0')

    # Update the indicies of the matched stars to only include those which
    # pass the restrict condition
    idx_label_f = idx_label[idx_restrict]
    idx_starlist_f = idx_starlist[idx_restrict]

    # Restrict tables to only these stars.
    #label_orig_trim = label_mat_orig[idx_restrict]
    #label_trim = label_mat[idx_restrict]
    #starlist_trim = starlist_mat[idx_restrict]

    #return label_orig_trim, label_trim, starlist_trim
    print( 'Restrict option activated')
    print(( 'Keeping {0} of {1} stars'.format(len(idx_restrict),
                                            len(label_mat))))
    
    return idx_label_f, idx_starlist_f


def read_label(labelFile, prop_to_time=None, flipX=True):
    """
    Read in a label.dat file, rename columns with standard
    names. Use velocities to convert positions into epoch
    t0, and then flip x positions and velocities such that
    +x is to the west.

    Update values in columns of position and velocity

    Parameters:
    ----------
    labelFile: text file. containing
        col1: name
        col2: mag
        col3: x0 (arcsec)
        col4: y0 (arcsec)
        col5: x0err
        col6: y0err
        col7: vx (mas/yr)
        col8: vy (mas/yr)
        col9: vxerr
        col10: vyerr
        col11: t0
        col12: use
        col13: r0 (arcsec)

    prop_to_time: None or float (default = None)
        If float, use velocities to propogate positions to defined time.

    flipX: boolean (default = True)
         If true, multiply the x positions and velocities by -1.0. This is
         useful when label.dat has +x to the east, while reference starlist
         has +x to the west.
         
    #OLD# tref: reference epoch that label.dat is converted to.

    Output:
    ------
    labelFile: astropy.table.
    containing name, m, x0, y0, x0e, y0e, vx, vy, vxe, vye, t0, use, r0,
    (if prop_to_time: x, y, xe, ye, t)
    
    x and y is in arcsec,
    converted to tref epoch,
    *(-1) so it increases to west
    
    vx, vy, vxe, vye is converted to arcsec/yr

    """
    t_label = Table.read(labelFile, format='ascii.no_header')
    t_label.rename_column('col1', 'name')
    t_label.rename_column('col2', 'm')
    t_label.rename_column('col3', 'x0')
    t_label.rename_column('col4', 'y0')
    t_label.rename_column('col5', 'x0e')
    t_label.rename_column('col6', 'y0e')
    t_label.rename_column('col7', 'vx')
    t_label.rename_column('col8', 'vy')
    t_label.rename_column('col9', 'vxe')
    t_label.rename_column('col10','vye')
    t_label.rename_column('col11','t0')
    t_label.rename_column('col12','use')
    t_label.rename_column('col13','r0')

    # Convert velocities from mas/yr to arcsec/year
    t_label['vx'] *= 0.001
    t_label['vy'] *= 0.001
    t_label['vxe'] *= 0.001
    t_label['vye'] *= 0.001

    # propogate to prop_to_time if prop_to_time is given
    if prop_to_time != None:
        x0 = t_label['x0']
        x0e = t_label['x0e']
        vx = t_label['vx']
        vxe = t_label['vxe']
        y0 = t_label['y0']
        y0e = t_label['y0e']
        vy = t_label['vy']
        vye = t_label['vye']
        t0 = t_label['t0']
        t_label['x'] = x0 + vx*(prop_to_time - t0)
        t_label['y'] = y0 + vy*(prop_to_time - t0)
        t_label['xe'] = np.sqrt(x0e**2 + (prop_to_time - t0)**2 * vxe**2)
        t_label['ye'] = np.sqrt(y0e**2 + (prop_to_time - t0)**2 * vye**2)
        t_label['x'].format = '.5f'
        t_label['y'].format = '.5f'
        t_label['xe'].format = '.5f'
        t_label['ye'].format = '.5f'
    
    # flip the x axis if flipX is True
    if flipX == True:
        t_label['x0'] = t_label['x0'] * (-1.0)
        t_label['vx'] = t_label['vx'] * (-1.0)
        if prop_to_time != None:
            t_label['x'] = t_label['x'] * (-1.0)

    return t_label


def read_starlist(starlistFile, error=True):
    """
    Read in a starlist file, rename columns with standard names.
    Assumes the starlist is the reference, so we have time as
    t and don't try to propogate positions to a different time.

    Parameter:
    ---------
    starlistFile: text file, containing:
        col1: name
        col2: mag
        col3: t
        col4: x (pix)
        col5: y (pix)
        if error==True:
            col6: xerr
            col7: yerr
            col8: SNR
            col9: corr
            col10: N_frames
            col11: flux
        else:
            col6: ? (left as default)
            col7: corr
            col8: N_frames
            col9: ? (left as default)
        
    error: boolean (default=True)
        If true, assumes starlist has error columns. This significantly
        changes the order of the columns.
    
    Output:
    ------
    starlist astropy table.
    containing: name, m, x, y, xe, ye, t
    """
    t_ref = Table.read(starlistFile, format='ascii', delimiter='\s')

    # Check if this already has column names:
    cols = t_ref.colnames
    
    if cols[0] != 'col1':
        t_ref['name'] = t_ref['name'].astype(str)
        return t_ref

    t_ref.rename_column(cols[0], 'name')
    t_ref['name'] = t_ref['name'].astype(str)
    t_ref.rename_column(cols[1], 'm')
    t_ref.rename_column(cols[2], 't')
    t_ref.rename_column(cols[3], 'x')
    t_ref.rename_column(cols[4], 'y')
    
    if error==True:
        t_ref.rename_column(cols[5], 'xe')
        t_ref.rename_column(cols[6], 'ye')
        t_ref.rename_column(cols[7], 'snr')
        t_ref.rename_column(cols[8], 'corr')
        t_ref.rename_column(cols[9], 'N_frames')
        t_ref.rename_column(cols[10], 'flux')
    else:
        t_ref.rename_column(cols[5], 'snr')
        t_ref.rename_column(cols[6], 'corr')
        t_ref.rename_column(cols[7], 'N_frames')
        t_ref.rename_column(cols[8], 'flux')
        
    return t_ref


class StarList(Table):
    """
    A StarList is an astropy.Table with star catalog from a single image.

    Required table columns (input as keywords):
    -------------------------
    name : 1D numpy.array with shape = N_stars
        List of names of the stars in the table.

    x : 1D numpy.array with shape = N_stars
        Positions of N_stars in the x dimension.

    y : 1D numpy.array with shape = N_stars
        Positions of N_stars in the y dimension.

    m : 1D numpy.array with shape = N_stars
        Magnitudes of N_stars.

    Optional table columns (input as keywords):
    -------------------------
    xe : 1D numpy.array with shape = N_stars
        Position uncertainties of N_stars in the x dimension.

    ye : 1D numpy.array with shape = N_stars
        Position uncertainties of N_stars in the y dimension.

    me : 1D numpy.array with shape = N_stars
        Magnitude uncertainties of N_stars.
        
    corr : 1D numpy.array with shape = N_stars
        Fitting correlation of N_stars.

    Optional table meta data
    -------------------------
    list_name : str
        Name of the starlist.

    list_time : int or float
        Time/date of the starlist.


    """
    
    def __init__(self, *args, **kwargs):
        """
        """
        # Check if the required arguments are present
        arg_req = ('name', 'x', 'y', 'm')

        found_all_required = True
        
        for arg_test in arg_req:
            if arg_test not in kwargs:
                found_all_required = False

        if not found_all_required:
            if not ('copy' in kwargs) | ('names' in kwargs.keys()) | \
                ('masked' in kwargs.keys()): # If it's not making a copy of the
                # StarList or replacing columns or selecting from slices
                err_msg = "The StarList class requires a arguments" + str(arg_req)
                warnings.warn(err_msg, UserWarning)
            Table.__init__(self, *args, **kwargs)
        else:
            # If we have errors, we need them in both dimensions.
            if ('xe' in kwargs) ^ ('ye' in kwargs):
                raise TypeError("The StarList class requires both 'xe' and" +
                                " 'ye' arguments")

            # Figure out the shape
            n_stars = kwargs['x'].shape[0]

            # Check if the type and size of the arguments are correct.
            # Name checking: type and shape
            if (not isinstance(kwargs['name'], np.ndarray)) or (
                len(kwargs['name']) != n_stars):
                err_msg = "The '{0:s}' argument has to be a numpy array "
                err_msg += "with length = {1:d}"
                raise TypeError(err_msg.format('name', n_stars))

            # Check all the arrays.
            arg_tab = ('x', 'y', 'm', 'xe', 'ye', 'me', 'corr')

            for arg_test in arg_tab:
                if arg_test in kwargs:
                    if not isinstance(kwargs[arg_test], np.ndarray):
                        err_msg = "The '{0:s}' argument has to be a numpy array"
                        raise TypeError(err_msg.format(arg_test))

                    if kwargs[arg_test].shape != (n_stars,):
                        err_msg = "The '{0:s}' argument has to have shape = ({1:d},)"
                        raise TypeError(err_msg.format(arg_test, n_stars))

            # We have to have special handling of meta-data
            meta_tab = ('list_time', 'list_name')
            meta_type = ((float, int), str)
            for mm in range(len(meta_tab)):
                meta_test = meta_tab[mm]
                meta_type_test = meta_type[mm]

                if meta_test in kwargs:

                    if not isinstance(kwargs[meta_test], meta_type_test):
                        err_msg = "The '{0:s}' argument has to be a {1:s}."
                        raise TypeError(
                            err_msg.format(meta_test, str(meta_type_test)))

            #####
            # Create the starlist
            #####
            Table.__init__(self,
                           (kwargs['name'], kwargs['x'], kwargs['y'], kwargs['m']),
                           names=('name', 'x', 'y', 'm'))
            self.meta = {'n_stars': n_stars}

            for meta_arg in meta_tab:
                if meta_arg in kwargs:
                    self.meta[meta_arg] = kwargs[meta_arg]

            for arg in arg_tab:
                if arg in ['name', 'x', 'y', 'm']:
                    continue
                if arg in kwargs:
                    self.add_column(Column(data=kwargs[arg], name=arg))
        
        return

    @classmethod
    def from_lis_file(cls, filename, error=True, fvu_file=None):
        """
        Read in a starlist file, rename columns with standard names.
        Assumes the starlist is the reference, so we have time as
        t and don't try to propogate positions to a different time.

        Parameter:
        ---------
        starlistFile: text file, containing:
            col1: name
            col2: mag (name=m)
            col3: t
            col4: x (pix)
            col5: y (pix)
            if error==True:
                col6: xerr (name=xe)
                col7: yerr (name=ye)
                col8: SNR (name=snr)
                col9: corr
                col10: N_frames
                col11: flux
            else:
                col6: ? (left as default)
                col7: corr
                col8: N_frames
                col9: ? (left as default)

            Note that an 'me' column will be added if error=True set to 1.0 / snr.

        error: boolean (default=True)
            If true, assumes starlist has error columns. This significantly
            changes the order of the columns.

        Output:
        ------
        starlists.StarList() object (subclass of Astropy Table).
        """
        t_ref = Table.read(filename, format='ascii', delimiter='\s')


        # Check if this already has column names:
        cols = t_ref.colnames

        if cols[0] != 'col1':
            t_ref['name'] = t_ref['name'].astype(str)
            return cls.from_table(t_ref)

        t_ref.rename_column(cols[0], 'name')
        t_ref['name'] = t_ref['name'].astype(str)
        t_ref.rename_column(cols[1], 'm')
        t_ref.rename_column(cols[2], 't')
        t_ref.rename_column(cols[3], 'x')
        t_ref.rename_column(cols[4], 'y')

        if error==True:
            t_ref.rename_column(cols[5], 'xe')
            t_ref.rename_column(cols[6], 'ye')
            t_ref.rename_column(cols[7], 'snr')
            t_ref.rename_column(cols[8], 'corr')
            t_ref.rename_column(cols[9], 'N_frames')
            t_ref.rename_column(cols[10], 'flux')
            t_ref['me'] = 1.0 / t_ref['snr']
        else:
            t_ref.rename_column(cols[5], 'snr')
            t_ref.rename_column(cols[6], 'corr')
            t_ref.rename_column(cols[7], 'N_frames')
            t_ref.rename_column(cols[8], 'flux')

        if fvu_file is not None:
            t_fvu = Table.read(fvu_file, format='ascii.no_header')
            t_fvu.rename_column('col1', 'sqrt_fvu')
            t_fvu.add_column(t_fvu['sqrt_fvu']**2, name='fvu')

            if len(t_fvu) != len(t_ref):
                msg = 'Star list and metric list have different lengths.\n'
                msg += '\t len(stars) = {0:d}\n'
                msg += '\t len(fvu) = {1:d}\n'
                
                raise RuntimeError(msg.format(len(t_ref), len(t_fvu)))
            
            t_ref = astropy.table.hstack([t_ref, t_fvu])        

        return cls.from_table(t_ref)

    def to_lis_file(self, filename):
        _out = open(filename, 'w')
        
        hdr = '{name:13s}  {mag:>6s}  {year:>8s}  '
        hdr += '{x:>9s}  {y:>9s}  {xe:>9s}  {ye:>9s}  '
        hdr += '{snr:>20s}  {corr:>6s}  {nimg:>8s}  {flux:>20s}\n'

        _out.write(hdr.format(name='# name', mag='m', year='t',
                              x='x', y='y', xe='xe', ye='ye',
                              snr='snr', corr='corr', nimg='N_frames', flux='flux'))
    

        fmt = '{name:13s}  {mag:6.3f}  {year:8.3f}  '
        fmt += '{x:9.3f}  {y:9.3f}  {xe:9.3f}  {ye:9.3f}  '
        fmt += '{snr:20.4f}  {corr:6.2f}  {nimg:8d}  {flux:20.0f}\n'

        for ss in range(len(self)):
            _out.write(fmt.format(name=self['name'][ss], mag=self['m'][ss], year=self['t'][ss],
                                  x=self['x'][ss], y=self['y'][ss], xe=self['xe'][ss], ye=self['ye'][ss],
                                  snr=self['snr'][ss], corr=self['corr'][ss], nimg=self['N_frames'][ss],
                                  flux=self['flux'][ss]))

        _out.close()
        
        return
    
    
    @classmethod
    def from_table(cls, table):
        """
        Make a StarList from an astropy.table.Table object. Note that the input table
        must have the columns ['name', 'x', 'y', 'm']. All other columns and meta data
        will be added to the new StarList object that is returned.
        """
        starlist = cls(name=table['name'], x=table['x'], y=table['y'], m=table['m'], meta=table.meta)
        
        for col in table.colnames:
            if col in ['name', 'x', 'y', 'm']:
                continue
            else:
                starlist.add_column(table[col])

        return starlist

    def fubar(self):
        print('This is in StarList')
        return
    
    def restrict_by_value(self, **kwargs):
        """
        Restrict a table to any min/max range of column values. For instance, 
        to restrict to only stars between 10 <= m <= 15, use:

        starlist.restrict_by_value(m_min=10, m_max=15)

        where 'm' was the column name.

        This function acts on self, so the rows are removed
        forever. 
        """
        # Loop through all conditions and build up
        # an array of indicies of rows to remove. 
        remove_flag = np.zeros(len(self), dtype=bool)
        
        for kwarg in kwargs:
            if kwargs[kwarg] is not None:
                # Get the name of the column to act on and
                # whether the condition is min or max.
                kwarg_split = kwarg.split('_')

                # Support column names such as x_0. 
                col = '_'.join(kwarg_split[:-1])

                if kwarg_split[-1] == 'min':
                    remove_flag = np.logical_or(remove_flag, self[col] <= kwargs[kwarg])
                
                if kwarg_split[-1] == 'max':
                    remove_flag = np.logical_or(remove_flag, self[col] >= kwargs[kwarg])

        rem_idx = np.where(remove_flag == True)[0]
        
        self.remove_rows(rem_idx)
        
        return

    def transform_xym(self, trans):
        """
        Apply a transformation (instance of flystar.transforms.Transform2D)
        to the x, y, m, xe, ye, me columns.

        Note that this case will handle trans == None (nothing is done).
        """

        self.transform_xy(trans)
        self.transform_m(trans)
            
        return


    def transform_xy(self, trans):
        """
        Apply a transformation (instance of flystar.transforms.Transform2D)
        to the x, y, xe, ye columns.

        Note that this case will handle trans == None (nothing is done).
        """
        if trans == None:
            return

        x_T, y_T = trans.evaluate(self['x'], self['y'])
        self['x'] = x_T
        self['y'] = y_T

        if 'xe' in self.colnames:
            xe_T, ye_T = trans.evaluate_error(self['x'], self['y'], self['xe'], self['ye'])
            self['xe'] = xe_T
            self['ye'] = ye_T

        return

    def transform_xym_CTE(self, trans):
        """
        FIXME

        Apply a transformation (instance of flystar.transforms.Transform2D)
        to the x, y, xe, ye columns.

        Note that this case will handle trans == None (nothing is done).
        """
        if trans == None:
            return
        
        x_T, y_T, m_T = trans.evaluate(self['x'], self['y'], self['m'])
        self['x'] = x_T
        self['y'] = y_T
        self['m'] = m_T

        # FIXME: I don't understand how this works?? This uses the updated values to calculate the error?
        if 'xe' in self.colnames:
            xe_T, ye_T, me_T = trans.evaluate_error(self['x'], self['y'], self['m'],
                                                        self['xe'], self['ye'], self['me'])
            self['xe'] = xe_T
            self['ye'] = ye_T
            self['me'] = me_T

        return
    
    def transform_m(self, trans):
        """
        Apply a transformation (instance of flystar.transforms.Transform2D)
        to the m and me column.

        Note that this case will handle trans == None (nothing is done).
        """
        if trans == None:
            return
    
        m_T = trans.evaluate_mag(self['m'])
        self['m'] = m_T

        if 'me' in self.colnames:
            me_T = trans.evaluate_magerror(self['m'], self['me'])
            self['me'] = me_T
    
        return
