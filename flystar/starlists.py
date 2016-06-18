import numpy as np
from sets import Set
from astropy.table import Table

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
    Name = Set(name1) & Set(name2)

    idx1 = np.array([list(name1).index(i) for i in Name])
    idx2 = np.array([list(name2).index(i) for i in Name])

    return idx1, idx2, len(idx1)

def restrict_by_area(table1, area):
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
    good = np.where( (xpos > x_range[0]) & (xpos < x_range[1]) &
                      (ypos > y_range[0]) & (ypos < y_range[1]) )

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
    print 'Restrict option activated'

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
    print 'Restrict option activated'
    print 'Keeping {0} of {1} stars'.format(len(idx_restrict),
                                            len(label_mat))
    
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
        col13: r2d (arcsec)

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
    containing name, m, x0, y0, x0e, y0e, vx, vy, vxe, vye, t0, use, r2d,
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
        t_label['x'].format = '.3f'
        t_label['y'].format = '.3f'
        t_label['xe'].format = '.3f'
        t_label['ye'].format = '.3f'
    
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
        return t_ref

    t_ref.rename_column(cols[0], 'name')
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
