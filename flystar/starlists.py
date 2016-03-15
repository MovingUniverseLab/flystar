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
    table1_trim, table2_trim
    """

    name1 = table1['name']
    name2 = table2['name']
    Name = Set(name1) & Set(name2)

    idx1 = np.array([list(name1).index(i) for i in Name])
    idx2 = np.array([list(name2).index(i) for i in Name])

    return table1[idx1], table2[idx2], len(idx1)

def restrict_by_area(table1, area):
    """
    Restrict starlist to those within a specific area. Note: the output starlist
    will ONLY contain stars that fulfills the area condition. 

    Parameters:
    ----------
    table1: astropy table
         Starlist to be restricted. Must have standard column headers
         (i.e. x, y)

    area: 2x2 array [[x1, x2], [y1, y2]]
        X and Y coordinate range to restric the stars too. Only stars with
        coordinates with x1 < X < x2 and y1 < Y < y2 will be allowed.
    
    Output:
    ------
    astropy table: same as input table, only with stars that pass the area
    restriction
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

    table_out = table1[good]

    return table_out

def restrict_by_use(label_mat_orig, label_mat, starlist_mat):
    """
    Restrict matching starlist to only those where the label_mat['use'] > 2.
    This behaves the same way as the -restrict flag in java align. 

    Parameters:
    -----------
    label_mat_orig: astropy table
         Label.dat table containing the matched stars in the original coordinates.
         Must have standard column headers.

    label_mat: astropy table
         Label.dat table containing the matched stars in the reference coordinates.
         Must have standard column headers.

    starlist_mat: astropy table
         Reference table containing the matched stars. Must have standard column
         headers
         
    Output:
    -------
    label_trim: astropy table
         label table with only use > 2 stars

    starlist_trim: astropy table
         reference table with only stars that correspond to use > 2 stars
         in the label_mat table.
    
    """
    print 'Restrict option activated'

    # Among label.dat matched stars, determine which are allowed by -restrict
    idx_restrict = np.where(label_mat['use'] !='0')

    # Restrict tables to only these stars.
    label_orig_trim = label_mat_orig[idx_restrict]
    label_trim = label_mat[idx_restrict]
    starlist_trim = starlist_mat[idx_restrict]

    return label_orig_trim, label_trim, starlist_trim
