import match
import transforms
from astropy.table import Table

def initial_align(table1, table2, briteN, transformModel, order):
    """
    Calculates an initial (unweighted) transformation between two sets of
    starlists. Matching is done using a blind triangle-matching algorithm
    of the brightest briteN stars in both starlists. Transformation is done
    using the transformModel in the input parameter.

    Starlists must be astropy tables with standard column headers. All
    positions must be at the same epoch, and +x must be in the same
    direction.

    Standard column headers:
    name: name
    x: x position
    y: y position
    xe: error in x position
    ye: error in y position
    
    vx: proper motion in x direction
    vy proper motion in y direction
    vxe: error in x proper motion
    vye: error in y proper motion

    m: magnitude
    me: magnitude error
    
    t0: linear motion time zero point

    use: specify use in transformation 
    

    Parameters:
    ----------
    -table1: astropy.table 
        contains name,m,x,y,xe,ye,vx,vy,vxe,vye,t0. 

    -table2: astropy.table
        contains name,m,x,y,xe,ye.
        this is the reference template

    -briteN: int
        The number of brightest stars used to match two starlists.    

    -transformModel:  transformation model object (class)
        The transformation model class that will be instantiated to find the
        best-fit transformation parameters between matched table1 and table2.
        eg: transforms.four_paramNW, transforms.PolyTransform

    -order: int
         Order of the transformation. Not relevant for 4 parameter or spline fit


    Output:
    ------
    Transformation object
    
    """
    # Extract necessary information from tables (x, y, m)
    table1 = Table.read(table1, format='ascii')
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']
    table2 = Table.read(table2, format='ascii')
    x2 = table2['x']
    y2 = table2['y']
    m2 = table2['m']


        
    # Run the blind triangle-matching algorithm to find the matches between the starlists
    print 'Attempting match with {0} and {1} stars from starlist1 and starlist2'.format(len(table1), len(table2))
    print 'Beginning match'
    N, x1m, y1m, m1m, x2m, y2m, m2m = match.miracle_match_briteN(x1, y1, m1, x2, y2, m2, briteN)
    assert len(x1m) > req_match#, 'Failed to find at least '+str(req_match+' matches, giving up'
    print '{0} stars matched wetween starlist1 and starlist2'.format(len(N))

    # Calculate transformation based on matches
    if transformModel == transforms.four_paramNW:
        t = transformModel(x1m, y1m ,x2m, y2m, order=order, weights=weights)
    else:
        t = transformModel(x1m, y1m ,x2m, y2m, degree=order, weights=weights)

    return t



def transform_and_match(table1, table2, transform, dr_tol, dm_tol=None):
    """
    apply transformation to starlist1 and 
    match stars to given radius and magnitude tolerance.

    Starlists must be astropy tables with standard columns names as specified
    in initial_align. 

    Parameters:
   -----------
    -table1: astropy.table 
        contains name,m,x,y,xe,ye,vx,vy,vxe,vye,t0. 

    -table2: astropy.table
        contains name,m,x,y,xe,ye.
        this is the reference template

    -transform: transformation object


    Output:
    -------
    matched lists of transformed and untransformed starlists.
    
    """

    # Extract necessary information from tables (x, y, m)
    table1 = Table.read(table1, format='ascii')
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']
    table2 = Table.read(table2, format='ascii')
    x2 = table2['x']
    y2 = table2['y']
    m2 = table2['m']


    # Transform x, y coordinates from starlist 1 into starlist 2
    x1t, y1t = transform.evaluate(x1, y1)

    # Match starlist 1 and 2
    idx1, idx2, dm, dr = match.match(x1t, y1t, m1, x2, y2, m2, dr_tol) 
    
    # Output matched starlists
    table1 = table1[idx1]
    table1T = table1
    table1T['x'] = x1t
    table1T['y'] = y1t
    table2 = table2[idx2]

    return table1, table1T, table2



def find_transform(matchedStarlist, transModel, weights=False):
    """
    Given a matched starlist, derive a new transform

    """
    # Read matched starlist

    # Calculate transformation based on matched stars


    # Output transformation

    return





def readLabel(labelFile, tref):
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
        col3: x (arcsec)
        col4: y 
        col5: xerr
        col6: yerr
        col7: vx (mas/yr)
        col8: vy 
        col9: vxerr
        col10: vyerr
        col11: t0
        col12: use
    tref: reference epoch that label.dat is converted to. 


    Output:
    ------
    labelFile: astropy.table. 
    containing name, m, x, y, xe, ye, vx, vy, vxe, vye, use
    
    x and y is in arcsec, 
    converted to tref epoch, 
    *(-1) so it increases to west
    
    vx, vy, vxe, vye is converted to arcsec/yr

    """
    t_label = Table.read(labelFile, format='ascii')
    t_label.rename_column('col1', 'name')
    t_label.rename_column('col2', 'm')
    t_label.rename_column('col3', 'x')
    t_label.rename_column('col4', 'y')
    t_label.rename_column('col5', 'xe')
    t_label.rename_column('col6', 'ye')
    t_label.rename_column('col7', 'vx')
    t_label.rename_column('col8', 'vy')
    t_label.rename_column('col9', 'vxe')
    t_label.rename_column('col10','vye')
    t_label.rename_column('col11','t0')
    t_label.rename_column('col12','use')
    
    t_label['vx'] *= 0.001
    t_label['vy'] *= 0.001
    t_label['vxe'] *= 0.001
    t_label['vye'] *= 0.001

    t_label['x'] = t_label['x'] + t_label['vx']*(tref - t_label['t0'])
    t_label['y'] = t_label['y'] + t_label['vy']*(tref - t_label['t0'])
    
    # flip the x axis, because lable.dat increase to the east,
    # reference frame increase to the west.
    t_label['x'] = t_label['x'] * (-1.0)
    
    return t_label



def readStarlist(starlistFile):
    """
    Read in a starlist file, rename columns with standard names

    Parameter:
    ---------
    starlistFile: text file, containing:
        col1: name
        col2: mag
        col4: x (pix)
        col5: y
        col6: xerr
        col7: yerr
        col3: t

    Output:
    ------
    starlist astropy table. 
    containing: name, m, x, y, xe, ye, t  
    """

    t_ref = Table.read(starlistFile, format='ascii')
    t_ref.rename_column('col1', 'name')
    t_ref.rename_column('col2', 'm')
    t_ref.rename_column('col4', 'x')
    t_ref.rename_column('col5', 'y')
    t_ref.rename_column('col6', 'xe')
    t_ref.rename_column('col7', 'ye')
    t_ref.rename_column('col3', 't')

    return t_ref 



def align_example():
    """"
    Example of how to use the flystar code.
    """
    # Read in label.dat, reference starlists






    
    return
    
