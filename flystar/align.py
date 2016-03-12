import match
import transforms
from astropy.table import Table, Column
import datetime
import os
import pdb

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



def find_transform(table1_mat, table2_mat, transModel=transform.four_paramNW, order=1, weights=False):
    """
    Given a matched starlist, derive a new transform. This transformation is
    calculated for starlist 1 into starlist 2

    Parameters:
    -----------
    table1_mat: astropy table
        Table with matched stars from starlist 1, with original positions
        (not transformed into starlist 2 frame)

    table2_mat: astropy table
        Table with matched stars from starlist 2, in starlist 2 frame.

    transModel: transformation class (default: transform.four_paramNW)
        Specify desired transform, e.g. four_paramNW or PolyTransform. If
        PolyTransform is selected, order defines the order of polynomial used

    order: int (default=1)
        Order of polynomial to use in the transformation. Only active if
        PolyTransform is selected

    weights: boolean (default=False)
        If true, use weights when calculating transformation. Weights are
        calculated in function find_weights

    Output:
    ------
    transformation object
    """
    # First, check that desired transform is supported
    if ( (transModel != transform.four_paramNW) & (transModel != transform.PolyTransform) ):
        print '{0} not supported yet!'.format(transModel)
        return
    
    # Extract *untransformed* coordinates from starlist 1 
    # and the matching coordinates from starlist 2
    x1 = table1_mat['x']
    y1 = table1_mat['y']
    x2 = table2_mat['x']
    y2 = table2_mat['y']

    # Calculate transform based on the matched stars    
    t = transModel(x1, y1, x2, y2, order=order, weights=weights)

    print '{0} stars used in transform'.format(len(x1))

    # Return transformation object
    return t


def write_transform(transformation, starlist, reference, N_trans, restrict=False, outFile='outTrans.txt'):
    """
    Given a transformation object, write out the coefficients in a java align
    readable format. Outfile name is specified by user

    Parameters:
    ----------
    transformation: transformation object
        Transformation object we want to feed into java align

    starlist: string
        File name of starlist; this is the starlist the transformation should
        be applied to. For output purposes only

    reference: string
        File name of reference; this is what the starlist is transformed to.
        For output purposes only

    N_trans: int
        Number of stars used in the transformation

    restrict: boolean (default=False)
        Set to True if transformation restricted to stars with use > 2. Purely
        for output purposes

    outFile: string (default: 'outTrans.txt')
        Name of output text file
         
    Output:
    ------
    txt file with the file name outFile
    """    
    # Extract X, Y coefficients from transform
    if transformation.__class__.__name__ == 'four_paramNW':
        Xcoeff = transformation.px
        Ycoeff = transformation.py
    elif transformation.__class__.__name__ == 'PolyTransform':
        Xcoeff = transformation.px.parameters
        Ycoeff = transformation.py.parameters
    else:
        print '{0} not yet supported!'.format(transType)
        return
        
    # Write output
    _out = open(outFile, 'w')
    
    # Write the header. DO NOT CHANGE, HARDCODED IN JAVA ALIGN
    _out.write('## Date: {0}'.format(datetime.date.today()) )
    _out.write('## File: {0}, Reference: {1}'.format(starlist, reference) )
    _out.write('## Directory: {0}'.format(os.getcwd()) )
    _out.write('## Transform Class: {0}'.format(transformation.__class__.__name__))
    _out.write('## Order: {0}'.format(transformation.order))
    _out.write('## N_coeff: {0}'.format(len(Xcoeff)))
    _out.write('## N_trans: {0}'.format(N_trans))
    _out.write('{0:-16s} {1:-16s}'.format('# Xcoeff', 'Ycoeff')
    
    # Write the coefficients
    for i in range(len(Xcoeff)):
        _out.write('{0:16.6e}  {1:16.6e}'.format(Xcoeff[i], Ycoeff[i]) )
    
    _out.close()
    
    return


def transform(starlist, transFile):
    """
    Apply transformation from transFile to starlist. Returns astropy table with
    added columns with the transformed coordinates. NOTE: Transforms
    positions/position errors, plus velocities and velocity errors if they
    are present in starlist
    
    Parameters:
    ----------
    starlist: astropy table
         Starlist we want to apply the transformation too. Must already
         have standard column headers

    transFile: ascii file
        File with the transformation coefficients. Assumed to be output of
        write_transform

    Output:
    ------
    starlist astropy table with additional columns (x_trans/y_trans, xe_trans/ye_trans,
    vx_trans/vy_trans, vxe_trans/vye_trans) containing the transformed coordinates. 
    """
    # Check to see if velocities are present in starlist. If so, we will
    # need to transform these as well as positions
    vel = False
    keys = starlist.keys()
    if 'vx' in keys:
        vel = True 
    
    # Extract needed information from starlist
    x_orig = starlist['x']
    y_orig = starlist['y']
    xe_orig = starlist['xe']
    ye_orig = starlist['ye']

    if vel:
        vx_orig = starlist['vx']
        vy_orig = starlist['vy']
        vxe_orig = starlist['vxe']
        vye_orig = starlist['vye']
    
    # Read transFile
    trans = Table.read(transFile, format='ascii')
    Xcoeff = trans['col1']
    Ycoeff = trans['col2']

    # How the transformation is applied depends on the type of transform.
    # This can be determined by the length of Xcoeff, Ycoeff
    if len(Xcoeff) == 3:
        x_new = Xcoeff[0] + Xcoeff[1] * x_orig + Xcoeff[2] * y_orig
        y_new = Ycoeff[0] + Ycoeff[1] * x_orig + Ycoeff[2] * y_orig
        xe_new = Xcoeff[0] + Xcoeff[1] * xe_orig + Xcoeff[2] * ye_orig
        ye_new = Ycoeff[0] + Ycoeff[1] * xe_orig + Ycoeff[2] * ye_orig

        if vel:
            vx_new = Xcoeff[1] * vx_orig + Xcoeff[2] * vy_orig
            vy_new = Ycoeff[1] * vx_orig + Ycoeff[2] * vy_orig
            vxe_new = Xcoeff[1] * vxe_orig + Xcoeff[2] * vye_orig
            vye_new = Ycoeff[1] * vxe_orig + Ycoeff[2] * vye_orig

    elif len(Xcoeff) == 6:
        x_new = Xcoeff[0] + Xcoeff[1]*x_orig + Xcoeff[2]*y_orig + Xcoeff[3]*x_orig**2. + \
          Xcoeff[4]*y_orig**2. + Xcoeff[5]*x_orig*y_orig
          
        y_new = Ycoeff[0] + Ycoeff[1]*x_orig + Ycoeff[2]*y_orig + Ycoeff[3]*x_orig**2. + \
          Ycoeff[4]*y_orig**2. + Ycoeff[5]*x_orig*y_orig
          
        xe_new = Xcoeff[0] + Xcoeff[1]*xe_orig + Xcoeff[2]*ye_orig + Xcoeff[3]*xe_orig**2. + \
          Xcoeff[4]*ye_orig**2. + Xcoeff[5]*xe_orig*ye_orig
          
        ye_new = Ycoeff[0] + Ycoeff[1]*xe_orig + Ycoeff[2]*ye_orig + Ycoeff[3]*xe_orig**2. + \
          Ycoeff[4]*ye_orig**2. + Ycoeff[5]*xe_orig*ye_orig

        if vel:
            vx_new = Xcoeff[1]*vx_orig + Xcoeff[2]*vy_orig + 2.*Xcoeff[3]*x_orig*vx_orig + \
                2.*Xcoeff[4]*y_orig*vy_orig. + Xcoeff[5]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vy_new = Ycoeff[1]*vx_orig + Ycoeff[2]*vy_orig + 2.*Ycoeff[3]*x_orig*vx_orig + \
                2.*Ycoeff[4]*y_orig*vy_orig. + Ycoeff[5]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vxe_new = Xcoeff[1]*vxe_orig + Xcoeff[2]*vye_orig + 2.*Xcoeff[3]*xe_orig*vxe_orig + \
                2.*Xcoeff[4]*ye_orig*vye_orig. + Xcoeff[5]*(xe_orig*vye_orig + vxe_orig*ye_orig)
          
            vye_new = Ycoeff[1]*vxe_orig + Ycoeff[2]*vye_orig + 2.*Ycoeff[3]*xe_orig*vxe_orig + \
                2.*Ycoeff[4]*ye_orig*vye_orig. + Ycoeff[5]*(xe_orig*vye_orig + vxe_orig*ye_orig)
        
    # Add transformed coords to astropy table
    xCol = Column(x_new, name='x_trans')
    yCol = Column(y_new, name='y_trans')
    xeCol = Column(xe_new, name='xe_trans')
    yeCol = Column(ye_new, name='ye_trans')
    
    starlist.add_column(xCol)
    starlist.add_column(yCol)
    starlist.add_column(xeCol)
    starlist.add_column(yeCol)    

    if vel:
        vxCol = Column(vx_new, name='vx_trans')
        vyCol = Column(vy_new, name='vy_trans')
        vxeCol = Column(vxe_new, name='vxe_trans')
        vyeCol = Column(vye_new, name='vye_trans')

        starlist.add_column(vxCol)
        starlist.add_column(vyCol)
        starlist.add_column(vxeCol)
        starlist.add_column(vyeCol) 

        
    return starlist



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

