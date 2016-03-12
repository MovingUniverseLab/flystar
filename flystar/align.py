import match
import transforms
import datetime
import os
import pdb

def initial_align(table1, table2, transformModel, briteN):
    """
    Calculates an initial (unweighted) transformation between two sets of
    starlists. Matching is done using a blind triangle-matching algorithm
    of the brightest briteN stars in both starlists.

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
    
    t: epoch of observation

    use?: specify use in transformation 
    

    Parameters:
    ----------





    Output:
    ------
    Transformation object
    
    """
    # Extract necessary information from tables (x, y, m)
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']




    # Check to make sure epoch match
    if table1['t'][0] != table2['t'][0]:
        print 'Warning: starlist epochs do not match'
        
    # Run the blind triangle-matching algorithm to find the matches between
    # the starlists


    # Calculate transformation based on matches


    return transform



def transform_and_match(table1, table2, transform, dr_tol, dm_tol=None):
    """
    Transform starlist 2 into starlist 1 frame and match stars to given radius
    and magnitude tolerance.

    Starlists must be astropy tables with standard columns names as specified
    in initial_align. 

    Parameters:
    -----------
    table1: astropy table

    transform: transformation object



    Output:
    -------
    
    """
    # Transform x, y coordinates from starlist 2 into starlist 1


    # Match starlist 1 and 2


    
    # Output matched starlists


    return




def find_transform(matchedStarlist, transModel=transform.four_paramNW, order=1, weights=False):
    """
    Given a matched starlist, derive a new transform. This transformation is
    calculated for starlist 2 into starlist 1

    Parameters:
    -----------
    matchedStarlist: fits table
        Matched starlist output from transform_and_match.

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
    
    # Read matched starlist, extract *untransformed* coordinates from starlist 2
    # and the matching coordinates from starlist 1
    match = Table.read(matchedStarlist, format='fits')

    x1 = match[XX]
    y1 = match[XX]

    x2 = match[XX]
    y2 = match[XX]

    # Calculate transform based on the matched stars    
    t = transModel(x2, y2, x1, y1, order=order, weights=weights)

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
        be applied to

    reference: string
        File name of reference; this is what the starlist is transformed to

    N_trans: int
        Number of stars used in the transformation

    restrict: boolean (default=False)
        Set to True if transformation restricted to stars with use? > 2 

    outFile: string (default: 'outTrans.txt')
        Name of output text file
         
    Output:
    ------
    txt file with the file name outFile
    """
    transType = transformation.type
    
    # Extract X, Y coefficients from transform
    if transType == 'fourParam':
        Xcoeff = transformation.px
        Ycoeff = transformation.py
    elif transType.startswidth('PolyTransform'):
        Xcoeff = transformation.px.parameters
        Ycoeff = transformation.py.parameters
    else:
        print '{0} not yet supported!'.format(transType)
        return
        
    # Write output
    _out = open(outFile, 'w')
    
    # Write the header. DO NOT CHANGE, HARDCODED IN JAVA ALIGN
    _out.write('## Transformation: {0}'.format(datetime.date.today()) )
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
    Xcoeff = trans['Xcoeff']
    Ycoeff = trans['Ycoeff']

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



def readLabel(labelFile, t0):
    """
    Read in a label.dat file, rename columns with standard
    names. Use velocities to convert positions into epoch
    t0, and then flip x positions and velocities such that
    +x is to the west.

    Update values in columns as necessary

    Parameters:
    ----------





    Output:
    ------
    
    """



    return



def readStarlist(starlistFile):
    """
    Read in a starlist file, rename columns with standard
    names

    Parameter:
    ---------


    Output:
    ------
    """



    return 



def align_example(labelFile, reference, transModel=transform.four_paramNW, order=1, N_loop=2):
    """
    Base example of how to use the flystar code.

    Parameters:
    -----------
    labelFile: ascii file
        Starlist we would like to transform into the reference frame. For this
        code, we expect a label.dat file

    reference: ascii file
        Starlist that defines the reference frame

    transModel: transformation class (default: transform.four_paramNW)
        Defines which transformation model to use. Only the four-parameter and
        1st/2nd order polynomial transformations are supported

    order: int (default=1)
        Order of the polynomial transformation. Only used for polynomial transform

    N_loop: int (default=2)
        How many times to iterate on the transformation calculation. Ideally,
        each iteration adds more stars and thus a better transform, to some
        limit.

    Output:
    ------
        
    """
    # Read in label.dat file and reference starlist, changing columns to their
    # standard column headers/epochs/orientations
    starlist = readStarlist(reference)
    t0 = starlist['t'][0]

    label = readLabel(labelFile, t0)

    # Perform blind matching and calculate initial transform
    trans = initial_align()


    # Use transformation to match starlists, then recalculate transformation.
    # Iterate on this as many times as desired
    for i in range(N_loop):
        matched = transform_and_match()

        trans = find_transform()


    # Write final transform
    write_transform()


    
    # Test transform: apply to starlists, make diagnostic plots
    label_trans = transform(label, 'outTrans.txt')
    

    # Diagnostic plots


    
    return
    
