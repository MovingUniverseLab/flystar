import match
import transforms
from astropy.table import Table, Column
import datetime
import os
import pdb
import numpy as np

def initial_align(table1, table2, briteN=100, transformModel=transforms.four_paramNW, order=1):
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
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']
    x2 = table2['x']
    y2 = table2['y']
    m2 = table2['m']


        
    # Run the blind triangle-matching algorithm to find the matches between the starlists
    print 'Attempting match with {0} and {1} stars from starlist1 and starlist2'.format(len(table1), len(table2))
    print 'Beginning match'

    # number of required matches of the input catalog to the total reference
    req_match = 5

    N, x1m, y1m, m1m, x2m, y2m, m2m = match.miracle_match_briteN(x1, y1, m1, x2, y2, m2, briteN)
    assert len(x1m) > req_match#, 'Failed to find at least '+str(req_match+' matches, giving up'
    print '{0} stars matched between starlist1 and starlist2'.format(N)

    # Calculate transformation based on matches
    t = transformModel(x1m, y1m ,x2m, y2m, order=order, weights=None)

    return t



def transform_and_match(table1, table2, transform, dr_tol=1.0, dm_tol=None):
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

    -dr_tol: float (default=1.0)
        The search radius for the matching algorithm, in the same units as the
        starlist file positions.

    -transform: transformation object


    Output:
    -------
    -astropy table with matched stars from table 1, with *original* coordinates
    -astropy table with matched stars from table 1, with *transformed* coordinates
    -astropy table iwth matched stars from table 2, with original coordinates
    """

    # Extract necessary information from tables (x, y, m)
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']
    x2 = table2['x']
    y2 = table2['y']
    m2 = table2['m']


    # Transform x, y coordinates from starlist 1 into starlist 2
    x1t, y1t = transform.evaluate(x1, y1)

    # Match starlist 1 and 2
    idx1, idx2, dm, dr = match.match(x1t, y1t, m1, x2, y2, m2, dr_tol) 
    
    # Output matched starlists
    table1 = table1[idx1]
    table2 = table2[idx2]
    table1T = Table.copy(table1)
    #table1T = transformAll(table1T, transform)
    table1T['x'] = x1t[idx1]
    table1T['y'] = y1t[idx1]

    print '{0} of {1} stars matched'.format(len(table1), len(x1t))

    return table1, table1T, table2



def find_transform(table1_mat, table1T_mat, table2_mat, transModel=transforms.four_paramNW, order=1, 
                weights='both'):
    """
    Given a matched starlist, derive a new transform. This transformation is
    calculated for starlist 1 into starlist 2

    Parameters:
    -----------
    table1_mat: astropy table
        Table with matched stars from starlist 1, with original positions
        (not transformed into starlist 2 frame)

    table1T_mat: astropy table
        Table with matched stars from starlist 1, with transformed position
        into starlist 2. This is used to calculate weights.

    table2_mat: astropy table
        Table with matched stars from starlist 2, in starlist 2 frame.

    transModel: transformation class (default: transform.four_paramNW)
        Specify desired transform, e.g. four_paramNW or PolyTransform. If
        PolyTransform is selected, order defines the order of polynomial used

    order: int (default=1)
        Order of polynomial to use in the transformation. Only active if
        PolyTransform is selected

    weights: string (default='both')
        if weights=='both', we use both position error and velocity error in table1T_mat 
            and position error table2_mat as uncertanties. And weights is the reciprocal 
            of this uncertanty.
        if weights=='table1', we only use postion error and velocity error in table1T_mat
            as uncertainty.
        if weights=='table2', we only use position error in table2 as uncertainty.
        otherwise, we don't use weights.

    Output:
    ------
    -transformation object
    -number of stars used in transform
    """
    # First, check that desired transform is supported
    if ( (transModel != transforms.four_paramNW) & (transModel != transforms.PolyTransform) ):
        print '{0} not supported yet!'.format(transModel)
        return
    
    # Extract *untransformed* coordinates from starlist 1 
    # and the matching coordinates from starlist 2
    x1 = table1_mat['x']
    y1 = table1_mat['y']
    x2 = table2_mat['x']
    y2 = table2_mat['y']


    # calculate weights.
    x1e = table1T_mat['xe']
    y1e = table1T_mat['ye']
    vx1e = table1T_mat['vxe']
    vy1e = table1T_mat['vye']
    t0 = table1T_mat['t0']

    x2e = table2_mat['x']
    y2e = table2_mat['y']
    t2 = table2_mat['t']
    delt = t2-t0

    if weights=='both':
        weight = 1/np.sqrt( x1e**2 + (vx1e * delt)**2 + x2e**2 + 
                            y1e**2 + (vy1e * delt)**2 + y2e**2)
    elif weights=='table1':
        weight = 1/np.sqrt( x1e**2 + (vx1e * delt)**2 + 
                            y1e**2 + (vy1e * delt)**2 )
    elif weights=='table2':
        weight = 1/np.sqrt( x2e**2 +  y2e**2)
    else:
        weight = None

    # Calculate transform based on the matched stars    
    t = transModel(x1, y1, x2, y2, order=order, weights=weight)

    N_trans = len(x1)
    print '{0} stars used in transform'.format(N_trans)

    # Return transformation object and number of stars used in transform
    return t, N_trans


def write_transform(transformation, starlist, reference, N_trans, deltaMag=0, restrict=False, outFile='outTrans.txt'):
    """
    Given a transformation object, write out the coefficients in a java align
    readable format. Outfile name is specified by user.

    Coefficients are output in file in the following way:
    x' = a0 + a1*x + a2*y + a3*x**2. + a4*x*y  + a5*y**2. + ...
    y' = b0 + b1*x + b2*y + b3*x**2. + b4*x*y  + b5*y**2. + ...

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

    deltaMag: float (default = 0)
        Average magnitude difference between reference and starlist
        (reference - starlist)

    restrict: boolean (default=False)
        Set to True if transformation restricted to stars with use > 2. Purely
        for output purposes

    outFile: string (default: 'outTrans.txt')
        Name of output text file
         
    Output:
    ------
    txt file with the file name outFile
    """
    # Extract info about transformation
    trans_name = transformation.__class__.__name__
    trans_order = transformation.order
      
    # Extract X, Y coefficients from transform
    if trans_name == 'four_paramNW':
        Xcoeff = transformation.px
        Ycoeff = transformation.py
    elif trans_name == 'PolyTransform':
        Xcoeff = transformation.px.parameters
        Ycoeff = transformation.py.parameters
    else:
        print '{0} not yet supported!'.format(transType)
        return
        
    # Write output
    _out = open(outFile, 'w')
    
    # Write the header. DO NOT CHANGE, HARDCODED IN JAVA ALIGN
    _out.write('## Date: {0}\n'.format(datetime.date.today()) )
    _out.write('## File: {0}, Reference: {1}\n'.format(starlist, reference) )
    _out.write('## Directory: {0}\n'.format(os.getcwd()) )
    _out.write('## Transform Class: {0}\n'.format(transformation.__class__.__name__))
    _out.write('## Order: {0}\n'.format(transformation.order))
    _out.write('## Restrict: {0}\n'.format(restrict))
    _out.write('## N_coeff: {0}\n'.format(len(Xcoeff)))
    _out.write('## N_trans: {0}\n'.format(N_trans))
    _out.write('## Delta Mag: {0}\n'.format(deltaMag))
    _out.write('{0:16s} {1:16s}\n'.format('# Xcoeff', 'Ycoeff'))
    

    # Write the coefficients such that the orders are together as defined in
    # documentation. This is a pain because PolyTransform output is weird.
    # (see astropy Polynomial2D documentation)
    if (trans_name == 'four_paramNW') | (trans_order == 1):
        for i in range(len(Xcoeff)):
            _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[i], Ycoeff[i]) )
    elif (trans_name == 'PolyTransform') & (trans_order == 2):
        _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[0], Ycoeff[0]) )
        _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[1], Ycoeff[1]) )
        _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[3], Ycoeff[3]) )
        _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[2], Ycoeff[2]) )
        _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[5], Ycoeff[5]) )
        _out.write('{0:16.6e}  {1:16.6e}'.format(Xcoeff[4], Ycoeff[4]) )

    else:
        print '{0} order {1} not yet supported in write_transform'.format(trans_name,
                                                                          trans_order)
        print 'Stopping'
        pdb.set_trace()

    _out.close()
    
    return


def transform_by_file(starlist, transFile):
    """
    Apply transformation from transFile to starlist. Returns astropy table with
    added columns with the transformed coordinates. NOTE: Transforms
    positions/position errors, plus velocities and velocity errors if they
    are present in starlist.
    
    Parameters:
    ----------
    starlist: astropy table
         Starlist we want to apply the transformation too. Must already
         have standard column headers

    transFile: ascii file
        File with the transformation coefficients. Assumed to be output of
        write_transform, with coefficients specified as code documents

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
    trans = Table.read(transFile, format='ascii.commented_header', header_start=-1)
    Xcoeff = trans['Xcoeff']
    Ycoeff = trans['Ycoeff']

    # How the transformation is applied depends on the type of transform.
    # This can be determined by the length of Xcoeff, Ycoeff
    if len(Xcoeff) == 3:
        x_new = Xcoeff[0] + Xcoeff[1] * x_orig + Xcoeff[2] * y_orig
        y_new = Ycoeff[0] + Ycoeff[1] * x_orig + Ycoeff[2] * y_orig
        xe_new = Xcoeff[1] * xe_orig + Xcoeff[2] * ye_orig
        ye_new = Ycoeff[1] * xe_orig + Ycoeff[2] * ye_orig

        if vel:
            vx_new = Xcoeff[1] * vx_orig + Xcoeff[2] * vy_orig
            vy_new = Ycoeff[1] * vx_orig + Ycoeff[2] * vy_orig
            vxe_new = Xcoeff[1] * vxe_orig + Xcoeff[2] * vye_orig
            vye_new = Ycoeff[1] * vxe_orig + Ycoeff[2] * vye_orig

    elif len(Xcoeff) == 6:
        x_new = Xcoeff[0] + Xcoeff[1]*x_orig + Xcoeff[2]*y_orig + Xcoeff[3]*x_orig**2. + \
          Xcoeff[5]*y_orig**2. + Xcoeff[4]*x_orig*y_orig
          
        y_new = Ycoeff[0] + Ycoeff[1]*x_orig + Ycoeff[2]*y_orig + Ycoeff[3]*x_orig**2. + \
          Ycoeff[5]*y_orig**2. + Ycoeff[4]*x_orig*y_orig
          
        xe_new = Xcoeff[1]*xe_orig + Xcoeff[2]*ye_orig + Xcoeff[3]*xe_orig**2. + \
          Xcoeff[5]*ye_orig**2. + Xcoeff[4]*xe_orig*ye_orig
          
        ye_new = Ycoeff[1]*xe_orig + Ycoeff[2]*ye_orig + Ycoeff[3]*xe_orig**2. + \
          Ycoeff[5]*ye_orig**2. + Ycoeff[4]*xe_orig*ye_orig

        if vel:
            vx_new = Xcoeff[1]*vx_orig + Xcoeff[2]*vy_orig + 2.*Xcoeff[3]*x_orig*vx_orig + \
                2.*Xcoeff[5]*y_orig*vy_orig + Xcoeff[4]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vy_new = Ycoeff[1]*vx_orig + Ycoeff[2]*vy_orig + 2.*Ycoeff[3]*x_orig*vx_orig + \
                2.*Ycoeff[5]*y_orig*vy_orig + Ycoeff[4]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vxe_new = Xcoeff[1]*vxe_orig + Xcoeff[2]*vye_orig + 2.*Xcoeff[3]*xe_orig*vxe_orig + \
                2.*Xcoeff[5]*ye_orig*vye_orig + Xcoeff[4]*(xe_orig*vye_orig + vxe_orig*ye_orig)
          
            vye_new = Ycoeff[1]*vxe_orig + Ycoeff[2]*vye_orig + 2.*Ycoeff[3]*xe_orig*vxe_orig + \
                2.*Ycoeff[5]*ye_orig*vye_orig + Ycoeff[4]*(xe_orig*vye_orig + vxe_orig*ye_orig)
        
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


def transform_by_object(starlist, transform):
    """
    Apply transformation to starlist. Returns astropy table with
    transformed positions/position errors, velocities and velocity errors 
    if they are present in starlist
    
    Parameters:
    ----------
    starlist: astropy table
         Starlist we want to apply the transformation too. Must already
         have standard column headers

    transform: transformation object
        File with the transformation coefficients. Assumed to be output of
        write_transform

    Output:
    ------
    starlist astropy table with transformed x,y,xe,ye,vx,vy,vxe,vye
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
    
    # Read transformation: Extract X, Y coefficients from transform
    if transform.__class__.__name__ == 'four_paramNW':
        Xcoeff = transform.px
        Ycoeff = transform.py
    elif transform.__class__.__name__ == 'PolyTransform':
        Xcoeff = transform.px.parameters
        Ycoeff = transform.py.parameters
    else:
        print '{0} not yet supported!'.format(transType)
        return
        
    # How the transformation is applied depends on the type of transform.
    # This can be determined by the length of Xcoeff, Ycoeff
    if len(Xcoeff) == 3:
        x_new = Xcoeff[0] + Xcoeff[1] * x_orig + Xcoeff[2] * y_orig
        y_new = Ycoeff[0] + Ycoeff[1] * x_orig + Ycoeff[2] * y_orig
        xe_new = np.sqrt( (Xcoeff[1] * xe_orig)**2 + (Xcoeff[2] * ye_orig)**2 )
        ye_new = np.sqrt( (Ycoeff[1] * xe_orig)**2 + (Ycoeff[2] * ye_orig)**2 )

        if vel:
            vx_new = Xcoeff[1] * vx_orig + Xcoeff[2] * vy_orig
            vy_new = Ycoeff[1] * vx_orig + Ycoeff[2] * vy_orig
            vxe_new = np.sqrt( (Xcoeff[1] * vxe_orig)**2 + (Xcoeff[2] * vye_orig)**2 )
            vye_new = np.sqrt( (Ycoeff[1] * vxe_orig)**2 + (Ycoeff[2] * vye_orig)**2 )

    elif len(Xcoeff) == 6:
        x_new = Xcoeff[0] + Xcoeff[1]*x_orig + Xcoeff[2]*x_orig**2 + Xcoeff[3]*y_orig + \
                Xcoeff[4]*y_orig**2. + Xcoeff[5]*x_orig*y_orig
          
        y_new = Ycoeff[0] + Ycoeff[1]*x_orig + Ycoeff[2]*x_orig**2 + Ycoeff[3]*y_orig + \
                Ycoeff[4]*y_orig**2. + Ycoeff[5]*x_orig*y_orig
          
        xe_new = np.sqrt( (Xcoeff[1] + 2*Xcoeff[2]*x_orig + Xcoeff[5]*y_orig)**2 * xe_orig**2 + \
                          (Xcoeff[3] + 2*Xcoeff[4]*y_orig + Xcoeff[5]*x_orig)**2 * ye_orig**2 )
          
        ye_new = np.sqrt( (Ycoeff[1] + 2*Ycoeff[2]*x_orig + Ycoeff[5]*y_orig)**2 * xe_orig**2 + \
                          (Ycoeff[3] + 2*Ycoeff[4]*y_orig + Ycoeff[5]*x_orig)**2 * ye_orig**2 )

        if vel:
            vx_new = Xcoeff[1]*vx_orig + 2*Xcoeff[2]*x_orig*vx_orig + Xcoeff[3]*vy_orig + \
                    2.*Xcoeff[4]*y_orig*vy_orig + Xcoeff[5]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vy_new = Ycoeff[1]*vx_orig + 2*Ycoeff[2]*x_orig*vx_orig + Ycoeff[3]*vy_orig + \
                    2.*Ycoeff[4]*y_orig*vy_orig + Ycoeff[5]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vxe_new = np.sqrt( (Xcoeff[1] + 2*Xcoeff[2]*x_orig + Xcoeff[5]*y_orig)**2 * vxe_orig**2 + \
                               (Xcoeff[3] + 2*Xcoeff[4]*y_orig + Xcoeff[5]*x_orig)**2 * vye_orig**2 + \
                               (2*Xcoeff[2]*vx_orig + Xcoeff[5]*vy_orig)**2 * xe_orig**2 + \
                               (2*Xcoeff[4]*vy_orig + Xcoeff[5]*vx_orig)**2 * ye_orig**2 )
                                
            vye_new = np.sqrt( (Ycoeff[1] + 2*Ycoeff[2]*x_orig + Ycoeff[5]*y_orig)**2 * vxe_orig**2 + \
                               (Ycoeff[3] + 2*Ycoeff[4]*y_orig + Ycoeff[5]*x_orig)**2 * vye_orig**2 + \
                               (2*Ycoeff[2]*vx_orig + Ycoeff[5]*vy_orig)**2 * xe_orig**2 + \
                               (2*Ycoeff[4]*vy_orig + Ycoeff[5]*vx_orig)**2 * ye_orig**2 )
        
    # update transformed coords to astropy table

    starlist['x'] = x_new
    starlist['y'] = y_new
    starlist['xe'] = xe_new
    starlist['ye'] = ye_new
    
    if vel:
        starlist['vx'] = vx_new
        starlist['vy'] = vy_new
        starlist['vxe'] = vxe_new
        starlist['vye'] = vye_new
        
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
    # reference frame increase to the west. Do this for velocity
    # as well
    t_label['x'] = t_label['x'] * (-1.0)
    t_label['vx'] = t_label['vx'] * (-1.0)
    
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

