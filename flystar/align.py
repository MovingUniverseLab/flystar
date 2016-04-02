import numpy as np
import copy
import match
import transforms
from astropy.table import Table, Column
import datetime
import copy
import os
import pdb


def initial_align(table1, table2, briteN=100, transformModel=transforms.four_paramNW, order=1):
    """
    Calculates an initial (unweighted) transformation from table1 starlist into
    table2 starlist (i.e., table2 is the reference starlist). Matching is done using
    a blind triangle-matching algorithm of the brightest briteN stars in both starlists.
    Transformation is done using the transformModel in the input parameter.

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
    print 'Begin initial match'

    # number of required matches of the input catalog to the total reference
    req_match = 5

    N, x1m, y1m, m1m, x2m, y2m, m2m = match.miracle_match_briteN(x1, y1, m1, x2, y2, m2, briteN)
    assert len(x1m) > req_match#, 'Failed to find at least '+str(req_match+' matches, giving up'
    print '{0} stars matched between starlist1 and starlist2'.format(N)

    # Calculate transformation based on matches
    t = transformModel(x1m, y1m ,x2m, y2m, order=order, weights=None)

    print 'End initial match \n'
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
    -idx1: indicies of matched stars from table1
    -idx2: indicies of matched stars from tabel2
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

    print '{0} of {1} stars matched'.format(len(idx1), len(x1t))

    return idx1, idx2


def find_transform(table1, table1_trans, table2, transModel=transforms.four_paramNW, order=1, 
                weights=None):
    """
    Given a matched starlist, derive a new transform. This transformation is
    calculated for starlist 1 into starlist 2

    Parameters:
    -----------
    table1: astropy table
        Table which we have calculated the transformation for, trimmed to only
        stars which match with table2. Original coords, not transformed into
        reference frame.

    table1_trans: astropy table
        Table which we calculated the transformation fo, trimmed to only
        stars which match with table2. Contains transformed coords. Only
        used when calculating weights.

    table2: astropy table
        Table with the reference starlist. Trimmed to only stars which
        match table1.

    trans: transformation object
        Transformation used to transform table1 coords in transform_and_match
        in order to do the star matching. 

    transModel: transformation class (default: transform.four_paramNW)
        Desired transform to apply to matched stars, e.g. four_paramNW or PolyTransform.
        If PolyTransform is selected, order defines the order of polynomial used

    order: int (default=1)
        Order of polynomial to use in the transformation. Only active if
        PolyTransform is selected

    weights: string (default=None)
        if weights=='both', we use both position error and velocity error in transformed
        starlist and reference starlist as uncertanties. And weights is the reciprocal 
            of this uncertanty.
        if weights=='starlist', we only use postion error and velocity error in transformed
        starlist as uncertainty.
        if weights=='reference', we only use position error in reference starlist as uncertainty.
        if weights==None, we don't use weights.

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
    x1 = table1['x']
    y1 = table1['y']
    x2 = table2['x']
    y2 = table2['y']

    # calculate weights from *transformed* coords. This is where we use the
    # transformation object
    x1e = table1_trans['xe']
    y1e = table1_trans['ye']

    x2e = table2['xe']
    y2e = table2['ye']

    # Calculate weights as to user specification
    if weights=='both':
        weight = 1/np.sqrt( x1e**2 + y1e**2 + x2e**2 + y2e**2)
    elif weights=='starlist':
        weight = 1/np.sqrt( x1e**2 + y1e**2 )
    elif weights=='reference':
        weight = 1/np.sqrt( x2e**2 +  y2e**2)
    else:
        weight = None

    # Calculate transform based on the matched stars    
    t = transModel(x1, y1, x2, y2, order=order, weights=weight)

    N_trans = len(x1)
    print '{0} stars used in transform\n'.format(N_trans)

    # Ret3urn transformation object and number of stars used in transform
    return t, N_trans


def write_transform(transform, starlist, reference, N_trans, deltaMag=0, restrict=False, weights=None,
                    outFile='outTrans.txt'):
    """
    Given a transformation object, write out the coefficients in a java align
    readable format. Outfile name is specified by user.

    Coefficients are output in file in the following way:
    x' = a0 + a1*x + a2*y + a3*x**2. + a4*x*y  + a5*y**2. + ...
    y' = b0 + b1*x + b2*y + b3*x**2. + b4*x*y  + b5*y**2. + ...

    Parameters:
    ----------
    transform: transformation object
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

    weights: string (default=None)
        if weights=='both', we use both position error and velocity error in transformed
        starlist and reference starlist as uncertanties. And weights is the reciprocal 
            of this uncertanty.
        if weights=='starlist', we only use postion error and velocity error in transformed
        starlist as uncertainty.
        if weights=='reference', we only use position error in reference starlist as uncertainty
        if weights==None, we don't use weights.

    outFile: string (default: 'outTrans.txt')
        Name of output text file
         
    Output:
    ------
    txt file with the file name outFile
    """
    # Extract info about transformation
    trans_name = transform.__class__.__name__
    trans_order = transform.order
      
    # Extract X, Y coefficients from transform
    if trans_name == 'four_paramNW':
        Xcoeff = transform.px
        Ycoeff = transform.py
    elif trans_name == 'PolyTransform':
        Xcoeff = transform.px.parameters
        Ycoeff = transform.py.parameters
    else:
        print '{0} not yet supported!'.format(transType)
        return
        
    # Write output
    _out = open(outFile, 'w')
    
    # Write the header. DO NOT CHANGE, HARDCODED IN JAVA ALIGN
    _out.write('## Date: {0}\n'.format(datetime.date.today()) )
    _out.write('## File: {0}, Reference: {1}\n'.format(starlist, reference) )
    _out.write('## Directory: {0}\n'.format(os.getcwd()) )
    _out.write('## Transform Class: {0}\n'.format(transform.__class__.__name__))
    _out.write('## Order: {0}\n'.format(transform.order))
    _out.write('## Restrict: {0}\n'.format(restrict))
    _out.write('## Weights: {0}\n'.format(weights))
    _out.write('## N_coeff: {0}\n'.format(len(Xcoeff)))
    _out.write('## N_trans: {0}\n'.format(N_trans))
    _out.write('## Delta Mag: {0}\n'.format(deltaMag))
    _out.write('{0:16s} {1:16s}\n'.format('# Xcoeff', 'Ycoeff'))
    
    # Write the coefficients such that the orders are together as defined in
    # documentation. This is a pain because PolyTransform output is weird.
    # (see astropy Polynomial2D documentation)
    if (trans_name == 'four_paramNW'):
        for i in range(len(Xcoeff)):
            _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[i], Ycoeff[i]) )
    elif (trans_name == 'PolyTransform'):
        # CODE TO GET INDICIES 
        N = trans_order - 1 
        idx_list = list()
        
        # when trans_order=1, N=0
        idx_list.append(0)
        idx_list.append(1)
        idx_list.append(N+2)
        
        if trans_order >= 2:
            for k in range(2, N+2):
                idx_list.append(k)
                for j in range(1, k):
                    i = k-j
                    idx_list.append(int(2*N +2 +j + (2*N+2-i)*(i-1)/2.))
                idx_list.append(N+1+k)

        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[0], Ycoeff[0]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[1], Ycoeff[1]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[3], Ycoeff[3]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[2], Ycoeff[2]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[5], Ycoeff[5]) )
        #_out.write('{0:16.6e}  {1:16.6e}'.format(Xcoeff[4], Ycoeff[4]) )

        for i in idx_list:
            _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[i], Ycoeff[i]) )


    _out.close()
    
    return


def transform_from_file(starlist, transFile):
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
    Copy of starlist astropy table with transformed coordinates.  
    """
    # Make a copy of starlist. This is what we will eventually modify with
    # the transformed coordinates
    starlist_f = copy.deepcopy(starlist)
    
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

    #-----------------------------------------------#
    # General equation for applying the transform
    #-----------------------------------------------#
    # First determine the order based on the number of terms
    # Comes from Nterms = (N+1)*(N+2) / 2.
    order = (np.sqrt(1 + 8*len(Xcoeff)) - 3) / 2.

    if order%1 != 0:
        print 'Incorrect number of coefficients for polynomial'
        print 'Stopping'
        return
    order = int(order)

    # Position loop
    idx = 0 # coeff index
    x_new = 0.0
    y_new = 0.0
    for i in range(order+1):
        for j in range(i+1):
            x_new += Xcoeff[idx] * x_orig**(i-j) * y_orig**j
            y_new += Ycoeff[idx] * x_orig**(i-j) * y_orig**j

            idx += 1

    # Position error loop
    idx = 0
    xe_new_sq = 0.0
    ye_new_sq = 0.0
    # First loop: dx'/dx
    for i in range(order+1):
        for j in range(i+1):
            xe_new_sq += (Xcoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j)**2. * xe_orig**2.
            ye_new_sq += (Ycoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j)**2. * xe_orig**2.

            idx += 1
                
    # Second loop: dy'/dy
    idx = 0
    for i in range(order+1):
        for j in range(i+1):
            xe_new_sq += (Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1))**2. * ye_orig**2.
            ye_new_sq += (Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1))**2. * ye_orig**2.    

            idx += 1
    # Take square root for xe/ye_new
    xe_new = np.sqrt(xe_new_sq)
    ye_new = np.sqrt(ye_new_sq)

    if vel:
        # Velocity loop
        idx = 0
        vx_new = 0.0
        vy_new = 0.0
        # First loop: dx'/dx
        for i in range(order+1):
            for j in range(i+1):
                vx_new += Xcoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j * vx_orig
                vy_new += Ycoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j * vx_orig

                idx += 1
        # Second loop: dy'/dy
        idx = 0
        for i in range(order+1):
            for j in range(i+1):
                vx_new += Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1) * ye_orig
                vy_new += Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1) * ye_orig         

                idx += 1
            
        # Velocity error loop
        idx = 0
        vxe_new_sq = 0.0
        vye_new_sq = 0.0
        # First loop: dvx' / dx
        for i in range(order+1):
            for j in range(i+1):
                vxe_new_sq += (Xcoeff[idx] * (i-j) * (i-j-1) * x_orig**(i-j-2) * y_orig**j * vx_orig)**2. * xe_orig**2. + \
                  (Xcoeff[idx] * (j) * (i-j) * x_orig**(i-j-1) * y_orig**(j-1) * vy_orig)**2. * xe_orig**2.
                vye_new_sq += (Ycoeff[idx] * (i-j) * (i-j-1) * x_orig**(i-j-2) * y_orig**j * vx_orig)**2. * xe_orig**2. + \
                  (Xcoeff[idx] * (j) * (i-j) * x_orig**(i-j-1) * y_orig**(j-1) * vy_orig)**2. * xe_orig**2.                  

                idx += 1
        # Second loop: dvx' / dy
        idx = 0
        for i in range(order+1):
            for j in range(i+1):
                vxe_new_sq += (Xcoeff[idx] * (i-j) * (j-1) * x_orig**(i-j-1) * y_orig**(j-1) * vx_orig)**2. * ye_orig**2. + \
                  (Xcoeff[idx] * (j) * (j-1) * x_orig**(i-j-1) * y_orig**(j-2) * vy_orig)**2. * ye_orig**2.
                vye_new_sq += (Ycoeff[idx] * (i-j) * (j-1) * x_orig**(i-j-1) * y_orig**(j-1) * vx_orig)**2. * ye_orig**2. + \
                  (Xcoeff[idx] * (j) * (j-1) * x_orig**(i-j-1) * y_orig**(j-2) * vy_orig)**2. * ye_orig**2.

                idx += 1
        # Third loop: dvx' / dvx
        idx = 0
        for i in range(order+1):
            for j in range(i+1):
                vxe_new_sq += (Xcoeff[idx] * (i-j) * x_orig**(i-j-1) * y_orig**j)**2. * vxe_orig**2.
                vye_new_sq += (Ycoeff[idx] * (i-j) * x_orig**(i-j-1) * y_orig**j)**2. * vxe_orig**2.

                idx += 1
        # Fourth loop: dvx' / dvy
        idx = 0
        for i in range(order+1):
            for j in range(i+1):
                vxe_new_sq += (Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1))**2. * vye_orig**2.        
                vye_new_sq += (Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1))**2. * vye_orig**2. 


        vxe_new = np.sqrt(vxe_new_sq)
        vye_new = np.sqrt(vye_new_sq)

    """
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
        x_new = Xcoeff[0] + Xcoeff[1]*x_orig + Xcoeff[3]*x_orig**2 + Xcoeff[2]*y_orig + \
                Xcoeff[5]*y_orig**2. + Xcoeff[4]*x_orig*y_orig
          
        y_new = Ycoeff[0] + Ycoeff[1]*x_orig + Ycoeff[3]*x_orig**2 + Ycoeff[2]*y_orig + \
                Ycoeff[5]*y_orig**2. + Ycoeff[4]*x_orig*y_orig
          
        xe_new = np.sqrt( (Xcoeff[1] + 2*Xcoeff[3]*x_orig + Xcoeff[4]*y_orig)**2 * xe_orig**2 + \
                          (Xcoeff[2] + 2*Xcoeff[5]*y_orig + Xcoeff[4]*x_orig)**2 * ye_orig**2 )
          
        ye_new = np.sqrt( (Ycoeff[1] + 2*Ycoeff[3]*x_orig + Ycoeff[4]*y_orig)**2 * xe_orig**2 + \
                          (Ycoeff[2] + 2*Ycoeff[5]*y_orig + Ycoeff[4]*x_orig)**2 * ye_orig**2 )

        if vel:
            vx_new = Xcoeff[1]*vx_orig + 2*Xcoeff[3]*x_orig*vx_orig + Xcoeff[2]*vy_orig + \
                    2.*Xcoeff[5]*y_orig*vy_orig + Xcoeff[4]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vy_new = Ycoeff[1]*vx_orig + 2*Ycoeff[3]*x_orig*vx_orig + Ycoeff[2]*vy_orig + \
                    2.*Ycoeff[5]*y_orig*vy_orig + Ycoeff[4]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vxe_new = np.sqrt( (Xcoeff[1] + 2*Xcoeff[3]*x_orig + Xcoeff[4]*y_orig)**2 * vxe_orig**2 + \
                               (Xcoeff[2] + 2*Xcoeff[5]*y_orig + Xcoeff[4]*x_orig)**2 * vye_orig**2 + \
                               (2*Xcoeff[3]*vx_orig + Xcoeff[4]*vy_orig)**2 * xe_orig**2 + \
                               (2*Xcoeff[5]*vy_orig + Xcoeff[4]*vx_orig)**2 * ye_orig**2 )
                                
            vye_new = np.sqrt( (Ycoeff[1] + 2*Ycoeff[3]*x_orig + Ycoeff[4]*y_orig)**2 * vxe_orig**2 + \
                               (Ycoeff[2] + 2*Ycoeff[5]*y_orig + Ycoeff[4]*x_orig)**2 * vye_orig**2 + \
                               (2*Ycoeff[3]*vx_orig + Ycoeff[4]*vy_orig)**2 * xe_orig**2 + \
                               (2*Ycoeff[5]*vy_orig + Ycoeff[4]*vx_orig)**2 * ye_orig**2 )
    """
    #Update transformed coords to copy of astropy table
    starlist_f['x'] = x_new
    starlist_f['y'] = y_new
    starlist_f['xe'] = xe_new
    starlist_f['ye'] = ye_new
    
    if vel:
        #starlist_f['x0'] = x0_new
        #starlist_f['y0'] = y0_new
        #starlist_f['x0e'] = x0e_new
        #starlist_f['y0e'] = y0e_new
        starlist_f['vx'] = vx_new
        starlist_f['vy'] = vy_new
        starlist_f['vxe'] = vxe_new
        starlist_f['vye'] = vye_new
    
    #xCol = Column(x_new, name='x_trans')
    #yCol = Column(y_new, name='y_trans')
    #xeCol = Column(xe_new, name='xe_trans')
    #yeCol = Column(ye_new, name='ye_trans')
    
    #starlist.add_column(xCol)
    #starlist.add_column(yCol)
    #starlist.add_column(xeCol)
    #starlist.add_column(yeCol)    

    #if vel:
    #    vxCol = Column(vx_new, name='vx_trans')
    #    vyCol = Column(vy_new, name='vy_trans')
    #    vxeCol = Column(vxe_new, name='vxe_trans')
    #    vyeCol = Column(vye_new, name='vye_trans')

    #    starlist.add_column(vxCol)
    #    starlist.add_column(vyCol)
    #    starlist.add_column(vxeCol)
    #    starlist.add_column(vyeCol) 

    return starlist_f


def transform_from_object(starlist, transform):
    """
    Apply transformation to starlist. Returns astropy table with
    transformed positions/position errors, velocities and velocity errors 
    if they are present in starlist
    
    Parameters:
    ----------
    starlist: astropy table
         Starlist we want to apply the transformation too. Must already
         have standard column headers
         x0, y0, x0e, y0e, vx, vy, vxe, vye, x, y, xe, ye

    transform: transformation object

    Output:
    ------
    Copy of starlist astropy table with transformed x0, y0, x0e, y0e,
    vx, vy, vxe, vye, x, y, xe, ye

    """
    # Make a copy of starlist. This is what we will eventually modify with
    # the transformed coordinates
    starlist_f = copy.deepcopy(starlist)

    # Check to see if velocities are present in starlist. If so, we will
    # need to transform these as well as positions
    vel = False
    keys = starlist.keys()
    if 'vx' in keys:
        vel = True 
    
    # Extract needed information from starlist
    x = starlist['x']
    y = starlist['y']
    xe = starlist['xe']
    ye = starlist['ye']

    if vel:
        x0 = starlist['x0']
        y0 = starlist['y0']
        x0e = starlist['x0e']
        y0e = starlist['y0e']
        vx = starlist['vx']
        vy = starlist['vy']
        vxe = starlist['vxe']
        vye = starlist['vye']
    
    # Read transformation: Extract X, Y coefficients from transform
    if transform.__class__.__name__ == 'four_paramNW':
        Xcoeff = transform.px
        Ycoeff = transform.py
        order = 1
    elif transform.__class__.__name__ == 'PolyTransform':
        Xcoeff = transform.px.parameters
        Ycoeff = transform.py.parameters
        order = transform.order
    else:
        print '{0} not yet supported!'.format(transType)
        return
        
    # How the transformation is applied depends on the type of transform.
    # This can be determined by the length of Xcoeff, Ycoeff
    #"""
    N = order - 1

    # x_new & y_new in (x,y)
    x_new = 0
    for i in range(0, N+2):
        x_new += Xcoeff[i] * (x**i)
    for j in range(1, N+2):
        x_new += Xcoeff[N+1+j] * (y**j)
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            x_new += Xcoeff[sub] * (x**i) * (y**j)

    y_new = 0
    for i in range(0, N+2):
        y_new += Ycoeff[i] * (x**i)
    for j in range(1, N+2):
        y_new += Ycoeff[N+1+j] * (y**j)
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            y_new += Ycoeff[sub] * (x**i) * (y**j)

    # xe_new & ye_new in (x,y,xe,ye)
    xe_new = 0
    temp1 = 0
    temp2 = 0
    for i in range(1, N+2):
        temp1 += i * Xcoeff[i] * (x**(i-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp1 += i * Xcoeff[sub] * (x**(i-1)) * (y**j)
    for j in range(1, N+2):
        temp2 += j * Xcoeff[N+1+j] * (y**(j-1))
    for i in range(1, N+1):
        for j in range(N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp2 += j * Xcoeff[sub] * (x**i) * (y**(j-1))
    xe_new = np.sqrt((temp1*xe)**2 + (temp2*ye)**2)

    ye_new = 0
    temp1 = 0
    temp2 = 0
    for i in range(1, N+2):
        temp1 += i * Ycoeff[i] * (x**(i-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp1 += i * Ycoeff[sub] * (x**(i-1)) * (y**j)
    for j in range(1, N+2):
        temp2 += j * Ycoeff[N+1+j] * (y**(j-1))
    for i in range(1, N+1):
        for j in range(N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp2 += j * Ycoeff[sub] * (x**i) * (y**(j-1))
    ye_new = np.sqrt((temp1*xe)**2 + (temp2*ye)**2)


    if vel:
        # x0_new & y0_new in (x0, y0)
        x0_new = 0
        for i in range(0, N+2):
            x0_new += Xcoeff[i] * (x0**i)
        for j in range(1, N+2):
            x0_new += Xcoeff[N+1+j] * (y0**j)
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                x0_new += Xcoeff[sub] * (x0**i) * (y0**j)

        y0_new = 0
        for i in range(0, N+2):
            y0_new += Ycoeff[i] * (x0**i)
        for j in range(1, N+2):
            y0_new += Ycoeff[N+1+j] * (y0**j)
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                y0_new += Ycoeff[sub] * (x0**i) * (y0**j)

        # x0e_new & y0e_new in (x0, y0, x0e, y0e)
        x0e_new = 0
        temp1 = 0
        temp2 = 0
        for i in range(1, N+2):
            temp1 += i * Xcoeff[i] * (x0**(i-1))
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp1 += i * Xcoeff[sub] * (x0**(i-1)) * (y0**j)
        for j in range(1, N+2):
            temp2 += j * Xcoeff[N+1+j] * (y0**(j-1))
        for i in range(1, N+1):
            for j in range(N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp2 += j * Xcoeff[sub] * (x0**i) * (y0**(j-1))
        x0e_new = np.sqrt((temp1*x0e)**2 + (temp2*y0e)**2)

        y0e_new = 0
        temp1 = 0
        temp2 = 0
        for i in range(1, N+2):
            temp1 += i * Ycoeff[i] * (x0**(i-1))
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp1 += i * Ycoeff[sub] * (x0**(i-1)) * (y0**j)
        for j in range(1, N+2):
            temp2 += j * Ycoeff[N+1+j] * (y0**(j-1))
        for i in range(1, N+1):
            for j in range(N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp2 += j * Ycoeff[sub] * (x0**i) * (y0**(j-1))
        y0e_new = np.sqrt((temp1*x0e)**2 + (temp2*y0e)**2)

        # vx_new & vy_new in (x0, y0, vx, vy)
        vx_new = 0
        for i in range(1, N+2):
            vx_new += i * Xcoeff[i] * (x0**(i-1)) * vx
        for j in range(1, N+2):
            vx_new += j * Xcoeff[N+1+j] * (y0**(j-1)) * vy
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                vx_new += i * Xcoeff[sub] * (x0**(i-1)) * (y0**j) * vx
                vx_new += j * Xcoeff[sub] * (x0**i) * (y0**(j-1)) * vy

        vy_new = 0
        for i in range(1, N+2):
            vy_new += i * Ycoeff[i] * (x0**(i-1)) * vx
        for j in range(1, N+2):
            vy_new += j * Ycoeff[N+1+j] * (y0**(j-1)) * vy
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                vy_new += i * Ycoeff[sub] * (x0**(i-1)) * (y0**j) * vx
                vy_new += j * Ycoeff[sub] * (x0**i) * (y0**(j-1)) * vy

        # vxe_new & vye_new in (x0, y0, x0e, y0e, vx, vy, vxe, vye)
        vxe_new = 0
        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0
        for i in range(2, N+2):
            temp1 += i * (i-1) * Xcoeff[i] * (x0**(i-2)) * vx
        for i in range(2, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp1 += i * (i-1) * Xcoeff[sub] * (x0**(i-2)) * (y0**j) * vx
        for i in range(1,N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp1 += i * j * Xcoeff[sub] * (x0**(i-1)) * (y0**(j-1)) * vy

        for j in range(2, N+2):
            temp2 += j * (j-1) * Xcoeff[N+1+j] * (y0**(j-2)) * vy
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp2 += i * j * Xcoeff[sub] * (x0**(i-1)) * (y0**(j-1)) * vx
        for i in range(1, N+1):
            for j in range(2, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp2 += j * (j-1) * Xcoeff[sub] * (x0**i) * (y0**(j-2)) * vy

        for i in range(1, N+2):
            temp3 += i * Xcoeff[i] * (x0**(i-1))
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp3 += i * Xcoeff[sub] * (x0**(i-1)) * (y0**j) 

        for j in range(1, N+2):
            temp4 += j * Xcoeff[N+1+j] * (y0**(j-1))
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp4 += j * Xcoeff[sub] * (x0**i) * (y0**(j-1))
        vxe_new = np.sqrt((temp1*x0e)**2 + (temp2*y0e)**2 + (temp3*vxe)**2 + (temp4*vye)**2)


        vye_new = 0
        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0
        for i in range(2, N+2):
            temp1 += i * (i-1) * Ycoeff[i] * (x0**(i-2)) * vx
        for i in range(2, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp1 += i * (i-1) * Ycoeff[sub] * (x0**(i-2)) * (y0**j) * vx
        for i in range(1,N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp1 += j * i * Ycoeff[sub] * (x0**(i-1)) * (y0**(j-1)) * vy

        for j in range(2, N+2):
            temp2 += j * (j-1) * Ycoeff[N+1+j] * (y0**(j-2)) * vy
        for i in range(1, N+1):
            for j in range(2, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp2 += j * (j-1) * Ycoeff[sub] * (x0**i) * (y0**(j-2)) * vy
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp2 += i * j * Ycoeff[sub] * (x0**(i-1)) * (y0**(j-1)) * vx

        for i in range(1, N+2):
            temp3 += i * Ycoeff[i] * (x0**(i-1))
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp3 += i * Ycoeff[sub] * (x0**(i-1)) * (y0**j) 

        for j in range(1, N+2):
            temp4 += j * Ycoeff[N+1+j] * (y0**(j-1))
        for i in range(1, N+1):
            for j in range(1, N+2-i):
                sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
                temp4 += j * Ycoeff[sub] * (x0**i) * (y0**(j-1))
        vye_new = np.sqrt((temp1*x0e)**2 + (temp2*y0e)**2 + (temp3*vxe)**2 + (temp4*vye)**2)



    """

    if len(Xcoeff) == 3:
        x_new = Xcoeff[0] + Xcoeff[1] * x + Xcoeff[2] * y
        y_new = Ycoeff[0] + Ycoeff[1] * x + Ycoeff[2] * y
        xe_new = np.sqrt( (Xcoeff[1] * xe)**2 + (Xcoeff[2] * ye)**2 )
        ye_new = np.sqrt( (Ycoeff[1] * xe)**2 + (Ycoeff[2] * ye)**2 )

        if vel:
            vx_new = Xcoeff[1] * vx + Xcoeff[2] * vy
            vy_new = Ycoeff[1] * vx + Ycoeff[2] * vy
            vxe_new = np.sqrt( (Xcoeff[1] * vxe)**2 + (Xcoeff[2] * vye)**2 )
            vye_new = np.sqrt( (Ycoeff[1] * vxe)**2 + (Ycoeff[2] * vye)**2 )

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
    """

    # update transformed coords to copy of astropy table
    starlist_f['x'] = x_new
    starlist_f['y'] = y_new
    starlist_f['xe'] = xe_new
    starlist_f['ye'] = ye_new
    
    if vel:
        starlist_f['x0'] = x0_new
        starlist_f['y0'] = y0_new
        starlist_f['x0e'] = x0e_new
        starlist_f['y0e'] = y0e_new
        starlist_f['vx'] = vx_new
        starlist_f['vy'] = vy_new
        starlist_f['vxe'] = vxe_new
        starlist_f['vye'] = vye_new
        
    return starlist_f





