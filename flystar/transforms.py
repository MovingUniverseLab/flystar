from astropy.modeling import models, fitting
import numpy as np
from scipy.interpolate import LSQBivariateSpline as spline
from scipy import stats
from astropy.table import Table
import collections
import re
import pdb

class Transform2D(object):
    '''
    Base class for transformations. It contains the properties common to all
    transformation objects.
    '''

    """def __init__(self, x, y, xref, yref):
        self.x = x
        self.y = y
        self.xref = xref
        self.yref = yref
        #self.px = None
        #self.px = None"""

    def __init__(self):
        '''
        Initialization of base class Transform2D.
        '''
        return
    
    @classmethod
    def from_file(cls, filename):
        """
        Initialize a Transform2D object (or a specified sub-class) from a file
        containing the transformation coefficients. 
        """
        # Read the header of the coefficients file and get the
        # transformation type and order. 
        transf_file = open(filename, "r")
        for line in transf_file:
            if "## Transform Class:"  in line:
                model = line.split(": ", 1)[1].rstrip("\n")
            if "## Order:" in line:
                order = int(re.search(r'\d+$', line).group(0))
                break
        transf_file.close()

        # Read in the coefficients.
        transf_tab = Table.read(filename, format='ascii.commented_header',
                                header_start=-1)
        px = transf_tab.as_array()['Xcoeff']
        py = transf_tab.as_array()['Ycoeff']

        if model == "PolyTransform":
            transf = PolyTransform(order, px, py)
    
        return transf

    def evaluate(self, x, y):
        # method should be defined in the subclasses
        pass

    def evaluate_starlist(self, star_list):
        new_list = copy.deepcopy(star_list)

        # Cases we need to consider:
        # 1. x and y, no errors, no velocities
        # 2. x, y, x errors, y errors, no velocities
        # 3. x, y, vx, vy, no xy or vel errors
        # 4. x, y, vx, vy, xe, ye, vxe, vye
        #
        # We will assume that all input is symmetric/2D (i.e. only check on x)

        # Positions
        vals = self.evaluate(star_list['x'], star_list['y'])
        
        new_list['x'] = vals[0]
        new_list['y'] = vals[1]

        # Positional errors (if they exist)
        if 'xe' in new_list.colnames:
            vals = self.evaluate_error(star_list['x'], star_list['y'],
                                       star_list['xe'], star_list['ye'])
            new_list['xe'] = vals[0]
            new_list['ye'] = vals[1]

        # Velocities (if they exist)
        if 'vx' in new_list.colnames:
            vals = self.evaluate_vel(star_list['x'], star_list['y'],
                                     star_list['vx'], star_list['vy'])
            new_list['vx'] = vals[0]
            new_list['vy'] = vals[1]

            # Velocity errors (if they exist)
            if 'vxe' in new_list.colnames:
                vals = self.evaluate_vel_error(star_list['x'], star_list['y'],
                                               star_list['vx'], star_list['vy'],
                                               star_list['xe'], star_list['ye'],
                                               star_list['vxe'], star_list['vye'])
                new_list['vxe'] = vals[0]
                new_list['vye'] = vals[1]
                
        return new_list
    
    def evaluate_MC_errors(self, x, x_err, y, y_err, nsim=500):
        '''
        Run a MC simulation to figure out what the uncertainty from the
        transformation should be.

        Parameters:
        -----------
        x, x_err, y, y_err - x,y position and their uncertainties

        Keywords:
        ---------
        nsim - number of simulations to run (default: 1000)

        Outputs:
        --------
        x_trans, x_trans_err, y_trans, y_trans_err
        '''

        # evaluate the transformation for the input arrays
        x_trans,y_trans = self.evaluate(x,y)
        
        # now run MC simulation to determine the uncertainties
        
        x_trans_all_stack = np.zeros((len(x),nsim))
        y_trans_all_stack = np.zeros((len(y),nsim))

        x_all_len = len(x)
        y_all_len = len(y)

        for i in np.arange(nsim):
            xsample = np.random.normal(loc=0.0,scale=1.0,size=len(x))
            ysample = np.random.normal(loc=0.0,scale=1.0,size=len(y))

            xsample_ref = np.random.normal(loc=0.0,scale=1.0,size=len(x))
            ysample_ref = np.random.normal(loc=0.0,scale=1.0,size=len(y))

            x_temp = x + xsample*x_err
            y_temp = y + ysample*y_err

            x_trans_temp,y_trans_temp = self.evaluate(x_temp,y_temp)
            x_trans_all_stack[:,i] = x_trans_temp
            y_trans_all_stack[:,i] = y_trans_temp

        x_trans_all_err = np.std(x_trans_all_stack, axis=1)
        y_trans_all_err = np.std(y_trans_all_stack, axis=1)


        # the transformed positions of the points that were used to derive the transformation

        return (x_trans, x_trans_all_err, y_trans, y_trans_all_err)
        
        
        
    
class four_paramNW:
    '''
    defines parameter tranformation between x,y and xref, yref
    does not weight the points
    '''

    def __init__(self, x, y,xref, yref, order=None, weights=None):
        self.px, self.py =  four_param(x, y, xref, yref)

        self.order = order

        
    def evaluate(self, x, y):
        xn = self.px[0] + self.px[1]*x + self.px[2]*y
        yn = self.py[0] + self.py[1]*x + self.py[2]*y
        return xn, yn 
        

class PolyTransform(Transform2D):
    """
    Defines a 2D affine polynomial transform between x, y -> xref, yref
    The tranformation is independent for x and y and has the form (for 2nd order fit):

    x' = a0 + a1*x + a2*y. + a3*x**2 + a4*x*y. + a5*y**2
    y' = b0 + b1*x + b2*y. + b3*x**2 + b4*x*y. + b5*y**2
    """
    def __init__(self, order, px, py, pxerr=None, pyerr=None):
        """
        Specify the order of the affine transformation (0th, 1st, 2nd, etc.)
        and the coefficients for the x transformation and y transformation. 
        
        Parameters
        ----------
        px : list or array [a0, a1, a2, ...] 
            coefficients to transform input x coordinates into output x' coordinates.

        py : list or array [b0, b1, b2, ...] 
            coefficients to transform input y coordinates into output y' coordinates.
        
        order : int
            The order of the transformation. 0 = 2 free parameters, 1 = 6 free parameters.

        pxerr : array or list
            array or list of errors of the coefficients to transform input x coordinates 
            into output x' coordinates.
        
        pyerr : array or list
            array or list of errors of the coefficients to transform input y coordinates 
            into output y' coordinates.
        """
        px_dict = PolyTransform.make_param_dict(px, order, isY=False)
        py_dict = PolyTransform.make_param_dict(py, order, isY=True)
        
        self.px = models.Polynomial2D(order, **px_dict)
        self.py = models.Polynomial2D(order, **py_dict)
        self.pxerr = pxerr
        self.pyerr = pyerr
        self.order = order

        return

    @staticmethod
    def make_param_dict(initial_param, order, isY=False):
        """
        Convert initial parameter arrays into a format that astropy model Polynomial2D
        can understand. We expect input arrays in the form:

        a0 + a1*x + a2*y + a3*x^2 + a4*x*y + a5*y^2 + a6*x^3 + a7*x^2*y + a8*x*y^2 + a9*y^3

        and conver this into a dictionary where:

        c0_0 = a0
        c1_0 = a1
        c0_1 = a2
        c2_0 = a3
        c1_1 = a4
        c0_2 = a5
        c3_0 = a6
        c2_1 = a7
        c1_2 = a8
        c0_3 = a9

        The input/output ordering is set for easy coding using:

        for i in range(self.order + 1):
            for j in range(i + 1):
                coeff[i-j, j] for term x**(i-j) * y**(j)

        But astropy models Polynomial2D has its own special order... we try to 
        hide this entirely inside our object. 
        """
        idx = 0

        param_dict = {}

        for i in range(order + 1):
            for j in range(i + 1):
                xid = i - j
                yid = j

                coeff_key = 'c{0}_{1}'.format(xid, yid)
                
                if isinstance(initial_param, (list, tuple, np.ndarray)):
                    coeff = initial_param[idx]
                else:
                    # Handle case of no initial guess
                    coeff = 0.0
                    if isY == False and xid == 1 and yid == 0:
                        coeff = 1.0
                    if isY == True and xid == 0 and yid == 1:
                        coeff = 1.0

                param_dict[coeff_key] = coeff

                idx += 1

        return param_dict
    
    
    def evaluate(self, x, y):
        """
        Apply the transformation to a starlist.

        Parameters: 
        ----------
        x : numpy array
            The raw x coordinates to be transformed.
        y : numpy array
            The raw y coordinates to be transformed.

        Returns:
        ----------
        x' : array
            The transformed x coordinates.
        y' : array
            The transformed y coordinates. 
        """
        return self.px(x, y), self.py(x, y)

    
    def evaluate_error(self, x, y, xe, ye):
        """
        Transform positional uncertainties. 

        Parameters: 
        ----------
        x : numpy array
            The original x coordinates to be used in the transformation.
        y : numpy array
            The original y coordinates to be used in the transformation.
        xe : numpy array
            The raw x errors to be transformed.
        ye : numpy array
            The raw y errors to be transformed.

        Returns:
        ----------
        xe' : array
            The transformed x errors.
        ye' : array
            The transformed y errors. 

        """
        dxnew_dx = 0.0
        dxnew_dy = 0.0
        
        dynew_dx = 0.0
        dynew_dy = 0.0

        for i in range(self.order + 1):
            for j in range(i + 1):
                xid = i - j
                yid = j

                coeff_idx = self.px.param_names.index( 'c{0}_{1}'.format(xid, yid) )
                Xcoeff = self.px.parameters[coeff_idx]
                Ycoeff = self.py.parameters[coeff_idx]
                
                # First loop: df'/dx
                dxnew_dx += Xcoeff * (i - j) * x**(i-j-1) * y**j
                dynew_dx += Ycoeff * (i - j) * x**(i-j-1) * y**j

                # Second loop: df'/dy
                dxnew_dy += Xcoeff * (j) * x**(i-j) * y**(j-1)
                dynew_dy += Ycoeff * (j) * x**(i-j) * y**(j-1)

                
        # Take square root for xe/ye_new
        xe_new = np.sqrt((dxnew_dx * xe)**2 + (dxnew_dy * ye)**2)
        ye_new = np.sqrt((dynew_dx * xe)**2 + (dynew_dy * ye)**2)

        return xe_new, ye_new

    def evaluate_vel(self, x, y, vx, vy):
        """
        Transform velocities.

        Parameters: 
        ----------
        x : numpy array
            The original x coordinates to be used in the transformation.
        y : numpy array
            The original y coordinates to be used in the transformation.
        vx : numpy array
            The raw vx to be transformed.
        vy : numpy array
            The raw vy to be transformed.

        Returns:
        ----------
        vx' : array
            The transformed vx errors.
        vy' : array
            The transformed vy errors. 

        """
        vx_new = 0.0
        vy_new = 0.0

        for i in range(self.order + 1):
            for j in range(i + 1):
                xid = i - j
                yid = j

                coeff_idx = self.px.param_names.index( 'c{0}_{1}'.format(xid, yid) )
                Xcoeff = self.px.parameters[coeff_idx]
                Ycoeff = self.py.parameters[coeff_idx]
                
                # First term: df'/dx
                vx_new += Xcoeff * (i - j) * x**(i-j-1) * y**j * vx
                vy_new += Ycoeff * (i - j) * x**(i-j-1) * y**j * vx

                # Second term: df'/dy
                vx_new += Xcoeff * (j) * x**(i-j) * y**(j-1) * vy
                vy_new += Ycoeff * (j) * x**(i-j) * y**(j-1) * vy
                
        return vx_new, vy_new

    def evaluate_vel_err(self, x, y, vx, vy, xe, ye, vxe, vye):
        """
        Transform velocities.

        Parameters: 
        ----------
        x : numpy array
            The original x coordinates to be used in the transformation.
        y : numpy array
            The original y coordinates to be used in the transformation.
        vx : numpy array
            The raw vx to be transformed.
        vy : numpy array
            The raw vy to be transformed.
        xe : numpy array
            The original x coordinates to be used in the transformation.
        ye : numpy array
            The original y coordinates to be used in the transformation.
        vxe : numpy array
            The raw vx to be transformed.
        vye : numpy array
            The raw vy to be transformed.


        Returns:
        ----------
        vxe' : array
            The transformed vx errors.
        vye' : array
            The transformed vy errors. 

        """
        dvxnew_dx = 0.0
        dvxnew_dy = 0.0
        dvxnew_dvx = 0.0
        dvxnew_dvy = 0.0

        dvynew_dx = 0.0
        dvynew_dy = 0.0
        dvynew_dvx = 0.0
        dvynew_dvy = 0.0
        
        for i in range(self.order+1):
            for j in range(i+1):
                coeff_idx = self.px.param_names.index( 'c{0}_{1}'.format((i-j), j) )
                Xcoeff = self.px.parameters[coeff_idx]
                Ycoeff = self.py.parameters[coeff_idx]
                
                # First term: dv' / dx
                dvxnew_dx += Xcoeff * (i-j) * (i-j-1) * x**(i-j-2) * y**j * vx
                dvxnew_dx += Xcoeff * (j) * (i-j) * x**(i-j-1) * y**(j-1) * vy
                    
                dvynew_dx += Ycoeff * (i-j) * (i-j-1) * x**(i-j-2) * y**j * vx
                dvynew_dx += Ycoeff * (j) * (i-j) * x**(i-j-1) * y**(j-1) * vy
    
                # Second term: dvx' / dy
                dvxnew_dy += Xcoeff * (i-j) * (j) * x**(i-j-1) * y**(j-1) * vx
                dvxnew_dy += Xcoeff * (j) * (j-1) * x**(i-j-1) * y**(j-2) * vy
                    
                dvynew_dy += Ycoeff * (i-j) * (j) * x**(i-j-1) * y**(j-1) * vx
                dvynew_dy += Ycoeff * (j) * (j-1) * x**(i-j-1) * y**(j-2) * vy
    
                # Third term: dvx' / dvx
                dvxnew_dvx += Xcoeff * (i-j) * x**(i-j-1) * y**j
                dvynew_dvx += Ycoeff * (i-j) * x**(i-j-1) * y**j
    
                # Fourth term: dvx' / dvy
                dvxnew_dvy += Xcoeff * (j) * x**(i-j) * y**(j-1)
                dvynew_dvy += Ycoeff * (j) * x**(i-j) * y**(j-1)
    
        vxe_new = np.sqrt((dvxnew_dx * xe)**2 + (dvxnew_dy * ye)**2 + (dvxnew_dvx * vxe)**2 + (dvxnew_dvy * vye)**2)
        vye_new = np.sqrt((dvynew_dx * xe)**2 + (dvynew_dy * ye)**2 + (dvynew_dvx * vxe)**2 + (dvynew_dvy * vye)**2)

        return vxe_new, vye_new
    
    @classmethod
    def derive_transform(cls, x, y, xref, yref, order,
                         init_gx=None, init_gy=None, weights=None):
        
        # now, if the initial guesses are not none, fill in terms until 
        init_gx = PolyTransform.make_param_dict(init_gx, order, isY=False)
        init_gy = PolyTransform.make_param_dict(init_gy, order, isY=True)
        
        p_init_x = models.Polynomial2D(order, **init_gx)
        p_init_y = models.Polynomial2D(order, **init_gy)
        
        fit_p  = fitting.LinearLSQFitter()

        px = fit_p(p_init_x, x, y, xref, weights=weights)
        py = fit_p(p_init_y, x, y, yref, weights=weights)

        # Re-order the parameters for ingest by PolyTransform.
        Xcoeff = []
        Ycoeff = []
        for i in range(order+1):
            for j in range(i+1):
                coeff_idx = px.param_names.index( 'c{0}_{1}'.format((i-j), j) )
                Xcoeff.append( px.parameters[coeff_idx] )
                Ycoeff.append( py.parameters[coeff_idx] )
        
        trans = cls(order, Xcoeff, Ycoeff)

        return trans

    @classmethod
    def from_file(cls, trans_file):
        """
        Given a transformation coefficients file, read in the coefficients and create
        a PolyTransform object.
    
        Coefficients in the input file should have the following order:
        x' = a0 + a1*x + a2*y + a3*x**2. + a4*x*y  + a5*y**2. + ...
        y' = b0 + b1*x + b2*y + b3*x**2. + b4*x*y  + b5*y**2. + ...
    
        Parameters:
        ----------
        trans_file : str
            The name of the input file to read in.

        Returns:
        ----------
        trans_obj: PolyTransform
            A transformation object instance.
        """
        trans_table = Table.read(trans_file, format='ascii.commented_header', header_start=-1)
        Xcoeff = trans_table['Xcoeff']
        Ycoeff = trans_table['Ycoeff']

        # First determine the order based on the number of terms
        # Comes from Nterms = (N + 1) * (N + 2) / 2
        order = int((np.sqrt(1 + 8*len(Xcoeff)) - 3) / 2.)

        trans_obj = cls(order, Xcoeff, Ycoeff)
        
        return trans_obj

    def to_file(self, trans_file):
        """
        Given a transformation object, write out the coefficients in a text file
        (readable by java align). Outfile name is specified by user.
    
        Coefficients are output in file in the following way:
        x' = a0 + a1*x + a2*y + a3*x**2. + a4*x*y  + a5*y**2. + ...
        y' = b0 + b1*x + b2*y + b3*x**2. + b4*x*y  + b5*y**2. + ...
    
        Parameters:
        ----------
        trans_file : str
            The name of the output file to save the coefficients and meta data to. 
            This file can be read back in with 

                trans_obj = PolyTransfrom.from_file(trans_file).

        """
        # Extract info about transformation
        trans_name = transform.__class__.__name__
        trans_order = transform.order
        
        # Extract X, Y coefficients from transform
        Xcoeff = transform.px.parameters
        Ycoeff = transform.py.parameters
        coeff_names = transform.px.param_names  # same for both.
            
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
        # CODE TO GET INDICIES
        for i in range(self.order+1):
            for j in range(i+1):
                coeff_idx = self.px.param_names.index( 'c{0}_{1}'.format((i-j), j) )
                Xcoeff = self.px.parameters[coeff_idx]
                Ycoeff = self.py.parameters[coeff_idx]
                
                _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff, Ycoeff) )
    
        _out.close()

        return
        
class Shift(PolyTransform):
    '''
    Defines shift tranformation between x,y and xref, yref
    Does not weight the points.
    '''

    def __init__(self, xshift, yshift, xshift_err=None, yshift_err=None):
        px = [xshift]
        py = [yshift]
        pxerr = [xshift_err]
        pyerr = [yshift_err]
        order = 0
        
        px_dict = make_param_dict(px, 0, isY=False)
        py_dict = make_param_dict(py, 0, isY=True)
        
        self.px = models.Polynomial2D(order, **px_dict)
        self.py = models.Polynomial2D(order, **py_dict)
        self.pxerr = pxerr
        self.pyerr = pyerr
        self.order = order

        return

    @classmethod
    def derive_transform(cls, x, y, xref, yref, weights=None):
        dx = xref - x
        dy = yref - y
        
        xshift = np.average(dx, weights=weights)
        yshift = np.average(dy, weights=weights)

        # Error is estimated as the error on the mean (optionally weighted)
        if weights != None:
            wgt_sum = np.sum(weights)
            xshift_err = np.sqrt( np.sum(weights * (dx - xshift)**2) / wgt_sum) / len(x)**0.5
            yshift_err = np.sqrt( np.sum(weights * (dy - yshift)**2) / wgt_sum) / len(x)**0.5
        else:
            xshift_err = stats.sem(dx)
            yshift_err = stats.sem(dy)
        
        trans = Shift(xshift, yshift,
                          xshift_err=xshift_err, yshift_err=yshift_err)

        return trans

    def evaluate_error(self, x, y, xe, ye):
        xe_new = np.hypot(xe, self.pxerr[0])
        ye_new = np.hypot(ye, self.pyerr[0])
        
        return (xe_new, ye_new)

    def evaluate_vel(self, x, y, vx, vy):
        """
        Evaluate positions and velocities. Errors are only propogated IF all 4 errors
        (position and velocity) are input. 
        """
        return (vx, vy)

    def evaluate_vel_error(self, x, y, vx, vy, xe, ye, vxe, vye):
        """
        Evaluate positions and velocities. Errors are only propogated IF all 4 errors
        (position and velocity) are input. 
        """
        return (vxe, vye)
    
    
class LegTransform(Transform2D):
    def __init__(self, order, px, py):
        """
        Specify the order of the Legendre transformation (0th, 1st, 2nd, etc.)
        and the coefficients for the x transformation and y transformation. 
        
        Parameters
        ----------
        px : list or array [a0, a1, a2, ...] 
            coefficients to transform input x coordinates into output x' coordinates.

        py : list or array [b0, b1, b2, ...] 
            coefficients to transform input y coordinates into output y' coordinates.
        
        order : int
            The order of the transformation.

        pxerr : array or list
            array or list of errors of the coefficients to transform input x coordinates 
            into output x' coordinates.
        
        pyerr : array or list
            array or list of errors of the coefficients to transform input y coordinates 
            into output y' coordinates.
        """
        px_dict = make_param_dict(px)
        py_dict = make_param_dict(py)
        
        self.px = models.Legendre2D(order, order, **px_dict)
        self.py = models.Legendre2D(order, order, **py_dict)
        self.order = order

        return


    @classmethod
    def derive_transform(cls, x, y, xref, yref, order,
                 init_gx=None, init_gy=None, weights=None):
        """
        Defines a bivariate legendre tranformation from x,y -> xref,yref using Legnedre polynomials as the basis.
        Transforms are independent for x and y and of the form:
            x' = c0_0 + c1_0 * L_1(x) + c0_1*L_1(y) + ....
            y' = d0_0 + d1_0 * L_1(x) + d0_1*L_1(y) + ....
        Note that all input coorindates will be renomalized to be on the interval of [-1:1] before fitting.
        The evaulate function will use the same renomralization procedure.
        """
        # Initialize transformation object.
        init_gx = LegTransform.make_param_dict(init_gx, order)
        init_gy = LegTransform.make_param_dict(init_gy, order)

        p_init_x = models.Legendre2D(order, order, **init_gx)
        p_init_y = models.Legendre2D(order, order, **init_gy)
        
        fit_p  = fitting.LinearLSQFitter()

        px = fit_p(p_init_x, x, y, x_ref, weights=weights)
        py = fit_p(p_init_y, x, y, y_ref, weights=weights)

        trans = cls(order, px.parameters, py.parameters)
        

    def evaluate(self, x, y):
        """
        Apply the transformation to a starlist.

        Parameters: 
        ----------
        x : numpy array
            The raw x coordinates to be transformed.
        y : numpy array
            The raw y coordinates to be transformed.

        Returns:
        ----------
        x' : array
            The transformed x coordinates.
        y' : array
            The transformed y coordinates. 
        """
        return self.px(x, y), self.py(x, y)

    
    def evaluate_error(self, x, y, xe, ye):
        """
        Transform positional uncertainties. 

        Parameters: 
        ----------
        x : numpy array
            The original x coordinates to be used in the transformation.
        y : numpy array
            The original y coordinates to be used in the transformation.
        xe : numpy array
            The raw x errors to be transformed.
        ye : numpy array
            The raw y errors to be transformed.

        Returns:
        ----------
        xe' : array
            The transformed x errors.
        ye' : array
            The transformed y errors. 

        """
        dxnew_dx = 0.0
        dxnew_dy = 0.0
        
        dynew_dx = 0.0
        dynew_dy = 0.0

        for i in range(self.order + 1):
            for j in range(i + 1):
                xid = i - j
                yid = j

                coeff_idx = self.px.param_names.index( 'c{0}_{1}'.format(xid, yid) )
                Xcoeff = self.px.parameters[coeff_idx]
                Ycoeff = self.py.parameters[coeff_idx]
                
                # First loop: df'/dx
                dxnew_dx += Xcoeff * (i - j) * x**(i-j-1) * y**j
                dynew_dx += Ycoeff * (i - j) * x**(i-j-1) * y**j

                # Second loop: df'/dy
                dxnew_dy += Xcoeff * (j) * x**(i-j) * y**(j-1)
                dynew_dy += Ycoeff * (j) * x**(i-j) * y**(j-1)

                
        # Take square root for xe/ye_new
        xe_new = np.sqrt((dxnew_dx * xe)**2 + (dxnew_dy * ye)**2)
        ye_new = np.sqrt((dynew_dx * xe)**2 + (dynew_dy * ye)**2)

        return xe_new, ye_new

    def evaluate_vel(self, x, y, vx, vy):
        """
        Transform velocities.

        Parameters: 
        ----------
        x : numpy array
            The original x coordinates to be used in the transformation.
        y : numpy array
            The original y coordinates to be used in the transformation.
        vx : numpy array
            The raw vx to be transformed.
        vy : numpy array
            The raw vy to be transformed.

        Returns:
        ----------
        vx' : array
            The transformed vx errors.
        vy' : array
            The transformed vy errors. 

        """
        vx_new = 0.0
        vy_new = 0.0

        for i in range(self.order + 1):
            for j in range(i + 1):
                xid = i - j
                yid = j

                coeff_idx = self.px.param_names.index( 'c{0}_{1}'.format(xid, yid) )
                Xcoeff = self.px.parameters[coeff_idx]
                Ycoeff = self.py.parameters[coeff_idx]
                
                # First term: df'/dx
                vx_new += Xcoeff * (i - j) * x**(i-j-1) * y**j * vx
                vy_new += Ycoeff * (i - j) * x**(i-j-1) * y**j * vx

                # Second term: df'/dy
                vx_new += Xcoeff * (j) * x**(i-j) * y**(j-1) * vy
                vy_new += Ycoeff * (j) * x**(i-j) * y**(j-1) * vy
                
        return vx_new, vy_new

    def evaluate_vel_err(self, x, y, vx, vy, xe, ye, vxe, vye):
        """
        Transform velocities.

        Parameters: 
        ----------
        x : numpy array
            The original x coordinates to be used in the transformation.
        y : numpy array
            The original y coordinates to be used in the transformation.
        vx : numpy array
            The raw vx to be transformed.
        vy : numpy array
            The raw vy to be transformed.
        xe : numpy array
            The original x coordinates to be used in the transformation.
        ye : numpy array
            The original y coordinates to be used in the transformation.
        vxe : numpy array
            The raw vx to be transformed.
        vye : numpy array
            The raw vy to be transformed.


        Returns:
        ----------
        vxe' : array
            The transformed vx errors.
        vye' : array
            The transformed vy errors. 

        """
        dvxnew_dx = 0.0
        dvxnew_dy = 0.0
        dvxnew_dvx = 0.0
        dvxnew_dvy = 0.0

        dvynew_dx = 0.0
        dvynew_dy = 0.0
        dvynew_dvx = 0.0
        dvynew_dvy = 0.0
        
        for i in range(self.order+1):
            for j in range(i+1):
                coeff_idx = self.px.param_names.index( 'c{0}_{1}'.format((i-j), j) )
                Xcoeff = self.px.parameters[coeff_idx]
                Ycoeff = self.py.parameters[coeff_idx]
                
                # First term: dv' / dx
                dvxnew_dx += Xcoeff * (i-j) * (i-j-1) * x**(i-j-2) * y**j * vx
                dvxnew_dx += Xcoeff * (j) * (i-j) * x**(i-j-1) * y**(j-1) * vy
                    
                dvynew_dx += Ycoeff * (i-j) * (i-j-1) * x**(i-j-2) * y**j * vx
                dvynew_dx += Ycoeff * (j) * (i-j) * x**(i-j-1) * y**(j-1) * vy
    
                # Second term: dvx' / dy
                dvxnew_dy += Xcoeff * (i-j) * (j) * x**(i-j-1) * y**(j-1) * vx
                dvxnew_dy += Xcoeff * (j) * (j-1) * x**(i-j-1) * y**(j-2) * vy
                    
                dvynew_dy += Ycoeff * (i-j) * (j) * x**(i-j-1) * y**(j-1) * vx
                dvynew_dy += Ycoeff * (j) * (j-1) * x**(i-j-1) * y**(j-2) * vy
    
                # Third term: dvx' / dvx
                dvxnew_dvx += Xcoeff * (i-j) * x**(i-j-1) * y**j
                dvynew_dvx += Ycoeff * (i-j) * x**(i-j-1) * y**j
    
                # Fourth term: dvx' / dvy
                dvxnew_dvy += Xcoeff * (j) * x**(i-j) * y**(j-1)
                dvynew_dvy += Ycoeff * (j) * x**(i-j) * y**(j-1)
    
        vxe_new = np.sqrt((dvxnew_dx * xe)**2 + (dvxnew_dy * ye)**2 + (dvxnew_dvx * vxe)**2 + (dvxnew_dvy * vye)**2)
        vye_new = np.sqrt((dvynew_dx * xe)**2 + (dvynew_dy * ye)**2 + (dvynew_dvx * vxe)**2 + (dvynew_dvy * vye)**2)

        return vxe_new, vye_new

class PolyClipTransform(Transform2D):

    def __init__(self,x , y , xref, yref, degree,
                 niter=3, sig_clip =3 , weights=None):

        Transform2D.__init__(self,x,y,xref,yref)
        self.s_bool = np.ones(x.shape, dtype='bool')

        if weights is None:
            weights = np.ones(x.shape)
        c_x, c_y = four_param(x, y, xref, yref)
        
        for i in range(niter+1):
            t = PolyTransform(x[self.s_bool], y[self.s_bool], xref[self.s_bool], yref[self.s_bool], degree, init_gx=c_x, init_gy=c_y, weights=weights[self.s_bool])
            #reset the initial guesses based on the previous tranforamtion
            #it is not clear to me that using these values is better than recalculating an intial guess from a 4 parameter tranform
            c_x[0] = t.px.c0_0.value
            c_x[1] = t.px.c1_0.value
            c_x[2] = t.px.c0_1.value

            c_y[0] = t.py.c0_0.value
            c_y[1] = t.py.c1_0.value
            c_y[2] = t.py.c0_1.value

            xev, yev = t.evaluate(x, y)
            dx = xref - xev
            dy = yref - yev
            mx = np.mean(dx[self.s_bool])
            my = np.mean(dy[self.s_bool])
            
            sigx = np.std(dx[self.s_bool])
            sigy = np.std(dy[self.s_bool])
            sigr = np.sqrt(sigx**2 + sigy**2)
            mr = np.sqrt(mx**2+my**2)
            dr = np.sqrt(dx**2 + dy**2)
                   
            
            if i != niter :
                #do not update the star boolean if we have performed the final tranformation
                #self.s_bool = self.s_bool - ((dx > mx + sig_clip * sigx) + (dx < mx - sig_clip * sigx) + (dy > my + sig_clip * sigy) + (dy < my - sig_clip * sigy))
                self.s_bool = self.s_bool - ((dr > mr + sig_clip * sigr) + (dr < mr - sig_clip * sigr))

        self.t = t

    def evaluate(self,x,y):
        return self.t.evaluate(x,y)
            
class LegClipTransform(Transform2D):

    def __init__(self,x , y , xref, yref, degree,
                 niter=3, sig_clip =3 , weights=None):

        Transform2D.__init__(self,x,y,xref,yref)
        
        self.s_bool = np.ones(x.shape, dtype='bool')
        self.s_bool = np.array(self.s_bool)
        
        if weights is None:
            weights = np.ones(x.shape)
        c_x, c_y = four_param(x, y, xref, yref)
        
        for i in range(niter+1):
            t = LegTransform(x[self.s_bool], y[self.s_bool], xref[self.s_bool], yref[self.s_bool], degree, init_gx=c_x, init_gy=c_y, weights=weights[self.s_bool])
            #reset the initial guesses based on the previous tranforamtion
            #it is not clear to me that using these values is better than recalculating an intial guess from a 4 parameter tranform
            c_x[0] = t.px.c0_0.value
            c_x[1] = t.px.c1_0.value
            c_x[2] = t.px.c0_1.value

            c_y[0] = t.py.c0_0.value
            c_y[1] = t.py.c1_0.value
            c_y[2] = t.py.c0_1.value

            xev, yev = t.evaluate(x, y)
            dx = xref - xev
            dy = yref - yev
            mx = np.mean(dx[self.s_bool])
            my = np.mean(dy[self.s_bool])
            
            sigx = np.std(dx[self.s_bool])
            sigy = np.std(dy[self.s_bool])
            sigr = np.sqrt(sigx**2 + sigy**2)
            mr = np.sqrt(mx**2+my**2)
            dr = np.sqrt(dx**2 + dy**2)
                   
            
            if i != niter :
                #do not update the star boolean if we have performed the final tranformation
                #self.s_bool = self.s_bool - ((dx > mx + sig_clip * sigx) + (dx < mx - sig_clip * sigx) + (dy > my + sig_clip * sigy) + (dy < my - sig_clip * sigy))
                self.s_bool = self.s_bool - ((dr > mr + sig_clip * sigr) + (dr < mr - sig_clip * sigr))
                self.s_bool = np.array(self.s_bool) # do this to prevent s_bool to become a pandas series

        self.t = t

    def evaluate(self,x,y):
        return self.t.evaluate(x,y)
            
        
class PolyClipSplineTransform(Transform2D):
    """
    Performs polynomail fit, then a spline fit on the residual
    optionally performs signma clipping, if niter > 0 (default is zero)
    """
    
    def __init__(self, x, y, xref, yref, degree,
                  weights=None,niter=0,sigma=3,
                  kx=None, ky=None):

        '''
        '''

        Transform2D.__init__(self,x,y,xref,yref)        
        self.poly = PolyTransform(x, y, xref, yref, degree, weights=weights)
        xev, yev = self.poly.evaluate(x, y)

        
        self.spline = SplineTransform(xev, yev, xref, yref, weights=weights, kx=kx, ky=ky)

    def evaluate(self, x, y):
        xev, yev = self.poly.evaluate(x, y)
        return self.spline.evaluate(xev, yev)
        
class LegClipSplineTransform:
    """
    Performas a Legendre fit, then fits the residual with a spline
    can optinall y perform sigma clipping in the legendre step, by setting niter as > 0 (default to zero)
    """
    
    def __init__(self, x, y, xref, yref, degree,
                weights=None, kx=None, ky=None,
                 niter=0, sigma=3):

        '''
        '''
        self.leg = LegClipTransform(x, y, xref, yref, degree, weights=weights, niter=niter, sigma=sigma)
        xev, yev = self.leg.evaluate(x[self.leg.s_bool], y[self.leg.s_bool])

        
        self.spline = SplineTransform(xev, yev, xref, yref, weights=weights, kx=kx, ky=ky)

    def evaluate(self, x, y):
        xev, yev = self.poly.evaluate(x, y)
        return self.spline.evaluate(xev, yev)
        
              

class SplineTransform(Transform2D):


    def __init__(self, x, y, xref, yref,weights=None, kx=None,ky=None):
        
        Transform2D.__init__(self,x,y,xref,yref)
        
        if weights==None:
            weights = np.ones(x.shape)
        if kx == None:
            kx = np.linspace(x.min(), x.max())
        if ky == None:
            ky = np.linspace(y.min(), y.max())

        self.spline_x = spline(x,y,xref, tx=kx,ty=ky,w=weights)
        self.spline_y = spline(x,y,yref, tx=kx,ty=ky,w=weights)

    def evaluate(self,x,y):
        return self.spline_x.ev(x,y), self.spline_y.ev(x,y)


def make_param_dict(initial_param):
    '''
    Convert an array of initial guesses
    '''
    ord_dict = {1:0, 3:1, 6:2, 10:3, 15:4, 21:5, 28:6, 36:7}

    # If the input variable is not a list or array and is None,
    # then we will set new defaults. 
    if not isinstance(initial_param, collections.Iterable) and (initial_param == None):
        return  {'c0_0':0, 'c1_0':0, 'c0_1':0}

    # If we have input values, make sure the length makes sense for some
    # specific order of a polynomial.
    assert len(initial_param) in list(ord_dict.keys())

    # Now set the coefficients into the dictionary. 
    var_name = models.Polynomial2D(ord_dict[len(initial_param)]).param_names
    i_d = {}
    for i in range(len(initial_param)):
        i_d[var_name[i]] = initial_param[i]

    return i_d
      
def four_param(x,y,x_ref,y_ref):
    '''
    calulates the 4 parameter tranfrom between the inputs
    does not weight the fit

    returns two vecotors, with correct fromat to be the intitial guesses
    want to solve for four parameters
    equations
    x' = a0 + a1*x + a2 * y
    y' = b0 + -a2 * x + a1 * y

    Add in matrix notation of exactly what is going in here
    x'_0      x_0  y_0  1  0        a1
    y'_0      y_0  -x_0 0  1        a2 
    x'_1      x_1  y_1  1  0     *  a0
    y'_1   =  y_1  x_1  0  1        b0

    Above is the first 4 line of the matrix equation, the LHS and matrix with
    the coordiantes set the pattern that contines through te entire list of coordinates
    To solve, I take the psuedo inverse of  the coordinate matrix, and then take the 
    dot product of coo_mat^-1 * LHS to give the tranformation coefficients

    As a final step, I recalucate the translation terms based on fixed value of a1 and a2, 
    as the translatoin term is prone to numerical error
    '''

    mat_ref = []
    mat_coo = []
    for i in range(len(x_ref)):
        mat_ref.append(x_ref[i])
        mat_ref.append(y_ref[i])

        mat_coo.append([x[i],y[i],1,0])
        mat_coo.append([y[i],-1.0*x[i],0,1])


    imat = np.linalg.pinv(mat_coo)

    trans = np.dot(imat, mat_ref)
    #print( 'trans all', trans)
    #print( 'first guess x', np.array([trans[2],trans[0],trans[1]]))
    #print( 'first guess y', np.array([trans[3],trans[1],trans[0]]))

    #before returning, recalculate the offsets (a0,b0), as they are susceptible to numberical error
    #we take the linear terms as constant, and compute the average offset
    #a0 = x' - a1*x - a2*y
    #b0 = y' +a2 * x - a1 *y
    a0 = np.mean(x_ref - trans[0] * x - trans[1]*y)
    b0 = np.mean(y_ref + trans[1] *x - trans[0] *y)
    
    #returning the reaulting coeficient to match fitting
    #[x0,a1,a2], [y0,-a2,a1], should be applied ot x,y vector
    
    return np.array([a0,trans[0],trans[1]]), np.array([b0,-1.0*trans[1],trans[0]])

############# Tests

def test_PolyTransform():
    # test the transformation
    x = np.random.uniform(low=0,high=1000,size=1000)
    y = np.random.uniform(low=0,high=1000,size=1000)
    x_err = np.zeros(len(x))+0.1
    y_err = np.zeros(len(x))+0.1
    
    xref = (x + 100.0)*.5
    yref = (y + 100.0)*.5

    t = PolyTransform(x,y,xref,yref,2)
    x_trans, x_trans_err, y_trans, y_trans_err = t.evaluate_errors(x,x_err,y,y_err)

    for i in np.arange(len(x_trans)):
        print(( '%5.4f %5.4f %5.4f %5.4f %5.4f %5.4f' % (x[i],x_trans[i],x_trans_err[i],y[i],y_trans[i],y_trans_err[i])))
    return t

def test_LegTransform():
    # test the transformation
    x = np.random.uniform(low=0,high=1000,size=1000)
    y = np.random.uniform(low=0,high=1000,size=1000)
    x_err = np.random.normal(loc=0.0,scale=0.1,size=len(x))
    y_err = np.random.normal(loc=0.0,scale=0.1,size=len(x))
    
    xref = (x + 100.0)*.5
    yref = (y + 100.0)*.5

    x = x+x_err
    y = y+y_err
    
    t = LegTransform(x,y,xref,yref,3)
    x_trans, x_trans_err, y_trans, y_trans_err = t.evaluate_errors(x,x_err,y,y_err)

    
    for i in np.arange(len(x_trans)):
        print(( '%5.4f %5.4f %5.4f %5.4f %5.4f %5.4f' % (xref[i],x_trans[i],x_trans_err[i],yref[i],y_trans[i],y_trans_err[i])))

    # make sure the real and transformed positions are close
    np.testing.assert_allclose(xref,x_trans,atol=0.1,rtol=1e-3)
    np.testing.assert_allclose(yref,y_trans,atol=0.1,rtol=1e-3)
    
    print( 'PASSED!')
    return t

    
