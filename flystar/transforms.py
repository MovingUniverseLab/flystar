from astropy.modeling import models, fitting
import numpy as np
from scipy.interpolate import LSQBivariateSpline as spline
from astropy.table import Table
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
    
    def evaluate_errors(self, x, x_err, y, y_err, nsim=500):
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

        x_trans_all_err = np.std(x_trans_all_stack,axis=1)
        y_trans_all_err = np.std(y_trans_all_stack,axis=1)


        # the transformed positions of the points that were used to derive the transformation

        return (x_trans, x_trans_all_err, y_trans, y_trans_all_err)
        
        
        
class Shift:
    '''
    Defines shift tranformation between x,y and xref, yref
    Does not weight the points.
    '''

    def __init__(self, x, y, xref, yref, order=None, weights=None):
        self.px = np.array([ np.average(xref - x, weights=weights) ])
        self.py = np.array([ np.average(yref - y, weights=weights) ])
    
        self.order = 0
        
        return

    def evaluate(self, x, y):
        xn = self.px[0] + x
        yn = self.py[0] + y
        
        return xn, yn 

    
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
    x' = a0 + a1*x + a2*x**2. + a3*y + a4*y**2. + a5*x*y
    y' = b0 + b1*x + b2*x**2. + b3*y + b4*y**2. + b5*x*y

    Currently, this only supports an initial guess of the linear terms.
    """
    def __init__(self, order, px, py, pxerr=None, pyerr=None):
        """
        Specify the order of the affine transformation (0th, 1st, 2nd, etc.)
        and the coefficients for the x transformation and y transformation. 
        
        Parameters
        ----------
        order : int
            The order of the transformation. 0 = 2 free parameters, 1 = 6 free parameters.

        px : Polynomial2D
            array or list of coefficients to transform input x coordinates into output x' coordinates.

        py : Polynomial2D
            array or list of coefficients to transform input y coordinates into output y' coordinates.
        
        pxerr : array or list
            array or list of errors of the coefficients to transform input x coordinates into output x' coordinates.
        
        pyerr : array or list
            array or list of errors of the coefficients to transform input y coordinates into output y' coordinates.
        """
        px_dict = check_initial_guess(px.parameters)
        py_dict = check_initial_guess(py.parameters)
        self.px = models.Polynomial2D(order, **px_dict)
        self.py = models.Polynomial2D(order, **py_dict)
        self.pxerr = pxerr
        self.pyerr = pyerr
        self.order = order

        return
    
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
        return self.px(x,y), self.py(x,y)
    
    @classmethod
    def derive_transform(cls, x, y, xref, yref, order,
                         init_gx=None, init_gy=None, weights=None):

        p0 = models.Polynomial2D(order)
        
        # now, if the initial guesses are not none, fill in terms until 
        init_gx = check_initial_guess(init_gx)
        init_gy = check_initial_guess(init_gy)
        
        p_init_x = models.Polynomial2D(order, **init_gx )
        p_init_y = models.Polynomial2D(order, **init_gy )
        
        fit_p  = fitting.LinearLSQFitter()

        px = fit_p(p_init_x, x, y, xref, weights=weights)
        py = fit_p(p_init_y, x, y, yref, weights=weights)

        return PolyTransform(order, px, py)

    
    
class LegTransform(Transform2D):

    def __init__(self, x, y, xref, yref, order,
                 init_gx=None,init_gy=None, weights=None):
        '''
        defines a 2d polnyomial tranformation fomr x,y -> xref,yref using Legnedre polynomials as the basis
        transforms are independent for x and y, of the form
        x' = c0_0 + c1_0 * L_1(x) + c0_1*L_1(y) + ....
        y' = d0_0 + d1_0 * L_1(x) + d0_1*L_1(y) + ....
        Note that all input coorindates will be renomalized to be on the interval of [-1:1] for fitting
        The evaulate function will use the same renomralization procedure
        '''

        # initialize parent class
        Transform2D.__init__(self,x,y,xref,yref)
        
        init_gx = check_initial_guess(init_gx)
        init_gy = check_initial_guess(init_gy)
        
        self.x_nc , x_norm= self.norm0(x)
        self.x_ncr, x_norm_ref = self.norm0(xref)
        self.y_nc , y_norm = self.norm0(y)
        self.y_ncr , y_norm_ref = self.norm0(yref)
        self.order = order
        
        p_init_x = models.Legendre2D(order, order,**init_gx)
        p_init_y = models.Legendre2D(order, order, **init_gy)
       
        fit_p  = fitting.LinearLSQFitter()

        self.px = fit_p(p_init_x, x_norm, y_norm, x_norm_ref, weights=weights)
        self.py = fit_p(p_init_y, x_norm, y_norm, y_norm_ref, weights=weights)
        

    def evaluate(self, x, y):
        xnew = self.rnorm(self.px(self.norm(x, self.x_nc), self.norm(y, self.y_nc)),
                          self.x_ncr)
        ynew = self.rnorm(self.py(self.norm(x, self.x_nc), self.norm(y, self.y_nc)),
                          self.y_ncr)
        return xnew, ynew

        


    def norm0(self, x):

        
        xmin = np.min(x)
        xdiv = np.max(x- xmin)/2.0
        n_param = np.array([xmin, xdiv])
        return n_param, self.norm(x, n_param)
        
    def norm(self, x, n_param):
        '''
        x is vector to be normalized, n_param is an array of [offset to subtract, value to divide]
        '''
        return (x - n_param[0]) / n_param[1] - 1.0
       
        
    def rnorm(self,x,n_param):
        '''
        reverses normalization process 
        '''
        
        return (x+1.0)   * n_param[1] + n_param[0]

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





def check_initial_guess(initial_param):
    '''
    Checks initial guesses for polynomial (and LEgendre) tranformations
    '''
    ord_dict = {1:0, 3:1, 6:2, 10:3, 15:4, 21:5, 28:6, 36:7}
    if initial_param == None:
        return  {'c0_0':0, 'c1_0':0, 'c0_1':0}
    assert len(initial_param) in list(ord_dict.keys())
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

    
