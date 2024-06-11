from astropy.modeling import models, fitting
import numpy as np
from abc import ABC

class MotionModel(ABC):
    # Fit paramters: Shared fit parameters
    fitter_param_names = []

    # Fixed parameters: These are parameters that are required for the model, but are not 
    # fit quantities. For example, RA and Dec in a parallax model.
    fixed_param_names = []

    # Non-fit paramters: Custom paramters that will not be fit.
    # These parameters should be derived from the fit parameters and
    # they must exist as a variable on the model object
    optional_param_names = []

    def __init__(self, *args, **kwargs):
        # Check that required phot_params are proper arrays.
        # If not, then make them arrays of len(1).
        for param in self.fitter_param_names:
            param_var = getattr(self, param)
            if not isinstance(param_var, (list, np.ndarray)):
                setattr(self, param, np.array([param_var]))

        return

    def get_pos_at_time(self, t):
        #return x, y
        pass

    def fit_pos_at_time(self, t, x, y, xe, ye, update=True):
        """
        Fit the input positions on the sky and errors
        to determine new parameters for this motion model (MM).
        Current MM parameters are used as the initial guess.
        Best-fit parameters will be returned along with uncertainties
        and updated if update=True. 
        """
        #return params, param_errors
        pass
    
class Linear(MotionModel):
    """
    A 2D linear motion model for a star on the sky.
    """
    fitter_param_names = ['x0', 'vx', 'y0', 'vy']
    
    def __init__(x0, vx, y0, vy, t0):
        self.x0 = x0
        self.vx = vx
        self.y0 = y0
        self.vy = vy
        self.t0 = t0

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        self.poly_order = 1
        self.px = models.Polynomial1D(self.poly_order, c0=self.x0, c1=self.vx)
        self.py = models.Polynomial1D(self.poly_order, c0=self.y0, c1=self.vy)
        
        return

    def get_pos_at_time(self, t):
        dt = t - self.t0
        
        x = self.px(dt)
        y = self.py(dt)

        return x, y

    def fit_pos_at_time(self, t, x, y, xe, ye, update=False, fixed_t0=False):
        fitter = fitting.LevMarLSQFitter()

        # Determine the new optimal t0.
        # Unless t0 is fixed, calculate the t0 for the stars.
        if fixed_t0 is False:
            t_weight = 1.0 / np.hypot(xe, ye)
            t0 = np.average(t, weights=t_weight)
        elif fixed_t0 is True:
            t0 = self.t0
        else:
            t0 = fixed_t0
            
        dt = t - t0

        px_new = fitter(self.px, dt, x, weights=1/xe)
        px_cov = fitter.fit_into['param_cov']
        
        py_new = fitter(self.py, dt, y, weights=1/ye)
        py_cov = fitter.fit_into['param_cov']

        x0 = px.c0
        vx = px.c1
        y0 = py.c0
        vy = py.c1

        px_param_errs = dict(zip(self.px.param_names, np.diag(px_cov)**0.5))
        py_param_errs = dict(zip(self.py.param_names, np.diag(py_cov)**0.5))

        x0e = px_param_errors['c0']
        vxe = px_param_errors['c1']
        y0e = py_param_errors['c0']
        vye = py_param_errors['c1']
        
        if update:
            self.px = px_new
            self.py = py_new

            self.x0 = self.px.c0
            self.vx = self.px.c1
            self.y0 = self.py.c0
            self.vy = self.py.c1

        params = [x0, vx, y0, vy, t0]
        param_errors = [x0e, vxe, y0e, vye]
        
        return params, param_errors

        

        
    
        
        
        
