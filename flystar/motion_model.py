from astropy.modeling import models, fitting
import numpy as np
from abc import ABC
import pdb

class MotionModel(ABC):
    # Number of data points required to fit model
    n_pts_req = 0
    # Degrees of freedom for model
    dof = 0

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
        # TODO: do we need this?
        '''for param in self.fitter_param_names:
            param_var = getattr(self, param)
            if not isinstance(param_var, (list, np.ndarray)):
                setattr(self, param, np.array([param_var]))'''

        return

    def get_pos_at_time(self, t):
        #return x, y
        pass
        
    def get_pos_err_at_time(self, t):
        #return x_err, y_err
        pass
        
    def get_batch_pos_at_time(self, t):
        #return x, y, x_err, y_err
        pass

    def fit_motion_model(self, t, x, y, xe, ye, update=True):
        """
        Fit the input positions on the sky and errors
        to determine new parameters for this motion model (MM).
        Current MM parameters are used as the initial guess.
        Best-fit parameters will be returned along with uncertainties
        and updated if update=True. 
        """
        #return params, param_errors
        pass
        
    def get_chi2(self,dt,x,y,xe,ye):
        """
        Get the chi^2 value for the current MM and
        the input data.
        """
        # TODO: confirm whether we want reduced chi^2 or anything special - maybe kwarg option
        x_pred,y_pred = self.get_pos_at_time(dt)
        chi2x = np.sum((x-x_pred)**2 / xe**2)
        chi2y = np.sum((y-y_pred)**2 / ye**2)
        return chi2x,chi2y
    
class Fixed(MotionModel):
    """
    A non-moving motion model for a star on the sky.
    """
    n_pts_req = 1
    dof=1
    fitter_param_names = ['x0','y0']
    fixed_param_names = []
    
    def __init__(self, x0=0, y0=0, t0=2025.0,
                        x0_err=0, y0_err=0):
        self.x0 = x0
        self.y0 = y0
        self.t0 = t0
        self.x0_err = x0_err
        self.y0_err = y0_err
        
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()
        
        return
        
    def get_pos_at_time(self,t):
        return self.x0, self.y0
        
    def get_pos_err_at_time(self,t):
        return self.x0_err, self.y0_err
        
    def get_batch_pos_at_time(self,t,
                                x0=[],y0=[],t0=[],
                                x0_err=[], y0_err=[]):
        return x0,y0,x0_err,y0_err
            
    def fit_motion_model(self, dt, x, y, xe, ye, update=False, bootstrap=0):
        # Handle single data point case
        if len(x)==1:
            return [x[0],y[0]],[xe[0],ye[0]]
    
        #TODO: it seems like sometimes it's weighted by std and sometimes by var - confirm which to do here
        x0 = np.average(x, weights=1/xe**2)
        x0e = np.sqrt(np.average((x-x0)**2,weights=1/xe))
        y0 = np.average(y, weights=1/ye**2)
        y0e = np.sqrt(np.average((y-y0)**2,weights=1/ye))
        
        params = [x0, y0]
        param_errors = [x0e, y0e]
        
        if update:
            self.x0 = x0
            self.y0 = y0
            self.x0_err = x0e
            self.y0_err = y0e
        
        return params, param_errors
    
class Linear(MotionModel):
    """
    A 2D linear motion model for a star on the sky.
    """
    n_pts_req = 2
    dof=2
    fitter_param_names = ['x0', 'vx', 'y0', 'vy']
    fixed_param_names = ['t0']
    
    def __init__(self, x0=0, vx=0, y0=0, vy=0, t0=2025.0,
                            x0_err=0, vx_err=0, y0_err=0, vy_err=0):
        self.x0 = x0
        self.vx = vx
        self.y0 = y0
        self.vy = vy
        self.t0 = t0
        self.x0_err = x0_err
        self.vx_err = vx_err
        self.y0_err = y0_err
        self.vy_err = vy_err
        
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        self.poly_order = 1
        self.px = models.Polynomial1D(self.poly_order, c0=self.x0, c1=self.vx)
        self.py = models.Polynomial1D(self.poly_order, c0=self.y0, c1=self.vy)
        
        return

    def get_pos_at_time(self, dt):
        x = self.px(dt)
        y = self.py(dt)
        return x, y
        
    def get_pos_err_at_time(self, dt):
        x_err = np.hypot(self.x0_err, self.vx_err*dt)
        y_err = np.hypot(self.y0_err, self.vy_err*dt)
        return x_err, y_err
        
    def get_batch_pos_at_time(self,t,
                                x0=[],vx=[], y0=[],vy=[], t0=[],
                                x0_err=[],vx_err=[], y0_err=[],vy_err=[]):
        dt = t-t0
        x = x0 + dt*vx
        y = y0 + dt*vy
        x_err = np.hypot(x0_err, vx_err*dt)
        y_err = np.hypot(y0_err, vy_err*dt)
        return x,y,x_err,y_err

    def fit_motion_model(self, dt, x, y, xe, ye, update=False, bootstrap=0):
        fitter = fitting.LevMarLSQFitter()

        # Handle 2-data point case
        # TODO: is this the best way to handle this case ? Altered it to be consistent with t0
        if len(x)==2:
            ix = int(xe[0]>xe[1])
            iy = int(ye[0]>ye[1])
            dx = np.diff(x)[0]
            dy = np.diff(y)[0]
            dt_diff = np.diff(dt)[0]
            vx = dx / dt_diff
            vy = dy / dt_diff
            x0 = x[ix]-vx*dt[ix]
            y0 = y[iy]-vy*dt[iy]
            vxe = np.hypot(*xe)/dt_diff
            vye = np.hypot(*ye)/dt_diff
            x0e = np.sqrt(xe[ix]**2 + (dt[ix]*vxe)**2)
            y0e = np.sqrt(ye[iy]**2 + (dt[iy]*vye)**2)
            return [x0, vx, y0, vy],[x0e, vxe, y0e, vye]

        px_new = fitter(self.px, dt, x, weights=1/xe)
        px_cov = fitter.fit_info['param_cov']
        py_new = fitter(self.py, dt, y, weights=1/ye)
        py_cov = fitter.fit_info['param_cov']

        x0 = px_new.c0.value
        vx = px_new.c1.value
        y0 = py_new.c0.value
        vy = py_new.c1.value
        
        # Run the bootstrap
        if bootstrap > 0:
            edx = np.arange(len(x), dtype=int)

            fit_x0_b = np.zeros(bootstrap, dtype=float)
            fit_vx_b = np.zeros(bootstrap, dtype=float)
            fit_y0_b = np.zeros(bootstrap, dtype=float)
            fit_vy_b = np.zeros(bootstrap, dtype=float)
        
            for bb in range(bootstrap):
                bdx = np.random.choice(edx, len(x))
                
                px_b = fitter(self.px, dt[bdx], x[bdx], weights=1/xe[bdx])
                px_b_cov = fitter.fit_info['param_cov']
                py_b = fitter(self.py, dt[bdx], y[bdx], weights=1/ye[bdx])
                py_b_cov = fitter.fit_info['param_cov']

                fit_x0_b[bb] = px_b.c0.value
                fit_vx_b[bb] = px_b.c1.value
                fit_y0_b[bb] = py_b.c0.value
                fit_vy_b[bb] = py_b.c1.value
        
            # Save the errors from the bootstrap
            x0e = fit_x0_b.std()
            vxe = fit_vx_b.std()
            y0e = fit_y0_b.std()
            vye = fit_vy_b.std()
        else:
            px_param_errs = dict(zip(self.px.param_names, np.diag(px_cov)**0.5))
            py_param_errs = dict(zip(self.py.param_names, np.diag(py_cov)**0.5))
            x0e = px_param_errs['c0']
            vxe = px_param_errs['c1']
            y0e = py_param_errs['c0']
            vye = py_param_errs['c1']
        
        if update:
            self.px = px_new
            self.py = py_new

            self.x0 = x0
            self.vx = vx
            self.y0 = y0
            self.vy = vy
            
            self.x0_err = x0e
            self.vx_err = vxe
            self.y0_err = y0e
            self.vy_err = vye

        params = [x0, vx, y0, vy]
        param_errors = [x0e, vxe, y0e, vye]
        
        return params, param_errors

class Acceleration(MotionModel):
    """
    A 2D accelerating motion model for a star on the sky.
    """
    n_pts_req = 4 # TODO: consider special case for 3 pts
    dof=3
    fitter_param_names = ['x0', 'vx0', 'ax', 'y0', 'vy0', 'ay']
    fixed_param_names = ['t0']
    
    def __init__(self, x0=0, vx0=0, ax=0, y0=0, vy0=0, ay=0, t0=2025.0,
                            x0_err=0, vx0_err=0, ax_err=0, y0_err=0, vy0_err=0, ay_err=0):
        self.x0 = x0
        self.vx0 = vx0
        self.ax = ax
        self.y0 = y0
        self.vy0 = vy0
        self.ay = ay
        self.t0 = t0
        self.x0_err = x0_err
        self.vx0_err = vx0_err
        self.ax_err = ax_err
        self.y0_err = y0_err
        self.vy0_err = vy0_err
        self.ay_err = ay_err

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        self.poly_order = 2
        self.px = models.Polynomial1D(self.poly_order, c0=self.x0, c1=self.vx0, c2=self.ax)
        self.py = models.Polynomial1D(self.poly_order, c0=self.y0, c1=self.vy0, c2=self.ay)
    
        return

    def get_pos_at_time(self, t):
        dt = t - self.t0
        x = self.px(dt)
        y = self.py(dt)
        return x, y
        
    def get_pos_err_at_time(self, t):
        dt = t - self.t0
        x_err = np.sqrt(self.x0_err**2 + (self.vx_err*dt)**2 + (self.ax_err*dt**2)**2)
        y_err = np.sqrt(self.y0_err**2 + (self.vy_err*dt)**2 + (self.ay_err*dt**2)**2)
        return x_err, y_err
        
    def get_batch_pos_at_time(self,t,
                                x0=[],vx0=[],ax=[], y0=[],vy0=[],ay=[], t0=[],
                                x0_err=[],vx0_err=[],ax_err=[], y0_err=[],vy0_err=[],ay_err=[]):
        dt = t-t0
        x = x0 + dt*vx0 + 0.5*dt**2*ax
        y = y0 + dt*vy0 + 0.5*dt**2*ay
        x_err = np.sqrt(x0_err**2 + (vx0_err*dt)**2 + (0.5*ax_err*dt**2)**2)
        y_err = np.sqrt(y0_err**2 + (vy0_err*dt)**2 + (0.5*ay_err*dt**2)**2)
        return x,y,x_err,y_err

    def fit_motion_model(self, dt, x, y, xe, ye, update=False, bootstrap=0):
        fitter = fitting.LevMarLSQFitter()

        px_new = fitter(self.px, dt, x, weights=1/xe)
        px_cov = fitter.fit_info['param_cov']
        py_new = fitter(self.py, dt, y, weights=1/ye)
        py_cov = fitter.fit_info['param_cov']

        x0 = px_new.c0.value
        vx0 = px_new.c1.value
        ax = px_new.c2.value
        y0 = py_new.c0.value
        vy0 = py_new.c1.value
        ay = py_new.c2.value
        
        # Run the bootstrap
        if bootstrap > 0:
            edx = np.arange(len(x), dtype=int)

            fit_x0_b = np.zeros(bootstrap, dtype=float)
            fit_vx0_b = np.zeros(bootstrap, dtype=float)
            fit_ax_b = np.zeros(bootstrap, dtype=float)
            fit_y0_b = np.zeros(bootstrap, dtype=float)
            fit_vy0_b = np.zeros(bootstrap, dtype=float)
            fit_ay_b = np.zeros(bootstrap, dtype=float)

            for bb in range(bootstrap):
                bdx = np.random.choice(edx, len(x))
                
                px_b = fitter(self.px, dt[bdx], x[bdx], weights=1/xe[bdx])
                px_b_cov = fitter.fit_info['param_cov']
                py_b = fitter(self.py, dt[bdx], y[bdx], weights=1/ye[bdx])
                py_b_cov = fitter.fit_info['param_cov']

                fit_x0_b[bb] = px_b.c0.value
                fit_vx0_b[bb] = px_b.c1.value
                fit_ax_b[bb] = px_b.c2.value
                fit_y0_b[bb] = py_b.c0.value
                fit_vy0_b[bb] = py_b.c1.value
                fit_ay_b[bb] = py_b.c2.value

            # Save the errors from the bootstrap
            x0e = fit_x0_b.std()
            vx0e = fit_vx0_b.std()
            axe = fit_ax_b.std()
            y0e = fit_y0_b.std()
            vy0e = fit_vy0_b.std()
            aye = fit_ay_b.std()
        else:
            px_param_errs = dict(zip(self.px.param_names, np.diag(px_cov)**0.5))
            py_param_errs = dict(zip(self.py.param_names, np.diag(py_cov)**0.5))
            x0e = px_param_errs['c0']
            vx0e = px_param_errs['c1']
            axe = px_param_errs['c2']
            y0e = py_param_errs['c0']
            vy0e = py_param_errs['c1']
            aye = py_param_errs['c2']
        
        if update:
            self.px = px_new
            self.py = py_new

            self.x0 = x0
            self.vx0 = vx0
            self.ax = ax
            self.y0 = y0
            self.vy0 = vy0
            self.ay = ay
            
            self.x0_err = x0e
            self.vx0_err = vx0e
            self.ax_err = axe
            self.y0_err = y0e
            self.vy0_err = vy0e
            self.ay_err = aye

        params = [x0, vx0, ax, y0, vy0, ay]
        param_errors = [x0e, vx0e, axe, y0e, vy0e, aye]
        
        return params, param_errors

"""
Get all the motion model parameters for a given motion_model_name.
Optionally, include fixed and error parameters (included by default).
"""
def get_one_motion_model_param_names(motion_model_name, with_errors=True, with_fixed=True):
    mod = eval(motion_model_name)
    list_of_parameters = []
    list_of_parameters += getattr(mod, 'fitter_param_names')
    if with_fixed:
        list_of_parameters += getattr(mod, 'fixed_param_names')
    if with_errors:
        list_of_parameters += [par+'_err' for par in getattr(mod, 'fitter_param_names')]
    return list_of_parameters

"""
Get all the motion model parameters for all models given in motion_model_list.
Optionally, include fixed and error parameters (included by default).
"""
def get_list_motion_model_param_names(motion_model_list, with_errors=True, with_fixed=True):
    list_of_parameters = []
    all_motion_models = [eval(mm) for mm in np.unique(motion_model_list).tolist()]
    for aa in range(len(all_motion_models)):
        param_names = getattr(all_motion_models[aa], 'fitter_param_names')
        param_fixed_names = getattr(all_motion_models[aa], 'fixed_param_names')
        param_err_names = [par+'_err' for par in param_names]

        list_of_parameters += param_names
        if with_fixed:
            list_of_parameters += param_fixed_names
        if with_errors:
            list_of_parameters += param_err_names
    
    return np.unique(list_of_parameters).tolist()

"""
Get all the motion model parameters for all models defined in this module.
Optionally, include fixed and error parameters (included by default).
"""
def get_all_motion_model_param_names(with_errors=True, with_fixed=True):
    list_of_parameters = []
    all_motion_models = MotionModel.__subclasses__()
    for aa in range(len(all_motion_models)):
        param_names = getattr(all_motion_models[aa], 'fitter_param_names')
        param_fixed_names = getattr(all_motion_models[aa], 'fixed_param_names')
        param_err_names = [par+'_err' for par in param_names]

        list_of_parameters += param_names
        if with_fixed:
            list_of_parameters += param_fixed_names
        if with_errors:
            list_of_parameters += param_err_names
    
    return np.unique(list_of_parameters).tolist()
    
        
        
        