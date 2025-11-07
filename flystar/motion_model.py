import numpy as np
from abc import ABC
import pdb
from flystar import parallax
from astropy.time import Time
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

class MotionModel(ABC):
    # Number of data points required to fit model
    n_pts_req = 0
    # Degrees of freedom for model
    n_params = 0

    # Fit paramters: Shared fit parameters
    fitter_param_names = []

    # Fixed parameters: These are parameters that are required for the model, but are not 
    # fit quantities. For example, RA and Dec in a parallax model.
    fixed_param_names = []
    fixed_meta_data = []

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

    def get_pos_at_time(self, params, t):
        #return x, y
        pass
        
    def get_batch_pos_at_time(self, t):
        #return x, y, x_err, y_err
        pass
        
    def run_fit(
        self, t, x, y, xe, ye, t0, 
        weighting='var',
        use_scipy=True, 
        absolute_sigma=True, 
        fill_value=np.inf
    ):
        # Run a single fit (used both for overall fit + bootstrap iterations)
        pass
        
    def get_weights(self, xe, ye, weighting='var'):
        if weighting=='std':
            return 1./xe, 1./ye
        elif weighting=='var':
            return 1./xe**2, 1./ye**2
        else:
            warnings.warn("Invalid weighting, using default weighting scheme var.", UserWarning)
            return 1./xe**2, 1./ye**2

    def fit_motion_model(
        self, t, x, y, xe, ye, t0, 
        bootstrap=0, 
        weighting='var',
        use_scipy=True, 
        absolute_sigma=True,
        fill_value=np.inf,
        rng=None
    ):
        """
        Fit the input positions on the sky and errors
        to determine new parameters for this motion model (MM).
        Best-fit parameters will be returned along with uncertainties.
        """
        params, param_errs, chi2x, chi2y = self.run_fit(
            t, x, y, xe, ye, t0=t0, 
            weighting=weighting,
            use_scipy=use_scipy, 
            absolute_sigma=absolute_sigma,
            fill_value=fill_value
        )

        if bootstrap > 0 and len(x) > (self.n_pts_req):
            edx = np.arange(len(x), dtype=int)
            bb_params = []
            bb_params_errs = []
            for bb in range(bootstrap):
                bdx = np.random.choice(edx, len(x), replace=False)
                params_bdx, param_errs_bdx, chi2x_bdx, chi2y_bdx = self.run_fit(
                    t[bdx], x[bdx], y[bdx], xe[bdx], ye[bdx], t0=t0,
                    weighting=weighting, 
                    use_scipy=use_scipy, 
                    absolute_sigma=absolute_sigma, 
                    params_guess=params, 
                    fill_value=fill_value
                )
                bb_params.append(params_bdx)
                bb_params_errs.append(param_errs_bdx)
        
            # Save the errors from the bootstrap
            param_errs = np.std(bb_params, axis=0)
            
            # Account for odd case
            inf_errs = [np.all(arr==np.inf) for arr in np.transpose(np.array(bb_params_errs))]
            param_errs[inf_errs] = 0.0

        return params, param_errs, chi2x, chi2y

    def get_chi2(self, fit_params, fixed_params, t, x, y, xe, ye, reduced=False):
        """
        Get the chi^2 value for the current MM and
        the input data.
        """
        x_pred, y_pred = self.get_pos_at_time(fit_params, fixed_params, t)
        chi2x = np.sum((x - x_pred)**2 / xe**2)
        chi2y = np.sum((y - y_pred)**2 / ye**2)
        if reduced:
            if len(t) == self.n_params:
                chi2x, chi2y = 0, 0
            else:
                dof = len(x) - self.n_params
                chi2x, chi2y = chi2x / dof, chi2y / dof
        return chi2x, chi2y

class Fixed(MotionModel):
    """
    A non-moving motion model for a star on the sky.
    """
    
    n_pts_req = 1
    n_params = 1
    fitter_param_names = ['x0','y0']
    fixed_param_names = []

    def __init__(self, **kwargs):
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()
        return
        
    def get_pos_at_time(self, fit_params, fixed_params, t):
        fit_params_dict = dict(zip(self.fitter_param_names, fit_params))
        if hasattr(t, "__len__"):
            return np.repeat(fit_params_dict['x0'], len(t)), np.repeat(fit_params_dict['y0'], len(t))
        else:
            return fit_params_dict['x0'], fit_params_dict['y0']
        
    def get_batch_pos_at_time(self,t,
                                x0=[],y0=[],t0=[],
                                x0_err=[], y0_err=[]):
        if hasattr(t, "__len__"):
            return np.repeat(x0[:,np.newaxis],len(t),axis=1), np.repeat(y0[:,np.newaxis],len(t),axis=1), np.repeat(x0_err[:,np.newaxis],len(t),axis=1), np.repeat(y0_err[:,np.newaxis],len(t),axis=1)
        else:
            return x0,y0,x0_err,y0_err
            
    def run_fit(
            self, t, x, y, xe, ye, t0, 
            weighting='var', 
            use_scipy=True, 
            absolute_sigma=True,
            params_guess=None,
            fill_value=np.inf
    ):
        if not use_scipy:
            Warning("Fixed model has no non-scipy fitter option. Running with scipy.")
        # Handle single data point case
        if len(x)==1:
            x0, y0, x0e, y0e = x[0], y[0], xe[0], ye[0]
    
        else:
            x_wt, y_wt = self.get_weights(xe, ye, weighting=weighting)
            x0 = np.average(x, weights=x_wt)
            x0e = np.sqrt(np.average((x - x0)**2, weights=x_wt))
            y0 = np.average(y, weights=y_wt)
            y0e = np.sqrt(np.average((y - y0)**2, weights=y_wt))

        params = np.array([x0, y0])
        param_errors = np.array([x0e, y0e])
        
        chi2x, chi2y = self.get_chi2(params, [], t, x, y, xe, ye)
        degree_of_freedom = len(x) - 1
        if not absolute_sigma:
            if degree_of_freedom > 0:
                reduced_chi2x = chi2x / degree_of_freedom
                reduced_chi2y = chi2y / degree_of_freedom

                param_errors[0] *= reduced_chi2x**0.5
                param_errors[1] *= reduced_chi2y**0.5
            else:
                warnings.warn(
                    f'Degree of freedom <= 0. Covariance of the parameters could not be estimated. Setting parameter uncertainties to fill value {fill_value}.',
                    OptimizeWarning, stacklevel=2
                )
                param_errors = np.full_like(param_errors, fill_value)

        return params, param_errors, chi2x, chi2y

class Linear(MotionModel):
    """
    A 2D linear motion model for a star on the sky.
    """
    
    n_pts_req = 2
    n_params = 2
    fitter_param_names = ['x0', 'vx', 'y0', 'vy']
    fixed_param_names = ['t0']

    def __init__(self, **kwargs):
        
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()
        return

    def get_pos_at_time(self, fit_params, fixed_params, t):
        fit_params_dict = dict(zip(self.fitter_param_names, fit_params))
        fixed_params_dict = dict(zip(self.fixed_param_names, fixed_params))
        dt = t-fixed_params_dict['t0']
        return fit_params_dict['x0'] + fit_params_dict['vx']*dt, fit_params_dict['y0'] + fit_params_dict['vy']*dt

    def get_batch_pos_at_time(self, t, x0=[],vx=[], y0=[],vy=[], t0=[],
                x0_err=[],vx_err=[], y0_err=[],vy_err=[], **kwargs):
        if hasattr(t, "__len__"):
            dt = t-t0[:,np.newaxis]
            x = x0[:,np.newaxis] + dt*vx[:,np.newaxis]
            y = y0[:,np.newaxis] + dt*vy[:,np.newaxis]
            x_err = np.hypot(x0_err[:,np.newaxis], vx_err[:,np.newaxis]*dt)
            y_err = np.hypot(y0_err[:,np.newaxis], vy_err[:,np.newaxis]*dt)
        else:
            dt = t-t0
            x = x0 + dt*vx
            y = y0 + dt*vy
            x_err = np.hypot(x0_err, vx_err*dt)
            y_err = np.hypot(y0_err, vy_err*dt)
        return x,y,x_err,y_err

    def run_fit(
            self, t, x, y, xe, ye, t0, 
            weighting='var', 
            use_scipy=True, 
            absolute_sigma=True,
            params_guess=None,
            fill_value=np.inf
    ):
        dt = t - t0
        x_wt, y_wt = self.get_weights(xe, ye, weighting=weighting)
        if params_guess is None:
            params_guess = [x.mean(), 0., y.mean(), 0.]

        if use_scipy:
            def linear(t, c0, c1):
                return c0 + c1*t
            x_opt, x_cov = curve_fit(linear, dt, x, p0=np.array(params_guess[:2]), sigma=1/np.sqrt(x_wt), absolute_sigma=absolute_sigma)
            y_opt, y_cov = curve_fit(linear, dt, y, p0=np.array(params_guess[2:]), sigma=1/np.sqrt(y_wt), absolute_sigma=absolute_sigma)
            x0, vx = x_opt
            y0, vy = y_opt
            x0e, vxe = np.sqrt(x_cov.diagonal())
            y0e, vye = np.sqrt(y_cov.diagonal())
            params = np.array([x0, vx, y0, vy])
            param_errors = np.array([x0e, vxe, y0e, vye])
            chi2_x, chi2_y = self.get_chi2(params, [t0], t, x, y, xe, ye)
            
        else:
            # Use  https://en.wikipedia.org/wiki/Weighted_least_squares#Solution scheme
            x = np.array(x)
            y = np.array(y)
            dt = np.array(dt)
            X_mat_t = np.vander(dt, 2)
            # x calculation
            W_mat_x = np.diag(x_wt)
            XTWX_mat_x = X_mat_t.T @ W_mat_x @ X_mat_t
            pcov_x = np.linalg.inv(XTWX_mat_x)  # Covariance Matrix
            popt_x = pcov_x @ X_mat_t.T @ W_mat_x @ x   # Linear Solution
            perr_x = np.sqrt(np.diag(pcov_x))   # Uncertainty of Linear Solution
            # y calculation
            W_mat_y = np.diag(y_wt)
            XTWX_mat_y = X_mat_t.T @ W_mat_y @ X_mat_t
            pcov_y = np.linalg.inv(XTWX_mat_y)  # Covariance Matrix
            popt_y = pcov_y @ X_mat_t.T @ W_mat_y @ y   # Linear Solution
            perr_y = np.sqrt(np.diag(pcov_y))   # Uncertainty of Linear Solution
            # prepare values to return
            vx, x0 = popt_x
            vy, y0 = popt_y
            vxe, x0e = perr_x
            vye, y0e = perr_y

            # Does not use get_chi2 to accelerate calculation
            residual_x = x - X_mat_t @ popt_x
            residual_y = y - X_mat_t @ popt_y
            
            
            chi2_x = residual_x.T @ W_mat_x @ residual_x
            chi2_y = residual_y.T @ W_mat_y @ residual_y

            params = np.array([x0, vx, y0, vy])
            param_errors = np.array([x0e, vxe, y0e, vye])

            if not absolute_sigma:
                degree_of_freedom = len(x) - 2
                if degree_of_freedom > 0:
                    reduced_chi2_x = chi2_x/(len(x) - 2)
                    reduced_chi2_y = chi2_y/(len(x) - 2)
                    
                    param_errors[0:2] *= reduced_chi2_x**0.5
                    param_errors[2:4] *= reduced_chi2_y**0.5

                else:
                    warnings.warn(
                        f'Degree of freedom <= 0. Covariance of the parameters could not be estimated. Setting parameter uncertainties to fill value {fill_value}.',
                        OptimizeWarning, stacklevel=2
                    )
                    param_errors = np.full_like(param_errors, fill_value)

        return params, param_errors, chi2_x, chi2_y

class Acceleration(MotionModel):
    """
    A 2D accelerating motion model for a star on the sky.
    """
    n_pts_req = 4 # TODO: consider special case for 3 pts
    n_params = 3
    fitter_param_names = ['x0', 'vx0', 'ax', 'y0', 'vy0', 'ay']
    fixed_param_names = ['t0']
    
    def __init__(self, x0=0, vx0=0, ax=0, y0=0, vy0=0, ay=0, t0=None,
                            x0_err=0, vx0_err=0, ax_err=0, y0_err=0, vy0_err=0, ay_err=0, **kwargs):
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()
        return
        
    def get_pos_at_time(self, fit_params, fixed_params, t):
        fit_params_dict = dict(zip(self.fitter_param_names, fit_params))
        fixed_params_dict = dict(zip(self.fixed_param_names, fixed_params))
        dt = t-fixed_params_dict['t0']
        x = fit_params_dict['x0'] + fit_params_dict['vx0']*dt + 0.5*fit_params_dict['ax']*dt**2
        y = fit_params_dict['y0'] + fit_params_dict['vy0']*dt + 0.5*fit_params_dict['ay']*dt**2
        return x, y
        
    def get_batch_pos_at_time(self,t,
                                x0=[],vx0=[],ax=[], y0=[],vy0=[],ay=[], t0=[],
                                x0_err=[],vx0_err=[],ax_err=[], y0_err=[],vy0_err=[],ay_err=[], **kwargs):
        if hasattr(t, "__len__"):
            dt = t-t0[:,np.newaxis]
            x = x0[:,np.newaxis] + dt*vx0[:,np.newaxis] + 0.5*dt**2*ax[:,np.newaxis]
            y = y0[:,np.newaxis] + dt*vy0[:,np.newaxis] + 0.5*dt**2*ay[:,np.newaxis]
            x_err = np.sqrt(x0_err[:,np.newaxis]**2 + (vx0_err[:,np.newaxis]*dt)**2 + (0.5*ax_err[:,np.newaxis]*dt**2)**2)
            y_err = np.sqrt(y0_err[:,np.newaxis]**2 + (vy0_err[:,np.newaxis]*dt)**2 + (0.5*ay_err[:,np.newaxis]*dt**2)**2)
        else:
            dt = t-t0
            x = x0 + dt*vx0 + 0.5*dt**2*ax
            y = y0 + dt*vy0 + 0.5*dt**2*ay
            x_err = np.sqrt(x0_err**2 + (vx0_err*dt)**2 + (0.5*ax_err*dt**2)**2)
            y_err = np.sqrt(y0_err**2 + (vy0_err*dt)**2 + (0.5*ay_err*dt**2)**2)
        return x,y,x_err,y_err

    def run_fit(
        self, t, x, y, xe, ye, t0, 
        weighting='var', 
        use_scipy=True, 
        absolute_sigma=True, 
        params_guess=None,
        fill_value=np.inf
    ):
        if not use_scipy:
            Warning("Acceleration model has no non-scipy fitter option. Running with scipy.")
        dt = t - t0
        x_wt, y_wt = self.get_weights(xe,ye, weighting=weighting)
        if params_guess is None:
            params_guess = [x.mean(), 0., 0., y.mean(), 0., 0.]

        def accel(t, c0, c1, c2):
            return c0 + c1*t + 0.5*c2*t**2

        x_opt, x_cov = curve_fit(accel, dt, x, p0=np.array(params_guess[:3]), sigma=1/x_wt**0.5, absolute_sigma=True)
        y_opt, y_cov = curve_fit(accel, dt, y, p0=np.array(params_guess[3:]), sigma=1/y_wt**0.5, absolute_sigma=True)
        x0 = x_opt[0]
        y0 = y_opt[0]
        vx0 = x_opt[1]
        vy0 = y_opt[1]
        ax = x_opt[2]
        ay = y_opt[2]

        x0e, vx0e, axe = np.sqrt(x_cov.diagonal())
        y0e, vy0e, aye = np.sqrt(y_cov.diagonal())

        params = [x0, vx0, ax, y0, vy0, ay]
        param_errors = [x0e, vx0e, axe, y0e, vy0e, aye]

        return params, param_errors

class Parallax(MotionModel):
    """
    Motion model for linear proper motion + parallax
    
    Requires RA, Dec, and PA parameters (degrees) for parallax calculation.
        RA, Dec in J2000
    Optional PA is counterclockwise offset of the image y-axis from North.
    Optional obs parameter describes observer location, default is 'earth'.
    """
    
    n_pts_req = 4
    n_params = 3
    fitter_param_names = ['x0', 'vx', 'y0', 'vy', 'pi']
    fixed_param_names = ['t0']
    fixed_meta_data = ['RA','Dec','PA','obs']
    
    def __init__(self, RA, Dec, PA=0.0, obs='earth', **kwargs):
        self.RA = RA
        self.Dec = Dec
        self.PA = PA
        self.obs = obs
        self.plx_vector_cached = None
        return
        
    def get_parallax_vector(self, t_mjd):
        recalc_plx = True
        if self.plx_vector_cached is not None:
            if hasattr(t_mjd, "__len__"):
                if list(t_mjd) == list(self.plx_vector_cached[0]):
                    pvec = self.plx_vector_cached[1:]
                    recalc_plx = False
                elif all([t_mjd_i in self.plx_vector_cached[0] for t_mjd_i in t_mjd]):
                    pvec_idxs = [np.argwhere(self.plx_vector_cached[0]==t_mjd_i)[0][0] for t_mjd_i in t_mjd]
                    pvec = [self.plx_vector_cached[1][pvec_idxs], self.plx_vector_cached[2][pvec_idxs]]
                    recalc_plx = False
            elif t_mjd in self.plx_vector_cached[0]:
                idx = np.where(t_mjd==self.plx_vector_cached[0])[0][0]
                pvec = np.array([self.plx_vector_cached[1][idx], self.plx_vector_cached[2][idx]])
                recalc_plx = False
        if recalc_plx:
            pvec = parallax.parallax_in_direction(self.RA, self.Dec, t_mjd, obsLocation=self.obs, PA=self.PA).T
            if hasattr(t_mjd, "__len__"):
                self.plx_vector_cached = [t_mjd, pvec[0], pvec[1]]
        return pvec
        
    def get_pos_at_time(self, fit_params, fixed_params, t):
        fit_params_dict = dict(zip(self.fitter_param_names, fit_params))
        fixed_params_dict = dict(zip(self.fixed_param_names, fixed_params))
        dt = t - fixed_params_dict['t0']
        
        t_mjd = Time(t, format='decimalyear', scale='utc').mjd
        pvec = self.get_parallax_vector(t_mjd)
        pvec_x = np.reshape(pvec[0], t.shape)
        pvec_y = np.reshape(pvec[1], t.shape)
        x = fit_params_dict['x0'] + fit_params_dict['vx']*dt + fit_params_dict['pi']*pvec_x
        y = fit_params_dict['y0'] + fit_params_dict['vy']*dt + fit_params_dict['pi']*pvec_y
        return x, y

    def get_batch_pos_at_time(self, t,
                                x0=[],vx=[], y0=[],vy=[], pi=[], t0=[],
                                x0_err=[],vx_err=[], y0_err=[],vy_err=[], pi_err=[], **kwargs):
        t_mjd = Time(t, format='decimalyear', scale='utc').mjd
        pvec = self.get_parallax_vector(t_mjd)
        if hasattr(t, "__len__"):
            dt = t-t0[:,np.newaxis]
            x = x0[:,np.newaxis] + dt*vx[:,np.newaxis] + pi[:,np.newaxis]*pvec[0].T
            y = y0[:,np.newaxis] + dt*vy[:,np.newaxis] + pi[:,np.newaxis]*pvec[1].T
            try:
                x_err = np.sqrt(x0_err[:,np.newaxis]**2 + (vx_err[:,np.newaxis]*dt)**2 + (pi_err[:,np.newaxis]*pvec[0].T)**2)
                y_err = np.sqrt(y0_err[:,np.newaxis]**2 + (vy_err[:,np.newaxis]*dt)**2 + (pi_err[:,np.newaxis]*pvec[1].T)**2)
            except:
                x_err,y_err = [],[]
        else:
            dt = t-t0
            x = x0 + dt*vx + pi*pvec[0]
            y = y0 + dt*vy + pi*pvec[1]
            try:
                x_err = np.sqrt(x0_err**2 + (vx_err*dt)**2 + (pi_err*pvec[0])**2)
                y_err = np.sqrt(y0_err**2 + (vy_err*dt)**2 + (pi_err*pvec[1])**2)
            except:
                x_err,y_err = [],[]
        return x,y,x_err,y_err

    def run_fit(
        self, t, x, y, xe, ye, t0, 
        weighting='var', 
        use_scipy=True, 
        absolute_sigma=True, 
        params_guess=None, 
        fill_value=np.inf
    ):
        if not use_scipy:
            Warning("Parallax model has no non-scipy fitter option. Running with scipy.")
        t_mjd = Time(t, format='decimalyear', scale='utc').mjd
        pvec = self.get_parallax_vector(t_mjd)
        x_wt, y_wt = self.get_weights(xe,ye, weighting=weighting)
        def fit_func(use_t, x0,vx, y0,vy, pi):
            x_res = x0 + vx*(use_t-t0) + pi*pvec[0]
            y_res = y0 + vy*(use_t-t0) + pi*pvec[1]
            return np.hstack([x_res, y_res])
        # Initial guesses, x0,y0 as x,y averages;
        #     vx,vy as average velocity if first and last points are perfectly measured;
        #     pi for 10 pc disance
        if params_guess is None:
            idx_first, idx_last = np.argmin(t), np.argmax(t)
            params_guess = [x.mean(),(x[idx_last]-x[idx_first])/(t[idx_last]-t[idx_first]),
                            y.mean(),(y[idx_last]-y[idx_first])/(t[idx_last]-t[idx_first]), 0.1]
        res = curve_fit(fit_func, t, np.hstack([x,y]),
                        p0=params_guess, sigma = 1.0/np.hstack([x_wt,y_wt]))
        x0, vx, y0, vy, pi = res[0]
        x0_err, vx_err, y0_err, vy_err, pi_err = np.sqrt(np.diag(res[1]))

        params = [x0, vx, y0, vy, pi]
        param_errors = [x0_err, vx_err, y0_err, vy_err, pi_err]
        return params, param_errors
        
"""
Check that everything is set up properly for motion models to run and their
required metadata.
"""
def validate_motion_model_dict(motion_model_dict, startable, default_motion_model):
    # Collect names of all motion models that might get used.
    all_motion_model_names = ['Fixed']
    if default_motion_model is not None:
        all_motion_model_names.append(default_motion_model)
    if 'motion_model_input' in startable.columns:
        all_motion_model_names += np.unique(startable['motion_model_input']).tolist()
    if 'motion_model_used' in startable.columns:
        all_motion_model_names += np.unique(startable['motion_model_used']).tolist()
    all_motion_model_names = np.unique(all_motion_model_names)
    
    # Check whether all motion models are in the dict, and if not, try to add them
    #   here or raise an error.
    for mm in all_motion_model_names:
        if mm not in motion_model_dict:
            mm_obj = eval(mm)
            if len(mm_obj.fixed_meta_data)>0:
                raise ValueError(f"Cannot use {mm} motion model without required metadata. Please initialize with required metadata and provide in motion_model_dict.")
            else:
                motion_model_dict[mm] = mm_obj()
                # warnings.warn(f"Using default model/fitter for {mm}.", UserWarning)

    return motion_model_dict
    
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
    motion_model_map = {
        'Fixed': Fixed,
        'Linear': Linear,
        'Acceleration': Acceleration,
        'Parallax': Parallax
    }
    
    list_of_parameters = []
    # all_motion_models = [eval(mm) for mm in np.unique(motion_model_list).tolist()]
    for mm in range(len(motion_model_list)):
        motion_model = motion_model_map[motion_model_list[mm]]
        param_names = motion_model.fitter_param_names
        param_fixed_names = motion_model.fixed_param_names
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
    
