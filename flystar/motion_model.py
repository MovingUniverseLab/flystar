import numpy as np
from abc import ABC
import pdb
from flystar import parallax
from astropy.time import Time
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

class MotionModel(ABC):
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

    def get_pos_at_time(self, fit_params, fixed_params, t):
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
        fill_value=np.nan,
        verbose=True
    ):
        # Run a single fit (used both for overall fit + bootstrap iterations)
        return np.full(self.n_params, fill_value), np.full(self.n_params, np.inf), np.nan, np.nan

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
        fill_value=np.nan,
        verbose=True,
        seed=None
    ):
        """Fit stellar motion parameters

        Parameters
        ----------
        t : array-like
            Times of measurements
        x : array-like
            x-coordinates
        y : array-like
            y-coordinates
        xe : array-like
            Uncertainty of x
        ye : array-like
            Uncertainty of y
        t0 : array-like
            Reference time for fitting, i.e. dt = t - t0 will be used in fitting
        bootstrap : int, optional
            Bootstrapping uncertainties, by default 0
        weighting : str, optional
            Use standard error weighting ('std': w=1/xe, 1/ye) or variance weighting ('var': w=1/xe**2, 1/ye**2), by default 'var'
        use_scipy : bool, optional
            Use scipy for optmization. Otherwise, use linear algebraic solution (Linear model only), which is faster for < 300 epochs, by default True
        absolute_sigma : bool, optional
            Absolute sigma. See scipy.optimize.curve_fit for details, by default True
        fill_value : float, optional
            Fill value for parameters when not enough data points to fit model, by default np.nan
        verbose : bool, optional
            Print warning messages, by default True
        seed : int, optional
            Seed for the random number generator, by default None
        Returns
        -------
        params, params_err, chi2_x, chi2_y
            Parameters, uncertainties, and chi squares. The corresponding parameter names are in self.fitter_param_names.
        """
        params, param_errs, chi2_x, chi2_y = self.run_fit(
            t, x, y, xe, ye, t0=t0, 
            weighting=weighting,
            use_scipy=use_scipy, 
            absolute_sigma=absolute_sigma,
            fill_value=fill_value,
            verbose=verbose
        )
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        # Bootstrap errors
        n_obs = len(t)
        if bootstrap > 0 and n_obs > (self.n_params):
            edx = np.arange(n_obs, dtype=int)
            bb_params = []
            bb_params_errs = []
            for bb in range(bootstrap):
                bdx = rng.choice(edx, n_obs, replace=False)
                params_bdx, param_errs_bdx, chi2x_bdx, chi2y_bdx = self.run_fit(
                    t[bdx], x[bdx], y[bdx], xe[bdx], ye[bdx], t0=t0,
                    weighting=weighting, 
                    use_scipy=use_scipy, 
                    absolute_sigma=absolute_sigma, 
                    params_guess=params, 
                    fill_value=fill_value,
                    verbose=verbose
                )
                bb_params.append(params_bdx)
                bb_params_errs.append(param_errs_bdx)
        
            # Save the errors from the bootstrap
            param_errs = np.std(bb_params, axis=0)

            # Account for odd case
            inf_errs = [np.all(arr==np.inf) for arr in np.transpose(np.array(bb_params_errs))]
            param_errs[inf_errs] = 0.0

        return params, param_errs, chi2_x, chi2_y

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

class Empty(MotionModel):
    n_params = 0
    fitter_param_names = []
    fixed_param_names = []

    def __init__(self, **kwargs):
        """Empty motion model, returns nan for values and inf for uncertainties.
        """
        super().__init__()
        return
        
    def get_pos_at_time(self, fit_params, fixed_params, t):
        if hasattr(t, "__len__"):
            return np.full(len(t), np.nan), np.full(len(t), np.nan)
        else:
            return np.nan, np.nan

    def get_batch_pos_at_time(self,t,
                                x0=[],y0=[],t0=[],
                                x0_err=[], y0_err=[]):
        if hasattr(t, "__len__"):
            return np.full((len(x0), len(t)), np.nan), np.full((len(y0), len(t)), np.nan), np.full((len(x0), len(t)), np.nan), np.full((len(y0), len(t)), np.nan)
        else:
            return np.nan, np.nan, np.nan, np.nan

    def run_fit(
            self, t, x, y, xe, ye, t0, 
            weighting='var', 
            use_scipy=True, 
            absolute_sigma=True,
            fill_value=np.nan,
            verbose=True
    ):
        if verbose:
            warnings.warn(f"Empty data cannot be fit. Setting parameters to {fill_value} and uncertainties to np.inf.", OptimizeWarning, stacklevel=2)
        params = np.full(self.n_params, fill_value)
        param_errors = np.full(self.n_params, np.inf)
        return params, param_errors, np.nan, np.nan


class Fixed(MotionModel):
    """
    A non-moving motion model for a star on the sky.
    """
    
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
            return x0, y0, x0_err, y0_err
            
    def run_fit(
            self, t, x, y, xe, ye, t0, 
            weighting='var', 
            use_scipy=True, 
            absolute_sigma=True,
            params_guess=None,
            fill_value=np.nan,
            verbose=True
    ):
        if verbose and (not use_scipy):
            warnings.warn("Fixed model has no non-scipy fitter option. Running with scipy.")

        n_obs = len(t)
        degree_of_freedom = n_obs - self.n_params
        # Not enough data points to fit model
        if degree_of_freedom < 0:
            if verbose:
                warnings.warn(
                    f'Not enough data points to fit model. Setting parameters to {fill_value} and uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )
            params = np.full(self.n_params, fill_value)
            param_errors = np.full(self.n_params, np.inf)
            return params, param_errors, np.nan, np.nan
        
        # degree_of_freedom >= 0
        # Calculate weighted average position
        x_wt, y_wt = self.get_weights(xe, ye, weighting=weighting)
        x_wt_norm = x_wt / np.sum(x_wt)
        y_wt_norm = y_wt / np.sum(y_wt)
        x0 = np.average(x, weights=x_wt)
        x0e = (np.sum(x_wt_norm**2 * xe**2))**0.5 / n_obs  # Error propagation
        y0 = np.average(y, weights=y_wt)
        y0e = (np.sum(y_wt_norm**2 * ye**2))**0.5 / n_obs  # Error propagation

        params = np.array([x0, y0])
        param_errors = np.array([x0e, y0e])
        
        chi2x, chi2y = self.get_chi2(params, [], t, x, y, xe, ye)
        
        if not absolute_sigma:
            if degree_of_freedom > 0:
                reduced_chi2x = chi2x / degree_of_freedom
                reduced_chi2y = chi2y / degree_of_freedom

                param_errors[0] *= reduced_chi2x**0.5
                param_errors[1] *= reduced_chi2y**0.5
            else:
                # degree_of_freedom == 0, as < 0 case already handled above
                warnings.warn(
                    f'Degree of freedom < 0. Covariance of the parameters could not be estimated. Setting parameter uncertainties to fill value np.inf.',
                    OptimizeWarning, stacklevel=2
                )
                # Set parameter uncertainties to np.inf, same behavior as scipy.optimize.curve_fit
                param_errors = np.full_like(param_errors, np.inf)

        return params, param_errors, chi2x, chi2y

class Linear(MotionModel):
    """
    A 2D linear motion model for a star on the sky.
    """
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
        dt = t - fixed_params_dict['t0']
        return fit_params_dict['x0'] + fit_params_dict['vx']*dt, fit_params_dict['y0'] + fit_params_dict['vy']*dt

    def get_batch_pos_at_time(self, t, x0=[],vx=[], y0=[],vy=[], t0=[],
                x0_err=[],vx_err=[], y0_err=[],vy_err=[], **kwargs):
        if hasattr(t, "__len__"):
            dt = t - t0[:,np.newaxis]
            x = x0[:,np.newaxis] + dt*vx[:,np.newaxis]
            y = y0[:,np.newaxis] + dt*vy[:,np.newaxis]
            x_err = np.hypot(x0_err[:,np.newaxis], vx_err[:,np.newaxis]*dt)
            y_err = np.hypot(y0_err[:,np.newaxis], vy_err[:,np.newaxis]*dt)
        else:
            dt = t - t0
            x = x0 + dt*vx
            y = y0 + dt*vy
            x_err = np.hypot(x0_err, vx_err*dt)
            y_err = np.hypot(y0_err, vy_err*dt)
        return x, y, x_err, y_err

    def run_fit(
            self, t, x, y, xe, ye, t0, 
            weighting='var', 
            use_scipy=True, 
            absolute_sigma=True,
            params_guess=None,
            fill_value=np.nan,
            verbose=True
    ):
        n_obs = len(t)
        degree_of_freedom = n_obs - self.n_params
        # Not enough data points to fit model
        if degree_of_freedom < 0:
            if verbose:
                warnings.warn(
                    f'Not enough data points to fit model. Setting parameters to {fill_value} and uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )
            params = np.full(self.n_params, fill_value)
            param_errors = np.full(self.n_params, np.inf)
            return params, param_errors, np.nan, np.nan
        
        # degree_of_freedom >= 0
        dt = t - t0
        x_wt, y_wt = self.get_weights(xe, ye, weighting=weighting)
        if params_guess is None:
            params_guess = [x.mean(), 0., y.mean(), 0.]

        if use_scipy:
            def linear(t, c0, c1):
                return c0 + c1*t
            x_opt, x_cov = curve_fit(linear, dt, x, p0=np.array(params_guess[:2]), sigma=1/x_wt**0.5, absolute_sigma=absolute_sigma)
            y_opt, y_cov = curve_fit(linear, dt, y, p0=np.array(params_guess[2:]), sigma=1/y_wt**0.5, absolute_sigma=absolute_sigma)
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
                if degree_of_freedom > 0:
                    reduced_chi2_x = chi2_x / degree_of_freedom
                    reduced_chi2_y = chi2_y / degree_of_freedom
                    
                    param_errors[0:2] *= reduced_chi2_x**0.5
                    param_errors[2:4] *= reduced_chi2_y**0.5

                else:
                    # degree_of_freedom == 0, as < 0 case already handled above
                    warnings.warn(
                        f'Degree of freedom < 0. Covariance of the parameters could not be estimated. Setting parameter uncertainties to fill value np.inf.',
                        OptimizeWarning, stacklevel=2
                    )
                    # Set parameter uncertainties to np.inf, same behavior as scipy.optimize.curve_fit
                    param_errors = np.full_like(param_errors, np.inf)

        return params, param_errors, chi2_x, chi2_y

class Acceleration(MotionModel):
    """
    A 2D accelerating motion model for a star on the sky.
    """
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
        dt = t - fixed_params_dict['t0']
        x = fit_params_dict['x0'] + fit_params_dict['vx0']*dt + 0.5*fit_params_dict['ax']*dt**2
        y = fit_params_dict['y0'] + fit_params_dict['vy0']*dt + 0.5*fit_params_dict['ay']*dt**2
        return x, y
        
    def get_batch_pos_at_time(self,t,
                                x0=[],vx0=[],ax=[], y0=[],vy0=[],ay=[], t0=[],
                                x0_err=[],vx0_err=[],ax_err=[], y0_err=[],vy0_err=[],ay_err=[], **kwargs):
        if hasattr(t, "__len__"):
            dt = t - t0[:,np.newaxis]
            x = x0[:, np.newaxis] + dt * vx0[:, np.newaxis] + 0.5 * ax[:, np.newaxis] * dt**2
            y = y0[:, np.newaxis] + dt * vy0[:, np.newaxis] + 0.5 * ay[:, np.newaxis] * dt**2
            x_err = np.sqrt(x0_err[:, np.newaxis]**2 + (vx0_err[:, np.newaxis]*dt)**2 + (0.5*ax_err[:, np.newaxis]*dt**2)**2)
            y_err = np.sqrt(y0_err[:, np.newaxis]**2 + (vy0_err[:, np.newaxis]*dt)**2 + (0.5*ay_err[:, np.newaxis]*dt**2)**2)
        else:
            dt = t - t0
            x = x0 + dt * vx0 + 0.5 * ax * dt**2
            y = y0 + dt * vy0 + 0.5 * ay * dt**2
            x_err = np.sqrt(x0_err**2 + (vx0_err * dt)**2 + (0.5 * ax_err * dt**2)**2)
            y_err = np.sqrt(y0_err**2 + (vy0_err * dt)**2 + (0.5 * ay_err * dt**2)**2)
        return x, y, x_err, y_err

    def run_fit(
        self, t, x, y, xe, ye, t0, 
        weighting='var', 
        use_scipy=True, 
        absolute_sigma=True, 
        params_guess=None,
        fill_value=np.nan,
        verbose=True
    ):
        if not use_scipy:
            if verbose:
                warnings.warn("Acceleration model has no non-scipy fitter option. Running with scipy.")
        
        n_obs = len(t)
        degree_of_freedom = n_obs - self.n_params
        # Not enough data points to fit model
        if degree_of_freedom < 0:
            if verbose:
                warnings.warn(
                    f'Not enough data points to fit model. Setting parameters to {fill_value} and uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )
            params = np.full(self.n_params, fill_value)
            param_errors = np.full(self.n_params, np.inf)
            return params, param_errors, np.nan, np.nan

        # degree_of_freedom >= 0
        dt = t - t0
        x_wt, y_wt = self.get_weights(xe,ye, weighting=weighting)
        if params_guess is None:
            # Initial guess for velocity:
            idx_first, idx_last = np.argmin(t), np.argmax(t)
            t_span = t[idx_last] - t[idx_first]
            params_guess = [x.mean(), (x[idx_last] - x[idx_first]) / t_span, 0., y.mean(), (y[idx_last] - y[idx_first]) / t_span, 0.]

        def accel(t, c0, c1, c2):
            return c0 + c1*t + 0.5*c2*t**2

        x_opt, x_cov = curve_fit(accel, dt, x, p0=np.array(params_guess[:3]), sigma=1/x_wt**0.5, absolute_sigma=absolute_sigma)
        y_opt, y_cov = curve_fit(accel, dt, y, p0=np.array(params_guess[3:]), sigma=1/y_wt**0.5, absolute_sigma=absolute_sigma)
        x0, vx0, ax = x_opt
        y0, vy0, ay = y_opt
        x0e, vx0e, axe = np.sqrt(x_cov.diagonal())
        y0e, vy0e, aye = np.sqrt(y_cov.diagonal())

        params = np.array([x0, vx0, ax, y0, vy0, ay])
        param_errors = np.array([x0e, vx0e, axe, y0e, vy0e, aye])
        chi2_x, chi2_y = self.get_chi2(params, [t0], t, x, y, xe, ye)

        return params, param_errors, chi2_x, chi2_y

class Parallax(MotionModel):
    """
    Motion model for linear proper motion + parallax
    
    Requires RA, Dec, and PA parameters (degrees) for parallax calculation.
        RA, Dec in J2000
    Optional PA is counterclockwise offset of the image y-axis from North.
    Optional obs parameter describes observer location, default is 'earth'.
    """
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
        return x, y, x_err, y_err

    def run_fit(
        self, t, x, y, xe, ye, t0, 
        weighting='var', 
        use_scipy=True, 
        absolute_sigma=True, 
        params_guess=None, 
        fill_value=np.nan,
        verbose=True
    ):
        if not use_scipy:
            if verbose:
                warnings.warn("Parallax model has no non-scipy fitter option. Running with scipy.", UserWarning)
        
        n_obs = len(t)
        degree_of_freedom = n_obs - self.n_params
        # Not enough data points to fit model
        if degree_of_freedom < 0:
            if verbose:
                warnings.warn(
                    f'Not enough data points to fit model. Setting parameters to {fill_value} and uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )
            params = np.full(self.n_params, fill_value)
            param_errors = np.full(self.n_params, np.inf)
            return params, param_errors
        
        # degree_of_freedom >= 0
        t_mjd = Time(t, format='decimalyear', scale='utc').mjd
        pvec = self.get_parallax_vector(t_mjd)
        x_wt, y_wt = self.get_weights(xe,ye, weighting=weighting)
        def fit_func(use_t, x0,vx, y0,vy, pi):
            x_res = x0 + vx*(use_t-t0) + pi*pvec[0]
            y_res = y0 + vy*(use_t-t0) + pi*pvec[1]
            return np.hstack([x_res, y_res])
        # Initial guesses, x0,y0 as x,y averages;
        #     vx,vy as average velocity if first and last points are perfectly measured;
        #     pi for 10 pc distance
        if params_guess is None:
            idx_first, idx_last = np.argmin(t), np.argmax(t)
            t_span = t[idx_last] - t[idx_first]
            params_guess = [
                x.mean(), (x[idx_last] - x[idx_first]) / t_span,
                y.mean(), (y[idx_last] - y[idx_first]) / t_span, 
                0.1
            ]
        res = curve_fit(
            fit_func, t, np.hstack([x,y]),
            p0=params_guess, sigma = 1.0/np.hstack([x_wt,y_wt]),
            absolute_sigma=absolute_sigma
        )
        x0, vx, y0, vy, pi = res[0]
        x0_err, vx_err, y0_err, vy_err, pi_err = np.sqrt(np.diag(res[1]))

        params = np.array([x0, vx, y0, vy, pi])
        param_errors = np.array([x0_err, vx_err, y0_err, vy_err, pi_err])
        chi2_x, chi2_y = self.get_chi2(params, [t0], t, x, y, xe, ye)
        
        return params, param_errors, chi2_x, chi2_y


def validate_motion_models(motion_models, startable, default_motion_model):
    """Validate that all the unique motion models in startable and default_motion_model are in the motion_models. If not, add available models to the list.

    Parameters
    ----------
    motion_models : list of MotionModels
        List of MotionModels that are expected to encompass all the motion models
    startable : StarTable
        Star table that possibly contains 'motion_model_input' and 'motion_model_used'
    default_motion_model : MotionModel
        Default MotionModel
    """
    motion_model_map = {
        'Fixed': Fixed,
        'Linear': Linear,
        'Acceleration': Acceleration,
        'Parallax': Parallax
    }
    # Collect names of all motion models that might get used.
    all_motion_model_names = set()
    all_motion_model_names.add('Fixed')
    if default_motion_model is not None:
        all_motion_model_names.add(default_motion_model.__name__)
    if 'motion_model_input' in startable.colnames:
        all_motion_model_names.update(startable['motion_model_input'].tolist())
    if 'motion_model_used' in startable.colnames:
        all_motion_model_names.update(startable['motion_model_used'].tolist())

    # Check whether all motion models are in the list, and if not, raise an error.
    all_motion_models = [motion_model_map[mm] for mm in all_motion_model_names]
    for mm in all_motion_models:
        if mm not in motion_models:
            if len(mm.fixed_meta_data) > 0:
                raise ValueError(f"Cannot use {mm} motion model without required metadata. Please initialize with required metadata and provide in motion_models.")
            else:
                motion_models.append(mm)
                warnings.warn(f"{mm} not found in motion_models list. Added default instance.", UserWarning)

    return motion_models


def get_one_motion_model_param_names(motion_model, with_errors=True, with_fixed=True):
    """Get all the motion model parameters

    Parameters
    ----------
    motion_model : MotionModel
        MotionModel instance
    with_errors : bool, optional
        Add uncertainty names with '_err' suffix or not, by default True
    with_fixed : bool, optional
        Add fixed param names with '_fixed' suffix or not, by default True
    
    Returns
    -------
    list
        List of all parameter names for the motion model
    """
    list_of_parameters = []
    
    def list_add(name):
        if name not in list_of_parameters:
            list_of_parameters.append(name)

    for param in motion_model.fitter_param_names:
        # Fitter params
        list_add(param)
        # Error params
        if with_errors:
            list_add(param + '_err')
    # Fixed params
    if with_fixed:
        for param in motion_model.fixed_param_names:
            list_add(param)
    return list_of_parameters


def get_list_motion_model_param_names(motion_model_list, with_errors=True, with_fixed=True):
    """Get all the motion model parameters

    Parameters
    ----------
    motion_model_list : list
        List of MotionModels
    with_errors : bool, optional
        Add uncertainty names with '_err' suffix or not, by default True
    with_fixed : bool, optional
        Add fixed param names with '_fixed' suffix or not, by default True
    
    Returns
    -------
    list
        List of all unique parameter names across all motion models
    """
    list_of_parameters = []

    def list_add(name):
        if name not in list_of_parameters:
            list_of_parameters.append(name)

    for mm in motion_model_list:
        for param in mm.fitter_param_names:
            # Fitter params
            list_add(param)
            # Error params
            if with_errors:
                list_add(param + '_err')
        # Fixed params
        if with_fixed:
            for param in mm.fixed_param_names:
                list_add(param)
    return list(list_of_parameters)


def get_all_motion_model_names(with_errors=True, with_fixed=True):
    return get_list_motion_model_param_names(MotionModel.__subclasses__(), with_errors=with_errors, with_fixed=with_fixed)

def motion_model_map():
    return {
        'Empty': Empty,
        'Fixed': Fixed,
        'Linear': Linear,
        'Acceleration': Acceleration,
        'Parallax': Parallax
    }