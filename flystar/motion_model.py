import numpy as np
from abc import ABC
import pdb
from flystar import parallax
from astropy.time import Time
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

class MotionModel(ABC):
    # Fit paramters: Shared fit parameters
    fit_param_names = []

    # Number of fit parameters/required observations in each direction
    n_params = int(np.ceil(len(fit_param_names) / 2))

    # Fixed parameters: These are parameters that are required for the model, but are not 
    # fit quantities. For example, RA and Dec in a parallax model.
    fixed_param_names = []
    fixed_meta_data = []

    # Non-fit paramters: Custom paramters that will not be fit.
    # These parameters should be derived from the fit parameters and
    # they must exist as a variable on the model object
    optional_param_names = []
    name = "MotionModel"

    def __init__(self, *args, **kwargs):
        return
    
    def model_fit(self, dt):
        return np.full_like(dt, np.nan)
    
    def model(self, t, fit_params, fixed_params=None, fit_param_errs=None):
        if fit_param_errs is None:
            return np.full_like(t, np.nan), np.full_like(t, np.nan)
        return np.full_like(t, np.nan), np.full_like(t, np.nan), np.full_like(t, np.inf), np.full_like(t, np.inf)

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

    def fit(
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
            Parameters, uncertainties, and chi squares. The corresponding parameter names are in self.fit_param_names.
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

    def calc_chi2(self, fit_params, fixed_params, t, x, y, xe, ye, reduced=False):
        """
        Get the chi^2 value for the current MM and
        the input data.
        """
        x_pred, y_pred = self.model(t, fit_params, fixed_params)
        chi2x = np.sum((x - x_pred)**2 / xe**2)
        chi2y = np.sum((y - y_pred)**2 / ye**2)
        if reduced:
            if len(t) == self.n_params:
                return np.inf, np.inf
            else:
                degree_of_freedom = len(x) - self.n_params
                chi2x, chi2y = chi2x / degree_of_freedom, chi2y / degree_of_freedom
        return chi2x, chi2y

class Empty(MotionModel):
    fit_param_names = []
    fixed_param_names = []
    name = "Empty"
    # Number of fit parameters/required observations in each direction
    n_params = int(np.ceil(len(fit_param_names) / 2))

    def __init__(self, **kwargs):
        """Empty motion model, returns nan for values and inf for uncertainties.
        """
        super().__init__()
        return
    
    def model_fit(self, dt):
        return np.full_like(dt, np.nan)

    def model(self, t, fit_params, fixed_params, fixed_param_errs=None):
        t = np.atleast_1d(t)
        return np.full_like(t, np.nan), np.full_like(t, np.nan), np.full_like(t, np.inf), np.full_like(t, np.inf)

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
    
    fit_param_names = ['x0','y0']
    fixed_param_names = []
    # Number of fit parameters/required observations in each direction
    n_params = int(np.ceil(len(fit_param_names) / 2))

    name = "Fixed"

    def __init__(self, **kwargs):
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()
        return

    def model_fit(self, dt, x0):
        """Fit function for Fixed motion model

        Parameters
        ----------
        dt : array-like
            Time offset, shape (N_times,)
        x0 : float or array-like
            Average positions, scalar or shape (N_stars,)
        
        Returns
        -------
        x : array-like
            Predicted positions, shape (N_times,) if scalar x0, else (N_stars, N_times)
        """
        dt = np.atleast_1d(dt)
        x0 = np.asarray(x0)
        return np.broadcast_to(x0[:, np.newaxis], (x0.shape[0], dt.shape[0])) if x0.ndim > 0 else np.full_like(dt, x0)

    def model(self, t, fit_params, fixed_params=None, fit_param_errs=None):
        """Predicted positions (and uncertainties, if fit_param_errs is provided) at time t of Fixed model.

        Parameters
        ----------
        t : float or array-like
            Time array, shape (N_times,)
        fit_params : array-like
            Fit parameters x0, y0 in shape (N_params,) or (N_stars, N_params)
        fixed_params : array-like, optional
            Not applicable for Fixed, by default None
        fit_param_errs : array-like, optional
            Uncertainties for x0, y0 in shape (N_params,) or (N_stars, N_params), by default None

        Returns
        -------
        x, y (, xe, ye)
            Predicted position (and uncertainties) of Fixed model, shape (N_stars, N_times), or (N_times,) if N_stars=1, or (N_stars,) if N_times=1
        """
        t = np.atleast_1d(t)
        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_params)

        N_stars = fit_params.shape[0] if fit_params.ndim > 1 else 1
        N_times = len(t)
        x0, y0 = fit_params.T  # Each shape (N_stars,)

        # Return results in (N_stars, N_times) shape
        x = self.model_fit(t, x0)  # Shape (N_stars, N_times)
        y = self.model_fit(t, y0)  # Shape (N_stars, N_times)

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x = x.flatten()
            y = y.flatten()

        if fit_param_errs is None:
            return x, y

        fit_param_errs = np.atleast_2d(fit_param_errs)  # (N_stars, N_params)
        x0_err, y0_err = fit_param_errs.T

        # Return results in (N_stars, N_times) shape
        x_err = np.broadcast_to(x0_err[:, np.newaxis], (N_stars, N_times))
        y_err = np.broadcast_to(y0_err[:, np.newaxis], (N_stars, N_times))

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x_err = x_err.flatten()
            y_err = y_err.flatten()

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
        x0e = (np.sum(x_wt_norm**2 * xe**2))**0.5  # Error propagation
        y0 = np.average(y, weights=y_wt)
        y0e = (np.sum(y_wt_norm**2 * ye**2))**0.5  # Error propagation

        params = np.array([x0, y0])
        param_errors = np.array([x0e, y0e])

        chi2x, chi2y = self.calc_chi2(params, [], t, x, y, xe, ye)

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
    fit_param_names = ['x0', 'vx', 'y0', 'vy']
    fixed_param_names = ['t0']

    # Number of fit parameters/required observations in each direction
    n_params = int(np.ceil(len(fit_param_names) / 2))

    name = "Linear"

    def __init__(self, **kwargs):
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()
        return

    def model_fit(self, dt, x0, v):
        """Linear motion model fit function

        Parameters
        ----------
        dt : array-like
            Time offset, shape (N_times,)
        x0 : float or array-like
            Initial position, shape (N_stars,) or scalar
        v : float or array-like
            Velocity, shape (N_stars,) or scalar

        Returns
        -------
        x : array-like
            Predicted position(s)
        """
        return x0 + v * dt

    def model(self, t, fit_params, fixed_params, fit_param_errs=None):
        """Model positions (and uncertainties, if fit_param_errs is provided) at time t of Linear model.

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to evaluate the model
        fit_params : array-like
            Fit parameters x0, vx, y0, vy in shape (N_params,) or (N_stars, N_params)
        fixed_params : array-like
            Fixed parameters t0 in shape (1,) or (N_stars, 1)
        fit_param_errs : array-like, optional
            Uncertainties of fit parameters in shape (N_params,) or (N_stars, N_params), by default None

        Returns
        -------
        x, y (, xe, ye)
            Predicted positions (and uncertainties, if fit_param_errs is provided) with shape (N_stars, N_times), or (N_times,) if N_stars=1, or (N_stars,) if N_times=1
        """
        t = np.atleast_1d(t)
        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_params)

        N_stars = fit_params.shape[0] if fit_params.ndim > 1 else 1
        N_times = len(t)

        x0, vx, y0, vy = fit_params.T  # Each shape (N_stars,)
        t0 = np.atleast_1d(fixed_params[0])  # Shape (N_stars,) or (1,)

        dt = t[np.newaxis, :] - t0[:, np.newaxis]  # Shape (N_stars, N_times)
        
        x = self.model_fit(dt, x0[:, np.newaxis], vx[:, np.newaxis])  # Shape (N_stars, N_times)
        y = self.model_fit(dt, y0[:, np.newaxis], vy[:, np.newaxis])  # Shape (N_stars, N_times)

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x = x.flatten()
            y = y.flatten()

        if fit_param_errs is None:
            return x, y
        
        fit_param_errs = np.atleast_2d(fit_param_errs)  # (N_stars, N_params)
        x0_err, vx_err, y0_err, vy_err = fit_param_errs.T   # Each shape (N_stars,)
        x_err = np.hypot(x0_err[:, np.newaxis], vx_err[:, np.newaxis] * dt)  # Shape (N_stars, N_times)
        y_err = np.hypot(y0_err[:, np.newaxis], vy_err[:, np.newaxis] * dt)  # Shape (N_stars, N_times)
        
        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x_err = x_err.flatten()
            y_err = y_err.flatten()
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
            x_opt, x_cov = curve_fit(self.model_fit, dt, x, p0=np.array(params_guess[:2]), sigma=1/x_wt**0.5, absolute_sigma=absolute_sigma)
            y_opt, y_cov = curve_fit(self.model_fit, dt, y, p0=np.array(params_guess[2:]), sigma=1/y_wt**0.5, absolute_sigma=absolute_sigma)
            x0, vx = x_opt
            y0, vy = y_opt
            x0e, vxe = np.sqrt(x_cov.diagonal())
            y0e, vye = np.sqrt(y_cov.diagonal())
            params = np.array([x0, vx, y0, vy])
            param_errors = np.array([x0e, vxe, y0e, vye])
            chi2_x, chi2_y = self.calc_chi2(params, [t0], t, x, y, xe, ye)
            
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
    fit_param_names = ['x0', 'vx0', 'ax', 'y0', 'vy0', 'ay']
    fixed_param_names = ['t0']
    name = "Acceleration"

    # Number of fit parameters/required observations in each direction
    n_params = int(np.ceil(len(fit_param_names) / 2))

    def __init__(self):
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()
        return
    
    def model_fit(self, t, x0, v0, a):
        """Model positions at time t of Acceleration model.

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to evaluate the model
        x0 : float or array-like
            Initial position(s)
        v0 : float or array-like
            Initial velocity(ies)
        a : float or array-like
            Acceleration(s)

        Returns
        -------
        float or array-like
            Model positions at time t of Acceleration model
        """
        return x0 + v0*t + 0.5*a*t**2

    def model(self, t, fit_params, fixed_params, fit_param_errs=None):
        """Model positions (and uncertainties, if fit_param_errs is provided) at time t of Acceleration model.

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to evaluate the model
        fit_params : array-like
            Fit parameters with shape (N_stars, N_params) or (N_params,)
        fixed_params : array-like
            Fixed parameters with shape (N_stars, N_fixed_params) or (N_fixed_params,)
        fit_param_errs : array-like, optional
            Fit parameter uncertainties with shape (N_stars, N_params) or (N_params,), by default None

        Returns
        -------
        x, y (, xe, ye)
            Predicted positions (and uncertainties, if fit_param_errs is provided) with shape (N_stars, N_times), or (N_times,) if N_stars=1, or (N_stars,) if N_times=1
        """
        t = np.atleast_1d(t)
        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_params)

        N_stars = fit_params.shape[0] if fit_params.ndim > 1 else 1
        N_times = len(t)
        
        x0, vx0, ax, y0, vy0, ay = fit_params.T  # Each shape (N_stars,)
        t0 = np.atleast_1d(fixed_params[0])  # Shape (N_stars,) or (1,)
        
        dt = t[np.newaxis, :] - t0[:, np.newaxis]  # Shape (N_stars, N_times)
        
        x = self.model_fit(dt, x0[:, np.newaxis], vx0[:, np.newaxis], ax[:, np.newaxis])  # Shape (N_stars, N_times)
        y = self.model_fit(dt, y0[:, np.newaxis], vy0[:, np.newaxis], ay[:, np.newaxis])  # Shape (N_stars, N_times)
        
        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x = x.flatten()
            y = y.flatten()
        
        if fit_param_errs is None:
            return x, y
        
        fit_param_errs = np.atleast_2d(fit_param_errs)  # (N_stars, N_params)
        x0_err, vx0_err, ax_err, y0_err, vy0_err, ay_err = fit_param_errs.T
        x_err = np.sqrt(x0_err[:, np.newaxis]**2 + (vx0_err[:, np.newaxis] * dt)**2 + (0.5 * ax_err[:, np.newaxis] * dt**2)**2)  # Shape (N_stars, N_times)
        y_err = np.sqrt(y0_err[:, np.newaxis]**2 + (vy0_err[:, np.newaxis] * dt)**2 + (0.5 * ay_err[:, np.newaxis] * dt**2)**2)  # Shape (N_stars, N_times)
        
        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x_err = x_err.flatten()
            y_err = y_err.flatten()

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

        x_opt, x_cov = curve_fit(self.model_fit, dt, x, p0=np.array(params_guess[:3]), sigma=1/x_wt**0.5, absolute_sigma=absolute_sigma)
        y_opt, y_cov = curve_fit(self.model_fit, dt, y, p0=np.array(params_guess[3:]), sigma=1/y_wt**0.5, absolute_sigma=absolute_sigma)
        x0, vx0, ax = x_opt
        y0, vy0, ay = y_opt
        x0e, vx0e, axe = np.sqrt(x_cov.diagonal())
        y0e, vy0e, aye = np.sqrt(y_cov.diagonal())

        params = np.array([x0, vx0, ax, y0, vy0, ay])
        param_errors = np.array([x0e, vx0e, axe, y0e, vy0e, aye])
        chi2_x, chi2_y = self.calc_chi2(params, [t0], t, x, y, xe, ye)

        return params, param_errors, chi2_x, chi2_y

class Parallax(MotionModel):
    """
    Motion model for linear proper motion + parallax
    
    Requires RA, Dec, and PA parameters (degrees) for parallax calculation.
        RA, Dec in J2000
    Optional PA is counterclockwise offset of the image y-axis from North.
    Optional obs parameter describes observer location, default is 'earth'.
    """
    fit_param_names = ['x0', 'vx', 'y0', 'vy', 'pi']
    fixed_param_names = ['t0']
    fixed_meta_data = ['RA','Dec','PA','obs']
    name = "Parallax"
        
    # Number of fit parameters/required observations in each direction
    n_params = int(np.ceil(len(fit_param_names) / 2))

    def __init__(self, ra, dec, pa=0., obs='earth'):
        super().__init__()
        self.ra = ra
        self.dec = dec
        self.pa = pa
        self.obs = obs
        self.plx_vector_cached = None
        return

    def calc_parallax_vector(self, t_mjd):
        """Calculate parallax vector of shape (2, N_times)

        Parameters
        ----------
        t_mjd : array-like
            Time array in mjd

        Returns
        -------
        pvec
            Parallax vector of shape (2, N_times)
        """
        if self.plx_vector_cached is not None:
            t_mjd = np.atleast_1d(t_mjd)
            t_mjd_cached = self.plx_vector_cached[0]
            if np.allclose(t_mjd, t_mjd_cached):
                # If cached values match input times, return cached values
                return self.plx_vector_cached[1]

            elif all(np.isin(t_mjd, t_mjd_cached)):
                # If all input times are in cached values, return those
                # Calculate pvec_idxs such that t_mjd_cached[ pvec_idxs ] == t_mjd
                pvec_idxs = np.array([np.where(t_mjd_cached==t_mjd_i)[0][0] for t_mjd_i in t_mjd])
                pvec = self.plx_vector_cached[1][:, pvec_idxs]
                return pvec

        pvec = parallax.parallax_in_direction(self.ra, self.dec, t_mjd, obsLocation=self.obs, PA=self.pa).T
        self.plx_vector_cached = [t_mjd, pvec]
        return pvec

    def model_fit(self, dt, x0, vx, y0, vy, pi):
        """Model positions at time t of Parallax model.

        Parameters
        ----------
        dt : float or array-like
            Time(s) at which to evaluate the model
        x0 : float or array-like
            Initial position(s)
        vx : float or array-like
            Velocity(ies)
        y0 : float or array-like
            Initial position(s)
        vy : float or array-like
            Velocity(ies)
        pi : float or array-like
            Parallax factor(s)

        Returns
        -------
        2d array
            Model positions at time t of Parallax model, shape (2, N_times)
        """
        x_res = x0 + vx*dt + pi * self.pvec[0]
        y_res = y0 + vy*dt + pi * self.pvec[1]
        return np.vstack([x_res, y_res])


    def model(self, t, fit_params, fixed_params, fit_param_errs=None):
        """Model positions (and uncertainties, if fit_param_errs is provided) at time t of Parallax model.

        Parameters
        ----------
        t : float or array-like
            Times at which to evaluate the model
        fit_params : array-like
            Fit parameters for the model
        fixed_params : array-like
            Fixed parameters for the model
        fit_param_errs : array-like, optional
            Uncertainties in fit parameters, by default None

        Returns
        -------
        x, y (, xe, ye)
            Predicted positions (and uncertainties, if fit_param_errs is provided) with shape (N_stars, N_times), or (N_times,) if N_stars=1, or (N_stars,) if N_times=1
        """
        t = np.atleast_1d(t)
        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_params)
        
        N_stars = fit_params.shape[0] if fit_params.ndim > 1 else 1
        N_times = len(t)

        x0, vx, y0, vy, pi = fit_params.T  # Each shape (N_stars,)
        t0 = np.atleast_1d(fixed_params[0])  # Shape (N_stars,) or (1,)

        dt = t[np.newaxis, :] - t0[:, np.newaxis]  # Shape (N_stars, N_times)
        t_mjd = Time(t, format='decimalyear', scale='utc').mjd  # Shape (N_times,)
        self.pvec = self.calc_parallax_vector(t_mjd) # Shape (2, N_times)
        x, y = self.model_fit(dt, x0[:, np.newaxis], vx[:, np.newaxis], y0[:, np.newaxis], vy[:, np.newaxis], pi[:, np.newaxis])  # Shape (N_stars, N_times)

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x = x.flatten()
            y = y.flatten()

        if fit_param_errs is None:
            return x, y

        fit_param_errs = np.atleast_2d(fit_param_errs)  # (N_stars, N_params)
        x0_err, vx_err, y0_err, vy_err, pi_err = fit_param_errs.T
        x_err = np.sqrt(x0_err[:, np.newaxis]**2 + (vx_err[:, np.newaxis] * dt)**2 + (pi_err[:, np.newaxis] * self.pvec[0][np.newaxis, :])**2)  # Shape (N_stars, N_times)
        y_err = np.sqrt(y0_err[:, np.newaxis]**2 + (vy_err[:, np.newaxis] * dt)**2 + (pi_err[:, np.newaxis] * self.pvec[1][np.newaxis, :])**2)  # Shape (N_stars, N_times)
        
        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x_err = x_err.flatten()
            y_err = y_err.flatten()
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
        
        t = np.atleast_1d(t)
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
        self.pvec = self.get_parallax_vector(t_mjd)
        x_wt, y_wt = self.get_weights(xe,ye, weighting=weighting)

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
            self.model_fit, t - t0, np.vstack([x, y]),
            p0=params_guess, sigma = 1.0/np.hstack([x_wt,y_wt]),
            absolute_sigma=absolute_sigma
        )
        x0, vx, y0, vy, pi = res[0]
        x0_err, vx_err, y0_err, vy_err, pi_err = np.sqrt(np.diag(res[1]))

        params = np.array([x0, vx, y0, vy, pi])
        param_errors = np.array([x0_err, vx_err, y0_err, vy_err, pi_err])
        chi2_x, chi2_y = self.calc_chi2(params, [t0], t, x, y, xe, ye)
        
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

    for param in motion_model.fit_param_names:
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
        for param in mm.fit_param_names:
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
    mm_map = {
        'Empty': Empty,
        'Fixed': Fixed,
        'Linear': Linear,
        'Acceleration': Acceleration,
        'Parallax': Parallax
    }
    
    # Sort by n_params
    mm_map = dict(sorted(mm_map.items(), key=lambda item: item[1].n_params))
    return mm_map