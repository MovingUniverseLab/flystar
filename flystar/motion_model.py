import numpy as np
from abc import ABC
from flystar import parallax
from astropy.time import Time
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

class MotionModel(ABC):
    name = "MotionModel"

    # Fit paramters: Shared fit parameters
    fit_param_names = []
    n_fit_params = len(fit_param_names)
    # Number of fit parameters/required observations in each direction
    n_params = int((n_fit_params + 1) / 2)

    # Fixed parameters: These are parameters that are required for the model, but are not
    # fit quantities. For example, RA and Dec in a parallax model.
    fixed_param_names = []
    required_fixed_param_names = []
    optional_fixed_params = {}

    fixed_meta_data = []

    # Non-fit paramters: Custom paramters that will not be fit.
    # These parameters should be derived from the fit parameters and
    # they must exist as a variable on the model object

    def __init__(self, *args, **kwargs):
        """
        Make a motion model object. This object defines the fit and fixed parameters,
        and contains functions to fit the model to data and infer positions at given times.
        Each instance corresponds to a given motion model, not an individual star,
        and thus the fit values are only input/returned in functions, not stored in the object.
        """
        return

    def _check_param_dimensions(self, fit_params, fit_params_errs, fixed_params_dict):
        """Check that parameters is either a scalar or length of N_stars

        Parameters
        ----------
        fit_params: array-like
            Fit parameters, shape (N_fit_params,) or (n_stars, N_fit_params)
        fit_params_errs: array-like
            Errors of fit parameters, shape (N_fit_params,) or (n_stars, N_fit_params)
        fixed_params_dict : dict
            Dictionary of fixed parameters
        """
        N_stars = fit_params.shape[0] if fit_params.ndim > 1 else 1
        if fit_params_errs is not None:
            assert fit_params_errs.shape == fit_params.shape, "fit_params and fit_params_errs must have the same shape!"

        if fixed_params_dict is not None:
            for key, value in fixed_params_dict.items():
                # assert key in fixed_params_dict, f"Missing fixed parameter {key} in fixed_params_dict!"
                value = fixed_params_dict[key]
                if np.isscalar(value):
                    continue
                else:
                    assert len(value) == N_stars, f"Length of fixed parameter {key} must be either 1 or N_stars={N_stars}!"

    def model_fit(self, dt):
        return np.full_like(dt, np.nan)

    def model(self, t, fit_params, fit_param_errs=None, fixed_params_dict=None):
        self._check_param_dimensions(fit_params, fit_param_errs, fixed_params_dict)
        if fit_param_errs is None:
            return np.full_like(t, np.nan), np.full_like(t, np.nan)
        return np.full_like(t, np.nan), np.full_like(t, np.nan), np.full_like(t, np.inf), np.full_like(t, np.inf)

    def run_fit(
        self, t, x, y, xe, ye,
        fixed_params_dict=None,
        weighting='var',
        use_scipy=True,
        absolute_sigma=True,
        params_guess=None,
        fill_value=np.nan,
        return_chi2=False,
        verbose=True
    ):
        # Run a single fit (used both for overall fit + bootstrap iterations)
        if return_chi2:
            return np.full(self.n_fit_params, fill_value), np.full(self.n_fit_params, np.inf), np.nan, np.nan
        return np.full(self.n_fit_params, fill_value), np.full(self.n_fit_params, np.inf)

    def calc_weights(self, xe, ye, weighting='var'):
        if weighting=='std':
            return 1./xe, 1./ye
        elif weighting=='var':
            return 1./xe**2, 1./ye**2
        else:
            warnings.warn("Invalid weighting, using default weighting scheme var.", UserWarning)
            return 1./xe**2, 1./ye**2

    def fit(
        self, t, x, y, xe, ye,
        fixed_params_dict=None,
        weighting='var',
        use_scipy=True,
        absolute_sigma=True,
        fill_value=np.nan,
        params_guess=None,
        return_chi2=False,
        bootstrap=0,
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
        fixed_params_dict : dict, optional
            Dictionary of fixed parameters, see each motion model's fixed_param_names for details, by default None
        weighting : str, optional
            Use standard error weighting ('std': w=1/xe, 1/ye) or variance weighting ('var': w=1/xe**2, 1/ye**2), by default 'var'
        use_scipy : bool, optional
            Use scipy for optimization. Otherwise, use linear algebraic solution (Linear model only), which is faster for < 300 epochs, by default True
        absolute_sigma : bool, optional
            Absolute sigma. See scipy.optimize.curve_fit for details, by default True
        fill_value : float, optional
            Fill value for parameters when not enough data points to fit model, by default np.nan
        params_guess : array-like, optional
            Initial guess for the fit parameters used in scipy curve_fit, by default None
        return_chi2 : bool, optional
            Return chi^2 values along with parameters and uncertainties in params, param_errs, chi2_x, chi2_y, by default False
        bootstrap : int, optional
            Bootstrapping uncertainties, by default 0
        verbose : bool, optional
            Print warning messages, by default True
        seed : int, optional
            Seed for the random number generator, by default None
        Returns
        -------
        params, param_errs(, chi2_x, chi2_y)
            Parameters, uncertainties, and chi squares if return_chi2 is True. The corresponding parameter names are in self.fit_param_names.
        """
        assert np.ndim(t) == np.ndim(x) == np.ndim(y) == np.ndim(xe) == np.ndim(ye) == 1, "Input arrays must be 1D! Motion model can only fit individual stars"
        assert len(t) == len(x) == len(y) == len(xe) == len(ye), "Input arrays must have the same length!"

        if not verbose:
            warnings.filterwarnings("ignore", category=OptimizeWarning)

        fit_result = self.run_fit(
            t, x, y, xe, ye,
            fixed_params_dict=fixed_params_dict,
            weighting=weighting,
            use_scipy=use_scipy,
            absolute_sigma=absolute_sigma,
            fill_value=fill_value,
            params_guess=params_guess,
            return_chi2=return_chi2,
            verbose=verbose
        )

        if return_chi2:
            params, param_errs, chi2_x, chi2_y = fit_result
        else:
            params, param_errs = fit_result


        # Bootstrap errors
        n_obs = len(t)

        if (bootstrap > 0) and (n_obs > self.n_params):
            rng = np.random.default_rng(seed)
            edx = np.arange(n_obs, dtype=int)
            # Precompute All Bootstrap Draws at Once
            # Ensure there are enough unique points in each bootstrap sample
            bdx_unique = np.stack([
                rng.choice(edx, size=self.n_params, replace=False)
                for _ in range(bootstrap)
            ])
            # Draw with replacement for the rest
            bdx_extra = np.stack([
                rng.choice(edx, size=n_obs - self.n_params, replace=True)
                for _ in range(bootstrap)
            ])
            bdx_all = np.hstack((bdx_unique, bdx_extra))

            bb_params = []
            bb_params_errs = []
            for bdx in bdx_all:
                params_bdx, param_errs_bdx = self.run_fit(
                    t[bdx], x[bdx], y[bdx], xe[bdx], ye[bdx],
                    fixed_params_dict=fixed_params_dict,
                    weighting=weighting,
                    use_scipy=use_scipy,
                    absolute_sigma=absolute_sigma,
                    params_guess=params,
                    fill_value=fill_value,
                    return_chi2=False,
                    verbose=verbose
                )
                bb_params.append(params_bdx)
                bb_params_errs.append(param_errs_bdx)

            # Save the errors from the bootstrap
            param_errs = np.std(bb_params, axis=0)

            # Account for odd case
            inf_errs = [np.all(arr==np.inf) for arr in np.transpose(np.array(bb_params_errs))]
            param_errs[inf_errs] = 0.0

        if not verbose:
            warnings.resetwarnings()

        if return_chi2:
            return params, param_errs, chi2_x, chi2_y
        else:
            return params, param_errs


    # def calc_chi2(self, dt, x, y, x_wt, y_wt, popt_x, popt_y, reduced=False, parallax=False):
    #     X_mat_t = np.vander(dt, 2)
    #     residual_x = x - X_mat_t @ popt_x
    #     residual_y = y - X_mat_t @ popt_y

    #     W_mat_x = np.diag(x_wt)
    #     W_mat_y = np.diag(y_wt)

    #     chi2_x = residual_x.T @ W_mat_x @ residual_x
    #     chi2_y = residual_y.T @ W_mat_y @ residual_y

    #     if reduced:
    #         if len(dt) == self.n_params:
    #             return np.inf, np.inf
    #         if not parallax:
    #             degree_of_freedom = len(x) - self.n_params
    #         else:
    #             degree_of_freedom = 2*len(x) - len(self.fit_param_names)
    #         chi2_x, chi2_y = chi2_x / degree_of_freedom, chi2_y / degree_of_freedom
    #     return chi2_x, chi2_y

    def calc_chi2(self, t, x, y, xe, ye, fit_params, fixed_params_dict=None, reduced=False, parallax=False):
        """
        Get the chi^2 value for the input motion model parameters and data.
        """
        x_pred, y_pred = self.model(t, fit_params, fixed_params_dict=fixed_params_dict)
        chi2x = np.sum((x - x_pred)**2 / xe**2)
        chi2y = np.sum((y - y_pred)**2 / ye**2)
        if reduced:
            if len(t) == self.n_params:
                return np.inf, np.inf
            if not parallax:
                degree_of_freedom = len(x) - self.n_params
            else:
                degree_of_freedom = 2*len(x) - len(self.fit_param_names)
            chi2x, chi2y = chi2x / degree_of_freedom, chi2y / degree_of_freedom
        return chi2x, chi2y

class Empty(MotionModel):
    name = "Empty"
    fit_param_names = []
    fixed_param_names = []
    required_fixed_param_names = []
    optional_fixed_params = {}

    n_fit_params = len(fit_param_names)
    # Number of fit parameters/required observations in each direction
    n_params = int((n_fit_params + 1) / 2)

    def __init__(self, **kwargs):
        """Empty motion model, returns nan for values and inf for uncertainties.
        """
        super().__init__()
        return

    def model_fit(self, dt):
        return np.full_like(dt, np.nan)

    def model(self, t, fit_params, fit_param_errs=None, fixed_params_dict=None):
        """Predicted positions (and uncertainties, if fit_param_errs is provided) at time t of Empty model.

        Parameters
        ----------
        t : float or array-like
            Time array, shape (N_times,)
        fit_params : array-like
            Fit parameters, shape (N_fit_params,) or (N_stars, N_fit_params)
        fit_param_errs : array-like, optional
            Uncertainties for fit parameters, not applicable for Empty model, by default None
        fixed_params_dict : dict, optional
            Not applicable for Empty model, by default None

        Returns
        -------
        x, y (, xe, ye)
            Predicted position (and uncertainties) of Empty model, shape (N_times,)
        """
        self._check_param_dimensions(fit_params, fit_param_errs, fixed_params_dict)

        t = np.atleast_1d(t)
        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_fit_params)

        N_stars = fit_params.shape[0]
        N_times = len(t)

        if N_times == N_stars or N_times == 1 or N_stars == 1:
            # Assume each time corresponds to each star, so N_times = 1
            x = np.full(N_stars, np.nan)
            y = np.full(N_stars, np.nan)
        else:
            x = np.full((N_stars, N_times), np.nan)
            y = np.full((N_stars, N_times), np.nan)

        if fit_param_errs is None:
            return x, y
        return x, y, np.full_like(x, np.inf), np.full_like(y, np.inf)

    def run_fit(
        self, t, x, y, xe, ye,
        fixed_params_dict=None,
        weighting='var',
        use_scipy=True,
        absolute_sigma=True,
        params_guess=None,
        fill_value=np.nan,
        return_chi2=False,
        verbose=True
    ):
        """Fit stellar motion parameters

        Parameters
        ----------
        t : float or array-like
            Time array, shape (N_times,)
        x : array-like
            Observed x positions, shape (N_times,)
        y : array-like
            Observed y positions, shape (N_times,)
        xe : array-like
            Observed uncertainties in x positions, shape (N_times,)
        ye : array-like
            Observed uncertainties in y positions, shape (N_times,)
        fixed_params_dict : dict, optional
            Dictionary of fixed parameters, not applicable for Empty model, by default None
        weighting : str, optional
            Weighting scheme to use, 'var' or 'std', by default 'var'
        use_scipy : bool, optional
            Whether to use scipy.optimize for fitting, by default True
        absolute_sigma : bool, optional
            Whether to treat sigma as absolute, by default True
        fill_value : float, optional
            Value to fill parameters with when fitting is not possible, by default np.nan
        params_guess : array-like, optional
            Initial guess for parameters, by default None
        return_chi2 : bool, optional
            Whether to return chi-squared value, by default False
        verbose : bool, optional
            Whether to print verbose output, by default True

        Returns
        -------
        params, param_errors (, chi2_x, chi2_y)
            Fitted parameters, their uncertainties, and optionally chi-squared values
        """
        self.fixed_params_dict = fixed_params_dict
        if verbose:
            warnings.warn(f"Empty data cannot be fit. Setting parameters to {fill_value} and uncertainties to np.inf.", OptimizeWarning, stacklevel=2)
        params = np.full(self.n_fit_params, fill_value)
        param_errors = np.full(self.n_fit_params, np.inf)
        if return_chi2:
            return params, param_errors, np.nan, np.nan
        else:
            return params, param_errors


class Fixed(MotionModel):
    """
    A non-moving motion model for a star on the sky.
    """
    name = "Fixed"
    fit_param_names = ['x0','y0']
    fixed_param_names = []
    required_fixed_param_names = []
    optional_fixed_params = {}

    n_fit_params = len(fit_param_names)
    # Number of fit parameters/required observations in each direction
    n_params = int((n_fit_params + 1) / 2)

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
        return x0 + np.zeros_like(x0) * dt

    def model(self, t, fit_params, fit_param_errs=None, fixed_params_dict=None):
        """Predicted positions (and uncertainties, if fit_param_errs is provided) at time t of Fixed model.

        Parameters
        ----------
        t : float or array-like
            Time array, shape (N_times,)
        fit_params : array-like
            x0, y0 in shape (N_fit_params,) or (N_stars, N_fit_params)
        fit_param_errs : array-like, optional
            Uncertainties for x0, y0 in shape (N_fit_params,) or (N_stars, N_fit_params), by default None
        fixed_params_dict : dict, optional
            Not applicable for Fixed, by default None


        Returns
        -------
        x, y (, xe, ye)
            Predicted position (and uncertainties) of Fixed model, shape (N_stars, N_times), or (N_times,) if N_stars=1, or (N_stars,) if N_times=1
        """
        self.fixed_params_dict = fixed_params_dict
        t = np.atleast_1d(t)
        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_fit_params)
        self._check_param_dimensions(fit_params, fit_param_errs, fixed_params_dict)

        N_stars = fit_params.shape[0]
        N_times = len(t)
        x0, y0 = fit_params.T  # Each shape (N_stars,)

        # FIXME: Do we want this assumption?
        if N_times == N_stars:
            # Assume each time corresponds to each star, so N_times = 1
            dt = t[:, np.newaxis]  # Shape (N_stars, 1)
            N_times = 1
        else:
            # Else, calculate each time for each star
            dt = t[np.newaxis, :]  - np.zeros(N_stars)[:, np.newaxis]  # Shape (N_stars, N_times)

        # Return results in (N_stars, N_times) shape
        x = self.model_fit(t, x0[:, np.newaxis])  # Shape (N_stars, N_times)
        y = self.model_fit(t, y0[:, np.newaxis])  # Shape (N_stars, N_times)

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x = x.flatten()
            y = y.flatten()

        if fit_param_errs is None:
            return x, y

        fit_param_errs = np.atleast_2d(fit_param_errs)  # (N_stars, N_fit_params)
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
        self, t, x, y, xe, ye,
        fixed_params_dict=None,
        weighting='var',
        use_scipy=True,
        absolute_sigma=True,
        params_guess=None,
        fill_value=np.nan,
        return_chi2=False,
        verbose=True
    ):
        if verbose and (not use_scipy):
            warnings.warn("Fixed model has no non-scipy fitter option. Running with scipy.")

        n_obs = len(t)
        degree_of_freedom = n_obs - self.n_params
        # Not enough data points to fit model
        if degree_of_freedom < 0:
            warnings.warn(
                f'Not enough data points to fit model. Setting parameters to {fill_value} and uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )
            params = np.full(self.n_fit_params, fill_value)
            param_errors = np.full(self.n_fit_params, np.inf)
            return params, param_errors, np.nan, np.nan

        # degree_of_freedom >= 0
        # Calculate weighted average position
        x_wt, y_wt = self.calc_weights(xe, ye, weighting=weighting)
        x_wt_norm = x_wt / np.sum(x_wt)
        y_wt_norm = y_wt / np.sum(y_wt)
        x0 = np.average(x, weights=x_wt)
        x0e = (np.sum(x_wt_norm**2 * xe**2))**0.5  # Error propagation
        y0 = np.average(y, weights=y_wt)
        y0e = (np.sum(y_wt_norm**2 * ye**2))**0.5  # Error propagation

        params = np.array([x0, y0])
        param_errors = np.array([x0e, y0e])

        if (not absolute_sigma) or return_chi2:
            chi2x, chi2y = self.calc_chi2(t, x, y, xe, ye, params)

        if not absolute_sigma:
            if degree_of_freedom > 0:
                reduced_chi2x = chi2x / degree_of_freedom
                reduced_chi2y = chi2y / degree_of_freedom

                param_errors[0] *= reduced_chi2x**0.5
                param_errors[1] *= reduced_chi2y**0.5
            else:
                # degree_of_freedom == 0, as < 0 case already handled above
                warnings.warn(
                    f'Degree of freedom < 0. Covariance of the parameters could not be estimated. Setting parameter uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )
                # Set parameter uncertainties to np.inf, same behavior as scipy.optimize.curve_fit
                param_errors = np.full_like(param_errors, np.inf)

        if return_chi2:
            return params, param_errors, chi2x, chi2y
        else:
            return params, param_errors

class Linear(MotionModel):
    """
    A 2D linear motion model for a star on the sky.
    """
    name = "Linear"
    fit_param_names = ['x0', 'vx', 'y0', 'vy']
    required_fixed_param_names = ['t0']
    optional_fixed_params = {}
    fixed_param_names = required_fixed_param_names + list(optional_fixed_params.keys())

    n_fit_params = len(fit_param_names)
    # Number of fit parameters/required observations in each direction
    n_params = int((n_fit_params + 1) / 2)

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

    def model(self, t, fit_params, fit_param_errs=None, fixed_params_dict=None):
        """Model positions (and uncertainties, if fit_param_errs is provided) at time t of Linear model.

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to evaluate the model
        fit_params : array-like
            x0, vx, y0, vy in shape (N_fit_params,) or (N_stars, N_fit_params)
        fit_param_errs : array-like, optional
            Uncertainties of fit parameters in shape (N_fit_params,) or (N_stars, N_fit_params), by default None
        fixed_params_dict : dict
            t0, shape (1,) or (N_stars,)

        Returns
        -------
        x, y (, xe, ye)
            Predicted positions (and uncertainties, if fit_param_errs is provided) with shape (N_stars, N_times), or (N_times,) if N_stars=1, or (N_stars,) if N_times=1
        """
        if fixed_params_dict is None:
            fixed_params_dict = self.fixed_params_dict
        assert 't0' in fixed_params_dict, "Fixed parameter t0 is required for Linear model."
        self._check_param_dimensions(fit_params, fit_param_errs, fixed_params_dict)

        t = np.atleast_1d(t)
        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_fit_params)

        N_stars = fit_params.shape[0]
        N_times = len(t)

        x0, vx, y0, vy = fit_params.T  # Each shape (N_stars,)
        t0 = np.atleast_1d(fixed_params_dict['t0'])  # Shape (N_stars,) or (1,)

        if N_times == N_stars:
            # Assume each time corresponds to each star, so N_times = 1
            dt = t - t0 # Shape (N_stars,)
            dt = dt[:, np.newaxis]  # Shape (N_stars, 1)
            N_times = 1
        else:
            dt = t[np.newaxis, :] - t0[:, np.newaxis]  # Shape (N_stars, N_times)

        x = self.model_fit(dt, x0[:, np.newaxis], vx[:, np.newaxis])  # Shape (N_stars, N_times)
        y = self.model_fit(dt, y0[:, np.newaxis], vy[:, np.newaxis])  # Shape (N_stars, N_times)

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x = x.flatten()
            y = y.flatten()

        if fit_param_errs is None:
            return x, y

        fit_param_errs = np.atleast_2d(fit_param_errs)  # (N_stars, N_fit_params)
        x0_err, vx_err, y0_err, vy_err = fit_param_errs.T   # Each shape (N_stars,)
        x_err = np.hypot(x0_err[:, np.newaxis], vx_err[:, np.newaxis] * dt)  # Shape (N_stars, N_times)
        y_err = np.hypot(y0_err[:, np.newaxis], vy_err[:, np.newaxis] * dt)  # Shape (N_stars, N_times)

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x_err = x_err.flatten()
            y_err = y_err.flatten()
        return x, y, x_err, y_err

    def run_fit(
        self, t, x, y, xe, ye,
        fixed_params_dict=None,
        weighting='var',
        use_scipy=True,
        absolute_sigma=True,
        params_guess=None,
        fill_value=np.nan,
        return_chi2=False,
        verbose=True
    ):
        if fixed_params_dict is None:
            fixed_params_dict = {}
        if 't0' not in fixed_params_dict:
            # Default t0 to weighted average time
            fixed_params_dict['t0'] = np.average(t, weights=1./np.hypot(xe, ye))
        self.fixed_params_dict = fixed_params_dict
        t0 = np.atleast_1d(fixed_params_dict['t0'])
        t = np.atleast_1d(t)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        xe = np.atleast_1d(xe)
        ye = np.atleast_1d(ye)

        n_obs = len(t)
        degree_of_freedom = n_obs - self.n_params
        # Not enough data points to fit model
        if degree_of_freedom < 0:
            warnings.warn(
                f'Not enough data points to fit model. Setting parameters to {fill_value} and uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )
            params = np.full(self.n_fit_params, fill_value)
            param_errors = np.full(self.n_fit_params, np.inf)
            if return_chi2:
                return params, param_errors, np.nan, np.nan
            else:
                return params, param_errors

        # degree_of_freedom >= 0
        dt = t - t0
        x_wt, y_wt = self.calc_weights(xe, ye, weighting=weighting)
        if params_guess is None:
            params_guess = [x.mean(), 0., y.mean(), 0.]

        if use_scipy:
            x_opt, x_cov, x_info, x_msg, x_ier = curve_fit(self.model_fit, dt, x, p0=np.array(params_guess[:2]), sigma=1/x_wt**0.5, absolute_sigma=absolute_sigma, full_output=True)
            y_opt, y_cov, y_info, y_msg, y_ier = curve_fit(self.model_fit, dt, y, p0=np.array(params_guess[2:]), sigma=1/y_wt**0.5, absolute_sigma=absolute_sigma, full_output=True)
            x0, vx = x_opt
            y0, vy = y_opt
            x0e, vxe = np.sqrt(x_cov.diagonal())
            y0e, vye = np.sqrt(y_cov.diagonal())
            params = np.array([x0, vx, y0, vy])
            param_errors = np.array([x0e, vxe, y0e, vye])
            if return_chi2:
                # chi2_x, chi2_y = self.calc_chi2(t, x, y, xe, ye, params, fixed_params_dict)
                chi2_x = np.sum(x_info['fvec']**2)
                chi2_y = np.sum(y_info['fvec']**2)
                return params, param_errors, chi2_x, chi2_y
            else:
                return params, param_errors

        # Linear algebraic solution
        # Use  https://en.wikipedia.org/wiki/Weighted_least_squares#Solution_scheme
        X_mat_t = np.vander(dt, 2)

        # x calculation
        W_mat_x = np.diag(x_wt)
        XTWX_mat_x = X_mat_t.T @ W_mat_x @ X_mat_t # Shape (2, 2)
        pcov_x = np.linalg.pinv(XTWX_mat_x)  # Covariance Matrix
        popt_x = pcov_x @ X_mat_t.T @ W_mat_x @ x   # Linear Solution

        # Singular matrix (not enough unique times): Fill uncertainty with Inf.
        if np.linalg.matrix_rank(XTWX_mat_x) < 2:
            warnings.warn(
                f'Singular matrix. Covariance of the parameters could not be estimated. Setting parameter uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )
            perr_x = np.full_like(popt_x, np.inf)
        else:
            perr_x = np.sqrt(np.diag(pcov_x))   # Uncertainty of Linear Solution

        # y calculation
        W_mat_y = np.diag(y_wt)
        XTWX_mat_y = X_mat_t.T @ W_mat_y @ X_mat_t # Shape (2, 2)
        pcov_y = np.linalg.pinv(XTWX_mat_y)  # Covariance Matrix
        popt_y = pcov_y @ X_mat_t.T @ W_mat_y @ y   # Linear Solution

        # Singular matrix (not enough unique times): Fill uncertainty with Inf.
        if np.linalg.matrix_rank(XTWX_mat_y) < 2:
            warnings.warn(
                f'Singular matrix. Covariance of the parameters could not be estimated. Setting parameter uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )
            perr_y = np.full_like(popt_y, np.inf)
        else:
            perr_y = np.sqrt(np.diag(pcov_y))   # Uncertainty of Linear Solution

        # prepare values to return
        vx, x0 = popt_x
        vy, y0 = popt_y
        vxe, x0e = perr_x
        vye, y0e = perr_y

        params = np.array([x0, vx, y0, vy])
        param_errors = np.array([x0e, vxe, y0e, vye])

        # Does not use get_chi2 to accelerate calculation
        if return_chi2 or (not absolute_sigma):
            residual_x = x - X_mat_t @ popt_x
            residual_y = y - X_mat_t @ popt_y

            chi2_x = residual_x.T @ W_mat_x @ residual_x
            chi2_y = residual_y.T @ W_mat_y @ residual_y

        if not absolute_sigma:
            if degree_of_freedom > 0:
                reduced_chi2_x = chi2_x / degree_of_freedom
                reduced_chi2_y = chi2_y / degree_of_freedom

                param_errors[0:2] *= reduced_chi2_x**0.5
                param_errors[2:4] *= reduced_chi2_y**0.5

            else:
                # degree_of_freedom == 0, as < 0 case already handled above
                warnings.warn(
                    f'Degree of freedom < 0. Covariance of the parameters could not be estimated. Setting parameter uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )
                # Set parameter uncertainties to np.inf, same behavior as scipy.optimize.curve_fit
                param_errors = np.full_like(param_errors, np.inf)

        if return_chi2:
            return params, param_errors, chi2_x, chi2_y
        else:
            return params, param_errors

class Acceleration(MotionModel):
    """
    A 2D accelerating motion model for a star on the sky.
    """
    name = "Acceleration"
    fit_param_names = ['x0', 'vx0', 'ax', 'y0', 'vy0', 'ay']
    required_fixed_param_names = ['t0']
    optional_fixed_params = {}
    fixed_param_names = required_fixed_param_names + list(optional_fixed_params.keys())

    n_fit_params = len(fit_param_names)
    # Number of required observations in each direction
    n_params = int((n_fit_params + 1) / 2)

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

    def model(self, t, fit_params, fit_param_errs=None, fixed_params_dict=None):
        """Model positions (and uncertainties, if fit_param_errs is provided) at time t of Acceleration model.

        Parameters
        ----------
        t : float or array-like
            Time(s) at which to evaluate the model
        fit_params : array-like
            x0, vx, ax, y0, vy, ay in shape (N_fit_params,) or (N_stars, N_fit_params)
        fit_param_errs : array-like, optional
            Fit parameter uncertainties with shape (N_stars, N_fit_params) or (N_fit_params,), by default None
        fixed_params_dict : dict
            t0, shape (1,) or (N_stars,)

        Returns
        -------
        x, y (, xe, ye)
            Predicted positions (and uncertainties, if fit_param_errs is provided) with shape (N_stars, N_times), or (N_times,) if N_stars=1, or (N_stars,) if N_times=1
        """
        if fixed_params_dict is None:
            fixed_params_dict = self.fixed_params_dict
        assert 't0' in fixed_params_dict, "Fixed parameter t0 is required for Acceleration model."
        self._check_param_dimensions(fit_params, fit_param_errs, fixed_params_dict)

        t = np.atleast_1d(t)
        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_fit_params)

        N_stars = fit_params.shape[0]
        N_times = len(t)

        x0, vx0, ax, y0, vy0, ay = fit_params.T  # Each shape (N_stars,)
        t0 = np.atleast_1d(fixed_params_dict['t0'])  # Shape (N_stars,) or (1,)

        if N_times == N_stars:
            # Assume each time corresponds to each star, so N_times = 1
            dt = t - t0 # Shape (N_stars,)
            dt = dt[:, np.newaxis]  # Shape (N_stars, 1)
            N_times = 1
        else:
            dt = t[np.newaxis, :] - t0[:, np.newaxis]  # Shape (N_stars, N_times)

        x = self.model_fit(dt, x0[:, np.newaxis], vx0[:, np.newaxis], ax[:, np.newaxis])  # Shape (N_stars, N_times)
        y = self.model_fit(dt, y0[:, np.newaxis], vy0[:, np.newaxis], ay[:, np.newaxis])  # Shape (N_stars, N_times)

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x = x.flatten()
            y = y.flatten()

        if fit_param_errs is None:
            return x, y

        fit_param_errs = np.atleast_2d(fit_param_errs)  # (N_stars, N_fit_params)
        x0_err, vx0_err, ax_err, y0_err, vy0_err, ay_err = fit_param_errs.T
        x_err = np.sqrt(x0_err[:, np.newaxis]**2 + (vx0_err[:, np.newaxis] * dt)**2 + (0.5 * ax_err[:, np.newaxis] * dt**2)**2)  # Shape (N_stars, N_times)
        y_err = np.sqrt(y0_err[:, np.newaxis]**2 + (vy0_err[:, np.newaxis] * dt)**2 + (0.5 * ay_err[:, np.newaxis] * dt**2)**2)  # Shape (N_stars, N_times)

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x_err = x_err.flatten()
            y_err = y_err.flatten()

        return x, y, x_err, y_err


    def run_fit(
        self, t, x, y, xe, ye,
        fixed_params_dict=None,
        weighting='var',
        use_scipy=True,
        absolute_sigma=True,
        params_guess=None,
        fill_value=np.nan,
        return_chi2=False,
        verbose=True
    ):
        if fixed_params_dict is None:
            fixed_params_dict = {}
        if 't0' not in fixed_params_dict:
            # Default t0 to weighted average time
            fixed_params_dict['t0'] = np.average(t, weights=1./np.hypot(xe, ye))
        self.fixed_params_dict = fixed_params_dict
        t0 = np.atleast_1d(fixed_params_dict['t0'])
        t = np.atleast_1d(t)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        xe = np.atleast_1d(xe)
        ye = np.atleast_1d(ye)

        if not use_scipy:
            if verbose:
                warnings.warn("Acceleration model has no non-scipy fitter option. Running with scipy.")

        n_obs = len(t)
        degree_of_freedom = n_obs - self.n_params
        # Not enough data points to fit model
        if degree_of_freedom < 0:
            warnings.warn(
                f'Not enough data points to fit model. Setting parameters to {fill_value} and uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )
            params = np.full(self.n_fit_params, fill_value)
            param_errors = np.full(self.n_fit_params, np.inf)
            if return_chi2:
                return params, param_errors, np.nan, np.nan
            else:
                return params, param_errors

        # degree_of_freedom >= 0
        dt = t - t0
        x_wt, y_wt = self.calc_weights(xe,ye, weighting=weighting)
        if params_guess is None:
            # Initial guess for velocity:
            idx_first, idx_last = np.argmin(t), np.argmax(t)
            t_span = t[idx_last] - t[idx_first]
            params_guess = [x.mean(), (x[idx_last] - x[idx_first]) / t_span, 0., y.mean(), (y[idx_last] - y[idx_first]) / t_span, 0.]

        x_opt, x_cov, x_info, x_msg, x_ier = curve_fit(self.model_fit, dt, x, p0=np.array(params_guess[:3]), sigma=1/x_wt**0.5, absolute_sigma=absolute_sigma, full_output=True)
        y_opt, y_cov, y_info, y_msg, y_ier = curve_fit(self.model_fit, dt, y, p0=np.array(params_guess[3:]), sigma=1/y_wt**0.5, absolute_sigma=absolute_sigma, full_output=True)
        x0, vx0, ax = x_opt
        y0, vy0, ay = y_opt
        x0e, vx0e, axe = np.sqrt(x_cov.diagonal())
        y0e, vy0e, aye = np.sqrt(y_cov.diagonal())

        params = np.array([x0, vx0, ax, y0, vy0, ay])
        param_errors = np.array([x0e, vx0e, axe, y0e, vy0e, aye])
        if return_chi2:
            # chi2_x, chi2_y = self.calc_chi2(t, x, y, xe, ye, params, fixed_params_dict)
            chi2_x = np.sum(x_info['fvec']**2)
            chi2_y = np.sum(y_info['fvec']**2)
            return params, param_errors, chi2_x, chi2_y
        else:
            return params, param_errors

class Parallax(MotionModel):
    """
    Motion model for linear proper motion + parallax

    Requires RA and Dec J2000 (degrees) for parallax calculation.
    Optional PA is counterclockwise offset of the image y-axis from North.
    Optional obs parameter describes observer location, default is 'earth'.
    """
    name = "Parallax"
    fit_param_names = ['x0', 'vx', 'y0', 'vy', 'pi']
    required_fixed_param_names = ['t0', 'ra', 'dec']
    optional_fixed_params = {'pa': 0., 'obsLocation': 'earth'}
    fixed_param_names = required_fixed_param_names + list(optional_fixed_params.keys())


    n_fit_params = len(fit_param_names)
    # Number of required observations in each direction
    n_params = int((n_fit_params + 1) / 2)

    def __init__(self):
        super().__init__()
        self.pvec_cached = None  # Cache for parallax vector
        self.t_mjd_cached = None  # Cache for times corresponding to cached parallax vector
        return

    def calc_parallax_vector(self, t_mjd, ra, dec, pa=0., obsLocation='earth'):
        """Calculate parallax vector of shape (N_stars, 2, N_times)

        Parameters
        ----------
        t_mjd : array-like
            Time array in mjd
        ra : float or array-like
            Right ascension(s) in degrees
        dec : float or array-like
            Declination(s) in degrees
        pa : float or array-like, optional
            Position angle(s) of image y-axis from North in degrees, by default 0.
        obsLocation : str, optional
            Observer location, by default 'earth'

        Returns
        -------
        pvec
            Parallax vector of shape (N_stars, 2, N_times)
        """
        if self.pvec_cached is not None:
            t_mjd = np.atleast_1d(t_mjd)
            t_mjd_cached = self.t_mjd_cached
            if np.array_equal(t_mjd, t_mjd_cached):
                # If cached values match input times, return cached values
                return self.pvec_cached

            elif all(np.isin(t_mjd, t_mjd_cached)):
                # If all input times are in cached values, return those
                # Calculate pvec_idxs such that t_mjd_cached[ pvec_idxs ] == t_mjd
                pvec_idxs = np.array([np.where(t_mjd_cached == t_mjd_i)[0][0] for t_mjd_i in t_mjd])
                pvec = self.pvec_cached[:, :, pvec_idxs]
                return pvec

        pvec = parallax.parallax_in_direction(ra, dec, t_mjd, obsLocation=obsLocation, pa=pa)   # Shape (N_stars, 2, N_times)
        # self.plx_vector_cached = [t_mjd, pvec]
        self.t_mjd_cached = t_mjd
        self.pvec_cached = pvec
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
        x_result, y_result : array-like
            Model positions at time t of Parallax model, shape (N_stars, N_times)
        """
        # x0, vx, y0, vy, pi are all shape (N_stars, N_times)
        x_result = x0 + vx * dt + pi * self.pvec[:, 0, :]  # Parallax contribution in x direction
        y_result = y0 + vy * dt + pi * self.pvec[:, 1, :]  # Parallax contribution in y direction
        return x_result, y_result

    def _model_fit(self, dt, x0, vx, y0, vy, pi):
        """Wrapper for model_fit to return concatenated results for scipy fitting."""
        x_result, y_result = self.model_fit(dt, x0, vx, y0, vy, pi)
        # scipy.optimize.curve_fit expects a 1D output array with the same length
        # as the input ydata. For single-star fits, intermediate broadcasting can
        # yield arrays with shape (1, N_times); flatten to avoid M=1 interpretation.
        return np.hstack([np.ravel(x_result), np.ravel(y_result)])  # Shape (2*N_times,)

    def model(self, t, fit_params, fit_param_errs=None, fixed_params_dict=None):
        """Model positions (and uncertainties, if fit_param_errs is provided) at time t of Parallax model.

        Parameters
        ----------
        t : float or array-like
            Times at which to evaluate the model
        fit_params : array-like
            x0, vx, y0, vy, pi in shape (N_fit_params,) or (N_stars, N_fit_params)
        fit_param_errs : array-like, optional
            Uncertainties in fit parameters, by default None
        fixed_params : dict
            - t0, shape (N_stars,) or (1,).
            - ra, shape (N_stars,) or (1,).
            - dec, shape (N_stars,) or (1,).
            - pa, optional, shape (N_stars,) or (1,), by default 0.
            - obsLocation, optional, string, by default 'earth'

        Returns
        -------
        x, y (, xe, ye)
            Predicted positions (and uncertainties, if fit_param_errs is provided) with shape (N_stars, N_times), or (N_times,) if N_stars=1, or (N_stars,) if N_times=1
        """
        if fixed_params_dict is None:
            fixed_params_dict = self.fixed_params_dict
        assert all([_ in fixed_params_dict for _ in ['t0', 'ra', 'dec']]), "Fixed parameters t0, ra, and dec are required for Parallax model."
        self._check_param_dimensions(fit_params, fit_param_errs, fixed_params_dict)

        t = np.atleast_1d(t)
        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_fit_params)

        N_stars = fit_params.shape[0]
        N_times = len(t)

        x0, vx, y0, vy, pi = fit_params.T  # Each shape (N_stars,)
        t0 = np.atleast_1d(fixed_params_dict['t0'])  # Shape (N_stars,) or (1,)
        ra = np.atleast_1d(fixed_params_dict['ra'])
        dec = np.atleast_1d(fixed_params_dict['dec'])
        pa = np.atleast_1d(fixed_params_dict.get('pa', 0.0))
        obsLocation = fixed_params_dict.get('obsLocation', 'earth')

        # TODO: vectorize parallax.parallax_in_direction to handle multiple obsLocation?
        assert isinstance(obsLocation, str) or (np.unique(obsLocation).size == 1), "obsLocation must be a single string for all stars at this time."
        if not isinstance(obsLocation, str):
            obsLocation = np.unique(obsLocation)[0]


        if N_times == N_stars:
            # Assume each time corresponds to each star, so N_times = 1
            dt = t - t0 # Shape (N_stars,)
            dt = dt[:, np.newaxis]  # Shape (N_stars, 1)
            N_times = 1
        else:
            dt = t[np.newaxis, :] - t0[:, np.newaxis]  # Shape (N_stars, N_times)

        t_mjd = Time(t, format='decimalyear', scale='utc').mjd  # Shape (N_times,)
        self.pvec = self.calc_parallax_vector(t_mjd, ra, dec, pa=pa, obsLocation=obsLocation) # Shape (N_stars, 2, N_times)
        x, y = self.model_fit(dt, x0[:, np.newaxis], vx[:, np.newaxis], y0[:, np.newaxis], vy[:, np.newaxis], pi[:, np.newaxis])  # Shape (N_stars, N_times)

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x = x.flatten()
            y = y.flatten()

        if fit_param_errs is None:
            return x, y

        fit_param_errs = np.atleast_2d(fit_param_errs)  # (N_stars, N_fit_params)
        x0_err, vx_err, y0_err, vy_err, pi_err = fit_param_errs.T
        x_err = np.sqrt(x0_err[:, np.newaxis]**2 + (vx_err[:, np.newaxis] * dt)**2 + (pi_err[:, np.newaxis] * self.pvec[:, 0, :])**2)  # Shape (N_stars, N_times)
        y_err = np.sqrt(y0_err[:, np.newaxis]**2 + (vy_err[:, np.newaxis] * dt)**2 + (pi_err[:, np.newaxis] * self.pvec[:, 1, :])**2)  # Shape (N_stars, N_times)

        if N_stars == 1 or N_times == 1:
            # If only one star, return flattened arrays
            x_err = x_err.flatten()
            y_err = y_err.flatten()
        return x, y, x_err, y_err


    def run_fit(
        self, t, x, y, xe, ye,
        fixed_params_dict,
        weighting='var',
        use_scipy=True,
        absolute_sigma=True,
        params_guess=None,
        fill_value=np.nan,
        return_chi2=False,
        verbose=True
    ):
        if not use_scipy:
            if verbose:
                warnings.warn("Parallax model has no non-scipy fitter option. Running with scipy.", UserWarning)

        assert all([k in fixed_params_dict for k in ['ra', 'dec']]), "Parallax model requires 'ra' and 'dec' in fixed_params."
        t = np.atleast_1d(t)

        if 't0' not in fixed_params_dict:
            # Default t0 to weighted average time
            fixed_params_dict['t0'] = np.average(t, weights=1./np.hypot(xe, ye))
        if 'obsLocation' not in fixed_params_dict:
            fixed_params_dict['obsLocation'] = 'earth'
        self.fixed_params_dict = fixed_params_dict
        t0 = np.atleast_1d(fixed_params_dict['t0'])
        ra = np.atleast_1d(fixed_params_dict['ra'])
        dec = np.atleast_1d(fixed_params_dict['dec'])
        pa = np.atleast_1d(fixed_params_dict.get('pa', 0.0))
        obsLocation = fixed_params_dict['obsLocation']

        n_fit = len(t)
        degree_of_freedom = n_fit - self.n_params
        # Not enough data points to fit model
        if degree_of_freedom < 0:
            warnings.warn(
                f'Not enough data points to fit model. Setting parameters to {fill_value} and uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )
            params = np.full(self.n_fit_params, fill_value)
            param_errors = np.full(self.n_fit_params, np.inf)
            if return_chi2:
                return params, param_errors, np.nan, np.nan
            else:
                return params, param_errors

        # degree_of_freedom >= 0
        t_mjd = Time(t, format='decimalyear', scale='utc').mjd
        self.pvec = self.calc_parallax_vector(t_mjd, ra, dec, pa=pa, obsLocation=obsLocation) # Shape (2, N_times)
        x_wt, y_wt = self.calc_weights(xe, ye, weighting=weighting)

        # Initial guesses, x0,y0 as x,y averages;
        #     vx,vy as average velocity if first and last points are perfectly measured;
        #     pi for 10 pc distance
        if params_guess is None:
            idx_first, idx_last = np.argmin(t), np.argmax(t)
            t_span = t[idx_last] - t[idx_first]
            params_guess = np.array([
                x.mean(), (x[idx_last] - x[idx_first]) / t_span,
                y.mean(), (y[idx_last] - y[idx_first]) / t_span,
                0.1
            ])

        # Convert weights to 1-sigma uncertainties for curve_fit.
        # calc_weights returns w = 1/sigma^2 for 'var' and w = 1/sigma for 'std'.
        if weighting == 'std':
            sigma_x = 1.0 / x_wt
            sigma_y = 1.0 / y_wt
        else:
            sigma_x = 1.0 / np.sqrt(x_wt)
            sigma_y = 1.0 / np.sqrt(y_wt)

        popt, pcov, infodict, mesg, ier = curve_fit(
            self._model_fit, t - t0, np.hstack([x, y]),
            p0=params_guess, sigma=np.hstack([sigma_x, sigma_y]),
            absolute_sigma=absolute_sigma, full_output=True
        )
        x0, vx, y0, vy, pi = popt
        x0_err, vx_err, y0_err, vy_err, pi_err = np.sqrt(pcov.diagonal())

        params = np.array([x0, vx, y0, vy, pi])
        param_errors = np.array([x0_err, vx_err, y0_err, vy_err, pi_err])

        if return_chi2:
            # chi2_x, chi2_y = self.calc_chi2(t, x, y, xe, ye, params, fixed_params_dict)
            chi2_x = np.sum(infodict['fvec'][:len(t)]**2)
            chi2_y = np.sum(infodict['fvec'][len(t):]**2)
            return params, param_errors, chi2_x, chi2_y
        else:
            return params, param_errors


def motion_model_param_names(motion_models, with_errors=True, with_fixed=True):
    """Get the motion model parameter names from a list of MotionModels.

    Parameters
    ----------
    motion_models : MotionModel, str, or list of MotionModels/strings.
        Motion model to query parameter names from. If str, should be the name of a MotionModel class.
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

    motion_models = np.atleast_1d(motion_models)
    mm_map = motion_model_map()
    for mm in motion_models:
        if isinstance(mm, str):
            mm = mm_map[mm]
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
    return list_of_parameters


def all_motion_model_param_names(with_errors=True, with_fixed=True):
    """Get all motion model parameter names from all available MotionModels.

    Parameters
    ----------
    with_errors : bool, optional
        Add uncertainty names with '_err' suffix or not, by default True
    with_fixed : bool, optional
        Add fixed param names with '_fixed' suffix or not, by default True

    Returns
    -------
    list
        List of all unique parameter names across all motion models
    """
    return motion_model_param_names(MotionModel.__subclasses__(), with_errors=with_errors, with_fixed=with_fixed)

def motion_model_map():
    """Get a dictionary mapping motion model names to MotionModel classes.

    Returns
    -------
    mm_map : dict
        Dictionary mapping motion model names to MotionModel classes.
    """
    mm_map = dict(
        [(mm.__name__, mm) for mm in MotionModel.__subclasses__()]
    )
    # Sort by required epochs
    mm_map = dict(sorted(mm_map.items(), key=lambda item: item[1].n_params))
    return mm_map