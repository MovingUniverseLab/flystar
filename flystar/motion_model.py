import warnings
import numpy as np
from abc import ABC
from flystar import parallax
from astropy.time import Time
from scipy.optimize import OptimizeWarning


def weight_from_sigma(sigma, valid=None):
    """
    Convert an uncertainty (sigma) array into a safe inverse-variance
    weight (1/sigma**2), for use in a weighted sum/average.

    A point with no real uncertainty information should contribute
    nothing to a weighted sum -- but naively computing 1/sigma**2 can
    instead produce an infinite or NaN weight (sigma is NaN/inf/exactly
    zero, or so small that squaring it underflows to zero), which would
    corrupt rather than exclude that point. This handles all of those
    cases uniformly: any sigma that doesn't produce a finite weight, or
    any point explicitly marked invalid via `valid`, gets a weight of
    exactly 0.

    This does NOT handle the "every point has weight 0" case for you --
    a weighted average built from these weights still needs its own
    explicit fallback for that (see combine_lists/fit_motion_models),
    since there's no single value this function could return that fixes
    an otherwise-undefined 0/0 average.

    Parameters
    ----------
    sigma : array-like
        Uncertainty values (any invalid/zero/overflow-inducing value is
        safely handled).
    valid : array-like of bool, optional
        If given, points where this is False also get weight 0,
        regardless of sigma, by default None.

    Returns
    -------
    weight : ndarray
        Same shape as sigma.
    """
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = 1. / sigma**2
        if valid is not None:
            weight = np.where(valid, weight, 0.0)
    weight[~np.isfinite(weight)] = 0.0
    return weight


def broadcast_times(t, n_stars, caller='model'):
    """
    Normalize a time argument into an explicit (n_stars, n_times) grid.

    The SHAPE of `t` alone decides whether the times are shared across stars
    or are per-star. There is deliberately no inference from
    len(t) == n_stars: that used to be read as "one time per star", so the
    very same 1D array changed meaning depending on how many stars the table
    happened to contain, and a table whose star count coincided with its
    epoch count silently took the wrong branch.

    ==========================  ====================================
    `t`                         meaning
    ==========================  ====================================
    scalar                      one time, every star
    ``(n_times,)``              one shared grid, every star -- always,
                                even when n_times == n_stars
    ``(1, n_times)``            the same, written explicitly
    ``(n_stars, n_times)``      each star has its own times
    ==========================  ====================================

    To evaluate every star at its own single time, pass a column vector --
    ``t[:, np.newaxis]``, of shape ``(n_stars, 1)`` -- not a bare 1D array.
    Note that propagating a whole table to one new epoch does NOT need this:
    pass the scalar epoch and a per-star ``t0`` fixed parameter, and each
    star's dt = t - t0[star] already differs.

    Parameters
    ----------
    t : scalar or array-like
        Times, in one of the shapes above.
    n_stars : int
        Number of stars the times are being broadcast against.
    caller : str, optional
        Name used in the error message, by default 'model'.

    Returns
    -------
    ndarray, shape (n_stars, n_times)
        May be a read-only broadcast view -- do not write into it.
    """
    t = np.asarray(t, dtype=float)

    if t.ndim == 0:
        return np.full((n_stars, 1), float(t))
    if t.ndim == 1:
        return np.broadcast_to(t[np.newaxis, :], (n_stars, t.shape[0]))
    if t.ndim == 2:
        if t.shape[0] == n_stars:
            return t
        if t.shape[0] == 1:
            return np.broadcast_to(t, (n_stars, t.shape[1]))
        raise ValueError(
            f"{caller}: 2D time array must have one row per star -- got shape "
            f"{t.shape} for {n_stars} star(s). Pass a shared time grid as 1D "
            f"(n_times,), or per-star times as (n_stars, n_times); for one "
            f"time per star use t[:, np.newaxis], of shape ({n_stars}, 1)."
        )
    raise ValueError(
        f"{caller}: time array must be scalar, 1D (n_times,), or 2D "
        f"(n_stars, n_times) -- got a {t.ndim}D array of shape {t.shape}."
    )


def sigma_from_error(xe, ye, weighting='var'):
    """
    Convert x/y position errors into the sigma values a weighted fit
    should use, based on the requested weighting scheme.

    weighting : str, optional
        'var': sigma = ``abs(xe)``, ``abs(ye)``, so a later 1/sigma**2 gives
        true inverse-variance weighting (w=1/xe**2, 1/ye**2).
        'std': sigma = ``sqrt(abs(xe))``, ``sqrt(abs(ye))``, so the same later
        1/sigma**2 instead gives standard-error weighting (w=1/xe, 1/ye).
        By default 'var'.
    """
    if weighting=='std':
        return np.sqrt(np.abs(xe)), np.sqrt(np.abs(ye))
    elif weighting=='var':
        return np.abs(xe), np.abs(ye)
    else:
        warnings.warn("Invalid weighting, using default weighting scheme var.", UserWarning)
        return np.abs(xe), np.abs(ye)


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
        """Evaluate the model at the given time(s).

        Every concrete subclass overrides this. The time argument follows one
        contract across all of them, resolved by :func:`broadcast_times`:

        Parameters
        ----------
        t : scalar or array-like
            Time(s) at which to evaluate the model. The SHAPE decides
            whether the times are shared across stars or are per-star --
            nothing is inferred from ``len(t)`` matching ``N_stars``:

            ======================  =========================================
            ``t``                   meaning
            ======================  =========================================
            scalar                  one time, every star
            ``(N_times,)``          one shared grid, every star -- always,
                                    even when ``N_times == N_stars``
            ``(1, N_times)``        the same, written explicitly
            ``(N_stars, N_times)``  each star has its own times
            ======================  =========================================

            For one time per star, pass a column vector
            ``t[:, np.newaxis]`` of shape ``(N_stars, 1)`` -- not a bare 1D
            array. Any other shape raises ``ValueError`` rather than being
            guessed at. See :func:`broadcast_times`.
        fit_params : array-like
            Fit parameters, shape (N_fit_params,) or (N_stars, N_fit_params).
        fit_param_errs : array-like, optional
            Uncertainties on fit_params, same shape, by default None.
        fixed_params_dict : dict, optional
            Fixed (non-fit) parameters; see each subclass's
            fixed_param_names, by default None.

        Returns
        -------
        x, y (, xe, ye)
            Predicted positions, and uncertainties if fit_param_errs is
            given, with shape (N_stars, N_times) -- flattened when
            N_stars == 1 or N_times == 1.
        """
        self._check_param_dimensions(fit_params, fit_param_errs, fixed_params_dict)
        if fit_param_errs is None:
            return np.full_like(t, np.nan), np.full_like(t, np.nan)
        return np.full_like(t, np.nan), np.full_like(t, np.nan), np.full_like(t, np.inf), np.full_like(t, np.inf)

    def run_fit(self, t, x, y, xe, ye, valid, fixed_params_dict=None, weighting='var',
                       absolute_sigma=True, fill_value=np.nan, verbose=True):
        """
        Fit a batch of stars at once (used both for the main fit and for
        fit()'s bootstrap iterations). Every concrete MotionModel subclass
        must override this -- there is no per-star fallback anymore, so a
        subclass that doesn't override it would otherwise silently inherit
        this stub and produce all-fill_value/inf fits with no warning. A
        subclass whose fit genuinely can't be vectorized across stars can
        still satisfy this same batch-in/batch-out signature by looping
        over stars internally (e.g. calling scipy.optimize.curve_fit once
        per row) -- nothing requires the implementation to be closed-form,
        only the interface to accept/return a whole batch at once.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_fit(t, x, y, xe, ye, valid, ...)."
        )

    def fit(
        self, t, x, y, xe, ye,
        fixed_params_dict=None,
        weighting='var',
        absolute_sigma=True,
        fill_value=np.nan,
        return_chi2=False,
        bootstrap=0,
        seed=None,
        verbose=True
    ):
        """Fit stellar motion parameters -- for one star, or for a whole
        batch of stars at once.

        t, x, y, xe, ye : 1D, shape (n_epochs,)
            A single star's data. The caller is expected to have already
            filtered this down to that star's own real epochs -- no
            padding, nothing to mask.
        t, x, y, xe, ye : 2D, shape (n_stars, n_epochs)
            A batch of many stars packed into one rectangular array (this
            is the path StarTable.fit_motion_models uses for real
            performance -- run_fit is never called directly from outside
            this module). Since stars don't all have the same number of
            real epochs, some cells are padding; padding is marked by nan
            in x and/or y (this codebase's existing "no data" convention),
            not by a separate mask the caller has to build. valid =
            isfinite(x) & isfinite(y) is derived here and handed to
            run_fit(), which does the actual (closed-form, vectorized)
            solve for the whole batch in one call.

        Every concrete model's run_fit is closed-form: the single-star
        case just wraps the star's data into a batch of one row (and, for
        bootstrap, into a batch of `bootstrap` rows -- one resampled
        subset/order of this star's epochs per row), so every case above
        goes through the same vectorized, non-iterative solve.

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
        absolute_sigma : bool, optional
            Absolute sigma. If False, parameter errors are rescaled by the reduced chi^2, by default True
        fill_value : float, optional
            Fill value for parameters when not enough data points to fit model, by default np.nan
        return_chi2 : bool, optional
            Return chi^2 values along with parameters and uncertainties in params, param_errs, chi2_x, chi2_y, by default False.
            Ignored for the 2D (batch) case, which always returns all four.
        bootstrap : int, optional
            Bootstrapping uncertainties (single-star case only), by default 0
        seed : int, optional
            Seed for the random number generator, by default None
        verbose : bool, optional
            Print warning messages, by default True

        Returns
        -------
        params, param_errs(, chi2_x, chi2_y)
            Parameters, uncertainties, and chi squares if return_chi2 is True (always for the batch case). The corresponding parameter names are in self.fit_param_names.
        """
        t = np.asarray(t)
        x = np.asarray(x)
        y = np.asarray(y)
        xe = np.asarray(xe)
        ye = np.asarray(ye)

        if not verbose:
            warnings.filterwarnings("ignore", category=OptimizeWarning)

        if t.ndim == 2:
            # Batch path: many stars packed into one rectangular array.
            # No separate mask is built or passed by the caller -- padding
            # epochs are wherever x or y is nan, and that's exactly what
            # isfinite picks out.
            valid = np.isfinite(x) & np.isfinite(y)
            result = self.run_fit(
                t, x, y, xe, ye, valid,
                fixed_params_dict=fixed_params_dict, weighting=weighting, absolute_sigma=absolute_sigma,
                fill_value=fill_value, verbose=verbose
            )
            if not verbose:
                warnings.resetwarnings()
            return result

        for variable, name in zip([t, x, y, xe, ye], ['t', 'x', 'y', 'xe', 'ye']):
            assert np.ndim(variable) == 1, f"Input {name} array must be 1D (single star) or 2D (batch)! Got shape {np.shape(variable)}"
            if name != 't':
                assert len(t) == len(variable), f'Input {name} must have the same length as t! Got len(t)={len(t)}, len({name})={len(variable)}'

        # Copy (rather than mutate the caller's dict) before filling in a
        # default t0 -- same convenience the old per-star run_fit() gave
        # Linear/Acceleration/Parallax when t0 wasn't supplied.
        fixed_params_dict = dict(fixed_params_dict) if fixed_params_dict is not None else {}
        if ('t0' in self.required_fixed_param_names) and ('t0' not in fixed_params_dict):
            fixed_params_dict['t0'] = np.average(t, weights=1. / np.hypot(xe, ye))
        # Remembered so a later self.model(t, params) call (without its own
        # fixed_params_dict) can fall back to what this fit used -- same
        # convenience the old per-star run_fit() provided.
        self.fixed_params_dict = fixed_params_dict

        n_obs = len(t)
        valid = np.ones((1, n_obs), dtype=bool)
        params, param_errs, chi2_x, chi2_y = self.run_fit(
            t[np.newaxis, :], x[np.newaxis, :], y[np.newaxis, :], xe[np.newaxis, :], ye[np.newaxis, :], valid,
            fixed_params_dict=fixed_params_dict, weighting=weighting, absolute_sigma=absolute_sigma,
            fill_value=fill_value, verbose=verbose
        )
        params, param_errs, chi2_x, chi2_y = params[0], param_errs[0], chi2_x[0], chi2_y[0]

        # Bootstrap errors
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
            bdx_all = np.hstack((bdx_unique, bdx_extra))  # shape (bootstrap, n_obs)

            # All bootstrap draws of this one star are fit in a single
            # run_fit call -- each draw is just a "row" with its own
            # resampled subset/order of this star's epochs (valid is
            # all-True since every entry in a row is a real, if repeated,
            # epoch).
            valid_boot = np.ones_like(bdx_all, dtype=bool)
            bb_params, bb_param_errs, _, _ = self.run_fit(
                t[bdx_all], x[bdx_all], y[bdx_all], xe[bdx_all], ye[bdx_all], valid_boot,
                fixed_params_dict=fixed_params_dict, weighting=weighting, absolute_sigma=absolute_sigma,
                fill_value=fill_value, verbose=verbose
            )

            # Save the errors from the bootstrap
            param_errs = np.std(bb_params, axis=0)

            # Account for odd case
            inf_errs = np.all(bb_param_errs == np.inf, axis=0)
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
            if parallax:
                degree_of_freedom = 2*len(x) - len(self.fit_param_names)
            else:
                degree_of_freedom = len(x) - self.n_params
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
        t : scalar or array-like
            Time(s) at which to evaluate the model. The shape decides
            whether the times are shared across stars or are per-star --
            nothing is inferred from ``len(t)`` matching ``N_stars``. Accepts
            a scalar (one time, every star), ``(N_times,)`` or
            ``(1, N_times)`` (one shared grid for every star -- always, even
            when ``N_times == N_stars``), or ``(N_stars, N_times)`` (each star
            its own times); for one time per star pass ``t[:, np.newaxis]``.
            Any other shape raises ``ValueError``. See
            :func:`broadcast_times` for the full table.
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

        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_fit_params)

        N_stars = fit_params.shape[0]
        # See broadcast_times: t's shape alone says shared-grid vs per-star.
        N_times = broadcast_times(t, N_stars, caller='Empty.model').shape[1]

        x = np.full((N_stars, N_times), np.nan)
        y = np.full((N_stars, N_times), np.nan)

        if N_stars == 1 or N_times == 1:
            # Same squeeze convention as every other model
            x = x.flatten()
            y = y.flatten()

        if fit_param_errs is None:
            return x, y
        return x, y, np.full_like(x, np.inf), np.full_like(y, np.inf)

    def run_fit(self, t, x, y, xe, ye, valid, fixed_params_dict=None, weighting='var',
                       absolute_sigma=True, fill_value=np.nan, verbose=True):
        """
        Batch fit for many stars at once. Empty's "fit" never looks at any
        data -- it's always fill_value/inf regardless of what's passed in
        -- so there's no actual computation to batch. This exists purely
        so that a table containing some Empty stars (there is almost
        always at least a handful, e.g. stars with 0 valid epochs)
        doesn't force the caller to spin up a
        multiprocessing pool -- and pay its real, fixed per-worker spawn
        cost -- just to run this trivial, zero-cost case one star at a time.

        Parameters
        ----------
        t, x, y, xe, ye, valid : array-like, shape (n_stars, n_epochs)
            Unused -- accepted only for interface consistency with other
            motion models' run_fit.
        fixed_params_dict, weighting, absolute_sigma : unused.
        fill_value : float, optional
            Fill value for parameters when not enough data points to fit model, by default np.nan
        verbose : bool, optional
            Print warning messages, by default True

        Returns
        -------
        params : ndarray, shape (n_stars, 0)
        param_errs : ndarray, shape (n_stars, 0)
        chi2_x, chi2_y : ndarray, shape (n_stars,), all nan
        """
        n_stars = t.shape[0]
        params = np.full((n_stars, self.n_fit_params), fill_value)
        param_errs = np.full((n_stars, self.n_fit_params), np.inf)
        chi2x = np.full(n_stars, np.nan)
        chi2y = np.full(n_stars, np.nan)
        return params, param_errs, chi2x, chi2y


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
        t : scalar or array-like
            Time(s) at which to evaluate the model. The shape decides
            whether the times are shared across stars or are per-star --
            nothing is inferred from ``len(t)`` matching ``N_stars``. Accepts
            a scalar (one time, every star), ``(N_times,)`` or
            ``(1, N_times)`` (one shared grid for every star -- always, even
            when ``N_times == N_stars``), or ``(N_stars, N_times)`` (each star
            its own times); for one time per star pass ``t[:, np.newaxis]``.
            Any other shape raises ``ValueError``. See
            :func:`broadcast_times` for the full table.
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
        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_fit_params)
        self._check_param_dimensions(fit_params, fit_param_errs, fixed_params_dict)

        N_stars = fit_params.shape[0]
        # See broadcast_times: t's shape alone says shared-grid vs per-star.
        dt = broadcast_times(t, N_stars, caller='Fixed.model')  # (N_stars, N_times)
        N_times = dt.shape[1]
        x0, y0 = fit_params.T  # Each shape (N_stars,)

        # Return results in (N_stars, N_times) shape. Fixed is
        # time-independent, so dt only sets the output shape here.
        x = self.model_fit(dt, x0[:, np.newaxis])  # Shape (N_stars, N_times)
        y = self.model_fit(dt, y0[:, np.newaxis])  # Shape (N_stars, N_times)

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

    def run_fit(self, t, x, y, xe, ye, valid, fixed_params_dict=None, weighting='var',
                       absolute_sigma=True, fill_value=np.nan, verbose=True):
        """
        Batch fit for many stars at once. Fixed's fit is closed-form (a
        weighted average -- no iterative optimizer), so nothing about it
        actually requires fitting one star at a time; this fits the whole
        batch in one pass instead of looping (or spinning up
        multiprocessing for) each star individually.

        Parameters
        ----------
        t, x, y, xe, ye : array-like, shape (n_stars, n_epochs)
            Per-star, per-epoch data. Entries where `valid` is False are
            ignored -- their content does not matter (e.g. they can be NaN
            placeholders for undetected epochs).
        valid : array-like of bool, shape (n_stars, n_epochs)
            Which entries are usable for each star.
        fixed_params_dict : dict, optional
            Unused -- Fixed has no fixed params -- accepted only so callers
            can call run_fit() uniformly across motion models (e.g.
            Linear requires fixed_params_dict={'t0': ...}), by default None.
        weighting : str, optional
            'var' (w=1/xe**2, 1/ye**2) or 'std' (w=1/xe, 1/ye), by default 'var'
        absolute_sigma : bool, optional
            If False, parameter errors are rescaled by the reduced chi^2, by default True
        fill_value : float, optional
            Fill value for parameters when not enough data points to fit model, by default np.nan
        verbose : bool, optional
            Print warning messages, by default True

        Returns
        -------
        params : ndarray, shape (n_stars, 2)
        param_errs : ndarray, shape (n_stars, 2)
        chi2_x, chi2_y : ndarray, shape (n_stars,)
        """
        n_valid = valid.sum(axis=1)
        has_data = n_valid >= self.n_params  # degree_of_freedom >= 0

        if verbose and np.any(~has_data):
            warnings.warn(
                f'Not enough data points to fit model for {np.sum(~has_data)} star(s). '
                f'Setting parameters to {fill_value} and uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )

        sigma_x, sigma_y = sigma_from_error(xe, ye, weighting=weighting)
        x_wt = weight_from_sigma(sigma_x, valid)
        y_wt = weight_from_sigma(sigma_y, valid)

        x_wt_sum = x_wt.sum(axis=1)
        y_wt_sum = y_wt.sum(axis=1)
        x_masked = np.where(valid, x, 0.0)
        y_masked = np.where(valid, y, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            x0 = (x_masked * x_wt).sum(axis=1) / x_wt_sum
            y0 = (y_masked * y_wt).sum(axis=1) / y_wt_sum
            x0e = 1. / np.sqrt(x_wt_sum)
            y0e = 1. / np.sqrt(y_wt_sum)

        params = np.column_stack([x0, y0])
        param_errs = np.column_stack([x0e, y0e])

        # chi2: Fixed's prediction is time-independent (x_pred == x0 for every
        # epoch). Weighted with x_wt/y_wt -- the same (weighting-scheme)
        # weights the fit itself used, as in Linear/Acceleration/Parallax --
        # NOT a re-derived 1/xe**2. Those two only coincide for
        # weighting='var'; under weighting='std' the fit's weight is 1/xe, so
        # dividing by xe**2 here would report a chi2 inconsistent with the fit
        # (and, via the absolute_sigma=False rescaling below, wrong parameter
        # errors that disagree with scipy.optimize.curve_fit). Using the
        # weights also makes an epoch with unusable xe/ye (weight 0) drop out
        # of chi2 cleanly instead of poisoning the whole sum with nan.
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2x = (x_wt * (x_masked - x0[:, np.newaxis])**2).sum(axis=1)
            chi2y = (y_wt * (y_masked - y0[:, np.newaxis])**2).sum(axis=1)

        if not absolute_sigma:
            dof = n_valid - self.n_params
            dof_pos = dof > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                reduced_chi2x = np.where(dof_pos, chi2x / np.where(dof_pos, dof, 1), 1.0)
                reduced_chi2y = np.where(dof_pos, chi2y / np.where(dof_pos, dof, 1), 1.0)
            param_errs[:, 0] = np.where(dof_pos, param_errs[:, 0] * np.sqrt(reduced_chi2x), np.inf)
            param_errs[:, 1] = np.where(dof_pos, param_errs[:, 1] * np.sqrt(reduced_chi2y), np.inf)
            if verbose and np.any(has_data & ~dof_pos):
                warnings.warn(
                    'Degree of freedom <= 0 for some star(s). Covariance of the parameters could not be '
                    'estimated. Setting parameter uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )

        # Not-enough-data stars: overwrite with fill_value/inf/nan regardless
        # of whatever the (meaningless, e.g. 0/0) computation above produced.
        params[~has_data] = fill_value
        param_errs[~has_data] = np.inf
        chi2x[~has_data] = np.nan
        chi2y[~has_data] = np.nan

        return params, param_errs, chi2x, chi2y

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
        t : scalar or array-like
            Time(s) at which to evaluate the model. The shape decides
            whether the times are shared across stars or are per-star --
            nothing is inferred from ``len(t)`` matching ``N_stars``. Accepts
            a scalar (one time, every star), ``(N_times,)`` or
            ``(1, N_times)`` (one shared grid for every star -- always, even
            when ``N_times == N_stars``), or ``(N_stars, N_times)`` (each star
            its own times); for one time per star pass ``t[:, np.newaxis]``.
            Any other shape raises ``ValueError``. See
            :func:`broadcast_times` for the full table.
        fit_params : array-like
            x0, vx, y0, vy in shape (N_fit_params,) or (N_stars, N_fit_params)
        fit_param_errs : array-like, optional
            Uncertainties of fit parameters in shape (N_fit_params,) or (N_stars, N_fit_params), by default None
        fixed_params_dict : dict, optional
            t0, shape (1,) or (N_stars,), by default None.

        Returns
        -------
        x, y (, xe, ye)
            Predicted positions (and uncertainties, if fit_param_errs is provided) with shape (N_stars, N_times), or (N_times,) if N_stars=1, or (N_stars,) if N_times=1
        """
        if fixed_params_dict is None:
            fixed_params_dict = self.fixed_params_dict
        assert 't0' in fixed_params_dict, "Fixed parameter t0 is required for Linear model."
        self._check_param_dimensions(fit_params, fit_param_errs, fixed_params_dict)

        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_fit_params)

        N_stars = fit_params.shape[0]
        # See broadcast_times: t's shape alone says shared-grid vs per-star.
        t_grid = broadcast_times(t, N_stars, caller='Linear.model')  # (N_stars, N_times)
        N_times = t_grid.shape[1]

        x0, vx, y0, vy = fit_params.T  # Each shape (N_stars,)
        t0 = np.broadcast_to(np.atleast_1d(fixed_params_dict['t0']), (N_stars,))

        dt = t_grid - np.asarray(t0)[:, np.newaxis]  # Shape (N_stars, N_times)

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


    def run_fit(self, t, x, y, xe, ye, valid, fixed_params_dict=None, weighting='var',
                       absolute_sigma=True, fill_value=np.nan, verbose=True):
        """
        Batch fit for many stars at once. Linear's weighted least-squares
        fit is closed-form (the normal equations, no iterative optimizer)
        -- so, like Fixed, it doesn't actually need to run one star at a
        time. Rather than building a full (n_epochs, n_epochs) diagonal
        weight matrix and calling np.linalg.pinv/matrix_rank (SVD-based)
        per star -- wasteful work for what's always exactly a 2x2 system
        -- this computes the five weighted sums the 2x2 normal-equations
        matrix needs via vectorized .sum(axis=1) calls across the whole
        batch, and solves/inverts that 2x2 system with its closed-form
        (adjugate-over-determinant) formula.

        Parameters
        ----------
        t, x, y, xe, ye : array-like, shape (n_stars, n_epochs)
            Per-star, per-epoch data. Entries where `valid` is False are
            ignored -- their content does not matter (e.g. they can be NaN
            placeholders for undetected epochs).
        valid : array-like of bool, shape (n_stars, n_epochs)
            Which entries are usable for each star.
        fixed_params_dict : dict, optional
            Must contain 't0', either a scalar or shape (n_stars,), by default None.
        weighting : str, optional
            'var' (w=1/xe**2, 1/ye**2) or 'std' (w=1/xe, 1/ye), by default 'var'
        absolute_sigma : bool, optional
            If False, parameter errors are rescaled by the reduced chi^2, by default True
        fill_value : float, optional
            Fill value for parameters when not enough data points to fit model, by default np.nan
        verbose : bool, optional
            Print warning messages, by default True

        Returns
        -------
        params : ndarray, shape (n_stars, 4) -- [x0, vx, y0, vy]
        param_errs : ndarray, shape (n_stars, 4)
        chi2_x, chi2_y : ndarray, shape (n_stars,)
        """
        assert fixed_params_dict is not None and 't0' in fixed_params_dict, \
            "Linear.run_fit requires fixed_params_dict={'t0': ...}."

        n_stars, n_epochs = t.shape
        t0 = np.broadcast_to(np.atleast_1d(fixed_params_dict['t0']), (n_stars,)).astype(float)
        dt = t - t0[:, np.newaxis]

        n_valid = valid.sum(axis=1)
        has_data = n_valid >= self.n_params  # degree_of_freedom >= 0

        if verbose and np.any(~has_data):
            warnings.warn(
                f'Not enough data points to fit model for {np.sum(~has_data)} star(s). '
                f'Setting parameters to {fill_value} and uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )

        sigma_x, sigma_y = sigma_from_error(xe, ye, weighting=weighting)
        x_wt = weight_from_sigma(sigma_x, valid)
        y_wt = weight_from_sigma(sigma_y, valid)

        dt_m = np.where(valid, dt, 0.0)
        x_m = np.where(valid, x, 0.0)
        y_m = np.where(valid, y, 0.0)

        def solve(wt, val_m):
            # Weighted normal-equations matrix for [v, x0] (basis [dt, 1]):
            #   [[Swdt2, Swdt], [Swdt, Sw]] @ [v, x0] = [Swdtv, Swv]
            # Solved and inverted in closed form (2x2 adjugate/det) rather
            # than via np.linalg.pinv/matrix_rank.
            Sw = wt.sum(axis=1)
            Swdt = (wt * dt_m).sum(axis=1)
            Swdt2 = (wt * dt_m**2).sum(axis=1)
            Swv = (wt * val_m).sum(axis=1)
            Swdtv = (wt * dt_m * val_m).sum(axis=1)

            det = Swdt2 * Sw - Swdt**2
            # Singular (e.g. every valid epoch at the same time): a direct
            # determinant tolerance instead of an SVD-based matrix_rank check.
            scale = np.maximum(Sw * Swdt2, np.finfo(float).tiny)
            singular = has_data & (np.abs(det) <= 1e-12 * scale)

            with np.errstate(divide='ignore', invalid='ignore'):
                v = (Sw * Swdtv - Swdt * Swv) / det
                v0 = (Swdt2 * Swv - Swdt * Swdtv) / det
                v_err = np.sqrt(Sw / det)
                v0_err = np.sqrt(Swdt2 / det)

            if verbose and np.any(singular):
                warnings.warn(
                    'Singular matrix. Covariance of the parameters could not be estimated. '
                    'Setting parameter uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )
            v_err[singular] = np.inf
            v0_err[singular] = np.inf
            # A singular system (e.g. every valid epoch at the same time) has
            # no well-defined [v, x0] split -- only their particular combination
            # is constrained -- so unlike np.linalg.pinv's arbitrary
            # minimum-norm choice, report fill_value here rather than a
            # specific-but-meaningless number. The error is inf either way,
            # so nothing downstream should be trusting this value regardless.
            v[singular] = fill_value
            v0[singular] = fill_value

            return v0, v, v0_err, v_err, singular

        x0, vx, x0e, vxe, singular_x = solve(x_wt, x_m)
        y0, vy, y0e, vye, singular_y = solve(y_wt, y_m)

        params = np.column_stack([x0, vx, y0, vy])
        param_errs = np.column_stack([x0e, vxe, y0e, vye])

        # chi2 = weighted sum of squared residuals (residual.T @ W @ residual),
        # using the same (weighting-scheme) weights the fit itself used.
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2x = (x_wt * (x_m - (vx[:, np.newaxis] * dt_m + x0[:, np.newaxis]))**2).sum(axis=1)
            chi2y = (y_wt * (y_m - (vy[:, np.newaxis] * dt_m + y0[:, np.newaxis]))**2).sum(axis=1)
        # A singular fit has no real params to compute a residual from
        # (regardless of what fill_value happens to be) -- nan them
        # explicitly rather than relying on fill_value being nan.
        chi2x[singular_x] = np.nan
        chi2y[singular_y] = np.nan

        if not absolute_sigma:
            dof = n_valid - self.n_params
            dof_pos = dof > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                reduced_chi2x = np.where(dof_pos, chi2x / np.where(dof_pos, dof, 1), 1.0)
                reduced_chi2y = np.where(dof_pos, chi2y / np.where(dof_pos, dof, 1), 1.0)
            param_errs[:, 0] = np.where(dof_pos, param_errs[:, 0] * np.sqrt(reduced_chi2x), np.inf)
            param_errs[:, 1] = np.where(dof_pos, param_errs[:, 1] * np.sqrt(reduced_chi2x), np.inf)
            param_errs[:, 2] = np.where(dof_pos, param_errs[:, 2] * np.sqrt(reduced_chi2y), np.inf)
            param_errs[:, 3] = np.where(dof_pos, param_errs[:, 3] * np.sqrt(reduced_chi2y), np.inf)
            if verbose and np.any(has_data & ~dof_pos):
                warnings.warn(
                    'Degree of freedom <= 0 for some star(s). Covariance of the parameters could not be '
                    'estimated. Setting parameter uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )

        # Not-enough-data and singular stars: overwrite with fill_value/inf/nan
        # regardless of whatever the (meaningless, e.g. 0/0, or inf*nan from
        # the absolute_sigma=False rescaling above) computation produced.
        # This must come last -- e.g. the rescaling above would otherwise
        # silently turn a singular star's correct inf error into nan
        # (inf * sqrt(nan) == nan, not inf).
        params[~has_data] = fill_value
        param_errs[~has_data] = np.inf
        chi2x[~has_data] = np.nan
        chi2y[~has_data] = np.nan
        param_errs[singular_x, 0] = np.inf
        param_errs[singular_x, 1] = np.inf
        param_errs[singular_y, 2] = np.inf
        param_errs[singular_y, 3] = np.inf

        return params, param_errs, chi2x, chi2y

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
        t : scalar or array-like
            Time(s) at which to evaluate the model. The shape decides
            whether the times are shared across stars or are per-star --
            nothing is inferred from ``len(t)`` matching ``N_stars``. Accepts
            a scalar (one time, every star), ``(N_times,)`` or
            ``(1, N_times)`` (one shared grid for every star -- always, even
            when ``N_times == N_stars``), or ``(N_stars, N_times)`` (each star
            its own times); for one time per star pass ``t[:, np.newaxis]``.
            Any other shape raises ``ValueError``. See
            :func:`broadcast_times` for the full table.
        fit_params : array-like
            x0, vx, ax, y0, vy, ay in shape (N_fit_params,) or (N_stars, N_fit_params)
        fit_param_errs : array-like, optional
            Fit parameter uncertainties with shape (N_stars, N_fit_params) or (N_fit_params,), by default None
        fixed_params_dict : dict, optional
            t0, shape (1,) or (N_stars,), by default None.

        Returns
        -------
        x, y (, xe, ye)
            Predicted positions (and uncertainties, if fit_param_errs is provided) with shape (N_stars, N_times), or (N_times,) if N_stars=1, or (N_stars,) if N_times=1
        """
        if fixed_params_dict is None:
            fixed_params_dict = self.fixed_params_dict
        assert 't0' in fixed_params_dict, "Fixed parameter t0 is required for Acceleration model."
        self._check_param_dimensions(fit_params, fit_param_errs, fixed_params_dict)

        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_fit_params)

        N_stars = fit_params.shape[0]
        # See broadcast_times: t's shape alone says shared-grid vs per-star.
        t_grid = broadcast_times(t, N_stars, caller='Acceleration.model')  # (N_stars, N_times)
        N_times = t_grid.shape[1]

        x0, vx0, ax, y0, vy0, ay = fit_params.T  # Each shape (N_stars,)
        t0 = np.broadcast_to(np.atleast_1d(fixed_params_dict['t0']), (N_stars,))

        dt = t_grid - np.asarray(t0)[:, np.newaxis]  # Shape (N_stars, N_times)

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



    def run_fit(self, t, x, y, xe, ye, valid, fixed_params_dict=None, weighting='var',
                       absolute_sigma=True, fill_value=np.nan, verbose=True):
        """
        Batch fit for many stars at once. Acceleration's model
        (x0 + vx0*dt + 0.5*ax*dt**2) is linear
        in its fit parameters despite being quadratic in time, so its
        weighted least-squares fit is closed-form too -- same situation as
        Linear, just with a 3-parameter (instead of 2-parameter) basis
        [1, dt, 0.5*dt**2] per direction. Unlike Linear, this solves the
        3x3 normal-equations system with a batched np.linalg.inv rather
        than a hand-derived closed-form adjugate -- deriving that by hand
        for a 3x3 system is error-prone for little extra speed over
        LAPACK's own (also closed-form, non-iterative) solver.

        Parameters
        ----------
        t, x, y, xe, ye : array-like, shape (n_stars, n_epochs)
            Per-star, per-epoch data. Entries where `valid` is False are
            ignored -- their content does not matter (e.g. they can be NaN
            placeholders for undetected epochs).
        valid : array-like of bool, shape (n_stars, n_epochs)
            Which entries are usable for each star.
        fixed_params_dict : dict, optional
            Must contain 't0', either a scalar or shape (n_stars,), by default None.
        weighting : str, optional
            'var' (w=1/xe**2, 1/ye**2) or 'std' (w=1/xe, 1/ye), by default 'var'
        absolute_sigma : bool, optional
            If False, parameter errors are rescaled by the reduced chi^2, by default True
        fill_value : float, optional
            Fill value for parameters when not enough data points to fit model, by default np.nan
        verbose : bool, optional
            Print warning messages, by default True

        Returns
        -------
        params : ndarray, shape (n_stars, 6) -- [x0, vx0, ax, y0, vy0, ay]
        param_errs : ndarray, shape (n_stars, 6)
        chi2_x, chi2_y : ndarray, shape (n_stars,)
        """
        assert fixed_params_dict is not None and 't0' in fixed_params_dict, \
            "Acceleration.run_fit requires fixed_params_dict={'t0': ...}."

        n_stars, n_epochs = t.shape
        t0 = np.broadcast_to(np.atleast_1d(fixed_params_dict['t0']), (n_stars,)).astype(float)
        dt = t - t0[:, np.newaxis]

        n_valid = valid.sum(axis=1)
        has_data = n_valid >= self.n_params  # degree_of_freedom >= 0

        if verbose and np.any(~has_data):
            warnings.warn(
                f'Not enough data points to fit model for {np.sum(~has_data)} star(s). '
                f'Setting parameters to {fill_value} and uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )

        sigma_x, sigma_y = sigma_from_error(xe, ye, weighting=weighting)
        x_wt = weight_from_sigma(sigma_x, valid)
        y_wt = weight_from_sigma(sigma_y, valid)

        dt_m = np.where(valid, dt, 0.0)
        dt2_m = 0.5 * dt_m**2
        x_m = np.where(valid, x, 0.0)
        y_m = np.where(valid, y, 0.0)

        def solve(wt, val_m):
            # Weighted normal-equations matrix for [x0, v0, a] built from
            # basis [1, dt, 0.5*dt**2].
            S0 = wt.sum(axis=1)
            S1 = (wt * dt_m).sum(axis=1)
            S2 = (wt * dt2_m).sum(axis=1)
            S11 = (wt * dt_m**2).sum(axis=1)
            S12 = (wt * dt_m * dt2_m).sum(axis=1)
            S22 = (wt * dt2_m**2).sum(axis=1)

            r0 = (wt * val_m).sum(axis=1)
            r1 = (wt * val_m * dt_m).sum(axis=1)
            r2 = (wt * val_m * dt2_m).sum(axis=1)

            M = np.zeros((n_stars, 3, 3))
            M[:, 0, 0] = S0
            M[:, 0, 1] = M[:, 1, 0] = S1
            M[:, 0, 2] = M[:, 2, 0] = S2
            M[:, 1, 1] = S11
            M[:, 1, 2] = M[:, 2, 1] = S12
            M[:, 2, 2] = S22
            r = np.column_stack([r0, r1, r2])

            # Singular (e.g. fewer than 3 unique valid times): a direct
            # determinant tolerance instead of an SVD-based matrix_rank check.
            det = np.linalg.det(M)
            scale = np.maximum(np.abs(S0 * S11 * S22), np.finfo(float).tiny)
            singular = has_data & (np.abs(det) <= 1e-12 * scale)
            unsafe = singular | ~has_data

            if verbose and np.any(singular):
                warnings.warn(
                    'Singular matrix. Covariance of the parameters could not be estimated. '
                    'Setting parameter uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )

            # Stars with too little/degenerate data would otherwise send a
            # singular matrix into np.linalg.inv for the whole batch (which
            # raises, unlike a per-star pinv) -- swap those in for a safe
            # placeholder first; their real params/errs get overwritten
            # with fill_value/inf below regardless of what this produces.
            M_safe = M.copy()
            M_safe[unsafe] = np.eye(3)

            cov = np.linalg.inv(M_safe)
            params = np.einsum('nij,nj->ni', cov, r)
            param_errs = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))

            params[unsafe] = fill_value
            param_errs[unsafe] = np.inf

            return params, param_errs, singular

        x_params, x_errs, singular_x = solve(x_wt, x_m)
        y_params, y_errs, singular_y = solve(y_wt, y_m)

        params = np.column_stack([x_params, y_params])
        param_errs = np.column_stack([x_errs, y_errs])

        # chi2 = weighted sum of squared residuals (residual.T @ W @ residual),
        # using the same (weighting-scheme) weights the fit itself used.
        with np.errstate(divide='ignore', invalid='ignore'):
            x_model = x_params[:, 0:1] + x_params[:, 1:2] * dt_m + x_params[:, 2:3] * dt2_m
            y_model = y_params[:, 0:1] + y_params[:, 1:2] * dt_m + y_params[:, 2:3] * dt2_m
            chi2x = (x_wt * (x_m - x_model)**2).sum(axis=1)
            chi2y = (y_wt * (y_m - y_model)**2).sum(axis=1)
        # A singular fit has no real params to compute a residual from
        # (regardless of what fill_value happens to be) -- nan them
        # explicitly rather than relying on fill_value being nan.
        chi2x[singular_x] = np.nan
        chi2y[singular_y] = np.nan

        if not absolute_sigma:
            dof = n_valid - self.n_params
            dof_pos = dof > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                reduced_chi2x = np.where(dof_pos, chi2x / np.where(dof_pos, dof, 1), 1.0)
                reduced_chi2y = np.where(dof_pos, chi2y / np.where(dof_pos, dof, 1), 1.0)
            for jj in range(3):
                param_errs[:, jj] = np.where(dof_pos, param_errs[:, jj] * np.sqrt(reduced_chi2x), np.inf)
                param_errs[:, 3 + jj] = np.where(dof_pos, param_errs[:, 3 + jj] * np.sqrt(reduced_chi2y), np.inf)
            if verbose and np.any(has_data & ~dof_pos):
                warnings.warn(
                    'Degree of freedom <= 0 for some star(s). Covariance of the parameters could not be '
                    'estimated. Setting parameter uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )

        # Not-enough-data and singular stars: overwrite with fill_value/inf/nan
        # regardless of whatever the (meaningless, e.g. 0/0, or inf*nan from
        # the absolute_sigma=False rescaling above) computation produced.
        # This must come last -- e.g. the rescaling above would otherwise
        # silently turn a singular star's correct inf error into nan
        # (inf * sqrt(nan) == nan, not inf).
        params[~has_data] = fill_value
        param_errs[~has_data] = np.inf
        chi2x[~has_data] = np.nan
        chi2y[~has_data] = np.nan
        for jj in range(3):
            param_errs[singular_x, jj] = np.inf
            param_errs[singular_y, 3 + jj] = np.inf

        return params, param_errs, chi2x, chi2y

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
            Time array in MJD, **in the TDB scale**. parallax_in_direction
            declares its input TDB without converting, so a UTC MJD would be
            read as TDB and shift every epoch by ~69 s. Callers converting from
            decimal years should use
            ``Time(t, format='decimalyear', scale='utc').tdb.mjd``.
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
            Parallax vector of shape (N_stars, 2, N_times), where 2 corresponds to (x, y) components.
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

    def model(self, t, fit_params, fit_param_errs=None, fixed_params_dict=None):
        """Model positions (and uncertainties, if fit_param_errs is provided) at time t of Parallax model.

        Parameters
        ----------
        t : scalar or array-like
            Time(s) at which to evaluate the model. The shape decides
            whether the times are shared across stars or are per-star --
            nothing is inferred from ``len(t)`` matching ``N_stars``. Accepts
            a scalar (one time, every star), ``(N_times,)`` or
            ``(1, N_times)`` (one shared grid for every star -- always, even
            when ``N_times == N_stars``), or ``(N_stars, N_times)`` (each star
            its own times); for one time per star pass ``t[:, np.newaxis]``.
            Any other shape raises ``ValueError``. See
            :func:`broadcast_times` for the full table.
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

        fit_params = np.atleast_2d(fit_params)  # (N_stars, N_fit_params)

        N_stars = fit_params.shape[0]
        # See broadcast_times: t's shape alone says shared-grid vs per-star.
        t_grid = broadcast_times(t, N_stars, caller='Parallax.model')  # (N_stars, N_times)
        N_times = t_grid.shape[1]

        x0, vx, y0, vy, pi = fit_params.T  # Each shape (N_stars,)
        # Broadcast every per-star fixed param to (N_stars,), as run_fit does.
        # A scalar ra/dec would otherwise leave the parallax vector with a
        # single row, which only happens to broadcast for a shared time grid
        # and indexes out of bounds for a per-star one.
        t0 = np.broadcast_to(np.atleast_1d(fixed_params_dict['t0']), (N_stars,))
        ra = np.broadcast_to(np.atleast_1d(fixed_params_dict['ra']), (N_stars,))
        dec = np.broadcast_to(np.atleast_1d(fixed_params_dict['dec']), (N_stars,))
        pa = np.broadcast_to(np.atleast_1d(fixed_params_dict.get('pa', 0.0)), (N_stars,))
        obsLocation = fixed_params_dict.get('obsLocation', 'earth')

        # TODO: vectorize parallax.parallax_in_direction to handle multiple obsLocation?
        assert isinstance(obsLocation, str) or (np.unique(obsLocation).size == 1), "obsLocation must be a single string for all stars at this time."
        if not isinstance(obsLocation, str):
            obsLocation = np.unique(obsLocation)[0]


        dt = t_grid - np.asarray(t0)[:, np.newaxis]  # Shape (N_stars, N_times)

        # parallax_in_direction takes a single shared mjd axis, but t_grid may
        # give each star its own times -- so evaluate it once for the unique
        # times across the whole grid and gather back per (star, epoch). When
        # every star shares one grid (the common case) unique_t is just that
        # grid and this reduces to a reshape, same as run_fit does.
        unique_t, inverse_idx = np.unique(t_grid, return_inverse=True)
        inverse_idx = inverse_idx.reshape(t_grid.shape)
        # Observation epochs come from UTC timestamps, so convert UTC -> TDB
        # properly rather than relabelling the number. parallax_in_direction
        # declares its input TDB without converting, so handing it a bare UTC
        # MJD shifts every epoch by TDB-UTC (69.184 s as of 2026).
        #
        # UTC conversion consults the leap-second table, and ERFA warns
        # ("dubious year") for dates more than ~5 years past the table in the
        # installed pyerfa -- around 2028 for 2.0.1.5. That only affects
        # epochs in the future; any real observation is inside the table.
        t_mjd = Time(unique_t, format='decimalyear', scale='utc').tdb.mjd
        pvec_unique = self.calc_parallax_vector(t_mjd, ra, dec, pa=pa, obsLocation=obsLocation)  # (N_stars, 2, n_unique)
        star_idx = np.arange(N_stars)[:, np.newaxis]
        self.pvec = np.stack(
            [pvec_unique[:, 0, :][star_idx, inverse_idx],
             pvec_unique[:, 1, :][star_idx, inverse_idx]],
            axis=1
        )  # Shape (N_stars, 2, N_times)
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



    def run_fit(self, t, x, y, xe, ye, valid, fixed_params_dict=None, weighting='var',
                       absolute_sigma=True, fill_value=np.nan, verbose=True):
        """
        Batch fit for many stars at once. Parallax's model
        (x0 + vx*dt + pi*Px(t), y0 + vy*dt + pi*Py(t)) is linear in its
        fit parameters once the parallax factors Px, Py are precomputed
        from each star's fixed ra/dec -- so, like Linear/Acceleration, it
        has a closed-form weighted least-squares solution. Unlike those
        two, x and y are NOT independent here: pi is shared between both
        directions (all 5 params are fit jointly from the stacked [x, y]
        data), so this is one coupled 5x5 normal-equations system per
        star -- the x0/vx block and y0/vy block only interact with each
        other through the shared pi row/column -- solved via a batched
        np.linalg.inv.

        Parameters
        ----------
        t, x, y, xe, ye : array-like, shape (n_stars, n_epochs)
            Per-star, per-epoch data. Entries where `valid` is False are
            ignored -- their content does not matter (e.g. they can be NaN
            placeholders for undetected epochs). All stars must share the
            same observation times (parallax_in_direction only supports
            one shared time axis for the whole batch, and ref_table's 2D
            't' column is populated the same way for every star at a
            given epoch) -- t[0] is taken as that shared grid.
        valid : array-like of bool, shape (n_stars, n_epochs)
            Which entries are usable for each star.
        fixed_params_dict : dict, optional
            Must contain 't0', 'ra', 'dec' (each scalar or shape
            (n_stars,)), and optionally 'pa', 'obsLocation', by default None.
        weighting : str, optional
            'var' (w=1/xe**2, 1/ye**2) or 'std' (w=1/xe, 1/ye), by default 'var'
        absolute_sigma : bool, optional
            If False, parameter errors are rescaled by the reduced chi^2, by default True
        fill_value : float, optional
            Fill value for parameters when not enough data points to fit model, by default np.nan
        verbose : bool, optional
            Print warning messages, by default True

        Returns
        -------
        params : ndarray, shape (n_stars, 5) -- [x0, vx, y0, vy, pi]
        param_errs : ndarray, shape (n_stars, 5)
        chi2_x, chi2_y : ndarray, shape (n_stars,)
        """
        assert fixed_params_dict is not None and all(k in fixed_params_dict for k in ['t0', 'ra', 'dec']), \
            "Parallax.run_fit requires fixed_params_dict={'t0':..., 'ra':..., 'dec':...}."

        n_stars, n_epochs = t.shape
        t0 = np.broadcast_to(np.atleast_1d(fixed_params_dict['t0']), (n_stars,)).astype(float)
        ra = np.broadcast_to(np.atleast_1d(fixed_params_dict['ra']), (n_stars,)).astype(float)
        dec = np.broadcast_to(np.atleast_1d(fixed_params_dict['dec']), (n_stars,)).astype(float)
        pa = np.broadcast_to(np.atleast_1d(fixed_params_dict.get('pa', 0.0)), (n_stars,)).astype(float)
        obsLocation = fixed_params_dict.get('obsLocation', 'earth')
        assert isinstance(obsLocation, str) or (np.unique(obsLocation).size == 1), \
            "obsLocation must be a single string for all stars at this time."
        if not isinstance(obsLocation, str):
            obsLocation = np.unique(obsLocation)[0]

        dt = t - t0[:, np.newaxis]

        n_valid = valid.sum(axis=1)
        has_data = n_valid >= self.n_params  # degree_of_freedom >= 0

        if verbose and np.any(~has_data):
            warnings.warn(
                f'Not enough data points to fit model for {np.sum(~has_data)} star(s). '
                f'Setting parameters to {fill_value} and uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )

        # Rows don't have to share the same times (e.g. fit()'s bootstrap
        # resampling gives each row its own resampled subset/order of one
        # star's epochs) -- parallax_in_direction only takes one shared mjd
        # axis, so compute it once for the *unique* times across the whole
        # batch, then gather back per (row, epoch). When every row does
        # share the same grid (the normal multi-star case), unique_t is
        # just that grid and this is a no-op reshape.
        unique_t, inverse_idx = np.unique(t, return_inverse=True)
        inverse_idx = inverse_idx.reshape(t.shape)
        t_mjd = Time(unique_t, format='decimalyear', scale='utc').tdb.mjd
        pvec_unique = self.calc_parallax_vector(t_mjd, ra, dec, pa=pa, obsLocation=obsLocation)  # (n_stars, 2, n_unique_times)
        star_idx = np.arange(n_stars)[:, np.newaxis]
        Px = pvec_unique[:, 0, :][star_idx, inverse_idx]  # (n_stars, n_epochs)
        Py = pvec_unique[:, 1, :][star_idx, inverse_idx]

        sigma_x, sigma_y = sigma_from_error(xe, ye, weighting=weighting)
        x_wt = weight_from_sigma(sigma_x, valid)
        y_wt = weight_from_sigma(sigma_y, valid)

        dt_m = np.where(valid, dt, 0.0)
        x_m = np.where(valid, x, 0.0)
        y_m = np.where(valid, y, 0.0)
        Px_m = np.where(valid, Px, 0.0)
        Py_m = np.where(valid, Py, 0.0)

        # Per-direction weighted sums -- the x0/vx block and y0/vy block
        # never mix with each other, only (separately) with the shared pi
        # row/column below.
        Sx0 = x_wt.sum(axis=1)
        Sx1 = (x_wt * dt_m).sum(axis=1)
        Sx11 = (x_wt * dt_m**2).sum(axis=1)
        SxP = (x_wt * Px_m).sum(axis=1)
        Sx1P = (x_wt * dt_m * Px_m).sum(axis=1)
        SxPP = (x_wt * Px_m**2).sum(axis=1)

        Sy0 = y_wt.sum(axis=1)
        Sy1 = (y_wt * dt_m).sum(axis=1)
        Sy11 = (y_wt * dt_m**2).sum(axis=1)
        SyP = (y_wt * Py_m).sum(axis=1)
        Sy1P = (y_wt * dt_m * Py_m).sum(axis=1)
        SyPP = (y_wt * Py_m**2).sum(axis=1)

        rx0 = (x_wt * x_m).sum(axis=1)
        rx1 = (x_wt * x_m * dt_m).sum(axis=1)
        ry0 = (y_wt * y_m).sum(axis=1)
        ry1 = (y_wt * y_m * dt_m).sum(axis=1)
        # Both x and y data feed into the pi row/column -- this is exactly
        # where x and y stop being independent.
        rP = (x_wt * x_m * Px_m).sum(axis=1) + (y_wt * y_m * Py_m).sum(axis=1)

        M = np.zeros((n_stars, 5, 5))
        M[:, 0, 0] = Sx0
        M[:, 0, 1] = M[:, 1, 0] = Sx1
        M[:, 1, 1] = Sx11
        M[:, 0, 4] = M[:, 4, 0] = SxP
        M[:, 1, 4] = M[:, 4, 1] = Sx1P

        M[:, 2, 2] = Sy0
        M[:, 2, 3] = M[:, 3, 2] = Sy1
        M[:, 3, 3] = Sy11
        M[:, 2, 4] = M[:, 4, 2] = SyP
        M[:, 3, 4] = M[:, 4, 3] = Sy1P

        M[:, 4, 4] = SxPP + SyPP

        r = np.column_stack([rx0, rx1, ry0, ry1, rP])

        # Singular (e.g. fewer than 3 unique valid times, or a star whose
        # parallax factor is degenerate over its valid epochs): a direct
        # determinant tolerance instead of an SVD-based matrix_rank check.
        det = np.linalg.det(M)
        scale = np.maximum(np.abs(Sx0 * Sx11 * Sy0 * Sy11 * M[:, 4, 4]), np.finfo(float).tiny)
        singular = has_data & (np.abs(det) <= 1e-12 * scale)
        unsafe = singular | ~has_data

        if verbose and np.any(singular):
            warnings.warn(
                'Singular matrix. Covariance of the parameters could not be estimated. '
                'Setting parameter uncertainties to np.inf.',
                OptimizeWarning, stacklevel=2
            )

        # Stars with too little/degenerate data would otherwise send a
        # singular matrix into np.linalg.inv for the whole batch (which
        # raises, unlike a per-star pinv) -- swap those in for a safe
        # placeholder first; their real params/errs get overwritten with
        # fill_value/inf below regardless of what this produces.
        M_safe = M.copy()
        M_safe[unsafe] = np.eye(5)

        cov = np.linalg.inv(M_safe)
        params = np.einsum('nij,nj->ni', cov, r)
        param_errs = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))

        params[unsafe] = fill_value
        param_errs[unsafe] = np.inf

        x0, vx, y0, vy, pi = params.T
        with np.errstate(divide='ignore', invalid='ignore'):
            x_model = x0[:, np.newaxis] + vx[:, np.newaxis] * dt_m + pi[:, np.newaxis] * Px_m
            y_model = y0[:, np.newaxis] + vy[:, np.newaxis] * dt_m + pi[:, np.newaxis] * Py_m
            chi2x = (x_wt * (x_m - x_model)**2).sum(axis=1)
            chi2y = (y_wt * (y_m - y_model)**2).sum(axis=1)
        # A singular fit has no real params to compute a residual from
        # (regardless of what fill_value happens to be) -- nan them
        # explicitly rather than relying on fill_value being nan.
        chi2x[singular] = np.nan
        chi2y[singular] = np.nan

        if not absolute_sigma:
            # Unlike Linear/Acceleration (two independent per-direction
            # fits, each rescaled by its own reduced chi2), this is one
            # joint 5-parameter fit over the combined [x, y] data -- so
            # scipy's own curve_fit(absolute_sigma=False) rescales the
            # whole covariance by a single reduced chi2 built from the
            # combined residuals and the true combined degrees of freedom
            # (2*n_valid data points minus all 5 params), not
            # self.n_params (which is only a "min epochs needed" heuristic
            # shared with the has_data check above, not the real dof here).
            dof = 2 * n_valid - self.n_fit_params
            dof_pos = dof > 0
            chi2 = chi2x + chi2y
            with np.errstate(divide='ignore', invalid='ignore'):
                reduced_chi2 = np.where(dof_pos, chi2 / np.where(dof_pos, dof, 1), 1.0)
            for jj in range(5):
                param_errs[:, jj] = np.where(dof_pos, param_errs[:, jj] * np.sqrt(reduced_chi2), np.inf)
            if verbose and np.any(has_data & ~dof_pos):
                warnings.warn(
                    'Degree of freedom <= 0 for some star(s). Covariance of the parameters could not be '
                    'estimated. Setting parameter uncertainties to np.inf.',
                    OptimizeWarning, stacklevel=2
                )

        # Not-enough-data and singular stars: overwrite with fill_value/inf/nan
        # regardless of whatever the (meaningless, e.g. 0/0, or inf*nan from
        # the absolute_sigma=False rescaling above) computation produced.
        # This must come last -- e.g. the rescaling above would otherwise
        # silently turn a singular star's correct inf error into nan
        # (inf * sqrt(nan) == nan, not inf).
        params[~has_data] = fill_value
        param_errs[~has_data] = np.inf
        chi2x[~has_data] = np.nan
        chi2y[~has_data] = np.nan
        param_errs[singular] = np.inf

        return params, param_errs, chi2x, chi2y


def determine_motion_models(startable, motion_models=None, fixed_params_dict=None):
    """Determine, per star, which motion model to use.

    Precedence:

    1. A ``motion_model_input`` column -- the caller's explicit per-star
       request -- wherever that model can actually be evaluated for that star
       (every parameter it needs present and finite). This is the same
       priority fit_motion_models gives the column.
    2. Otherwise the most complex model in `motion_models` whose parameters
       are all present and finite for that star.
    3. `motion_models=None` means "any model", so step 2 becomes "the most
       complex model this star's parameters support".

    The distinction between a restricted list and None is what separates the
    two questions this answers. Which model a star was FIT with is confined to
    the models that were requested, so callers pass their list. How far a star
    must move to reach some other epoch is a property of the star, not of what
    you chose to fit -- a reference imported from an external catalog can carry
    vx/vy/t0 that were never fit here and still has to move with Linear -- so
    propagation passes None.

    Parameters
    ----------
    startable : startable
        Startable with motion model parameter columns
    motion_models : list of MotionModel or str, optional
        List of motion model classes or their names to select from.
        If None, all available motion models will be considered, by default None
    fixed_params_dict : dict, optional
        Dictionary of fixed parameters, by default None

    Returns
    -------
    motion_model_used : list
        List of motion model used for each star
    n_params : list
        List of n parameters per direction for each star
    """

    # Needed unconditionally: both for resolving a list of model names and for
    # resolving 'motion_model_input' requests further down, which are looked up
    # against every known model rather than just the ones passed in.
    all_mm_map = motion_model_map()

    if motion_models is None:
        motion_models = MotionModel.__subclasses__()
    elif all(isinstance(mm, str) for mm in motion_models):
        motion_models = [all_mm_map[mm] for mm in motion_models]

    if fixed_params_dict is None:
        fixed_params_dict = {}

    motion_models_possible = []
    for mm in motion_models:
        required_columns = mm.fit_param_names + mm.fixed_param_names
        req_col_in_table = [col for col in required_columns if (col in startable.colnames)]
        req_col_in_dict = [col for col in required_columns if (col in fixed_params_dict.keys())]
        req_cols = startable[req_col_in_table]
        if all((col in startable.colnames) or (col in fixed_params_dict.keys()) for col in required_columns):
            motion_models_possible.append((mm, req_col_in_table, req_cols, req_col_in_dict))

    # Vectorized replacement for the old per-star Python loop (which called
    # np.isfinite/np.issubdtype once per star per required column -- millions
    # of times for large mosaics). For each candidate motion model, checked in
    # the same priority order as before (last-declared model first), compute a
    # whole-table boolean mask of which stars have all of that model's required
    # *numeric* columns finite, then assign that model to every not-yet-assigned
    # star the mask covers. Whether the fixed_params_dict entries are finite
    # doesn't depend on the star, so it's checked once per model instead of once
    # per star. This makes the `processes`/`chunksize` arguments unnecessary for
    # this function; they are kept in the signature for backward compatibility.
    n_stars = len(startable)
    motion_model_used = np.empty(n_stars, dtype=object)
    n_params = np.empty(n_stars, dtype=int)
    assigned = np.zeros(n_stars, dtype=bool)

    for mm, req_col_in_table, req_cols, req_col_in_dict in motion_models_possible[::-1]:
        fixed_ok = all(
            np.isfinite(fixed_params_dict[col])
            for col in req_col_in_dict
            if np.issubdtype(np.array(fixed_params_dict[col]).dtype, np.number)
        )
        if not fixed_ok:
            continue

        satisfies = np.ones(n_stars, dtype=bool)
        for col in req_col_in_table:
            col_data = req_cols[col]
            if np.issubdtype(col_data.dtype, np.number):
                satisfies &= np.isfinite(col_data)

        newly_assigned = satisfies & ~assigned
        motion_model_used[newly_assigned] = mm.name
        n_params[newly_assigned] = mm.n_params
        assigned |= newly_assigned

    # Highest priority: an explicit per-star request in 'motion_model_input',
    # wherever that model can actually be evaluated for that star. Applied
    # last so it overrides the choice made from `motion_models` above.
    #
    # This is the same priority fit_motion_models already gives the column --
    # it resolves requests through the full model map rather than the
    # restricted list -- so honoring it here keeps the two in agreement
    # instead of having this function silently re-derive something else.
    #
    # "Can be evaluated" means every parameter that model needs is present (a
    # table column or a fixed_params_dict entry) and finite for that star. So
    # a request downgrades by itself when its parameters are missing: a star
    # asking for Acceleration with no ax/ay, or with ax nan because it had too
    # few epochs to fit, falls through to the choice above rather than
    # silently producing nan positions.
    if 'motion_model_input' in startable.colnames:
        requested = np.asarray(startable['motion_model_input'])
        for name in np.unique(requested):
            if name not in all_mm_map:
                # Unrecognized request -- leave those rows as assigned above.
                continue
            mm = all_mm_map[name]
            rows = np.flatnonzero(requested == name)
            if rows.size == 0:
                continue

            usable = np.ones(rows.size, dtype=bool)
            for col in mm.fit_param_names + mm.fixed_param_names:
                if col in startable.colnames:
                    col_data = np.asarray(startable[col][rows])
                    if np.issubdtype(col_data.dtype, np.number):
                        usable &= np.isfinite(col_data)
                elif col in fixed_params_dict:
                    value = np.asarray(fixed_params_dict[col])
                    if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                        usable[:] = False
                else:
                    # The requested model needs something this table lacks.
                    usable[:] = False
                    break

            honored = rows[usable]
            motion_model_used[honored] = mm.name
            n_params[honored] = mm.n_params
            assigned[honored] = True

    # Stars that matched no motion model are dropped, matching the old
    # behavior of simply never appending an entry for them.
    motion_model_used = motion_model_used[assigned].tolist()
    n_params = n_params[assigned].tolist()

    return motion_model_used, n_params


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

    # Callers (e.g. align.update_ref_table_aggregates) may pass one entry per
    # star -- mostly repeats of the same handful of motion model names/classes.
    # Re-expanding fit_param_names/fixed_param_names for every repeat is pure
    # waste, since list_add() is a no-op for names already seen. Dedup up front
    # (preserving first-occurrence order, which is what determines the order of
    # list_of_parameters below) so each distinct motion model is expanded once.
    seen = set()
    unique_motion_models = []
    for mm in motion_models:
        key = mm if isinstance(mm, str) else id(mm)
        if key not in seen:
            seen.add(key)
            unique_motion_models.append(mm)
    motion_models = unique_motion_models

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

def organize_motion_models(motion_models):
    """
    Organize a list of motion models of type str or MotionModel into a list of MotionModel classes,
    sorted by increasing number of required parameters. Empty and Fixed are always added if not already present.
    To be used in align and StarTable.fit_motion_models.

    Parameters
    ----------
    motion_models : MotionModel, str, or list of MotionModels/strings.
        Motion model(s) to organize. Names are matched case-insensitively --
        'linear' and 'Linear' are the same model -- and only the canonical
        spelling propagates, so the caller's casing never reaches the output.

    Returns
    -------
    list
        List of MotionModel classes sorted by increasing number of required parameters.
    """

    all_mm_map = motion_model_map()

    def class_from_name(name):
        """
        Resolve one model name to its class, case-insensitively.

        Every model name is a single word ('Empty', 'Fixed', 'Linear',
        'Acceleration', 'Parallax'), so str.capitalize() is an exact
        normalization: it upper-cases the first character and lower-cases the
        rest, mapping 'linear', 'LINEAR' and 'lInEaR' all onto 'Linear'. Only
        the canonical name goes any further -- what is returned is the class
        itself, and the name that reaches the ref_table comes from that class's
        .name attribute, so nothing downstream ever sees the caller's casing.
        """
        canonical = name.capitalize()
        assert canonical in all_mm_map.keys(), \
            f"motion_model must be in {list(all_mm_map.keys())}, but got '{name}'"
        return all_mm_map[canonical]

    # Change to list if not
    motion_model_classes = []
    if motion_models is None:
        motion_models = [Empty, Fixed]
    elif isinstance(motion_models, str):
        motion_model_classes = [class_from_name(motion_models)]
    elif isinstance(motion_models, type) and issubclass(motion_models, MotionModel):
        motion_model_classes = [motion_models]
    elif isinstance(motion_models, (list, tuple, np.ndarray)):
        for mm in motion_models:
            if isinstance(mm, str):
                motion_model_classes.append(class_from_name(mm))
            else:
                assert issubclass(mm, MotionModel), f"motion_model must be a string or a MotionModel object, but got {type(mm)}"
                motion_model_classes.append(mm)

    mm_names = [mm.name for mm in motion_model_classes]
    if 'Empty' not in mm_names:
        motion_model_classes.append(all_mm_map['Empty'])
    if 'Fixed' not in mm_names:
        motion_model_classes.append(all_mm_map['Fixed'])

    # Sort by increasing n_params
    motion_model_classes = sorted(motion_model_classes, key=lambda mm: mm.n_params)
    return motion_model_classes
