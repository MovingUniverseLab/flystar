# Plan: Keplerian `Orbit` motion model

Plan only. No source changes in this branch. The base is `mm_rework_lingfeng`.

This adds a Keplerian model for Galactic Center stars orbiting Sgr A*, read from `orbits.dat`, and wires it into `MosaicToRef` so those stars are propagated to each epoch while every other star keeps using `Empty`, `Fixed`, `Linear`, `Acceleration`, or `Parallax`.

Two classes, because `fit_param_names` and `n_params` are class attributes. `Orbit` predicts from the catalog elements and does not solve for them. `OrbitFit` fits the six elements into new columns and leaves the input `orb_*` columns alone. A generic align option says which motion models are not refit. Fixed `Orbit` stars are the usual entry in that list. `OrbitFit` is a later phase, after fixed mode works.

## Comparison with the existing framework

`Orbit` and `OrbitFit` are direct subclasses of `MotionModel`. Fixed mode adds the shared-machinery edits at the end of this section. Fitting mode adds four more, also listed there: case-insensitive name lookup, an optional diagnostics return from `run_fit`, a demotion exception so `OrbitFit` is not rewritten as a simpler model, and a partial restore so an unfrozen `OrbitFit` star is still solved when `update_ref_orig` would have held it.

### The base class today

`MotionModel` is an abstract base class in `flystar/motion_model.py`. One instance is one functional form, not one star. Fit results are arguments and return values. They are not stored on the instance, apart from the `fixed_params_dict` that `fit` remembers so a later `model` call can reuse it (`motion_model.py:366-369`).

Class attributes (`motion_model.py:138-152`):

| Attribute | Lines | What it is |
|---|---|---|
| `name` | 138 | Public name. `motion_model_map` keys on the class `__name__` (`motion_model.py:2017-2018`), which matches `name` for every subclass. |
| `fit_param_names` | 141 | Parameters `run_fit` solves for. Order is the x block, then the y block, then any term shared by both. |
| `n_fit_params` | 142 | `len(fit_param_names)`. |
| `n_params` | 143-144 | `int((n_fit_params + 1) / 2)`. Epochs required to fit, and the sort key for complexity. |
| `fixed_param_names` | 148 | Every non-fitted name. Subclasses set this to the required list plus the optional keys. |
| `required_fixed_param_names` | 149 | Must be resolved or `fit_motion_models` raises `KeyError`. |
| `optional_fixed_params` | 150 | `{name: default}`. A missing value falls back to the default inside `fit_motion_models` and `infer_positions`. |
| `fixed_meta_data` | 152 | Declared on the base class and unused. `Parallax` does not set it. |

Methods:

| Method | Lines | Who overrides it |
|---|---|---|
| `model_fit` | 192-193 | Each subclass. Nothing outside the class calls it. |
| `model` | 195-240 | Every subclass. This is prediction and error propagation. `t` goes through `broadcast_times` (`motion_model.py:52`). With `fit_param_errs is None` it returns `(x, y)`. With errors it returns `(x, y, xe, ye)`. |
| `run_fit` | 242-258 | Every subclass. The base raises `NotImplementedError`. Batch in, batch out: `(params, param_errs, chi2x, chi2y)`. |
| `fit` | 260-422 | Not overridden. Builds `valid` from finite `x` and `y`, calls `run_fit`, and bootstraps by calling `run_fit` again. If `t0` is required and missing, it fills the weighted-mean epoch (`motion_model.py:364-365`). |
| `calc_chi2` | 446-461 | Not overridden. Calls `model` and sums squared residuals. |
| `_check_param_dimensions` | 167-190 | Not overridden. A fixed-parameter array must be a scalar or length `N_stars`. |

There is no separate `predict` method and no separate error-propagation method. Both are `model`.

How a model is selected and used:

1. `motion_model_map()` (`motion_model.py:2009-2022`) discovers direct subclasses. `organize_motion_models` (`motion_model.py:2024`) sorts the caller's list by `n_params` and always adds `Empty` and `Fixed`.
2. `StarTable.fit_motion_models` (`startables.py:856`) assigns `motion_model_used`. With a `motion_model_input` column, the request stands when `n_fit >= n_params`; otherwise `np.digitize` demotes the star (`startables.py:1242-1255`). It then calls `run_fit` through `MotionModel.fit` for each used model.
3. `StarTable.infer_positions` (`startables.py:1683`) calls `determine_motion_models` with `motion_models=None` (`startables.py:1739-1741`), groups rows by the chosen name, and calls `model` (`startables.py:1830-1836`). It honors `motion_model_input` when that model can be evaluated. It does not read `motion_model_used`.
4. `MosaicToRef` (`align.py:2650`) subclasses `MosaicSelfRef`. Its constructor takes `motion_models=['Empty', 'Fixed']` and `fixed_params_dict=None` (`align.py:2704-2705`). `update_ref_table_aggregates` (`align.py:1818`) sends stars with at most one valid epoch to `combine_lists_xym` and the rest to `fit_motion_models` (`align.py:1926-1942`). `get_ref_list_from_table` (`align.py:2132`) propagates with `infer_positions(epoch, fixed_params_dict=self.fixed_params_dict)` (`align.py:2194-2196`). The fitting list and the propagation choice are separate: propagation is not restricted to `self.motion_models` (`align.py:2190-2193`).

### Subclasses side by side

`n_params` is `int((n_fit_params + 1) / 2)` for every row. That formula gives 0 for `Orbit` and 3 for `OrbitFit`.

| | Empty | Fixed | Linear | Acceleration | Parallax | Orbit (fixed) | OrbitFit (fit) |
|---|---|---|---|---|---|---|---|
| Fit parameters | none | `x0`, `y0` | `x0`, `vx`, `y0`, `vy` | `x0`, `vx0`, `ax`, `y0`, `vy0`, `ay` | `x0`, `vx`, `y0`, `vy`, `pi` | none | `fit_orb_P`, `fit_orb_t0`, `fit_orb_e`, `fit_orb_i`, `fit_orb_Omega`, `fit_orb_omega`. These names are not the input `orb_*` columns, so the fit does not overwrite the reference elements |
| Required fixed | none | none | `t0` | `t0` | `t0`, `ra`, `dec` | `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, `orb_omega` | the same six `orb_*` elements, used only as the starting guess |
| Optional fixed | none | none | none | none | `pa=0`, `obsLocation='earth'` | `mass=4.0e6` Msun, `dist=8.0e3` pc, `x_bh=0`, `y_bh=0`, `vx_bh=0`, `vy_bh=0`, `t_bh=2000` | the same seven. `mass`, `dist`, and the black-hole offsets are not fit. A joint black-hole fit across stars is future work |
| Catalog columns | none | `x0`, `y0` and `_err` | those plus `vx`, `vy` and `_err`, and `t0` | those plus `vx0`, `ax`, `vy0`, `ay` and `_err`, and `t0` | Linear's columns plus `pi`, `pi_err`, `ra`, `dec`, `pa`, `obsLocation` | the six `orb_*` elements. `mass`, `dist`, and the black-hole offsets land in `meta` when uniform, otherwise in columns. File fields `A` and `search` are parsed to check the line and are not written to the catalog | input `orb_*` left as written, plus `fit_orb_*`, `fit_orb_*_err`, `fit_orb_cov` (6×6), `fit_orb_converged`, `fit_orb_n_iter`. `chi2_x` and `chi2_y` are the existing chi-squared columns. `A` and `search` are still not stored |
| Meaning of `t0` | none | none | Epoch of `x0`, `y0`, `vx`, `vy`. `fit` fills the weighted-mean epoch when `t0` is missing (`motion_model.py:364-365`) | Epoch of `x0`, `y0`, `vx0`, `vy0`, `ax`, `ay`. Same default fill | Same epoch as `Linear`, plus the epoch subtracted before the parallax vector | `orb_t0` is periapse time. It is a different column from stellar `t0`. `t_bh` is the epoch of the black-hole offset | `orb_t0` is the initial periapse time. `fit_orb_t0` is the fitted periapse time. Neither is stellar `t0` |
| `n_params` and demotion | `n_params` is the fewest distinct epochs the model needs. `Empty` needs 0, which is the floor. | `Fixed` needs 1. A star with one valid epoch stays `Fixed`. A star with none becomes `Empty`. | `Linear` needs 2. A star with fewer than two distinct valid epochs is demoted to `Fixed` or `Empty` (`startables.py:1244-1255`). | `Acceleration` needs 3. That is the same number as `Parallax`. With no `motion_model_input` column the fitter chooses by `n_params` alone and requires unique values, so both in one list raises (`startables.py:1082-1086`). | `Parallax` needs 3. Same collision with `Acceleration` when the column is absent. | `Orbit` needs 0. `n_fit` is the number of distinct valid epochs for that star, and it is never negative, so `n_fit < n_params` never happens and an Orbit star is never demoted (`startables.py:1244-1255`). `Empty` also needs 0. Without `motion_model_input`, selection is by `n_params` alone and those values must be unique (`startables.py:1082-1086`), so `Orbit` and `Empty` in one list raises. The orbits reader always writes `motion_model_input='Orbit'`, so the catalogs this plan attaches do not hit that check. | `OrbitFit` has six fit parameters, so `n_params = int((6+1)/2) = 3`. Each epoch supplies two sky coordinates, so six elements need at least three epochs. That is the same `n_params` as `Acceleration` and `Parallax`, and a list with no `motion_model_input` column raises (`startables.py:1082-1086`). With the column, `n_fit < 3` would normally demote the star (`startables.py:1244-1255`). `OrbitFit` sets `demote = False`, so it is not demoted. The solver is skipped, and prediction uses the input elements. See section 2A. |
| Fittable | `run_fit` returns the fill value. Nothing is solved | Closed-form weighted mean | Closed-form 2x2 normal equations | Closed-form quadratic | Closed-form joint 5-parameter fit. `pi` is shared by x and y | `run_fit` returns arrays with shape `(n_stars, 0)`, like `Empty`. Elements are not solved | Nonlinear least squares. `run_fit` calls `scipy.optimize.least_squares` once per star. The batch signature is unchanged. Other models stay closed-form |
| Prediction | NaN at every time (`motion_model.py:517-518`) | `x0`, `y0`, constant in time (`motion_model.py:602`) | `x0 + vx*(t - t0)` (`motion_model.py:791`, `854-855`) | `x0 + vx0*dt + 0.5*ax*dt**2` (`motion_model.py:1073`) | `x0 + vx*dt + pi*pvec_x`, and the same in y (`motion_model.py:1408-1409`) | Newtonian `kep2xyz` east/north, then `x = x_bh + vx_bh*(t - t_bh) - r_east` and `y = y_bh + vy_bh*(t - t_bh) + r_north`. The signs are hardcoded | The same sky formula. Use `fit_orb_*` when all six are finite. Otherwise use the input `orb_*` |
| Error propagation | `xe = ye = inf` when errors are requested | `x0_err`, `y0_err` broadcast across time (`motion_model.py:659-660`) | `hypot(x0_err, vx_err*dt)` (`motion_model.py:867-868`) | `sqrt(x0_err**2 + (vx0_err*dt)**2 + (0.5*ax_err*dt**2)**2)` (`motion_model.py:1132-1133`) | That linear sum plus `(pi_err * pvec)**2` (`motion_model.py:1510-1511`) | `xe = ye = 0` when errors are requested. No fit parameters and no element uncertainties, so there is nothing to propagate and no `pos_err` knob | Numerical Jacobian of `(x, y)` with respect to the six elements, times `fit_orb_cov`. Diagonal `_err` values alone are not propagated. A singular covariance (exactly 3 epochs) returns `xe = ye = inf`. There is no `pos_err` knob |
| `fixed_motion_models` / `fix_motion` | Does not exist yet. After the change, a listed model or a `True` `fix_motion` row keeps its input columns and is not demoted. Unlisted stars are unchanged | same rule | same rule. This is how one `Linear` star stays frozen while another is refit | same rule | same rule | same rule. The recommended align passes `fixed_motion_models=['Orbit']`. `Orbit` has no `x0` or `vx`. An unfrozen Orbit star still has every existing fit-parameter column it does not own cleared by the reset (`startables.py:1450-1461`): `x0`, `y0`, `vx`, `vy`, `vx0`, `ax`, `vy0`, `ay`, `pi`, and their `_err` columns, whenever those columns are already on the table. The six `orb_*` elements are fixed parameters and are not cleared. Freezing skips the reset, which is what keeps the catalog's `x0`, `y0`, `vx`, and `vy`. | Listing `Orbit` does not freeze `OrbitFit`. Listing `OrbitFit`, or setting `fix_motion`, does. An unfrozen `OrbitFit` star is fit. The input `orb_*` columns are fixed parameters and stay. The reset would still clear `x0` and `vx`; section 2A.6 restores those columns after the fit and keeps `fit_orb_*`. |

### Class skeletons

`Parallax` as it exists, cut down to the declarations and the real signatures. The body of `model` is the formula at `motion_model.py:1408-1409` and `1510-1511`. `run_fit` is the batched solve at `motion_model.py:1521`.

```python
import numpy as np


class Parallax(MotionModel):
    """Linear proper motion plus parallax.

    RA and Dec are J2000 degrees. ``pa`` is the counterclockwise offset of
    the image y-axis from north, in degrees.
    """

    name = "Parallax"
    fit_param_names = ['x0', 'vx', 'y0', 'vy', 'pi']
    required_fixed_param_names = ['t0', 'ra', 'dec']
    optional_fixed_params = {'pa': 0., 'obsLocation': 'earth'}
    fixed_param_names = (
        required_fixed_param_names + list(optional_fixed_params.keys())
    )
    n_fit_params = len(fit_param_names)
    n_params = int((n_fit_params + 1) / 2)  # 3

    def model(self, t, fit_params, fit_param_errs=None,
              fixed_params_dict=None):
        """Predict positions, and uncertainties if errors are given.

        Parameters
        ----------
        t : scalar or array-like
            Shared time grid or per-star times. See ``broadcast_times``.
        fit_params : array-like, shape (5,) or (n_stars, 5)
            ``x0``, ``vx``, ``y0``, ``vy``, ``pi``.
        fit_param_errs : array-like, optional
            Same shape as ``fit_params``. Omit to skip uncertainties.
        fixed_params_dict : dict, optional
            Required keys ``t0``, ``ra``, ``dec``. Optional ``pa`` and
            ``obsLocation``.

        Returns
        -------
        x, y : ndarray
            Predicted positions.
        xe, ye : ndarray
            Returned only when ``fit_param_errs`` is given.
        """
        # x = x0 + vx * (t - t0) + pi * pvec_x
        # y = y0 + vy * (t - t0) + pi * pvec_y

    def run_fit(self, t, x, y, xe, ye, valid, fixed_params_dict=None,
                weighting='var', absolute_sigma=True, fill_value=np.nan,
                verbose=True):
        """Closed-form joint fit of the five parameters.

        Parameters
        ----------
        t, x, y, xe, ye : array-like, shape (n_stars, n_epochs)
            Measurements. Padding epochs are ignored via ``valid``.
        valid : ndarray of bool, shape (n_stars, n_epochs)
            Epochs that enter the fit.
        fixed_params_dict : dict, optional
            Must contain ``t0``, ``ra``, and ``dec``.
        weighting : {'var', 'std'}, optional
            ``'var'`` uses ``1/sigma**2``. ``'std'`` uses ``1/sigma``.
        absolute_sigma : bool, optional
            When False, rescale parameter errors by the reduced chi2.
        fill_value : float, optional
            Parameter value for a star with too few epochs.
        verbose : bool, optional
            Warn when a star cannot be fit.

        Returns
        -------
        params, param_errs : ndarray, shape (n_stars, 5)
            ``[x0, vx, y0, vy, pi]`` and their uncertainties.
        chi2x, chi2y : ndarray, shape (n_stars,)
            Chi-squared in each coordinate.
        """
```

`Orbit` as proposed. Same two methods, same signatures. No new method on the base class.

```python
import numpy as np


class Orbit(MotionModel):
    """Prediction-only Newtonian Keplerian orbit about Sgr A*.

    Elements are fixed. ``mass`` and ``dist`` default to the pair that
    reproduces the ``A`` column of ``orbits.dat`` v2.0.2.

    ``kep2xyz`` returns east and north. This class hardcodes the FlyStar
    frame: ``x = -east`` (west is positive) and ``y = +north``.
    """

    name = "Orbit"
    fit_param_names = []
    required_fixed_param_names = [
        'orb_P', 'orb_t0', 'orb_e', 'orb_i', 'orb_Omega', 'orb_omega',
    ]
    optional_fixed_params = {
        'mass': 4.0e6,
        'dist': 8.0e3,
        'x_bh': 0.0,
        'y_bh': 0.0,
        'vx_bh': 0.0,
        'vy_bh': 0.0,
        't_bh': 2000.0,
    }
    fixed_param_names = (
        required_fixed_param_names + list(optional_fixed_params.keys())
    )
    n_fit_params = len(fit_param_names)
    n_params = int((n_fit_params + 1) / 2)  # 0, same slot as Empty

    def model(self, t, fit_params, fit_param_errs=None,
              fixed_params_dict=None):
        """Predict the star relative to the black-hole offset.

        Parameters
        ----------
        t : scalar or array-like
            Decimal years. Shared grid or per-star times. See
            ``broadcast_times``.
        fit_params : array-like, shape (0,) or (n_stars, 0)
            No free parameters. Accepted so the call matches ``model``.
        fit_param_errs : array-like, optional
            Accepted so the call matches ``model``. There are no
            uncertainties to propagate.
        fixed_params_dict : dict, optional
            The six elements are required. ``mass``, ``dist``, and the
            black-hole offsets fall back to ``optional_fixed_params``.

        Returns
        -------
        x, y : ndarray
            FlyStar frame. ``x = -east``, ``y = +north``, plus the
            black-hole offset.
        xe, ye : ndarray
            Returned only when ``fit_param_errs`` is given. Both are 0.
            ``orbits.dat`` has no element errors, and this model has no
            fit parameters.
        """
        # r_east, r_north, _ = kep2xyz(...)  # arcsec; east, north
        # x = x_bh + vx_bh * (t - t_bh) - r_east
        # y = y_bh + vy_bh * (t - t_bh) + r_north

    def run_fit(self, t, x, y, xe, ye, valid, fixed_params_dict=None,
                weighting='var', absolute_sigma=True, fill_value=np.nan,
                verbose=True):
        """No-op. Elements are not fit parameters.

        Parameters
        ----------
        t, x, y, xe, ye : array-like, shape (n_stars, n_epochs)
            Accepted and unused.
        valid : ndarray of bool, shape (n_stars, n_epochs)
            Accepted and unused.
        fixed_params_dict : dict, optional
            Accepted and unused. Elements stay where the caller put them.
        weighting, absolute_sigma, fill_value, verbose
            Accepted so the signature matches ``MotionModel.run_fit``.

        Returns
        -------
        params, param_errs : ndarray, shape (n_stars, 0)
            Empty. Nothing was solved.
        chi2x, chi2y : ndarray, shape (n_stars,)
            NaN.
        """
```

`OrbitFit` is the fitting-mode class. It is added in phase B, after `Orbit` works. The six `fit_orb_*` names are the columns `fit_motion_models` writes, each with an `_err` column. The input `orb_*` columns are required fixed parameters and are only the starting guess.

```python
import numpy as np


class OrbitFit(MotionModel):
    """Fit a Newtonian orbit. Do not overwrite the reference elements.

    ``fit_orb_*`` are the solution. ``orb_*`` are the starting guess and
    stay on the catalog. The sky frame is the same as ``Orbit``:
    ``x = -east``, ``y = +north``.

    ``mass``, ``dist``, and the black-hole offsets are fixed. They are
    not in ``fit_param_names``.
    """

    name = "OrbitFit"
    fit_param_names = [
        'fit_orb_P', 'fit_orb_t0', 'fit_orb_e', 'fit_orb_i',
        'fit_orb_Omega', 'fit_orb_omega',
    ]
    required_fixed_param_names = [
        'orb_P', 'orb_t0', 'orb_e', 'orb_i', 'orb_Omega', 'orb_omega',
    ]
    optional_fixed_params = {
        'mass': 4.0e6,
        'dist': 8.0e3,
        'x_bh': 0.0,
        'y_bh': 0.0,
        'vx_bh': 0.0,
        'vy_bh': 0.0,
        't_bh': 2000.0,
    }
    fixed_param_names = (
        required_fixed_param_names + list(optional_fixed_params.keys())
    )
    n_fit_params = len(fit_param_names)
    n_params = int((n_fit_params + 1) / 2)  # 3 epochs, see section 2A
    demote = False  # stay OrbitFit when n_fit < 3; see section 2A.3

    def model(self, t, fit_params, fit_param_errs=None,
              fixed_params_dict=None):
        """Predict from the fit when it exists, else from ``orb_*``.

        Parameters
        ----------
        t : scalar or array-like
            Decimal years. See ``broadcast_times``.
        fit_params : array-like, shape (6,) or (n_stars, 6)
            ``fit_orb_P``, ``fit_orb_t0``, ``fit_orb_e``, ``fit_orb_i``,
            ``fit_orb_Omega``, ``fit_orb_omega``.
        fit_param_errs : array-like, optional
            Diagonal errors. Position errors use ``fit_orb_cov`` when
            that column is present, not these diagonals alone.
        fixed_params_dict : dict, optional
            Input ``orb_*`` elements, ``mass``, ``dist``, and the
            black-hole offsets.

        Returns
        -------
        x, y : ndarray
            FlyStar frame. ``x = -east``, ``y = +north``.
        xe, ye : ndarray
            Returned only when errors are requested. Jacobian times
            ``fit_orb_cov``, or ``inf`` when that covariance is singular.
        """

    def run_fit(self, t, x, y, xe, ye, valid, fixed_params_dict=None,
                weighting='var', absolute_sigma=True, fill_value=np.nan,
                verbose=True):
        """Per-star ``least_squares``, started from ``orb_*``.

        Parameters
        ----------
        t, x, y, xe, ye : array-like, shape (n_stars, n_epochs)
            Astrometry. Invalid epochs are marked by ``valid``.
        valid : ndarray of bool, shape (n_stars, n_epochs)
            Epochs that enter the fit.
        fixed_params_dict : dict, optional
            Must contain the six input elements. ``mass`` and ``dist``
            fall back to the class defaults.
        weighting : {'var', 'std'}, optional
            Same meaning as ``Linear.run_fit``.
        absolute_sigma : bool, optional
            When False, scale the covariance by the reduced chi-squared.
        fill_value : float, optional
            Value written when the star is not solved.
        verbose : bool, optional
            Warn when a star is skipped or does not converge.

        Returns
        -------
        params, param_errs : ndarray, shape (n_stars, 6)
            Fitted elements and the square root of the covariance
            diagonal. ``fill_value`` and ``inf`` when not solved.
        chi2x, chi2y : ndarray, shape (n_stars,)
            Weighted squared residuals in each coordinate.
        diagnostics : dict
            ``fit_orb_converged`` (bool), ``fit_orb_n_iter`` (int),
            ``fit_orb_cov`` with shape ``(n_stars, 6, 6)``.
        """
```

### Changes to shared machinery

Phase A does not edit `MotionModel`. Phase B edits `fit` only so a fifth return value from `run_fit` does not raise (item 8). `model`, `run_fit`, and `calc_chi2` keep their signatures. `Orbit` and `OrbitFit` are picked up by `motion_model_map` because each is a direct subclass. The longest existing name is `Acceleration` (12 characters). `Orbit` is 5 and `OrbitFit` is 8, so `_MOTION_MODEL_NAME_WIDTH` in `startables.py:17-18` does not change.

1. **`determine_motion_models` treats optional parameters as optional.** Today `fixed_param_names` includes the optional keys, and both loops require every one of those names to be present. `infer_positions` already falls back to the class default after selection (`startables.py:1806-1818`). The gate runs first, so the default is never reached when the name is absent. The explicit `motion_model_input` loop also skips `table.meta`.

   Candidate loop, before (`motion_model.py:1824-1833`):

```python
required_columns = mm.fit_param_names + mm.fixed_param_names
if all((col in startable.colnames) or (col in fixed_params_dict)
       or (col in meta_keys) for col in required_columns):
    motion_models_possible.append(...)
```

   Candidate loop, after:

```python
required_columns = mm.fit_param_names + mm.required_fixed_param_names
if all((col in startable.colnames) or (col in fixed_params_dict)
       or (col in meta_keys) for col in required_columns):
    # A present optional value that is non-finite still rejects the model.
    # A missing optional value does not. model() uses the class default.
    motion_models_possible.append(...)
```

   Explicit request, before (`motion_model.py:1908-1920`):

```python
for col in mm.fit_param_names + mm.fixed_param_names:
    if col in startable.colnames:
        usable &= np.isfinite(startable[col][rows])  # numeric columns
    elif col in fixed_params_dict:
        ...
    else:
        usable[:] = False
        break
```

   Explicit request, after:

```python
for col in mm.fit_param_names + mm.required_fixed_param_names:
    if col in startable.colnames:
        usable &= np.isfinite(...)
    elif col in fixed_params_dict:
        ...
    elif col in startable.meta:
        ...
    else:
        usable[:] = False
        break
for col, default in mm.optional_fixed_params.items():
    # Absent: leave usable alone. Present and non-finite: usable = False.
    ...
```

   Existing models. `Empty`, `Fixed`, `Linear`, and `Acceleration` have `optional_fixed_params = {}`, so both loops see the same names as today. `Parallax` changes only for a star that is missing `pa` or `obsLocation`. Today that star cannot be selected. After the change it can, with `pa=0` and `obsLocation='earth'`. A `Parallax` star that already carries those values is unchanged. A non-finite `pa` still rejects `Parallax`.

2. **`fit_motion_models` grows an optional argument.** The default is no freeze.

   Before (`startables.py:856`):

```python
def fit_motion_models(self, motion_models=None, fixed_params_dict=None, ...):
```

   After:

```python
def fit_motion_models(self, motion_models=None, fixed_params_dict=None,
                      fixed_motion_models=None, ...):
    # frozen = motion_model_input in fixed_motion_models, or fix_motion
    # Drop frozen rows from select_stars before demotion (line 1242)
    # and before the fit-parameter reset (line 1453).
```

   Existing models. A call that omits `fixed_motion_models` and has no `fix_motion` column takes the same path as today, including demotion and the reset of fit-parameter columns the used model does not own. A caller who passes `fixed_motion_models=['Linear']`, or sets `fix_motion` on one row, freezes those stars and refits the others. That is new behavior only for the rows the caller marked.

3. **`update_ref_table_aggregates` unions the same mask into `keep_orig`.** This is the align path. Stars with one valid epoch never reach `fit_motion_models`; they go through `combine_lists_xym` (`align.py:1909`, `1926-1937`). The mask has to be applied before that split.

   Before (`align.py:1852-1868`):

```python
if (keep_orig is not None) and (np.count_nonzero(keep_orig) > 0):
    vals_orig = {...}          # save m0 and motion columns
    fit_star_idxs = ~keep_orig
else:
    fit_star_idxs = None
```

   After:

```python
frozen = _frozen_motion_mask(self.ref_table, self.fixed_motion_models)
if frozen.any():
    keep_orig = frozen if keep_orig is None else (keep_orig | frozen)
# then the existing save / fit_star_idxs / restore, unchanged
```

   Existing models. With no frozen rows, `keep_orig` is the `update_ref_orig` mask from section 1.6 and the function behaves as it does now. Frozen rows of any model, including `Linear` and `Parallax`, are saved and restored and are absent from both `simple_idxs` and `complex_idxs`, so they are not demoted. Phase B changes the restore for unfrozen `OrbitFit` rows only (item 10).

4. **`MosaicToRef.__init__` stores the list.** `MosaicSelfRef.__init__` sets `self.fixed_motion_models` to an empty collection so the inherited aggregate method can read it.

   Before (`align.py:2704-2705`):

```python
motion_models=['Empty', 'Fixed'],
fixed_params_dict=None,
```

   After:

```python
motion_models=['Empty', 'Fixed'],
fixed_params_dict=None,
fixed_motion_models=None,   # None means freeze nobody
```

   Existing models. Callers that do not pass the new argument get today's refit. Passing `['Orbit']` freezes only stars whose `motion_model_input` is `Orbit`.

5. **New columns and metadata keys.** The reader writes `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, and `orb_omega`. It does not write `A` or `search`. No `catalog_meta_names` hook is added. That hook is not needed: `A` is checked inside the reader test and then dropped, `search` is not a match radius, and uniform values such as `mass` and `dist` already go to `table.meta` through `fit_motion_models` (`startables.py:1422-1429`). The reset loop builds its column set from `fit_param_names` of every subclass (`startables.py:1450-1452`). `Orbit.fit_param_names` is empty, so fixed mode adds no names to that set. `OrbitFit` adds the six `fit_orb_*` names. The reset then clears those columns on any star whose used model is not `OrbitFit`. Columns that already belonged to `Empty`, `Fixed`, `Linear`, `Acceleration`, or `Parallax` are unchanged.

   Existing models. Their columns and their `meta` keys are untouched. The one new effect is on a star whose `motion_model_used` becomes `Orbit` and that is not frozen. `Orbit` fits no parameters, so the reset (`startables.py:1450-1461`) treats every fit-parameter column as not belonging to that star and sets it to the fill value. On a catalog that already has them, that is `x0`, `y0`, `vx`, `vy`, `vx0`, `ax`, `vy0`, `ay`, `pi`, and each `_err` column. The `orb_*` elements are not in that list. Freezing the star skips the reset. An unfrozen `OrbitFit` star is cleared of those same columns, because it does not own them. Item 10 puts `x0`, `y0`, `vx`, and `vy` back and leaves `fit_orb_*` in place.

6. **`n_params` uniqueness.** The assert at `startables.py:1082-1086` is unchanged. `n_params` is the fewest distinct epochs a model needs, and without a `motion_model_input` column the fitter selects by that number alone, so two models that share it raise. `Orbit` and `Empty` both need 0 epochs. `OrbitFit` needs 3, the same as `Acceleration` and `Parallax`. Pairing `Orbit` with `Empty`, or `OrbitFit` with `Acceleration` or `Parallax`, raises when the column is missing. Lists that do not include the new classes are unaffected. The orbits reader always writes `motion_model_input`.

7. **Name lookup, in the fitting phase.** `organize_motion_models` turns a string into a class with `str.capitalize()` (`motion_model.py:2057-2059`). That maps `linear` to `Linear`. It also maps `OrbitFit` to `Orbitfit`, which is not the class name, so the lookup would miss.

   Before:

```python
canonical = name.capitalize()
assert canonical in all_mm_map.keys(), ...
return all_mm_map[canonical]
```

   After:

```python
folded = {key.casefold(): key for key in all_mm_map}
canonical = folded.get(name.casefold())
assert canonical is not None, ...
return all_mm_map[canonical]
```

   Existing models. `linear`, `LINEAR`, and `Linear` still resolve to `Linear`. The same for `Empty`, `Fixed`, `Acceleration`, `Parallax`, and `Orbit`. The only new success is a name whose canonical spelling is not a single capital followed by lowercase letters, which today is `OrbitFit`.

8. **Optional diagnostics from `run_fit`, in the fitting phase.** The base signature still returns four arrays. `OrbitFit` also needs a convergence flag, an iteration count, and the 6×6 covariance. Those are not fit parameters, so they must not go into `fit_param_names` or they would change `n_params`.

   The batch path already returns `run_fit`'s tuple unchanged (`motion_model.py:346-353`). The single-star path and the bootstrap path unpack four names (`motion_model.py:373` and `403`). `fit_motion_models` unpacks four names on the batch call (`startables.py:1584`), the per-star call (`startables.py:1622`), and the pool worker's result (`startables.py:1605`). A fifth value would raise at each of those sites, and the single-star path would drop it even if the unpack were widened, because `fit` returns only the first four when `return_chi2=True` (`motion_model.py:419-420`). The worker (`startables.py:2063`) returns whatever `fit` returns.

   Before, the single-star path:

```python
params, param_errs, chi2_x, chi2_y = self.run_fit(...)
```

   After, a fifth value is optional. The batch path needs no change.

```python
result = self.run_fit(...)
params, param_errs, chi2_x, chi2_y = result[:4]
diagnostics = result[4] if len(result) > 4 else None
```

   `fit_motion_models` does the same slice at the three unpack sites and writes each entry of `diagnostics` as a column on the stars just fit. `fit` with `return_chi2=True` appends `diagnostics` when it is present, so the per-star loop and the worker can see it. The docstring at `motion_model.py:290-294`, which says every `run_fit` is closed-form, gains one sentence: `OrbitFit` loops per star inside `run_fit`.

   Existing models. They keep returning four arrays. Slicing `[:4]` is the same four values, `diagnostics` is absent, and no new column is written. `OrbitFit` returns a dict with `fit_orb_converged`, `fit_orb_n_iter`, and `fit_orb_cov`. `chi2_x` and `chi2_y` stay the coordinate chi-squareds. Their sum is the joint chi-squared. There is no extra chi-squared column.

9. **Do not demote `OrbitFit`, in the fitting phase.** `n_params` stays 3, so the uniqueness assert (`startables.py:1082-1086`) still fires when `OrbitFit` shares a list with `Acceleration` or `Parallax` and the table has no `motion_model_input` column. The demotion test is a separate line.

   Before (`startables.py:1245-1246`):

```python
required_params = np.array([all_mm_map[mm_name].n_params
                            for mm_name in self['motion_model_input']])
reassign_mm = n_fit < required_params
```

   After:

```python
required_params = np.array([all_mm_map[mm_name].n_params
                            for mm_name in self['motion_model_input']])
reassign_mm = n_fit < required_params
# Missing attribute means True. Only OrbitFit sets demote = False.
reassign_mm &= np.array([
    getattr(all_mm_map[name], 'demote', True)
    for name in self['motion_model_input']
])
```

   Existing models. They do not set `demote`, so `getattr` is `True` and `reassign_mm` is unchanged. `Orbit` has `n_params = 0`, so `n_fit < n_params` is already never true. `OrbitFit` stays `OrbitFit` when `n_fit < 3`. `run_fit` then skips the solver, and `model` predicts from `orb_*` (section 2A.3). The base class is not given a new attribute.

10. **Partial restore for an unfrozen `OrbitFit` star, in the fitting phase.** Item 3 unions `frozen` into `keep_orig` and then restores those rows in full. That is still what happens to a frozen star, including a frozen `OrbitFit` star. An unfrozen `OrbitFit` star is different, because `update_ref_orig=False` puts every original row in `keep_orig`. Restoring that row in full would throw away `fit_orb_*`. Leaving it out of the fit would mean it is never solved.

   Before, after item 3's mask is built: every `keep_orig` row is saved and restored, and excluded from both `simple_idxs` and `complex_idxs`.

   After, in `update_ref_table_aggregates`:

```python
frozen = _frozen_motion_mask(self.ref_table, self.fixed_motion_models)
held = frozen if keep_orig is None else (keep_orig | frozen)
# Unfrozen OrbitFit rows stay in the fit. Frozen rows do not.
solve = (~frozen) & (self.ref_table['motion_model_input'] == 'OrbitFit')
exclude = held & ~solve
# Save orb_* and the other models' fit-parameter columns on `solve`.
# Fit the complement of `exclude`.
# Restore `exclude` in full.
# On `solve`, restore only the saved columns. Keep fit_orb_*, the
# diagnostics, chi2_x, chi2_y, n_fit, n_params, and motion_model_used.
```

   Existing models. A row whose `motion_model_input` is not `OrbitFit` has `solve` false, so `exclude` is `held` and the restore is the one in item 3. `Linear`, `Parallax`, and fixed `Orbit` are unchanged. The same partial restore runs when `update_ref_orig` is `True`, `'periter'`, or `'atend'`, on the passes that refit original stars. Section 2A.6 lists the columns.

## 1. Architecture on `mm_rework_lingfeng`

The motion-model machinery on this branch is not the `mm_rework` API. Implementation follows the names below.

### 1.1 Models are discovered, not registered

`flystar/motion_model.py` defines `MotionModel` and the direct subclasses `Empty`, `Fixed`, `Linear`, `Acceleration`, and `Parallax`. `motion_model_map()` builds `{class __name__: class}` from `MotionModel.__subclasses__()`. That map is not recursive, so `Orbit` and `OrbitFit` must subclass `MotionModel` itself. The class name and the `name` attribute are `Orbit` and `OrbitFit`. `organize_motion_models` currently matches names with `str.capitalize()`. That accepts `Orbit`. It rejects `OrbitFit`, because `str.capitalize()` turns it into `Orbitfit`. The fitting phase replaces that with a case-insensitive lookup on the real class name (section "Changes to shared machinery", item 7).

There is no `eval`, no `motion_model_dict` of instances, and no `default_motion_model` string.

### 1.2 What a model declares

From `docs/motion_models.rst`:

| Attribute | Role |
|---|---|
| `name` | String callers pass. |
| `fit_param_names` | Fitted parameters. `x` block, then `y` block, shared terms last. |
| `n_fit_params` | `len(fit_param_names)`. |
| `n_params` | `int((n_fit_params + 1) / 2)`. Epochs required, and the complexity sort key. |
| `required_fixed_param_names` | Must be resolved or fitting raises `KeyError`. |
| `optional_fixed_params` | `{name: default}`. Missing values fall back to the default. |
| `fixed_param_names` | `required_fixed_param_names + list(optional_fixed_params)`. |

`Parallax` is the pattern for shared, non-fitted quantities:

```python
required_fixed_param_names = ['t0', 'ra', 'dec']
optional_fixed_params = {'pa': 0., 'obsLocation': 'earth'}
```

`Parallax()` takes no constructor arguments. `ra` and `dec` are required because a catalog has no universal sky position. `pa` and `obsLocation` have defaults, so they are optional. `Orbit` uses that same split. It does not go back to the old `fixed_meta_data` list or to required constructor arguments. The base-class `fixed_meta_data` attribute is unused by `Parallax`.

Methods to implement:

- `model(t, fit_params, fit_param_errs=None, fixed_params_dict=None)` returns `(x, y)` or `(x, y, xe, ye)`. Time shapes go through `broadcast_times`.
- `run_fit(t, x, y, xe, ye, valid, fixed_params_dict=None, ...)` returns `(params, param_errs, chi2x, chi2y)` for the whole batch. `OrbitFit` may append a diagnostics dict. Callers take the first four values, so the extra value is optional (shared-machinery item 8).

`model_fit` is a local convention. Nothing outside the class calls it.

### 1.3 Where fixed parameters come from

`StarTable.fit_motion_models` and `StarTable.infer_positions` use one order:

1. `fixed_params_dict` (scalar applies to every star; an array must have length `N_stars`)
2. a column of that name
3. `table.meta` of that name
4. for an optional parameter only, the class default

A missing required parameter raises `KeyError`. Fitting writes the values it used back under the same name: one `meta` entry when the value is uniform and no column exists, otherwise a column. A column that disagrees is moved to `<param>_orig` on the first write. Disagreeing metadata is overwritten and not kept.

`MosaicToRef` stores the caller's dict on `self.fixed_params_dict` and passes it into `infer_positions` and `fit_motion_models`.

### 1.4 Choosing a model, and demotion

Two columns:

- `motion_model_input` is the per-star request. It is not filled automatically.
- `motion_model_used` is what the fit wrote.

`StarTable.fit_motion_models` (`flystar/startables.py`):

- With no `motion_model_input` column, each star gets the most complex model in the `motion_models` list with `n_fit >= n_params`. Duplicate `n_params` in that list raises `AssertionError`.
- With the column, the request is kept when `n_fit >= n_params`. Otherwise the star is reassigned with `np.digitize` to the most complex model it can support, from the union of `motion_models` and the names in the column. `Empty` and `Fixed` are always in that union.
- After assignment, any fit-parameter column that the used model does not own is reset to the fill value. Fixed-parameter columns are not in that reset.

There is no `default_motion_model`. `MosaicToRef.__init__` takes `motion_models=['Empty', 'Fixed']`. `organize_motion_models` sorts that list by `n_params`. New unmatched stars get `motion_models[-1].name` as `motion_model_input` inside `add_rows_for_new_stars`.

`motion_model_input` / `motion_model_used` use a derived string width, `_MOTION_MODEL_NAME_WIDTH` in `startables.py`, not a hard-coded `U20`. The longest name on this branch is `Acceleration` (12 characters). `Orbit` is shorter, so the width does not change.

### 1.5 Prediction

The live path is `StarTable.infer_positions`. `MosaicToRef.get_ref_list_from_table` calls it with `self.fixed_params_dict`. `MosaicToRef.fit` calls it again for the final chi-squared. `determine_motion_models` picks the model per star: an explicit `motion_model_input` wins when that model can be evaluated, otherwise the most complex model whose parameters are present and finite.

`StarTable.get_star_positions_at_time` is still in the file and still calls `get_batch_pos_at_time` and `get_one_motion_model_param_names`. Those are gone. Do not extend it. `align.infer_positions` is a thin wrapper around the table method.

### 1.6 How an align refits the reference

`MosaicToRef` subclasses `MosaicSelfRef`. The refit lives in `MosaicSelfRef.update_ref_table_aggregates`, which `MosaicToRef.fit` and `match_and_transform` both call. `fit_velocities` does not exist.

Inside one aggregate update:

1. If `keep_orig` is set, copy `m0`, `motion_model_used`, `n_params`, and the motion-parameter columns of those rows into `vals_orig`.
2. Stars that need an update and have at most one valid epoch go through `combine_lists_xym` (`simple_idxs`). That writes `x0`, `y0`, `m0` and their errors. It does not look at `motion_model_input`.
3. The rest go through `fit_motion_models` (`complex_idxs`).
4. `determine_motion_models` rewrites `motion_model_used` and `n_params`.
5. `vals_orig` is written back onto the `keep_orig` rows.

`keep_orig` comes from `update_ref_orig`:

| `update_ref_orig` | During `match_and_transform` | Final pass in `MosaicToRef.fit` |
|---|---|---|
| `False` | Original reference rows are kept. On intermediate lists, stars with no detection in the current list are kept too. | `keep_orig = ref_orig` |
| `True` | Only stars not yet detected in the current list are kept. The last list of the iteration refits everyone. | `keep_orig = None`. Original catalog values are replaced. |
| `'periter'` | Same as `False` until the last list of the iteration, which refits the original rows. | `keep_orig = None`, because the flag is truthy. |
| `'atend'` | Original rows are kept on every list. | `keep_orig = None`. The final aggregate is the one that refits them. |

`iters` is not a constructor argument. It is the longest of the tolerance and `trans_args` schedules.

The `pdb.set_trace()` that used to sit on a hard-coded iteration is gone.

### 1.7 A gap the orbit work has to close

`determine_motion_models` treats every name in `fixed_param_names` as mandatory. That includes optional parameters. The candidate loop accepts a value from the dict, a column, or metadata. The explicit `motion_model_input` loop (the one orbit stars hit) accepts only a column or `fixed_params_dict`. It does not read metadata, and it does not accept "missing, so use the class default".

`infer_positions` and `fit_motion_models` already fall back to `optional_fixed_params` after the model has been selected. The selection gate runs first, so a model whose only source for an optional parameter is the class default is rejected, and a requested model whose optional parameter lives only in `meta` is rejected too.

Orbit stars are requested by name, and `mass` / `dist` are optional with defaults. Without a change here, `motion_model_input='Orbit'` is dropped unless the caller also puts `mass` and `dist` in a column or in `fixed_params_dict`.

Change, in both loops of `determine_motion_models`:

- A missing optional fixed parameter does not disqualify the model.
- An optional parameter that is present but non-finite still does.
- The explicit-request loop consults `table.meta`, as the candidate loop already does.

Required parameters are unchanged: the six elements must be present and finite. Add a test that a `Parallax` row whose `pa` is only in `meta` survives an explicit `motion_model_input` request, so the meta fix is not orbit-only.

## 2. The `Orbit` model

Fixed mode. `fit_param_names = []`, so `n_fit_params = 0` and `n_params = 0`. `run_fit` returns empty parameter arrays, the same shape contract as `Empty`, and does not touch the elements. Fitting is a different class, `OrbitFit`, in section 2A. The reader sets `motion_model_input='Orbit'`. A caller who wants a fit changes that column to `'OrbitFit'` for those stars.

`n_params` is the fewest distinct epochs the model needs. `Orbit` needs 0, and so does `Empty`. When the table has no `motion_model_input` column, `fit_motion_models` picks a model from `n_params` alone and raises if two models share a value (`startables.py:1082-1086`). Listing `Orbit` and `Empty` together then raises. The orbits reader always writes `motion_model_input`, which is how those two models share a catalog. Do not invent a different `n_params` to dodge the check.

### 2.1 Parameters

| Kind | Names |
|---|---|
| Required fixed | `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, `orb_omega` |
| Optional fixed | see the table below |

`Orbit()` takes none of these as constructor arguments. The defaults live on `optional_fixed_params`, which is how this branch gives `Parallax` its `pa=0` and `obsLocation='earth'`. Overrides use the normal order: `fixed_params_dict`, then a column, then `meta`, then the default.

| Name | Default | Unit | Meaning |
|---|---|---|---|
| `mass` | `4.0e6` | solar masses | Black-hole mass. |
| `dist` | `8.0e3` | pc | Distance `R0`. |
| `x_bh` | `0.0` | arcsec, FlyStar frame | Black-hole `x` at `t_bh`. `+x` is west. |
| `y_bh` | `0.0` | arcsec, FlyStar frame | Black-hole `y` at `t_bh`. `+y` is north. |
| `vx_bh` | `0.0` | arcsec/yr, FlyStar frame | Black-hole proper motion. |
| `vy_bh` | `0.0` | arcsec/yr, FlyStar frame | Black-hole proper motion. |
| `t_bh` | `2000.0` | decimal year | Epoch of `(x_bh, y_bh)`. |

`mass = 4.0e6` and `dist = 8.0e3` are the values that reproduce the `A` column of `orbits.dat` v2.0.2. They are not the gcwork `Constants` pair (`4.07e6` Msun, `7960.1` pc). A caller who wants the gcwork pair passes them explicitly:

```python
MosaicToRef(..., fixed_params_dict={'mass': 4.07e6, 'dist': 7960.1})
```

Because the defaults are uniform, the first fit that actually includes an `Orbit` star stores them in `table.meta` unless a column already exists. `MosaicToRef.fixed_params_dict` outranks both.

`t_bh` is the epoch of the black-hole offset. It is not periapse and it is not the stellar `t0`. With the black hole at the origin and at rest, `t_bh` drops out.

Sky position at decimal year `t`. The signs are part of the model, not parameters. `parallax.py` uses the same frame: `x = -east * cos(pa) + north * sin(pa)` at `pa = 0` is `x = -east`.

```text
r_east, r_north, r_los = kep2xyz(...)     # arcsec, relative to the black hole
x = x_bh + vx_bh * (t - t_bh) - r_east    # +x is west
y = y_bh + vy_bh * (t - t_bh) + r_north   # +y is north
```

`r_los` is not written to the catalog. There is no `x_sign` or `y_sign` argument.

### 2.2 Catalog columns from `orbits.dat`

v2.0.2, whitespace-separated, no header. Thirty-two stars. Columns in order:

| File column | Unit | Catalog column | Used to predict? |
|---|---|---|---|
| name | | `name` | match key only |
| P | yr | `orb_P` | yes |
| A | mas | not stored | parsed so the line can be checked. The `a_mas` test compares it with `P` and the default mass and distance, then drops it |
| t0 | decimal year | `orb_t0` | yes, time of periapse |
| e | | `orb_e` | yes |
| i | deg | `orb_i` | yes |
| Omega | deg | `orb_Omega` | yes, PA of the ascending node |
| omega | deg | `orb_omega` | yes, argument of periapse |
| search | pix | not stored | parsed so the line can be checked. Not a match radius |

`A` and `search` never become catalog columns, metadata, or model parameters. No `catalog_meta_names` list is added for them. Nothing else in this plan needs that hook: the six elements are ordinary columns, and `mass`, `dist`, and the black-hole offsets already use the fixed-parameter lookup in section 1.3.

Do not reuse `t0`, `x0`, `y0`, `vx`, or `vy` for elements. `t0` is the epoch of the linear model and is replaced by the weighted mean detection time. `x0` and `vx` are the linear model's position and proper motion. An instantaneous orbital velocity is a different number; writing it into `vx` would make `x0 + vx*(t - t0)` wrong for any star later demoted to `Linear`.

Non-orbit stars have NaN in the `orb_*` columns. Their `motion_model_input` stays whatever the catalog already said.

### 2.3 Errors

`orbits.dat` has no uncertainties, and `Orbit` has no fit parameters, so there is nothing to propagate. When `model` is asked for errors it returns `xe = ye = 0`. When it is not asked, it returns only `x, y`, and `infer_positions` fills `inf` if the table has no error columns (`startables.py:1754-1763`). There is no `pos_err` argument. `OrbitFit` position errors are in section 2A.5.

Matching ignores errors, so a zero reference error does not change who matches. Transform weights do not ignore them. `get_weights_for_lists` turns a non-finite weight into 0, which drops that star from the transform.

- `trans_weights is None`: the zero error is unused.
- `'both,var'`, `'both,std'`, `'list,var'`, `'list,std'`: the science-list variance is what the weight uses. A zero reference error leaves that variance alone, so the star stays in the transform when the science list has `xe` and `ye`.
- `'ref,var'` and `'ref,std'`: the reference variance is 0, the weight is non-finite, and the star is dropped.

That is the same outcome as a `Fixed` or `Linear` reference star whose `x0_err` and `y0_err` are 0. This version does not add a floor. Do not invent a covariance.

### 2.4 What a fit does to an orbit star if it is not frozen

`run_fit` does not change `orb_*`. `Orbit` has no `x0` or `vx` to fit. The reset in `fit_motion_models` (`startables.py:1450-1461`) still clears every fit-parameter column the used model does not own. For an unfrozen star left on `Orbit`, that is every such column already on the table: `x0`, `y0`, `vx`, `vy`, `vx0`, `ax`, `vy0`, `ay`, `pi`, and their `_err` columns. A Galactic Center reference list usually already has `x0`, `y0`, `vx`, and `vy`, and those input values are what get replaced with the fill value. The six element columns are fixed parameters, so they stay. The align option in section 5 skips that reset. Callers who want the input `x0` and `vx` kept pass `fixed_motion_models=['Orbit']`. The model class does not freeze itself. An unfrozen `OrbitFit` star is different: section 2A.6 restores `x0` and `vx` after the fit and keeps `fit_orb_*`.

## 2A. Fitting mode (`OrbitFit`)

Fixed mode stays as section 2. This section is phase B. It does not change `Orbit`.

### 2A.1 One class or two

Two classes. `fit_param_names`, `n_fit_params`, and `n_params` are class attributes. `determine_motion_models`, the demotion check, and the column reset all read those attributes off the class. A flag on one instance would not change which columns get written or how many epochs the star needs. `Orbit` has no fit parameters and `n_params = 0`. `OrbitFit` has six fit parameters and `n_params = 3`, and it sets `demote = False` so that floor does not rewrite the star. The mode for a star is `motion_model_input`: `'Orbit'` or `'OrbitFit'`.

### 2A.2 What is fit

The six elements, by default, and nothing else.

| Quantity | Role in `OrbitFit` |
|---|---|
| `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, `orb_omega` | Required fixed parameters. Starting guess only. Never overwritten. |
| `fit_orb_P`, `fit_orb_t0`, `fit_orb_e`, `fit_orb_i`, `fit_orb_Omega`, `fit_orb_omega` | Fit parameters. The solution and the values used for prediction when they are all finite. |
| `fit_orb_*_err` | Square root of the diagonal of `fit_orb_cov`. Written by the existing `_err` rule. |
| `mass`, `dist`, `x_bh`, `y_bh`, `vx_bh`, `vy_bh`, `t_bh` | Optional fixed parameters, same defaults as `Orbit`. Not fit. |
| `fit_orb_cov` | Per-star array, shape `(6, 6)`, in the order of `fit_param_names`. |
| `fit_orb_converged` | Bool. False when the star was skipped or the solver did not converge. |
| `fit_orb_n_iter` | Int. Function evaluations or iterations reported by `least_squares`. 0 when the solver was not called. |
| `chi2_x`, `chi2_y` | Existing columns. Weighted squared residuals in each sky coordinate. The joint chi-squared is their sum. |

`mass` and `dist` stay at `4.0e6` Msun and `8.0e3` pc unless the caller overrides them. A joint fit of one black-hole mass, distance, or offset shared by many stars is not a per-star `run_fit`. It is future work.

There is no `x_sign`, `y_sign`, `pos_err`, `gr_orbit`, or `rel_redshift`. The sky signs stay hardcoded. `A` and `search` stay unstored. Matching stays on `dr_tol`.

### 2A.3 Epochs and demotion

Six elements and two sky coordinates per epoch. The number of constraints is `2 * n_fit`. The fit is determined when `2 * n_fit >= 6`, so `n_fit >= 3`. The class formula gives the same floor: `n_params = int((6 + 1) / 2) = 3`.

That number is also `Acceleration.n_params` and `Parallax.n_params`. Without `motion_model_input`, `fit_motion_models` raises (`startables.py:1082-1086`). Catalogs that use `OrbitFit` set the column.

`n_fit < 3` would normally demote the star (`startables.py:1244-1255`). Do not demote `OrbitFit`. The class sets `demote = False`. The demotion line keeps a star when that attribute is false and treats a missing attribute as true, so no other class changes (item 9 in the shared-machinery list). The input elements are still a complete predictor. Demoting the star to `Linear` would move it with a proper motion instead of on the orbit, and the reset would then clear `fit_orb_*` because `Linear` does not own those names. Skip the solver, set `fit_orb_converged` false, leave `fit_orb_*` at the fill value, and let `model` predict from `orb_*`. A frozen `OrbitFit` star is not fit and is not demoted, same as any other frozen star.

Exactly three epochs can be solved and have no residual degree of freedom. Call the solver if it converges, and set the covariance, the `_err` columns, and `xe`/`ye` to non-finite. Four or more epochs are where the reported errors are meaningful.

`n_fit` counts distinct valid epochs, the same count the other models use.

### 2A.4 Solver

`run_fit` keeps the batch signature and loops over stars inside it. `docs/motion_models.rst` already describes that pattern for a nonlinear model: the closed-form models stay vectorized, and only the stars assigned to this model pay for a per-star optimizer. A Galactic Center list is tens of orbits, not the whole mosaic. One shared `least_squares` across stars is a poor fit here. Each star has its own elements, bounds, and angle branch.

Use `scipy.optimize.least_squares` with `method='trf'`. The residual is the weighted sky offset, `weighting='var'` or `'std'` the same way as `Linear.run_fit`. Initialize from the input `orb_*` elements. If any of those six is non-finite, do not search. Set `fit_orb_converged` false and the fit columns to the fill value. There is no grid search in this phase.

The internal vector has six unconstrained numbers. The catalog columns stay in the usual units (years, degrees, dimensionless eccentricity).

| Internal parameter | Maps to | Why |
|---|---|---|
| `ln(P)` | `fit_orb_P = exp(ln P)` | Period stays positive. |
| `Δt0` | `fit_orb_t0`, then shifted by an integer number of periods so it lies within half a period of the input `orb_t0` | Periapse time is periodic. The reported value stays near the guess. |
| `h = sqrt(e) cos ω`, `k = sqrt(e) sin ω` | `e = h² + k²`, `ω = atan2(k, h)` in degrees | Puts the eccentricity and the argument of periapse in a form without a hard angle cut. If `h² + k² >= 1`, the residual returns a large penalty so `e` stays below 1. |
| `i` with bounds `(0, 180)` degrees | `fit_orb_i` | Inclination stays in the usual range. |
| `Ω` unbounded, in degrees | `fit_orb_Omega`, wrapped to the turn nearest the input `orb_Omega` | Node angle has no preferred cut. |

After a successful solve, also evaluate the twin `(Ω + 180°, ω + 180°)`. Sky positions are unchanged under that pair, and the line-of-sight velocity flips. This fit has no radial velocities, so the twin is a real degeneracy, not a bug. Keep the branch whose angles are closer to the input `orb_Omega` and `orb_omega`. Do not flip `i`. Radial velocities that would break the degeneracy are future work.

`absolute_sigma=True` uses the covariance `inv(Jᵀ W J)` in the internal parameters, then the analytic Jacobian of the map above to convert that matrix into `fit_orb_cov` in the reported elements. `absolute_sigma=False` multiplies by the reduced chi-squared when the degree of freedom is positive. The diagonal of `fit_orb_cov` supplies `fit_orb_*_err`.

If `least_squares` does not converge, or the star has fewer than three epochs, do not write a partial element set. `fit_orb_*` are the fill value, errors are non-finite, `fit_orb_converged` is false, and `fit_orb_n_iter` records how far the solver got (zero if it was not called). The input `orb_*` columns are not arguments of the write-back, so they stay byte for byte.

### 2A.5 Prediction and position errors

`model` uses `fit_orb_*` when all six are finite. Otherwise it uses `orb_*`, so a skipped or failed fit still places the star on the reference orbit. The sky formula is the one in section 2, including `x = -east` and `y = +north`.

When errors are requested and `fit_orb_cov` is finite, `xe` and `ye` come from a numerical Jacobian of `(x, y)` with respect to the six reported elements:

```text
xe**2 = J_x  fit_orb_cov  J_x.T
ye**2 = J_y  fit_orb_cov  J_y.T
```

The Jacobian is a central difference of the same `kep2xyz` path the prediction uses. Diagonal `fit_orb_*_err` alone is the wrong input. Period, periapse time, eccentricity, and the two angles are correlated, and `Ω` with `ω` is exactly degenerate on the sky. Propagating only the diagonal would invent a position error. If the covariance is missing or singular, return `xe = ye = inf`, not zero. Fixed `Orbit` still returns zero. There is no `pos_err` argument in either class.

### 2A.6 How an align chooses, and what it keeps

`attach_orbits` still sets `motion_model_input='Orbit'`. Fitting is opt-in: set `'OrbitFit'` on the stars that should be solved.

`fixed_motion_models=['Orbit']` freezes fixed orbits and does not freeze `OrbitFit`. Add `'OrbitFit'` to that list, or set `fix_motion`, to hold a fitted star at its input columns and skip the solver. A frozen star of either class is restored in full, including `motion_model_used`.

`update_ref_orig` is unchanged for every star that is not an unfrozen `OrbitFit` star. The table in section 5.3 still describes those stars.

An unfrozen `OrbitFit` star has to be fit even when `update_ref_orig=False` would have put it in `keep_orig`. Otherwise the original-row restore would throw the solution away, or the star would never enter `fit_motion_models`. The rule in `update_ref_table_aggregates`:

1. Build `frozen` as in section 5. Build `held` from `update_ref_orig` and then union `frozen`, as today.
2. Remove unfrozen `OrbitFit` rows from the mask that excludes stars from the fit. They are solved.
3. Save, before the fit, the input `orb_*` columns and the other models' fit-parameter columns (`x0`, `y0`, `vx`, `vy`, and the acceleration and parallax names) on those rows.
4. After the fit, restore those saved columns on the unfrozen `OrbitFit` rows. Do not restore `fit_orb_*`, `fit_orb_*_err`, `fit_orb_cov`, `fit_orb_converged`, `fit_orb_n_iter`, `chi2_x`, `chi2_y`, `n_fit`, `n_params`, or `motion_model_used`.

The input elements therefore survive even if a later edit writes them by mistake. `x0` and `vx` survive the reset that would otherwise clear them, because `OrbitFit` does not own those names. The new solution columns survive. `held` rows that are not unfrozen `OrbitFit` stars are still restored in full.

`update_ref_orig=True`, `'periter'`, and `'atend'` still refit unfrozen `OrbitFit` stars on the passes where original stars are refit. The same partial restore keeps `orb_*`, `x0`, and `vx`. Frozen stars are still not refit on those passes.

## 3. Kepler solver

New module `flystar/orbits.py`. Port the Newtonian solver from the gcwork `Orbit.kep2xyz` / `eccen_anomaly` used by `orbits_jlu_python_gcwork_2024-10-03`. Do not import `gcwork`, `pylab`, or `numpy.core.umath_tests`. No `print` of `sys.path`.

```text
a_AU = (P_yr**2 * M_Msun) ** (1/3)      # Gaussian: years, solar masses, AU
a_arcsec = a_AU / dist_pc
a_mas = a_arcsec * 1000
```

`kep2xyz` takes arrays of epochs and scalar elements. It returns `r_arcsec`, `v_mas_yr`, and `a_mas_yr2`, each shape `(N, 3)`, with index 0 east, 1 north, 2 line of sight. The anomaly is solved with Newton-Raphson. Guard the `sqrt(1 - e**2)` division; the largest eccentricity in this file is about 0.98.

The port is Newtonian only. It has no GR periapse-advance term and no relativistic redshift term, and no flags that would turn them on. Those paths are future work (section 9).

With `M = 4.0e6` and `R0 = 8000`, `a_mas` from the printed `P` matches column `A` for all 32 stars to well under 0.02 mas. The largest residual is S0-105, about 0.013 mas, because `P` is printed as `152.76` (0.01 yr). That is the rounding of `P`, not a different mass. The test asserts `abs(a_mas - A) <= 0.02` for every star. It reads `A` from the file while it parses the line. It does not write `A` onto a catalog. The gcwork cross-check does not use these defaults; it passes `mass=4.07e6` and `dist=7960.1` and compares sky positions to `kep2xyz`.

## 4. Reader

`flystar/orbits.py` function `read_orbits_dat(path) -> astropy.Table`. Each data line has nine whitespace-separated fields and no header. The parser reads all nine, so a short or long line fails. The returned table has `name`, `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, and `orb_omega` only. Names stay as written (`S0-2`, not a different spelling). `A` and `search` are not columns of that table.

`attach_orbits(starlist, orbits)` matches on `name`:

- Matched rows: set `motion_model_input` to `'Orbit'` and copy the six element columns.
- Unmatched catalog rows: leave `motion_model_input` unchanged and set those six columns to NaN.
- Names in the orbit file that are not in the catalog: warn, do not add rows.
- Do not add `orb_A`, `orb_search`, or any metadata entry for `A` or `search`.

Attach is explicit. Loading a starlist does not look for `orbits.dat`.

Matching uses `MosaicToRef`'s `dr_tol` for every star, including stars with `motion_model_input='Orbit'`. The `search` field in the file is not a radius. There is no per-star search radius and no later phase that adds one.

## 5. Not refitting selected models

### 5.1 Choice of mechanism

An aligner argument, not a flag on the model class.

```python
MosaicToRef(..., fixed_motion_models=['Orbit'])
```

`fixed_motion_models` is a sequence of model names. Default `None`, which means no model is frozen. A star is frozen when either of these is true:

- its `motion_model_input` is in `fixed_motion_models`, or
- the boolean column `fix_motion` is present and `True` for that row.

`StarTable.fit_motion_models` takes the same argument and honors the same column. `MosaicToRef` passes the list through. `update_ref_table_aggregates` is the method that has to apply it, because the one-epoch fast path never reaches `fit_motion_models`.

Why this and not a class attribute such as `Orbit.refit = False`:

- A class flag freezes every star of that class, in every align. The requested test freezes one `Linear` star and refits the other `Linear` stars in the same table. A name list cannot say that either, which is why the per-star `fix_motion` column exists. The name list is the convenient form for "every `Orbit` star".
- A class default would change behavior when the caller does not pass the option. The default must stay identical to today.
- `motion_models`, `fixed_params_dict`, and `update_ref_orig` are already aligner settings. This is the same kind of setting.
- Phase A does not fit elements. Phase B fits them on `OrbitFit`, and only for stars the caller marks with `motion_model_input`. The freeze stays the caller's choice. It is not a property of either class.

`MosaicSelfRef.__init__` sets `self.fixed_motion_models` to an empty collection so the inherited aggregate method is safe. Only `MosaicToRef.__init__` exposes the argument. `getattr` is a reasonable belt if a subclass forgets the attribute.

### 5.2 What "not refit" means

For a frozen star, every motion-parameter column from the input reference catalog is unchanged after every iteration and in the final table. That includes `x0`, `vx`, the `orb_*` elements, `motion_model_used`, and `n_params`. Stars of other models, and unfrozen stars of the same model, are refit exactly as they are today.

Implementation: build the frozen mask at the start of `update_ref_table_aggregates` and union it into `keep_orig` before the save. If `keep_orig` was `None`, the mask becomes the new `keep_orig`. The existing `vals_orig` save then covers the frozen rows, those rows are absent from both `simple_idxs` and `complex_idxs`, and the restore at the end writes the saved values back after `determine_motion_models`.

`fit_motion_models` does the same exclusion when it is called directly. Frozen rows are taken out of `select_stars` before the demotion block and before the fit-parameter reset, and the scatter back into the parent table does not write them. Excluding them only inside `fit_motion_models` is not enough for the aligner: `combine_lists_xym` would still overwrite `x0` and `y0` for a frozen star with one detection.

Do not special-case the string `'Orbit'` inside either function. The freeze mask is the whole policy for fixed mode. Unfrozen `OrbitFit` stars are the one exception, and only in phase B: they stay in the fit when `keep_orig` would have held every original row, and only their solution columns are kept afterward. Section 2A.6 has the steps. Input `orb_*` columns are still not overwritten.

### 5.3 `update_ref_orig`

The frozen mask is applied on top of the table in section 1.6. It does not change what `update_ref_orig` does to everyone else.

| Setting | Unfrozen original stars | Frozen stars |
|---|---|---|
| `False` | Already preserved by `keep_orig = ref_orig`. | Also preserved. The union is redundant for original rows and necessary for a frozen star that is not `ref_orig`. |
| `True` | Refit on the last list of each iteration, and again in the final pass. | Saved and restored on those passes too. Input values survive. |
| `'periter'` | Refit on the last list of each iteration. | Held on that last list, not only on the intermediate lists. |
| `'atend'` | Held during the iteration, refit in the final pass. | Held during the iteration and still held in the final pass. |

The saved values are the values at the start of that aggregate call. For a frozen star that was also frozen on the previous call, those are still the input catalog values.

### 5.4 Demotion and new stars

`n_params` is the minimum number of distinct epochs a model needs. `n_fit` is the per-star count of distinct valid epochs. Demotion is the reassignment in `fit_motion_models` when `n_fit < n_params` (`startables.py:1244-1255`). It only sees stars that were not frozen. A `Linear` star needs two epochs, so one detection becomes `Fixed` when the star is not frozen. A frozen `Linear` star with one detection stays `Linear` and keeps its input `x0` and `vx`.

`motion_models` is still the list demotion chooses from. Freezing `Orbit` does not add `Orbit` to that list and does not change `motion_models[-1]`, so new stars are unaffected.

An `Orbit` star is never demoted. Its `n_params` is 0, and `n_fit` cannot be negative, so `n_fit < n_params` is never true. `Orbit` does not have `x0` or `vx`. The reset in the same function (`startables.py:1450-1461`) still runs for every star that was not frozen, and it clears each fit-parameter column the used model does not own. Those names come from the other models: `x0` and `y0` (`Fixed`); `x0`, `vx`, `y0`, and `vy` (`Linear` and `Parallax`); `vx0`, `ax`, `vy0`, and `ay` (`Acceleration`); and `pi` (`Parallax`), plus each name's `_err` column. A star whose `motion_model_used` is `Orbit` owns none of them. If the input catalog already has those columns, they are set to the fill value for that star, and the errors are set to inf. The six `orb_*` columns are fixed parameters, so this loop does not touch them. Freezing removes the star from `fit_motion_models` before the reset. That is what keeps the catalog's `x0`, `y0`, `vx`, and `vy`.

Without a `motion_model_input` column, selection uses `n_params` alone, and two models with the same `n_params` raise (`startables.py:1082-1086`). `Empty` also has `n_params = 0`, so a fit list that contains both `Orbit` and `Empty` raises when the column is missing. The orbits reader always writes `motion_model_input`.

`OrbitFit` needs three epochs, because six elements and two coordinates per epoch require `n_fit >= 3`. It shares that `n_params` with `Acceleration` and `Parallax`, so it also needs `motion_model_input`. It sets `demote = False`, so it is not demoted when `n_fit < 3`. The solver is skipped and prediction uses `orb_*`. Section 2A.3 is the full account. An unfrozen `OrbitFit` star does not own `x0` or `vx` either. Section 2A.6 restores those columns after the fit so the reset does not discard them, and it leaves `fit_orb_*` in place.

### 5.5 Recommended Galactic Center call

```python
MosaicToRef(
    ref_list,
    starlists,
    dr_tol=...,
    motion_models=['Empty', 'Fixed', 'Linear', 'Acceleration', 'Parallax',
                   'Orbit', 'OrbitFit'],
    fixed_motion_models=['Orbit'],
    fixed_params_dict={'mass': 4.0e6, 'dist': 8.0e3},  # optional; these are the defaults
    update_ref_orig=False,
)
```

`mass` and `dist` can be omitted. Passing them makes the choice visible at the call. `fixed_motion_models=['Orbit']` freezes fixed orbits and leaves `OrbitFit` stars free to be solved. Add `'OrbitFit'` to that list to skip the solver for those stars too. `dr_tol` is the match radius for every star. With `update_ref_orig=False`, unfrozen `OrbitFit` stars are still fit, and their input `orb_*`, `x0`, and `vx` are restored afterward (section 2A.6). Stars whose `motion_model_input` stays `'Orbit'` are unchanged from fixed mode.

## 6. Tests

New files: `flystar/tests/test_orbits.py` and additions to `flystar/tests/test_motion_model.py` and `flystar/tests/test_align.py`. Use the existing pytest style. No live download of gcwork.

Solver and model:

- Circular and eccentric analytic positions match Newtonian `kep2xyz` at a grid of epochs. The port has no GR or redshift switch to test.
- With the black hole at the origin, `x` equals minus the east offset and `y` equals the north offset. The class has no `x_sign` or `y_sign` to override.
- When `fit_param_errs` is passed, `model` returns `xe = ye = 0`. `optional_fixed_params` has no `pos_err`.
- `Orbit()` has `optional_fixed_params['mass'] == 4.0e6` and `optional_fixed_params['dist'] == 8.0e3`. `model` with no `mass` or `dist` uses those. A `fixed_params_dict` override, a column, and a `meta` entry each win in that order.
- For every star in `orbits.dat` v2.0.2, `a_mas` from the printed `P` with those defaults is within 0.02 mas of the file's `A` field. The test reads `A` during the parse. After `read_orbits_dat` and after `attach_orbits`, the table has no `A`, `orb_A`, `search`, or `orb_search` column and no metadata entry for either field.
- Against gcwork `kep2xyz`, pass `mass=4.07e6` and `dist=7960.1` explicitly. Compare east and north at several epochs, including periapse and a time far from it. Do not use the FlyStar defaults for this comparison.
- `read_orbits_dat` returns 32 rows and the S0-2 elements. `attach_orbits` sets `motion_model_input` only on name matches.
- An orbit star and a linear star in one `MosaicToRef` are matched with the same `dr_tol`. No column on the orbit star changes that radius.
- `determine_motion_models` keeps `motion_model_input='Orbit'` when `mass` and `dist` are absent, and when they exist only in `meta`. A non-finite `orb_e` still rejects the request. The same meta path keeps an explicit `Parallax` request whose `pa` is only in `meta`.

Freeze option:

- One table, two `Linear` stars with the same epochs and different motions. `fix_motion` is `True` for the first only. After `fit_motion_models`, the first star's `x0` and `vx` are exactly the input values. The second star's `vx` has changed.
- The same table with `fixed_motion_models=['Linear']` leaves both stars' `x0` and `vx` unchanged.
- With the option unset and no `fix_motion` column, both `Linear` stars are refit. This is the regression check that the default did not change.
- An `Orbit` star in a mixed table, with `fixed_motion_models=['Orbit']`, keeps every `orb_*` value and its input `x0` and `vx` through `MosaicToRef`, including when it has only one detection (the `combine_lists_xym` path) and when `update_ref_orig` is `True`. A neighboring `Linear` star is refit.
- A frozen `Linear` star with one detection stays `motion_model_used='Linear'`. An unfrozen `Linear` star with one detection is demoted to `Fixed`.
- Repeat the mixed table for `update_ref_orig` in `False`, `True`, `'periter'`, and `'atend'`. Frozen values after the final table match the input. Unfrozen original stars change when the flag says they should (`True`, `'periter'`, `'atend'`) and do not change when it is `False`.

Align integration, without the freeze as the thing under test:

- A tiny mosaic: one orbit star plus one `Linear` star, three epochs. At each epoch the reference position of the orbit star matches `model()`, and the linear star matches `x0 + vx*(t - t0)`.
- `update_ref_orig=False` and no freeze: the six `orb_*` elements are still the input elements, because they are fixed parameters. `Orbit` does not fit `x0` or `vx`, but the reset still sets those columns, and any other fit-parameter columns the table already has (`y0`, `vy`, `pi`, `ax`, and the rest listed in section 5.4), to the fill value. The test that wants those input values kept uses the freeze.

Fitting mode, phase B. Fixed-mode tests above still pass when no star is `OrbitFit`.

- Inject an S0-2-like orbit (`e` about 0.9, `P` about 16 yr) at 15 or more epochs spanning at least one period. Add 0.5 mas Gaussian noise. Recover `|ΔP|/P < 0.02`, `|Δe| < 0.02`, `|Δt0| < 0.1` yr, and angle errors under 5 degrees after folding `(Ω, ω)` onto the branch nearer the input elements.
- Inject an S0-16-like orbit (`e` about 0.97, `P` about 55 yr) with epochs that include periapse and 1 mas noise. Recover `|Δe| < 0.03` and `|ΔP|/P < 0.05`, with the same angle rule.
- After every fit, the six input `orb_*` columns are bitwise unchanged.
- With four or more epochs and a converged fit, `fit_orb_*` is finite, each `fit_orb_*_err` is finite and positive, `fit_orb_cov` has shape `(6, 6)` and is symmetric, `fit_orb_converged` is true, and `fit_orb_n_iter` is positive. `xe` and `ye` from `model` are finite and come from that covariance.
- With `n_fit = 2`, the star stays `OrbitFit`, the solver is not treated as converged, `fit_orb_*` is the fill value, and the predicted position matches `Orbit` on the input elements.
- With `n_fit = 3`, a converged fit may exist, and the errors, the covariance used for `xe`/`ye`, and `xe`/`ye` themselves are non-finite.
- One `MosaicToRef` with `fixed_motion_models=['Orbit']` and `update_ref_orig=False`: an `Orbit` star keeps its input elements and does not gain a solution; an `OrbitFit` star keeps its input `orb_*` and `x0`/`vx` and gains `fit_orb_*`; a `Linear` star is refit. All three match with the same `dr_tol`.
- The same align with `fixed_motion_models=['Orbit', 'OrbitFit']` does not populate `fit_orb_*`.

## 7. Decisions

1. **Coordinate frame.** Resolved. Hardcoded inside `Orbit.model` and `OrbitFit.model`: `x = -east`, `y = +north`. There is no `x_sign` or `y_sign` parameter.
2. **Black-hole mass and distance.** Resolved. Optional fixed parameters, defaults `mass=4.0e6` Msun and `dist=8.0e3` pc, same lookup as `Parallax`'s `pa` and `obsLocation`. These defaults reproduce the `A` field of `orbits.dat`. The gcwork pair is an explicit override, not the default.
3. **Fixed versus fitted elements.** Resolved. Two classes, because `fit_param_names` and `n_params` are class attributes. `Orbit` does not fit and has `n_params = 0`, so it is never demoted. `OrbitFit` fits the six elements into `fit_orb_*`, has `n_params = 3` (six elements, two sky coordinates per epoch), and sets `demote = False` so fewer than three epochs do not rewrite it as `Linear`. It does not write the input `orb_*` columns. `mass`, `dist`, and the black-hole offsets stay fixed. A joint black-hole fit across stars is future work.
4. **Which stars are frozen.** Resolved. Caller-supplied `fixed_motion_models` plus an optional `fix_motion` column. The recommended align passes `fixed_motion_models=['Orbit']`, which does not freeze `OrbitFit`. Listing `OrbitFit` or setting `fix_motion` does.
5. **Match radius.** Resolved. Every star, fixed orbit or fitted orbit, uses the same `MosaicToRef` `dr_tol`. The file's `search` column is parsed and discarded. There is no per-star search radius.
6. **Position errors.** Resolved for `Orbit`: `xe = ye = 0` when errors are requested. Resolved for `OrbitFit`: `xe` and `ye` come from the Jacobian and `fit_orb_cov`, or `inf` when that matrix is missing or singular. There is no `pos_err` parameter on either class. For a fixed `Orbit` star, `'ref,var'` and `'ref,std'` still drop the star; schemes that use the science-list errors do not.

Still open, and not blocking the plan:

- Whether to ship `orbits.dat` inside the package or only accept a path. Tests can use a checked-in copy of the 32-line file either way.

## 8. Implementation order

Phase A is fixed mode. Do not start phase B until phase A behaves as this plan says.

Phase A:

1. `flystar/orbits.py`: Newtonian solver only, `kep2xyz`, `read_orbits_dat`, `attach_orbits`. Parse `A` and `search`, do not store them. Tests against the analytic orbit, the `a_mas` check against file `A`, and gcwork with explicit mass and distance.
2. `Orbit` in `motion_model.py`, using `optional_fixed_params` for `mass` and `dist`. Tests for defaults, overrides, and `infer_positions`.
3. `determine_motion_models`: optional defaults count as available, and the explicit-request loop reads `meta`.
4. `fixed_motion_models` on `MosaicToRef`, honored in `update_ref_table_aggregates` and `fit_motion_models`, including the one-epoch path, demotion, and all four `update_ref_orig` settings.
5. One mixed-catalog align test: fixed orbit star frozen, linear star refit, positions at each epoch coming from the right model.

Phase B, fitting, after phase A:

1. Case-insensitive name lookup so `OrbitFit` resolves, and the optional fifth return from `run_fit` for diagnostics. Existing models keep their four-value return.
2. `OrbitFit`: per-star `least_squares`, the internal parameterization, the 180 degree branch choice, and `fit_orb_cov`. Set `demote = False` and teach the demotion line to honor it, so `n_fit < 3` does not rewrite the star. Tests that recover the injected S0-2-like and S0-16-like orbits, leave `orb_*` unchanged, fill the new columns, and skip the solver below three epochs.
3. The `update_ref_table_aggregates` exception in section 2A.6, so an unfrozen `OrbitFit` star is fit under every `update_ref_orig` setting and its input `orb_*`, `x0`, and `vx` stay.
4. One `MosaicToRef` test with a frozen `Orbit` star, an `OrbitFit` star, and a `Linear` star.

## 9. Out of scope

Posterior samples. Light-time delay. Radial velocities, including using them to break the `(Ω + 180°, ω + 180°)` degeneracy. A joint fit of black-hole mass, distance, or offset across stars. GR periapse advance and relativistic redshift: not parameters, not flags, and not branches in `flystar/orbits.py`. A per-star match radius, including any use of the file's `search` column. Editing `MosaicSelfRef` beyond storing an empty freeze list so the shared aggregate method can read it. Any change to stars whose model was not listed and whose `fix_motion` is not set, except the `OrbitFit` partial restore in section 2A.6.
