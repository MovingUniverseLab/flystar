# Plan: Keplerian `Orbit` motion model

Implemented on a branch off `mm_rework_lingfeng`. This file is the plan as approved at commit `911d066`, plus the deviations in section 10. The code and the tests follow section 10 where it disagrees with the text above it.

This adds one Keplerian model, `Orbit`, for Galactic Center stars orbiting Sgr A*. Elements are read from `orbits.dat` and wired into `MosaicToRef` so those stars are propagated to each epoch while every other star keeps using `Empty`, `Fixed`, `Linear`, `Acceleration`, or `Parallax`.

`Orbit` covers a fixed orbit and a fitted orbit. There is no second class. `fit_param_names` are `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, and `orb_omega`. A fit updates those columns in place, the same way `Linear` updates `x0` and `vx`. The new columns are `orb_*_err`, plus `orb_cov`, `orb_fit_converged`, and `orb_fit_n_iter`. Whether a star is fixed or fit is decided only by the generic freeze mechanism: the `fixed_motion_models` list, and a per-star string column `fit_motion`. The same mechanism freezes or refits `Linear` and the other models. Original elements survive when the star is frozen or when `update_ref_orig` holds the row. Fitting the elements is phase B, after fixed prediction and the freeze mechanism work.

## Comparison with the existing framework

`Orbit` is a direct subclass of `MotionModel`. Phase A adds the shared-machinery edits at the end of this section: optional fixed parameters stay optional, a missing `orb_*_err` column does not block prediction, and the freeze list and `fit_motion` column exist. Phase A also stops demotion from rewriting an `Orbit` star that has too few epochs. Phase B adds one more edit: an optional diagnostics return from `run_fit`. Name lookup is unchanged. `str.capitalize()` already maps `orbit` to `Orbit`. There is no second class and no exception that pulls one model out of `keep_orig`.

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
2. `StarTable.fit_motion_models` (`startables.py:856`) assigns `motion_model_used`. With a `motion_model_input` column, the request stands when `n_fit >= n_params`; otherwise `np.digitize` demotes the star (`startables.py:1242-1255`). It then calls `run_fit` through `MotionModel.fit` for each used model. The freeze mask is applied before that demotion.
3. `StarTable.infer_positions` (`startables.py:1683`) calls `determine_motion_models` with `motion_models=None` (`startables.py:1739-1741`), groups rows by the chosen name, and calls `model` (`startables.py:1830-1836`). It honors `motion_model_input` when that model can be evaluated. It does not read `motion_model_used`.
4. `MosaicToRef` (`align.py:2650`) subclasses `MosaicSelfRef`. Its constructor takes `motion_models=['Empty', 'Fixed']` and `fixed_params_dict=None` (`align.py:2704-2705`). `update_ref_table_aggregates` (`align.py:1818`) sends stars with at most one valid epoch to `combine_lists_xym` and the rest to `fit_motion_models` (`align.py:1926-1942`). `get_ref_list_from_table` (`align.py:2132`) propagates with `infer_positions(epoch, fixed_params_dict=self.fixed_params_dict)` (`align.py:2194-2196`). The fitting list and the propagation choice are separate: propagation is not restricted to `self.motion_models` (`align.py:2190-2193`).

### Subclasses side by side

`n_params` is `int((n_fit_params + 1) / 2)` for every row, including `Orbit`.

| | Empty | Fixed | Linear | Acceleration | Parallax | Orbit |
|---|---|---|---|---|---|---|
| Fit parameters | none | `x0`, `y0` | `x0`, `vx`, `y0`, `vy` | `x0`, `vx0`, `ax`, `y0`, `vy0`, `ay` | `x0`, `vx`, `y0`, `vy`, `pi` | `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, `orb_omega`. A fit writes the solution back into these columns, the same way `Linear` writes `x0` and `vx`. A fixed star is not refit, so the columns stay at the catalog values |
| Required fixed | none | none | `t0` | `t0` | `t0`, `ra`, `dec` | none. The elements are fit parameters, not fixed parameters. A fixed star is predicted from those same columns, which the fitter does not rewrite |
| Optional fixed | none | none | none | none | `pa=0`, `obsLocation='earth'` | `mass=4.0e6` Msun, `dist=8.0e3` pc, `x_bh=0`, `y_bh=0`, `vx_bh=0`, `vy_bh=0`, `t_bh=2000`. `mass`, `dist`, and the black-hole offsets are not fit. A joint black-hole fit across stars is future work |
| Catalog columns | none | `x0`, `y0` and `_err` | those plus `vx`, `vy` and `_err`, and `t0` | those plus `vx0`, `ax`, `vy0`, `ay` and `_err`, and `t0` | Linear's columns plus `pi`, `pi_err`, `ra`, `dec`, `pa`, `obsLocation` | the six `orb_*` elements, their `orb_*_err` columns, `orb_cov` (6×6), `orb_fit_converged`, and `orb_fit_n_iter`. `chi2_x` and `chi2_y` are the existing chi-squared columns. `mass`, `dist`, and the black-hole offsets land in `meta` when uniform, otherwise in columns. File fields `a` and `search` are parsed to check the line and are not written to the catalog |
| Meaning of `t0` | none | none | Epoch of `x0`, `y0`, `vx`, `vy`. `fit` fills the weighted-mean epoch when `t0` is missing (`motion_model.py:364-365`) | Epoch of `x0`, `y0`, `vx0`, `vy0`, `ax`, `ay`. Same default fill | Same epoch as `Linear`, plus the epoch subtracted before the parallax vector | `orb_t0` is periapse time. A fit updates that same column. It is not stellar `t0`. `t_bh` is the epoch of the black-hole offset |
| `n_params` and demotion | `n_params` is the fewest distinct epochs the model needs. `Empty` needs 0, which is the floor. | `Fixed` needs 1. A star with one valid epoch stays `Fixed`. A star with none becomes `Empty`. | `Linear` needs 2. A star with fewer than two distinct valid epochs is demoted to `Fixed` or `Empty` (`startables.py:1244-1255`). A frozen `Linear` star is removed before that test, so one detection stays `Linear`. | `Acceleration` needs 3. That is the same number as `Parallax`. With no `motion_model_input` column the fitter chooses by `n_params` alone and requires unique values, so both in one list raises (`startables.py:1082-1086`). | `Parallax` needs 3. Same collision with `Acceleration` when the column is absent. A frozen star is not demoted. | `Orbit` has six fit parameters, so the class formula gives `n_params = 3`. Each epoch supplies two sky coordinates, so three epochs are the algebraic minimum. That is the same `n_params` as `Acceleration` and `Parallax`, and a list with no `motion_model_input` column raises (`startables.py:1082-1086`). The reader always writes `motion_model_input`. A fixed star is removed before demotion, so the test never sees it and the star is never demoted, including with zero epochs. A fit star with fewer than three epochs is not demoted to `Linear`. `Orbit` sets `demote = False`. The solver is skipped and prediction uses `orb_*`. Four epochs are where the covariance is finite. See section 2.5. |
| Fittable | `run_fit` returns the fill value. Nothing is solved | Closed-form weighted mean | Closed-form 2x2 normal equations | Closed-form quadratic | Closed-form joint 5-parameter fit. `pi` is shared by x and y | A fixed star does not call `run_fit`. A fit star, in phase B, calls `scipy.optimize.least_squares` once per star inside `run_fit`, seeded from the current `orb_*`. On failure it returns those same values and sets `orb_fit_converged` false. The batch signature is unchanged. Other models stay closed-form |
| Prediction | NaN at every time (`motion_model.py:517-518`) | `x0`, `y0`, constant in time (`motion_model.py:602`) | `x0 + vx*(t - t0)` (`motion_model.py:791`, `854-855`) | `x0 + vx0*dt + 0.5*ax*dt**2` (`motion_model.py:1073`) | `x0 + vx*dt + pi*pvec_x`, and the same in y (`motion_model.py:1408-1409`) | Newtonian `kep2xyz` east/north, then `x = x_bh + vx_bh*(t - t_bh) - r_east` and `y = y_bh + vy_bh*(t - t_bh) + r_north`. The signs are hardcoded. Fixed and fit both read the current `orb_*` columns |
| Error propagation | `xe = ye = inf` when errors are requested | `x0_err`, `y0_err` broadcast across time (`motion_model.py:659-660`) | `hypot(x0_err, vx_err*dt)` (`motion_model.py:867-868`) | `sqrt(x0_err**2 + (vx0_err*dt)**2 + (0.5*ax_err*dt**2)**2)` (`motion_model.py:1132-1133`) | That linear sum plus `(pi_err * pvec)**2` (`motion_model.py:1510-1511`) | A fixed star returns `xe = ye = 0`. A fit with a finite `orb_cov` returns the numerical Jacobian of `(x, y)` times that covariance. Diagonal `orb_*_err` values alone are not propagated. A fit with no finite covariance returns `xe = ye = inf`. There is no `pos_err` knob |
| `fixed_motion_models` / `fit_motion` | Does not exist yet. After the change, a listed model is frozen unless the row's `fit_motion` says `'fit'`. A row that says `'fixed'` is frozen even if the model is not in the list. A missing or blank cell follows the list. Unset list and unset column means fit, which is today's behavior | same rule. A frozen `Fixed` star keeps `x0` and `y0` | same rule. This is how one `Linear` star stays frozen while another is refit. `keep_orig` restores `x0` and `vx` the same way it will restore `orb_*` | same rule | same rule | same rule. The recommended align passes `fixed_motion_models=['Orbit']`. A row with `fit_motion='fit'` is solved on the passes where `update_ref_orig` refits original stars, and `orb_*` move. A frozen star, or a row held by `update_ref_orig=False`, keeps the input `orb_*`. The reset (`startables.py:1450-1461`) does not clear `orb_*` on an `Orbit` star, because those names belong to it. It still clears `x0` and `vx` on a fit `Orbit` star. Freezing skips that reset |

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

`Orbit` as proposed. Same two methods, same signatures. No new method on the base class. Fixed versus fit is not a class flag. The elements are fit parameters. A fixed star simply is not refit, so `model` reads the catalog columns. Phase B fills in `run_fit`.

```python
import numpy as np


class Orbit(MotionModel):
    """Newtonian orbit. Fixed or fit is chosen per star, not here.

    ``fixed_motion_models`` and the ``fit_motion`` column decide.
    Both modes read ``orb_*``. A fit updates those columns in place.
    A failed fit returns the values it was seeded with.

    The sky frame is hardcoded: ``x = -east``, ``y = +north``.
    ``mass`` and ``dist`` default to the pair that reproduces the
    ``a`` column of ``orbits.dat`` v2.0.2.
    """

    name = "Orbit"
    fit_param_names = [
        'orb_P', 'orb_t0', 'orb_e', 'orb_i', 'orb_Omega', 'orb_omega',
    ]
    required_fixed_param_names = []
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
    n_params = int((n_fit_params + 1) / 2)  # 3; see section 2.5
    demote = False  # too few epochs fall back; see section 2.5
    prediction_columns = ['orb_cov']

    def model(self, t, fit_params, fit_param_errs=None,
              fixed_params_dict=None):
        """Predict from the current ``orb_*`` elements.

        Parameters
        ----------
        t : scalar or array-like
            Decimal years. See ``broadcast_times``.
        fit_params : array-like, shape (6,) or (n_stars, 6)
            ``orb_P``, ``orb_t0``, ``orb_e``, ``orb_i``,
            ``orb_Omega``, ``orb_omega``.
        fit_param_errs : array-like, optional
            Diagonal ``orb_*_err``. Position errors use ``orb_cov``
            when that covariance is finite, not these diagonals alone.
        fixed_params_dict : dict, optional
            ``mass``, ``dist``, and the black-hole offsets.
            ``orb_cov`` is passed here when the column exists.
            ``mass`` and ``dist`` fall back to the defaults.

        Returns
        -------
        x, y : ndarray
            FlyStar frame. ``x = -east``, ``y = +north``.
        xe, ye : ndarray
            Returned only when errors are requested. ``0`` when no
            fit was run. Jacobian times ``orb_cov`` when that matrix
            is finite. ``inf`` when a fit has no finite covariance.
        """
        # r_east, r_north, _ = kep2xyz(...)  # arcsec; east, north
        # x = x_bh + vx_bh * (t - t_bh) - r_east
        # y = y_bh + vy_bh * (t - t_bh) + r_north

    def run_fit(self, t, x, y, xe, ye, valid, fixed_params_dict=None,
                weighting='var', absolute_sigma=True, fill_value=np.nan,
                verbose=True):
        """Per-star ``least_squares``, seeded from the current ``orb_*``.

        On failure, return that seed and flag non-convergence. Do not
        return ``fill_value``. Phase A returns the seed and does not
        call the solver. Phase B is section 2.6. Frozen stars never
        reach this method.

        Parameters
        ----------
        t, x, y, xe, ye : array-like, shape (n_stars, n_epochs)
            Astrometry. Invalid epochs are marked by ``valid``.
        valid : ndarray of bool, shape (n_stars, n_epochs)
            Epochs that enter the fit.
        fixed_params_dict : dict, optional
            ``mass`` and ``dist`` fall back to the class defaults.
            The six elements come from the table, not from here.
        weighting : {'var', 'std'}, optional
            Same meaning as ``Linear.run_fit``.
        absolute_sigma : bool, optional
            When False, scale the covariance by the reduced chi-squared.
        fill_value : float, optional
            Unused for the elements. A failed fit keeps its seed.
        verbose : bool, optional
            Warn when a star is skipped or does not converge.

        Returns
        -------
        params, param_errs : ndarray, shape (n_stars, 6)
            Updated elements, or the seed if not solved. Uncertainties
            are the square root of the covariance diagonal, or ``inf``
            when not solved.
        chi2x, chi2y : ndarray, shape (n_stars,)
            Weighted squared residuals in each coordinate.
        diagnostics : dict
            Phase B only. ``orb_fit_converged`` (bool),
            ``orb_fit_n_iter`` (int), and ``orb_cov`` with shape
            ``(n_stars, 6, 6)``. Phase A returns four arrays.
        """
```

### Changes to shared machinery

Phase A does not edit `model`, `run_fit`, or `calc_chi2` on the base class. Phase B edits `fit` only so a fifth return value from `run_fit` does not raise (item 8). `Orbit` is picked up by `motion_model_map` because it is a direct subclass. The longest existing name is `Acceleration` (12 characters). `Orbit` is 5, so `_MOTION_MODEL_NAME_WIDTH` in `startables.py:17-18` does not change. `organize_motion_models` keeps `str.capitalize()` (`motion_model.py:2057-2059`). That already resolves `Orbit`.

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

2. **Fixed-mode prediction reads the fit-parameter columns.** `orb_*` are `fit_param_names`, so the existing finiteness check is the right gate (`motion_model.py:1824`, `1908-1912`). A fixed star is selectable because the catalog already has finite elements. A missing or non-finite `orb_e` drops the request, the same way a missing `vx` drops `Linear`. There is no `predict_without_fit_params` attribute. A fixed star is predicted from those columns because it was not refit, not because the elements live in a second list.

   `infer_positions` builds `fit_param_errs` by indexing `param + '_err'` whenever `x0_err` and `y0_err` exist (`startables.py:1754-1787`). `orbits.dat` has no uncertainties, and a frozen star never enters the fitter that creates `orb_P_err`. A missing `_err` column becomes an array of `inf`, which is the same default the fitter uses when it adds an error column (`startables.py:1350-1353`). Columns that already exist are read as they are.

   Before:

```python
fit_param_errs = np.array([
    self[param_name + '_err'][unique_index]
    for param_name in motion_model_instance.fit_param_names
]).T
```

   After:

```python
fit_param_errs = np.array([
    self[name][unique_index] if name in self.colnames
    else np.full(len(unique_index), np.inf)
    for name in (p + '_err' for p in motion_model_instance.fit_param_names)
]).T
```

   When `prediction_columns` names a column that exists, copy it into the dict handed to `model`. `Orbit` lists `orb_cov`. A missing covariance column means the fixed-star errors, `xe = ye = 0`. `Orbit.model` does not propagate the diagonal `orb_*_err` values.

   Existing models. They already have their `_err` columns in the catalogs this branch fits, so the new branch is not taken and the numbers are unchanged. A `Linear` star with non-finite `vx` is still rejected.

3. **`fit_motion_models` grows a freeze list and reads `fit_motion`.** The default is no freeze. The column is a string, not a boolean. See section 5 for why, and for precedence.

   Before (`startables.py:856`):

```python
def fit_motion_models(self, motion_models=None, fixed_params_dict=None, ...):
```

   After:

```python
def fit_motion_models(self, motion_models=None, fixed_params_dict=None,
                      fixed_motion_models=None, ...):
    # frozen from fixed_motion_models, then fit_motion overrides per row
    # Drop frozen rows from select_stars before demotion (line 1242)
    # and before the fit-parameter reset (line 1453).
```

   Existing models. A call that omits `fixed_motion_models` and has no `fit_motion` column takes the same path as today, including demotion and the reset of fit-parameter columns the used model does not own. A caller who passes `fixed_motion_models=['Linear']`, or sets `fit_motion='fixed'` on one row, freezes those stars and refits the others. That is new behavior only for the rows the caller marked. `fit_motion='fit'` on one row of a frozen model refits that row and leaves the other rows frozen.

4. **`update_ref_table_aggregates` unions the same mask into `keep_orig`.** This is the align path. Stars with one valid epoch never reach `fit_motion_models`; they go through `combine_lists_xym` (`align.py:1909`, `1926-1937`). The mask has to be applied before that split. There is no later exception that removes a model from this mask.

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

   A non-frozen star whose class sets `demote = False` is not eligible for `simple_idxs`, even with one detection. It goes to `fit_motion_models`. Today only `Orbit` sets that attribute. `combine_lists_xym` would otherwise replace `x0` and `y0` and ignore the orbit. Frozen stars of every model stay out of both `simple_idxs` and `complex_idxs`.

   Existing models. With no frozen rows, `keep_orig` is the `update_ref_orig` mask from section 1.6 and the function behaves as it does now. Frozen rows of any model, including `Linear` and `Parallax`, are saved and restored in full. `fit_motion='fit'` does not pull a star out of the `update_ref_orig` mask. It only stops the freeze list from adding that star.

5. **`MosaicToRef.__init__` stores the list.** `MosaicSelfRef.__init__` sets `self.fixed_motion_models` to an empty collection so the inherited aggregate method can read it.

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

   Existing models. Callers that do not pass the new argument get today's refit. Passing `['Orbit']` freezes stars whose `motion_model_input` is `Orbit`, except rows whose `fit_motion` is `'fit'`.

6. **`orb_*` are fit parameters, so the reset and `keep_orig` already know them.** The reader writes `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, and `orb_omega`. It does not write `a` or `search`, and it does not write a second set of element columns. No `catalog_meta_names` hook is added. That hook is not needed: `a` is checked inside the reader test and then dropped, `search` is not a match radius, and uniform values such as `mass` and `dist` already go to `table.meta` through `fit_motion_models` (`startables.py:1422-1429`).

   The reset builds its column set from `fit_param_names` of every subclass (`startables.py:1450-1452`). `Orbit` adds the six `orb_*` names to that set. A star whose `motion_model_used` is `Orbit` owns them, so the reset does not clear them. The seed is still in the columns when `run_fit` reads it. The ordinary write at `startables.py:1667-1669` then stores whatever `run_fit` returned, which is the same write that stores `x0` and `vx` for `Linear`. On success that is the solution. On failure `run_fit` returns the seed, so the write puts the same numbers back. `orb_*_err` follows the framework's `_err` suffix and is created with the other error columns (`startables.py:1350-1353`).

   A star whose used model is not `Orbit` does not own `orb_*`. If those columns exist, the reset sets them to the fill value and their `_err` columns to `inf`. Non-orbit rows from `attach_orbits` are already NaN there. Frozen stars are not in the fitted subtable (`startables.py:1001-1046`), so the reset never sees them and the scatter does not write them.

   `keep_orig` saves every name from `motion_model_param_names` for the models named on the table (`align.py:1857-1867`), which is `fit_param_names`, each `_err`, and `fixed_param_names` (`motion_model.py:1978-1988`). Once `Orbit` is one of those models, `orb_*` and `orb_*_err` are in that list and are restored with `x0` and `vx` (`align.py:1987-1989`). `orb_cov`, `orb_fit_converged`, and `orb_fit_n_iter` are not fit parameters, so they are not in the save. They are written only for stars that enter `run_fit`, and held rows do not.

   Existing models. Their own columns are unchanged. On a star whose `motion_model_used` is `Orbit` and that is not frozen, the reset still clears every fit-parameter column `Orbit` does not own. On a catalog that already has them, that is `x0`, `y0`, `vx`, `vy`, `vx0`, `ax`, `vy0`, `ay`, `pi`, and each `_err` column. Prediction does not use those columns. It matters only for a reader that wanted the catalog proper motion to survive on the same row as a fitted orbit. Freezing the star, or holding it with `update_ref_orig`, restores them, because the save list includes every model's fit parameters. There is no second restore that puts `x0` back onto a star that was actually fit.

7. **`n_params` uniqueness, and demotion.** The assert at `startables.py:1082-1086` is unchanged. `n_params` is the class formula `int((n_fit_params + 1) / 2)`. Without a `motion_model_input` column the fitter selects by that number alone, so two models that share it raise. `Orbit` needs 3 epochs, the same as `Acceleration` and `Parallax`. Pairing `Orbit` with either of those raises when the column is missing. `Orbit` does not share 0 with `Empty`. Lists that do not include `Orbit` are unaffected. The orbits reader always writes `motion_model_input`.

   Do not set the class `n_params` to 0 to dodge demotion. That would claim six fit parameters need no epochs, and it would collide with `Empty`. Do not set it to 4 either. Four epochs are where the covariance has a residual degree of freedom. The class attribute stays the formula's 3. Section 2.5 is the full account.

   Frozen stars are removed before the demotion block (`startables.py:1244-1255`). Their effective requirement in that test is 0 epochs. The class attribute is not rewritten, and the `n_params` column on a frozen row is left at its input value because the fitter does not write that row. This is the whole of the per-star change, and it applies to every model. A frozen `Linear` star with one detection stays `Linear`. A frozen `Orbit` star with zero detections stays `Orbit`.

   An unfrozen `Orbit` star would still be demoted when `n_fit < 3`, because the class `n_params` is 3. That demotion is the wrong fallback: `Linear` would move the star with a proper motion, and the reset would then clear `orb_*` because `Linear` does not own those names. The only elements would be gone. `Orbit` sets `demote = False`. The base class does not grow the attribute. The skipped solver returns the current `orb_*` values, so the write-back does not replace them with the fill value.

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
# Missing attribute means True. Only Orbit sets demote = False.
reassign_mm &= np.array([
    getattr(all_mm_map[name], 'demote', True)
    for name in self['motion_model_input']
])
```

   Existing models. They do not set `demote`, so `getattr` is `True` and `reassign_mm` is unchanged aside from the frozen rows already removed. `Orbit` stays `Orbit` when `n_fit < 3`. `run_fit` then skips the solver, and `model` predicts from `orb_*`.

8. **Optional diagnostics from `run_fit`, in phase B.** The base signature still returns four arrays. `Orbit` also needs a convergence flag, an iteration count, and the 6×6 covariance. Those are not fit parameters, so they must not go into `fit_param_names` or they would change `n_params`.

   The batch path already returns `run_fit`'s tuple unchanged (`motion_model.py:346-353`). The single-star path and the bootstrap path unpack four names (`motion_model.py:373` and `403`). `fit_motion_models` unpacks four names on the batch call (`startables.py:1584`), the per-star call (`startables.py:1622`), and the pool worker's result (`startables.py:1605`). A fifth value would raise at each of those sites. The worker (`startables.py:2063`) returns whatever `fit` returns.

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

   `fit_motion_models` does the same slice at the three unpack sites and writes each entry of `diagnostics` as a column on the stars just fit. `fit` with `return_chi2=True` appends `diagnostics` when it is present, so the per-star loop and the worker can see it. The docstring at `motion_model.py:290-294`, which says every `run_fit` is closed-form, gains one sentence: `Orbit` loops per star inside `run_fit`.

   Existing models. They keep returning four arrays. Slicing `[:4]` is the same four values, `diagnostics` is absent, and no new column is written. `Orbit` returns a dict with `orb_fit_converged`, `orb_fit_n_iter`, and `orb_cov`. `chi2_x` and `chi2_y` stay the coordinate chi-squareds. Their sum is the joint chi-squared. There is no extra chi-squared column. Phase A does not return the dict.

## 1. Architecture on `mm_rework_lingfeng`

The motion-model machinery on this branch is not the `mm_rework` API. Implementation follows the names below.

### 1.1 Models are discovered, not registered

`flystar/motion_model.py` defines `MotionModel` and the direct subclasses `Empty`, `Fixed`, `Linear`, `Acceleration`, and `Parallax`. `motion_model_map()` builds `{class __name__: class}` from `MotionModel.__subclasses__()`. That map is not recursive, so `Orbit` must subclass `MotionModel` itself. The class name and the `name` attribute are `Orbit`. `organize_motion_models` matches names with `str.capitalize()` (`motion_model.py:2057-2059`). That accepts `Orbit`. No lookup change is required.

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

`Parallax()` takes no constructor arguments. `ra` and `dec` are required because a catalog has no universal sky position. `pa` and `obsLocation` have defaults, so they are optional. `Orbit` uses that same split for `mass`, `dist`, and the black-hole offsets. It does not go back to the old `fixed_meta_data` list or to required constructor arguments. The base-class `fixed_meta_data` attribute is unused by `Parallax`.

Methods to implement:

- `model(t, fit_params, fit_param_errs=None, fixed_params_dict=None)` returns `(x, y)` or `(x, y, xe, ye)`. Time shapes go through `broadcast_times`.
- `run_fit(t, x, y, xe, ye, valid, fixed_params_dict=None, ...)` returns `(params, param_errs, chi2x, chi2y)` for the whole batch. `Orbit` may append a diagnostics dict in phase B. Callers take the first four values, so the extra value is optional (shared-machinery item 8).

`model_fit` is a local convention. Nothing outside the class calls it.

### 1.3 Where fixed parameters come from

`StarTable.fit_motion_models` and `StarTable.infer_positions` use one order:

1. `fixed_params_dict` (scalar applies to every star; an array must have length `N_stars`)
2. a column of that name
3. `table.meta` of that name
4. for an optional parameter only, the class default

A missing required parameter raises `KeyError`. Fitting writes the values it used back under the same name: one `meta` entry when the value is uniform and no column exists, otherwise a column. A column that disagrees is moved to `<param>_orig` on the first write. Disagreeing metadata is overwritten and not kept.

The fitted elements are not fixed parameters, so this write-back does not touch them. They are updated by the fit-parameter write at `startables.py:1667-1669`, the same write that stores `x0` and `vx`. On a failed fit that write stores the seed `run_fit` returned, which is the values already in the columns.

`MosaicToRef` stores the caller's dict on `self.fixed_params_dict` and passes it into `infer_positions` and `fit_motion_models`.

### 1.4 Choosing a model, and demotion

Two columns:

- `motion_model_input` is the per-star request. It is not filled automatically.
- `motion_model_used` is what the fit wrote.

`StarTable.fit_motion_models` (`flystar/startables.py`):

- With no `motion_model_input` column, each star gets the most complex model in the `motion_models` list with `n_fit >= n_params`. Duplicate `n_params` in that list raises `AssertionError`. `Orbit` shares 3 with `Acceleration` and `Parallax`, so those combinations require the column.
- With the column, the request is kept when `n_fit >= n_params`. Otherwise the star is reassigned with `np.digitize` to the most complex model it can support, from the union of `motion_models` and the names in the column. `Empty` and `Fixed` are always in that union. Frozen stars are removed before this test. `Orbit` also sets `demote = False`, so an unfrozen orbit star is not reassigned when `n_fit < 3`.
- After assignment, any fit-parameter column that the used model does not own is reset to the fill value. Fixed-parameter columns are not in that reset. Frozen stars are removed before the reset.

There is no `default_motion_model`. `MosaicToRef.__init__` takes `motion_models=['Empty', 'Fixed']`. `organize_motion_models` sorts that list by `n_params`. New unmatched stars get `motion_models[-1].name` as `motion_model_input` inside `add_rows_for_new_stars`.

`motion_model_input` / `motion_model_used` use a derived string width, `_MOTION_MODEL_NAME_WIDTH` in `startables.py`, not a hard-coded `U20`. The longest name on this branch is `Acceleration` (12 characters). `Orbit` is shorter, so the width does not change.

### 1.5 Prediction

The live path is `StarTable.infer_positions`. `MosaicToRef.get_ref_list_from_table` calls it with `self.fixed_params_dict`. `MosaicToRef.fit` calls it again for the final chi-squared. `determine_motion_models` picks the model per star: an explicit `motion_model_input` wins when that model can be evaluated, otherwise the most complex model whose parameters are present and finite.

`StarTable.get_star_positions_at_time` is still in the file and still calls `get_batch_pos_at_time` and `get_one_motion_model_param_names`. Those are gone. Do not extend it. `align.infer_positions` is a thin wrapper around the table method.

For `Orbit`, "can be evaluated" means the six `orb_*` elements are present and finite. They are fit parameters, and the catalog supplies them. A missing `orb_*_err` does not drop the star. Item 2 in the shared-machinery list is that error-column lookup.

### 1.6 How an align refits the reference

`MosaicToRef` subclasses `MosaicSelfRef`. The refit lives in `MosaicSelfRef.update_ref_table_aggregates`, which `MosaicToRef.fit` and `match_and_transform` both call. `fit_velocities` does not exist.

Inside one aggregate update:

1. If `keep_orig` is set, copy `m0`, `motion_model_used`, `n_params`, and the motion-parameter columns of those rows into `vals_orig`.
2. Stars that need an update and have at most one valid epoch go through `combine_lists_xym` (`simple_idxs`). That writes `x0`, `y0`, `m0` and their errors. It does not look at `motion_model_input`. A non-frozen `Orbit` star skips this fast path (item 4).
3. The rest go through `fit_motion_models` (`complex_idxs`).
4. `determine_motion_models` rewrites `motion_model_used` and `n_params`.
5. `vals_orig` is written back onto the `keep_orig` rows, in full.

`keep_orig` comes from `update_ref_orig`:

| `update_ref_orig` | During `match_and_transform` | Final pass in `MosaicToRef.fit` |
|---|---|---|
| `False` | Original reference rows are kept. On intermediate lists, stars with no detection in the current list are kept too. | `keep_orig = ref_orig` |
| `True` | Only stars not yet detected in the current list are kept. The last list of the iteration refits everyone. | `keep_orig = None`. Original catalog values are replaced. |
| `'periter'` | Same as `False` until the last list of the iteration, which refits the original rows. | `keep_orig = None`, because the flag is truthy. |
| `'atend'` | Original rows are kept on every list. | `keep_orig = None`. The final aggregate is the one that refits them. |

The freeze mask is unioned into that `keep_orig` for every model. A row with `fit_motion='fit'` is not frozen, and it is not removed from the `update_ref_orig` mask. Held rows are restored in full. Because `orb_*` are fit parameters, `motion_model_param_names` includes them and their `_err` columns, and the restore writes those saved values back (`align.py:1857-1867`, `1987-1989`). That is the same restore that puts `x0` and `vx` back on a held `Linear` star.

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

Required element columns are unchanged in spirit: the six `orb_*` values must be present and finite, because they are now fit parameters and the existing gate already requires that. Add a test that a `Parallax` row whose `pa` is only in `meta` survives an explicit `motion_model_input` request, so the meta fix is not orbit-only. `mass` can be missing. `orb_P` cannot.

## 2. The `Orbit` model

One class. `fit_param_names` is `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, and `orb_omega`, so `n_fit_params = 6` and `n_params = 3`. `required_fixed_param_names` is empty. The elements are not fixed parameters. A fixed star uses them because the fitter never replaces them. A fit star uses them as the seed and then as the updated solution. The reader sets `motion_model_input='Orbit'`. It does not set `fit_motion`. A caller who wants every attached orbit held passes `fixed_motion_models=['Orbit']`. A caller who wants some of them solved sets `fit_motion='fit'` on those rows.

Phase A implements `model` and a `run_fit` that returns the current `orb_*` values and does not call the solver. Phase B replaces that body with the solver in section 2.6. The declarations do not change between the phases.

### 2.1 Parameters

| Kind | Names |
|---|---|
| Fit | `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, `orb_omega` |
| Uncertainties | `orb_P_err`, `orb_t0_err`, `orb_e_err`, `orb_i_err`, `orb_Omega_err`, `orb_omega_err` |
| Required fixed | none |
| Optional fixed | see the table below |
| Diagnostics, not fit parameters | `orb_cov`, `orb_fit_converged`, `orb_fit_n_iter` |

`Orbit()` takes none of these as constructor arguments. The defaults live on `optional_fixed_params`, which is how this branch gives `Parallax` its `pa=0` and `obsLocation='earth'`. Overrides use the normal order: `fixed_params_dict`, then a column, then `meta`, then the default.

| Name | Default | Unit | Meaning |
|---|---|---|---|
| `mass` | `4.0e6` | solar masses | Black-hole mass. Not fit. |
| `dist` | `8.0e3` | pc | Distance `R0`. Not fit. |
| `x_bh` | `0.0` | arcsec, FlyStar frame | Black-hole `x` at `t_bh`. `+x` is west. Not fit. |
| `y_bh` | `0.0` | arcsec, FlyStar frame | Black-hole `y` at `t_bh`. `+y` is north. Not fit. |
| `vx_bh` | `0.0` | arcsec/yr, FlyStar frame | Black-hole proper motion. Not fit. |
| `vy_bh` | `0.0` | arcsec/yr, FlyStar frame | Black-hole proper motion. Not fit. |
| `t_bh` | `2000.0` | decimal year | Epoch of `(x_bh, y_bh)`. Not fit. |

`mass = 4.0e6` and `dist = 8.0e3` are the values that reproduce the `a` column of `orbits.dat` v2.0.2. They are not the gcwork `Constants` pair (`4.07e6` Msun, `7960.1` pc). A caller who wants the gcwork pair passes them explicitly:

```python
MosaicToRef(..., fixed_params_dict={'mass': 4.07e6, 'dist': 7960.1})
```

Because the defaults are uniform, the first fit that actually includes an `Orbit` star stores them in `table.meta` unless a column already exists. `MosaicToRef.fixed_params_dict` outranks both.

`t_bh` is the epoch of the black-hole offset. It is not periapse and it is not the stellar `t0`. With the black hole at the origin and at rest, `t_bh` drops out.

A joint fit of one black-hole mass, distance, or offset shared by many stars is not a per-star `run_fit`. It is future work.

Sky position at decimal year `t`. The signs are part of the model, not parameters. `parallax.py` uses the same frame: `x = -east * cos(pa) + north * sin(pa)` at `pa = 0` is `x = -east`. The same formula is used for a fixed star and a fit star. Only the element source changes.

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
| P | yr | `orb_P` | yes. A fit updates this column |
| a | mas | not stored | parsed so the line can be checked. The on-disk file is unchanged; code calls this field `a` (it used to be written `A`). The `a_mas` test compares it with `P` and the default mass and distance, then drops it |
| t0 | decimal year | `orb_t0` | yes, time of periapse. A fit updates this column |
| e | | `orb_e` | yes. A fit updates this column |
| i | deg | `orb_i` | yes. A fit updates this column |
| Omega | deg | `orb_Omega` | yes, PA of the ascending node |
| omega | deg | `orb_omega` | yes, argument of periapse |
| search | pix | not stored | parsed so the line can be checked. Not a match radius |

`a` and `search` never become catalog columns, metadata, or model parameters. No `catalog_meta_names` list is added for them. Nothing else in this plan needs that hook: the six elements are ordinary columns, and `mass`, `dist`, and the black-hole offsets already use the fixed-parameter lookup in section 1.3.

Do not reuse `t0`, `x0`, `y0`, `vx`, or `vy` for elements. `t0` is the epoch of the linear model and is replaced by the weighted mean detection time when a star is actually fit (`startables.py:1672`). `x0` and `vx` are the linear model's position and proper motion. An instantaneous orbital velocity is a different number; writing it into `vx` would make `x0 + vx*(t - t0)` wrong for any star later demoted to `Linear`. `Orbit` is not demoted, but the columns stay distinct anyway.

Non-orbit stars have NaN in the `orb_*` columns. Their `motion_model_input` stays whatever the catalog already said.

### 2.3 Errors

A fixed star has no solved elements. When `model` is asked for errors it returns `xe = ye = 0`. When it is not asked, it returns only `x, y`, and `infer_positions` fills `inf` if the table has no error columns (`startables.py:1754-1763`). There is no `pos_err` argument.

A fit star with a finite `orb_cov` propagates that matrix. Section 2.7 has the formula. A fit star whose covariance is missing, singular, or left non-finite because the solver did not converge returns `xe = ye = inf`. The elements themselves stay at the seed. That `inf` is how a failed fit stays distinct from the fixed star's zero.

Matching ignores errors, so a zero reference error does not change who matches. Transform weights do not ignore them. `get_weights_for_lists` turns a non-finite weight into 0, which drops that star from the transform.

- `trans_weights is None`: the zero error is unused.
- `'both,var'`, `'both,std'`, `'list,var'`, `'list,std'`: the science-list variance is what the weight uses. A zero reference error leaves that variance alone, so the star stays in the transform when the science list has `xe` and `ye`.
- `'ref,var'` and `'ref,std'`: the reference variance is 0, the weight is non-finite, and the star is dropped. The same drop happens when a failed fit reports `xe = ye = inf`.

That is the same outcome as a `Fixed` or `Linear` reference star whose `x0_err` and `y0_err` are 0. This version does not add a floor. A fit star with a finite covariance uses that covariance's position error in `'ref,var'` and `'ref,std'`, the same way a `Linear` star uses `x0_err`.

### 2.4 What a fit does to an orbit star

A fixed star never enters `run_fit`. Its `orb_*` columns stay, and so do `x0`, `vx`, `motion_model_used`, and `n_params`. `keep_orig` would also restore `orb_*` if a fit had touched the row, because those names are now fit parameters and `motion_model_param_names` saves them (`align.py:1857-1867`).

A fit star does enter `run_fit`. The solver is seeded from the current `orb_*` values. On success the ordinary fit-parameter write (`startables.py:1667-1669`) stores the solution back into those same columns, and stores `orb_*_err`. On failure it stores the seed again, so the elements are unchanged, and it stores non-finite errors. Phase B also writes `orb_fit_converged`, `orb_fit_n_iter`, and `orb_cov`.

Writing the fill value on failure would erase the only elements that can still place the star, and the next fit would have nothing to start from. A `Linear` star can survive a fill in `x0` because the next detection refits a position from the data. An orbit with a NaN period cannot be integrated. The reset does not clear `orb_*` on an `Orbit` star (`startables.py:1456-1458`), so the pre-fit numbers are still in the columns. Returning that seed makes the generic write a no-op on the values. No special case is added to the write loop. `orb_fit_converged = False` is what distinguishes a kept catalog orbit from a solution.

`Orbit` does not own `x0` or `vx`. The same reset still clears every fit-parameter column the used model does not own, for every star that was not frozen. For a fit `Orbit` star that is `x0`, `y0`, `vx`, `vy`, `vx0`, `ax`, `vy0`, `ay`, `pi`, and their `_err` columns, whenever those columns are already on the table. This still happens. It does not matter for prediction: `infer_positions` reads `orb_*` for an `Orbit` star, not `x0 + vx*(t - t0)`. It matters for a catalog that wanted the old linear solution to remain on the same row as the fitted orbit. Those callers freeze the star, or hold it with `update_ref_orig`, which restores `x0` and `vx` along with `orb_*`. There is no extra restore for a star that was actually fit. The existing `t0` write at `startables.py:1672` also runs, because the star was fit. `orb_t0` is a different column.

### 2.5 Epochs and demotion

`n_params` on the class is `int((n_fit_params + 1) / 2) = int((6 + 1) / 2) = 3`. That is the number the uniqueness assert and the complexity sort already understand. Six elements and two sky coordinates per epoch give the same floor: the fit is determined when `2 * n_fit >= 6`, so `n_fit >= 3`.

`n_fit = 3` has no residual degree of freedom. `inv(Jᵀ W J)` is singular, and a reduced-chi-squared scale is undefined. The elements may still be solvable. The covariance, the `_err` columns, and `xe` / `ye` are non-finite. `n_fit >= 4` leaves `2 * n_fit - 6 >= 2` residual measurements, which is where a converged fit reports a finite covariance. Four is an error-reporting threshold. It is not a second `n_params`. Raising the class attribute to 4 would break the formula every other model uses, and it would hide a solvable three-epoch case inside demotion.

`n_fit` counts distinct valid epochs, the same count the other models use (`startables.py:1215-1236`).

What each star does:

- Fixed, by the list or by `fit_motion='fixed'`. Removed before demotion. Effective requirement in that test: 0 epochs. Never demoted, including with no detections. Predicts from `orb_*`. `xe = ye = 0`. The solver is not called.
- Fit, and `n_fit < 3`. Stays `Orbit` because `demote = False`. The solver is not called. `run_fit` returns the current `orb_*` values, so those columns are bitwise unchanged. `orb_fit_converged` is false, `orb_fit_n_iter` is 0, and `orb_*_err` is non-finite. Prediction uses the kept elements. Errors on the sky are `inf`, because a fit was requested and no covariance exists.
- Fit, and `n_fit = 3`. The solver runs. A converged solution updates `orb_*`. Covariance and `orb_*_err` stay non-finite.
- Fit, and `n_fit >= 4`. The solver runs. A converged solution updates `orb_*` and stores a finite covariance and finite positive `orb_*_err`.

The uniqueness assert is untouched. Catalogs that place `Orbit` in a list with `Acceleration` or `Parallax` need `motion_model_input`. The reader writes it. A frozen star does not get a private `n_params` of 0 stored on the class or, by this plan, written into the column. Skipping the row is the whole framework change for the fixed case.

### 2.6 Solver (phase B)

`run_fit` keeps the batch signature and loops over stars inside it. `docs/motion_models.rst` already describes that pattern for a nonlinear model: the closed-form models stay vectorized, and only the stars assigned to this model pay for a per-star optimizer. A Galactic Center list is tens of orbits, not the whole mosaic. One shared `least_squares` across stars is a poor fit here. Each star has its own elements, bounds, and angle branch. `Linear`, `Fixed`, `Acceleration`, and `Parallax` stay on their closed-form vectorized fits.

Use `scipy.optimize.least_squares` with `method='trf'`. The residual is the weighted sky offset, `weighting='var'` or `'std'` the same way as `Linear.run_fit`. Initialize from the current `orb_*` elements, copied before the solve. If any of those six is non-finite, do not search. Set `orb_fit_converged` false, leave the elements at that copy, and set the errors non-finite. There is no grid search in this phase.

The internal vector has six unconstrained numbers. The catalog columns stay in the usual units (years, degrees, dimensionless eccentricity).

| Internal parameter | Maps to | Why |
|---|---|---|
| `ln(P)` | `orb_P = exp(ln P)` | Period stays positive. |
| `Δt0` | `orb_t0`, then shifted by an integer number of periods so it lies within half a period of the seed | Periapse time is periodic. The reported value stays near the guess. |
| `h = sqrt(e) cos ω`, `k = sqrt(e) sin ω` | `e = h² + k²`, `ω = atan2(k, h)` in degrees | Puts the eccentricity and the argument of periapse in a form without a hard angle cut. If `h² + k² >= 1`, the residual returns a large penalty so `e` stays below 1. |
| `i` with bounds `(0, 180)` degrees | `orb_i` | Inclination stays in the usual range. |
| `Ω` unbounded, in degrees | `orb_Omega`, wrapped to the turn nearest the seed | Node angle has no preferred cut. |

After a successful solve, also evaluate the twin `(Ω + 180°, ω + 180°)`. Sky positions are unchanged under that pair, and the line-of-sight velocity flips. This fit has no radial velocities, so the twin is a real degeneracy, not a bug. Keep the branch whose angles are closer to the input `orb_Omega` and `orb_omega`. Do not flip `i`. Radial velocities that would break the degeneracy are future work.

`absolute_sigma=True` uses the covariance `inv(Jᵀ W J)` in the internal parameters, then the analytic Jacobian of the map above to convert that matrix into `orb_cov` in the reported elements. `absolute_sigma=False` multiplies by the reduced chi-squared when the degree of freedom is positive. The diagonal of `orb_cov` supplies `orb_*_err`.

If `least_squares` does not converge, or the star has fewer than three epochs, return the seed copied before the solve, not a partial step and not `fill_value`. `orb_*_err` is non-finite, `orb_fit_converged` is false, `orb_cov` is non-finite, and `orb_fit_n_iter` records how far the solver got (zero if it was not called). The generic column write then stores the seed, so the elements stay byte for byte. A partial solution would be a set of elements that was never accepted, and a fill value would leave the next epoch with no orbit to propagate.

### 2.7 Prediction and position errors

`model` always reads the current `orb_*` columns. A fixed star, a skipped fit, and a failed fit all still have the seed there. A successful fit has the updated elements there. The sky formula is the one in section 2.1, including `x = -east` and `y = +north`.

When errors are requested and `orb_cov` is finite:

```text
xe**2 = J_x  orb_cov  J_x.T
ye**2 = J_y  orb_cov  J_y.T
```

The Jacobian is a central difference of the same `kep2xyz` path the prediction uses. Diagonal `orb_*_err` alone is the wrong input. Period, periapse time, eccentricity, and the two angles are correlated, and `Ω` with `ω` is exactly degenerate on the sky. Propagating only the diagonal would invent a position error. If a fit was requested and the covariance is missing or singular, return `xe = ye = inf`. If the solver was not called because the star is fixed, return `xe = ye = 0`. There is no `pos_err` argument.

## 3. Kepler solver

New module `flystar/orbits.py`. Port the Newtonian solver from the gcwork `Orbit.kep2xyz` / `eccen_anomaly` used by `orbits_jlu_python_gcwork_2024-10-03`. Do not import `gcwork`, `pylab`, or `numpy.core.umath_tests`. No `print` of `sys.path`.

```text
a_AU = (P_yr**2 * M_Msun) ** (1/3)      # Gaussian: years, solar masses, AU
a_arcsec = a_AU / dist_pc
a_mas = a_arcsec * mas_per_arcsec       # mas_per_arcsec from astropy, stored as a float
```

`G`, `Msun`, the AU, the Julian year, and mas-per-arcsec are read from astropy once at import and stored as plain floats. The Gaussian axis above is not replaced by `(G M P**2 / 4 pi**2)`. That physical axis is about 1.3e-5 smaller and misses both the 0.02 mas `a`-column tolerance and the gcwork position tolerance. Positions do not use `G` or `Msun`. Those two constants convert only the acceleration.

`kep2xyz` takes arrays of epochs and scalar elements. It returns `r_arcsec`, `v_mas_yr`, and `acc_mas_yr2`, each shape `(N, 3)`, with index 0 east, 1 north, 2 line of sight. The acceleration array is `acc`, not `a`, so it does not collide with the semi-major axis. The anomaly is solved with Newton-Raphson. Guard the `sqrt(1 - e**2)` division; the largest eccentricity in this file is about 0.98.

The port is Newtonian only. It has no GR periapse-advance term and no relativistic redshift term, and no flags that would turn them on. Those paths are future work (section 9).

With `M = 4.0e6` and `R0 = 8000`, `a_mas` from the printed `P` matches column `a` for all 32 stars to well under 0.02 mas. The largest residual is S0-105, about 0.013 mas, because `P` is printed as `152.76` (0.01 yr). That is the rounding of `P`, not a different mass. The test asserts `abs(a_mas - a) <= 0.02` for every star. It reads `a` from the file while it parses the line. It does not write `a` onto a catalog. The gcwork cross-check does not use these defaults; it passes `mass=4.07e6` and `dist=7960.1` and compares sky positions to `kep2xyz`.

## 4. Reader

`flystar/orbits.py` function `read_orbits_dat(path) -> astropy.Table`. Each data line has nine whitespace-separated fields and no header. The parser reads all nine, so a short or long line fails. The returned table has `name`, `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, and `orb_omega` only. Names stay as written (`S0-2`, not a different spelling). `a` and `search` are not columns of that table. The file on disk still has nine numeric fields and no header.

`attach_orbits(starlist, orbits)` matches on `name`:

- Matched rows: set `motion_model_input` to `'Orbit'` and copy the six element columns. Do not set `fit_motion`. Fixed versus fit stays with the caller.
- Unmatched catalog rows: leave `motion_model_input` unchanged and set those six columns to NaN.
- Names in the orbit file that are not in the catalog: warn, do not add rows.
- Do not add `orb_a`, `orb_search`, or any metadata entry for `a` or `search`.
- Do not add `orb_*_err`, `orb_cov`, `orb_fit_converged`, or `orb_fit_n_iter`. The fitter creates the error columns. The diagnostics appear when a fit runs.

Attach is explicit. Loading a starlist does not look for `orbits.dat`.

Matching uses `MosaicToRef`'s `dr_tol` for every star, including stars with `motion_model_input='Orbit'`. The `search` field in the file is not a radius. There is no per-star search radius and no later phase that adds one.

## 5. Fixed versus fit

### 5.1 The column

Use a string column `fit_motion` with values `'fixed'` and `'fit'`. Do not keep a boolean `fix_motion`.

The override has to work in both directions, and it has to have a third state that means "no opinion". A boolean only has two states. A plain bool column cannot store null, so the default `False` would be read as an explicit "fit" and would silently un-freeze every star in `fixed_motion_models`. A masked bool can store null, but a missing mask is easy to confuse with `False`, and the name reads as a yes/no freeze rather than a mode. A string cell is `'fixed'`, `'fit'`, or blank. Blank, masked, and a missing column are the same third state.

Accepted values are exactly `'fixed'` and `'fit'`, lowercase. Any other non-blank value raises. Do not treat a typo as fit.

### 5.2 Precedence

For each star:

1. If the `fit_motion` column exists and that row's value is `'fixed'` or `'fit'`, the cell wins.
2. Otherwise, if `motion_model_input` is in `fixed_motion_models`, the star is fixed.
3. Otherwise the star is fit.

A missing column means every row uses step 2. A null, masked, or blank cell on one row means that row uses step 2 and the other rows still use their own cells. `fixed_motion_models=None`, and an omitted argument, means the list freezes nobody. Together with a missing column, that is today's behavior: every star is fit.

`StarTable.fit_motion_models` takes the list and honors the column. `MosaicToRef` passes the list through. `update_ref_table_aggregates` builds the same mask, because the one-epoch fast path never reaches `fit_motion_models`.

Why this and not a class attribute such as `Orbit.refit = False`:

- A class flag would freeze every star of that class, in every align. The requested test freezes one `Linear` star and refits the other `Linear` stars in the same table. The list cannot say that either, which is why the column exists. The list is the convenient form for "every `Orbit` star".
- A class default would change behavior when the caller does not pass the option. The default must stay identical to today.
- `motion_models`, `fixed_params_dict`, and `update_ref_orig` are already aligner settings. This is the same kind of setting.
- Fixed versus fit is a property of the star in this align, not of the class. Phase B fits elements only for stars the caller left unfrozen.

`MosaicSelfRef.__init__` sets `self.fixed_motion_models` to an empty collection so the inherited aggregate method is safe. Only `MosaicToRef.__init__` exposes the argument. `getattr` is a reasonable belt if a subclass forgets the attribute.

### 5.3 What "fixed" means

For a fixed star, every motion-parameter column from the input reference catalog is unchanged after every iteration and in the final table. That includes `x0`, `vx`, the `orb_*` elements, `motion_model_used`, and `n_params`. The solver does not run. The star is not demoted.

Implementation: build the frozen mask at the start of `update_ref_table_aggregates` and union it into `keep_orig` before the save. If `keep_orig` was `None`, the mask becomes the new `keep_orig`. The existing `vals_orig` save then covers the frozen rows, those rows are absent from both `simple_idxs` and `complex_idxs`, and the restore at the end writes the saved values back after `determine_motion_models`. The restore is the whole row. It is the same restore `Linear` gets.

`fit_motion_models` does the same exclusion when it is called directly. Frozen rows are taken out of `select_stars` before the demotion block and before the fit-parameter reset, and the scatter back into the parent table does not write them. Excluding them only inside `fit_motion_models` is not enough for the aligner: `combine_lists_xym` would still overwrite `x0` and `y0` for a frozen star with one detection.

Do not special-case the string `'Orbit'` inside the freeze mask. The mask is the whole policy, for every model.

### 5.4 `update_ref_orig`

The frozen mask is applied on top of the table in section 1.6. It does not change what `update_ref_orig` does to a star that is not frozen. `fit_motion='fit'` means "not frozen". It does not mean "refit this original row even when `update_ref_orig` is holding original rows".

| Setting | Stars that are not frozen | Frozen stars |
|---|---|---|
| `False` | Original rows stay as they are, including an `Orbit` row whose `fit_motion` is `'fit'`. Same rule as `Linear`. | Also preserved. The union matters for a frozen star that is not an original row. |
| `True` | Refit on the last list of each iteration, and again in the final pass. An `Orbit` row marked `'fit'` is solved on those passes. | Saved and restored on those passes too. Input values survive. |
| `'periter'` | Refit on the last list of each iteration. | Held on that last list, not only on the intermediate lists. |
| `'atend'` | Held during the iteration, refit in the final pass. | Held during the iteration and still held in the final pass. |

The saved values are the values at the start of that aggregate call. For a frozen star that was also frozen on the previous call, those are still the input catalog values.

To solve orbital elements inside `MosaicToRef`, the orbit row must be not frozen, and `update_ref_orig` must be `True`, `'periter'`, or `'atend'` if that row is an original reference star. `False` holds it, the same way it holds an original `Linear` star.

### 5.5 Demotion and new stars

`n_params` is the minimum number of distinct epochs a model needs. `n_fit` is the per-star count of distinct valid epochs. Demotion is the reassignment in `fit_motion_models` when `n_fit < n_params` (`startables.py:1244-1255`). It only sees stars that were not frozen. A `Linear` star needs two epochs, so one detection becomes `Fixed` when the star is not frozen. A frozen `Linear` star with one detection stays `Linear` and keeps its input `x0` and `vx`.

`motion_models` is still the list demotion chooses from. Freezing `Orbit` does not add `Orbit` to that list and does not change `motion_models[-1]`, so new stars are unaffected.

A fixed `Orbit` star is never demoted, because it is not in the test. Its class `n_params` is still 3. The effective requirement is 0 only in the sense that the comparison is not applied.

A fit `Orbit` star is not demoted when `n_fit < 3`, because `demote = False`. The solver is skipped and `orb_*` stay at the seed. Section 2.5 is the full account. The reset still runs for that star. It does not clear `orb_*`, because `Orbit` owns them. It does clear `x0` and `vx` when those columns exist. Prediction does not use the cleared columns.

Without a `motion_model_input` column, selection uses `n_params` alone, and two models with the same `n_params` raise (`startables.py:1082-1086`). `Acceleration` and `Parallax` also have `n_params = 3`, so a fit list that contains `Orbit` and either of them raises when the column is missing. The orbits reader always writes `motion_model_input`.

### 5.6 The same switch on the other models

| Model | Fixed | Fit |
|---|---|---|
| `Fixed` | Keep `x0`, `y0`, and their errors. Do not demote. | Weighted mean, as today. |
| `Linear` | Keep `x0`, `vx`, `y0`, `vy`. One detection stays `Linear`. | Closed-form refit, as today. Fewer than two epochs demotes. |
| `Acceleration` | Keep the quadratic coefficients and `t0`. | Closed-form refit, as today. |
| `Parallax` | Keep `x0`, `vx`, `y0`, `vy`, `pi`. | Closed-form refit, as today. |
| `Orbit` | Predict from `orb_*`. `xe = ye = 0`. No solver. Input `orb_*`, `x0`, and `vx` stay. | `least_squares` updates `orb_*` in place. A failed fit keeps the seed and sets `orb_fit_converged` false. `x0` and `vx` are cleared by the reset. |

### 5.7 Recommended Galactic Center call

```python
MosaicToRef(
    ref_list,
    starlists,
    dr_tol=...,
    motion_models=['Empty', 'Fixed', 'Linear', 'Acceleration', 'Parallax',
                   'Orbit'],
    fixed_motion_models=['Orbit'],
    fixed_params_dict={'mass': 4.0e6, 'dist': 8.0e3},  # optional
    update_ref_orig=False,
)
```

`mass` and `dist` can be omitted. Passing them makes the choice visible at the call. `fixed_motion_models=['Orbit']` freezes every orbit whose `fit_motion` is missing or blank. Set `fit_motion='fit'` on the rows to solve, and set `update_ref_orig` to `True`, `'periter'`, or `'atend'` if those rows are original reference stars. The cell overrides the list. It does not override `update_ref_orig`. `dr_tol` is the match radius for every star.

## 6. Tests

New files: `flystar/tests/test_orbits.py` and additions to `flystar/tests/test_motion_model.py` and `flystar/tests/test_align.py`. Use the existing pytest style. No live download of gcwork.

Solver and model, phase A:

- Circular and eccentric analytic positions match Newtonian `kep2xyz` at a grid of epochs. The port has no GR or redshift switch to test.
- With the black hole at the origin, `x` equals minus the east offset and `y` equals the north offset. The class has no `x_sign` or `y_sign` to override.
- A fixed star, asked for errors, returns `xe = ye = 0`. `optional_fixed_params` has no `pos_err`. Prediction reads `orb_*`. It does not need `orb_*_err`.
- `Orbit()` has `optional_fixed_params['mass'] == 4.0e6` and `optional_fixed_params['dist'] == 8.0e3`. `model` with no `mass` or `dist` uses those. A `fixed_params_dict` override, a column, and a `meta` entry each win in that order.
- For every star in `orbits.dat` v2.0.2, `a_mas` from the printed `P` with those defaults is within 0.02 mas of the file's `a` field. The test reads `a` during the parse. After `read_orbits_dat` and after `attach_orbits`, the table has no `a`, `orb_a`, `search`, or `orb_search` column and no metadata entry for either field.
- Against gcwork `kep2xyz`, pass `mass=4.07e6` and `dist=7960.1` explicitly. Compare east and north at several epochs, including periapse and a time far from it. Do not use the FlyStar defaults for this comparison.
- `read_orbits_dat` returns 32 rows and the S0-2 elements. `attach_orbits` sets `motion_model_input` only on name matches, and does not set `fit_motion`.
- An orbit star and a linear star in one `MosaicToRef` are matched with the same `dr_tol`. No column on the orbit star changes that radius.
- `determine_motion_models` keeps `motion_model_input='Orbit'` when `mass` and `dist` are absent, when they exist only in `meta`, and when `orb_*_err` is absent. A non-finite `orb_e` still rejects the request. The same meta path keeps an explicit `Parallax` request whose `pa` is only in `meta`. A non-finite `vx` still rejects `Linear`. `infer_positions` on a fixed orbit star with `x0_err` present and `orb_P_err` absent returns `xe = ye = 0` and does not raise.

Freeze list and column. These apply to phase A and stay green in phase B:

- One `Orbit` star with `fixed_motion_models=['Orbit']` and no `fit_motion` column. After `fit_motion_models`, and after `MosaicToRef.update_ref_table_aggregates`, every `orb_*` value and the input `x0` and `vx` are unchanged, including with one detection and with zero detections. `motion_model_used` stays `Orbit`. The solver is not called. `xe = ye = 0`. `MosaicToRef.fit` still drops a star with zero detections in its existing junk-source cleanup, so that case is not a surviving row after `fit`.
- The same star with the list unset and `fit_motion='fixed'`. Same outcome.
- The column overrides the list in both directions. `fixed_motion_models=['Orbit']` and one row `fit_motion='fit'`: that row is not frozen, and the other `Orbit` rows are. `fixed_motion_models=['Linear']` and one row `fit_motion='fit'`: that `Linear` row is refit and the other `Linear` rows keep `x0` and `vx`. A blank cell next to a populated cell follows the list.
- One table, two `Linear` stars with the same epochs and different motions. `fit_motion` is `'fixed'` for the first only. After `fit_motion_models`, the first star's `x0` and `vx` are exactly the input values. The second star's `vx` has changed.
- The same table with `fixed_motion_models=['Linear']` and no column leaves both stars' `x0` and `vx` unchanged.
- With the list unset and no `fit_motion` column, both `Linear` stars are refit. This is the regression check that the default did not change.
- A frozen `Linear` star with one detection stays `motion_model_used='Linear'`. An unfrozen `Linear` star with one detection is demoted to `Fixed`. A frozen `Fixed`, `Acceleration`, or `Parallax` star keeps its input fit parameters while an unfrozen neighbor of the same class is refit.
- A value other than `'fixed'`, `'fit'`, or blank raises.
- Repeat the frozen-orbit table for `update_ref_orig` in `False`, `True`, `'periter'`, and `'atend'`. Frozen values after the final table match the input. Stars that are not frozen change when the setting says they should (`True`, `'periter'`, `'atend'`) and do not change when it is `False`. An original row with `fit_motion='fit'` is held when the setting is `False`, the same way an original `Linear` row is held.

Align integration, phase A:

- A tiny mosaic: one orbit star frozen by the list, plus one `Linear` star, three epochs. At each epoch the reference position of the orbit star matches `model()` on `orb_*`, and the linear star matches `x0 + vx*(t - t0)`.
- `update_ref_orig=False` and no freeze, on an original `Orbit` row: the row is held by `keep_orig`, so `orb_*`, `x0`, and `vx` stay. This is the `update_ref_orig` rule, not a property of the orbit.

Fitting, phase B. The phase A tests above still pass.

- Inject an S0-2-like orbit (`e` about 0.9, `P` about 16 yr) at 15 or more epochs spanning at least one period, with the star not frozen. Add 0.5 mas Gaussian noise. The fit updates `orb_*` in place. Recover `|ΔP|/P < 0.02`, `|Δe| < 0.02`, `|Δt0| < 0.1` yr, and angle errors under 5 degrees after folding `(Ω, ω)` onto the branch nearer the seed.
- Inject an S0-16-like orbit (`e` about 0.97, `P` about 55 yr) with epochs that include periapse and 1 mas noise. Recover `|Δe| < 0.03` and `|ΔP|/P < 0.05`, with the same angle rule.
- `orb_*` is bitwise unchanged for a frozen star, and for an original row when `update_ref_orig=False`, including a row whose `fit_motion` is `'fit'`. It is not bitwise unchanged for a star that was actually fit.
- With four or more epochs and a converged fit, `orb_*` has moved toward the injected elements, each `orb_*_err` is finite and positive, `orb_cov` has shape `(6, 6)` and is symmetric, `orb_fit_converged` is true, and `orb_fit_n_iter` is positive. `xe` and `ye` from `model` are finite and come from that covariance.
- With `n_fit = 2`, the star stays `Orbit`, `orb_*` is bitwise unchanged, `orb_fit_converged` is false, and the predicted position matches the seed. `xe` and `ye` are non-finite.
- With `n_fit = 3`, a converged fit may update `orb_*`, and the errors, the covariance used for `xe` / `ye`, and `xe` / `ye` themselves are non-finite.
- A failed solve keeps the seed in `orb_*`, sets `orb_fit_converged` false, and does not leave a fill value in the element columns.
- One `MosaicToRef` with original rows, `fixed_motion_models=['Orbit']`, and `update_ref_orig=True`: an `Orbit` row with blank `fit_motion` keeps its input elements and its input `x0` and `vx`; an `Orbit` row with `fit_motion='fit'` has `orb_*` updated toward the data, finite `orb_*_err` when it has at least four epochs, and does not keep the input `x0` and `vx`; a `Linear` row is refit. All three match with the same `dr_tol`.
- The same three rows with `update_ref_orig=False`: every original row is held, including the one marked `'fit'`, and `orb_*` is bitwise unchanged. The `Linear` row is held too.

## 7. Decisions

1. **Coordinate frame.** Resolved. Hardcoded inside `Orbit.model`: `x = -east`, `y = +north`. There is no `x_sign` or `y_sign` parameter.
2. **Black-hole mass and distance.** Resolved. Optional fixed parameters, defaults `mass=4.0e6` Msun and `dist=8.0e3` pc, same lookup as `Parallax`'s `pa` and `obsLocation`. These defaults reproduce the `a` field of `orbits.dat`. The gcwork pair is an explicit override, not the default. They are not fit. A joint black-hole fit across stars is future work.
3. **Fixed versus fitted elements.** Resolved. One class. `fit_param_names` are `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, and `orb_omega`. A fit updates those columns in place. They are not also required fixed parameters. A fixed star is predicted from the same columns because it is not refit. Original values survive through `update_ref_orig` / `keep_orig`, `fixed_motion_models`, and `fit_motion='fixed'`, the same mechanisms that preserve `x0` and `vx`. A failed fit returns the seed and sets `orb_fit_converged` false. Class `n_params` stays 3. A fit star below that is not demoted. Finite `orb_*_err` starts at four epochs. A fit `Orbit` star still loses pre-existing `x0` and `vx` to the reset. That does not affect the orbit prediction. Callers who need those linear columns freeze the star or hold the row.
4. **Which stars are frozen.** Resolved. `fixed_motion_models` names whole classes. The string column `fit_motion` is `'fixed'` or `'fit'`. A non-blank cell overrides the list. A missing column, or a null or blank cell, follows the list. Both unset means fit, which is today's behavior. The recommended align passes `fixed_motion_models=['Orbit']`. The same column and list freeze or refit `Linear` and the other models. `fit_motion` does not override `update_ref_orig`.
5. **Match radius.** Resolved. Every star, fixed orbit or fitted orbit, uses the same `MosaicToRef` `dr_tol`. The file's `search` column is parsed and discarded. There is no per-star search radius.
6. **Position errors.** Resolved. A fixed star returns `xe = ye = 0` when errors are requested. A fit with a finite `orb_cov` returns the Jacobian times that covariance. A fit with no finite covariance returns `inf`, including a failed fit whose `orb_*` values were kept. There is no `pos_err` parameter. For a fixed star, `'ref,var'` and `'ref,std'` still drop the star; schemes that use the science-list errors do not.

Still open, and not blocking the plan:

- Whether to ship `orbits.dat` inside the package or only accept a path. Tests can use a checked-in copy of the 32-line file either way.

## 8. Implementation order

Phase A is the `Orbit` class with fixed prediction, plus the freeze list and the `fit_motion` column. Do not start phase B until phase A behaves as this plan says.

Phase A:

1. `flystar/orbits.py`: Newtonian solver only, `kep2xyz`, `read_orbits_dat`, `attach_orbits`. Parse `a` and `search`, do not store them. Tests against the analytic orbit, the `a_mas` check against file `a`, and gcwork with explicit mass and distance.
2. `Orbit` in `motion_model.py`. `fit_param_names` are the six `orb_*` names, `required_fixed_param_names` is empty, `n_params` is 3, and `demote = False`. `model` predicts from `orb_*` and returns `xe = ye = 0`. `run_fit` returns the current elements and does not call `least_squares`. Tests for defaults, overrides, and `infer_positions` when `orb_*_err` is absent.
3. `determine_motion_models`: optional defaults count as available, and the explicit-request loop reads `meta`. `infer_positions` treats a missing `orb_*_err` column as `inf` and passes `orb_cov` through when the column exists.
4. `fixed_motion_models` and `fit_motion` on `MosaicToRef` and `fit_motion_models`. Honor precedence, the one-epoch path, demotion (frozen rows out; `demote = False` for `Orbit`), and all four `update_ref_orig` settings. The same mask freezes `Linear`.
5. Tests: fixed via the list, fixed via the column, the column overriding the list in both directions, one `Linear` star frozen by the column while another is refit, and the default unchanged when neither is set.

Phase B, the solver inside the same class, after phase A:

1. The optional fifth return from `run_fit` for diagnostics. Existing models keep their four-value return.
2. `Orbit.run_fit`: per-star `least_squares`, the internal parameterization, the 180 degree branch choice, and `orb_cov`. A failed fit and a star with fewer than three epochs return the seed and set `orb_fit_converged` false. Tests that move `orb_*` toward the injected S0-2-like and S0-16-like orbits, leave `orb_*` unchanged when the star is frozen or when `update_ref_orig=False`, require finite `orb_*_err` at four or more epochs, and cover `n_fit` of 2 and of 3.
3. One `MosaicToRef` test with a fixed `Orbit` star, a fit `Orbit` star (`fit_motion='fit'`), and a `Linear` star. `update_ref_orig=True` so the fit runs. The same rows with `update_ref_orig=False` stay held.

## 9. Out of scope

A second orbit class. A boolean `fix_motion` column. Any exception that pulls one model out of `keep_orig` or restores only some of a row's columns. Posterior samples. Light-time delay. Radial velocities, including using them to break the `(Ω + 180°, ω + 180°)` degeneracy. A joint fit of black-hole mass, distance, or offset across stars. GR periapse advance and relativistic redshift: not parameters, not flags, and not branches in `flystar/orbits.py`. A per-star match radius, including any use of the file's `search` column. Editing `MosaicSelfRef` beyond storing an empty freeze list so the shared aggregate method can read it, and keeping `fit_motion` one-dimensional in `setup_ref_table_from_starlist` (section 10). Changing name lookup. Any change to stars that the list and the column do not mark, beyond the missing `orb_*_err` lookup and `demote = False` described above.

## 10. Deviations found while implementing

1. `get_star_positions_at_time` now calls `infer_positions`. `motion_model_dict` and `allow_alt_models` are accepted and unused. The old body raised `AttributeError`.
2. A seed with `P <= 0` or `e` outside `[0, 1)` is treated like a non-finite seed: the solver is not called, the seed is kept, and `orb_fit_converged` is false. `kep2xyz` cannot integrate those elements, and returning a fill value would erase the catalog.
3. `orb_fit_n_iter` stores `least_squares` `nfev` (residual evaluations). `OptimizeResult` has no `nit`.
4. An all-NaN `orb_cov` means the fitter never wrote that star, so `xe = ye = 0`. A failed fit stores inf, and `xe = ye = inf`. One column has to carry both, because a mixed table creates `orb_cov` for every row.
5. `fit_motion` is kept one-dimensional in `MosaicSelfRef.setup_ref_table_from_starlist`, next to `motion_model_input`. A two-dimensional column is cleared when the per-list values are reset, which erased the mode before the freeze mask could read it.
6. `MosaicToRef.fit` still removes stars with `n_detect == 0`. A frozen orbit with no detections is unchanged inside `fit_motion_models` and `update_ref_table_aggregates`, and is then dropped by that existing cleanup. The zero-detection check does not expect the star to survive `fit`.
7. The acceleration vector uses `_G_CGS`, `_MSUN_G`, `_CM_IN_AU`, and `_SEC_IN_YR`, each taken from astropy at import and stored as a float. The uploaded gcwork `Constants` class was not in the dependency file. Positions do not use `G` or `Msun`. The gcwork cross-check compares east and north only. `semimajor_axis_mas` and the AU axis inside `kep2xyz` keep the Gaussian `(P**2 * M)**(1/3)` relation. The physical `G M` form misses the 0.02 mas `a`-column tolerance and the 1e-12 arcsec gcwork tolerance. The semi-major-axis symbol in code, comments, and this plan is `a`. The orbits.dat file layout is unchanged. The acceleration array is `acc`, so it does not share that name.
