# Plan: Keplerian `Orbit` motion model

Plan only. No source changes in this branch. The base is `mm_rework_lingfeng`.

This adds a prediction-only Keplerian model for Galactic Center stars orbiting Sgr A*, read from `orbits.dat`, and wires it into `MosaicToRef` so those stars are propagated to each epoch while every other star keeps using `Empty`, `Fixed`, `Linear`, `Acceleration`, or `Parallax`.

Orbital elements are not free parameters in this version. A separate, generic align option says which motion models are not refit. `Orbit` is the reason for that option. It is not hard-coded.

## Comparison with the existing framework

`Orbit` is another direct subclass of `MotionModel`. It does not change the base class. The differences are which attributes it fills in, and four edits to the shared selection and refit path, listed at the end of this section.

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

`n_params` is `int((n_fit_params + 1) / 2)` for every row, including the proposed `Orbit`.

| | Empty | Fixed | Linear | Acceleration | Parallax | Orbit (proposed) |
|---|---|---|---|---|---|---|
| Fit parameters | none | `x0`, `y0` | `x0`, `vx`, `y0`, `vy` | `x0`, `vx0`, `ax`, `y0`, `vy0`, `ay` | `x0`, `vx`, `y0`, `vy`, `pi` | none |
| Required fixed | none | none | `t0` | `t0` | `t0`, `ra`, `dec` | `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, `orb_omega` |
| Optional fixed | none | none | none | none | `pa=0`, `obsLocation='earth'` | `mass=4.0e6` Msun, `dist=8.0e3` pc, `x_bh=0`, `y_bh=0`, `vx_bh=0`, `vy_bh=0`, `t_bh=2000` |
| Catalog columns | none | `x0`, `y0` and `_err` | those plus `vx`, `vy` and `_err`, and `t0` | those plus `vx0`, `ax`, `vy0`, `ay` and `_err`, and `t0` | Linear's columns plus `pi`, `pi_err`, `ra`, `dec`, `pa`, `obsLocation` | the six `orb_*` elements. `mass`, `dist`, and the black-hole offsets land in `meta` when uniform, otherwise in columns. File fields `A` and `search` are parsed to check the line and are not written to the catalog |
| Meaning of `t0` | none | none | Epoch of `x0`, `y0`, `vx`, `vy`. `fit` fills the weighted-mean epoch when `t0` is missing (`motion_model.py:364-365`) | Epoch of `x0`, `y0`, `vx0`, `vy0`, `ax`, `ay`. Same default fill | Same epoch as `Linear`, plus the epoch subtracted before the parallax vector | `orb_t0` is periapse time. It is a different column from stellar `t0`. `t_bh` is the epoch of the black-hole offset |
| `n_params` and demotion | `0`. Already the floor | `1`. One epoch keeps it; zero epochs become `Empty` | `2`. Fewer than two distinct epochs demotes to `Fixed` or `Empty` (`startables.py:1245-1255`) | `3`. Shares that number with `Parallax`, so both in one list without `motion_model_input` raises (`startables.py:1082-1086`) | `3`. Same collision with `Acceleration` | `0`, same as `Empty`. Both in one list without `motion_model_input` raises. With the column, `n_fit < 0` never happens, so an unfrozen `Orbit` star is not demoted for lack of epochs |
| Fittable | `run_fit` returns the fill value. Nothing is solved | Closed-form weighted mean | Closed-form 2x2 normal equations | Closed-form quadratic | Closed-form joint 5-parameter fit. `pi` is shared by x and y | `run_fit` returns arrays with shape `(n_stars, 0)`, like `Empty`. Elements are not solved |
| Prediction | NaN at every time (`motion_model.py:517-518`) | `x0`, `y0`, constant in time (`motion_model.py:602`) | `x0 + vx*(t - t0)` (`motion_model.py:791`, `854-855`) | `x0 + vx0*dt + 0.5*ax*dt**2` (`motion_model.py:1073`) | `x0 + vx*dt + pi*pvec_x`, and the same in y (`motion_model.py:1408-1409`) | Newtonian `kep2xyz` east/north, then `x = x_bh + vx_bh*(t - t_bh) - r_east` and `y = y_bh + vy_bh*(t - t_bh) + r_north`. The signs are hardcoded |
| Error propagation | `xe = ye = inf` when errors are requested | `x0_err`, `y0_err` broadcast across time (`motion_model.py:659-660`) | `hypot(x0_err, vx_err*dt)` (`motion_model.py:867-868`) | `sqrt(x0_err**2 + (vx0_err*dt)**2 + (0.5*ax_err*dt**2)**2)` (`motion_model.py:1132-1133`) | That linear sum plus `(pi_err * pvec)**2` (`motion_model.py:1510-1511`) | `xe = ye = 0` when errors are requested. No fit parameters and no element uncertainties, so there is nothing to propagate and no `pos_err` knob |
| `fixed_motion_models` / `fix_motion` | Does not exist yet. After the change, a listed model or a `True` `fix_motion` row keeps its input columns and is not demoted. Unlisted stars are unchanged | same rule | same rule. This is how one `Linear` star stays frozen while another is refit | same rule | same rule | same rule. The recommended align passes `fixed_motion_models=['Orbit']`. An unfrozen `Orbit` star keeps its elements (they are fixed parameters) and loses `x0` and `vx` to the fit-parameter reset (`startables.py:1450-1458`) |

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

### Changes to shared machinery

`MotionModel` itself is not edited. `model`, `fit`, `run_fit`, and `calc_chi2` keep the signatures above. `Orbit` is picked up by `motion_model_map` because it is a direct subclass. The longest existing name is `Acceleration` (12 characters), and `Orbit` is shorter, so `_MOTION_MODEL_NAME_WIDTH` in `startables.py:17-18` does not change.

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

   Existing models. With no frozen rows, `keep_orig` is the `update_ref_orig` mask from section 1.6 and the function behaves as it does now. Frozen rows of any model, including `Linear` and `Parallax`, are saved and restored and are absent from both `simple_idxs` and `complex_idxs`, so they are not demoted.

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

5. **New columns and metadata keys.** The reader writes `orb_P`, `orb_t0`, `orb_e`, `orb_i`, `orb_Omega`, and `orb_omega`. It does not write `A` or `search`. No `catalog_meta_names` hook is added. That hook is not needed: `A` is checked inside the reader test and then dropped, `search` is not a match radius, and uniform values such as `mass` and `dist` already go to `table.meta` through `fit_motion_models` (`startables.py:1422-1429`). The reset loop builds its column set from `fit_param_names` of every subclass (`startables.py:1450-1452`). `Orbit.fit_param_names` is empty, so that set does not grow and no existing column starts being cleared because `Orbit` exists.

   Existing models. Their columns and their `meta` keys are untouched. The one new effect is on a star whose `motion_model_used` becomes `Orbit` and that is not frozen: `x0`, `vx`, and the other fit-parameter columns it does not own are cleared by the reset that already does this for any model. Freezing the star skips that reset.

6. **`n_params` uniqueness.** The assert at `startables.py:1082-1086` is unchanged. `Orbit.n_params` is 0, like `Empty`. A `motion_models` list that contains both, and a table with no `motion_model_input` column, raises. Lists that do not include `Orbit` are unaffected. Galactic Center catalogs set the column, which is the supported way to mix them.

## 1. Architecture on `mm_rework_lingfeng`

The motion-model machinery on this branch is not the `mm_rework` API. Implementation follows the names below.

### 1.1 Models are discovered, not registered

`flystar/motion_model.py` defines `MotionModel` and the direct subclasses `Empty`, `Fixed`, `Linear`, `Acceleration`, and `Parallax`. `motion_model_map()` builds `{class __name__: class}` from `MotionModel.__subclasses__()`. That map is not recursive, so `Orbit` must subclass `MotionModel` itself. The class name and the `name` attribute are both `"Orbit"`. `organize_motion_models` matches names with `str.capitalize()`, so the public string is the one capitalised word `Orbit`.

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
- `run_fit(t, x, y, xe, ye, valid, fixed_params_dict=None, ...)` returns `(params, param_errs, chi2x, chi2y)` for the whole batch.

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

Prediction only. `fit_param_names = []`, so `n_fit_params = 0` and `n_params = 0`. `run_fit` returns empty parameter arrays, the same shape contract as `Empty`, and does not touch the elements.

`n_params = 0` collides with `Empty`. `fit_motion_models` raises if both are candidates and the table has no `motion_model_input` column. Galactic Center catalogs set that column, which is the supported way to mix `Orbit` with `Empty`. Do not invent a fake `n_params` to dodge the check.

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

`orbits.dat` has no uncertainties, and `Orbit` has no fit parameters, so there is nothing to propagate. When `model` is asked for errors it returns `xe = ye = 0`. When it is not asked, it returns only `x, y`, and `infer_positions` fills `inf` if the table has no error columns (`startables.py:1754-1763`). There is no `pos_err` argument.

Matching ignores errors, so a zero reference error does not change who matches. Transform weights do not ignore them. `get_weights_for_lists` turns a non-finite weight into 0, which drops that star from the transform.

- `trans_weights is None`: the zero error is unused.
- `'both,var'`, `'both,std'`, `'list,var'`, `'list,std'`: the science-list variance is what the weight uses. A zero reference error leaves that variance alone, so the star stays in the transform when the science list has `xe` and `ye`.
- `'ref,var'` and `'ref,std'`: the reference variance is 0, the weight is non-finite, and the star is dropped.

That is the same outcome as a `Fixed` or `Linear` reference star whose `x0_err` and `y0_err` are 0. This version does not add a floor. Do not invent a covariance.

### 2.4 What a fit does to an orbit star if it is not frozen

`run_fit` does not change `orb_*`. The generic reset in `fit_motion_models` still clears fit-parameter columns the used model does not own, so an unfrozen `Orbit` star loses `x0`, `vx`, and the rest of the linear block. The align option in section 5 is what keeps the input catalog row intact. Callers who want that pass `fixed_motion_models=['Orbit']`. The model class does not freeze itself.

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
- Version 1 does not fit elements. A later version might. The freeze should be the caller's choice when that happens, not a permanent property of the class.

`MosaicSelfRef.__init__` sets `self.fixed_motion_models` to an empty collection so the inherited aggregate method is safe. Only `MosaicToRef.__init__` exposes the argument. `getattr` is a reasonable belt if a subclass forgets the attribute.

### 5.2 What "not refit" means

For a frozen star, every motion-parameter column from the input reference catalog is unchanged after every iteration and in the final table. That includes `x0`, `vx`, the `orb_*` elements, `motion_model_used`, and `n_params`. Stars of other models, and unfrozen stars of the same model, are refit exactly as they are today.

Implementation: build the frozen mask at the start of `update_ref_table_aggregates` and union it into `keep_orig` before the save. If `keep_orig` was `None`, the mask becomes the new `keep_orig`. The existing `vals_orig` save then covers the frozen rows, those rows are absent from both `simple_idxs` and `complex_idxs`, and the restore at the end writes the saved values back after `determine_motion_models`.

`fit_motion_models` does the same exclusion when it is called directly. Frozen rows are taken out of `select_stars` before the demotion block and before the fit-parameter reset, and the scatter back into the parent table does not write them. Excluding them only inside `fit_motion_models` is not enough for the aligner: `combine_lists_xym` would still overwrite `x0` and `y0` for a frozen star with one detection.

Do not special-case the string `'Orbit'` inside either function. The mask is the whole policy.

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

Demotion is the `n_fit < n_params` reassignment inside `fit_motion_models`. It only sees stars that were not frozen. A `Linear` star with one detection still becomes `Fixed` when it is not frozen. A frozen `Linear` star with one detection stays `Linear` and keeps its input `x0` and `vx`, even though it does not have enough epochs to support that model.

`motion_models` is still the list demotion chooses from. Freezing `Orbit` does not add `Orbit` to that list and does not change `motion_models[-1]`, so new stars are unaffected.

`n_params = 0` means an unfrozen `Orbit` star is never demoted for lack of epochs (`n_fit < 0` cannot happen). It can still lose `x0` and `vx` to the fit-parameter reset. The freeze is what prevents that.

### 5.5 Recommended Galactic Center call

```python
MosaicToRef(
    ref_list,
    starlists,
    dr_tol=...,
    motion_models=['Empty', 'Fixed', 'Linear', 'Acceleration', 'Parallax', 'Orbit'],
    fixed_motion_models=['Orbit'],
    fixed_params_dict={'mass': 4.0e6, 'dist': 8.0e3},  # optional; these are the defaults
    update_ref_orig=False,
)
```

`mass` and `dist` can be omitted. Passing them makes the choice visible at the call. `fixed_motion_models=['Orbit']` is not optional if the input `x0` and `vx` of those stars must survive. `dr_tol` is the match radius for the orbit stars and for every other star.

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
- `update_ref_orig=False` and no freeze: the orbit elements are still the input elements (they are fixed parameters). Document that `x0` and `vx` of that star are cleared by the fit-parameter reset, so the test that wants them kept uses the freeze.

## 7. Decisions

1. **Coordinate frame.** Resolved. Hardcoded inside `Orbit.model`: `x = -east`, `y = +north`. There is no `x_sign` or `y_sign` parameter.
2. **Black-hole mass and distance.** Resolved. Optional fixed parameters, defaults `mass=4.0e6` Msun and `dist=8.0e3` pc, same lookup as `Parallax`'s `pa` and `obsLocation`. These defaults reproduce the `A` field of `orbits.dat`. The gcwork pair is an explicit override, not the default.
3. **Elements are fixed.** Resolved for this version. No least-squares orbit fit.
4. **Which stars are frozen.** Resolved. Caller-supplied `fixed_motion_models` plus an optional `fix_motion` column. `Orbit` is not frozen unless the caller says so. The recommended align passes `fixed_motion_models=['Orbit']`.
5. **Match radius.** Resolved. Orbit stars use the same `MosaicToRef` `dr_tol` as every other star. The file's `search` column is parsed and discarded. There is no per-star search radius in this version and none planned as a later phase.
6. **Position errors.** Resolved. `model` returns `xe = ye = 0` when errors are requested. There is no `pos_err` parameter. `'ref,var'` and `'ref,std'` drop those stars; schemes that use the science-list errors do not.

Still open, and not blocking the plan:

- Whether to ship `orbits.dat` inside the package or only accept a path. Tests can use a checked-in copy of the 32-line file either way.

## 8. Implementation order

1. `flystar/orbits.py`: Newtonian solver only, `kep2xyz`, `read_orbits_dat`, `attach_orbits`. Parse `A` and `search`, do not store them. Tests against the analytic orbit, the `a_mas` check against file `A`, and gcwork with explicit mass and distance.
2. `Orbit` in `motion_model.py`, using `optional_fixed_params` for `mass` and `dist`. Tests for defaults, overrides, and `infer_positions`.
3. `determine_motion_models`: optional defaults count as available, and the explicit-request loop reads `meta`.
4. `fixed_motion_models` on `MosaicToRef`, honored in `update_ref_table_aggregates` and `fit_motion_models`, including the one-epoch path, demotion, and all four `update_ref_orig` settings.
5. One mixed-catalog align test: orbit star frozen, linear star refit, positions at each epoch coming from the right model.

## 9. Out of scope

Fitting orbital elements. Posterior samples. Light-time delay. GR periapse advance and relativistic redshift: not parameters, not flags, and not branches in `flystar/orbits.py`. A per-star match radius, including any use of the file's `search` column. Editing `MosaicSelfRef` beyond storing an empty freeze list so the shared aggregate method can read it. Any change to stars whose model was not listed and whose `fix_motion` is not set.
