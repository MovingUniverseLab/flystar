# Plan: Keplerian `Orbit` motion model

Plan only. No source changes in this branch. The base is `mm_rework_lingfeng`.

This adds a prediction-only Keplerian model for Galactic Center stars orbiting Sgr A*, read from `orbits.dat`, and wires it into `MosaicToRef` so those stars are propagated to each epoch while every other star keeps using `Empty`, `Fixed`, `Linear`, `Acceleration`, or `Parallax`.

Orbital elements are not free parameters in this version. A separate, generic align option says which motion models are not refit. `Orbit` is the reason for that option. It is not hard-coded.

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

`motion_model_input` / `motion_model_used` use a derived string width, `_MOTION_MODEL_NAME_WIDTH` in `startables.py`, not a hard-coded `U20`. Adding `Orbit` widens the column.

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
| `x_sign` | `-1.0` | | Multiplies the east offset from `kep2xyz`. |
| `y_sign` | `+1.0` | | Multiplies the north offset. |
| `pos_err` | `0.0` | arcsec | Constant `xe` and `ye` when errors are requested. |
| `gr_orbit` | `False` | | Secular GR periapse advance. Off in this version. |
| `rel_redshift` | `False` | | Line-of-sight redshift term. No effect on sky position. Off in this version. |

`mass = 4.0e6` and `dist = 8.0e3` are the values that reproduce the `A` column of `orbits.dat` v2.0.2. They are not the gcwork `Constants` pair (`4.07e6` Msun, `7960.1` pc). A caller who wants the gcwork pair passes them explicitly:

```python
MosaicToRef(..., fixed_params_dict={'mass': 4.07e6, 'dist': 7960.1})
```

Because the defaults are uniform, the first fit that actually includes an `Orbit` star stores them in `table.meta` unless a column already exists. `MosaicToRef.fixed_params_dict` outranks both.

`t_bh` is the epoch of the black-hole offset. It is not periapse and it is not the stellar `t0`. With the black hole at the origin and at rest, `t_bh` drops out.

Sky position at decimal year `t`:

```text
r_east, r_north, r_los = kep2xyz(...)     # arcsec, relative to the black hole
x = x_bh + vx_bh * (t - t_bh) + x_sign * r_east
y = y_bh + vy_bh * (t - t_bh) + y_sign * r_north
```

`r_los` is not written to the catalog. `parallax.py` already uses this frame: `x = -east * cos(pa) + north * sin(pa)` at `pa = 0` is `x = -east`.

### 2.2 Catalog columns from `orbits.dat`

v2.0.2, whitespace-separated, no header. Thirty-two stars. Columns in order:

| File column | Unit | Catalog column | Used to predict? |
|---|---|---|---|
| name | | `name` | match key only |
| P | yr | `orb_P` | yes |
| A | mas | `orb_A` | stored, not an input to `kep2xyz` |
| t0 | decimal year | `orb_t0` | yes, time of periapse |
| e | | `orb_e` | yes |
| i | deg | `orb_i` | yes |
| Omega | deg | `orb_Omega` | yes, PA of the ascending node |
| omega | deg | `orb_omega` | yes, argument of periapse |
| search | pix | `orb_search` | stored, not a motion parameter |

`orb_A` and `orb_search` are not in `fixed_param_names`. The fit-parameter reset does not clear them.

Do not reuse `t0`, `x0`, `y0`, `vx`, or `vy` for elements. `t0` is the epoch of the linear model and is replaced by the weighted mean detection time. `x0` and `vx` are the linear model's position and proper motion. An instantaneous orbital velocity is a different number; writing it into `vx` would make `x0 + vx*(t - t0)` wrong for any star later demoted to `Linear`.

Non-orbit stars have NaN in the `orb_*` columns. Their `motion_model_input` stays whatever the catalog already said.

### 2.3 Errors

`orbits.dat` has no uncertainties. This version does not propagate element errors. When `model` is asked for errors it returns `xe = ye = pos_err`.

Matching ignores errors. Transform weights do not. `pos_err = 0` is fine for `trans_weights is None` and for schemes that also use the science-list errors. `'ref,var'` and `'ref,std'` turn a zero reference error into a non-finite weight, and `get_weights_for_lists` then sets that weight to 0, which drops the star from the transform. Callers who want orbit stars to carry weight under those schemes set `pos_err` to a floor. Do not invent a covariance.

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

`gr_orbit=True` and `rel_redshift=True` raise `NotImplementedError` in this version. The flags exist so a later version can turn them on without a new parameter name.

With `M = 4.0e6` and `R0 = 8000`, `a_mas` from the printed `P` matches column `A` for all 32 stars to well under 0.02 mas. The largest residual is S0-105, about 0.013 mas, because `P` is printed as `152.76` (0.01 yr). That is the rounding of `P`, not a different mass. The test asserts `abs(a_mas - A) <= 0.02` for every star. The gcwork cross-check does not use these defaults; it passes `mass=4.07e6` and `dist=7960.1` and compares sky positions to `kep2xyz`.

## 4. Reader

`flystar/orbits.py` function `read_orbits_dat(path) -> astropy.Table` with the columns in section 2.2. No header. Names stay as written (`S0-2`, not a different spelling).

`attach_orbits(starlist, orbits)` matches on `name`:

- Matched rows: set `motion_model_input` to `'Orbit'` and copy the `orb_*` columns.
- Unmatched catalog rows: leave `motion_model_input` unchanged and set `orb_*` to NaN.
- Names in the orbit file that are not in the catalog: warn, do not add rows.

Attach is explicit. Loading a starlist does not look for `orbits.dat`.

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

`mass` and `dist` can be omitted. Passing them makes the choice visible at the call. `fixed_motion_models=['Orbit']` is not optional if the input `x0` and `vx` of those stars must survive.

## 6. Tests

New files: `flystar/tests/test_orbits.py` and additions to `flystar/tests/test_motion_model.py` and `flystar/tests/test_align.py`. Use the existing pytest style. No live download of gcwork.

Solver and model:

- Circular and eccentric analytic positions match `kep2xyz` at a grid of epochs.
- `x` equals minus the east offset and `y` equals the north offset when the black hole is at the origin.
- `Orbit()` has `optional_fixed_params['mass'] == 4.0e6` and `optional_fixed_params['dist'] == 8.0e3`. `model` with no `mass` or `dist` uses those. A `fixed_params_dict` override, a column, and a `meta` entry each win in that order.
- For every star in `orbits.dat` v2.0.2, `a_mas` from the printed `P` with those defaults is within 0.02 mas of column `A`.
- Against gcwork `kep2xyz`, pass `mass=4.07e6` and `dist=7960.1` explicitly. Compare east and north at several epochs, including periapse and a time far from it. Do not use the FlyStar defaults for this comparison.
- `read_orbits_dat` returns 32 rows and the S0-2 elements. `attach_orbits` sets `motion_model_input` only on name matches.
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

1. **Coordinate frame.** Resolved. FlyStar `+x` is west. `x_sign=-1`, `y_sign=+1`.
2. **Black-hole mass and distance.** Resolved. Optional fixed parameters, defaults `mass=4.0e6` Msun and `dist=8.0e3` pc, same lookup as `Parallax`'s `pa` and `obsLocation`. These defaults reproduce `orbits.dat` column `A`. The gcwork pair is an explicit override, not the default.
3. **Elements are fixed.** Resolved for this version. No least-squares orbit fit.
4. **Which stars are frozen.** Resolved. Caller-supplied `fixed_motion_models` plus an optional `fix_motion` column. `Orbit` is not frozen unless the caller says so. The recommended align passes `fixed_motion_models=['Orbit']`.

Still open, and not blocking the plan:

- Whether to ship `orbits.dat` inside the package or only accept a path. Tests can use a checked-in copy of the 32-line file either way.
- GR periapse advance and the redshift term stay off until a later version turns the existing flags on.
- `orb_search` is stored and not used as a match radius in this version.

## 8. Implementation order

1. `flystar/orbits.py`: Newton solver, `kep2xyz`, `read_orbits_dat`, `attach_orbits`. Tests against the analytic orbit, column `A`, and gcwork with explicit mass and distance.
2. `Orbit` in `motion_model.py`, using `optional_fixed_params` for `mass` and `dist`. Tests for defaults, overrides, and `infer_positions`.
3. `determine_motion_models`: optional defaults count as available, and the explicit-request loop reads `meta`.
4. `fixed_motion_models` on `MosaicToRef`, honored in `update_ref_table_aggregates` and `fit_motion_models`, including the one-epoch path, demotion, and all four `update_ref_orig` settings.
5. One mixed-catalog align test: orbit star frozen, linear star refit, positions at each epoch coming from the right model.

## 9. Out of scope

Fitting orbital elements. Posterior samples. Light-time delay. GR, other than the flags that raise `NotImplementedError`. Changing the match radius from `orb_search`. Editing `MosaicSelfRef` beyond storing an empty freeze list so the shared aggregate method can read it. Any change to stars whose model was not listed and whose `fix_motion` is not set.
