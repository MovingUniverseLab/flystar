=============
Motion models
=============

Once every epoch sits in a common frame, each star has a time series of
positions, and the question becomes what curve to put through it. A *motion
model* is that curve. On this branch these are first-class pluggable classes in
:mod:`flystar.motion_model`, all deriving from
:class:`~flystar.motion_model.MotionModel`, and FlyStar chooses between them
**per star** rather than imposing one on the whole table.

This is the part of the API that differs most from ``main``.

.. _the-models:

The models
==========

Every model predicts a position at time :math:`t` from parameters it *fits* and
parameters you *fix*. ``n_params`` is how many epochs a star needs before the
model is fittable, per coordinate; it is derived as
``int((n_fit_params + 1) / 2)``, not set by hand, and it is also the quantity
models are ranked by when FlyStar decides what a star can support.

Throughout, :math:`\Delta t \equiv t - t_0`.

.. list-table::
   :header-rows: 1
   :widths: 14 8 24 54

   * - Model
     - ``n_params``
     - Fit parameters
     - Position model
   * - :class:`~flystar.motion_model.Empty`
     - 0
     - --
     - :math:`x(t) =` ``fill_value`` (NaN by default), :math:`\sigma_x(t) = \infty`.
       For stars with no usable detection at all.
   * - :class:`~flystar.motion_model.Fixed`
     - 1
     - :math:`x_0,\ y_0`
     - :math:`x(t) = x_0`, the weighted mean position -- no time dependence.
       For stars seen once, or held still on purpose.
   * - :class:`~flystar.motion_model.Linear`
     - 2
     - :math:`x_0, v_x,\ y_0, v_y`
     - :math:`x(t) = x_0 + v_x\,\Delta t`. Constant proper motion; the
       workhorse.
   * - :class:`~flystar.motion_model.Acceleration`
     - 3
     - :math:`x_0, v_{x0}, a_x,\ y_0, v_{y0}, a_y`
     - :math:`x(t) = x_0 + v_{x0}\,\Delta t + \tfrac{1}{2} a_x\,\Delta t^2`.
       For stars whose motion visibly curves.
   * - :class:`~flystar.motion_model.Parallax`
     - 3
     - :math:`x_0, v_x, \pi,\ y_0, v_y`
     - :math:`x(t) = x_0 + v_x\,\Delta t + \pi\,P_x(t)`. Linear motion plus
       annual parallax.

Note where :math:`\pi` sits in the parallax model: it multiplies the **parallax
vector** :math:`\boldsymbol{P}(t)`, not :math:`\Delta t`. Written out for both
coordinates,

.. math::

   x(t) &= x_0 + v_x\,(t - t_0) + \pi\,P_x(t) \\
   y(t) &= y_0 + v_y\,(t - t_0) + \pi\,P_y(t)

:math:`\boldsymbol{P}(t)` is computed by
:meth:`~flystar.motion_model.Parallax.calc_parallax_vector` from the star's
``ra``/``dec``, the position angle ``pa``, and the observatory location
``obsLocation``. Note that ``fit_motion_models`` currently applies a single
``obsLocation`` to every star in the table.

Fixed parameters
----------------

*Fit* parameters are solved for; *fixed* parameters you supply, either as a
column on the table or through ``fixed_params_dict`` (scalars apply to every
star, arrays must have length ``N_stars``).

.. list-table::
   :header-rows: 1
   :widths: 20 22 58

   * - Model
     - Required
     - Optional
   * - ``Empty``, ``Fixed``
     - --
     - --
   * - ``Linear``, ``Acceleration``
     - :math:`t_0`
     - --
   * - ``Parallax``
     - :math:`t_0`, ``ra``, ``dec``
     - ``pa`` (default 0), ``obsLocation`` (default ``'earth'``)

:math:`t_0` is the one you can always omit. Left unset, it is computed per star
as the uncertainty-weighted mean of that star's own epochs,

.. math::

   t_0 = \frac{\sum_i t_i / \sqrt{\sigma_{x,i}^2 + \sigma_{y,i}^2}}
              {\sum_i 1 / \sqrt{\sigma_{x,i}^2 + \sigma_{y,i}^2}}

which puts the reference epoch near the star's best-measured time and so
minimises the covariance between :math:`x_0` and :math:`v_x`.

Fitted parameters land in per-star columns named after the parameter, with
uncertainties in ``<param>_err`` -- ``vx`` and ``vx_err``, ``pi`` and
``pi_err``. See :doc:`uncertainties` for how those errors are computed.

How a model gets chosen
=======================

Two columns govern this, and the asymmetry between them matters:

``motion_model_input``
    Optional, supplied by you. A per-star *request*.

``motion_model_used``
    Always written by the fit. What actually *happened*.

:meth:`~flystar.startables.StarTable.fit_motion_models` resolves them so:

1. **No** ``motion_model_input`` **column.** Each star gets the most complex
   model in your ``motion_models`` list it has enough epochs to support
   (``n_fit >= n_params``). If two candidates share the same ``n_params`` this
   is ambiguous and raises ``AssertionError`` rather than guessing -- supply
   ``motion_model_input`` to disambiguate.
2. **With a** ``motion_model_input`` **column.** Your request wins wherever the
   star can support it. A star that cannot falls back to the most complex model
   it *can* support, drawn from the union of ``motion_models`` and the models
   named in the column.

``Empty`` and ``Fixed`` are always added to the candidate list even if you did
not ask for them, so stars detected zero or one time still get a well-defined
model instead of failing.

.. code-block:: python

   # Everything linear where possible; a few known parallax targets get more.
   table['motion_model_input'] = 'Linear'
   table['motion_model_input'][is_target] = 'Parallax'

   table.fit_motion_models(motion_models=['Linear', 'Parallax'],
                           fixed_params_dict={'ra': ra_deg, 'dec': dec_deg})

   # Always check what you got, not what you asked for.
   import numpy as np
   print(np.unique(table['motion_model_used'], return_counts=True))

A star demoted to a simpler model has any leftover parameters from its
previously-assigned model reset, so ``vx`` is never left holding a stale value
from an earlier, more complex fit.

Choosing a model versus propagating with one
--------------------------------------------

:func:`~flystar.motion_model.determine_motion_models` answers a related but
distinct question, and the ``motion_models`` argument separates them:

* Pass your list to ask *which of the models I requested was this star fit
  with* -- the answer must stay inside the set you asked for.
* Pass ``None`` to ask *how well can this star be propagated at all* -- a
  property of the star's own parameters, not of what you chose to fit. A
  reference star imported with a full parallax solution should be propagated
  with it even if this run only fit linear motion.

The time-argument contract
==========================

Every model's :meth:`~flystar.motion_model.MotionModel.model`, and
:meth:`~flystar.startables.StarTable.infer_positions`, take times under one
rule: **shape decides meaning.** Nothing is inferred from ``len(t)`` happening
to equal ``N_stars``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``t``
     - Meaning
   * - scalar
     - One time, every star.
   * - ``(N_times,)``
     - One shared grid, every star -- always, even when
       ``N_times == N_stars``.
   * - ``(1, N_times)``
     - The same, written explicitly.
   * - ``(N_stars, N_times)``
     - Each star has its own times.

For **one time per star**, pass a column vector ``t[:, np.newaxis]`` of shape
``(N_stars, 1)``. A bare 1D array of length ``N_stars`` means a shared grid, not
per-star times. Any other shape raises ``ValueError`` rather than being guessed
at. See :func:`~flystar.motion_model.broadcast_times`.

Propagating the whole table to a single new epoch does not need the
column-vector form -- pass the scalar epoch and let each star's own :math:`t_0`
supply the difference:

.. code-block:: python

   x, y, xe, ye = table.infer_positions(2026.5)

Fitting: one star or a whole table
==================================

:meth:`~flystar.motion_model.MotionModel.fit` handles both, dispatching on
dimensionality:

* **1D** arrays of shape ``(n_epochs,)`` -- a single star, already filtered to
  its real epochs.
* **2D** arrays of shape ``(n_stars, n_epochs)`` -- a batch packed
  rectangularly, with ``nan`` marking padding where a star has fewer real
  epochs than the widest row.

Every model's solve is closed-form and vectorized across the batch, so both
paths run through the same non-iterative code -- the single-star case is just a
batch of one row. There is no ``scipy.optimize.curve_fit`` call and no
``use_scipy`` switch on this branch; both were removed in favour of the
closed-form solves, which are tested for agreement with ``curve_fit``. In
practice you rarely call ``fit`` directly:
:meth:`~flystar.startables.StarTable.fit_motion_models` is the entry point and
takes the 2D path.

Two asymmetries between the two paths
-------------------------------------

Worth knowing before you call ``fit`` directly, because neither is obvious:

**The batch path does not fill in** :math:`t_0`. The single-star (1D) path
computes the weighted-mean :math:`t_0` for you and remembers it on
``self.fixed_params_dict``, so a later ``model(t, params)`` call works with no
further arguments. The batch (2D) path does neither -- pass
``fixed_params_dict={'t0': ...}`` to ``fit``, and the same ``t0`` again to
``model``:

.. code-block:: python

   t2d = np.broadcast_to(t, x.shape)                  # fit dispatches on t.ndim
   t0 = np.average(t2d, weights=1./np.hypot(xe, ye), axis=1)
   params, param_errs, chi2_x, chi2_y = mm.fit(t2d, x, y, xe, ye,
                                               fixed_params_dict={'t0': t0})
   x_model, y_model, xe_model, ye_model = mm.model(t_new, params, param_errs,
                                                  {'t0': t0})

Note also that dispatch is on ``t.ndim``, not on ``x`` -- a 1D ``t`` with 2D
``x`` takes the single-star path and then fails an assertion about ``x``.

**The two paths return different numbers of values.** The single-star path
returns ``(params, param_errs)``, honouring ``return_chi2``. The batch path
returns ``(params, param_errs, chi2_x, chi2_y)`` regardless of
``return_chi2``. Unpack accordingly.

Bootstrap and parallelism
-------------------------

``bootstrap=N`` resamples each star's epochs ``N`` times for empirical
parameter errors. It is the one path *not* vectorized across stars, so it is
also the only reason to reach for multiprocessing:

.. code-block:: python

   table.fit_motion_models(motion_models=['Linear'],
                           bootstrap=100, seed=42,
                           processes=8)

``processes > 1`` only spins up a pool once the number of stars needing the
per-star path exceeds ``mp_star_threshold`` (default 100,000); below that, pool
startup and pickling the shared arrays cost more than they save, so fitting
stays serial. Measured break-even was between 20,000 and 100,000 stars on a
10-core machine.
