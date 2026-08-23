=============
Motion models
=============

A motion model describes how one star's position changes with time. On this
branch these are first-class, pluggable classes in
:mod:`flystar.motion_model`, all deriving from
:class:`~flystar.motion_model.MotionModel`, and the alignment code selects
between them **per star** rather than applying one model to the whole table.

This is the part of the API that has changed most relative to ``main``.

The available models
====================

.. list-table::
   :header-rows: 1
   :widths: 16 10 34 40

   * - Class
     - ``n_params``
     - Fit parameters
     - Required fixed parameters
   * - :class:`~flystar.motion_model.Empty`
     - 0
     - --
     - --
   * - :class:`~flystar.motion_model.Fixed`
     - 1
     - ``x0``, ``y0``
     - --
   * - :class:`~flystar.motion_model.Linear`
     - 2
     - ``x0``, ``vx``, ``y0``, ``vy``
     - ``t0``
   * - :class:`~flystar.motion_model.Acceleration`
     - 3
     - ``x0``, ``vx0``, ``ax``, ``y0``, ``vy0``, ``ay``
     - ``t0``
   * - :class:`~flystar.motion_model.Parallax`
     - 3
     - ``x0``, ``vx``, ``y0``, ``vy``, ``pi``
     - ``t0``, ``ra``, ``dec`` (optional: ``pa``, ``obsLocation``)

``n_params`` is the number of epochs a star needs before the model is
fittable, per coordinate -- it is derived as ``int((n_fit_params + 1) / 2)``,
not set by hand. It is also the quantity models are ranked by when FlyStar
decides which model a given star can support.

*Fit* parameters are solved for. *Fixed* parameters are supplied by the caller,
either as a column on the table or through ``fixed_params_dict``; scalars apply
to every star, arrays must have length ``N_stars``. ``t0`` is the one exception:
if you do not supply it, it is computed per star as the uncertainty-weighted
mean of that star's epochs, ``np.average(t, weights=1/np.hypot(xe, ye))``.

Fitted parameters land in per-star columns named after the parameter, with
uncertainties in ``<param>_err`` -- ``vx`` and ``vx_err``, ``pi`` and
``pi_err``.

How a model gets chosen
=======================

Two columns govern this, and the asymmetry between them matters:

``motion_model_input``
    Optional, supplied by you. A per-star request.

``motion_model_used``
    Always written by the fit. What actually happened.

:meth:`~flystar.startables.StarTable.fit_motion_models` resolves the two like
this:

1. **No** ``motion_model_input`` **column.** Each star gets the most complex
   model in your ``motion_models`` list that it has enough epochs to support
   (``n_fit >= n_params``). If two candidate models have the same
   ``n_params``, this is ambiguous and raises ``AssertionError`` rather than
   picking one -- supply ``motion_model_input`` to disambiguate.
2. **With a** ``motion_model_input`` **column.** Your request wins, wherever the
   star can actually support it. A star that cannot falls back to the most
   complex model it *can* support, drawn from the union of ``motion_models``
   and the models named in the column.

``Empty`` and ``Fixed`` are always added to the candidate list even if you did
not ask for them, so stars detected zero or one time still get a well-defined
model instead of failing.

.. code-block:: python

   # Everything linear where possible; a handful of known parallax targets
   # get the parallax model.
   table['motion_model_input'] = 'Linear'
   table['motion_model_input'][is_target] = 'Parallax'

   table.fit_motion_models(motion_models=['Linear', 'Parallax'],
                           fixed_params_dict={'ra': ra_deg, 'dec': dec_deg})

   # Check what you actually got, not what you asked for.
   import numpy as np
   print(np.unique(table['motion_model_used'], return_counts=True))

A star that gets demoted to a simpler model has any leftover parameters from
its previously-assigned model reset, so ``vx`` is never left holding a stale
value from an earlier, more complex fit.

Choosing a model versus propagating with one
--------------------------------------------

:func:`~flystar.motion_model.determine_motion_models` answers a related but
distinct question, and the ``motion_models`` argument is what separates them:

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
contract: **shape decides meaning.** Nothing is inferred from ``len(t)``
happening to equal ``N_stars``.

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

Propagating the whole table to a single new epoch does not need the column-vector
form -- pass the scalar epoch and let each star's own ``t0`` supply the
difference:

.. code-block:: python

   x, y, xe, ye = table.infer_positions(2026.5)

Uncertainties: the ``absolute_sigma`` convention
================================================

Fitting and averaging both follow :func:`scipy.optimize.curve_fit`'s
``absolute_sigma`` convention, and the same flag means the same thing in
:meth:`~flystar.motion_model.MotionModel.fit`,
:meth:`~flystar.startables.StarTable.fit_motion_models` and
:meth:`~flystar.startables.StarTable.combine_lists`:

``absolute_sigma=True`` (default)
    Your ``xe``/``ye`` are trusted as given, and the reported parameter errors
    are propagated from them directly.

``absolute_sigma=False``
    Errors are rescaled by ``sqrt(chi2/dof)``, so only the *relative*
    magnitudes of ``xe``/``ye`` matter and the result reflects the epochs' own
    scatter.

In :meth:`~flystar.startables.StarTable.combine_lists`, the reported value is
always an uncertainty **of the mean**, never the scatter of the points, in
every weighting branch.

Fitting: one star or a whole table
==================================

:meth:`~flystar.motion_model.MotionModel.fit` handles both, dispatching on
dimensionality:

* **1D** arrays of shape ``(n_epochs,)`` -- a single star, already filtered down
  to its real epochs.
* **2D** arrays of shape ``(n_stars, n_epochs)`` -- a batch, packed
  rectangularly, with ``nan`` marking the padding where a star has fewer real
  epochs than the widest row.

Every concrete model's solve is closed-form and vectorized across the batch, so
both paths run through the same non-iterative code -- the single-star case is
just a batch of one row. In practice you rarely call this directly;
:meth:`~flystar.startables.StarTable.fit_motion_models` is the entry point, and
it takes the 2D path.

Bootstrap and parallelism
-------------------------

``bootstrap=N`` resamples each star's epochs ``N`` times to get empirical
parameter errors. This is the one path that is *not* vectorized across stars,
so it is also the only reason to reach for multiprocessing:

.. code-block:: python

   table.fit_motion_models(motion_models=['Linear'],
                           bootstrap=100, seed=42,
                           processes=8)

``processes > 1`` only actually spins up a pool once the number of stars needing
the per-star path exceeds ``mp_star_threshold`` (default 100,000); below that,
pool startup and pickling the shared arrays cost more than they save, so fitting
stays serial. Measured break-even was between 20,000 and 100,000 stars on a
10-core machine.
