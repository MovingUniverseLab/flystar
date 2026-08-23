========
Overview
========

The data model
==============

FlyStar has two container types, both subclasses of :class:`astropy.table.Table`,
and the distinction between them is the shape of their columns.

:class:`~flystar.starlists.StarList`
    **One epoch.** Columns are 1D, with one row per star: ``x``, ``y``, ``m``
    and, optionally, the uncertainties ``xe``, ``ye``, ``me``. This is what a
    single detection list from a single image looks like after being read off
    disk by :func:`~flystar.starlists.read_starlist` or
    :meth:`~flystar.starlists.StarList.from_lis_file`.

:class:`~flystar.startables.StarTable`
    **Many epochs, cross-matched.** Columns are 2D with shape
    ``(N_stars, N_lists)``, so ``table['x'][i, j]`` is star ``i`` as measured in
    list ``j``. A star not detected in list ``j`` has ``nan`` in that cell --
    that is the "no data" convention throughout the codebase, and it is what
    ``valid = isfinite(x) & isfinite(y)`` is derived from internally rather than
    from a mask the caller has to build.

    Alongside the 2D per-list columns sit 1D per-star columns holding
    quantities collapsed along the list axis: ``x0``, ``y0``, ``m0`` and their
    ``_err`` counterparts, plus the motion-model parameters.

Column naming conventions
=========================

These conventions are load-bearing -- much of the code dispatches on the
suffix of a column name, so they are worth learning before reading further.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Pattern
     - Shape
     - Meaning
   * - ``x``, ``y``, ``m``
     - 2D
     - Per-list measurement, one column per star list.
   * - ``xe``, ``ye``, ``me``
     - 2D
     - Per-list uncertainty on the above.
   * - ``x0``, ``y0``, ``m0``
     - 1D
     - Per-star value collapsed across lists, produced by
       :meth:`~flystar.startables.StarTable.combine_lists` /
       :meth:`~flystar.startables.StarTable.combine_lists_xym`.
   * - ``<name>_err``
     - 1D
     - Uncertainty on a per-star quantity -- ``x0_err``, ``vx_err``, ``pi_err``.
       Note the suffix is ``_err`` for per-star columns and a bare ``e`` for the
       2D per-list ones.
   * - ``t0``
     - 1D or meta
     - The reference epoch each star's motion model is expressed about.
   * - ``motion_model_input``
     - 1D
     - *Optional input.* The caller's per-star request for which model to use.
   * - ``motion_model_used``
     - 1D
     - *Output.* Which model was actually fit for that star.

A missing per-star uncertainty is filled with ``inf``, not ``nan``, and not with
a fabricated finite number: a star with no uncertainty information reports an
infinite error rather than a misleadingly precise one.

The typical workflow
====================

.. code-block:: python

   from flystar import align, starlists

   # 1. One StarList per epoch.
   lists = [starlists.read_starlist(f) for f in my_files]

   # 2. Match and transform them into a common frame.
   msc = align.MosaicSelfRef(lists, iters=3,
                             dr_tol=[1.0, 0.5, 0.3],
                             dm_tol=[2.0, 1.0, 1.0],
                             motion_models=['Linear'])
   msc.fit()

   # 3. The result is a StarTable with averaged positions and fitted motions.
   ref = msc.ref_table
   ref['x0'], ref['vx'], ref['motion_model_used']

   # 4. Predict where each star was, or will be, at some other time.
   x, y, xe, ye = ref.infer_positions(2026.5)

Step 2 is covered in :doc:`alignment`; steps 3 and 4 in :doc:`motion_models`.

Where the pieces live
=====================

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Module
     - Responsibility
   * - :mod:`flystar.starlists`
     - Reading, writing and trimming single-epoch lists.
   * - :mod:`flystar.startables`
     - The cross-matched multi-epoch table and its per-star operations.
   * - :mod:`flystar.match`
     - Star matching, including the triangle-based blind match
       (:func:`~flystar.match.miracle_match_briteN`) and the position/magnitude
       tolerance match (:func:`~flystar.match.match`,
       :func:`~flystar.match.match_chi2`).
   * - :mod:`flystar.transforms`
     - Coordinate transformation models -- shifts, four-parameter,
       polynomial, Legendre, and spline/clipped variants.
   * - :mod:`flystar.motion_model`
     - Per-star motion models and the machinery that chooses between them.
   * - :mod:`flystar.align`
     - The iterative match/transform/average loop that drives everything else.
   * - :mod:`flystar.analysis`
     - Gaia cross-matching and absolute-frame helpers.
   * - :mod:`flystar.plots`
     - Diagnostic plots: residual quivers, VPDs, chi-squared distributions.
