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

Putting it together
===================

The landing page has the end-to-end example. In short: build one
:class:`~flystar.starlists.StarList` per epoch, hand the list of them to
:class:`~flystar.align.MosaicSelfRef` (self-defined frame) or
:class:`~flystar.align.MosaicToRef` (external reference), call ``fit()``, and
read the resulting :class:`~flystar.startables.StarTable` off ``ref_table``.

:doc:`alignment` covers the aligners; :doc:`motion_models` covers the per-star
motion fit; :doc:`uncertainties` covers what the error columns mean.

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
