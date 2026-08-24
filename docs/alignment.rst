=========
Alignment
=========

Alignment is the iterative loop that turns a pile of single-epoch star lists
into one cross-matched :class:`~flystar.startables.StarTable` in a common
coordinate frame. Two driver classes in :mod:`flystar.align` implement it, and
they differ only in what the common frame *is*.

:class:`~flystar.align.MosaicSelfRef`
    The frame is defined by the data itself. One list is nominated as the
    reference for the first pass (``ref_index``, default 0); from the second
    iteration onward the reference becomes the sigma-clipped mean position of
    all lists. Use this when you have no external astrometric anchor and want a
    self-consistent relative frame.

:class:`~flystar.align.MosaicToRef`
    The frame is an external reference list you supply -- Gaia, an HST catalog,
    a previous alignment. The reference is not redefined by the data.

Both are constructed with their parameters and then run with ``fit()``:

.. code-block:: python

   from flystar import align, transforms

   msc = align.MosaicSelfRef(
       list_of_starlists,
       iters=3,
       dr_tol=[1.0, 0.5, 0.3],       # one entry per iteration
       dm_tol=[2.0, 1.0, 1.0],
       outlier_tol=[None, 3.0, 3.0],
       trans_class=transforms.PolyTransform,
       trans_args=[{'order': 1}, {'order': 2}, {'order': 2}],
       motion_models=['Linear'],
       matching='chi2',
   )
   msc.fit()

   ref = msc.ref_table          # the StarTable result
   trans = msc.trans_list       # the derived transform objects, one per list

Note the per-iteration lists. ``dr_tol``, ``dm_tol``, ``outlier_tol`` and
``trans_args`` each take one entry per iteration, which is how you tighten
tolerances and raise the transformation order as the solution converges.

The parameters, group by group
==============================

Both aligners take the same arguments except where noted; there are a lot, so
they are grouped by what they control. Only ``list_of_starlists`` (and, for
:class:`~flystar.align.MosaicToRef`, ``ref_list``) is required.

Per-iteration schedules
-----------------------

These are the ones that take **one entry per iteration**, and are how the
solution converges: match loosely at first, then tighten as the transformation
becomes better known.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``iters``
     - ``1``
     - Number of match/transform/average passes. Sets the length every
       schedule below must have.
   * - ``dr_tol``
     - ``[1.0]``
     - Match radius per iteration, in reference-frame units. Must be generous
       enough on the first pass to cover the initial frame offsets.
   * - ``dm_tol``
     - ``[1.0]``
     - Match magnitude tolerance per iteration.
   * - ``outlier_tol``
     - ``None``
     - Sigma clipping on the transformation residuals, per iteration. ``None``
       in a slot means no rejection that pass. Rejection re-derives the
       transformation, so the rejected stars stop influencing it.
   * - ``trans_args``
     - ``{'order': 1}``
     - Arguments to ``trans_class``. A single dict applies to every iteration;
       a list of dicts raises the order as the fit converges.
   * - ``mag_lim``
     - ``None``
     - Magnitude range for deriving the transformation. ``[min, max]`` applies
       everywhere; ``(N_iters, 2)`` varies by iteration; ``(N_iters, N_lists,
       2)`` varies by list as well.

Matching
--------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``matching``
     - ``'legacy'``
     - How a star with several candidates inside the tolerances is resolved.
       See the section above; use ``'chi2'`` in crowded fields.
   * - ``dchi2_tol``
     - ``9.0``
     - ``'chi2'`` only. How much better the winner must be than the runner-up,
       in :math:`\chi^2`. Below it the star is treated as ambiguous and left
       unmatched. 9.0 is a 3-sigma margin.
   * - ``match_sigma_pos``
     - ``None``
     - ``'chi2'`` only. Position scale for the :math:`\chi^2`. ``None``
       measures it from each list's own unambiguous pairs, so no error columns
       are needed.
   * - ``match_sigma_mag``
     - ``None``
     - ``'chi2'`` only. Magnitude scale, measured the same way.

The initial guess
-----------------

The first pass has no transformation yet, so one has to be found blind.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``init_guess_mode``
     - ``'miracle'``
     - ``'miracle'`` blind-matches triangles among the brightest stars, using
       only positions and magnitudes. ``'name'`` intersects the ``name``
       columns -- cheaper and more robust, but only if your lists share a
       naming scheme.
   * - ``init_order``
     - ``1``
     - Order of the transformation used for that first guess.
   * - ``briteN``
     - ``None``
     - ``'miracle'`` only. How many of the brightest stars to triangle-match.
       ``None`` uses ``min(50, len(star_list))``.
   * - ``ignore_contains``
     - ``'star'``
     - ``'name'`` only. Excludes names containing this substring; see the note
       above. ``None`` matches on every name.
   * - ``trans_input``
     - ``None``
     - A list of ready-made transform objects to start from, one per list. Skips
       the blind guess entirely.
   * - ``starlist_vertices``
     - ``None``
     - Polygon vertices per list, shape ``(N_lists, N_vertices, 2)``, confining
       the initial guess to stars inside them. For partially overlapping lists.
       :class:`~flystar.align.MosaicToRef` also takes ``reflist_vertex`` for the
       reference.

The transformation
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``trans_class``
     - ``PolyTransform``
     - The transformation model. See the table below.
   * - ``trans_weights``
     - ``None``
     - Which uncertainties weight the transformation fit:
       ``'both,var'``, ``'list,var'``, ``'ref,var'`` (or the ``,std``
       variants). ``None`` is unweighted. Needs position errors to exist.
   * - ``calc_trans_inverse``
     - ``False``
     - Also derive the inverse transformations, into ``trans_list_inverse``.
   * - ``mag_trans``
     - ``True``
     - Fit a magnitude offset per list alongside the positional transformation.

Which stars drive the fit
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``mag_lim``
     - ``None``
     - Magnitude range used for deriving transformations, before ``mag_trans``.
       Accepts ``[min, max]``, ``(N_iters, 2)`` for per-iteration limits, or
       ``(N_iters, N_lists, 2)`` to vary by list as well. The single axis is
       iterations, like ``dr_tol``.
   * - ``iter_callback``
     - ``None``
     - Called with ``(ref_table, iteration)`` after every iteration, and once
       more with ``iters``. Set ``use_in_trans = False`` on a row to drop that
       star from later transformations while keeping it in the output.
   * - ``ref_index``
     - ``0``
     - :class:`~flystar.align.MosaicSelfRef` only. Which list seeds the frame on
       the first pass; later passes use the sigma-clipped mean of all lists.
   * - ``ref_mag_lim``
     - ``None``
     - :class:`~flystar.align.MosaicToRef` only. Magnitude range on the
       reference list.
   * - ``use_ref_new``
     - ``False``
     - :class:`~flystar.align.MosaicToRef` only. Whether stars newly added to
       ``ref_table`` are used in subsequent transformations, or merely carried.
   * - ``update_ref_orig``
     - ``False``
     - :class:`~flystar.align.MosaicToRef` only. Whether to update the reference
       positions and motions as lists are transformed. Keep ``False`` to hold an
       absolute frame such as Gaia; ``True`` treats the reference as an initial
       guess.

Motion fitting
--------------

These are handed straight to
:meth:`~flystar.startables.StarTable.fit_motion_models`; see
:doc:`motion_models` and :doc:`uncertainties`.

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Parameter
     - Default
     - What it does
   * - ``motion_models``
     - ``['Empty', 'Fixed']``
     - Candidate models. Each star gets the most complex one it has enough
       epochs to support, unless a ``motion_model_input`` column requests
       otherwise. Add ``'Linear'`` if you want proper motions at all -- the
       default fits none.
   * - ``fixed_params_dict``
     - ``None``
     - Fixed model parameters, e.g. ``{'ra': ..., 'dec': ...}`` for
       ``Parallax``. ``t0`` is computed per star if omitted.
   * - ``vel_weights``
     - ``'var'``
     - Weighting scheme for the motion fit: ``'var'`` for
       :math:`1/\sigma^2`, ``'std'`` for :math:`1/|\sigma|`.
   * - ``absolute_sigma``
     - ``True``
     - ``scipy``'s convention for the reported errors.

Output and bookkeeping
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - What it does
   * - ``save_path``
     - ``None``
     - Directory for the results: ``PREFIX_input.txt``,
       ``PREFIX_ref_table.hdf5``, ``PREFIX_trans_list.pkl``.
   * - ``prefix_name``
     - ``'msr'`` / ``'mtr'``
     - Prefix for those file names.
   * - ``save_object``
     - ``True``
     - Also pickle the whole mosaic object. Heavier and tied to flystar's class
       definitions, but lets you reload and call methods like
       ``calc_bootstrap_errors`` later.
   * - ``save_plot``
     - ``True``
     - Also save a transformation diagnostic plot per (list, iteration). Real
       cost on large lists -- an unthinned scatter at dpi=300 each time.
   * - ``inherit_n_detect``
     - ``True``
     - If an input list carries its own ``n_detect`` (because it is itself the
       output of an earlier align), add that instead of counting 1, so
       ``n_detect`` totals raw detections.
   * - ``verbose``
     - ``True``
     - ``0``-``9``, or ``False``/``True`` for least/most.

What ``fit()`` produces
=======================

``fit()``'s output is ``self.ref_table``. It carries both the full 2D record of
every matched measurement -- ``x``, ``y``, ``m`` and their uncertainties, one
column per input list -- and the collapsed per-star quantities: ``x0``, ``y0``,
``m0``, their ``_err`` counterparts, ``n_detect``, and the motion-model
parameters with ``motion_model_used`` recording which model each star got.

``self.trans_list`` holds the derived :class:`~flystar.transforms.Transform2D`
objects, one per input list. If ``calc_trans_inverse=True`` was set,
``self.trans_list_inverse`` holds their inverses.

Passing ``save_path`` writes the results to disk: ``PREFIX_input.txt`` (the fit
parameters), ``PREFIX_ref_table.hdf5``, and ``PREFIX_trans_list.pkl`` (the
transform objects need pickling, as they are not plain data).

Matching
========

The ``matching`` keyword selects how a star with more than one candidate inside
the tolerances is resolved.

``'legacy'`` (default)
    The historical rule: a multi-candidate star is matched only if its nearest
    candidate in position is also its nearest in magnitude. In a crowded field
    this discards good matches -- a candidate 20x closer in position loses to
    one a few hundredths of a magnitude nearer -- and each discarded star then
    becomes a duplicate reference row that makes the next star list ambiguous in
    turn, so one split seeds the next.

``'chi2'``
    Candidates are scored as ``(dr/sigma_pos)**2 + (dm/sigma_mag)**2`` and only
    reciprocal best pairs winning by ``dchi2_tol`` (default 9.0, a 3-sigma
    margin) are kept; below that margin the star is treated as genuinely
    ambiguous and left unmatched. The scales are measured from the star lists'
    own unambiguous pairs by default, so no error columns are required. See
    :func:`~flystar.match.match_chi2`.

If you are aligning a crowded field and seeing single stars split into several
rows, ``matching='chi2'`` is the fix.

The initial guess for the first transformation comes from
``init_guess_mode='miracle'`` (:func:`~flystar.match.miracle_match_briteN`,
a triangle match on the ``briteN`` brightest stars) at order ``init_order``.
Where the lists only partially overlap, ``starlist_vertices`` restricts the
initial guess to stars inside the given polygons.

``init_guess_mode='name'`` is the alternative, and is much more robust when
your star lists already carry consistent names. It has one sharp edge worth
knowing: ``ignore_contains`` (default ``'star'``) excludes any name containing
that substring from the match. The default is deliberate -- auto-detected
sources labelled ``star_1``, ``star_2``, ... carry per-epoch detection indices
rather than stable identities, so matching on them would pair unrelated stars.
But it means a catalog whose names happen to contain ``'star'`` gets filtered
away. Pass ``ignore_contains=None`` when your names are genuinely stable across
epochs. The filter warns when it excludes anything, and ``''`` is rejected
rather than treated as "off".

Choosing a transformation
=========================

``trans_class`` and ``trans_args`` select the transformation model from
:mod:`flystar.transforms`. The useful ones:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Class
     - Use for
   * - :class:`~flystar.transforms.Shift`
     - Translation only.
   * - :class:`~flystar.transforms.four_paramNW`
     - Translation, rotation, single scale.
   * - :class:`~flystar.transforms.PolyTransform`
     - General polynomial of ``order``; the default (``order=1``).
   * - :class:`~flystar.transforms.LegTransform`
     - Legendre basis -- better conditioned than a raw polynomial at high
       order.
   * - :class:`~flystar.transforms.PolyClipTransform`,
       :class:`~flystar.transforms.LegClipTransform`
     - Clipped variants, for keeping the fit inside a valid domain.
   * - :class:`~flystar.transforms.SplineTransform`, and the
       ``*ClipSplineTransform`` variants
     - Spatially varying distortion that a global polynomial cannot absorb.

Controlling which stars drive the fit
=====================================

Several mechanisms narrow the set of stars used to *derive* transformations,
without dropping them from the output. In order of increasing specificity:

``mag_lim``
    A magnitude range per list.

``outlier_tol``
    Per-iteration sigma clipping on the transformation residuals. Rejection
    triggers a re-derivation of the transformation, so the rejected stars do
    not continue to influence it.

The ``'w'`` column
    An optional per-star weight column on an input
    :class:`~flystar.starlists.StarList`. The weight entering the
    transformation is ``star_list['w'] * ref_list['w'] * weight_from_keyword``.
    This is the most flexible knob, letting you decide as a function of both
    star and epoch what is good enough to constrain the transformation. Being
    usable (``w > 0``) is necessary but not sufficient -- magnitude limits and
    outlier rejection still take precedence.

``iter_callback``
    A function called with ``(ref_table, iteration)`` at the end of every
    iteration, and once more after the final re-matching pass with an index of
    ``iters`` -- one past the last iteration -- so that final call can be told
    apart from the end of the last iteration. The table handed in is the live
    ``ref_table``, so setting ``use_in_trans = False`` on a row excludes that
    star from subsequent transformations while keeping it in the output.

.. code-block:: python

   def reject_bright_saturated(table, i):
       if i == 0:
           table['use_in_trans'][table['m0'] < 10.0] = False

   msc = align.MosaicSelfRef(lists, iters=3, iter_callback=reject_bright_saturated,
                             dr_tol=[1., .5, .3], dm_tol=[2., 1., 1.])
   msc.fit()

Parallelism
===========

``fit()`` takes ``processes`` for the per-star motion-model fitting (see
:doc:`motion_models` -- it only helps when ``bootstrap > 0``) and,
independently, ``match_workers`` for the KDTree neighbour search inside
:func:`~flystar.match.match`.

``match_workers`` defaults to 1, deliberately: grabbing every core is the wrong
default on a shared machine. Set ``-1`` to use all cores, which is measurably
faster on large star lists and does not change the matching result -- the
per-query neighbour lists are identical, order included, regardless of thread
count.

Getting to an absolute frame
============================

After a relative alignment, :mod:`flystar.analysis` provides the Gaia path:
:func:`~flystar.analysis.query_gaia`,
:func:`~flystar.analysis.prepare_gaia_for_flystar` and
:func:`~flystar.analysis.project_gaia` to build the reference list, then
:meth:`~flystar.startables.StarTable.shift_reference_frame` to apply the
resulting velocity and parallax offsets to both the fitted motion parameters
and the time-series astrometry.

Note that :func:`~flystar.analysis.query_gaia` contacts the live Gaia TAP
service and the astroquery version pinned here exposes no timeout, so a slow
day at ESA will block rather than fail. Cache the catalog once and read it from
disk in anything automated.
