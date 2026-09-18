===============
Transformations
===============

A transformation is the coordinate mapping that carries one star list into the
common reference frame. Every list gets its own: it is what absorbs the
arbitrary pixel origin, rotation, plate scale and distortion of the image the
list came from, so that a star's position means the same thing in every epoch.

The aligner derives these for you. ``trans_class`` picks the functional form
and ``trans_args`` supplies its arguments, both described in
:doc:`alignment`; the fitted objects come back as ``trans_list``, one
:class:`~flystar.transforms.Transform2D` per input list. This page is about
which form to pick.

Choosing a model
================

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

Raising the order as the fit converges
======================================

``trans_args`` takes either a single dict, applied everywhere, or one dict per
iteration. The per-iteration form is how you start loose and tighten:

.. code-block:: python

   trans_args=[{'order': 1}, {'order': 2}, {'order': 2}]

The first pass has only the blind initial guess to work from, so a low order is
all the matches can support. Once the frame is roughly right and the matching
has tightened, a higher order has enough well-matched stars to be worth
fitting. Going straight to a high order on the first pass fits the order to the
mismatches instead.

Giving one list its own transformation
======================================

``trans_args`` carries a per-starlist axis as well as a per-iteration one, and
is normalized to ``(N_iters, N_lists)``. Reach for the nested form when one
list needs a different transformation from the rest -- a different instrument,
or a detector whose distortion the shared order does not capture:

.. code-block:: python

   # Three lists, two iterations. The third list is the wide-field camera,
   # whose distortion a first-order transformation cannot absorb.
   trans_args=[[{'order': 1}, {'order': 1}, {'order': 2}],
               [{'order': 2}, {'order': 2}, {'order': 3}]]

The three accepted forms in full:

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Form
     - Meaning
   * - ``{'order': 2}``
     - Those arguments for every list, every iteration.
   * - ``[{...}, ...]``, length ``N_iters``
     - Per iteration, the same for every starlist.
   * - ``[[{...}, ...], ...]``, shape ``(N_iters, N_lists)``
     - Fully specified, one dict per starlist per iteration.

A flat list always indexes **iterations**, never starlists, even when its
length happens to equal the number of lists. This is the same convention as
``mag_lim``, so that one axis means the same thing across every schedule
argument. Per-list arguments therefore always use the nested form, and a
single iteration of them is ``[[...]]``, with an outer length of 1::

   # One pass, each list with its own order.
   trans_args=[[{'order': 1}, {'order': 1}, {'order': 2}]]

That outer length is a schedule like ``dr_tol``, so it takes part in setting
the iteration count and has to agree with the other schedules given. A row
whose length is not ``N_lists`` raises ``ValueError`` rather than being
broadcast, since there is no sensible way to spread the wrong number of dicts
over the lists.

:meth:`~flystar.align.MosaicSelfRef.calc_bootstrap_errors` re-fits with the
last iteration's arguments, which is the transformation the alignment
converged with.

See :doc:`alignment` for the per-iteration schedules in general, and for
``trans_weights``, ``trans_input`` and ``calc_trans_inverse``.
