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

``trans_args`` takes either a single dict, applied to every iteration, or one
dict per iteration. The per-iteration form is how you start loose and tighten:

.. code-block:: python

   trans_args=[{'order': 1}, {'order': 2}, {'order': 2}]

The first pass has only the blind initial guess to work from, so a low order is
all the matches can support. Once the frame is roughly right and the matching
has tightened, a higher order has enough well-matched stars to be worth
fitting. Going straight to a high order on the first pass fits the order to the
mismatches instead.

See :doc:`alignment` for the per-iteration schedules in general, and for
``trans_weights``, ``trans_input`` and ``calc_trans_inverse``.
