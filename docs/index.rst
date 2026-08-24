=======
FlyStar
=======

**FlyStar aligns many star lists into one common frame and fits each star's
motion there, choosing per star from a set of motion models.**

That one sentence is the whole package: *align the stars, then describe how they
move.* The rest of this page unpacks it.

Astrometry of a crowded field is never taken in one frame. You have many
images -- different nights, different instruments, different pointings, each
with its own distortion and its own arbitrary pixel coordinate system. Every
one of them tells you about the same stars, but none of them agrees with the
others about where those stars are.

FlyStar's job is to make them agree. It cross-matches the stars across all
those lists, solves for the coordinate transformation that carries each list
into one common reference frame, and then -- with every epoch on the same
footing -- fits each star's motion through time. What comes out is a single
table of positions, proper motions, and where wanted parallaxes, with
uncertainties that mean something.

That is the whole point of the package: **turn many disagreeing star lists into
one self-consistent astrometric solution.**

.. admonition:: Which version is this?
   :class: important

   Built from the ``mm_rework_lingfeng`` branch, and describes *that* branch's
   API. The motion-model framework in :doc:`motion_models` does not exist on
   ``main``. Every page here, the :doc:`API reference <api/flystar/index>`
   included, is generated from this branch's source.

How the pieces fit
==================

Three objects, in the order you meet them:

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Stage
     - Object
     - What it holds
   * - **1. One epoch**
     - :class:`~flystar.starlists.StarList`
     - One detection list from one image. 1D columns, one row per star:
       ``x``, ``y``, ``m`` and their uncertainties.
   * - **2. Many epochs**
     - :class:`~flystar.startables.StarTable`
     - The cross-matched result. 2D columns of shape
       ``(N_stars, N_lists)``, so ``t['x'][i, j]`` is star ``i`` in list ``j``.
   * - **3. The solution**
     - :class:`~flystar.align.MosaicSelfRef` /
       :class:`~flystar.align.MosaicToRef`
     - The iterative match/transform/average loop that builds the StarTable
       and fits each star's motion.

In one line: **StarList → StarTable → align**. You assemble star lists, hand
them to an aligner, and read the answer off the ``StarTable`` it produces.

Getting started
===============

:doc:`getting_started` installs FlyStar and then builds a synthetic four-epoch
data set from scratch with numpy, aligns it, and checks the recovered positions
and proper motions against the truth -- with plots at each step. Start there.

Where to go next
================

:doc:`overview`
    The data model in full: what lives in a ``StarList`` versus a
    ``StarTable``, and the column-naming conventions the code dispatches on.

:doc:`motion_models`
    The per-star motion models, their equations, how FlyStar picks one per star,
    and how the reported uncertainties are computed. Worth reading before the
    aligner, whose ``motion_models`` argument only makes sense once you know
    what it is choosing between.


:doc:`alignment`
    The aligners in depth, parameter by parameter -- matching strategies,
    transformation models, and how to control which stars drive the fit.

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Guide

   getting_started
   overview
   motion_models
   alignment

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   examples/alignment_example
   examples/motion_model_example

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   api/flystar/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Indices

   genindex
