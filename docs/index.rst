=======
FlyStar
=======

**FlyStar cross-matches and aligns many star lists into one common frame and fits each star's
motion there, choosing per star from a set of motion models.**

Astrometry of a crowded field is never taken in one frame. You have many
images -- different nights, different instruments, different pointings, each
with its own distortion and its own arbitrary pixel coordinate system. Every
one of them tells you about the same stars, but first we must cross-match and 
align them to a common frame.

FlyStar cross-matches the stars across a stack of star lists, solves for the 
coordinate transformation that carries each list
into one common reference frame, and then fits each star's motion through time. 
What comes out is a single table of positions, proper motions, and where requested, 
parallaxes, with uncertainties.

How the pieces fit
==================

FlyStar has three main components:

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
   * - **3. The machinary**
     - :class:`~flystar.align.MosaicSelfRef` /
       :class:`~flystar.align.MosaicToRef`
     - The iterative match/transform/average loop class in align.py 
       that builds the StarTable
       and fits each star's motion.

In one line: **StarList → StarTable → align**. You assemble star lists, hand
them to an aligner, and read the answer off the ``StarTable`` it produces.

Where to go next
================

:doc:`installation`
    Installing FlyStar from the repository, and the optional extras.

:doc:`first_alignment`
    A complete alignment of synthetic star lists that runs as written, and the
    same call on your own lists.

:doc:`overview`
    The two data objects in full -- what lives in a ``StarList`` versus a
    ``StarTable``, and the column-naming conventions the code dispatches on.

:doc:`motion_models`
    The per-star motion models, their equations, how FlyStar picks one per star,
    and how the reported uncertainties are computed. Worth reading before the
    aligner, whose ``motion_models`` argument only makes sense once you know
    what it is choosing between.

:doc:`transformations`
    The coordinate transformation models, which to pick for a given kind of
    frame difference, and how to raise the order as the fit converges.

:doc:`alignment`
    The aligners in depth, parameter by parameter -- matching strategies,
    the initial guess, and how to control which stars drive the fit.

:doc:`contributors`
    The people who have written code in FlyStar.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting started

   installation
   first_alignment

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Components

   overview
   motion_models
   transformations
   alignment

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   examples/alignment_example
   examples/motion_model_example
   examples/name_match_example

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   api/flystar/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Project

   contributors

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Indices

   genindex
