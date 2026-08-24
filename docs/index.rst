=======
FlyStar
=======

**FlyStar measures where stars are and how they move.**

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

Installation
------------

FlyStar is not on PyPI; install it from the repository:

.. code-block:: bash

   git clone https://github.com/MovingUniverseLab/flystar.git
   cd flystar
   pip install -e .

Be aware that ``setup.cfg`` currently declares only ``astropy`` as a
requirement, while the package actually imports rather more at module level.
Install these too, or you will hit ``ImportError`` on the first ``import
flystar``:

.. code-block:: bash

   pip install numpy scipy astropy matplotlib pandas joblib tqdm

Three further packages are imported lazily, so you only need them for the
features that use them: ``shapely`` (polygon-restricted initial guesses via
``starlist_vertices``), ``astroquery`` (Gaia queries in
:mod:`flystar.analysis`), and ``plotly`` (interactive plots in
:mod:`flystar.plots`). Python 3.7 or newer.

Your first alignment
--------------------

Align several epochs into a self-consistent frame and fit linear motion:

.. code-block:: python

   from flystar import align, starlists, transforms

   # 1. One StarList per epoch. Each needs x, y, m columns; xe, ye, me if you
   #    have them. Set list_time so the motion fit knows when each was taken.
   lists = []
   for path, year in [('epoch1.lis', 2015.5),
                      ('epoch2.lis', 2017.4),
                      ('epoch3.lis', 2019.3)]:
       sl = starlists.StarList.from_lis_file(path)
       sl.meta['list_time'] = year
       lists.append(sl)

   # 2. Align. One entry per iteration in dr_tol/dm_tol/trans_args: tolerances
   #    tighten and the transformation order rises as the solution converges.
   msc = align.MosaicSelfRef(
       lists,
       iters=3,
       dr_tol=[1.0, 0.5, 0.3],          # match radius, reference units
       dm_tol=[2.0, 1.0, 1.0],          # match magnitude tolerance
       trans_class=transforms.PolyTransform,
       trans_args=[{'order': 1}, {'order': 2}, {'order': 2}],
       motion_models=['Linear'],        # fit x0, vx, y0, vy per star
       matching='chi2',                 # better in crowded fields
   )
   msc.fit()

   # 3. Read the answer.
   ref = msc.ref_table
   ref['x0'], ref['y0']                 # mean position at t0
   ref['vx'], ref['vy']                 # proper motion
   ref['vx_err'], ref['vy_err']         # and its uncertainty
   ref['motion_model_used']             # which model each star actually got
   ref['n_detect']                      # epochs each star was found in

   # 4. Predict where any star was, or will be, at another time.
   x, y, xe, ye = ref.infer_positions(2026.5)

If you have an external astrometric anchor -- Gaia, an HST catalog, a previous
alignment -- use :class:`~flystar.align.MosaicToRef` instead and pass it as the
reference; the frame is then fixed by that catalog rather than by the data.

Where to go next
----------------

:doc:`overview`
    The data model in full: what lives in a ``StarList`` versus a
    ``StarTable``, and the column-naming conventions the code dispatches on.

:doc:`alignment`
    The aligners in depth -- matching strategies, transformation models, and
    how to control which stars drive the fit.

:doc:`motion_models`
    The per-star motion models, their equations, and how FlyStar picks one
    per star.

:doc:`uncertainties`
    How weights, parameter errors and ``absolute_sigma`` are actually
    computed, with the formulae.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Guide

   overview
   alignment
   motion_models
   uncertainties

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   examples/index

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
