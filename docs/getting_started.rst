===============
Getting started
===============

Installation
============

FlyStar is not on PyPI; install it from the repository:

.. code-block:: bash

   git clone https://github.com/MovingUniverseLab/flystar.git
   cd flystar
   pip install -e .

Be aware that ``setup.cfg`` currently declares only ``astropy`` as a
requirement, while the package imports rather more at module level. Install
these too, or the first ``import flystar`` will fail:

.. code-block:: bash

   pip install numpy scipy astropy matplotlib pandas joblib tqdm

Three further packages are imported lazily, so you only need them for the
features that use them: ``shapely`` (polygon-restricted initial guesses via
``starlist_vertices``), ``astroquery`` (Gaia queries in
:mod:`flystar.analysis`), and ``plotly`` (interactive plots in
:mod:`flystar.plots`). Python 3.7 or newer.

Your first alignment
====================

The shortest thing that works. Build one
:class:`~flystar.starlists.StarList` per epoch, hand them to an aligner, read
the answer off the :class:`~flystar.startables.StarTable` it produces:

.. code-block:: python

   from flystar import align, starlists, transforms

   lists = []
   for path, year in [('epoch1.lis', 2015.5), ('epoch2.lis', 2017.4)]:
       sl = starlists.StarList.from_lis_file(path)
       sl.meta['list_time'] = year          # decimal year, UTC
       lists.append(sl)

   msc = align.MosaicSelfRef(lists, iters=3,
                             dr_tol=[1.0, 0.5, 0.3],
                             dm_tol=[2.0, 1.0, 1.0],
                             trans_class=transforms.PolyTransform,
                             trans_args=[{'order': 1}, {'order': 2}, {'order': 2}],
                             motion_models=['Linear'])
   msc.fit()

   ref = msc.ref_table
   ref['x0'], ref['vx'], ref['vx_err'], ref['motion_model_used']

Note ``motion_models=['Linear']``: the default is ``['Empty', 'Fixed']``, which
fits no proper motions at all.

For a version you can actually run -- synthetic data built with numpy, aligned,
and checked against the truth, with plots -- see
:doc:`examples/alignment_example`.

Where to go next
================

:doc:`overview`
    The data model in full, and the column-naming conventions the code
    dispatches on.

:doc:`motion_models`
    The per-star motion models and their equations, and how one is chosen for
    each star -- read this before the aligner's ``motion_models`` argument.

:doc:`alignment`
    The aligners in depth, with every constructor argument described --
    matching strategies, transformation models, and how to control which stars
    drive the fit.

