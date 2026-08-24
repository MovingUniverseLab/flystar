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

A worked alignment, from scratch
================================

The rest of this page builds a synthetic data set with numpy, aligns it, and
checks the answer against the truth we put in. Everything below runs as-is --
no data files needed -- so you can paste it into a session and watch it work.

The problem we are setting up
-----------------------------

250 stars in a 1000-pixel field, observed at four epochs two years apart. Each
epoch is deliberately given its own coordinate system -- a shift of up to 40
pixels, a rotation of up to 0.7 degrees about the field centre -- because that
is the situation FlyStar exists to resolve. On top of that, each star has a
small proper motion, each measurement has 0.05 pixel noise, and roughly 12% of
stars go undetected in any given epoch.

.. code-block:: python

   import numpy as np
   from flystar import align, starlists, transforms

   rng = np.random.default_rng(42)
   N, YEARS, ERR = 250, np.array([2015.0, 2017.0, 2019.0, 2021.0]), 0.05
   t0 = YEARS.mean()

   # Truth: positions at t0, proper motions, magnitudes.
   x0 = rng.uniform(0, 1000, N); y0 = rng.uniform(0, 1000, N)
   vx = rng.normal(0, 0.3, N);   vy = rng.normal(0, 0.3, N)   # pixels / year
   mag = rng.uniform(12, 19, N)
   names = np.array([f'S{j:03d}' for j in range(N)])

   # Each epoch gets its own frame: a shift plus a small rotation.
   shift_x = np.array([0., 18., -25., 40.])
   shift_y = np.array([0., -12., 30., -20.])
   angle = np.deg2rad(np.array([0., 0.3, -0.5, 0.7]))

.. admonition:: Why the stars are called ``S000`` and not ``star_000``
   :class: note

   ``init_guess_mode='name'`` leaves out every star whose name contains
   ``ignore_contains``, which defaults to ``'star'``. That default is
   deliberate: auto-detected sources are conventionally labelled ``star_1``,
   ``star_2``, ... per epoch, and those indices are per-list detection numbers
   rather than stable identities, so matching on them would pair unrelated
   stars.

   If your names *are* stable across epochs -- a cross-matched catalog, or
   synthetic data like this -- pass ``ignore_contains=None`` and every name is
   used:

   .. code-block:: python

      align.MosaicSelfRef(lists, ..., init_guess_mode='name',
                          ignore_contains=None)

   Either way it is no longer silent: the filter warns when it excludes stars,
   and says so again in the error if nothing is left to match on. Note that
   ``''`` is rejected rather than meaning "off" -- every name contains the
   empty string -- so ``None`` is the off switch.

Building one StarList per epoch
-------------------------------

A :class:`~flystar.starlists.StarList` is one epoch's detection list. Note
``meta['list_time']``: that is how the motion fit learns when each list was
taken, and it is a **decimal year in UTC** (see :doc:`overview`).

.. code-block:: python

   lists = []
   for i, yr in enumerate(YEARS):
       dt = yr - t0

       # Where the stars really are on the sky at this epoch.
       xt, yt = x0 + vx * dt, y0 + vy * dt

       # Now push them into this epoch's own frame: rotate about the centre,
       # then shift. This is the distortion the alignment has to undo.
       xc, yc = xt - 500., yt - 500.
       c, s = np.cos(angle[i]), np.sin(angle[i])
       xo = (c * xc - s * yc) + 500. + shift_x[i] + rng.normal(0, ERR, N)
       yo = (s * xc + c * yc) + 500. + shift_y[i] + rng.normal(0, ERR, N)

       seen = rng.random(N) > 0.12          # ~12% non-detections this epoch

       sl = starlists.StarList(
           name=names[seen], x=xo[seen], y=yo[seen], m=mag[seen],
           xe=np.full(seen.sum(), ERR), ye=np.full(seen.sum(), ERR),
           me=np.full(seen.sum(), 0.05),
       )
       sl.meta['list_time'] = yr
       lists.append(sl)

Aligning them
-------------

``dr_tol``, ``dm_tol`` and ``trans_args`` take **one entry per iteration**.
That is how the solution converges: the first pass matches loosely, because
the frames are still tens of pixels apart, and each pass afterwards tightens
the tolerance now that the transformation is better known.

.. code-block:: python

   msc = align.MosaicSelfRef(
       lists,
       iters=3,
       dr_tol=[60., 10., 5.],               # match radius, pixels, per iteration
       dm_tol=[1., 1., 1.],                 # match magnitude tolerance
       trans_class=transforms.PolyTransform,
       trans_args=[{'order': 1}] * 3,       # order 1 = shift + rotation + scale
       motion_models=['Linear'],            # fit x0, vx, y0, vy per star
       init_guess_mode='name',              # our stars carry consistent names
   )
   msc.fit()

   ref = msc.ref_table
   print(f"{len(ref)} stars; {int((ref['n_detect'] == 4).sum())} seen in all four epochs")

.. code-block:: text

   254 stars; 144 seen in all four epochs

254 rows against the 250 we injected, and 144 stars detected in every epoch --
close to the :math:`250 \times 0.88^4 \approx 150` you would expect from a 12%
per-epoch drop-out. The handful of extra rows are stars matched in only one or
two epochs.

Did it work?
------------

The point of the alignment is that every epoch ends up in one frame. On the
left, each star is measured at four visibly different places, because each
epoch has its own coordinate system; grey lines join the four measurements of
a single star. On the right, the same stars after transformation.

.. image:: _static/align_before_after.png
   :alt: Star positions before and after alignment
   :align: center

The transformed positions live in ``ref_table['x']`` and ``['y']``, which are
2D, ``(N_stars, N_lists)`` -- one column per epoch. The per-star averages are
``x0``/``y0``.

Residuals are the real test. Subtracting each star's own fitted motion from its
measured positions should leave nothing but noise:

.. code-block:: python

   good = np.asarray(ref['n_detect']) == len(YEARS)
   xm, ym, _, _ = ref.infer_positions(YEARS)          # model positions
   dx = (np.asarray(ref['x']) - xm)[good] * 1000       # milli-pixels
   dy = (np.asarray(ref['y']) - ym)[good] * 1000
   print(f"residual scatter: {np.nanstd(dx):.1f} / {np.nanstd(dy):.1f} mpix")

.. code-block:: text

   residual scatter: 36.7 / 38.0 mpix

.. image:: _static/align_residuals.png
   :alt: Post-alignment residual distribution
   :align: center

We injected 50 milli-pixels of noise and recovered 37. That is not the
alignment beating the noise -- it is the expected effect of fitting two
parameters per coordinate to four epochs, which absorbs part of the scatter:
:math:`50 \times \sqrt{1 - 2/4} = 35`. Getting 37 is the sign the fit is
behaving.

Finally, the proper motions. We never told FlyStar what they were, and the
frames were rotating underneath them, so recovering them is the strongest check
that the alignment is right:

.. code-block:: python

   name_to_i = {n: j for j, n in enumerate(names)}
   idx = np.array([name_to_i.get(n, -1) for n in np.asarray(ref['name']).astype(str)])
   ok = good & (idx >= 0)
   print(f"vx recovered to {np.std(np.asarray(ref['vx'])[ok] - vx[idx[ok]]) * 1000:.1f} mpix/yr")

.. code-block:: text

   vx recovered to 19.3 mpix/yr

.. image:: _static/align_proper_motion.png
   :alt: Recovered versus true proper motion
   :align: center

Proper motions recovered to about 20 milli-pixels per year against a 0.05 pixel
per-epoch measurement error over a six-year baseline -- so the alignment has
not absorbed the stellar motion into the frame solution, which is the failure
mode that matters here.

Where to go next
================

:doc:`overview`
    The data model in full, and the column-naming conventions the code
    dispatches on.

:doc:`alignment`
    The aligners in depth -- matching strategies, transformation models, and
    how to control which stars drive the fit. If you do not have consistent
    star names, this is where ``init_guess_mode='miracle'`` is explained.

:doc:`motion_models`
    The per-star motion models and their equations.

:doc:`uncertainties`
    How weights and parameter errors are computed, with the formulae.
