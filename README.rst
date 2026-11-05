FlyStar
-------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

FlyStar aligns many star lists into one common frame and fits each star's motion
there, choosing per star from a set of motion models. It cross-matches the stars
across every list, solves for the transformation that carries each list into a
common reference frame, and fits positions, proper motions and where wanted
parallaxes, with uncertainties.

Documentation
-------------

Two builds of the same source, both from the ``dev`` branch:

* **Read the Docs** -- https://flystar.readthedocs.io/en/dev/
  Rebuilt on every push, and keeps a version per branch under ``/en/<branch>/``.
  Note that ``/en/latest/`` does not track ``dev`` and does not have these
  pages.
* **GitHub Pages** -- https://wei-lingfeng.github.io/flystar/
  A mirror without ads, currently published from Lingfeng Wei's fork. One
  version only. This repository's ``.github/workflows/docs.yml`` builds the
  same docs on each push to ``dev`` that changes ``docs/`` or ``flystar/``, and
  will publish them to https://movinguniverselab.github.io/flystar/ once GitHub
  Pages is enabled for the repository.

Start with "Getting started" in the sidebar: Installation, then First
alignment, which builds a synthetic four-epoch data set, aligns it, and checks
the recovered positions and proper motions against the truth that went in.

.. warning::

   These describe the ``dev`` branch, whose motion-model framework does not
   exist on ``main``. The API reference on both sites is generated from this
   branch's source.

Installation
------------

Not on PyPI; install from the repository::

    git clone https://github.com/MovingUniverseLab/flystar.git
    cd flystar
    pip install -e .

That pulls in the runtime dependencies listed in ``pyproject.toml``: numpy,
scipy, astropy, matplotlib, pandas, joblib, tqdm, h5py and jplephem (which
astropy needs for the JPL ephemeris behind the Parallax model). Python 3.9 or
newer. Three further
packages are imported lazily and only needed for the features that use them --
``shapely``, ``astroquery`` and ``plotly`` -- grouped as an extra::

    pip install -e '.[optional]'


License
-------

This project is Copyright (c) Jessica Lu and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.


Contributing
------------

We love contributions! FlyStar is open source,
built on open source, and we'd love to have you hang out in our community.
