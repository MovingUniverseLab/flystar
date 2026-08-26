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

Two builds of the same source, both from the ``mm_rework_lingfeng`` branch:

* **Read the Docs** -- https://flystar.readthedocs.io/en/mm_rework_lingfeng/
  Rebuilt on every push, and keeps a version per branch under ``/en/<branch>/``.
  Note that ``/en/latest/`` tracks the default branch, which does not have these
  pages.
* **GitHub Pages** -- https://wei-lingfeng.github.io/flystar/
  An ad-free mirror of the same build. One version only: whatever
  ``mm_rework_lingfeng`` last published.

Start on either front page: it installs FlyStar and then builds a synthetic
four-epoch data set, aligns it, and checks the recovered positions and proper
motions against the truth that went in.

.. warning::

   These describe the ``mm_rework_lingfeng`` branch, whose motion-model
   framework does not exist on ``main``. The API reference on both sites is
   generated from this branch's source.

Installation
------------

Not on PyPI; install from the repository::

    git clone https://github.com/MovingUniverseLab/flystar.git
    cd flystar
    pip install -e .

That pulls in numpy, scipy, astropy, matplotlib, pandas, joblib and tqdm, all of
which the package imports at module level. Python 3.7 or newer. Three further
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

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
FlyStar based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
