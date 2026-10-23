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

https://flystar.readthedocs.io

Start on the front page: it installs FlyStar and then builds a synthetic
four-epoch data set, aligns it, and checks the recovered positions and proper
motions against the truth that went in.

Installation
------------

From PyPI::

    pip install flystar

Python 3.9 or newer. That pulls in numpy, scipy, astropy, matplotlib, pandas,
joblib, tqdm, h5py and jplephem. Three further packages are imported lazily and
only needed for the features that use them -- ``shapely``, ``astroquery`` and
``plotly`` -- grouped as an extra::

    pip install 'flystar[optional]'

To work on the code, install from the repository instead::

    git clone https://github.com/MovingUniverseLab/flystar.git
    cd flystar
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
