============
Installation
============

FlyStar is not on PyPI; install it from the repository:

.. code-block:: bash

   git clone https://github.com/MovingUniverseLab/flystar.git
   cd flystar
   pip install -e .

That pulls in the whole runtime set listed in ``pyproject.toml`` -- numpy,
scipy, astropy, matplotlib, pandas, joblib, tqdm, h5py and jplephem (which
astropy needs for the JPL ephemeris behind the Parallax model). Python 3.9 or
newer.

Three further packages are imported lazily, inside the functions that use them,
so install them only if you want those features: ``shapely``
(polygon-restricted initial guesses via ``starlist_vertices``), ``astroquery``
(Gaia queries in :mod:`flystar.analysis`) and ``plotly`` (interactive plots in
:mod:`flystar.plots`). They are grouped as an extra:

.. code-block:: bash

   pip install -e '.[optional]'

