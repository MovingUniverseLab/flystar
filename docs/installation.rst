============
Installation
============

FlyStar is not on PyPI; install it from the repository:

.. code-block:: bash

   git clone https://github.com/MovingUniverseLab/flystar.git
   cd flystar
   pip install -e .

That pulls in the whole runtime set -- numpy, scipy, astropy, matplotlib,
pandas, joblib and tqdm -- all of which the package imports at module level.
Python 3.7 or newer.

Three further packages are imported lazily, inside the functions that use them,
so install them only if you want those features: ``shapely``
(polygon-restricted initial guesses via ``starlist_vertices``), ``astroquery``
(Gaia queries in :mod:`flystar.analysis`) and ``plotly`` (interactive plots in
:mod:`flystar.plots`). They are grouped as an extra:

.. code-block:: bash

   pip install -e '.[optional]'

