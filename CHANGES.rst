=========
Changelog
=========

0.1.0 (2026-08-26)
==================

First release on PyPI.

FlyStar cross-matches and aligns stellar astrometry catalogues. It derives
the transformations between a stack of starlists and a common reference
frame, matches stars across them, and fits a motion model per star
(``Empty``, ``Fixed``, ``Linear``, ``Acceleration`` or ``Parallax``).

``MosaicSelfRef`` builds the reference frame from the starlists themselves;
``MosaicToRef`` aligns them to an external catalogue such as Gaia.

Notes for this release
----------------------

- The reference table records each star's identity in each starlist as
  ``idx_in_list``, an integer index into that starlist, rather than as a copy
  of the name. Use :func:`flystar.align.names_in_list` to recover the names.
  Tables written by pre-release versions of the code carry a string
  ``name_in_list`` column instead; ``flystar.plots`` and ``flystar.analysis``
  read either.
- ``jplephem`` is a hard requirement, not an extra: astropy uses it whenever a
  solar-system ephemeris is evaluated, which the ``Parallax`` motion model
  does.
- ``shapely``, ``astroquery`` and ``plotly`` are imported lazily by the
  features that need them (polygon footprints for the initial guess, Gaia and
  JPL Horizons queries, and interactive diagnostic plots respectively). Install
  them with ``pip install flystar[optional]``.
- Tests are not included in the distributions; run them from a checkout of the
  repository.
