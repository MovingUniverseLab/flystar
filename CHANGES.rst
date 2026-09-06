=========
Changelog
=========

Unreleased
==========

API changes to ``MosaicSelfRef`` and ``MosaicToRef``
----------------------------------------------------

- ``dr_tol`` is now a required argument, given before the optional ones.
  Matching is a radius search, so there is no meaningful default: the old
  ``dr_tol=[1.0]`` silently applied a one-unit radius in whatever units the
  reference frame happened to use.
- The ``iters`` argument is gone. It had never been read -- the iteration
  count has always come from the length of ``dr_tol`` -- so calls that passed
  a disagreeing ``iters`` were already running the ``dr_tol`` schedule. Drop
  it from the call.
- The number of iterations is now the length of the longest schedule given:
  ``dr_tol``, ``dm_tol``, ``outlier_tol`` or ``trans_args``. Any of them may
  be a single value, broadcast to that length, so ``dr_tol=0.5,
  dm_tol=[1.0, 0.5]`` is two passes at a constant radius, and all single
  values is a single pass. Two schedules of differing length remain an error.
  ``mag_lim`` does not take part -- its ``[min, max]`` form is a pair, not a
  schedule -- and is still checked against the resulting count.
- ``dm_tol`` now defaults to ``None``, which places no magnitude cut on the
  match. The old default of ``[1.0]`` rejected pairs more than one magnitude
  apart, which is wrong across filters and arbitrary within one.

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
