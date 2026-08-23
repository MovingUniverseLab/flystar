=====================
FlyStar Documentation
=====================

FlyStar aligns stellar astrometry. Given many star lists of the same field --
different epochs, different instruments, different distortion solutions -- it
matches the stars across lists, solves for the coordinate transformation
between each list and a common reference frame, and fits each star's motion
through time.

.. admonition:: Which version is this?
   :class: important

   This documentation is built from the ``mm_rework_lingfeng`` branch and
   describes **that** branch's API, which differs substantially from ``main``.
   The motion-model machinery described in :doc:`motion_models` does not exist
   on ``main`` at all. Every page here, including the auto-generated
   :doc:`api/flystar/index`, is generated from the source tree of this branch.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   overview
   motion_models
   alignment

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/flystar/index

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
