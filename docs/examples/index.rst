========
Examples
========

These notebooks are rendered from their stored outputs and are not executed at
documentation build time, so what you see is the output as it was when the
notebook was last run.

.. toctree::
   :maxdepth: 1

   motion_model_example
   flystar_mosaic_to_gaia/gaia_flystar_example

:doc:`motion_model_example`
    Fitting each of the motion models, how ``motion_model_input`` and
    ``motion_model_used`` interact, and predicting positions at new epochs.

:doc:`flystar_mosaic_to_gaia/gaia_flystar_example`
    Querying Gaia for a field, preparing the catalog as a FlyStar reference
    list, and aligning NIRC2 epochs to it with
    :class:`~flystar.align.MosaicToRef`.
