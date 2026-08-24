=============
Uncertainties
=============

An astrometric solution is only as useful as its error bars, so it is worth
being explicit about what FlyStar computes and what convention it follows.
Every fit and every average in the package obeys the same two rules:

1. Weights come from the input uncertainties through one of two **weighting
   schemes**, selected by ``weighting``.
2. Reported parameter errors follow :func:`scipy.optimize.curve_fit`'s
   **``absolute_sigma``** convention, with the same meaning everywhere the flag
   appears.

Weighting schemes
=================

``weighting`` decides how a per-epoch uncertainty becomes a fit weight. Both
schemes go through :func:`~flystar.motion_model.sigma_from_error` and then
:func:`~flystar.motion_model.weight_from_sigma`, which computes
:math:`w = 1/\sigma^2`:

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - ``weighting``
     - :math:`\sigma_i` used
     - Resulting weight
   * - ``'var'`` (default)
     - :math:`\sigma_i = |\sigma_{x,i}|`
     - :math:`w_i = 1/\sigma_{x,i}^2` -- true inverse-variance weighting.
   * - ``'std'``
     - :math:`\sigma_i = \sqrt{|\sigma_{x,i}|}`
     - :math:`w_i = 1/|\sigma_{x,i}|` -- standard-error weighting, a gentler
       down-weighting of poorly measured epochs.

Use ``'var'`` unless you have a specific reason: it is the statistically
correct choice when your input errors are trustworthy, and it minimises the
propagated uncertainty on the result.

Unusable uncertainties get weight zero, not a bad weight
--------------------------------------------------------

A naive :math:`1/\sigma^2` turns a missing or pathological uncertainty into an
infinite or NaN weight, which corrupts the whole sum rather than excluding one
point. :func:`~flystar.motion_model.weight_from_sigma` instead assigns
**exactly zero** whenever :math:`\sigma` is NaN, infinite, exactly zero, or so
small that squaring it underflows -- and to any epoch explicitly marked
invalid. Such an epoch drops cleanly out of both the fit and the
:math:`\chi^2`.

If *every* epoch of a star ends up with weight zero, there is no weighted mean
to report. FlyStar does not invent one: the value falls back to the unweighted
mean where one is defined, and the uncertainty is reported as
:math:`\infty` -- never a fabricated finite number. Per-star error columns are
filled with ``inf``, not ``nan``, precisely so that "we don't know" is
distinguishable from "no data" and can never be mistaken for precision.

The ``absolute_sigma`` convention
=================================

This flag appears on :meth:`~flystar.motion_model.MotionModel.fit`,
:meth:`~flystar.startables.StarTable.fit_motion_models` and
:meth:`~flystar.startables.StarTable.combine_lists`, and means the same thing
in all three.

Let :math:`\hat{\sigma}_p` be the formal error on parameter :math:`p` from the
weighted normal equations, :math:`\chi^2` the weighted sum of squared residuals
of the fit, and :math:`\nu` its degrees of freedom.

``absolute_sigma=True`` (default)
    Your input errors are taken at face value and propagated as given:

    .. math:: \sigma_p = \hat{\sigma}_p

``absolute_sigma=False``
    Errors are rescaled by the square root of the reduced :math:`\chi^2`:

    .. math:: \sigma_p = \hat{\sigma}_p \sqrt{\chi^2 / \nu}

    Only the *relative* magnitudes of the input errors then matter, and the
    result reflects the epochs' own disagreement.

Which to use: ``True`` trusts your per-point errors; ``False`` is the more
honest choice when the input errors are known to be systematically
underestimated, since the scatter of the data then sets the scale. When
:math:`\nu \le 0` there is nothing to rescale by, and the error is reported as
:math:`\infty` rather than as a 0/0 NaN.

Motion-model fits
=================

Every model's ``run_fit`` is a closed-form weighted least-squares solve. Two
worked cases follow; ``Acceleration`` and ``Parallax`` extend the same normal
equations to more basis functions.

``Fixed`` -- the weighted mean
------------------------------

With weights :math:`w_i` over a star's valid epochs:

.. math::

   x_0 = \frac{\sum_i w_i x_i}{\sum_i w_i}
   \qquad
   \hat{\sigma}_{x_0} = \frac{1}{\sqrt{\sum_i w_i}}
   \qquad
   \chi^2 = \sum_i w_i (x_i - x_0)^2

with :math:`\nu = n_\mathrm{valid} - 1`.

``Linear`` -- weighted straight-line fit
----------------------------------------

Solved as :math:`2\times2` normal equations in the basis
:math:`[\Delta t,\, 1]`. Writing the weighted sums

.. math::

   S_w = \sum_i w_i, \quad
   S_{w\Delta t} = \sum_i w_i \Delta t_i, \quad
   S_{w\Delta t^2} = \sum_i w_i \Delta t_i^2, \\
   S_{wx} = \sum_i w_i x_i, \quad
   S_{w\Delta t x} = \sum_i w_i \Delta t_i x_i, \quad
   D = S_{w\Delta t^2} S_w - S_{w\Delta t}^2

the solution and its formal errors are

.. math::

   v_x &= \frac{S_w S_{w\Delta t x} - S_{w\Delta t} S_{wx}}{D}
   &\hat{\sigma}_{v_x} &= \sqrt{S_w / D} \\
   x_0 &= \frac{S_{w\Delta t^2} S_{wx} - S_{w\Delta t} S_{w\Delta t x}}{D}
   &\hat{\sigma}_{x_0} &= \sqrt{S_{w\Delta t^2} / D}

with

.. math::

   \chi^2 = \sum_i w_i \bigl(x_i - (x_0 + v_x \Delta t_i)\bigr)^2 ,
   \qquad \nu = n_\mathrm{valid} - n_\mathrm{params}

The determinant :math:`D` vanishes when the fit is degenerate -- every valid
epoch at the same time, say. FlyStar detects this against a scaled tolerance
and reports ``fill_value`` for the parameters with :math:`\infty` errors,
rather than the arbitrary minimum-norm answer a pseudo-inverse would return.
Only a particular combination of :math:`x_0` and :math:`v_x` is constrained in
that case, so no single split is meaningful.

Averaging across epochs
=======================

:meth:`~flystar.startables.StarTable.combine_lists` collapses a per-list column
into a per-star one (``x`` → ``x0``, ``x0_err``). The reported uncertainty is
always the uncertainty **of the mean**, never the scatter of the points, in all
four branches below.

With :math:`w_i = 1/\sigma_i^2`, :math:`S = \sum_i (x_i - \bar{x})^2`,
:math:`\chi^2 = \sum_i w_i (x_i - \bar{x})^2`, and
:math:`\nu = n_\mathrm{valid} - 1` -- one parameter, the mean, estimated from
the data:

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Branch
     - ``absolute_sigma=True``
     - ``absolute_sigma=False``
   * - **weighted**
       (``weights_col`` given)
     - :math:`\sqrt{1 / \sum_i w_i}`
     - :math:`\sqrt{\dfrac{\chi^2}{\nu \sum_i w_i}}`
   * - **unweighted**
     - :math:`\sqrt{\dfrac{S}{n_\mathrm{valid}\,\nu}}`
     - :math:`\sqrt{\dfrac{S}{n_\mathrm{valid}\,\nu}}`

``absolute_sigma`` deliberately does not reach the unweighted branch: it chooses
between propagating input errors and rescaling by observed scatter, and the
unweighted branch runs precisely when there are no input errors to propagate.
The scatter is already all the information available.

Outlier rejection happens before any of this, via sigma clipping controlled by
the ``sigma`` argument, so a single bad epoch does not drag the mean.

Empirical errors by bootstrap
=============================

The formulae above are analytic and assume the model is right. To get errors
that make no such assumption, resample:

.. code-block:: python

   table.fit_motion_models(motion_models=['Linear'], bootstrap=100, seed=42)

Each star's epochs are drawn with replacement ``bootstrap`` times, the model is
refit on each draw, and the spread of the resulting parameters becomes the
reported error. Pass ``seed`` for reproducibility. This is the one fitting path
not vectorized across stars -- see :doc:`motion_models` on ``processes`` and
``mp_star_threshold``.

At the alignment level, :meth:`~flystar.align.MosaicToRef.calc_bootstrap_errors`
does the analogous thing one level up: it resamples the *reference stars*,
re-derives the transformations, and takes the scatter of the transformed
positions as the transformation error -- capturing uncertainty in the frame
itself, which the per-star fit above cannot see.
