"""Newtonian orbit solver, orbits.dat reader, and per-star orbit fits.

Each test states the physical or numerical property it is checking,
where the reference numbers come from, and why the tolerance is that
tight. The executable steps are unchanged from the implementation of
the orbit plan.
"""

import pathlib

import numpy as np
import pytest
from astropy.table import Table

from flystar.motion_model import Orbit
from flystar.orbits import (
    _MAS_PER_ARCSEC,
    attach_orbits,
    eccen_anomaly,
    kep2xyz,
    read_orbits_dat,
    semimajor_axis_mas,
)
from flystar.startables import StarTable


# orbits.dat v2.0.2: 32 Galactic Center stars, nine whitespace fields,
# no header. The third field is the printed semi-major axis in mas.
DATA = pathlib.Path(__file__).resolve().parent / 'test_data' / 'orbits_v2.0.2.dat'
# East/north snapshot transcribed from gcwork kep2xyz, not from FlyStar.
# Mass and distance in that file are the gcwork pair, not the defaults.
GCWORK = (
    pathlib.Path(__file__).resolve().parent
    / 'test_data'
    / 'gcwork_kep2xyz_m4.07e6_r7960.1.csv'
)
# Order matches Orbit.fit_param_names. Indexes below assume this order.
_NAMES = ('orb_P', 'orb_t0', 'orb_e', 'orb_i', 'orb_Omega', 'orb_omega')


def _ang(angle, reference):
    """Absolute difference of two degrees, folded into [0, 180].

    Parameters
    ----------
    angle, reference : float
        Angles in degrees. The branch cut is arbitrary.

    Returns
    -------
    delta : float
        Smallest absolute separation, in degrees, on [0, 180].

    Notes
    -----
    Node and periapse are periodic. A raw subtraction would call a
    359 degree error a failure when the angles are 1 degree apart.
    Folding into [0, 180] makes the later 5 degree limits meaningful.
    """
    return abs((float(angle) - float(reference) + 180.0) % 360.0 - 180.0)


def _branch_near_seed(fitted, truth, seed):
    """Angle errors after folding onto the branch nearer the seed.

    Parameters
    ----------
    fitted, truth, seed : array-like, shape (6,)
        Elements in ``Orbit.fit_param_names`` order.

    Returns
    -------
    d_omega, d_arg : float
        Absolute degree errors in ``orb_Omega`` and ``orb_omega``.

    Notes
    -----
    Sky positions are unchanged under ``(Omega + 180, omega + 180)``.
    The line-of-sight velocity flips, and this fit has no radial
    velocities, so both branches are acceptable. The solver keeps the
    branch closer to the seed. Scoring the angles against the truth
    without that fold would fail a correct fit that landed on the twin.
    """
    # Indexes 4 and 5 are orb_Omega and orb_omega.
    fit_o, fit_w = fitted[4], fitted[5]
    true_o, true_w = truth[4], truth[5]
    seed_o, seed_w = seed[4], seed[5]
    # Distance of each branch from the seed, in degrees, both angles.
    stay = _ang(fit_o, seed_o) + _ang(fit_w, seed_w)
    flip = _ang(fit_o + 180.0, seed_o) + _ang(fit_w + 180.0, seed_w)
    if stay <= flip:
        return _ang(fit_o, true_o), _ang(fit_w, true_w)
    return _ang(fit_o + 180.0, true_o), _ang(fit_w + 180.0, true_w)


def _orbit_table(epochs, elements, x, y, sigma):
    """One-star table whose ``orb_*`` columns are the fit seed.

    Parameters
    ----------
    epochs : ndarray, shape (n_epochs,)
        Decimal years.
    elements : array-like, shape (6,)
        Seed written into the six ``orb_*`` columns.
    x, y : ndarray, shape (n_epochs,)
        Measured positions, arcseconds, already in the FlyStar frame.
    sigma : float
        Position uncertainty written into ``xe`` and ``ye``, arcseconds.

    Returns
    -------
    table : StarTable
        One row, ``motion_model_input='Orbit'``.

    Notes
    -----
    The linear columns are filled because the table fitter and
    ``infer_positions`` expect them even for an orbit star. They are
    not the orbit. ``motion_model_input`` is what selects ``Orbit``.
    """
    epochs = np.asarray(epochs, dtype=float)
    n_ep = epochs.size
    # One row, one uncertainty at every epoch. ye is a copy so a later
    # in-place edit of xe cannot alias ye.
    sigma = np.full((1, n_ep), float(sigma))
    tab = StarTable(
        name=np.array(['star']),
        x=np.asarray(x, dtype=float)[np.newaxis, :],
        y=np.asarray(y, dtype=float)[np.newaxis, :],
        m=np.full((1, n_ep), 15.0),
        xe=sigma,
        ye=sigma.copy(),
        me=np.full((1, n_ep), 0.01),
        t=epochs[np.newaxis, :],
    )
    # The seed lives in the element columns. A failed fit must return
    # these same numbers, so the test can compare them bitwise.
    for name, value in zip(_NAMES, elements):
        tab[name] = np.array([float(value)])
    tab['motion_model_input'] = np.array(['Orbit'], dtype='U12')
    # Placeholder linear solution. Orbit prediction does not read it.
    tab['x0'] = np.array([0.0])
    tab['y0'] = np.array([0.0])
    tab['vx'] = np.array([0.0])
    tab['vy'] = np.array([0.0])
    tab['t0'] = np.array([2000.0])
    tab['x0_err'] = np.array([0.001])
    tab['y0_err'] = np.array([0.001])
    tab['motion_model_used'] = np.array(['Orbit'], dtype='U12')
    return tab


def _parse_file_a(path):
    """Name, period, and the printed ``a`` field. ``a`` is not stored.

    Parameters
    ----------
    path : path-like
        Nine-field orbits.dat file. No header. Field order is fixed.

    Returns
    -------
    names : list of str
    period, a_mas : ndarray, shape (n_stars,)
        Period in years and the file's semi-major axis in mas.

    Notes
    -----
    ``read_orbits_dat`` drops the third field after checking that the
    line has nine values. This helper reads that field so the test can
    compare it with ``semimajor_axis_mas`` without the catalog storing
    it. The on-disk layout is unchanged.
    """
    names = []
    period = []
    a_mas = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            # fields[1] is P in years. fields[2] is a in milliarcseconds.
            names.append(fields[0])
            period.append(float(fields[1]))
            a_mas.append(float(fields[2]))
    return names, np.asarray(period), np.asarray(a_mas)


def test_kep2xyz_matches_orbit_model_on_a_grid():
    """Orbit.model is kep2xyz with x = -east and y = +north.

    Notes
    -----
    This checks the frame convention, not a fit. ``kep2xyz`` returns
    east, north, and the line of sight in arcseconds, black hole at
    the origin. ``Orbit.model`` must negate east and keep north.
    Three orbits cover a circular face-on case, an S0-2-like
    eccentricity near 0.89, and an S0-16-like eccentricity near 0.98,
    which is the near-parabolic end of this catalog. The year grid
    runs from 1995 to 2035 so the short period is covered more than
    once and the long period is sampled well away from periapse.
    The absolute tolerance is 1e-12 arcsec. That is a few times
    floating-point roundoff on a coordinate of about an arcsecond.
    Relative tolerance is zero so a coordinate that is exactly zero,
    such as east for the face-on circular orbit, is still checked.
    """
    model = Orbit()
    # 21 epochs, inclusive, spanning 40 years.
    epochs = np.linspace(1995.0, 2035.0, 21)
    # Each row is P, t0, e, i, Omega, omega. Defaults: 4e6 Msun, 8000 pc.
    cases = [
        np.array([16.0, 2010.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([16.0, 2002.31, 0.8891, 134.7, 48.4, 246.7]),
        np.array([54.709, 2000.156, 0.9772, 99.4, 227.0, 334.0]),
    ]
    for elements in cases:
        # Column 0 is east, column 1 is north. Acceleration is unused.
        r_au, _, _ = kep2xyz(epochs, *elements)
        x, y = model.model(epochs, elements)
        # FlyStar: x is west, so it is minus east. y is north.
        np.testing.assert_allclose(x, -r_au[:, 0], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(y, r_au[:, 1], rtol=0.0, atol=1e-12)

    # Face-on circular orbit at periapse. Inclination, node, and
    # argument of periapse are all zero, so the star sits on the
    # positive north axis at a distance of one semi-major axis.
    circular = cases[0]
    # semimajor_axis_mas returns milliarcseconds. Divide by the
    # import-time mas-per-arcsec factor to get arcseconds.
    a_as = semimajor_axis_mas(circular[0], 4.0e6, 8.0e3) / _MAS_PER_ARCSEC
    # Evaluate at t0, which is periapse for this row.
    x0, y0 = model.model(circular[1], circular)
    np.testing.assert_allclose(x0, 0.0, atol=1e-12)
    np.testing.assert_allclose(y0, a_as, rtol=0.0, atol=1e-12)

    return None


def test_black_hole_offset_is_added_in_the_flystar_frame():
    """Black-hole offset and motion are added after the sign flip.

    Notes
    -----
    The orbital piece is still east/north from ``kep2xyz``. The model
    then writes ``x = x_bh + vx_bh * (t - t_bh) - east`` and
    ``y = y_bh + vy_bh * (t - t_bh) + north``. An offset must not
    reverse those signs. The orbit is face-on and circular so the
    stellar offset is only the semi-major axis along north at
    periapse, and the epoch is exactly 10 years after ``t_bh``.
    No explicit tolerance is passed: NumPy's default relative
    tolerance of 1e-7 is enough for a closed-form shift of order
    0.01 arcsec, and the absolute default of 0 still checks a zero.
    """
    # P, t0, e, i, Omega, omega. Periapse is the same epoch as t_bh.
    elements = np.array([16.0, 2000.0, 0.0, 0.0, 0.0, 0.0])
    epoch = np.array([2010.0])
    # East/north of the star relative to the black hole, arcseconds.
    r_au, _, _ = kep2xyz(epoch, *elements)
    x, y = Orbit().model(
        epoch, elements,
        fixed_params_dict={
            'x_bh': 0.01, 'y_bh': -0.02,
            'vx_bh': 0.001, 'vy_bh': -0.002, 't_bh': 2000.0,
        },
    )
    # Ten years of black-hole motion, then the FlyStar sign on the orbit.
    dt = 10.0
    np.testing.assert_allclose(x, 0.01 + 0.001 * dt - r_au[0, 0])
    np.testing.assert_allclose(y, -0.02 - 0.002 * dt + r_au[0, 1])

    return None


def test_fixed_star_position_errors_are_zero():
    """A star with no written covariance has xe = ye = 0.

    Notes
    -----
    Asking for errors by passing ``fit_param_errs`` must not invent a
    position error from the diagonal, and it must not return infinity.
    Infinity is reserved for a fit that ran and left no finite
    covariance. There is no ``pos_err`` switch. The two epochs force
    a length-2 vector, which confirms the time axis is preserved.
    The zeros are exact assignments, so the comparison is exact.
    """
    elements = np.array([16.0, 2010.0, 0.2, 40.0, 10.0, 20.0])
    # Non-finite diagonals. They must not be propagated.
    errs = np.full(6, np.inf)
    x, y, xe, ye = Orbit().model(
        np.array([2010.0, 2012.0]), elements, fit_param_errs=errs,
    )
    # Two epochs, one star: model flattens to shape (n_epochs,).
    assert x.shape == (2,)
    np.testing.assert_allclose(xe, 0.0)
    np.testing.assert_allclose(ye, 0.0)
    # The plan removed any pos_err knob from the optional parameters.
    assert 'pos_err' not in Orbit.optional_fixed_params

    return None


def test_mass_and_dist_defaults_and_override_order():
    """Defaults are 4e6 Msun and 8000 pc. Dict beats column beats meta.

    Notes
    -----
    Those defaults are the pair that reproduces the printed ``a``
    column of ``orbits.dat``. The lookup order matches ``Parallax``:
    an explicit ``fixed_params_dict`` wins, then a column, then
    ``table.meta``, then the class default. The star is face-on and
    circular and is evaluated at periapse, so ``y`` is the semi-major
    axis in arcseconds and a mass scale is a pure ``M**(1/3)`` scale.
    Eight times the mass doubles the axis. Twenty-seven times the
    mass triples it. Those integers are exact in the cube root, so a
    wrong exponent would miss by a large fraction, not by roundoff.
    The default path uses an absolute tolerance of 1e-12 arcsec, the
    same numerical floor as the grid test. The meta and dict paths
    use a relative tolerance of 1e-12 because the check is a ratio.
    The column path uses an absolute tolerance of 1e-10 arcsec, still
    far below a milliarcsecond, for the table round-trip.
    """
    assert Orbit.optional_fixed_params['mass'] == 4.0e6
    assert Orbit.optional_fixed_params['dist'] == 8.0e3
    elements = np.array([16.0, 2010.0, 0.0, 0.0, 0.0, 0.0])
    # y at periapse is the semi-major axis in arcseconds.
    # Divide mas by the astropy mas-per-arcsec factor.
    y_default = semimajor_axis_mas(16.0, 4.0e6, 8.0e3) / _MAS_PER_ARCSEC
    epochs = np.array([2010.0])
    n_ep = 1
    tab = StarTable(
        name=np.array(['S']),
        x=np.zeros((1, n_ep)), y=np.zeros((1, n_ep)),
        m=np.full((1, n_ep), 15.0),
        xe=np.full((1, n_ep), 0.001), ye=np.full((1, n_ep), 0.001),
        me=np.full((1, n_ep), 0.01),
        t=epochs[np.newaxis, :],
    )
    for name, value in zip(_NAMES, elements):
        tab[name] = np.array([value])
    tab['motion_model_input'] = np.array(['Orbit'], dtype='U12')
    # Present so infer_positions takes the error-returning path.
    # They are not the orbit's covariance.
    tab['x0_err'] = np.array([0.001])
    tab['y0_err'] = np.array([0.001])

    # No mass anywhere: the class default, 4e6 Msun and 8000 pc.
    _, y, _, _ = tab.infer_positions(2010.0)
    np.testing.assert_allclose(y, y_default, rtol=0.0, atol=1e-12)

    # 8x mass scales the axis by 2. Meta is the only place it lives.
    tab.meta['mass'] = 4.0e6 * 8.0
    _, y_meta, _, _ = tab.infer_positions(2010.0)
    np.testing.assert_allclose(y_meta, 2.0 * y_default, rtol=1e-12, atol=0.0)

    # A column outranks meta. Back to the default mass.
    tab['mass'] = np.array([4.0e6])
    _, y_col, _, _ = tab.infer_positions(2010.0)
    np.testing.assert_allclose(y_col, y_default, rtol=0.0, atol=1e-10)

    # fixed_params_dict outranks the column. 27x mass scales by 3.
    _, y_dict, _, _ = tab.infer_positions(
        2010.0, fixed_params_dict={'mass': 4.0e6 * 27.0},
    )
    np.testing.assert_allclose(y_dict, 3.0 * y_default, rtol=1e-12, atol=0.0)

    return None


def test_printed_a_matches_default_mass_and_distance():
    """Printed ``a`` matches semimajor_axis_mas at 4e6 Msun and 8000 pc.

    Notes
    -----
    The reference is the third field of ``orbits.dat`` v2.0.2, in
    milliarcseconds. Periods in that file are Julian years. The
    Gaussian formula applied to a Julian year overestimates ``a`` by
    about 1.3e-5, and the file was printed from that same formula, so
    the test keeps it. The absolute tolerance is 0.02 mas. The largest
    residual is S0-105, about 0.013 mas, because ``P`` is printed to
    0.01 yr. A physical ``G M`` axis would miss by up to about 0.05 mas
    and would fail this tolerance. After the reader runs, ``a`` and
    ``search`` must not be columns or metadata. The file has 32 stars.
    """
    names, period, a_file = _parse_file_a(DATA)
    # Defaults, not the gcwork pair. Units of the result are mas.
    a_model = semimajor_axis_mas(period, 4.0e6, 8.0e3)
    np.testing.assert_allclose(a_model, a_file, rtol=0.0, atol=0.02)
    assert len(names) == 32
    table = read_orbits_dat(DATA)
    # The reader parsed these fields and then dropped them.
    for banned in ('a', 'orb_a', 'search', 'orb_search'):
        assert banned not in table.colnames
        assert banned not in table.meta

    return None


def test_gcwork_east_north_with_explicit_mass_and_distance():
    """East and north match the transcribed gcwork snapshot.

    Notes
    -----
    The CSV is produced by ``generate_gcwork_kep2xyz_fixture.py`` from
    ``orbits_jlu_python_gcwork_2024-10-03`` ``kep2xyz``, with
    ``M=4.07e6`` and ``R0=7960.1``. Those are not the FlyStar defaults.
    Thirty-two stars at five epochs (periapse, half a period later,
    and the years 1995, 2010, and 2030) give 160 rows. The comparison
    is east and north in arcseconds, before the FlyStar sign flip.
    Positions do not use ``G`` or the solar mass in grams. The
    absolute tolerance is 1e-12 arcsec, which is roundoff. A silent
    switch to a physical axis would move the sky by about 1e-4
    arcsec and would fail here. Near-parabolic stars in the file are
    included, so the anomaly solver is checked at high eccentricity.
    """
    rows = np.genfromtxt(
        GCWORK, delimiter=',', names=True, dtype=None, encoding='utf-8',
    )
    # 32 stars times 5 epochs.
    assert len(rows) == 160
    for name in np.unique(rows['name']):
        sub = rows[rows['name'] == name]
        # The fixture must not have been regenerated with the defaults.
        assert sub['mass'][0] == pytest.approx(4.07e6)
        assert sub['dist'][0] == pytest.approx(7960.1)
        # Pass the gcwork pair explicitly. Column 0 is east, 1 is north.
        radius, _, _ = kep2xyz(
            sub['epoch'], sub['P'][0], sub['t0'][0], sub['e'][0],
            sub['i'][0], sub['Omega'][0], sub['omega'][0],
            mass=4.07e6, dist=7960.1,
        )
        np.testing.assert_allclose(
            radius[:, 0], sub['east_arcsec'], rtol=0.0, atol=1e-12,
        )
        np.testing.assert_allclose(
            radius[:, 1], sub['north_arcsec'], rtol=0.0, atol=1e-12,
        )
    return None


def test_read_and_attach_orbits_dat():
    """32 rows, S0-2 elements, and attach only on a shared name.

    Notes
    -----
    S0-2's elements are the printed digits of ``orbits.dat`` v2.0.2.
    ``pytest.approx`` uses a relative tolerance of 1e-6, which is
    finer than the last printed digit and coarser than a binary
    roundoff argument. ``attach_orbits`` may set
    ``motion_model_input`` only on a shared name. It must not set
    ``fit_motion``, and it must not store ``a`` or ``search``.
    A name that is only in the orbit file warns and is not added.
    An unmatched catalog row stays on its previous model and gets a
    non-finite period. S0-16's eccentricity, 0.977, checks that a
    near-parabolic row survived the parse.
    """
    table = read_orbits_dat(DATA)
    assert len(table) == 32
    row = table[table['name'] == 'S0-2'][0]
    # Printed S0-2 elements, not a fit.
    assert row['orb_P'] == pytest.approx(16.067)
    assert row['orb_t0'] == pytest.approx(2002.310)
    assert row['orb_e'] == pytest.approx(0.8891)
    assert row['orb_i'] == pytest.approx(134.7)
    assert row['orb_Omega'] == pytest.approx(48.4)
    assert row['orb_omega'] == pytest.approx(246.7)

    # Two names that exist in the file, and one that does not.
    catalog = Table()
    catalog['name'] = ['S0-2', 'S0-16', 'not-in-file']
    catalog['motion_model_input'] = np.array(
        ['Linear', 'Linear', 'Linear'], dtype='U12',
    )
    # The file has 30 further names. Those must warn, not become rows.
    with pytest.warns(UserWarning, match='not in the catalog'):
        attach_orbits(catalog, table)
    assert 'fit_motion' not in catalog.colnames
    for banned in ('a', 'orb_a', 'search', 'orb_search'):
        assert banned not in catalog.colnames
        assert banned not in catalog.meta
    # Matches become Orbit. The unknown name keeps Linear.
    assert catalog['motion_model_input'][0] == 'Orbit'
    assert catalog['motion_model_input'][1] == 'Orbit'
    assert catalog['motion_model_input'][2] == 'Linear'
    assert catalog['orb_P'][0] == pytest.approx(16.067)
    # S0-16 is the high-eccentricity row in this file.
    assert catalog['orb_e'][1] == pytest.approx(0.9772)
    # Unmatched catalog row: elements are NaN, not a fill orbit.
    assert not np.isfinite(catalog['orb_P'][2])
    return None


def _recover(truth, seed, epochs, sigma, rng):
    """Fit one injected orbit and return the table.

    Parameters
    ----------
    truth, seed : array-like, shape (6,)
        Injected elements and the starting guess.
    epochs : ndarray, shape (n_epochs,)
        Observation times, decimal years.
    sigma : float
        Gaussian noise, arcseconds, added in the FlyStar frame.
    rng : numpy.random.Generator
        Noise source. The caller owns the seed so the two recoveries
        in one test are reproducible and independent.

    Returns
    -------
    table : StarTable
        After ``fit_motion_models``.

    Notes
    -----
    Positions are drawn from ``Orbit.model`` of the truth, so they
    already use ``x = -east`` and ``y = +north``. The table is seeded
    away from the truth. A frozen star, or a solver that returned the
    seed, would fail the recovery limits below.
    """
    # Noise-free sky position of the injected elements.
    x_true, y_true = Orbit().model(epochs, np.asarray(truth, dtype=float))
    x = rng.normal(x_true, sigma)
    y = rng.normal(y_true, sigma)
    tab = _orbit_table(epochs, seed, x, y, sigma)
    tab.fit_motion_models(motion_models=['Orbit'], verbose=False)
    return tab


def test_recover_s02_like_and_s016_like():
    """Noisy S0-2-like and S0-16-like orbits move back toward the injection.

    Notes
    -----
    The S0-2-like star has eccentricity about 0.88 and a period of
    about 16 yr. Sixteen epochs from 2000 to 2018 cover more than one
    period. The noise is 0.5 mas (0.0005 arcsec). The recovery limits
    are the plan's: period to 2 percent, eccentricity to 0.02,
    periapse time to 0.1 yr, and both angles to 5 degrees after the
    branch fold. Those limits are wide next to 0.5 mas of noise on an
    axis of about 100 mas, and tight enough that the starting offset
    (0.4 yr in period, 0.02 in eccentricity, a few degrees) would fail
    if the solver left the seed alone. With at least four epochs and
    a converged fit, each ``orb_*_err`` is finite and positive, the
    covariance is 6 by 6 and symmetric, and the sky errors from that
    covariance are finite and not the fixed-star value of zero.

    The S0-16-like star has eccentricity about 0.97 and a period of
    about 55 yr. Epochs are bunched around periapse and then sparse,
    because a near-parabolic arc is constrained near periapse. The
    noise is 1 mas. Eccentricity to 0.03 and period to 5 percent are
    wider than the S0-2 limits because the period is long and the arc
    is incomplete. The angle rule is the same 5 degrees.
    """
    rng = np.random.default_rng(0)
    # Injected S0-2-like elements, then a seed offset from them.
    s02_true = np.array([16.0, 2010.0, 0.88, 134.7, 48.4, 246.7])
    s02_seed = s02_true + np.array([0.4, 0.05, 0.02, 2.0, 3.0, -3.0])
    s02 = _recover(
        s02_true, s02_seed, np.linspace(2000.0, 2018.0, 16), 0.0005, rng,
    )
    got = np.array([s02[name][0] for name in _NAMES])
    # |dP|/P < 0.02, |de| < 0.02, |dt0| < 0.1 yr.
    assert abs(got[0] - s02_true[0]) / s02_true[0] < 0.02
    assert abs(got[2] - s02_true[2]) < 0.02
    assert abs(got[1] - s02_true[1]) < 0.1
    # Angles after choosing the (Omega, omega) branch nearer the seed.
    d_node, d_peri = _branch_near_seed(got, s02_true, s02_seed)
    assert d_node < 5.0
    assert d_peri < 5.0
    # Four or more epochs: a converged fit records diagnostics.
    assert s02['orb_fit_converged'][0]
    assert s02['orb_fit_n_iter'][0] > 0
    assert s02['orb_cov'][0].shape == (6, 6)
    # The covariance of the reported elements is symmetric.
    np.testing.assert_allclose(s02['orb_cov'][0], s02['orb_cov'][0].T)
    for name in _NAMES:
        err = s02[name + '_err'][0]
        assert np.isfinite(err) and err > 0.0
    # The elements actually moved. A frozen star would be bitwise equal.
    assert not np.array_equal(got, s02_seed)
    # Sky errors come from the covariance, so they are finite and not 0.
    _, _, xe, ye = s02.infer_positions(2012.0)
    assert np.isfinite(xe) and np.isfinite(ye)
    assert xe != 0.0 and ye != 0.0

    # S0-16-like: high eccentricity, long period, periapse in the window.
    s16_true = np.array([55.0, 2010.0, 0.97, 99.4, 227.0, 334.0])
    s16_seed = s16_true + np.array([1.5, 0.02, 0.004, 1.0, 2.0, -2.0])
    # Dense sample across periapse, then a sparse tail out to 2050.
    epochs = np.concatenate([
        np.linspace(2009.2, 2010.8, 10),
        np.array([2015.0, 2020.0, 2025.0, 2030.0, 2040.0, 2050.0]),
    ])
    s16 = _recover(s16_true, s16_seed, epochs, 0.001, rng)
    got16 = np.array([s16[name][0] for name in _NAMES])
    assert abs(got16[2] - s16_true[2]) < 0.03
    assert abs(got16[0] - s16_true[0]) / s16_true[0] < 0.05
    d_node, d_peri = _branch_near_seed(got16, s16_true, s16_seed)
    assert d_node < 5.0
    assert d_peri < 5.0
    return None


def test_too_few_epochs_and_a_failed_solve_keep_the_seed():
    """n_fit of 2 and 3, and a seed that cannot be integrated.

    Notes
    -----
    Six elements and two sky coordinates need three epochs. With two
    epochs the solver is not called. ``demote`` is false, so the star
    stays ``Orbit``. The elements stay bitwise equal to the seed,
    ``orb_fit_n_iter`` is 0, and ``orb_fit_converged`` is false.
    Prediction matches that seed. The sky errors are non-finite,
    because a fit was requested and no covariance exists. That is
    different from a fixed star, whose errors are zero. The position
    comparison uses 1e-12 arcsec, the same numerical floor as the
    other prediction tests.

    With three epochs the fit may converge and update the elements.
    There is no residual degree of freedom, so the errors, the
    covariance, and the sky errors stay non-finite.

    An eccentricity of 1.2 is finite, so the star is not dropped as a
    non-finite input, but ``kep2xyz`` cannot integrate ``e`` outside
    ``[0, 1)``. The seed is kept: eccentricity stays 1.2, the period
    stays finite rather than becoming a NaN fill, convergence is
    false, and the model stays ``Orbit``.
    """
    truth = np.array([16.0, 2010.0, 0.5, 60.0, 20.0, 30.0])
    # Seed differs in period only, so a solved fit would be visible.
    seed = truth + np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Two distinct epochs: below the algebraic minimum of three.
    epochs2 = np.array([2008.0, 2012.0])
    x, y = Orbit().model(epochs2, truth)
    tab2 = _orbit_table(epochs2, seed, x, y, 0.001)
    before = np.array([tab2[name][0] for name in _NAMES])
    tab2.fit_motion_models(motion_models=['Orbit'], verbose=False)
    after = np.array([tab2[name][0] for name in _NAMES])
    # Bitwise: the solver did not write a new period.
    assert np.array_equal(after, before)
    assert tab2['motion_model_used'][0] == 'Orbit'
    assert not tab2['orb_fit_converged'][0]
    assert tab2['orb_fit_n_iter'][0] == 0
    # Prediction uses the kept seed, not the noise-free truth.
    x_seed, y_seed = Orbit().model(2010.0, seed)
    x_hat, y_hat, xe, ye = tab2.infer_positions(2010.0)
    np.testing.assert_allclose(x_hat, x_seed, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(y_hat, y_seed, rtol=0.0, atol=1e-12)
    # A requested fit with no covariance: errors are non-finite, not 0.
    assert not np.isfinite(xe) and not np.isfinite(ye)

    # Three epochs: solvable, but no degrees of freedom for a covariance.
    epochs3 = np.array([2009.0, 2010.0, 2011.0])
    x3, y3 = Orbit().model(epochs3, truth)
    tab3 = _orbit_table(epochs3, seed, x3, y3, 0.001)
    tab3.fit_motion_models(motion_models=['Orbit'], verbose=False)
    assert tab3['motion_model_used'][0] == 'Orbit'
    assert tab3['orb_fit_converged'][0]
    for name in _NAMES:
        assert not np.isfinite(tab3[name + '_err'][0])
    assert not np.all(np.isfinite(tab3['orb_cov'][0]))
    _, _, xe3, ye3 = tab3.infer_positions(2010.0)
    assert not np.isfinite(xe3) and not np.isfinite(ye3)

    # e = 1.2 is finite, so the star stays Orbit, but it cannot be integrated.
    bad = np.array(truth, dtype=float)
    bad[2] = 1.2
    epochs = np.linspace(2005.0, 2015.0, 6)
    # Detections are a legal orbit. The seed is the illegal one.
    x_ok, y_ok = Orbit().model(epochs, truth)
    tab_bad = _orbit_table(epochs, bad, x_ok, y_ok, 0.001)
    tab_bad.fit_motion_models(motion_models=['Orbit'], verbose=False)
    # The illegal eccentricity is kept. It is not replaced by a fill.
    assert tab_bad['orb_e'][0] == pytest.approx(1.2)
    assert tab_bad['motion_model_used'][0] == 'Orbit'
    assert not tab_bad['orb_fit_converged'][0]
    assert np.isfinite(tab_bad['orb_P'][0])

    return None


def test_eccen_anomaly_at_half_eccentricity_and_zero_mean():
    """Mean anomaly 0 at e just below 1/2 must return the root.

    Notes
    -----
    This is the evaluation that used to raise on the three-epoch fit.
    ``least_squares`` steps off a seed eccentricity of 0.5 to
    ``e = 0.49999999999999084`` while the star is at periapse, so the
    mean anomaly is 0. The Mikkola starter then returns about
    ``-1e-32``. The gcwork wrap ``E < 0 → E + 2π`` turns that into
    ``2π``. The residual against ``M = 0`` is ``2π``, and the Newton
    step ``2π / (1 - e)`` swaps ``+2π`` and ``-2π`` forever. The step
    stays near ``4π``, so it never falls below ``1e-10``, and 50
    iterations raised ``RuntimeError``. The same wrap at ``e = 0.9``
    makes the step grow instead of oscillate. Both roots are ``E = 0``.
    The residual tolerance is ``1e-10`` radians, the Newton ``thresh``.
    That is the failure the old code hit. It is much tighter than the
    ``1e-12`` arcsec gcwork position check, which does not include
    this particular eccentricity.
    """
    # The float the macOS fit actually evaluated. Not a rounded 0.5.
    ecc = 0.49999999999999084
    mean_anomaly = np.array([0.0])
    eccentric = eccen_anomaly(mean_anomaly, ecc)
    # Kepler residual at the returned angle. E = 0 is the root.
    residual = (
        eccentric[0] - ecc * np.sin(eccentric[0]) - mean_anomaly[0]
    )
    assert np.isfinite(eccentric[0])
    assert abs(float(residual)) < 1e-10

    # Same mean anomaly, higher e: the old Newton step ran away.
    ecc_runaway = 0.9
    eccentric_high = eccen_anomaly(mean_anomaly, ecc_runaway)
    residual_high = (
        eccentric_high[0]
        - ecc_runaway * np.sin(eccentric_high[0])
        - mean_anomaly[0]
    )
    assert np.isfinite(eccentric_high[0])
    assert abs(float(residual_high)) < 1e-10

    return None
