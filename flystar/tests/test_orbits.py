"""Newtonian orbit solver, orbits.dat reader, and per-star orbit fits."""

import pathlib

import numpy as np
import pytest
from astropy.table import Table

from flystar.motion_model import Orbit
from flystar.orbits import (
    attach_orbits,
    kep2xyz,
    read_orbits_dat,
    semi_major_mas,
)
from flystar.startables import StarTable


DATA = pathlib.Path(__file__).resolve().parent / 'test_data' / 'orbits_v2.0.2.dat'
GCWORK = (
    pathlib.Path(__file__).resolve().parent
    / 'test_data'
    / 'gcwork_kep2xyz_m4.07e6_r7960.1.csv'
)
_NAMES = ('orb_P', 'orb_t0', 'orb_e', 'orb_i', 'orb_Omega', 'orb_omega')


def _ang(angle, reference):
    """Absolute difference of two degrees, folded into [0, 180]."""
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
    """
    fit_o, fit_w = fitted[4], fitted[5]
    true_o, true_w = truth[4], truth[5]
    seed_o, seed_w = seed[4], seed[5]
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
        Measured positions, arcseconds.
    sigma : float
        Position uncertainty written into ``xe`` and ``ye``.

    Returns
    -------
    table : StarTable
        One row, ``motion_model_input='Orbit'``.
    """
    epochs = np.asarray(epochs, dtype=float)
    n_ep = epochs.size
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
    for name, value in zip(_NAMES, elements):
        tab[name] = np.array([float(value)])
    tab['motion_model_input'] = np.array(['Orbit'], dtype='U12')
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
    """Name, period, and the printed ``A`` field. ``A`` is not stored.

    Parameters
    ----------
    path : path-like
        Nine-field orbits.dat file.

    Returns
    -------
    names : list of str
    period, a_mas : ndarray, shape (n_stars,)
        Period in years and the file's semi-major axis in mas.
    """
    names = []
    period = []
    a_mas = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            names.append(fields[0])
            period.append(float(fields[1]))
            a_mas.append(float(fields[2]))
    return names, np.asarray(period), np.asarray(a_mas)


def test_kep2xyz_matches_orbit_model_on_a_grid():
    """Circular and eccentric positions match kep2xyz, with x = -east."""
    model = Orbit()
    epochs = np.linspace(1995.0, 2035.0, 21)
    cases = [
        np.array([16.0, 2010.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([16.0, 2002.31, 0.8891, 134.7, 48.4, 246.7]),
        np.array([54.709, 2000.156, 0.9772, 99.4, 227.0, 334.0]),
    ]
    for elements in cases:
        r_au, _, _ = kep2xyz(epochs, *elements)
        x, y = model.model(epochs, elements)
        np.testing.assert_allclose(x, -r_au[:, 0], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(y, r_au[:, 1], rtol=0.0, atol=1e-12)

    # Face-on circular at periapse sits on the north axis: x = 0, y = a.
    circular = cases[0]
    a_as = semi_major_mas(circular[0], 4.0e6, 8.0e3) / 1000.0
    x0, y0 = model.model(circular[1], circular)
    np.testing.assert_allclose(x0, 0.0, atol=1e-12)
    np.testing.assert_allclose(y0, a_as, rtol=0.0, atol=1e-12)
    return None


def test_black_hole_offset_is_added_in_the_flystar_frame():
    """x_bh and vx_bh shift x. They do not flip the east/north signs."""
    elements = np.array([16.0, 2000.0, 0.0, 0.0, 0.0, 0.0])
    epoch = np.array([2010.0])
    r_au, _, _ = kep2xyz(epoch, *elements)
    x, y = Orbit().model(
        epoch, elements,
        fixed_params_dict={
            'x_bh': 0.01, 'y_bh': -0.02,
            'vx_bh': 0.001, 'vy_bh': -0.002, 't_bh': 2000.0,
        },
    )
    dt = 10.0
    np.testing.assert_allclose(x, 0.01 + 0.001 * dt - r_au[0, 0])
    np.testing.assert_allclose(y, -0.02 - 0.002 * dt + r_au[0, 1])
    return None


def test_fixed_star_position_errors_are_zero():
    """No covariance written means xe = ye = 0. There is no pos_err."""
    elements = np.array([16.0, 2010.0, 0.2, 40.0, 10.0, 20.0])
    errs = np.full(6, np.inf)
    x, y, xe, ye = Orbit().model(
        np.array([2010.0, 2012.0]), elements, fit_param_errs=errs,
    )
    assert x.shape == (2,)
    np.testing.assert_allclose(xe, 0.0)
    np.testing.assert_allclose(ye, 0.0)
    assert 'pos_err' not in Orbit.optional_fixed_params
    return None


def test_mass_and_dist_defaults_and_override_order():
    """Defaults are 4e6 Msun and 8000 pc. Dict beats column beats meta."""
    assert Orbit.optional_fixed_params['mass'] == 4.0e6
    assert Orbit.optional_fixed_params['dist'] == 8.0e3
    elements = np.array([16.0, 2010.0, 0.0, 0.0, 0.0, 0.0])
    # y at periapse is the semi-major axis in arcseconds.
    y_default = semi_major_mas(16.0, 4.0e6, 8.0e3) / 1000.0
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
    tab['x0_err'] = np.array([0.001])
    tab['y0_err'] = np.array([0.001])

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
    """File A is within 0.02 mas of a_mas(P) at the default mass and distance."""
    names, period, a_file = _parse_file_a(DATA)
    a_model = semi_major_mas(period, 4.0e6, 8.0e3)
    np.testing.assert_allclose(a_model, a_file, rtol=0.0, atol=0.02)
    assert len(names) == 32
    table = read_orbits_dat(DATA)
    for banned in ('A', 'orb_A', 'search', 'orb_search'):
        assert banned not in table.colnames
        assert banned not in table.meta
    return None


def test_gcwork_east_north_with_explicit_mass_and_distance():
    """East and north match the transcribed gcwork snapshot.

    The CSV is produced by ``generate_gcwork_kep2xyz_fixture.py`` from
    ``orbits_jlu_python_gcwork_2024-10-03`` ``kep2xyz``, with
    ``M=4.07e6`` and ``R0=7960.1``. Those are not the FlyStar defaults.
    """
    rows = np.genfromtxt(
        GCWORK, delimiter=',', names=True, dtype=None, encoding='utf-8',
    )
    assert len(rows) == 160
    for name in np.unique(rows['name']):
        sub = rows[rows['name'] == name]
        assert sub['mass'][0] == pytest.approx(4.07e6)
        assert sub['dist'][0] == pytest.approx(7960.1)
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
    """32 rows, S0-2 elements, and attach only on a shared name."""
    table = read_orbits_dat(DATA)
    assert len(table) == 32
    row = table[table['name'] == 'S0-2'][0]
    assert row['orb_P'] == pytest.approx(16.067)
    assert row['orb_t0'] == pytest.approx(2002.310)
    assert row['orb_e'] == pytest.approx(0.8891)
    assert row['orb_i'] == pytest.approx(134.7)
    assert row['orb_Omega'] == pytest.approx(48.4)
    assert row['orb_omega'] == pytest.approx(246.7)

    catalog = Table()
    catalog['name'] = ['S0-2', 'S0-16', 'not-in-file']
    catalog['motion_model_input'] = np.array(
        ['Linear', 'Linear', 'Linear'], dtype='U12',
    )
    with pytest.warns(UserWarning, match='not in the catalog'):
        attach_orbits(catalog, table)
    assert 'fit_motion' not in catalog.colnames
    for banned in ('A', 'orb_A', 'search', 'orb_search'):
        assert banned not in catalog.colnames
        assert banned not in catalog.meta
    assert catalog['motion_model_input'][0] == 'Orbit'
    assert catalog['motion_model_input'][1] == 'Orbit'
    assert catalog['motion_model_input'][2] == 'Linear'
    assert catalog['orb_P'][0] == pytest.approx(16.067)
    assert catalog['orb_e'][1] == pytest.approx(0.9772)
    assert not np.isfinite(catalog['orb_P'][2])
    return None


def _recover(truth, seed, epochs, sigma, rng):
    """Fit one injected orbit and return the table.

    Parameters
    ----------
    truth, seed : array-like, shape (6,)
        Injected elements and the starting guess.
    epochs : ndarray, shape (n_epochs,)
        Observation times.
    sigma : float
        Gaussian noise, arcseconds.
    rng : numpy.random.Generator
        Noise source.

    Returns
    -------
    table : StarTable
        After ``fit_motion_models``.
    """
    x_true, y_true = Orbit().model(epochs, np.asarray(truth, dtype=float))
    x = rng.normal(x_true, sigma)
    y = rng.normal(y_true, sigma)
    tab = _orbit_table(epochs, seed, x, y, sigma)
    tab.fit_motion_models(motion_models=['Orbit'], verbose=False)
    return tab


def test_recover_s02_like_and_s016_like():
    """Noisy S0-2-like and S0-16-like orbits move back toward the injection."""
    rng = np.random.default_rng(0)
    s02_true = np.array([16.0, 2010.0, 0.88, 134.7, 48.4, 246.7])
    s02_seed = s02_true + np.array([0.4, 0.05, 0.02, 2.0, 3.0, -3.0])
    s02 = _recover(
        s02_true, s02_seed, np.linspace(2000.0, 2018.0, 16), 0.0005, rng,
    )
    got = np.array([s02[name][0] for name in _NAMES])
    assert abs(got[0] - s02_true[0]) / s02_true[0] < 0.02
    assert abs(got[2] - s02_true[2]) < 0.02
    assert abs(got[1] - s02_true[1]) < 0.1
    d_node, d_peri = _branch_near_seed(got, s02_true, s02_seed)
    assert d_node < 5.0
    assert d_peri < 5.0
    assert s02['orb_fit_converged'][0]
    assert s02['orb_fit_n_iter'][0] > 0
    assert s02['orb_cov'][0].shape == (6, 6)
    np.testing.assert_allclose(s02['orb_cov'][0], s02['orb_cov'][0].T)
    for name in _NAMES:
        err = s02[name + '_err'][0]
        assert np.isfinite(err) and err > 0.0
    # The elements actually moved. A frozen star would be bitwise equal.
    assert not np.array_equal(got, s02_seed)
    _, _, xe, ye = s02.infer_positions(2012.0)
    assert np.isfinite(xe) and np.isfinite(ye)
    assert xe != 0.0 and ye != 0.0

    s16_true = np.array([55.0, 2010.0, 0.97, 99.4, 227.0, 334.0])
    s16_seed = s16_true + np.array([1.5, 0.02, 0.004, 1.0, 2.0, -2.0])
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
    """n_fit of 2 and 3, and a seed that cannot be integrated."""
    truth = np.array([16.0, 2010.0, 0.5, 60.0, 20.0, 30.0])
    seed = truth + np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    epochs2 = np.array([2008.0, 2012.0])
    x, y = Orbit().model(epochs2, truth)
    tab2 = _orbit_table(epochs2, seed, x, y, 0.001)
    before = np.array([tab2[name][0] for name in _NAMES])
    tab2.fit_motion_models(motion_models=['Orbit'], verbose=False)
    after = np.array([tab2[name][0] for name in _NAMES])
    assert np.array_equal(after, before)
    assert tab2['motion_model_used'][0] == 'Orbit'
    assert not tab2['orb_fit_converged'][0]
    assert tab2['orb_fit_n_iter'][0] == 0
    x_seed, y_seed = Orbit().model(2010.0, seed)
    x_hat, y_hat, xe, ye = tab2.infer_positions(2010.0)
    np.testing.assert_allclose(x_hat, x_seed, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(y_hat, y_seed, rtol=0.0, atol=1e-12)
    assert not np.isfinite(xe) and not np.isfinite(ye)

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
    x_ok, y_ok = Orbit().model(epochs, truth)
    tab_bad = _orbit_table(epochs, bad, x_ok, y_ok, 0.001)
    tab_bad.fit_motion_models(motion_models=['Orbit'], verbose=False)
    assert tab_bad['orb_e'][0] == pytest.approx(1.2)
    assert tab_bad['motion_model_used'][0] == 'Orbit'
    assert not tab_bad['orb_fit_converged'][0]
    assert np.isfinite(tab_bad['orb_P'][0])
    return None
