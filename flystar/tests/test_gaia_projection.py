"""
Tests for the projection modes of analysis.prepare_gaia_for_flystar and
analysis.project_gaia (docs/plans/tangent_plane, Section 11).
"""
import os
import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord, get_body_barycentric, solar_system_ephemeris
from astropy.time import Time

from flystar import analysis, projection
from flystar.tests.gaia_synthetic import make_synthetic_gaia, CENTER_RA_STR, CENTER_DEC_STR
from flystar.tests.generate_gaia_legacy_fixture import legacy_outputs

test_dir = os.path.dirname(__file__)
LEGACY_FILE = os.path.join(test_dir, 'test_data', 'gaia_legacy_outputs.npz')

CENTER = SkyCoord(CENTER_RA_STR, CENTER_DEC_STR, unit=(u.hourangle, u.deg))
RA0, DEC0 = CENTER.ra.deg, CENTER.dec.deg


def prepare(gaia, proj, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return analysis.prepare_gaia_for_flystar(gaia, CENTER_RA_STR, CENTER_DEC_STR,
                                                 projection=proj, **kw)


# TEMPORARY: removed together with the 'flat_legacy' mode, before release.
def test_flat_legacy_bit_for_bit():
    """
    projection='flat_legacy' reproduces the outputs of the old
    prepare_gaia_for_flystar and project_gaia exactly (snapshot taken before
    the projection keyword was added; see generate_gaia_legacy_fixture.py).
    """
    ref = np.load(LEGACY_FILE)
    new = legacy_outputs({'projection': 'flat_legacy'})

    assert set(new.keys()) == set(ref.keys())
    for key in ref.keys():
        assert new[key].dtype == ref[key].dtype, key
        assert new[key].shape == ref[key].shape, key
        assert new[key].tobytes() == ref[key].tobytes(), key


def test_center_input_forms():
    gaia = make_synthetic_gaia()
    t_str = prepare(gaia, 'gnomonic')
    for ra, dec in [(RA0, DEC0), (CENTER, None), (RA0 * u.deg, DEC0 * u.deg)]:
        t = analysis.prepare_gaia_for_flystar(gaia, ra, dec)
        for col in ['x0', 'y0', 'vx', 'vy', 'x0_err', 'vx_err']:
            np.testing.assert_array_equal(t[col], t_str[col])

    lis_deg = analysis.project_gaia(gaia, 2020.0, RA0, DEC0)
    lis_str = analysis.project_gaia(gaia, 2020.0, CENTER_RA_STR, CENTER_DEC_STR)
    lis_sky = analysis.project_gaia(gaia, 2020.0, CENTER)
    np.testing.assert_array_equal(lis_deg['x'], lis_str['x'])
    np.testing.assert_array_equal(lis_deg['x'], lis_sky['x'])


@pytest.mark.parametrize('proj', ['gnomonic', 'flat'])
def test_prepare_new_modes(proj):
    gaia = make_synthetic_gaia()
    t = prepare(gaia, proj)

    # Metadata.
    assert t.meta['projection'] == proj
    assert t.meta['proj_ra0'] == RA0 and t.meta['proj_dec0'] == DEC0
    assert t.meta['frame'] == 'icrs'
    assert t.meta['parallax_ref'] == 'ssb'
    assert t.meta['data_release'] == 'EDR3/DR3'
    assert t.meta['ref_epoch'] == 2016.0
    assert t.meta['bright_pm_corrected'] is False

    # Positions; ra and dec kept.
    xi, eta = projection.sky_to_plane(gaia['ra'], gaia['dec'], RA0, DEC0, projection=proj)
    np.testing.assert_array_equal(t['x0'], -xi)
    np.testing.assert_array_equal(t['y0'], eta)
    np.testing.assert_array_equal(t['ra'], gaia['ra'])
    np.testing.assert_array_equal(t['dec'], gaia['dec'])

    # Proper motions: (vx, vy) = F J mu, zero where missing.
    jac = projection.jacobian(gaia['ra'], gaia['dec'], RA0, DEC0, projection=proj)
    has_pm = ~gaia['pmra'].mask
    mu = np.stack([gaia['pmra'].filled(0), gaia['pmdec'].filled(0)], axis=-1)
    v = np.einsum('nij,nj->ni', jac, mu) / 1e3
    np.testing.assert_allclose(t['vx'][has_pm], -v[has_pm, 0], rtol=1e-12)
    np.testing.assert_allclose(t['vy'][has_pm], v[has_pm, 1], rtol=1e-12)
    assert np.all(t['vx'][~has_pm] == 0) and np.all(t['vx_err'][~has_pm] == 0)

    # The proper motion is rotated (gnomonic) or rescaled (flat), not copied.
    assert np.abs(t['vx'][has_pm] + gaia['pmra'][has_pm] / 1e3).max() > 0

    # Errors: J is a rotation/scale within ~1e-3 over 2', so sigma_x0 is
    # sigma_alpha* (not sigma_alpha* cos(dec0), the old bug).
    np.testing.assert_allclose(t['x0_err'], gaia['ra_error'] / 1e3, rtol=2e-3)
    np.testing.assert_allclose(t['y0_err'], gaia['dec_error'] / 1e3, rtol=2e-3)
    np.testing.assert_allclose(t['vx_err'][has_pm], gaia['pmra_error'][has_pm] / 1e3, rtol=2e-3)

    # Errors exactly: C' = F J C J^T F with Gaia's ra_dec_corr.
    i = 5
    sa, sd, rho = gaia['ra_error'][i], gaia['dec_error'][i], gaia['ra_dec_corr'][i]
    cov = np.array([[sa**2, rho * sa * sd], [rho * sa * sd, sd**2]])
    fj = np.diag([-1.0, 1.0]) @ jac[i]
    cp = fj @ cov @ fj.T
    np.testing.assert_allclose([t['x0_err'][i], t['y0_err'][i]], np.sqrt(np.diag(cp)) / 1e3, rtol=1e-12)

    # Parallax copied as before.
    legacy = prepare(make_synthetic_gaia(), 'flat_legacy')
    np.testing.assert_array_equal(t['pi'], legacy['pi'])
    np.testing.assert_array_equal(t['pi_err'], legacy['pi_err'])
    np.testing.assert_array_equal(t['t0'], legacy['t0'])


def test_project_gaia_epoch_equals_catalog():
    # At the catalog epoch and without parallax, project_gaia is the
    # catalog projection.
    gaia = make_synthetic_gaia()
    t = prepare(gaia, 'gnomonic')
    lis = analysis.project_gaia(gaia, 2016.0, RA0, DEC0)
    np.testing.assert_allclose(lis['x'], t['x0'], rtol=0, atol=1e-9)
    np.testing.assert_allclose(lis['y'], t['y0'], rtol=0, atol=1e-9)
    np.testing.assert_allclose(lis['xe'], t['x0_err'], rtol=1e-12)
    assert lis.meta['projection'] == 'gnomonic' and lis.meta['epoch'] == 2016.0


@pytest.mark.parametrize('proj', ['gnomonic', 'flat'])
def test_project_gaia_order_independent(proj):
    """
    Propagating on the sky then projecting (project_gaia) agrees with
    propagating in the plane with the rotated proper motion F J mu
    (prepare_gaia_for_flystar), to second order: the difference shown in
    Figure pmmap of the report is gone.
    """
    gaia = make_synthetic_gaia(half_width=250.0, n=250)
    t = prepare(gaia, proj)
    for epoch in [2006.0, 2026.0]:
        lis = analysis.project_gaia(gaia, epoch, RA0, DEC0, projection=proj)
        dt = epoch - 2016.0
        # 1e-4 mas.
        np.testing.assert_allclose(lis['x'], t['x0'] + t['vx'] * dt, rtol=0, atol=1e-7)
        np.testing.assert_allclose(lis['y'], t['y0'] + t['vy'] * dt, rtol=0, atol=1e-7)

    # Whereas the old unrotated propagation differs by more.
    old = analysis.project_gaia(gaia, 2026.0, RA0, DEC0, projection='flat_legacy')
    new = analysis.project_gaia(gaia, 2026.0, RA0, DEC0, projection='gnomonic')
    assert np.abs(np.asarray(old['x']) - new['x']).max() > 1e-5


def test_project_gaia_errors_with_correlations():
    gaia = make_synthetic_gaia()
    epoch = 2026.0
    dt = epoch - 2016.0
    lis = analysis.project_gaia(gaia, epoch, RA0, DEC0)
    has_pm = ~gaia['pmra'].mask

    sa, sm, rho = gaia['ra_error'], gaia['pmra_error'], gaia['ra_pmra_corr']
    var = sa**2 + 2 * dt * rho * sa * sm + dt**2 * sm**2
    # J ~ I to ~1e-3 here.
    np.testing.assert_allclose(lis['xe'][has_pm], np.sqrt(var[has_pm]) / 1e3, rtol=2e-3)
    np.testing.assert_allclose(lis['xe'][~has_pm], gaia['ra_error'][~has_pm] / 1e3, rtol=2e-3)

    # Without the correlation columns the errors are the plain quadrature sum.
    gaia_nc = make_synthetic_gaia(with_corr=False)
    lis_nc = analysis.project_gaia(gaia_nc, epoch, RA0, DEC0)
    var_nc = gaia_nc['ra_error']**2 + dt**2 * gaia_nc['pmra_error'].filled(0)**2
    np.testing.assert_allclose(lis_nc['xe'], np.sqrt(var_nc) / 1e3, rtol=2e-3)


def test_project_gaia_include_parallax():
    """
    include_parallax adds pi * F J (P_e, P_n), with the parallax factors
    computed directly from astropy's barycentric Earth position.
    """
    gaia = make_synthetic_gaia()
    epoch = 2021.3
    no_plx = analysis.project_gaia(gaia, epoch, RA0, DEC0)
    with_plx = analysis.project_gaia(gaia, epoch, RA0, DEC0, include_parallax=True)
    assert with_plx.meta['include_parallax'] is True

    t = Time(epoch, format='decimalyear', scale='utc').tdb
    with solar_system_ephemeris.set('jpl'):
        r_obs = get_body_barycentric('earth', t).xyz.to_value(u.au)
    _, e, n = projection.unit_vectors(gaia['ra'], gaia['dec'])
    p_e = -e @ r_obs
    p_n = -n @ r_obs
    plx = gaia['parallax'].filled(0.0) / 1e3   # arcsec
    jac = projection.jacobian(gaia['ra'], gaia['dec'], RA0, DEC0)
    shift = np.einsum('nij,nj->ni', jac, np.stack([p_e, p_n], axis=-1)) * plx[:, None]

    # 1 uas.
    np.testing.assert_allclose(with_plx['x'] - no_plx['x'], -shift[:, 0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(with_plx['y'] - no_plx['y'], shift[:, 1], rtol=0, atol=1e-6)
    assert np.abs(shift).max() > 1e-4

    # Stars without a parallax do not move.
    masked = gaia['parallax'].mask
    np.testing.assert_array_equal(with_plx['x'][masked], no_plx['x'][masked])


def test_legacy_rejects_new_options():
    gaia = make_synthetic_gaia()
    with pytest.raises(ValueError):
        analysis.project_gaia(gaia, 2020.0, RA0, DEC0, projection='flat_legacy', include_parallax=True)
    with pytest.raises(ValueError):
        analysis.prepare_gaia_for_flystar(gaia, RA0, DEC0, projection='flat_legacy',
                                          correct_bright_pm=True)


def test_prepare_correct_bright_pm():
    gaia = make_synthetic_gaia()
    t0 = prepare(gaia, 'gnomonic')
    t1 = prepare(gaia, 'gnomonic', correct_bright_pm=True)
    assert t1.meta['bright_pm_corrected'] is True

    bright = (gaia['phot_g_mean_mag'] < 13) & ~gaia['pmra'].mask
    assert bright.sum() > 0 and (~bright).sum() > 0
    np.testing.assert_array_equal(t1['vx'][~bright], t0['vx'][~bright])
    np.testing.assert_array_equal(t1['vy'][~bright], t0['vy'][~bright])
    # The spin is tens of micro-arcsec/yr.
    dv = np.hypot(t1['vx'] - t0['vx'], t1['vy'] - t0['vy'])[bright]
    assert np.all((dv > 1e-6) & (dv < 1e-4))

    lis0 = analysis.project_gaia(gaia, 2026.0, RA0, DEC0)
    lis1 = analysis.project_gaia(gaia, 2026.0, RA0, DEC0, correct_bright_pm=True)
    np.testing.assert_allclose(lis1['x'] - lis0['x'], (t1['vx'] - t0['vx']) * 10.0, rtol=0, atol=1e-9)

    # Other data releases are refused.
    gaia_dr2 = make_synthetic_gaia()
    gaia_dr2['ref_epoch'] = 2015.5
    with pytest.raises(ValueError):
        prepare(gaia_dr2, 'gnomonic', correct_bright_pm=True)
