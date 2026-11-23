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
from astropy.table import Table
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


RA_SEX = CENTER.ra.to_string(unit=u.hourangle, sep=':', precision=8)
DEC_SEX = CENTER.dec.to_string(unit=u.deg, sep=':', precision=7)


@pytest.mark.parametrize('ra, dec', [
    (RA_SEX, DEC_SEX),                          # sexagesimal strings: hours, degrees
    (RA_SEX.replace(':', ' '), DEC_SEX),        # space-separated hours
    (CENTER.ra.to_string(unit=u.hourangle, precision=8), DEC_SEX),   # 17h45m40.04s
    (RA0, DEC0),                                # decimal degrees
    (str(float(RA0)), str(float(DEC0))),        # numeric strings are degrees
    (RA0, DEC_SEX),                             # mixed: number RA, string Dec
    (RA_SEX, DEC0),                             # mixed: string RA, number Dec
    (RA0 * u.deg, DEC_SEX),                     # Quantity RA, string Dec
])
def test_parse_center_forms(ra, dec):
    ra0, dec0 = analysis._parse_center(ra, dec)
    assert abs(ra0 - RA0) < 1e-6 / 3600.0
    assert abs(dec0 - DEC0) < 1e-6 / 3600.0


def test_parse_center_rejects_bad_input():
    with pytest.raises(ValueError):
        analysis._parse_center(RA0)            # dec missing
    with pytest.raises(ValueError):
        analysis._parse_center(CENTER, DEC0)   # SkyCoord plus dec
    with pytest.raises(ValueError):
        analysis._parse_center(RA0, 120.0)     # dec out of range
    with pytest.raises(TypeError):
        analysis._parse_center([RA0], DEC0)


def test_query_gaia_center_forms(monkeypatch):
    """
    query_gaia searches about the same center for string and decimal-degree
    inputs (the Gaia archive call is replaced by a stub).
    """
    astroquery_gaia = pytest.importorskip('astroquery.gaia')
    centers = []

    class Job:
        def get_results(self):
            return Table({'SOURCE_ID': [1]})

    def fake_cone_search(coords, radius=None, table_name=None):
        centers.append((coords.ra.deg, coords.dec.deg, radius, table_name))
        return Job()

    monkeypatch.setattr(astroquery_gaia.Gaia, 'cone_search_async', fake_cone_search)

    for ra, dec in [(CENTER_RA_STR, CENTER_DEC_STR), (RA0, DEC0), (RA0, CENTER_DEC_STR),
                    (CENTER, None)]:
        gaia = analysis.query_gaia(ra, dec, search_radius=10.0)
        assert 'source_id' in gaia.colnames

    for ra, dec, radius, table_name in centers:
        assert abs(ra - RA0) < 1e-9 and abs(dec - DEC0) < 1e-9
        assert radius == 10.0 * u.arcsec and table_name == 'gaiadr3.gaia_source'


@pytest.mark.parametrize('proj', ['gnomonic', 'flat', 'flat_legacy'])
def test_project_gaia_without_parallax_columns(proj):
    """
    With include_parallax=False, project_gaia neither needs nor uses the
    parallax columns.
    """
    gaia = make_synthetic_gaia()
    lis = analysis.project_gaia(gaia, 2020.0, RA0, DEC0, projection=proj)

    no_plx = gaia.copy()
    no_plx.remove_columns([c for c in no_plx.colnames if 'parallax' in c])
    lis_no_plx = analysis.project_gaia(no_plx, 2020.0, RA0, DEC0, projection=proj)

    for col in ['x', 'y', 'xe', 'ye']:
        np.testing.assert_array_equal(lis_no_plx[col], lis[col])
    assert not any('parallax' in c or c in ('pi', 'pi_err') for c in lis_no_plx.colnames)

    if proj != 'flat_legacy':
        with pytest.raises(KeyError):
            analysis.project_gaia(no_plx, 2020.0, RA0, DEC0, projection=proj,
                                  include_parallax=True)


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


ROUND_TRIP_CENTERS = [(266.4168, -29.0078), (150.0, 60.0)]


@pytest.mark.parametrize('proj', ['gnomonic', 'flat'])
@pytest.mark.parametrize('ra0, dec0', ROUND_TRIP_CENTERS)
@pytest.mark.parametrize('half_width', [60.0, 300.0, 1800.0])
def test_startable_to_radec_round_trip(proj, ra0, dec0, half_width):
    """
    Gaia -> prepare_gaia_for_flystar -> startable_to_radec recovers the Gaia
    astrometry.
    """
    gaia = make_synthetic_gaia(ra0=ra0, dec0=dec0, half_width=half_width, n=200,
                               with_corr=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tab = analysis.prepare_gaia_for_flystar(gaia, ra0, dec0, projection=proj, pi_err_limit=10.0)
    radec = analysis.startable_to_radec(tab)

    assert radec.meta['projection'] == proj and radec.meta['frame'] == 'icrs'
    assert radec.meta['parallax_ref'] == 'ssb'

    sep = SkyCoord(np.asarray(radec['ra']) * u.deg, np.asarray(radec['dec']) * u.deg).separation(
        SkyCoord(np.asarray(gaia['ra']) * u.deg, np.asarray(gaia['dec']) * u.deg)).to_value(u.arcsec)
    assert sep.max() < 1e-6     # 1 uas

    has_pm = ~gaia['pmra'].mask
    np.testing.assert_allclose(radec['pmra'][has_pm], gaia['pmra'][has_pm], rtol=0, atol=1e-9)
    np.testing.assert_allclose(radec['pmdec'][has_pm], gaia['pmdec'][has_pm], rtol=0, atol=1e-9)
    assert np.all(radec['pmra'][~has_pm] == 0)

    has_plx = ~gaia['parallax'].mask
    np.testing.assert_allclose(radec['parallax'][has_plx], gaia['parallax'][has_plx], rtol=1e-12)
    np.testing.assert_allclose(radec['parallax_error'][has_plx], gaia['parallax_error'][has_plx],
                               rtol=1e-12)

    # Errors: exact up to the x-y covariance that StarTables drop. The frame
    # is rotated from each star's east/north by theta ~ -dra sin(dec0), so
    # dropping it changes each variance by at most ~2 theta^2 (sa^2 + sd^2).
    dra = np.radians(projection._wrap180(np.asarray(gaia['ra']) - ra0))
    theta2 = (dra * np.sin(np.radians(dec0)))**2 + 1e-12

    def check(err_out, err_a, err_b, mask=slice(None)):
        e_out, ea, eb = (np.asarray(v, dtype=float)[mask] for v in (err_out, err_a, err_b))
        assert np.all(np.abs(e_out**2 - ea**2) <= 4 * theta2[mask] * (ea**2 + eb**2) + 1e-12 * ea**2)

    check(radec['ra_error'], gaia['ra_error'], gaia['dec_error'])
    check(radec['dec_error'], gaia['dec_error'], gaia['ra_error'])
    check(radec['pmra_error'], gaia['pmra_error'].filled(0), gaia['pmdec_error'].filled(0), has_pm)
    check(radec['pmdec_error'], gaia['pmdec_error'].filled(0), gaia['pmra_error'].filled(0), has_pm)

    # Offsets from the center are astropy's spherical offsets.
    dlon, dlat = SkyCoord(ra0 * u.deg, dec0 * u.deg).spherical_offsets_to(
        SkyCoord(np.asarray(gaia['ra']) * u.deg, np.asarray(gaia['dec']) * u.deg))
    np.testing.assert_allclose(radec['dlon'], dlon.to_value(u.arcsec), rtol=0, atol=1e-6)
    np.testing.assert_allclose(radec['dlat'], dlat.to_value(u.arcsec), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(radec['x0'], tab['x0'])
    np.testing.assert_array_equal(radec['ref_epoch'], tab['t0'])

    # The output has Gaia's names and units, so it can be fed back in.
    radec['source_id'] = gaia['source_id']
    for name in ['phot_g_mean_mag', 'phot_g_mean_flux_over_error']:
        radec[name] = gaia[name]
    radec['parallax'][~has_plx] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tab2 = analysis.prepare_gaia_for_flystar(Table(radec, masked=True), ra0, dec0,
                                                 projection=proj, pi_err_limit=10.0)
    np.testing.assert_allclose(tab2['x0'], tab['x0'], rtol=0, atol=1e-6)
    np.testing.assert_allclose(tab2['vx'], tab['vx'], rtol=0, atol=1e-12)


def test_startable_to_radec_reference_point_and_rotation():
    ra0, dec0 = 150.0, 60.0
    gaia = make_synthetic_gaia(ra0=ra0, dec0=dec0, half_width=300.0, n=50, with_corr=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tab = analysis.prepare_gaia_for_flystar(gaia, ra0, dec0)
    ref = analysis.startable_to_radec(tab)

    # The frame origin is the tangent point, but the reference sky position
    # is star i (e.g. a science target) at frame position (x0_i, y0_i): the
    # tangent point is solved for and is the original center.
    i = 7
    tab_nometa = tab.copy()
    tab_nometa.meta = {}
    out = analysis.startable_to_radec(tab_nometa, ra_ref=gaia['ra'][i], dec_ref=gaia['dec'][i],
                                      x_ref=tab['x0'][i], y_ref=tab['y0'][i],
                                      projection='gnomonic')
    np.testing.assert_allclose(out.meta['proj_ra0'], ra0, rtol=0, atol=1e-11)
    np.testing.assert_allclose(out.meta['proj_dec0'], dec0, rtol=0, atol=1e-11)
    np.testing.assert_allclose(out['ra'], ref['ra'], rtol=0, atol=1e-11)
    np.testing.assert_allclose(out['dec'], ref['dec'], rtol=0, atol=1e-11)
    np.testing.assert_allclose(out['pmra'], ref['pmra'], rtol=0, atol=1e-9)
    # Offsets are from (ra_ref, dec_ref), i.e. from star i.
    assert abs(out['dlon'][i]) < 1e-6 and abs(out['dlat'][i]) < 1e-6

    # Using star i's sky position as if it were the tangent point is wrong.
    out = analysis.startable_to_radec(tab_nometa, ra_ref=gaia['ra'][i], dec_ref=gaia['dec'][i],
                                      projection='gnomonic')
    assert np.abs(out['dec'] - ref['dec']).max() * 3600 > 1.0

    # A frame in "pixels" of 0.01" with +y at PA = 30 deg.
    scale, pa = 0.01, 30.0
    c, s = np.cos(np.radians(pa)), np.sin(np.radians(pa))
    xi, eta = -tab['x0'], tab['y0']
    vxi, veta = -tab['vx'], tab['vy']
    tab_rot = tab.copy()
    # Invert xi = s(-x c + y s), eta = s(x s + y c).
    tab_rot['x0'] = (-xi * c + eta * s) / scale
    tab_rot['y0'] = (xi * s + eta * c) / scale
    tab_rot['vx'] = (-vxi * c + veta * s) / scale
    tab_rot['vy'] = (vxi * s + veta * c) / scale
    for col in ['x0_err', 'y0_err', 'vx_err', 'vy_err', 'pi', 'pi_err']:
        tab_rot[col] = tab[col] / scale
    out = analysis.startable_to_radec(tab_rot, scale=scale, pa=pa)
    for col in ['ra', 'dec']:
        np.testing.assert_allclose(out[col], ref[col], rtol=0, atol=1e-11)
    for col in ['pmra', 'pmdec', 'parallax']:
        np.testing.assert_allclose(out[col], ref[col], rtol=0, atol=1e-9)
    # Diagonal errors only rotate into themselves when x and y errors are
    # equal, so compare errors for that case.
    tab_iso = tab.copy()
    tab_iso['x0_err'] = tab['y0_err']
    tab_rot['x0_err'] = tab['y0_err'] / scale
    tab_rot['y0_err'] = tab['y0_err'] / scale
    ref_iso = analysis.startable_to_radec(tab_iso)
    out = analysis.startable_to_radec(tab_rot, scale=scale, pa=pa)
    np.testing.assert_allclose(out['ra_error'], ref_iso['ra_error'], rtol=1e-9)
    np.testing.assert_allclose(out['dec_error'], ref_iso['dec_error'], rtol=1e-9)


def test_startable_to_radec_epoch():
    gaia = make_synthetic_gaia(with_corr=False)
    tab = prepare(gaia, 'gnomonic')
    out = analysis.startable_to_radec(tab, epoch=2026.0)
    ra_t, dec_t = projection.propagate_on_sky(gaia['ra'], gaia['dec'], gaia['pmra'].filled(0),
                                              gaia['pmdec'].filled(0), 10.0)
    np.testing.assert_allclose(out['ra'], ra_t, rtol=0, atol=1e-11)
    np.testing.assert_allclose(out['dec'], dec_t, rtol=0, atol=1e-11)
    assert np.all(out['ref_epoch'] == 2026.0)
    var = gaia['ra_error']**2 + 100 * gaia['pmra_error'].filled(0)**2
    np.testing.assert_allclose(out['ra_error'], np.sqrt(var), rtol=1e-3)


# TEMPORARY: removed together with the 'flat_legacy' mode.
def test_startable_to_radec_flat_legacy():
    gaia = make_synthetic_gaia()
    tab = prepare(gaia, 'flat_legacy')
    out = analysis.startable_to_radec(tab)
    np.testing.assert_allclose(out['ra'], gaia['ra'], rtol=0, atol=1e-11)
    np.testing.assert_allclose(out['dec'], gaia['dec'], rtol=0, atol=1e-11)
    has_pm = ~gaia['pmra'].mask
    np.testing.assert_allclose(out['pmra'][has_pm], gaia['pmra'][has_pm], rtol=1e-12)


def test_startable_to_radec_needs_frame():
    gaia = make_synthetic_gaia()
    tab = prepare(gaia, 'gnomonic')
    tab.meta = {}
    with pytest.raises(ValueError):
        analysis.startable_to_radec(tab)
    with pytest.raises(ValueError):
        analysis.startable_to_radec(tab, projection='gnomonic')
    with pytest.raises(ValueError):
        analysis.startable_to_radec(tab, projection='gnomonic', x_ref=1.0)
