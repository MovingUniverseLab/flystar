"""
Tests for flystar.projection (docs/plans/tangent_plane, Section 11).
"""
import numpy as np
import erfa
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS

from flystar import projection

# Centers used throughout: the Galactic Center declination and a high one.
CENTERS = [(266.4168, -29.0078), (150.0, 60.0)]
FIELDS = [60.0, 300.0, 1800.0]   # field half-widths in arcsec (1', 5', 30')
UAS = 1e-6                       # 1 micro-arcsec in arcsec


def random_field(ra0, dec0, half_width, n=500, seed=0):
    """
    Random stars within half_width arcsec (each axis) of (ra0, dec0),
    placed with exact offsets on the sphere.
    """
    rng = np.random.default_rng(seed)
    dlon = rng.uniform(-half_width, half_width, n)
    dlat = rng.uniform(-half_width, half_width, n)
    # Include the center itself.
    dlon[0] = 0.0
    dlat[0] = 0.0
    center = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    stars = center.spherical_offsets_by(dlon * u.arcsec, dlat * u.arcsec)
    return stars.ra.deg, stars.dec.deg


@pytest.mark.parametrize('ra0, dec0', CENTERS)
@pytest.mark.parametrize('half_width', FIELDS)
def test_gnomonic_matches_wcs_tan(ra0, dec0, half_width):
    ra, dec = random_field(ra0, dec0, half_width)

    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [0.0, 0.0]
    # One "pixel" = 1 arcsec, +x east, +y north.
    w.wcs.cdelt = [1.0 / 3600.0, 1.0 / 3600.0]
    w.wcs.set()
    # WCS pixels are 0-based here with crpix = 0 in the 1-based FITS sense,
    # so use origin=1 to make (0, 0) the reference pixel.
    x_wcs, y_wcs = w.wcs_world2pix(ra, dec, 1)

    xi, eta = projection.sky_to_plane(ra, dec, ra0, dec0)

    np.testing.assert_allclose(xi, x_wcs, rtol=0, atol=UAS)
    np.testing.assert_allclose(eta, y_wcs, rtol=0, atol=UAS)


@pytest.mark.parametrize('proj', ['gnomonic', 'flat', 'flat_legacy'])
@pytest.mark.parametrize('ra0, dec0', CENTERS)
@pytest.mark.parametrize('half_width', FIELDS)
def test_round_trip(proj, ra0, dec0, half_width):
    ra, dec = random_field(ra0, dec0, half_width, seed=1)

    xi, eta = projection.sky_to_plane(ra, dec, ra0, dec0, projection=proj)
    ra2, dec2, _, _ = projection.plane_to_sky(xi, eta, ra0, dec0, projection=proj)

    c1 = SkyCoord(ra * u.deg, dec * u.deg)
    c2 = SkyCoord(ra2 * u.deg, dec2 * u.deg)
    assert c1.separation(c2).to_value(u.arcsec).max() < UAS


def test_flat_matches_equations():
    ra0, dec0 = 266.4168, -29.0078
    ra = np.array([266.41, 266.43])
    dec = np.array([-29.01, -29.0])
    xi, eta = projection.sky_to_plane(ra, dec, ra0, dec0, projection='flat')
    np.testing.assert_array_equal(xi, (ra - ra0) * np.cos(np.radians(dec0)) * 3600.0)
    np.testing.assert_array_equal(eta, (dec - dec0) * 3600.0)

    # 'flat' wraps the RA difference across 0/360; 'flat_legacy' does not.
    xi_f, _ = projection.sky_to_plane(359.99, 10.0, 0.01, 10.0, projection='flat')
    xi_l, _ = projection.sky_to_plane(359.99, 10.0, 0.01, 10.0, projection='flat_legacy')
    np.testing.assert_allclose(xi_f, -0.02 * np.cos(np.radians(10.0)) * 3600, rtol=1e-9)
    assert xi_l > 1e6


def test_unknown_projection():
    with pytest.raises(ValueError):
        projection.sky_to_plane(1.0, 1.0, 1.0, 1.0, projection='tan')


def numerical_jacobian(ra, dec, ra0, dec0, proj, step_as=0.1):
    """
    d(xi, eta) / d(alpha*, delta) by central differences along each star's
    own east and north (exact offsets on the sphere).
    """
    c = SkyCoord(ra * u.deg, dec * u.deg)
    h = step_as * u.arcsec
    zero = 0 * u.arcsec
    cols = []
    for dlon, dlat in [(h, zero), (zero, h)]:
        cp = c.spherical_offsets_by(dlon, dlat)
        cm = c.spherical_offsets_by(-dlon, -dlat)
        xp, yp = projection.sky_to_plane(cp.ra.deg, cp.dec.deg, ra0, dec0, projection=proj)
        xm, ym = projection.sky_to_plane(cm.ra.deg, cm.dec.deg, ra0, dec0, projection=proj)
        cols.append(np.stack([(xp - xm), (yp - ym)], axis=-1) / (2 * step_as))
    return np.stack(cols, axis=-1)


@pytest.mark.parametrize('proj', ['gnomonic', 'flat'])
@pytest.mark.parametrize('ra0, dec0', CENTERS)
@pytest.mark.parametrize('half_width', FIELDS)
def test_jacobian_numerical(proj, ra0, dec0, half_width):
    ra, dec = random_field(ra0, dec0, half_width, n=50, seed=2)
    jac = projection.jacobian(ra, dec, ra0, dec0, projection=proj)
    jac_num = numerical_jacobian(ra, dec, ra0, dec0, proj)
    assert jac.shape == (len(ra), 2, 2)
    np.testing.assert_allclose(jac, jac_num, rtol=0, atol=1e-8)


def test_jacobian_first_order_rotation():
    # J_g ~ R(theta), theta = -dra sin(dec0), Eq. (jrot).
    ra0, dec0 = 266.4168, -29.0078
    ra = ra0 + 60.0 / 3600.0 / np.cos(np.radians(dec0))
    jac = projection.jacobian(ra, dec0, ra0, dec0)
    theta = -np.radians(ra - ra0) * np.sin(np.radians(dec0))
    np.testing.assert_allclose(jac, [[1, theta], [-theta, 1]], atol=1e-7)


@pytest.mark.parametrize('ra0, dec0', CENTERS)
@pytest.mark.parametrize('half_width', FIELDS)
def test_pmout_vector_form(ra0, dec0, half_width):
    """
    Eq. (pmout) gives the same proper motion as J^{-1}.
    """
    rng = np.random.default_rng(3)
    ra, dec = random_field(ra0, dec0, half_width, n=100, seed=3)
    xi_dot = rng.normal(0, 5, len(ra))
    eta_dot = rng.normal(0, 5, len(ra))

    jinv = np.linalg.inv(projection.jacobian(ra, dec, ra0, dec0))
    mu = np.einsum('nij,nj->ni', jinv, np.stack([xi_dot, eta_dot], axis=-1))

    xi, eta = projection.sky_to_plane(ra, dec, ra0, dec0)
    u0, e0, n0 = projection.unit_vectors(ra0, dec0)
    P = u0 + (xi * projection.AS2RAD)[:, None] * e0 + (eta * projection.AS2RAD)[:, None] * n0
    Pnorm = np.linalg.norm(P, axis=-1)
    uu = P / Pnorm[:, None]
    Pdot = xi_dot[:, None] * e0 + eta_dot[:, None] * n0
    udot = (Pdot - uu * np.sum(uu * Pdot, axis=-1)[:, None]) / Pnorm[:, None]
    _, e, n = projection.unit_vectors(ra, dec)
    mu_vec = np.stack([np.sum(udot * e, axis=-1), np.sum(udot * n, axis=-1)], axis=-1)

    np.testing.assert_allclose(mu, mu_vec, rtol=0, atol=1e-9)


@pytest.mark.filterwarnings('ignore::erfa.ErfaWarning')
@pytest.mark.parametrize('ra0, dec0', CENTERS)
def test_propagate_on_sky_matches_astropy(ra0, dec0):
    rng = np.random.default_rng(4)
    ra, dec = random_field(ra0, dec0, 1800.0, n=200, seed=4)
    pmra = rng.normal(0, 50, len(ra))
    pmdec = rng.normal(0, 50, len(ra))

    for dt in [-15.0, 10.0, 30.0]:
        ra_t, dec_t = projection.propagate_on_sky(ra, dec, pmra, pmdec, dt)

        c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg,
                     pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                     obstime=Time(2016.0, format='jyear'))
        c_t = c.apply_space_motion(new_obstime=Time(2016.0 + dt, format='jyear'))
        sep = SkyCoord(ra_t * u.deg, dec_t * u.deg).separation(
            SkyCoord(c_t.ra, c_t.dec)).to_value(u.arcsec)
        assert sep.max() < UAS


def test_propagate_on_sky_parallax():
    # A star at (0, 0) and an observer 1 au along +y (toward the star's east):
    # the shift is pi * s, with s = -y, i.e. toward the west.
    ra, dec = 0.0, 0.0
    obs = np.array([0.0, 1.0, 0.0])
    s = projection.parallax_shift_vector(ra, dec, obs)
    np.testing.assert_allclose(s, [0.0, -1.0, 0.0], atol=1e-15)
    ra_t, dec_t = projection.propagate_on_sky(ra, dec, 0.0, 0.0, 0.0, parallax=1000.0, s=s)
    # Moved 1 arcsec toward -y, i.e. west (RA decreases).
    np.testing.assert_allclose(projection.spherical_offsets(ra_t, dec_t, ra, dec),
                               [-1.0, 0.0], atol=1e-9)


@pytest.mark.parametrize('proj', ['gnomonic', 'flat'])
@pytest.mark.parametrize('ra0, dec0', CENTERS)
@pytest.mark.parametrize('half_width', FIELDS)
def test_offsets_match_astropy(proj, ra0, dec0, half_width):
    ra, dec = random_field(ra0, dec0, half_width, n=200, seed=5)
    xi, eta = projection.sky_to_plane(ra, dec, ra0, dec0, projection=proj)
    _, _, dlon, dlat = projection.plane_to_sky(xi, eta, ra0, dec0, projection=proj)

    center = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    a_dlon, a_dlat = center.spherical_offsets_to(SkyCoord(ra * u.deg, dec * u.deg))
    np.testing.assert_allclose(dlon, a_dlon.to_value(u.arcsec), rtol=0, atol=UAS)
    np.testing.assert_allclose(dlat, a_dlat.to_value(u.arcsec), rtol=0, atol=UAS)


def test_offsets_do_not_round_trip():
    # The documented caveat: spherical offsets differ from (xi, eta) at third
    # order -- about 46 mas at 30' for gnomonic (report_numbers.tex).
    ra0, dec0 = 266.4168, -29.0078
    ra, dec = random_field(ra0, dec0, 1800.0, n=2000, seed=6)
    xi, eta = projection.sky_to_plane(ra, dec, ra0, dec0)
    _, _, dlon, dlat = projection.plane_to_sky(xi, eta, ra0, dec0)
    center = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    r = center.separation(SkyCoord(ra * u.deg, dec * u.deg)).to_value(u.arcsec)
    diff = np.hypot(dlon - xi, dlat - eta)[r <= 1800.0].max()
    assert 0.03 < diff < 0.06
