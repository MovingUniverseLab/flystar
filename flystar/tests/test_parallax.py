"""
Tests of the barycentric parallax reference point (docs/plans/tangent_plane,
Sections 7 and 11). These use the JPL ephemeris through astropy, which is
downloaded on first use.
"""
import numpy as np
from astropy import units as u
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from astropy.time import Time

from flystar import motion_model, parallax, projection


def test_parallax_factor_is_barycentric():
    """
    parallax_in_direction equals the east/north components of -r_earth, the
    Earth's barycentric position from astropy, rotated by pa.
    """
    ra = np.array([266.4168, 150.0, 10.0])
    dec = np.array([-29.0078, 60.0, 41.0])
    pa = np.array([0.0, 30.0, -75.0])
    t = Time(np.arange(2015.0, 2025.0, 0.37), format='decimalyear', scale='utc').tdb

    pvec = parallax.parallax_in_direction(ra, dec, t.mjd, pa=pa)

    with solar_system_ephemeris.set('jpl'):
        r_earth = get_body_barycentric('earth', t).xyz.to_value(u.au).T   # (N_times, 3)
    _, e, n = projection.unit_vectors(ra, dec)                           # (N_stars, 3)
    p_e = -e @ r_earth.T                                                 # (N_stars, N_times)
    p_n = -n @ r_earth.T
    c = np.cos(np.radians(pa))[:, None]
    s = np.sin(np.radians(pa))[:, None]
    x = -p_e * c + p_n * s
    y = p_e * s + p_n * c

    np.testing.assert_allclose(pvec[:, 0, :], x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pvec[:, 1, :], y, rtol=0, atol=1e-12)

    # Not the heliocentric factor: the Sun is ~0.005-0.01 au from the barycenter.
    with solar_system_ephemeris.set('jpl'):
        r_sun = get_body_barycentric('sun', t).xyz.to_value(u.au).T
    p_e_helio = e @ (r_sun - r_earth).T
    assert np.abs(p_e_helio - p_e).max() > 1e-3


def test_parallax_fit_recovers_barycentric_position():
    """
    Positions simulated with Gaia's barycentric astrometric model (Eq. prop,
    then gnomonic projection) are fitted by the Parallax model with
    x0, y0 recovered to < 1 uas.
    """
    ra0, dec0 = 266.4168, -29.0078
    # The star is at the projection center at t0 (where the center's
    # parallax factor is exact).
    ra_star, dec_star = ra0, dec0
    pmra, pmdec, plx = 3.0, -5.0, 2.0       # mas/yr, mas
    t0 = 2020.0
    t = np.arange(2016.0, 2024.0, 0.13)

    tt = Time(t, format='decimalyear', scale='utc').tdb
    with solar_system_ephemeris.set('jpl'):
        r_obs = get_body_barycentric('earth', tt).xyz.to_value(u.au).T   # (N_times, 3)
    s = projection.parallax_shift_vector(ra_star, dec_star, r_obs)
    ra_t, dec_t = projection.propagate_on_sky(ra_star, dec_star, pmra, pmdec, t - t0,
                                              parallax=plx, s=s)
    xi, eta = projection.sky_to_plane(ra_t, dec_t, ra0, dec0)
    x, y = -xi, eta
    err = np.full_like(t, 1e-4)

    mod = motion_model.Parallax()
    fixed = {'t0': t0, 'ra': ra0, 'dec': dec0, 'pa': 0.0}
    params, _ = mod.fit(t, x, y, err, err, fixed_params_dict=fixed)
    x0_fit, vx_fit, y0_fit, vy_fit, pi_fit = np.ravel(params)

    assert abs(x0_fit) < 1e-6
    assert abs(y0_fit) < 1e-6
    np.testing.assert_allclose(pi_fit, plx / 1e3, rtol=0, atol=1e-6)
    np.testing.assert_allclose([vx_fit, vy_fit], [-pmra / 1e3, pmdec / 1e3], rtol=0, atol=1e-6)
