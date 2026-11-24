"""
Tests of the barycentric parallax reference point (docs/plans/tangent_plane,
Sections 7 and 11). These use the JPL ephemeris through astropy, which is
downloaded on first use.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import (get_body_barycentric, get_body_barycentric_posvel,
                                 solar_system_ephemeris)
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

    with solar_system_ephemeris.set(parallax.EPHEMERIS):
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
    with solar_system_ephemeris.set(parallax.EPHEMERIS):
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
    with solar_system_ephemeris.set(parallax.EPHEMERIS):
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


def test_observer_barycentric_horizons_matches_astropy():
    """
    The JPL Horizons path of get_observer_barycentric returns barycentric,
    ICRS-equatorial positions that are accurate between widely spaced epochs.

    Earth is requested by its Horizons ID '399', which astropy does not know,
    so it goes through Horizons, and is checked against astropy's 'earth' in
    parallax.EPHEMERIS (DE440s, which shares DE441's barycenter; astropy's
    'jpl' default, DE430, is offset by ~116 km). Epochs ~4 months apart
    catch the old behaviour of querying at the median epoch spacing and
    linearly interpolating; the
    z component catches an ecliptic (rather than equatorial) frame; and the
    offset from the Sun-centred position catches a heliocentric origin.
    Needs network access to JPL Horizons; skipped when it is unavailable.
    """
    pytest.importorskip('astroquery.jplhorizons')

    # Irregular epochs spanning ~3 years, ~4 months apart, in TDB.
    t = Time(np.array([2021.03, 2021.41, 2021.70, 2022.12, 2022.55, 2023.01, 2023.87]),
             format='decimalyear', scale='utc').tdb

    # Horizons path (velocity too, to exercise that branch).
    try:
        pos_h, vel_h = parallax.get_observer_barycentric('399', t, velocity=True)
    except OSError as exc:  # network failure (requests errors are OSErrors)
        pytest.skip(f'JPL Horizons unavailable: {exc}')

    # Shapes of r_h, r_a: [N_times, 3] in au; v_h, v_a in au/day.
    r_h = pos_h.xyz.to_value(u.au).T
    v_h = vel_h.xyz.to_value(u.au / u.day).T
    with solar_system_ephemeris.set(parallax.EPHEMERIS):
        pos_a, vel_a = get_body_barycentric_posvel('earth', t)
        r_sun = get_body_barycentric('sun', t).xyz.to_value(u.au).T
    r_a = pos_a.xyz.to_value(u.au).T
    v_a = vel_a.xyz.to_value(u.au / u.day).T

    # The 1-day cubic spline reproduces the ephemeris to < 1e-9 au (150 m).
    np.testing.assert_allclose(r_h, r_a, rtol=0, atol=1e-9)
    np.testing.assert_allclose(v_h, v_a, rtol=0, atol=1e-10)

    # Sanity: the heliocentric position differs by the Sun's offset (>1e-3 au).
    assert np.abs((r_a - r_sun) - r_h).max() > 1e-3
