"""
Numbers and figure for tangent_plane_report.tex.

Compares FlyStar's current "flat" projection (as in
analysis.prepare_gaia_for_flystar and analysis.project_gaia) with a true
gnomonic (tangent-plane) projection, and estimates the proper-motion and
parallax errors from using the field center's axes for every star.

Writes, next to this script:
    projection_table.tex        LaTeX table body, included by the report
    report_numbers.tex          \\newcommand macros for numbers quoted in the text
    projection_differences.pdf  two-panel figure

Run with:  python make_figures.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ignore any personal matplotlibrc so the figures come out the same for everyone.
plt.style.use('default')
import numpy as np
from astropy.time import Time

from flystar import parallax

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Conversions.
AS2RAD = np.pi / 180.0 / 3600.0
RAD2AS = 1.0 / AS2RAD

# Grid used in the table and the figure.
TABLE_DECS = [-29.0, 0.0, 30.0, 60.0]           # deg
TABLE_OFFSETS = [10.0, 60.0, 300.0, 1800.0]     # arcsec
PM_REF = 5.0                                    # mas/yr, reference proper motion
PI_REF = 1.0                                    # mas, reference parallax


def flat_project(ra, dec, ra0, dec0):
    """
    FlyStar's current projection: an equirectangular map scaled at the center.

    Parameters
    ----------
    ra, dec : float or numpy array
        Star coordinates in radians.
    ra0, dec0 : float
        Field center in radians.

    Returns
    -------
    xi, eta : float or numpy array
        East and north coordinates in radians (FlyStar's x is -xi).
    """
    xi = (ra - ra0) * np.cos(dec0)
    eta = dec - dec0

    return xi, eta


def gnomonic_project(ra, dec, ra0, dec0):
    """
    True gnomonic (tangent-plane) projection about (ra0, dec0).

    Parameters
    ----------
    ra, dec : float or numpy array
        Star coordinates in radians.
    ra0, dec0 : float
        Tangent point in radians.

    Returns
    -------
    xi, eta : float or numpy array
        Standard coordinates (east, north) in radians.
    """
    dra = ra - ra0

    # Cosine of the angular distance from the tangent point.
    denom = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(dra)

    xi = np.cos(dec) * np.sin(dra) / denom
    eta = (np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)) / denom

    return xi, eta


def gnomonic_deproject(xi, eta, ra0, dec0):
    """
    Inverse gnomonic projection: tangent-plane coordinates to RA and Dec.

    Parameters
    ----------
    xi, eta : float or numpy array
        Standard coordinates (east, north) in radians.
    ra0, dec0 : float or numpy array
        Tangent point in radians.

    Returns
    -------
    ra, dec : float or numpy array
        Coordinates in radians.
    """
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)

    # At the tangent point itself (rho = 0) the ratio sin(c)/rho tends to 1.
    sinc_over_rho = np.where(rho > 0, np.sin(c) / np.where(rho > 0, rho, 1.0), 1.0)

    dec = np.arcsin(np.cos(c) * np.sin(dec0) + eta * sinc_over_rho * np.cos(dec0))
    ra = ra0 + np.arctan2(xi * sinc_over_rho * rho,
                          rho * np.cos(dec0) * np.cos(c) - eta * np.sin(dec0) * np.sin(c))

    # For rho = 0 the atan2 above is atan2(0, 0); the star is at the tangent point.
    ra = np.where(rho > 0, ra, ra0)

    return ra, dec


def diagonal_star(offset_as, dec0):
    """
    Star offset equally east and north of the center by a total of offset_as.

    Parameters
    ----------
    offset_as : float or numpy array
        Total angular offset from the center in arcsec.
    dec0 : float
        Center declination in radians (center RA is 0).

    Returns
    -------
    ra, dec : float or numpy array
        Star coordinates in radians.
    """
    # Equal east and north components, each offset / sqrt(2).
    d = offset_as / np.sqrt(2.0) * AS2RAD
    dec = dec0 + d
    ra = d / np.cos(dec0)

    return ra, dec


def north_angle(ra, dec, ra0, dec0, step=1e-8):
    """
    Angle of the star's local north in the tangent plane.

    Parameters
    ----------
    ra, dec : float or numpy array
        Star coordinates in radians.
    ra0, dec0 : float
        Tangent point in radians.
    step : float, optional
        Finite-difference step in declination (radians).

    Returns
    -------
    theta : float or numpy array
        Angle (radians) from the tangent plane's +eta axis to the star's local
        north, measured toward +xi (east).
    """
    # Move the star a little north and see which way it goes in the plane.
    xi1, eta1 = gnomonic_project(ra, dec, ra0, dec0)
    xi2, eta2 = gnomonic_project(ra, dec + step, ra0, dec0)
    theta = np.arctan2(xi2 - xi1, eta2 - eta1)

    return theta


def gnomonic_jacobian(ra, dec, ra0, dec0):
    """
    Jacobian of the gnomonic projection with respect to (alpha*, delta).

    Parameters
    ----------
    ra, dec : float or numpy array
        Star coordinates in radians.
    ra0, dec0 : float
        Tangent point in radians.

    Returns
    -------
    jac : numpy array
        Shape (..., 2, 2). Maps a sky motion (mu_alpha*, mu_delta) along the
        star's own east and north to the plane motion (d xi, d eta). This is
        J_g in the report.
    """
    dra = ra - ra0
    denom = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(dra)
    xi, eta = gnomonic_project(ra, dec, ra0, dec0)
    k = np.cos(dec) * np.sin(dec0) - np.sin(dec) * np.cos(dec0) * np.cos(dra)

    j11 = np.cos(dra) + xi * np.cos(dec0) * np.sin(dra)
    j12 = -np.sin(dec) * np.sin(dra) - xi * k
    j21 = np.sin(dra) * (np.sin(dec0) + eta * np.cos(dec0))
    j22 = np.sin(dec) * np.sin(dec0) * np.cos(dra) + np.cos(dec) * np.cos(dec0) - eta * k

    jac = np.stack([np.stack([j11, j12], axis=-1), np.stack([j21, j22], axis=-1)], axis=-2)
    jac = jac / np.asarray(denom)[..., np.newaxis, np.newaxis]

    return jac


def check_expansions():
    """
    Check the second-order expansions of (gnomonic - flat) and of theta.

    Returns
    -------
    worst : float
        Largest relative error of the second-order formulas over the test grid.

    Notes
    -----
    The expansions are
        xi_g  - xi_flat  ~ -dra * ddec * sin(dec0)
        eta_g - eta_flat ~ 0.5 * dra**2 * sin(dec0) * cos(dec0)
        theta            ~ -dra * sin(dec0)
    with dra = ra - ra0 and ddec = dec - dec0. Their relative error should
    shrink in proportion to the offset.
    """
    worst = 0.0

    for dec0_deg in (-29.0, 30.0, 60.0):
        dec0 = np.radians(dec0_deg)

        for offset_as in (10.0, 60.0):
            ra, dec = diagonal_star(offset_as, dec0)
            dra, ddec = ra, dec - dec0

            # Exact differences.
            xi_f, eta_f = flat_project(ra, dec, 0.0, dec0)
            xi_g, eta_g = gnomonic_project(ra, dec, 0.0, dec0)
            theta = north_angle(ra, dec, 0.0, dec0)

            # Second-order predictions.
            pred_xi = -dra * ddec * np.sin(dec0)
            pred_eta = 0.5 * dra**2 * np.sin(dec0) * np.cos(dec0)
            pred_theta = -dra * np.sin(dec0)

            rel = [abs((xi_g - xi_f) / pred_xi - 1), abs((eta_g - eta_f) / pred_eta - 1),
                   abs(theta / pred_theta - 1)]
            worst = max(worst, max(rel))

    return worst


def parallax_error_fraction(offset_as, dec0_deg, ra0_deg=266.4):
    """
    Error in the parallax displacement from using the center's direction.

    Parameters
    ----------
    offset_as : float
        Star offset from the center in arcsec (diagonal, east and north).
    dec0_deg : float
        Center declination in degrees.
    ra0_deg : float, optional
        Center right ascension in degrees.

    Returns
    -------
    frac : float
        Largest difference over a year between the star's true parallax
        displacement (in the tangent plane's axes) and the one a fit with the
        center's direction assumes, as a fraction of the displacement amplitude.
    """
    dec0 = np.radians(dec0_deg)
    ra_rel, dec = diagonal_star(offset_as, dec0)
    ra_star_deg = ra0_deg + np.degrees(ra_rel)
    dec_star_deg = np.degrees(dec)

    # One year of epochs (TDB MJD).
    mjd = Time(2024.0, format='decimalyear', scale='utc').tdb.mjd + np.arange(0.0, 366.0, 2.0)

    # Parallax vectors (FlyStar x, y; units of the parallax) at the center and at the star,
    # each in its own local east/north axes. Shape: (2, N_times).
    p_center = parallax.parallax_in_direction(ra0_deg, dec0_deg, mjd)[0]
    p_star = parallax.parallax_in_direction(ra_star_deg, dec_star_deg, mjd)[0]

    # Rotate the star's local (east, north) into the tangent plane's (xi, eta).
    theta = north_angle(ra_rel, dec, 0.0, dec0)
    p_e, p_n = -p_star[0], p_star[1]
    xi = p_e * np.cos(theta) + p_n * np.sin(theta)
    eta = -p_e * np.sin(theta) + p_n * np.cos(theta)

    # Difference in FlyStar's frame (x = -xi), relative to the amplitude.
    diff = np.hypot(-xi - p_center[0], eta - p_center[1])
    frac = diff.max() / np.hypot(p_center[0], p_center[1]).max()

    return frac


def simulate_gc_field(n_stars=250, radius_as=250.0, seed=22):
    """
    Simulated Gaia-like reference stars around Sgr A*.

    Parameters
    ----------
    n_stars : int, optional
        Number of stars.
    radius_as : float, optional
        Stars are spread uniformly over a disk of this radius (arcsec).
    seed : int, optional
        Random-number seed.

    Returns
    -------
    ra0, dec0 : float
        Field center (Sgr A*) in radians.
    ra, dec : numpy array
        Star positions in radians.
    pmra, pmdec : numpy array
        Proper motions (mu_alpha*, mu_delta) in mas/yr, each in the star's
        own local east and north.
    """
    rng = np.random.default_rng(seed)

    # Sgr A* (Reid & Brunthaler 2004).
    ra0, dec0 = np.radians(266.416837), np.radians(-29.007811)

    # Uniform over a disk in the tangent plane, then onto the sky.
    r = radius_as * np.sqrt(rng.uniform(0, 1, n_stars)) * AS2RAD
    phi = rng.uniform(0, 2 * np.pi, n_stars)
    ra, dec = gnomonic_deproject(r * np.sin(phi), r * np.cos(phi), ra0, dec0)

    # Proper motions scattered about Sgr A*'s apparent motion.
    pmra = rng.normal(-3.2, 3.0, n_stars)
    pmdec = rng.normal(-5.6, 3.0, n_stars)

    return ra0, dec0, ra, dec, pmra, pmdec


def propagation_cases(ra0, dec0, ra, dec, pmra, pmdec, dt):
    """
    Positions after dt years in the central tangent plane, two ways.

    Parameters
    ----------
    ra0, dec0 : float
        Field center (tangent point) in radians.
    ra, dec : numpy array
        Star positions at the start, in radians.
    pmra, pmdec : numpy array
        Proper motions in mas/yr in each star's local east and north.
    dt : float
        Time step in years.

    Returns
    -------
    tan_first : tuple of numpy array
        (xi, eta) in radians. Case 1, current FlyStar: project the starting
        position into the central plane, then add the proper motion there.
    pm_first : tuple of numpy array
        (xi, eta) in radians. Case 2: move each star in its own local tangent
        plane, back to the sky, then project into the central plane.
    """
    # Displacement over dt, in radians.
    d_east = pmra * dt * 1e-3 * AS2RAD
    d_north = pmdec * dt * 1e-3 * AS2RAD

    # Case 1: tangent plane first.
    xi0, eta0 = gnomonic_project(ra, dec, ra0, dec0)
    tan_first = (xi0 + d_east, eta0 + d_north)

    # Case 2: proper motion first, in each star's own tangent plane.
    ra_t, dec_t = gnomonic_deproject(d_east, d_north, ra, dec)
    pm_first = gnomonic_project(ra_t, dec_t, ra0, dec0)

    return tan_first, pm_first


def pm_map():
    """
    Compute the two-case comparison for the simulated Galactic Center field.

    Returns
    -------
    result : dict
        x, y (FlyStar frame, arcsec), dra and ddec (case 2 minus case 1, in
        mas, along east and north), r (arcsec), dt (yr), n (stars), and
        pred_resid (mas): the largest difference between the computed offsets
        and the rotation formula's prediction.
    """
    dt = 10.0
    ra0, dec0, ra, dec, pmra, pmdec = simulate_gc_field()
    (xi1, eta1), (xi2, eta2) = propagation_cases(ra0, dec0, ra, dec, pmra, pmdec, dt)

    # Difference (PM first minus tangent plane first), in mas.
    dra = (xi2 - xi1) * RAD2AS * 1e3
    ddec = (eta2 - eta1) * RAD2AS * 1e3

    # Starting positions in the FlyStar frame (x = -xi), arcsec.
    xi0, eta0 = gnomonic_project(ra, dec, ra0, dec0)
    x, y = -xi0 * RAD2AS, eta0 * RAD2AS

    # Prediction: the motion rotated by each star's local-north angle theta.
    theta = north_angle(ra, dec, ra0, dec0)
    mu_e, mu_n = pmra * dt, pmdec * dt
    pred_dra = mu_e * np.cos(theta) + mu_n * np.sin(theta) - mu_e
    pred_ddec = -mu_e * np.sin(theta) + mu_n * np.cos(theta) - mu_n
    pred_resid = np.max(np.hypot(dra - pred_dra, ddec - pred_ddec))

    result = dict(x=x, y=y, dra=dra, ddec=ddec, r=np.hypot(x, y), dt=dt, n=len(ra),
                  pred_resid=pred_resid)

    return result


def make_pm_map_figure(res):
    """
    Map of the two-case position difference, in the style of issue #22.

    Parameters
    ----------
    res : dict
        Output of pm_map().

    Returns
    -------
    None
    """
    # Shared, symmetric color scale.
    vmax = np.max(np.abs(np.concatenate([res['dra'], res['ddec']])))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)

    for ax, key, title in zip(axes, ['dra', 'ddec'], [r'$\Delta\alpha*$', r'$\Delta\delta$']):
        sc = ax.scatter(res['x'], res['y'], c=res[key], cmap='RdBu', vmin=-vmax, vmax=vmax,
                        s=18, edgecolors='none')
        ax.set_title(title)
        ax.set_xlabel('x (arcsec, +west)')
        # +x is west, so east is on the left, as on the sky.
        ax.set_aspect('equal')
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(f'{title} (mas)')
    axes[0].set_ylabel('y (arcsec, +north)')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pm_propagation_map.pdf'))
    plt.close(fig)

    return None


def error_maps(pm_res):
    """
    Errors in the simulated Galactic Center field from today's handling.

    Parameters
    ----------
    pm_res : dict
        Output of pm_map(), used to cross-check the proper-motion errors.

    Returns
    -------
    result : dict
        x, y (FlyStar frame, arcsec) and r (arcsec); then, each along east
        and north:
        pos_e, pos_n -- flat minus gnomonic position at one instant (mas);
        pm_e, pm_n -- proper motion copied in unchanged minus J_g mu (uas/yr);
        pi_e, pi_n -- largest difference over a year between the parallax
        displacement modeled with the center's parallax factor and the star's
        true one in the gnomonic frame, for a PI_REF parallax (uas).
        Also pm_check (mas): largest difference between pm error * dt and
        the propagation map of pm_map().
    """
    ra0, dec0, ra, dec, pmra, pmdec = simulate_gc_field()

    # Gnomonic positions; FlyStar frame x = -xi.
    xi_g, eta_g = gnomonic_project(ra, dec, ra0, dec0)
    x, y = -xi_g * RAD2AS, eta_g * RAD2AS

    # 1. Positions: flat minus gnomonic.
    xi_f, eta_f = flat_project(ra, dec, ra0, dec0)
    pos_e = (xi_f - xi_g) * RAD2AS * 1e3
    pos_n = (eta_f - eta_g) * RAD2AS * 1e3

    # 2. Proper motions: copied in (J = I) minus the exact J_g mu.
    jac = gnomonic_jacobian(ra, dec, ra0, dec0)               # (N, 2, 2)
    mu = np.stack([pmra, pmdec], axis=-1)                      # (N, 2), mas/yr
    mu_plane = np.einsum('nij,nj->ni', jac, mu)
    pm_e = (mu[:, 0] - mu_plane[:, 0]) * 1e3                   # uas/yr
    pm_n = (mu[:, 1] - mu_plane[:, 1]) * 1e3

    # Cross-check against the issue #22 map: there the difference is
    # (proper motion first) - (plane first) = (J_g mu - mu) * dt.
    pm_check = np.max(np.hypot(-pm_e * 1e-3 * pm_res['dt'] - pm_res['dra'],
                               -pm_n * 1e-3 * pm_res['dt'] - pm_res['ddec']))

    # 3. Parallax: center's factor (pa = 0) versus each star's own, through J_g.
    mjd = Time(2024.0, format='decimalyear', scale='utc').tdb.mjd + np.arange(0.0, 366.0, 2.0)
    p_center = parallax.parallax_in_direction(np.degrees(ra0), np.degrees(dec0), mjd)[0]   # (2, T)
    p_star = parallax.parallax_in_direction(np.degrees(ra), np.degrees(dec), mjd)         # (N, 2, T)

    # parallax_in_direction returns FlyStar (x, y) = (-east, north) for pa = 0.
    pe_star, pn_star = -p_star[:, 0, :], p_star[:, 1, :]
    xi_true = jac[:, 0, 0, None] * pe_star + jac[:, 0, 1, None] * pn_star
    eta_true = jac[:, 1, 0, None] * pe_star + jac[:, 1, 1, None] * pn_star
    xi_model, eta_model = -p_center[0], p_center[1]

    # Model minus truth, largest |difference| over the year, in uas.
    pi_e = np.max(np.abs(xi_model - xi_true), axis=1) * PI_REF * 1e3
    pi_n = np.max(np.abs(eta_model - eta_true), axis=1) * PI_REF * 1e3

    result = dict(x=x, y=y, r=np.hypot(x, y), pos_e=pos_e, pos_n=pos_n,
                  pm_e=pm_e, pm_n=pm_n, pi_e=pi_e, pi_n=pi_n, pm_check=pm_check)

    return result


def make_error_map_figure(res):
    """
    Three-row map: position, proper-motion and parallax errors, east and north.

    Parameters
    ----------
    res : dict
        Output of error_maps().

    Returns
    -------
    None
    """
    rows = [('pos_e', 'pos_n', 'Position: flat $-$ gnomonic', 'mas', 'RdBu', True),
            ('pm_e', 'pm_n', r'PM: copied $-$ $J_\mathrm{g}\mu$', r'$\mu$as yr$^{-1}$', 'RdBu', True),
            ('pi_e', 'pi_n', 'Parallax: max |model $-$ true|', r'$\mu$as', 'viridis', False)]
    comps = ['east', 'north']

    fig, axes = plt.subplots(3, 2, figsize=(8.0, 10.0), sharex=True, sharey=True,
                             layout='constrained')

    for irow, (key_e, key_n, title, unit, cmap, signed) in enumerate(rows):
        vals = np.concatenate([res[key_e], res[key_n]])
        if signed:
            vmax = np.max(np.abs(vals))
            vmin = -vmax
        else:
            vmin, vmax = 0.0, vals.max()

        for icol, key in enumerate([key_e, key_n]):
            ax = axes[irow, icol]
            sc = ax.scatter(res['x'], res['y'], c=res[key], cmap=cmap, vmin=vmin, vmax=vmax,
                            s=14, edgecolors='none')
            ax.set_title(f'{title}, {comps[icol]}', fontsize=10)
            ax.set_aspect('equal')
            # +x is west, so east is on the left, as on the sky.

        # One color bar per row, shared by its east and north panels.
        cb = fig.colorbar(sc, ax=axes[irow, :], shrink=0.9)
        cb.set_label(unit)

        axes[irow, 0].set_ylabel('y (arcsec, +north)')
    for ax in axes[-1]:
        ax.set_xlabel('x (arcsec, +west)')

    fig.savefig(os.path.join(OUT_DIR, 'error_maps.pdf'))
    plt.close(fig)

    return None


def aberration_residuals(radii_as=(60.0, 300.0, 1800.0), n_grid=41):
    """
    Differential annual aberration across a field, and what alignment leaves.

    Parameters
    ----------
    radii_as : tuple of float, optional
        Field radii in arcsec.
    n_grid : int, optional
        Grid points per axis across the field diameter.

    Returns
    -------
    result : dict
        For each radius (key), a dict with the largest over Earth-velocity
        directions (0 to 90 deg from the field center) of: raw -- position
        change relative to the field center (mas); affine -- residual after a
        per-epoch first-order (6-parameter) polynomial fit (mas); quad --
        residual after a second-order (12-parameter) fit (mas).

    Notes
    -----
    Uses classical aberration u' = (u + v/c) / |u + v/c| with |v| = 29.78 km/s,
    projects gnomonically about the aberrated field center, and fits the
    aberrated positions as polynomials of the unaberrated ones. That mimics
    FlyStar's per-epoch transformation into a reference frame without
    aberration (Gaia's barycentric positions).
    """
    kappa = 29.78 / 299792.458

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def fit_resid(x, y, xa, ya, order):
        terms = [np.ones_like(x), x, y]
        if order >= 2:
            terms += [x * x, x * y, y * y]
        a = np.stack(terms, axis=1)
        rx = xa - a @ np.linalg.lstsq(a, xa, rcond=None)[0]
        ry = ya - a @ np.linalg.lstsq(a, ya, rcond=None)[0]
        return np.hypot(rx, ry).max() * RAD2AS * 1e3

    result = {}
    for radius in radii_as:
        g = np.linspace(-radius, radius, n_grid) * AS2RAD
        gx, gy = np.meshgrid(g, g)
        inside = np.hypot(gx, gy) <= radius * AS2RAD
        xi, eta = gx[inside], gy[inside]
        # Field center along +z; xi along +x (east), eta along +y (north).
        u = unit(np.stack([xi, eta, np.ones_like(xi)], axis=1))

        out = dict(raw=0.0, affine=0.0, quad=0.0)
        for phi in np.radians([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]):
            v = kappa * np.array([np.sin(phi), 0.0, np.cos(phi)])
            ua = unit(u + v)
            uc = unit(np.array([0.0, 0.0, 1.0]) + v)
            # Tangent-plane triad at the aberrated center.
            e_c = unit(np.cross([0.0, 1.0, 0.0], uc))
            n_c = np.cross(uc, e_c)
            xa, ya = (ua @ e_c) / (ua @ uc), (ua @ n_c) / (ua @ uc)

            out['raw'] = max(out['raw'], np.hypot(xa - xi, ya - eta).max() * RAD2AS * 1e3)
            out['affine'] = max(out['affine'], fit_resid(xi, eta, xa, ya, 1))
            out['quad'] = max(out['quad'], fit_resid(xi, eta, xa, ya, 2))
        result[radius] = out

    return result


def spherical_offset_differences(radii_as=(60.0, 300.0, 1800.0)):
    """
    Spherical offsets versus gnomonic standard coordinates.

    Parameters
    ----------
    radii_as : tuple of float, optional
        Distances from the reference in arcsec.

    Returns
    -------
    result : dict
        For each radius, the largest difference (mas), over position angle,
        between the gnomonic standard coordinates (xi, eta) and astropy-style
        spherical offsets (dlon, dlat) in a frame rotated so that the
        reference is at the origin.
    """
    result = {}
    for radius in radii_as:
        pa = np.linspace(0.0, 2.0 * np.pi, 361)
        xi, eta = radius * AS2RAD * np.sin(pa), radius * AS2RAD * np.cos(pa)
        # Direction with the reference at +z, east +x, north +y.
        norm = np.sqrt(1.0 + xi**2 + eta**2)
        ux, uy, uz = xi / norm, eta / norm, 1.0 / norm
        dlon, dlat = np.arctan2(ux, uz), np.arcsin(uy)
        result[radius] = np.hypot(dlon - xi, dlat - eta).max() * RAD2AS * 1e3

    return result


def table_rows():
    """
    Compute the comparison table.

    Returns
    -------
    rows : list of dict
        One row per (dec0, offset): position difference (mas), |theta|
        (arcsec), proper-motion error for PM_REF (uas/yr) and parallax
        displacement error for PI_REF (uas).
    """
    rows = []

    for dec0_deg in TABLE_DECS:
        dec0 = np.radians(dec0_deg)

        for offset_as in TABLE_OFFSETS:
            ra, dec = diagonal_star(offset_as, dec0)

            # Position difference between the two projections.
            xi_f, eta_f = flat_project(ra, dec, 0.0, dec0)
            xi_g, eta_g = gnomonic_project(ra, dec, 0.0, dec0)
            dpos_mas = np.hypot(xi_f - xi_g, eta_f - eta_g) * RAD2AS * 1e3

            # Rotation of local north, and the proper-motion error it causes.
            theta = north_angle(ra, dec, 0.0, dec0)
            pm_err_uas = PM_REF * abs(theta) * 1e3

            # Parallax displacement error for a PI_REF parallax.
            pi_err_uas = PI_REF * parallax_error_fraction(offset_as, dec0_deg) * 1e3

            rows.append(dict(dec0=dec0_deg, offset=offset_as, dpos=dpos_mas,
                             theta=abs(theta) * RAD2AS, pm_err=pm_err_uas, pi_err=pi_err_uas))

    return rows


def fmt_offset(offset_as):
    """
    Format an offset in arcsec as 10'' or 5'.

    Parameters
    ----------
    offset_as : float
        Offset in arcsec.

    Returns
    -------
    label : str
        LaTeX label.
    """
    if offset_as < 60:
        label = f"{offset_as:.0f}$''$"
    else:
        label = f"{offset_as / 60:.0f}$'$"

    return label


def fmt_num(value, zero_below=None):
    """
    Format a number with about three significant figures.

    Parameters
    ----------
    value : float
        Number to format.
    zero_below : float, optional
        If given, values below this are numerical noise and print as 0.

    Returns
    -------
    text : str
    """
    if zero_below is not None and abs(value) < zero_below:
        text = '0'
    elif abs(value) < 1e-3:
        text = '$<0.001$'
    elif abs(value) < 10:
        text = f"{value:.3g}"
    else:
        text = f"{value:,.0f}".replace(',', '{,}')

    return text


def write_table(rows):
    """
    Write the LaTeX tabular for the report.

    Parameters
    ----------
    rows : list of dict
        Output of table_rows().

    Returns
    -------
    None

    Notes
    -----
    The whole tabular is written, not just its rows: LaTeX's \\input cannot be
    used inside a tabular when the included rows contain \\midrule.
    """
    lines = [r'\begin{tabular}{llrrrr}',
             r'\toprule',
             r'$\delta_0$ & Offset & Position & North rotation & PM error & Parallax error \\',
             r' & & (mas) & (arcsec) & ($\mu$as\,yr$^{-1}$) & ($\mu$as) \\',
             r'\midrule']
    for k, row in enumerate(rows):
        # Rule between declination blocks.
        if k > 0 and row['dec0'] != rows[k - 1]['dec0']:
            lines.append(r'\midrule')
        dec_label = f"${row['dec0']:+.0f}^\\circ$" if k == 0 or row['dec0'] != rows[k - 1]['dec0'] else ''
        # At the equator local north is not rotated; tiny values there are finite-difference noise.
        lines.append(f"{dec_label} & {fmt_offset(row['offset'])} & {fmt_num(row['dpos'])} & "
                     f"{fmt_num(row['theta'], zero_below=1e-3)} & "
                     f"{fmt_num(row['pm_err'], zero_below=1e-3)} & {fmt_num(row['pi_err'])} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']

    with open(os.path.join(OUT_DIR, 'projection_table.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    return None


def write_numbers(rows, worst_expansion, pm_res, err_res, aber, sph):
    """
    Write \\newcommand macros for the numbers quoted in the report text.

    Parameters
    ----------
    rows : list of dict
        Output of table_rows().
    worst_expansion : float
        Output of check_expansions().
    pm_res : dict
        Output of pm_map().
    err_res : dict
        Output of error_maps().
    aber : dict
        Output of aberration_residuals().
    sph : dict
        Output of spherical_offset_differences().

    Returns
    -------
    None
    """
    # Largest errors in the simulated field (total of east and north).
    err_inner = err_res['r'] <= 200.0
    pos_tot = np.hypot(err_res['pos_e'], err_res['pos_n'])
    pm_tot = np.hypot(err_res['pm_e'], err_res['pm_n'])
    pi_tot = np.maximum(err_res['pi_e'], err_res['pi_n'])

    # Largest position difference in the simulated field, overall and within 200''.
    dtot = np.hypot(pm_res['dra'], pm_res['ddec'])
    inner = pm_res['r'] <= 200.0

    def pick(dec0, offset):
        return next(r for r in rows if r['dec0'] == dec0 and r['offset'] == offset)

    gc_10 = pick(-29.0, 10.0)
    gc_1m = pick(-29.0, 60.0)
    hi_1m = pick(60.0, 60.0)

    macros = {
        'PosDiffGCTen': f"{gc_10['dpos']:.2f}",
        'PosDiffGCArcmin': f"{gc_1m['dpos']:.1f}",
        'PosDiffHiArcmin': f"{hi_1m['dpos']:.1f}",
        'NorthRotGCArcmin': f"{gc_1m['theta']:.0f}",
        'PMErrGCArcmin': f"{gc_1m['pm_err']:.1f}",
        'PiErrGCArcmin': f"{gc_1m['pi_err']:.2f}",
        'CosDecGC': f"{np.cos(np.radians(-29.0)):.3f}",
        'ExpansionWorst': f"{worst_expansion * 100:.2f}",
        'PMRef': f"{PM_REF:g}",
        'PiRef': f"{PI_REF:g}",
        'PMMapN': f"{pm_res['n']}",
        'PMMapDt': f"{pm_res['dt']:g}",
        'PMMapMax': f"{dtot.max():.3f}",
        'PMMapMaxInner': f"{dtot[inner].max():.3f}",
        'PMMapPredResid': f"{pm_res['pred_resid'] * 1e3:.1g}",
        'ErrMapPosMax': f"{pos_tot.max():.0f}",
        'ErrMapPosMaxInner': f"{pos_tot[err_inner].max():.0f}",
        'ErrMapPMMax': f"{pm_tot.max():.1f}",
        'ErrMapPMMaxInner': f"{pm_tot[err_inner].max():.1f}",
        'ErrMapPiMax': f"{pi_tot.max():.1f}",
        'ErrMapPiMaxInner': f"{pi_tot[err_inner].max():.1f}",
        'AberRawOne': f"{aber[60.0]['raw']:.0f}",
        'AberRawFive': f"{aber[300.0]['raw']:.0f}",
        'AberRawThirty': f"{aber[1800.0]['raw']:.0f}",
        'AberAffOne': f"{aber[60.0]['affine']:.3f}",
        'AberAffFive': f"{aber[300.0]['affine']:.2f}",
        'AberAffThirty': f"{aber[1800.0]['affine']:.2f}",
        'AberQuadThirty': f"{aber[1800.0]['quad']:.3f}",
        'SphOffOne': f"{sph[60.0]:.3f}",
        'SphOffFive': f"{sph[300.0]:.2f}",
        'SphOffThirty': f"{sph[1800.0]:.0f}",
    }

    with open(os.path.join(OUT_DIR, 'report_numbers.tex'), 'w') as fh:
        for name, value in macros.items():
            fh.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")

    return None


def make_figure():
    """
    Two panels against offset: position difference, and local-north rotation.

    Returns
    -------
    None
    """
    offsets = np.logspace(0, np.log10(3600.0), 200)   # 1'' to 1 deg

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8))

    # -29 and +30 deg nearly coincide (similar |tan(dec0)|), so vary the line style too.
    styles = {-29.0: '-', 0.0: ':', 30.0: '--', 60.0: '-.'}
    colors = {-29.0: 'C0', 0.0: 'C1', 30.0: 'C2', 60.0: 'C3'}

    for dec0_deg in TABLE_DECS:
        dec0 = np.radians(dec0_deg)
        ra, dec = diagonal_star(offsets, dec0)

        # Position difference (mas).
        xi_f, eta_f = flat_project(ra, dec, 0.0, dec0)
        xi_g, eta_g = gnomonic_project(ra, dec, 0.0, dec0)
        dpos = np.hypot(xi_f - xi_g, eta_f - eta_g) * RAD2AS * 1e3

        # Local-north rotation (arcsec); zero to first order at the equator.
        theta = np.abs(north_angle(ra, dec, 0.0, dec0)) * RAD2AS

        label = f"$\\delta_0 = {dec0_deg:+.0f}^\\circ$"
        ls, color = styles[dec0_deg], colors[dec0_deg]
        ax1.loglog(offsets, dpos, ls=ls, color=color, label=label)

        # At the equator the rotation is zero; values below 1e-3 arcsec are noise.
        good = theta > 1e-3
        if np.any(good):
            ax2.loglog(offsets[good], theta[good], ls=ls, color=color, label=label)

    # Reference precision for AO astrometry.
    ax1.axhline(0.1, color='k', ls=':', lw=1)
    ax1.text(1.3, 0.13, '0.1 mas', fontsize=8)

    ax1.set_xlabel('Offset from field center (arcsec)')
    ax1.set_ylabel('Flat minus gnomonic position (mas)')
    ax2.set_xlabel('Offset from field center (arcsec)')
    ax2.set_ylabel('Rotation of local north (arcsec)')
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(True, which='major', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'projection_differences.pdf'))
    plt.close(fig)

    return None


def main():
    """
    Compute everything and write the report inputs.

    Returns
    -------
    None
    """
    # Second-order expansions used in the text.
    worst = check_expansions()
    print(f'Second-order expansions: worst relative error {worst:.2e}')
    assert worst < 0.02, 'second-order expansions do not match the exact differences'

    rows = table_rows()
    for row in rows:
        print(f"dec0={row['dec0']:+5.0f}  offset={row['offset']:6.0f}\"  dpos={row['dpos']:12.4f} mas  "
              f"theta={row['theta']:9.2f}\"  pm_err={row['pm_err']:8.3f} uas/yr  "
              f"pi_err={row['pi_err']:8.4f} uas")

    # Issue #22: proper motion propagated in the central plane versus each star's own.
    pm_res = pm_map()
    dtot = np.hypot(pm_res['dra'], pm_res['ddec'])
    print(f"PM map: {pm_res['n']} stars, dt={pm_res['dt']:g} yr, max diff {dtot.max():.4f} mas, "
          f"max within 200\" {dtot[pm_res['r'] <= 200].max():.4f} mas, "
          f"rotation-formula residual {pm_res['pred_resid'] * 1e3:.2e} uas")
    assert pm_res['pred_resid'] < 0.1 * dtot.max(), 'differences do not follow the rotation formula'

    # Instantaneous position, proper-motion and parallax errors in the same field.
    err_res = error_maps(pm_res)
    print(f"Error maps: max position {np.hypot(err_res['pos_e'], err_res['pos_n']).max():.2f} mas, "
          f"max PM {np.hypot(err_res['pm_e'], err_res['pm_n']).max():.3f} uas/yr, "
          f"max parallax {max(err_res['pi_e'].max(), err_res['pi_n'].max()):.3f} uas, "
          f"PM cross-check vs propagation map {err_res['pm_check'] * 1e3:.2e} uas")
    assert err_res['pm_check'] < 1e-3, 'PM errors disagree with the propagation map'

    # Differential annual aberration, and spherical offsets versus standard coordinates.
    aber = aberration_residuals()
    for radius, val in aber.items():
        print(f"Aberration, radius {radius:.0f}\": raw {val['raw']:.1f} mas, "
              f"after affine {val['affine']:.4f} mas, after quadratic {val['quad']:.2e} mas")
    sph = spherical_offset_differences()
    for radius, val in sph.items():
        print(f"Spherical offsets vs standard coordinates, {radius:.0f}\": {val:.4g} mas")

    write_table(rows)
    write_numbers(rows, worst, pm_res, err_res, aber, sph)
    make_figure()
    make_pm_map_figure(pm_res)
    make_error_map_figure(err_res)
    print('wrote projection_table.tex, report_numbers.tex, projection_differences.pdf, '
          'pm_propagation_map.pdf, error_maps.pdf')

    return None


if __name__ == '__main__':
    main()
