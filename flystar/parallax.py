# Parallax calculation module for motion models involving parallax
# Adapted from BAGLE's parallax.py

import os
import numpy as np
from joblib import Memory
from astropy.time import Time
from astropy import units, units as u
from astropy.coordinates import SkyCoord, get_body_barycentric, get_body_barycentric_posvel, solar_system_ephemeris, \
    CartesianRepresentation

# FIXME: Do we still need this?
# Setup a parallax cache
try:
    cache_dir = os.environ['PARALLAX_CACHE_DIR']
except:
    cache_dir = os.path.dirname(__file__) + '/parallax_cache/'
cache_memory = Memory(cache_dir, verbose=0)
# Default cache size is 1 GB
cache_memory.reduce_size()

# JPL ephemeris used by astropy for the observer's barycentric position.
# DE440s matches the DE440/441 barycenter that JPL Horizons uses for
# spacecraft (astropy's 'jpl' default, DE430, is offset by ~116 km).
EPHEMERIS = 'de440s'

# @cache_memory.cache()
def parallax_in_direction(ra, dec, mjd, obsLocation='earth', pa=0.):
    """
    Calculate the parallax vector (parallax factors) in a given direction
    following MulensModel.

    The parallax factors are the east and north components of
    s(t) = -r_obs(t) / (1 au), with r_obs the barycentric position of the
    observer, i.e. Eq. (pfac) of docs/plans/tangent_plane with the Solar
    System barycenter as the reference point, as Gaia and Hipparcos use. A
    star with parallax pi is displaced by pi * pvec from its barycentric
    position, so the x0, y0 of a Parallax fit are positions as seen from the
    barycenter, the same as Gaia's ra, dec. (Earlier versions measured the
    observer from the Sun, which made x0, y0 heliocentric; the two differ by
    pi times the Sun's barycentric offset, <~0.01 au.)

    Parameters
    ----------
    RA : float or array-like
        Right Ascension in degrees (ICRS).
    Dec : float or array-like
        Declination in degrees (ICRS).
    mjd : float or array-like
        Modified Julian Date(s), **in the TDB scale**. Not converted; see the
        note in the body.
        Modified Julian Date.
    obsLocation : str, optional
        Observer location, by default 'earth'.
    PA : float, optional
        Position angle in degrees (counterclockwise offset of the image y-axis from North), by default 0.

    Returns
    -------
    pvec : ndarray
        Parallax vector components in au, shape of (N_stars, 2, N_times), where the second dimension corresponds to the x or y components
        (x west-positive for PA = 0).
    """
    # Munge inputs into astropy format.
    # times = Time(mjd + 2400000.5, format='jd', scale='tdb')
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    mjd = np.atleast_1d(mjd)
    pa = np.atleast_1d(pa)
    # Declares the incoming MJD to be TDB -- it does NOT convert. Callers
    # must hand over a TDB MJD; passing a UTC one silently shifts every
    # epoch by TDB-UTC (69.184 s as of 2026).
    times = Time(mjd, format='mjd', scale='tdb')
    coord = SkyCoord(ra, dec, unit=(units.deg, units.deg))  # Shape (N_stars,)

    directions = coord.cartesian.xyz.value.T # Shape (N_stars, 3)
    north = np.array([0., 0., 1.])
    # Cross product of each star with north vector
    _east_projected = np.cross(north, directions)
    _east_projected /= np.linalg.norm(_east_projected, axis=1)[:, np.newaxis]   # Shape (N_stars, 3)
    _north_projected = np.cross(directions, _east_projected)
    _north_projected /= np.linalg.norm(_north_projected, axis=1)[:, np.newaxis] # Shape (N_stars, 3)

    obs_pos = get_observer_barycentric(obsLocation, times)  # Shape (N_times,)

    # The reference point is the Solar System barycenter (r_ref = 0 in
    # Eq. pfac of docs/plans/tangent_plane), as in Gaia and Hipparcos, so the
    # star is displaced along -r_obs. (Earlier versions used the Sun,
    # sun_pos - obs_pos, which made x0, y0 heliocentric.)
    ref_obs_pos = -obs_pos

    pos = ref_obs_pos.xyz.T.to(units.au).value  # Shape (N_times, 3)
    # Broadcast pos to (N_stars, 3, N_times) and take dot product with east and north unit vectors to get components in those directions.
    pos = np.broadcast_to(pos.T, (directions.shape[0], 3, pos.shape[0])) # Shape (N_stars, 3, N_times)

    e = np.einsum('sdt,sd->st', pos, _east_projected)    # Shape (N_stars, N_times)
    n = np.einsum('sdt,sd->st', pos, _north_projected)   # Shape (N_stars, N_times)

    # Rotate frame e,n->x,y accounting for PA
    pa = np.deg2rad(pa) # shape (N_stars,)
    x = -e * np.cos(pa[:, np.newaxis]) + n * np.sin(pa[:, np.newaxis])  # Shape (N_stars, N_times)
    y =  e * np.sin(pa[:, np.newaxis]) + n * np.cos(pa[:, np.newaxis])  # Shape (N_stars, N_times)
    # pvec Shape (N_stars, 2, N_times)
    pvec = np.stack((x, y), axis=1)
    return pvec


def get_observer_barycentric(body, times, min_ephem_step=1, velocity=False):
    """
    Get the barycentric position of a satellite or other Solar System body
    in the ICRS (equatorial) frame.

    Bodies known to astropy's JPL ephemeris (e.g. 'earth', 'moon') are
    evaluated directly with astropy using the module-level ``EPHEMERIS``
    (DE440s), which shares its barycenter with the DE441 ephemeris behind
    Horizons. Anything else (e.g. a spacecraft such as
    JWST, Horizons ID '-170') is fetched from JPL Horizons as barycentric
    (location '@0'), ICRF equatorial (refplane 'earth') vectors on a fixed
    grid of ``min_ephem_step`` days, then cubic-spline interpolated onto the
    requested times.

    Parameters
    ----------
    body : str
        The name of the Solar System body. Must use the JPL Horizons
        naming scheme for bodies that astropy does not know.
    times : astropy.time.Time array
        Times at which to fetch the position of the body. Converted to TDB
        before matching against the Horizons grid.
    min_ephem_step : int, optional
        Horizons query step in integer days, by default 1. Must be >= 1. The
        step is fixed and does not depend on the spacing of ``times``.
    velocity : bool, optional
        If True, return both position and velocity vectors, by default False.

    Returns
    -------
    obs_pos : astropy.coordinates.CartesianRepresentation, shape = [N_times]
        Barycentric ICRS xyz position at the input times.
    obs_vel : astropy.coordinates.CartesianRepresentation, shape = [N_times]
        Barycentric ICRS xyz velocity. Only returned if ``velocity`` is True.

    Notes
    -----
    Earlier versions queried Horizons at the median spacing of ``times``
    (months for typical astrometry), linearly interpolated across the orbit,
    and used Horizons' defaults for ``vectors()``: Sun-centred and in the
    ecliptic plane. That gave spacecraft parallax factors that were wrong by
    up to a factor of ~2 (e.g. for JWST).
    """
    # Bodies that astropy's JPL ephemeris knows (without changing the
    # caller's global ephemeris setting).
    with solar_system_ephemeris.set(EPHEMERIS):
        astropy_bodies = solar_system_ephemeris.bodies

    if body in astropy_bodies:
        # Evaluate the DE440s ephemeris directly with astropy.
        if velocity:
            obs_pos, obs_vel = get_body_barycentric_posvel(body=body, time=times,
                                                           ephemeris=EPHEMERIS)
        else:
            obs_pos = get_body_barycentric(body=body, time=times, ephemeris=EPHEMERIS)
    else:
        from scipy.interpolate import CubicSpline
        from astroquery.jplhorizons import Horizons

        # Fixed query step in whole days, never coarser than requested.
        step_days = int(min_ephem_step)
        if step_days < 1:
            raise ValueError(f'min_ephem_step must be >= 1 day, got {min_ephem_step}')

        # Horizons vector tables are tabulated in TDB; shape of t_jd: [N_times]
        t_jd = np.atleast_1d(times.tdb.jd)

        # Pad the range by a few steps so the spline is well supported at both
        # ends, and snap to whole days so the grid starts at 0h TDB.
        pad = 3 * step_days
        jd_start = np.floor(t_jd.min() - pad - 0.5) + 0.5
        jd_stop = np.ceil(t_jd.max() + pad - 0.5) + 0.5
        t_start = Time(jd_start, format='jd', scale='tdb').iso.split()[0]
        t_stop = Time(jd_stop, format='jd', scale='tdb').iso.split()[0]

        # Fetch barycentric (@0), ICRF equatorial (refplane='earth') vectors.
        obj = Horizons(id=body, location='@0',
                       epochs={'start': t_start, 'stop': t_stop, 'step': f'{step_days}d'})
        obj_data = obj.vectors(refplane='earth')

        ephem_jd = np.asarray(obj_data['datetime_jd'])

        # Spline each position component onto the requested times (km).
        xyz = [CubicSpline(ephem_jd, obj_data[c].to(u.km).value)(t_jd)
               for c in ('x', 'y', 'z')]
        obs_pos = CartesianRepresentation(*xyz, unit=u.km)

        if velocity:
            # Same for the velocity components (km/s).
            vxyz = [CubicSpline(ephem_jd, obj_data[c].to(u.km / u.s).value)(t_jd)
                    for c in ('vx', 'vy', 'vz')]
            obs_vel = CartesianRepresentation(*vxyz, unit=u.km / u.s)

    if velocity:
        return (obs_pos, obs_vel)
    else:
        return obs_pos