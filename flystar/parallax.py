# Parallax calculation module for motion models involving parallax
# Adapted from BAGLE's parallax.py

import math

import numpy as np
from joblib import Memory
import os
from astropy import units, units as u
from astropy.coordinates import SkyCoord, get_body_barycentric, get_body_barycentric_posvel, solar_system_ephemeris, \
    CartesianRepresentation
from astropy.time import Time

# Use the JPL ephemerides.
solar_system_ephemeris.set('jpl')

# Setup a parallax cache
try:
    cache_dir = os.environ['PARALLAX_CACHE_DIR']
except:
    cache_dir = os.path.dirname(__file__) + '/parallax_cache/'
cache_memory = Memory(cache_dir, verbose=0)
# Default cache size is 1 GB
cache_memory.reduce_size()

# @cache_memory.cache()
def parallax_in_direction(ra, dec, mjd, obsLocation='earth', pa=0.):
    """
    Calculate the parallax vector in a given direction following MulensModel.

    Parameters
    ----------
    RA : float or array-like
        Right Ascension in degrees. (J2000)
    Dec : float or array-like
        Declination in degrees. (J2000)
    mjd : float or array-like
        Modified Julian Date.
    obsLocation : str, optional
        Observer location, by default 'earth'.
    PA : float, optional
        Position angle in degrees (counterclockwise offset of the image y-axis from North), by default 0.

    Returns
    -------
    pvec : ndarray
        Parallax vector components, shape of (N_stars, 2, N_times), where the second dimension corresponds to the x or y components.
    """
    # Munge inputs into astropy format.
    # times = Time(mjd + 2400000.5, format='jd', scale='tdb')
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    mjd = np.atleast_1d(mjd)
    pa = np.atleast_1d(pa)
    times = Time(mjd, format='mjd', scale='tdb')  # convert to TDB
    coord = SkyCoord(ra, dec, unit=(units.deg, units.deg))  # Shape (N_stars,)

    directions = coord.cartesian.xyz.value.T # Shape (N_stars, 3)
    north = np.array([0., 0., 1.])
    # Cross product of each star with north vector
    _east_projected = np.cross(north, directions)
    _east_projected /= np.linalg.norm(_east_projected, axis=1)[:, np.newaxis]   # Shape (N_stars, 3)
    _north_projected = np.cross(directions, _east_projected)
    _north_projected /= np.linalg.norm(_north_projected, axis=1)[:, np.newaxis] # Shape (N_stars, 3)

    obs_pos = get_observer_barycentric(obsLocation, times)  # Shape (N_times,)
    sun_pos = get_body_barycentric(body='sun', time=times)  # Shape (N_times,)

    sun_obs_pos = sun_pos - obs_pos

    pos = sun_obs_pos.xyz.T.to(units.au).value  # Shape (N_times, 3)
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
    using JPL emphemerides through the Horizon app.

    The ephemeris is queried at a decimated time step set by min_ephem_step
    (def=1 day) that must be 1 day or larger. The positions
    (and optionally velocities) are then interpolated onto the desired
    time array.

    Inputs
    ------
    body : str
        The name of the Solar System body. Must use the JPL Horizon
        naming scheme.

    times : astropy.time.Time array
        Array of times (astropy.time.core.Time) objects at which to
        fetch the position of the specified Solar System body.

    Optional Inputs
    ---------------
    min_ephem_step : int
        Minimum time step to query JPL in days. Must not be <1 and must
        be in integer days.

    veloctiy : bool
        If true, return both position and velocity vectors over time.

    Return
    ------
    coord : astropy.coordinates.CartesianRepresentation
        The xyz coordinates in the plane of the Solar System at the
        input times.
    """

    if body in solar_system_ephemeris.bodies:
        if velocity:
            obs_pos, obs_vel = get_body_barycentric_posvel(body=body, time=times)
        else:
            obs_pos = get_body_barycentric(body=body, time=times)
    else:
        # Figure out a cadence for the ephemerides, not smaller than 1 day.
        dt = np.median(np.diff(times)).jd
        if dt < min_ephem_step:
            dt = min_ephem_step

        # Get the date range, add some padding on each side.
        t_min = times.min()
        t_max = times.max()
        t_min.format = 'iso'
        t_max.format = 'iso'
        t_min = str(t_min - dt*u.day).split()[0]
        t_max = str(t_max + dt*u.day).split()[0]
        step = f'{dt:.0f}d'

        # Fetch the Horizons ephemeris.
        from astroquery.jplhorizons import Horizons
        obj = Horizons(id=body, epochs={'start':t_min, 'stop':t_max, 'step':step})
        obj_data = obj.vectors()

        ephem_jd = obj_data['datetime_jd']

        # Interpolate to the actual time array.
        obj_x_at_t = np.interp(times.jd, ephem_jd, obj_data['x'].to('km')) * u.km
        obj_y_at_t = np.interp(times.jd, ephem_jd, obj_data['y'].to('km')) * u.km
        obj_z_at_t = np.interp(times.jd, ephem_jd, obj_data['z'].to('km')) * u.km

        if velocity:
            obj_vx_at_t = np.interp(times.jd, ephem_jd, obj_data['vx'].to('km/s')) * u.km / u.s
            obj_vy_at_t = np.interp(times.jd, ephem_jd, obj_data['vy'].to('km/s')) * u.km / u.s
            obj_vz_at_t = np.interp(times.jd, ephem_jd, obj_data['vz'].to('km/s')) * u.km / u.s

            obs_vel = CartesianRepresentation(obj_vx_at_t, obj_vy_at_t, obj_vz_at_t)

        obs_pos = CartesianRepresentation(obj_x_at_t, obj_y_at_t, obj_z_at_t)

    if velocity:
        return (obs_pos, obs_vel)
    else:
        return obs_pos