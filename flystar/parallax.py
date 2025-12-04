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

@cache_memory.cache()
def parallax_in_direction(RA, Dec, mjd, obsLocation='earth', PA=0):
    """
    | R.A. in degrees. (J2000)
    | Dec. in degrees. (J2000)
    | MJD
    | PA in degrees. (counterclockwise offset of the image y-axis from North)

    Equations following MulensModel.
    """
    #print('parallax_in_direction: len(t) = ', len(mjd))

    # Munge inputs into astropy format.
    times = Time(mjd + 2400000.5, format='jd', scale='tdb')
    coord = SkyCoord(RA, Dec, unit=(units.deg, units.deg))

    direction = coord.cartesian.xyz.value
    north = np.array([0., 0., 1.])
    _east_projected = np.cross(north, direction) / np.linalg.norm(np.cross(north, direction))
    _north_projected = np.cross(direction, _east_projected) / np.linalg.norm(np.cross(direction, _east_projected))

    obs_pos = get_observer_barycentric(obsLocation, times)
    sun_pos = get_body_barycentric(body='sun', time=times)

    sun_obs_pos = sun_pos - obs_pos

    pos = sun_obs_pos.xyz.T.to(units.au)

    e = np.dot(pos, _east_projected)
    n = np.dot(pos, _north_projected)
    
    # Rotate frame e,n->x,y accounting for PA
    PA_rad = np.pi/180.0 * PA
    x = -e.value*np.cos(PA_rad) + n.value*np.sin(PA_rad)
    y =  e.value*np.sin(PA_rad) + n.value*np.cos(PA_rad)
    
    pvec = np.array([x, y]).T
    
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


