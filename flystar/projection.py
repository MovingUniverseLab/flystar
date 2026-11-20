"""
Projections between the sky (ICRS RA, Dec) and FlyStar's plane of arcsec
offsets, with the proper-motion Jacobians, propagation on the sky and the
bright-star proper-motion correction for Gaia (E)DR3.

The equations follow docs/plans/tangent_plane/tangent_plane_report.tex
(Sections 2-8 and 11). Conventions:

* Sky positions (ra, dec) are in degrees, ICRS.
* Plane coordinates (xi, eta) are the standard coordinates of the chosen
  projection about the reference (tangent) point, in arcsec, with +xi toward
  east and +eta toward north. FlyStar's x is west-positive, so callers use
  ``x = -xi`` and ``y = eta``.
* Proper motions are (pmra, pmdec) = (mu_alpha*, mu_delta), along each star's
  own local east and north, as in Gaia.

Projections:

``'gnomonic'``
    The tangent-plane (FITS TAN) projection, Eq. (gnom) of the report. This
    is what an ideal camera records. The default.
``'flat'``
    Scaled equirectangular: xi = (ra - ra_ref) cos(dec_ref),
    eta = dec - dec_ref (Eq. flat).
``'flat_legacy'``
    TEMPORARY, removed before release. The same position math as ``'flat'``
    (without wrapping the RA difference at 0/360 deg), kept so that the Gaia
    ingestion functions can reproduce FlyStar's old outputs bit for bit.
    The two flat modes differ only in how the callers treat proper motions
    and errors.
"""
import numpy as np

__all__ = ['PROJECTIONS', 'sky_to_plane', 'plane_to_sky', 'spherical_offsets',
           'jacobian', 'unit_vectors', 'parallax_shift_vector',
           'propagate_on_sky', 'correct_bright_pm']

#: Allowed values of the ``projection`` keyword.
#: 'flat_legacy' is temporary and will be removed before release.
PROJECTIONS = ('gnomonic', 'flat', 'flat_legacy')

# Conversions.
DEG2AS = 3600.0
AS2RAD = np.pi / 180.0 / 3600.0
MAS2RAD = AS2RAD / 1.0e3


def _check_projection(projection):
    """
    Raise a ValueError for an unknown projection name.
    """
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}, got {projection!r}")


def _wrap180(angle_deg):
    """
    Wrap an angle in degrees into [-180, 180), leaving values already in
    that range bit-for-bit unchanged.
    """
    angle_deg = np.asarray(angle_deg, dtype=float)
    return np.where(angle_deg >= 180.0, angle_deg - 360.0,
                    np.where(angle_deg < -180.0, angle_deg + 360.0, angle_deg))


def unit_vectors(ra, dec):
    """
    Direction, local east and local north unit vectors at (ra, dec).

    Parameters
    ----------
    ra, dec : float or array-like
        Sky coordinates in degrees.

    Returns
    -------
    u, e, n : numpy array
        Each of shape ``np.broadcast(ra, dec).shape + (3,)``: the unit vector
        toward the star, toward local east and toward local north
        (Eq. triad of the report).
    """
    a = np.radians(np.asarray(ra, dtype=float))
    d = np.radians(np.asarray(dec, dtype=float))
    a, d = np.broadcast_arrays(a, d)

    ca, sa = np.cos(a), np.sin(a)
    cd, sd = np.cos(d), np.sin(d)

    u = np.stack([cd * ca, cd * sa, sd], axis=-1)
    e = np.stack([-sa, ca, np.zeros_like(a)], axis=-1)
    n = np.stack([-sd * ca, -sd * sa, cd], axis=-1)

    return u, e, n


def _vector_to_radec(vec):
    """
    RA in [0, 360) and Dec (both degrees) of (not necessarily unit) vectors
    of shape (..., 3).
    """
    vec = np.asarray(vec, dtype=float)
    ra = np.degrees(np.arctan2(vec[..., 1], vec[..., 0])) % 360.0
    dec = np.degrees(np.arctan2(vec[..., 2], np.hypot(vec[..., 0], vec[..., 1])))

    return ra, dec


def sky_to_plane(ra, dec, ra_ref, dec_ref, projection='gnomonic'):
    """
    Project sky positions onto the plane about a reference (tangent) point.

    Parameters
    ----------
    ra, dec : float or array-like
        ICRS coordinates of the stars in degrees.
    ra_ref, dec_ref : float
        ICRS coordinates of the projection center (tangent point) in degrees.
    projection : {'gnomonic', 'flat', 'flat_legacy'}, optional
        Projection to use (see the module docstring). Default 'gnomonic'.
        'flat_legacy' is temporary and will be removed before release.

    Returns
    -------
    xi, eta : float or numpy array
        Standard coordinates of the chosen projection about the reference,
        in arcsec, east (xi) and north (eta) positive: Eq. (gnom) of the
        report for 'gnomonic', Eq. (flat) for the flat modes. FlyStar's
        west-positive x is ``-xi``.

    Notes
    -----
    For 'gnomonic', stars 90 deg or more from the tangent point cannot be
    projected and are returned as NaN. For 'flat', the RA difference is
    wrapped into [-180, 180) deg; 'flat_legacy' does not wrap it, as the old
    code did not.
    """
    _check_projection(projection)

    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)

    if projection == 'flat_legacy':
        # TEMPORARY (removed before release): exactly the arithmetic of the
        # old prepare_gaia_for_flystar / project_gaia, so results are
        # bit-for-bit identical.
        xi = (ra - ra_ref) * np.cos(np.radians(dec_ref)) * 3600.0
        eta = (dec - dec_ref) * 3600.0
        return xi, eta

    if projection == 'flat':
        # Same expression as 'flat_legacy' when no wrap is needed.
        xi = _wrap180(ra - ra_ref) * np.cos(np.radians(dec_ref)) * 3600.0
        eta = (dec - dec_ref) * 3600.0
        return xi, eta

    # Gnomonic, Eq. (gnom).
    a = np.radians(ra)
    d = np.radians(dec)
    a0 = np.radians(ra_ref)
    d0 = np.radians(dec_ref)
    da = a - a0

    denom = np.sin(d) * np.sin(d0) + np.cos(d) * np.cos(d0) * np.cos(da)
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = np.cos(d) * np.sin(da) / denom
        eta = (np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(da)) / denom
    bad = denom <= 0
    xi = np.where(bad, np.nan, xi) / AS2RAD
    eta = np.where(bad, np.nan, eta) / AS2RAD

    return xi, eta


def spherical_offsets(ra, dec, ra_ref, dec_ref):
    """
    Spherical offsets of sky positions from a reference position.

    These are the longitude and latitude in a frame rotated so that the
    reference is at the origin and its north is +latitude: the same as
    astropy's ``SkyOffsetFrame`` (with zero rotation) and
    ``SkyCoord.spherical_offsets_to``. The longitude offset is NOT multiplied
    by the cosine of the offset latitude.

    Parameters
    ----------
    ra, dec : float or array-like
        ICRS coordinates in degrees.
    ra_ref, dec_ref : float or array-like
        Reference position in degrees.

    Returns
    -------
    dlon, dlat : float or numpy array
        Offsets in arcsec, east and north positive.
    """
    u, _, _ = unit_vectors(ra, dec)
    u0, e0, n0 = unit_vectors(ra_ref, dec_ref)

    x = np.sum(u * u0, axis=-1)
    y = np.sum(u * e0, axis=-1)
    z = np.sum(u * n0, axis=-1)

    dlon = np.degrees(np.arctan2(y, x)) * DEG2AS
    dlat = np.degrees(np.arctan2(z, np.hypot(x, y))) * DEG2AS

    return dlon, dlat


def plane_to_sky(xi, eta, ra_ref, dec_ref, projection='gnomonic'):
    """
    Invert the projection: plane coordinates to RA and Dec, plus spherical
    offsets from the reference.

    Parameters
    ----------
    xi, eta : float or array-like
        Standard coordinates in arcsec, east and north positive (FlyStar's
        ``-x`` and ``y`` for a north-up arcsec frame).
    ra_ref, dec_ref : float
        ICRS coordinates of the projection center (tangent point) in degrees.
    projection : {'gnomonic', 'flat', 'flat_legacy'}, optional
        Projection that built the plane. Default 'gnomonic'. 'flat' and
        'flat_legacy' (temporary, removed before release) use the same
        inverse, Eq. (flatinv); 'gnomonic' uses Eq. (gnominv).

    Returns
    -------
    ra, dec : float or numpy array
        ICRS coordinates in degrees, RA in [0, 360).
    dlon, dlat : float or numpy array
        Spherical offsets from (ra_ref, dec_ref) in arcsec, east and north
        positive, as defined by astropy's ``SkyOffsetFrame`` and
        ``SkyCoord.spherical_offsets_to`` (see :func:`spherical_offsets`).

    Notes
    -----
    Three caveats about the offsets:

    * **They do not round-trip.** ``plane_to_sky`` applied to the output of
      :func:`sky_to_plane` returns the same RA and Dec, but its spherical
      offsets are not the (xi, eta) that went in. They differ at third order
      in the distance from the reference (up to about 0.002 mas at 1',
      0.2 mas at 5' and 46 mas at 30' for gnomonic). Do not compare them with FlyStar's
      ``x0``, ``y0`` or pass them back into ``sky_to_plane``; RA and Dec are
      the quantities that round-trip.
    * **Proper motions stay along local east and north.** The offset frame's
      axes at a star are rotated from that star's own east and north by about
      the meridian-convergence angle. Report proper motions as Gaia does,
      (mu_alpha*, mu_delta) in each star's own axes, not along the offset
      axes.
    * **Which definition.** Spherical offsets have two common definitions.
      These are astropy's: the longitude offset is not multiplied by the
      cosine of the offset latitude.
    """
    _check_projection(projection)

    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)

    if projection in ('flat', 'flat_legacy'):
        # Eq. (flatinv).
        ra = (ra_ref + xi / 3600.0 / np.cos(np.radians(dec_ref))) % 360.0
        dec = dec_ref + eta / 3600.0
    else:
        # Gnomonic: P = u0 + xi e0 + eta n0, u = P / |P|. Equivalent to
        # Eq. (gnominv) and free of the rho = 0 special case.
        u0, e0, n0 = unit_vectors(ra_ref, dec_ref)
        x = xi * AS2RAD
        y = eta * AS2RAD
        p = u0 + x[..., np.newaxis] * e0 + y[..., np.newaxis] * n0
        ra, dec = _vector_to_radec(p)

    dlon, dlat = spherical_offsets(ra, dec, ra_ref, dec_ref)

    return ra, dec, dlon, dlat


def jacobian(ra, dec, ra_ref, dec_ref, projection='gnomonic'):
    """
    Jacobian of the projection with respect to motions on the sky.

    Parameters
    ----------
    ra, dec : float or array-like
        ICRS coordinates of the stars in degrees (where J is evaluated).
    ra_ref, dec_ref : float
        Projection center in degrees.
    projection : {'gnomonic', 'flat', 'flat_legacy'}, optional
        Default 'gnomonic'.

    Returns
    -------
    jac : numpy array
        Shape ``np.broadcast(ra, dec).shape + (2, 2)``. Maps a small motion
        (d alpha*, d delta) along the star's own east and north to the plane
        motion (d xi, d eta), both in the same angular units:
        J_g of Eq. (jgnom) for 'gnomonic' and J_f of Eq. (jflat) for the
        flat modes. Proper motions map as (xi_dot, eta_dot) = J (pmra, pmdec)
        and covariances as J C J^T; FlyStar's (vx, vy) = F J (pmra, pmdec)
        with F = diag(-1, 1).

    Notes
    -----
    'flat_legacy' returns the true Jacobian of its position math, J_f. The
    old code ignored it (J = I); the legacy code paths in the callers do the
    same.
    """
    _check_projection(projection)

    d = np.radians(np.asarray(dec, dtype=float))
    a = np.radians(np.asarray(ra, dtype=float))
    a, d = np.broadcast_arrays(a, d)
    d0 = np.radians(dec_ref)

    if projection in ('flat', 'flat_legacy'):
        j11 = np.cos(d0) / np.cos(d)
        zero = np.zeros_like(d)
        one = np.ones_like(d)
        jac = np.stack([np.stack([j11, zero], axis=-1),
                        np.stack([zero, one], axis=-1)], axis=-2)
        return jac

    da = a - np.radians(ra_ref)
    denom = np.sin(d) * np.sin(d0) + np.cos(d) * np.cos(d0) * np.cos(da)
    xi = np.cos(d) * np.sin(da) / denom
    eta = (np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(da)) / denom
    k = np.cos(d) * np.sin(d0) - np.sin(d) * np.cos(d0) * np.cos(da)

    j11 = np.cos(da) + xi * np.cos(d0) * np.sin(da)
    j12 = -np.sin(d) * np.sin(da) - xi * k
    j21 = np.sin(da) * (np.sin(d0) + eta * np.cos(d0))
    j22 = np.sin(d) * np.sin(d0) * np.cos(da) + np.cos(d) * np.cos(d0) - eta * k

    jac = np.stack([np.stack([j11, j12], axis=-1),
                    np.stack([j21, j22], axis=-1)], axis=-2)
    jac = jac / denom[..., np.newaxis, np.newaxis]

    return jac


def parallax_shift_vector(ra, dec, obs_pos_au):
    """
    Parallax shift vector s(t) of Eq. (pfac), with the barycenter as the
    reference point.

    Parameters
    ----------
    ra, dec : float or array-like
        ICRS coordinates of the stars in degrees.
    obs_pos_au : array-like
        Barycentric position of the observer in au, shape (3,) or
        ``(N, 3)`` (ICRS axes).

    Returns
    -------
    s : numpy array
        Shape ``(..., 3)``. A star of parallax pi is displaced by ``pi * s``
        (radians, for pi in radians). Its components along the star's east
        and north are the parallax factors (P_e, P_n).
    """
    u, _, _ = unit_vectors(ra, dec)
    r = np.asarray(obs_pos_au, dtype=float)
    s = -r + u * np.sum(u * r, axis=-1)[..., np.newaxis]

    return s


def propagate_on_sky(ra, dec, pmra, pmdec, dt, parallax=None, s=None):
    """
    Move stars on the sky by their proper motion (and optionally parallax).

    Implements Eq. (prop) of the report: the standard astrometric model with
    the radial-velocity terms dropped, as in astropy's
    ``SkyCoord.apply_space_motion`` for sources with no distance or radial
    velocity.

    Parameters
    ----------
    ra, dec : float or array-like
        ICRS coordinates in degrees at the catalog epoch.
    pmra, pmdec : float or array-like
        Proper motions mu_alpha* (already times cos(dec)) and mu_delta in
        mas/yr.
    dt : float or array-like
        Time from the catalog epoch to the target epoch, in years.
    parallax : float or array-like, optional
        Parallax in mas. Used only together with ``s``.
    s : array-like, optional
        Parallax shift vector(s) at the target epoch, shape (3,) or (N, 3),
        in au (e.g. from :func:`parallax_shift_vector`, or ``-r_obs / 1 au``;
        a component along the star's direction has no first-order effect).

    Returns
    -------
    ra, dec : float or numpy array
        ICRS coordinates in degrees at the target epoch, RA in [0, 360).
    """
    u, e, n = unit_vectors(ra, dec)
    pmra = np.asarray(pmra, dtype=float)
    pmdec = np.asarray(pmdec, dtype=float)
    dt = np.asarray(dt, dtype=float)

    du = (dt * pmra * MAS2RAD)[..., np.newaxis] * e + (dt * pmdec * MAS2RAD)[..., np.newaxis] * n
    vec = u + du

    if (parallax is not None) and (s is not None):
        plx = np.asarray(parallax, dtype=float) * MAS2RAD
        vec = vec + plx[..., np.newaxis] * np.asarray(s, dtype=float)

    return _vector_to_radec(vec)


def correct_bright_pm(ra, dec, pmra, pmdec, gmag, data_release='DR3'):
    """
    Correct the proper motions of bright Gaia (E)DR3 stars for the
    magnitude-dependent spin of their reference frame.

    Not yet available: the spin table has not been filled in.
    """
    raise NotImplementedError("correct_bright_pm is not implemented yet.")
