"""Newtonian Kepler solver and ``orbits.dat`` reader.

The sky calculation is a port of the gcwork ``Orbit.kep2xyz`` /
``eccen_anomaly`` used by ``orbits_jlu_python_gcwork_2024-10-03``.
It is Newtonian only: no GR periapse advance and no relativistic
redshift. Index 0 of each returned vector is east, 1 is north, and
2 is the line of sight, matching that code.

Positions use the Gaussian axis, not G or Msun. Those two
constants, with the AU and the Julian year, convert only the
acceleration. The mas-per-arcsec factor converts velocity and
acceleration, and ``semimajor_axis_mas``.

The FlyStar frame is applied by ``Orbit.model``, not here:
``x = -east``, ``y = +north``.
"""

import warnings

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.table import Table, Column


# From astropy at import, then stored as plain floats. The hot path
# must not build Quantity objects. G and Msun are cgs. The AU is in
# cm. The year is the Julian year (365.25 d). mas-per-arcsec replaces
# the literal 1000. Positions do not use G or Msun.
_G_CGS = float(const.G.cgs.value)
_MSUN_G = float(const.M_sun.cgs.value)
_CM_IN_AU = float(const.au.cgs.value)
_SEC_IN_YR = float((1 * u.yr).to(u.s).value)
_MAS_PER_ARCSEC = float((1 * u.arcsec).to(u.mas).value)

# Nine whitespace-separated fields, no header. The file itself is
# unchanged. ``a`` is the semi-major axis (the old name was ``A``)
# and ``search`` is the last field. Both are parsed so a short line
# fails, then dropped. They are not catalog columns.
_ORBIT_FILE_FIELDS = (
    'name', 'P', 'a', 't0', 'e', 'i', 'Omega', 'omega', 'search',
)
_ELEMENT_COLUMNS = (
    'orb_P', 'orb_t0', 'orb_e', 'orb_i', 'orb_Omega', 'orb_omega',
)


def semimajor_axis_mas(period_yr, mass_msun, dist_pc):
    """Semi-major axis in milliarcseconds.

    Parameters
    ----------
    period_yr : float or ndarray
        Period in years. Catalog values are Julian years, not
        Gaussian years. See Notes.
    mass_msun : float
        Central mass in solar masses.
    dist_pc : float
        Distance in parsecs.

    Returns
    -------
    a_mas : float or ndarray
        Semi-major axis in milliarcseconds. Same shape as ``period_yr``.

    Notes
    -----
    ``a_AU = (P**2 * M)**(1/3)`` is exact only when ``P`` is in
    Gaussian years. A Gaussian year is ``2*pi/k``, where ``k`` is the
    Gaussian gravitational constant, about 365.2569 days. Periods
    passed here are Julian years, 365.25 days. Putting a Julian
    period into the Gaussian formula overestimates ``a`` by about
    1.3e-5.

    This uses the definition for a year = 365.25 days. It is
    not ``(G M P**2 / 4 pi**2)`` with ``_G_CGS`` and ``_MSUN_G``.
    That physical axis is smaller by about 1.3e-5. 
    """
    period_yr = np.asarray(period_yr, dtype=float)
    # Gaussian-year formula, applied to a Julian period on purpose.
    # See Notes. Do not replace this with G*M.
    a_au = (period_yr**2 * float(mass_msun))**(1.0 / 3.0)
    a_mas = a_au / float(dist_pc) * _MAS_PER_ARCSEC

    return a_mas


def _kepler_residual(eccentric_anomaly, mean_anomaly, ecc):
    """Residual of Kepler's equation, ``E - e sin E - M``.

    Parameters
    ----------
    eccentric_anomaly, mean_anomaly : float
        Angles in radians. They must already share one 2π convention.
    ecc : float
        Eccentricity.

    Returns
    -------
    residual : float
        Radians. Zero at the eccentric anomaly that belongs to
        ``mean_anomaly``.
    """
    residual = (
        float(eccentric_anomaly)
        - float(ecc) * np.sin(eccentric_anomaly)
        - float(mean_anomaly)
    )

    return residual


def _fold_two_pi(angle, reference):
    """Shift ``angle`` by multiples of 2π onto the turn nearest ``reference``.

    Parameters
    ----------
    angle, reference : float
        Radians.

    Returns
    -------
    folded : float
        ``angle - 2π * round((angle - reference) / 2π)``.
    """
    turns = np.round((angle - reference) / (2.0 * np.pi))
    folded = float(angle) - float(turns) * (2.0 * np.pi)

    return folded


def _bisect_kepler(mean_anomaly, ecc, thresh):
    """Eccentric anomaly by bisection.

    Parameters
    ----------
    mean_anomaly : float
        Mean anomaly in radians, on ``(-π, π]``.
    ecc : float
        Eccentricity, ``0 <= ecc < 1``.
    thresh : float
        Residual, in radians, that counts as converged.

    Returns
    -------
    eccentric_anomaly : float
        Root of ``E - e sin E - M``, in radians.

    Notes
    -----
    ``|E - M| = |e sin E| <= e``, so the root lies in
    ``[M - e, M + e]``. The residual is strictly increasing, so
    the bracket cannot be escaped. This is the fallback when a
    Newton step oscillates or runs away.
    """
    mean_anomaly = float(mean_anomaly)
    ecc = float(ecc)
    lo = mean_anomaly - ecc
    hi = mean_anomaly + ecc
    # f(M - e) = -e - e sin(M - e) <= 0, and f(M + e) >= 0.
    flo = _kepler_residual(lo, mean_anomaly, ecc)
    if abs(flo) <= thresh:
        return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = _kepler_residual(mid, mean_anomaly, ecc)
        if abs(fmid) <= thresh or abs(hi - lo) <= thresh:
            return mid
        # Keep the bound whose residual still brackets zero.
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fmid

    return 0.5 * (lo + hi)


def eccen_anomaly(mean_anomaly, ecc, thresh=1e-10):
    """Solve Kepler's equation for the eccentric anomaly.

    ``Orbit.eccen_anomaly``: a starter approximation
    followed by Newton-Raphson integration. Circular orbits return the mean
    anomaly reduced to ``(-pi, pi]``.

    A starter a hair below zero is wrapped to ``2π``. At mean anomaly
    0 the residual is then ``2π``, and Newton oscillates (``e`` near
    0.5, step ``4π``) or runs away (higher ``e``). That start is folded
    back onto the mean anomaly first. If the step still never drops
    below ``thresh``, bisection replaces the raise.

    Parameters
    ----------
    mean_anomaly : array-like, shape (n_epochs,)
        Mean anomaly in radians.
    ecc : float
        Eccentricity. Must satisfy ``0 <= ecc < 1``.
    thresh : float, optional
        Newton-Raphson tolerance in radians, by default 1e-10.

    Returns
    -------
    eccentric_anomaly : ndarray, shape (n_epochs,)
        Eccentric anomaly in radians, on ``(-pi, pi]``.

    """
    ecc = float(ecc)
    if ecc < 0.0 or ecc >= 1.0:
        raise ValueError(
            f"Eccentricity must satisfy 0 <= e < 1, got {ecc}."
        )

    mean_anomaly = np.atleast_1d(np.asarray(mean_anomaly, dtype=float))

    # Range reduction to -pi < m <= pi, matching the gcwork port.
    mx = np.array(mean_anomaly, dtype=float, copy=True)
    mx = np.where(mx > np.pi, np.mod(mx, 2.0 * np.pi), mx)
    mx = np.where(mx > np.pi, mx - 2.0 * np.pi, mx)
    mx = np.where(mx <= -np.pi, np.mod(mx, 2.0 * np.pi), mx)
    mx = np.where(mx <= -np.pi, mx + 2.0 * np.pi, mx)

    if ecc == 0.0:
        return mx

    # Starter approximation (Mikkola-style), then one Halley-like step,
    # then Newton-Raphson where the residual is still above 1e-10.
    aux = (4.0 * ecc) + 0.50
    alpha = (1.0 - ecc) / aux
    beta = mx / (2.0 * aux)
    aux_root = np.sqrt(beta**2 + alpha**3)
    z = beta + aux_root
    z = np.where(z <= 0.0, beta - aux_root, z)
    # Real cube root. Do not use z**(1/3): negatives become complex.
    z = np.sign(z) * np.abs(z)**(1.0 / 3.0)
    s0 = z - alpha / z
    s1 = s0 - (0.0780 * s0**5) / (1.0 + ecc)
    e0 = mx + ecc * ((3.0 * s1) - (4.0 * s1**3))

    se0 = np.sin(e0)
    ce0 = np.cos(e0)
    f = e0 - (ecc * se0) - mx
    f1 = 1.0 - (ecc * ce0)
    f2 = ecc * se0
    f3 = ecc * ce0
    f4 = -1.0 * f2
    u1 = -1.0 * f / f1
    u2 = -1.0 * f / (f1 + 0.50 * f2 * u1)
    u3 = -1.0 * f / (
        f1 + 0.50 * f2 * u2 + 0.166666666666670 * f3 * u2 * u2
    )
    u4 = -1.0 * f / (
        f1 + 0.50 * f2 * u3
        + 0.166666666666670 * f3 * u3 * u3
        + 0.0416666666666670 * f4 * u3**3
    )
    eccanom = e0 + u4
    eccanom = np.where(eccanom >= 2.0 * np.pi, eccanom - 2.0 * np.pi, eccanom)
    eccanom = np.where(eccanom < 0.0, eccanom + 2.0 * np.pi, eccanom)

    # Mean anomaly shifted onto [0, 2pi) for the residual the gcwork
    # Newton loop compares against.
    mmm = np.array(mx, dtype=float, copy=True)
    mmm = np.where(mmm < 0.0, mmm + 2.0 * np.pi, mmm)
    diff = eccanom - ecc * np.sin(eccanom) - mmm
    needs = np.flatnonzero(np.abs(diff) > 1e-10)

    for i in needs:
        # Prefer the 2π image closer to this mean anomaly. The gcwork
        # wrap above turns a tiny negative root into 2π. Against M = 0
        # the residual is then 2π, and the Newton step is 2π / (1 - e),
        # which swaps 2π and -2π forever when e = 0.5.
        folded = _fold_two_pi(eccanom[i], mmm[i])
        if abs(_kepler_residual(folded, mmm[i], ecc)) < abs(
            _kepler_residual(eccanom[i], mmm[i], ecc)
        ):
            eccanom[i] = folded
        # Same Newton update and the same step test as the gcwork
        # port. A start that already converges is not perturbed.
        settled = False
        for _ in range(50):
            fe = eccanom[i] - ecc * np.sin(eccanom[i]) - mmm[i]
            fs = 1.0 - ecc * np.cos(eccanom[i])
            # 1 - e cos E is at least 1 - e. A zero here is roundoff,
            # which used to divide and raise. Bisect that case.
            if (not np.isfinite(fs)) or fs == 0.0:
                break
            oldval = eccanom[i]
            eccanom[i] = oldval - fe / fs
            if abs(oldval - eccanom[i]) < thresh:
                settled = True
                break
        # Not settled, and the residual is still outside thresh.
        # A step that only chatters at roundoff keeps the Newton
        # value. Anything else is bisected on the (-π, π] anomaly
        # instead of raising.
        if not settled and abs(
            _kepler_residual(eccanom[i], mmm[i], ecc)
        ) > thresh:
            # Tighter than the Newton step test. A 1e-10 residual
            # at high e is a larger angle error than the gcwork
            # position tolerance can absorb.
            eccanom[i] = _bisect_kepler(float(mx[i]), ecc, 1.0e-14)
        while eccanom[i] >= np.pi:
            eccanom[i] = eccanom[i] - 2.0 * np.pi
        while eccanom[i] < -np.pi:
            eccanom[i] = eccanom[i] + 2.0 * np.pi

    return eccanom


def kep2xyz(epochs, period, t0, ecc, incl, big_omega, omega,
            mass=4.0e6, dist=8.0e3):
    """Cartesian offset of a Newtonian Keplerian orbit.

    Parameters
    ----------
    epochs : array-like, shape (n_epochs,)
        Decimal years at which to evaluate the orbit.
    period : float
        Period in years. Must be positive.
    t0 : float
        Time of periapse, decimal year.
    ecc : float
        Eccentricity, ``0 <= ecc < 1``.
    incl : float
        Inclination in degrees.
    big_omega : float
        Longitude of the ascending node in degrees.
    omega : float
        Argument of periapse in degrees.
    mass : float, optional
        Central mass in solar masses, by default 4.0e6.
    dist : float, optional
        Distance in parsecs, by default 8.0e3.

    Returns
    -------
    r_arcsec : ndarray, shape (n_epochs, 3)
        Offset from the central mass, in arcseconds. Column 0 is east,
        1 is north, 2 is the line of sight.
    v_mas_yr : ndarray, shape (n_epochs, 3)
        Velocity in milliarcseconds per year, same axes.
    acc_mas_yr2 : ndarray, shape (n_epochs, 3)
        Acceleration in milliarcseconds per year squared, same axes.
        Named ``acc`` so it is not the semi-major axis ``a``.

    """
    epochs = np.atleast_1d(np.asarray(epochs, dtype=float))
    period = float(period)
    ecc = float(ecc)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError(f"Period must be positive and finite, got {period}.")
    if ecc < 0.0 or ecc >= 1.0:
        raise ValueError(f"Eccentricity must satisfy 0 <= e < 1, got {ecc}.")

    mass = float(mass)
    dist = float(dist)
    # Semi-major axis in AU. Gaussian (P**2 * M)**(1/3), not G*M.
    # See semimajor_axis_mas. The physical axis misses the fixtures.
    axis = (period**2 * mass)**(1.0 / 3.0)
    mean_motion = 2.0 * np.pi / period
    ecc_sqrt = np.sqrt(1.0 - ecc**2)

    mean_anom = mean_motion * (epochs - float(t0))
    ecc_anom = eccen_anomaly(mean_anom, ecc)
    cos_e = np.cos(ecc_anom)
    sin_e = np.sin(ecc_anom)
    edot = mean_motion / (1.0 - ecc * cos_e)
    x_orb = cos_e - ecc
    y_orb = ecc_sqrt * sin_e

    cos_om = np.cos(np.radians(omega))
    sin_om = np.sin(np.radians(omega))
    cos_big = np.cos(np.radians(big_omega))
    sin_big = np.sin(np.radians(big_omega))
    cos_i = np.cos(np.radians(incl))
    sin_i = np.sin(np.radians(incl))

    # Thiele-Innes constants, in AU.
    con_a = axis * (cos_om * cos_big - sin_om * sin_big * cos_i)
    con_b = axis * (cos_om * sin_big + sin_om * cos_big * cos_i)
    con_c = axis * (sin_om * sin_i)
    con_f = axis * (-sin_om * cos_big - cos_om * sin_big * cos_i)
    con_g = axis * (-sin_om * sin_big + cos_om * cos_big * cos_i)
    con_h = axis * (cos_om * sin_i)

    n_epochs = epochs.size
    r = np.zeros((n_epochs, 3), dtype=float)
    v = np.zeros((n_epochs, 3), dtype=float)
    # acc, not a: a is the semi-major axis.
    acc = np.zeros((n_epochs, 3), dtype=float)

    r[:, 0] = (con_b * x_orb) + (con_g * y_orb)
    r[:, 1] = (con_a * x_orb) + (con_f * y_orb)
    r[:, 2] = (con_c * x_orb) + (con_h * y_orb)

    v[:, 0] = edot * ((-con_b * sin_e) + (con_g * ecc_sqrt * cos_e))
    v[:, 1] = edot * ((-con_a * sin_e) + (con_f * ecc_sqrt * cos_e))
    v[:, 2] = edot * ((-con_c * sin_e) + (con_h * ecc_sqrt * cos_e))

    # Acceleration in the AU frame, then convert with the cgs floats.
    # GM / r^2 points at the black hole. r is still in AU here.
    gm = mass * _MSUN_G * _G_CGS
    for ii in range(n_epochs):
        rmag_cm = np.sqrt(np.sum(r[ii, :]**2)) * _CM_IN_AU
        acc[ii, :] = -gm * r[ii, :] * _CM_IN_AU / rmag_cm**3

    # r: AU to arcsec (1 AU at 1 pc is 1 arcsec). v: AU/yr to mas/yr.
    # acc: cm/s^2 to mas/yr^2. Factors are the import-time floats.
    r = r / dist
    v = v * _MAS_PER_ARCSEC / dist
    acc = acc * _MAS_PER_ARCSEC * _SEC_IN_YR**2 / (_CM_IN_AU * dist)

    return r, v, acc


def read_orbits_dat(path):
    """Read an ``orbits.dat`` file into a table of six elements.

    Parameters
    ----------
    path : str or path-like
        Whitespace-separated file, no header. The on-disk layout is
        unchanged. Each data line has nine fields: name, P (yr),
        a (mas), t0, e, i (deg), Omega (deg), omega (deg),
        search (pix). ``a`` is the semi-major axis.

    Returns
    -------
    table : astropy.table.Table
        Columns ``name``, ``orb_P``, ``orb_t0``, ``orb_e``, ``orb_i``,
        ``orb_Omega``, ``orb_omega``. ``a`` and ``search`` are not
        columns and are not stored in ``table.meta``.

    Notes
    -----
    A line that does not have exactly nine fields raises ``ValueError``.
    Names are kept as written (``S0-2``, not a different spelling).
    """
    names = []
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped == '' or stripped.startswith('#'):
                continue
            fields = stripped.split()
            if len(fields) != 9:
                raise ValueError(
                    f"{path}:{line_number}: expected 9 fields, "
                    f"got {len(fields)}."
                )
            # a (fields[2]) and search (fields[8]) are checked by being
            # parsed, then discarded. They never become columns.
            name = fields[0]
            period, _a_mas, t0, ecc, incl, big_omega, arg_peri, _search = (
                float(fields[1]), float(fields[2]), float(fields[3]),
                float(fields[4]), float(fields[5]), float(fields[6]),
                float(fields[7]), float(fields[8]),
            )
            names.append(name)
            rows.append((period, t0, ecc, incl, big_omega, arg_peri))

    rows = np.asarray(rows, dtype=float) if rows else np.zeros((0, 6))
    table = Table()
    table['name'] = names
    for j, col in enumerate(_ELEMENT_COLUMNS):
        table[col] = rows[:, j] if len(rows) else np.array([], dtype=float)
        
    return table


def attach_orbits(starlist, orbits):
    """Copy orbit elements onto catalog rows that share a name.

    Parameters
    ----------
    starlist : astropy.table.Table
        Catalog with a ``name`` column. Modified in place.
    orbits : astropy.table.Table
        Table returned by :func:`read_orbits_dat`.

    Returns
    -------
    starlist : astropy.table.Table
        The same object. Matched rows get ``motion_model_input='Orbit'``
        and the six ``orb_*`` columns. Unmatched catalog rows get NaN
        elements and keep their existing ``motion_model_input``.
        ``fit_motion`` is not set.

    Notes
    -----
    Names present in ``orbits`` but missing from ``starlist`` are
    warned and not added as rows. Error columns and fit diagnostics
    are not created here; the fitter adds those.
    """
    if 'name' not in starlist.colnames:
        raise KeyError("attach_orbits: starlist has no 'name' column.")

    n_stars = len(starlist)
    catalog_names = np.asarray(starlist['name']).astype(str)
    orbit_names = np.asarray(orbits['name']).astype(str)

    # Build the six element columns. Unmatched rows stay NaN.
    for col in _ELEMENT_COLUMNS:
        values = np.full(n_stars, np.nan, dtype=float)
        if col not in starlist.colnames:
            starlist[col] = values
        else:
            # Keep any pre-existing numbers only until a match overwrites
            # them. Unmatched rows are set to NaN, per the plan.
            starlist[col] = values

    if 'motion_model_input' not in starlist.colnames:
        # Lazy import: motion_model imports this module, so the width
        # constant is only available once both modules have loaded.
        from flystar.startables import _MOTION_MODEL_NAME_WIDTH
        starlist.add_column(Column(
            data=np.full(n_stars, '', dtype=f'U{_MOTION_MODEL_NAME_WIDTH}'),
            name='motion_model_input',
        ))

    orbit_index = {name: i for i, name in enumerate(orbit_names)}
    matched = set()
    for i_star, name in enumerate(catalog_names):
        if name not in orbit_index:
            continue
        i_orb = orbit_index[name]
        matched.add(name)
        for col in _ELEMENT_COLUMNS:
            starlist[col][i_star] = orbits[col][i_orb]
        starlist['motion_model_input'][i_star] = 'Orbit'

    missing = [name for name in orbit_names if name not in matched]
    if missing:
        warnings.warn(
            "attach_orbits: orbit-file names not in the catalog, "
            f"not added: {', '.join(missing)}",
            UserWarning,
            stacklevel=2,
        )
    return starlist
