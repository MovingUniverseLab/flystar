"""Write the gcwork east/north cross-check used by test_orbits.py.

Transcription of ``Orbit.kep2xyz`` and ``Orbit.eccen_anomaly`` from the
attached file ``orbits_jlu_python_gcwork_2024-10-03_338c.py`` (Jessica
Lu, gcwork, 2024-10-03). This script does not import ``flystar.orbits``
and does not import gcwork. The checked-in CSV is the output of this
transcription, so the unit test compares the FlyStar port against that
snapshot rather than against itself.

Mass and distance are the gcwork pair, passed explicitly:
``M = 4.07e6`` solar masses, ``R0 = 7960.1`` pc. They are not the
FlyStar defaults (``4.0e6``, ``8000``). East and north do not use
``G``, solar mass in grams, or cm-per-AU. Those constants affect only
the acceleration, which this fixture does not store.

Elements come from ``flystar/tests/test_data/orbits_v2.0.2.dat``, a
copy of the attached ``orbits_a790.dat`` (32 stars, nine fields, no
header). Epochs are periapse, half a period later, and the years
1995, 2010, and 2030.

Run from the repository root:

    python3 flystar/tests/generate_gcwork_kep2xyz_fixture.py
"""

import math
import pathlib

import numpy as np


# gcwork pair. Not the FlyStar default mass and distance.
MASS_MSUN = 4.07e6
DIST_PC = 7960.1

HERE = pathlib.Path(__file__).resolve().parent
ORBITS_DAT = HERE / 'test_data' / 'orbits_v2.0.2.dat'
OUT_CSV = HERE / 'test_data' / 'gcwork_kep2xyz_m4.07e6_r7960.1.csv'


def eccen_anomaly(mean_anomaly, ecc, thresh=1e-10):
    """Eccentric anomaly, transcribed from gcwork ``Orbit.eccen_anomaly``.

    Parameters
    ----------
    mean_anomaly : array-like, shape (n_epochs,)
        Mean anomaly in radians.
    ecc : float
        Eccentricity, ``0 <= ecc < 1``.
    thresh : float, optional
        Newton-Raphson tolerance in radians, by default 1e-10.

    Returns
    -------
    eccentric_anomaly : ndarray, shape (n_epochs,)
        Eccentric anomaly in radians.

    Notes
    -----
    The cube-root step follows the gcwork source, including
    ``abs(z) ** (1/3)`` and the later sign test on that positive
    value. A negative mean anomaly is shifted onto ``[0, 2 pi)``
    before the Newton loop, matching that source.
    """
    mean_anomaly = np.atleast_1d(np.asarray(mean_anomaly, dtype=float))
    mx = np.array(mean_anomaly, dtype=float, copy=True)
    mx = np.where(mx > math.pi, np.mod(mx, 2.0 * math.pi), mx)
    mx = np.where(mx > math.pi, mx - 2.0 * math.pi, mx)
    mx = np.where(mx <= -math.pi, np.mod(mx, 2.0 * math.pi), mx)
    mx = np.where(mx <= -math.pi, mx + 2.0 * math.pi, mx)
    if ecc == 0.0:
        return mx

    aux = (4.0 * ecc) + 0.50
    alpha = (1.0 - ecc) / aux
    beta = mx / (2.0 * aux)
    aux = np.sqrt(beta**2 + alpha**3)
    z = beta + aux
    z = np.where(z <= 0.0, beta - aux, z)
    # gcwork: test = abs(z)**(1/3), then copy that into z.
    z = np.abs(z)**(1.0 / 3.0)
    z = np.where(z < 0.0, -z, z)
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
    eccanom = np.where(eccanom >= 2.0 * math.pi, eccanom - 2.0 * math.pi, eccanom)
    eccanom = np.where(eccanom < 0.0, eccanom + 2.0 * math.pi, eccanom)

    mmm = np.array(mx, dtype=float, copy=True)
    mmm = np.where(mmm < 0.0, mmm + 2.0 * math.pi, mmm)
    diff = eccanom - ecc * np.sin(eccanom) - mmm
    needs = np.flatnonzero(np.abs(diff) > 1e-10)
    for i in needs:
        loop_count = 0
        while True:
            fe = eccanom[i] - ecc * np.sin(eccanom[i]) - mmm[i]
            fs = 1.0 - ecc * np.cos(eccanom[i])
            oldval = eccanom[i]
            eccanom[i] = oldval - fe / fs
            loop_count += 1
            if abs(oldval - eccanom[i]) < thresh:
                break
            if loop_count > 10**6:
                raise RuntimeError(
                    f'eccen_anomaly did not converge for e = {ecc}.'
                )
        while eccanom[i] >= math.pi:
            eccanom[i] = eccanom[i] - 2.0 * math.pi
        while eccanom[i] < -math.pi:
            eccanom[i] = eccanom[i] + 2.0 * math.pi
    return eccanom


def kep2xyz_east_north(epochs, period, t0, ecc, incl, big_omega, omega,
                       mass=MASS_MSUN, dist=DIST_PC):
    """East and north offset, transcribed from gcwork ``Orbit.kep2xyz``.

    Parameters
    ----------
    epochs : array-like, shape (n_epochs,)
        Decimal years.
    period : float
        Period in years.
    t0 : float
        Time of periapse, decimal year.
    ecc : float
        Eccentricity.
    incl, big_omega, omega : float
        Inclination, longitude of the ascending node, and argument of
        periapse, in degrees.
    mass : float, optional
        Central mass in solar masses, by default 4.07e6.
    dist : float, optional
        Distance in parsecs, by default 7960.1.

    Returns
    -------
    east, north : ndarray, shape (n_epochs,)
        Offset from the central mass, in arcseconds. Index 0 of the
        gcwork radius vector is east and index 1 is north.

    Notes
    -----
    ``a_AU = (P**2 * M)**(1/3)``. The radius is divided by ``dist``
    to convert AU to arcseconds. Acceleration is not returned.
    """
    epochs = np.atleast_1d(np.asarray(epochs, dtype=float))
    axis = (period**2 * mass)**(1.0 / 3.0)
    mean_motion = 2.0 * math.pi / period
    ecc_sqrt = math.sqrt(1.0 - ecc**2)
    mean_anom = mean_motion * (epochs - t0)
    ecc_anom = eccen_anomaly(mean_anom, ecc)
    cos_e = np.cos(ecc_anom)
    sin_e = np.sin(ecc_anom)
    x_orb = cos_e - ecc
    y_orb = ecc_sqrt * sin_e
    cos_om = math.cos(math.radians(omega))
    sin_om = math.sin(math.radians(omega))
    cos_big = math.cos(math.radians(big_omega))
    sin_big = math.sin(math.radians(big_omega))
    cos_i = math.cos(math.radians(incl))
    sin_i = math.sin(math.radians(incl))
    con_a = axis * (cos_om * cos_big - sin_om * sin_big * cos_i)
    con_b = axis * (cos_om * sin_big + sin_om * cos_big * cos_i)
    con_f = axis * (-sin_om * cos_big - cos_om * sin_big * cos_i)
    con_g = axis * (-sin_om * sin_big + cos_om * cos_big * cos_i)
    east = (con_b * x_orb) + (con_g * y_orb)
    north = (con_a * x_orb) + (con_f * y_orb)
    east = east / dist
    north = north / dist
    return east, north


def read_elements(path):
    """Nine-field orbits.dat rows, including the printed ``a``.

    Parameters
    ----------
    path : path-like
        Whitespace-separated file, no header. The on-disk field
        order is unchanged. The third number is the semi-major axis.

    Returns
    -------
    rows : list of tuple
        ``(name, P, a, t0, e, i, Omega, omega)``. ``search`` is parsed
        so a short line fails, then dropped.
    """
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == '' or stripped.startswith('#'):
                continue
            fields = stripped.split()
            if len(fields) != 9:
                raise ValueError(f'{path}: expected 9 fields, got {fields}')
            rows.append((
                fields[0],
                float(fields[1]), float(fields[2]), float(fields[3]),
                float(fields[4]), float(fields[5]), float(fields[6]),
                float(fields[7]),
            ))
    return rows


def main():
    """Write the CSV. Returns None.

    Returns
    -------
    None
    """
    rows = read_elements(ORBITS_DAT)
    lines = [
        'name,epoch,east_arcsec,north_arcsec,P,t0,e,i,Omega,omega,mass,dist',
    ]
    for name, period, _a_mas, t0, ecc, incl, big_omega, omega in rows:
        epochs = np.array([
            t0,
            t0 + 0.5 * period,
            1995.0,
            2010.0,
            2030.0,
        ], dtype=float)
        east, north = kep2xyz_east_north(
            epochs, period, t0, ecc, incl, big_omega, omega,
            mass=MASS_MSUN, dist=DIST_PC,
        )
        for epoch, e_off, n_off in zip(epochs, east, north):
            lines.append(
                f'{name},{epoch:.10f},{e_off:.16e},{n_off:.16e},'
                f'{period:.10f},{t0:.10f},{ecc:.10f},{incl:.10f},'
                f'{big_omega:.10f},{omega:.10f},{MASS_MSUN:.10e},{DIST_PC:.10f}'
            )
    OUT_CSV.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {OUT_CSV} ({len(lines) - 1} rows, {len(rows)} stars)')
    return None


if __name__ == '__main__':
    main()
