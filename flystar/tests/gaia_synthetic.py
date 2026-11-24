"""
A synthetic, Gaia-like astropy Table for the Gaia ingestion tests.

The table mimics what astroquery returns for a Gaia DR3 cone search: masked
columns, some stars with no proper motion (two-parameter solutions) and some
with no parallax. It is deterministic, so legacy outputs captured from it can
be compared bit for bit.
"""
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table, MaskedColumn

# Field center used by the legacy fixtures (hourangle / degree strings, as the
# old prepare_gaia_for_flystar took them).
CENTER_RA_STR = '17:45:40.04'
CENTER_DEC_STR = '-29:00:28.1'


def make_synthetic_gaia(ra0=None, dec0=None, half_width=120.0, n=60, seed=42,
                        with_corr=True):
    """
    Make a synthetic Gaia-like table.

    Parameters
    ----------
    ra0, dec0 : float, optional
        Field center in degrees. Defaults to CENTER_RA_STR, CENTER_DEC_STR.
    half_width : float, optional
        Stars are placed uniformly within +/- half_width arcsec (spherical
        offsets) of the center.
    n : int, optional
        Number of stars.
    seed : int, optional
        Random seed.
    with_corr : bool, optional
        Include Gaia's correlation-coefficient columns.

    Returns
    -------
    gaia : astropy.table.Table
        Masked table with Gaia's column names and units.
    """
    if ra0 is None or dec0 is None:
        c = SkyCoord(CENTER_RA_STR, CENTER_DEC_STR, unit=(u.hourangle, u.deg), frame='icrs')
        ra0, dec0 = c.ra.deg, c.dec.deg

    rng = np.random.default_rng(seed)
    center = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    dlon = rng.uniform(-half_width, half_width, n) * u.arcsec
    dlat = rng.uniform(-half_width, half_width, n) * u.arcsec
    stars = center.spherical_offsets_by(dlon, dlat)

    # Every 7th star has no proper motion or parallax (2-parameter solution);
    # every 5th has a parallax but it is masked.
    no_pm = np.zeros(n, dtype=bool)
    no_pm[3::7] = True
    no_plx = no_pm.copy()
    no_plx[1::5] = True

    def masked(data, mask, unit=None):
        return MaskedColumn(data=data, mask=mask, unit=unit)

    nomask = np.zeros(n, dtype=bool)
    gaia = Table(masked=True)
    gaia['source_id'] = masked(4057000000000000000 + np.arange(n, dtype=np.int64) * 1013, nomask)
    gaia['ra'] = masked(stars.ra.deg, nomask, u.deg)
    gaia['ra_error'] = masked(rng.uniform(0.02, 1.5, n), nomask, u.mas)
    gaia['dec'] = masked(stars.dec.deg, nomask, u.deg)
    gaia['dec_error'] = masked(rng.uniform(0.02, 1.5, n), nomask, u.mas)
    gaia['parallax'] = masked(rng.normal(0.3, 0.5, n), no_plx, u.mas)
    gaia['parallax_error'] = masked(rng.uniform(0.02, 0.8, n), no_plx, u.mas)
    gaia['pmra'] = masked(rng.normal(-2.0, 4.0, n), no_pm, u.mas / u.yr)
    gaia['pmra_error'] = masked(rng.uniform(0.02, 1.5, n), no_pm, u.mas / u.yr)
    gaia['pmdec'] = masked(rng.normal(-4.0, 4.0, n), no_pm, u.mas / u.yr)
    gaia['pmdec_error'] = masked(rng.uniform(0.02, 1.5, n), no_pm, u.mas / u.yr)
    if with_corr:
        for name in ['ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
                     'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
                     'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']:
            # Small correlations keep every 5x5 covariance positive definite.
            mask = nomask if name == 'ra_dec_corr' else no_pm
            gaia[name] = masked(rng.uniform(-0.2, 0.2, n), mask)
    gaia['ref_epoch'] = masked(np.full(n, 2016.0), nomask, u.yr)
    gaia['phot_g_mean_mag'] = masked(rng.uniform(9.0, 20.0, n), nomask, u.mag)
    gaia['phot_g_mean_flux_over_error'] = masked(rng.uniform(20.0, 3000.0, n), nomask)
    gaia['astrometric_params_solved'] = masked(np.where(no_pm, 3, 31).astype(np.int16), nomask)

    # Masked entries hold junk underneath, as astroquery's can.
    for name in ['pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'parallax', 'parallax_error']:
        gaia[name].data.data[gaia[name].mask] = 1e20

    return gaia
