import copy
import numpy as np
import pylab as plt
from scipy.stats import f
from astropy import table
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from flystar import starlists, match, parallax
from flystar import projection as proj

##################################################
# New codes for velocity support in FlyStar and using
# the new StarTable and StarList format.
##################################################

def query_gaia(ra, dec, search_radius=30.0, table_name='gaiadr3'):
    """
    Query the Gaia database at the specified location
    and with the specified search radius

    Input
    ----------
    ra : string
        R.A. in hours in the format such as '17:45:40.3'

    dec : string
        Dec. in degrees in the format such as '-29:00:28.0'

    search_radius : float
        The search radius in arcseconds.

    Optional Input
    --------------
    table_name : string
        Options are 'gaiadr2' or 'gaiaedr3'
    """
    from astroquery.gaia import Gaia
    target_coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    ra = target_coords.ra.degree
    dec = target_coords.dec.degree

    search_radius *= u.arcsec

    Gaia.ROW_LIMIT = 50000
    gaia_job = Gaia.cone_search_async(target_coords, radius=search_radius, table_name=table_name + '.gaia_source')
    gaia = gaia_job.get_results()

    #Change new 'SOURCE_ID' column header back to lowercase 'source_id' so all subsequent functions still work:
    if 'SOURCE_ID' in gaia.colnames:
        gaia.rename_column('SOURCE_ID', 'source_id')

    return gaia

def check_gaia_parallaxes(ra,dec,search_radius=10.0,table_name='gaiadr3',target='(unnamed)',
    file_ext=''):
    """
    Query the Gaia database at the specified location
    and with the specified search radius, and plot
    parallaxes.

    Input
    ----------
    ra : string
        R.A. in hours in the format such as '17:45:40.3'

    dec : string
        Dec. in degrees in the format such as '-29:00:28.0'

    search_radius : float
        The search radius in arcseconds.

    Optional Input
    --------------
    table_name : string
        Options are 'gaiadr2' or 'gaiadr3'
    """
    # Query Gaia
    gaia = query_gaia(ra,dec,search_radius=search_radius,table_name=table_name)
    # Set up reasonable histogram bins
    plim0,plim1 = np.min(gaia['parallax']),np.max(gaia['parallax'])
    pplim0,pplim1 = np.min(gaia['parallax']/gaia['parallax_error']),np.max(gaia['parallax']/gaia['parallax_error'])
    binwidth = 1
    pbins = np.arange(np.floor(plim0),np.ceil(plim1)+binwidth,binwidth)
    ppbins = np.arange(np.floor(pplim0),np.ceil(pplim1)+binwidth,binwidth)
    # Find number where plx/plx_err>3
    p_perr = (gaia['parallax']/gaia['parallax_error']).compressed()
    nppe3 = sum((p_perr>3).astype(int))
    nppen3 = sum((p_perr<-3).astype(int))
    print(table_name,'stars within',search_radius,'\" with plx/plx_err>3: ', nppe3, ' of ', len(gaia['parallax']))
    print(table_name,'stars within',search_radius,'\" with plx/plx_err<-3: ', nppen3, ' of ', len(gaia['parallax']))
    # Plot
    plt.subplots(nrows=1,ncols=2,figsize=(12,6))
    plt.subplot(121)
    plt.xlabel('parallax (mas)'); plt.ylabel('N stars')
    plt.hist(gaia['parallax'],bins=pbins)
    plt.yscale('log')
    plt.title(table_name+' parallax histograms, '+str(search_radius)+'\" radius around '+target, loc='left')
    plt.subplot(122)
    plt.xlabel('parallax/parallax_error')
    plt.hist(gaia['parallax']/gaia['parallax_error'],bins=ppbins)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('gaiaplx'+file_ext+'.png')


def _parse_center(ra, dec=None):
    """
    Field center in decimal degrees (ICRS) from any accepted input form.

    Parameters
    ----------
    ra : SkyCoord, str, astropy Quantity/Angle, or float
        A SkyCoord (then ``dec`` must be None); a sexagesimal or other
        string, read as hourangle (e.g. '17:45:40.04', as accepted by
        :func:`query_gaia`); a Quantity or Angle; or a plain number in
        degrees.
    dec : str, astropy Quantity/Angle, or float, optional
        Declination. A string or number is read as degrees.

    Returns
    -------
    ra0, dec0 : float
        Center in decimal degrees, ICRS.
    """
    if isinstance(ra, SkyCoord):
        if dec is not None:
            raise ValueError("Pass dec=None when the center is a SkyCoord.")
        c = ra.icrs
        return float(c.ra.degree), float(c.dec.degree)

    if dec is None:
        raise ValueError("dec is required unless ra is a SkyCoord.")

    if isinstance(ra, str) or isinstance(dec, str):
        # The original hourangle / degree string input.
        c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
        return c.ra.degree, c.dec.degree

    if isinstance(ra, u.Quantity):
        ra = ra.to_value(u.deg)
    if isinstance(dec, u.Quantity):
        dec = dec.to_value(u.deg)

    return ra, dec


def _filled(col, fill_value):
    """
    A plain float array from a (possibly masked) column, plus its mask.
    """
    mask = np.ma.getmaskarray(col)
    data = np.ma.filled(np.ma.asarray(col, dtype=float), fill_value)
    data = np.array(data, dtype=float)
    data[mask] = fill_value

    return data, mask


def _gaia_data_release(gaia):
    """
    Gaia data release of a table: from ``gaia.meta['data_release']`` if set,
    otherwise inferred from ``ref_epoch`` (2016.0 is EDR3 or DR3, which share
    an astrometric solution; 2015.5 is DR2; 2015.0 is DR1).
    """
    for key in ['data_release', 'DATA_RELEASE']:
        if key in gaia.meta:
            return str(gaia.meta[key])

    epochs = np.unique(_filled(gaia['ref_epoch'], np.nan)[0])
    if len(epochs) == 1:
        known = {2016.0: 'EDR3/DR3', 2015.5: 'DR2', 2015.0: 'DR1'}
        if epochs[0] in known:
            return known[epochs[0]]

    return 'unknown'


def _gaia_ref_epoch(gaia):
    """
    Catalog reference epoch(s) for the table metadata: one float if all
    stars share it, otherwise the sorted list of values.
    """
    epochs = np.unique(_filled(gaia['ref_epoch'], np.nan)[0])
    if len(epochs) == 1:
        return float(epochs[0])

    return [float(e) for e in epochs]


def _gaia_meta(gaia, ra0, dec0, projection, bright_pm_corrected):
    """
    Metadata recording the frame of a table built from Gaia.
    """
    meta = {'projection': projection,
            'proj_ra0': float(ra0),
            'proj_dec0': float(dec0),
            'frame': 'icrs',
            'parallax_ref': 'ssb',
            'data_release': _gaia_data_release(gaia),
            'ref_epoch': _gaia_ref_epoch(gaia),
            'bright_pm_corrected': bool(bright_pm_corrected)}

    return meta


# Gaia's astrometric parameters, in the order used for covariance matrices
# here, and the prefix of their correlation columns.
_GAIA_PARAMS = ['ra', 'dec', 'pmra', 'pmdec', 'parallax']


def _gaia_covariance(gaia, pm_ok, plx_ok, params=_GAIA_PARAMS):
    """
    Gaia covariance matrices of the requested astrometric parameters.

    Parameters
    ----------
    gaia : astropy table
        Gaia catalog. Errors are read from ``<param>_error`` and correlations
        from Gaia's ``<p1>_<p2>_corr`` columns; absent or masked correlations
        are taken as zero.
    pm_ok, plx_ok : numpy array of bool
        Stars whose proper motions / parallaxes are used. The others get zero
        variance for those parameters.
    params : list of str, optional
        Subset of 'ra', 'dec', 'pmra', 'pmdec', 'parallax'.

    Returns
    -------
    cov : numpy array, shape (N, P, P)
        In mas and mas/yr; the 'ra' entry is for alpha* (Gaia's ra_error is
        already sigma_alpha* = sigma_alpha cos(dec)).
    """
    n = len(gaia)
    sig = np.zeros((n, len(params)))
    for i, p in enumerate(params):
        sig[:, i] = _filled(gaia[p + '_error'], 0.0)[0]
        if p in ('pmra', 'pmdec'):
            sig[~pm_ok, i] = 0.0
        if p == 'parallax':
            sig[~plx_ok, i] = 0.0

    corr = np.broadcast_to(np.eye(len(params)), (n, len(params), len(params))).copy()
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            for name in (f'{params[i]}_{params[j]}_corr', f'{params[j]}_{params[i]}_corr'):
                if name in gaia.colnames:
                    rho = _filled(gaia[name], 0.0)[0]
                    rho[~np.isfinite(rho)] = 0.0
                    corr[:, i, j] = rho
                    corr[:, j, i] = rho
                    break

    cov = sig[:, :, np.newaxis] * corr * sig[:, np.newaxis, :]

    return cov


def _flip_jacobian(jac):
    """
    F J with F = diag(-1, 1): maps sky (east, north) to FlyStar (x, y).
    """
    fj = jac.copy()
    fj[..., 0, :] *= -1.0

    return fj


def _gaia_proper_motions(gaia, correct_bright_pm):
    """
    Proper motions (mas/yr) with missing ones set to zero, optionally with
    the bright-star correction, and the mask of stars that have them.
    """
    pmra, pmra_mask = _filled(gaia['pmra'], 0.0)
    pmdec, pmdec_mask = _filled(gaia['pmdec'], 0.0)
    pm_ok = ~pmra_mask & ~pmdec_mask & np.isfinite(pmra) & np.isfinite(pmdec)
    pmra[~pm_ok] = 0.0
    pmdec[~pm_ok] = 0.0

    if correct_bright_pm:
        ra, _ = _filled(gaia['ra'], np.nan)
        dec, _ = _filled(gaia['dec'], np.nan)
        gmag, _ = _filled(gaia['phot_g_mean_mag'], np.nan)
        pmra_c, pmdec_c = proj.correct_bright_pm(ra, dec, pmra, pmdec, gmag,
                                                 data_release=_gaia_data_release(gaia))
        pmra = np.where(pm_ok, pmra_c, pmra)
        pmdec = np.where(pm_ok, pmdec_c, pmdec)

    return pmra, pmdec, pm_ok


def prepare_gaia_for_flystar(gaia, ra, dec=None, targets_dict=None, match_dr_max=0.2, pi_err_limit=0.4,
                             default_motion_model='Linear', projection='gnomonic', correct_bright_pm=False):
    """
    Take a Gaia table (from astroquery) and produce a new table projected onto
    a plane of arcsec offsets about the given center, with columns named
    such that they are ready for FlyStar input.

    Positions are each star's at its catalog ``ref_epoch`` (stored as
    ``t0``), in ICRS. FlyStar's x is west-positive: x0 = -xi, y0 = eta, with
    (xi, eta) the east/north standard coordinates of the projection
    (see :mod:`flystar.projection`). Proper motions and errors are mapped
    through the projection's Jacobian J: (vx, vy) = F J (pmra, pmdec) and
    C' = F J C J^T F with F = diag(-1, 1), where C includes Gaia's
    correlation coefficients when the table has them; only the square roots
    of the diagonal of C' are kept.

    Inputs
    ------------
    gaia : astropy table
        Gaia Catalog from query_gaia()

    ra : SkyCoord, str, Quantity, or float
        Projection center (tangent point), ICRS. A SkyCoord (then leave dec
        as None); a string, read as hourangle (e.g. '17:45:40.04', the
        original input form); an astropy Quantity/Angle; or a plain number,
        read as **degrees** (as in project_gaia).

    dec : str, Quantity, or float, optional
        Declination of the center. Strings and numbers are degrees.

    Optional Inputs
    ---------------
    targets_dict : dict, optional
        {name: [x, y]} positions (arcsec, east-positive x) of targets to
        rename in the output.

    match_dr_max : float, optional
        Matching radius for targets_dict (arcsec).

    pi_err_limit : float, optional
        Parallaxes with errors above this (mas) are set to zero.

    default_motion_model : str, optional
        'Linear', 'Parallax', 'Fixed' or 'Empty'.

    projection : {'gnomonic', 'flat', 'flat_legacy'}, optional
        'gnomonic' (default): tangent-plane projection, PMs and errors
        through J_g. 'flat': scaled equirectangular projection, PMs and
        errors through J_f (self-consistent flat frame). 'flat_legacy':
        TEMPORARY, REMOVED BEFORE RELEASE -- the old code path bit for bit,
        including unrotated proper motions and the x0_err = ra_error *
        cos(dec0) bug; for regression tests only. It adds no ra/dec columns.

    correct_bright_pm : bool, optional
        Apply the Cantat-Gaudin & Brandt (2021) bright-star proper-motion
        correction (Gaia EDR3/DR3 only; see
        :func:`flystar.projection.correct_bright_pm`). Default False.

    Returns
    -------
    gaia_new : astropy table
        With name, x0, y0, x0_err, y0_err, vx, vy, vx_err, vy_err (arcsec,
        arcsec/yr), t0, source_id, m, me, pi, pi_err (arcsec), the motion
        model columns and, except for 'flat_legacy', each star's ra and dec
        (deg). ``meta`` records projection, proj_ra0, proj_dec0 (deg),
        frame='icrs', parallax_ref='ssb', data_release, ref_epoch and
        bright_pm_corrected.
    """
    proj._check_projection(projection)
    ra, dec = _parse_center(ra, dec)   # center in decimal degrees, ICRS

    if projection == 'flat_legacy':
        # TEMPORARY -- REMOVED BEFORE RELEASE. The old code path, kept bit for
        # bit (unrotated PMs and the x0_err = ra_error * cos(dec0) bug, which
        # shrinks every x error by cos(dec0)) for regression tests.
        if correct_bright_pm:
            raise ValueError("correct_bright_pm is not supported with projection='flat_legacy'.")

        cos_dec = np.cos(np.radians(dec))
        x, y = proj.sky_to_plane(gaia['ra'], gaia['dec'], ra, dec, projection='flat_legacy')  # arcsec
        xe = gaia['ra_error'] * cos_dec / 1e3      # arcsec
        ye = gaia['dec_error'] / 1e3               # arcsec

        gaia_new = table.Table([gaia['source_id'].data.astype('S19')], names=['name'], masked=False)

        gaia_new['x0'] = x * -1.0
        gaia_new['y0'] = y
        gaia_new['x0_err'] = xe
        gaia_new['y0_err'] = ye

        # Also convert the velocities. Note that Gaia PM are already * cos(dec)
        gaia_new['vx'] = gaia['pmra'].data * -1.0 / 1e3 # asec/yr
        gaia_new['vy'] = gaia['pmdec'].data / 1e3
        gaia_new['vx_err'] = gaia['pmra_error'].data / 1e3
        gaia_new['vy_err'] = gaia['pmdec_error'].data / 1e3

        gaia_new['t0'] = gaia['ref_epoch'].data
        gaia_new['source_id'] = gaia['source_id'].data.astype('S19')

        # Find sources without velocities and fix them up.
        idx = np.where(gaia['pmdec'].mask == True)[0]
        gaia_new['vx'][idx] = 0.0
        gaia_new['vy'][idx] = 0.0
        gaia_new['vx_err'][idx] = 0.0
        gaia_new['vy_err'][idx] = 0.0

        gaia_new['m'] = gaia['phot_g_mean_mag']
        gaia_new['me'] = 1.09/gaia['phot_g_mean_flux_over_error']
        gaia_new['pi'] = gaia['parallax'].data*1e-3
        gaia_new['pi_err'] = gaia['parallax_error'].data*1e-3

        # Set the velocities (and uncertainties) to zero if they aren't measured.
        idx = np.where(np.isnan(gaia_new['vx']) == True)[0]
        gaia_new['vx'][idx] = 0.0
        gaia_new['vx_err'][idx] = 0.0
        gaia_new['vy'][idx] = 0.0
        gaia_new['vy_err'][idx] = 0.0
    else:
        ra_star, _ = _filled(gaia['ra'], np.nan)
        dec_star, _ = _filled(gaia['dec'], np.nan)

        # Positions: x0 = -xi, y0 = eta (arcsec).
        xi, eta = proj.sky_to_plane(ra_star, dec_star, ra, dec, projection=projection)

        # Proper motions (mas/yr), zero where missing; optional bright-star correction.
        pmra, pmdec, pm_ok = _gaia_proper_motions(gaia, correct_bright_pm)

        # F J at each star, and the covariances C' = F J C J^T F.
        fj = _flip_jacobian(proj.jacobian(ra_star, dec_star, ra, dec, projection=projection))
        cov = _gaia_covariance(gaia, pm_ok, np.ones(len(gaia), dtype=bool), params=_GAIA_PARAMS[:4])
        cov_pos = fj @ cov[:, 0:2, 0:2] @ np.swapaxes(fj, -1, -2)
        cov_pm = fj @ cov[:, 2:4, 2:4] @ np.swapaxes(fj, -1, -2)
        vel = np.einsum('nij,nj->ni', fj, np.stack([pmra, pmdec], axis=-1))

        gaia_new = table.Table([gaia['source_id'].data.astype('S19')], names=['name'], masked=False)

        gaia_new['x0'] = -xi
        gaia_new['y0'] = eta
        gaia_new['x0_err'] = np.sqrt(cov_pos[:, 0, 0]) / 1e3        # arcsec
        gaia_new['y0_err'] = np.sqrt(cov_pos[:, 1, 1]) / 1e3

        gaia_new['vx'] = vel[:, 0] / 1e3                            # arcsec/yr
        gaia_new['vy'] = vel[:, 1] / 1e3
        gaia_new['vx_err'] = np.sqrt(cov_pm[:, 0, 0]) / 1e3
        gaia_new['vy_err'] = np.sqrt(cov_pm[:, 1, 1]) / 1e3

        gaia_new['t0'] = gaia['ref_epoch'].data
        gaia_new['source_id'] = gaia['source_id'].data.astype('S19')

        # Keep each star's catalog position (deg, ICRS at t0).
        gaia_new['ra'] = ra_star
        gaia_new['dec'] = dec_star

        gaia_new['m'] = gaia['phot_g_mean_mag']
        gaia_new['me'] = 1.09/gaia['phot_g_mean_flux_over_error']
        gaia_new['pi'] = gaia['parallax'].data*1e-3
        gaia_new['pi_err'] = gaia['parallax_error'].data*1e-3

    # Cut out stars with high plx error and set motion models
    idx = np.where((gaia_new['pi_err']>(pi_err_limit/1e3)) | (gaia['parallax'].mask == True))[0]
    gaia_new['pi'][idx] = 0.0
    gaia_new['pi_err'][idx] = 0.0
    if default_motion_model=='Parallax':
        gaia_new['motion_model_input'] = 'Parallax'
        gaia_new['motion_model_used'] = 'Parallax'
        gaia_new['motion_model_used'][idx] = 'Linear'
        gaia_new['n_params'] = 3
        gaia_new['n_params'][idx] = 2
    elif default_motion_model=='Linear':
        gaia_new['motion_model_input'] = 'Linear'
        gaia_new['motion_model_used'] = 'Linear'
        gaia_new['n_params'] = 2
    elif default_motion_model=='Fixed':
        gaia_new['motion_model_input'] = 'Fixed'
        gaia_new['motion_model_used'] = 'Fixed'
        gaia_new['n_params'] = 1
    elif default_motion_model=='Empty':
        gaia_new['motion_model_input'] = 'Empty'
        gaia_new['motion_model_used'] = 'Empty'
        gaia_new['n_params'] = 0
    else:
        print("Invalid motion model",default_motion_model,"- none assigned")

    #macy additions to try to fix wild magnitude values
    #gaia_new['ruwe'] = gaia['ruwe']
    #try:
    #    gaia_new = gaia_new[~gaia_new['m'].mask]
    #except:
    #    print('no invalig mags')

    gaia_new = gaia_new.filled()  #convert masked colunms to regular columns

    # Record the frame so it can be inverted later (startable_to_radec).
    gaia_new.meta.update(_gaia_meta(gaia, ra, dec, projection, correct_bright_pm))

    if targets_dict != None:
#        for targ_name, targ_coo in targets_dict.items():
#            dx = gaia_new['x0'] - (targ_coo[0] * -1.0)
#            dy = gaia_new['y0'] - targ_coo[1]
#            dr = np.hypot(dx, dy)
#
#            idx = dr.argmin()
#
#            if dr[idx] < match_dr_max:
#                gaia_new['name'][idx] = targ_name
#                print('Found match for: ', targ_name, ' - ',gaia_new['source_id'][idx])
        targ_names = [x for x in targets_dict]
        targ_xs = np.array([targets_dict[x][0] for x in targets_dict])
        targ_ys = np.array([targets_dict[x][1] for x in targets_dict])
        for i_gaia in range(len(gaia_new)):
            dx = gaia_new['x0'][i_gaia] - (targ_xs * -1.0)
            dy = gaia_new['y0'][i_gaia] - targ_ys
            dr = np.hypot(dx, dy)

            idx = dr.argmin()

            if dr[idx] < match_dr_max:
                gaia_new['name'][i_gaia] = targ_names[idx]
                print('Found match for: ', targ_names[idx], ' - ',gaia_new['source_id'][i_gaia])

    return gaia_new


def run_flystar():

    test_file = '/u/jlu/work/microlens/OB150211/a_2018_10_19/a_ob150211_2018_10_19/lis/stars_matched2.fits'

    t = Table.read(test_file)
    print(t.colnames)
    print(t['x_orig'].shape)

    # Model parameters: N_stars
    x0 = np.zeros(t['x_orig'].shape[0], dtype=float)
    y0 = np.zeros(t['x_orig'].shape[0], dtype=float)
    vx = np.zeros(t['x_orig'].shape[0], dtype=float)
    vy = np.zeros(t['x_orig'].shape[0], dtype=float)

    # Order of the transformation  STOPPED HERE
    t_order = 'junk'
    t_nparams = 'junk'

    # N_epochs by N_params
    px = np.zeros((t['x_orig'].shape[1], t_nparams), dtype=float)
    py = np.zeros((t['x_orig'].shape[1], t_nparams), dtype=float)

    # N_stars by N_epochs
    weights = np.zeros(t['x_orig'].shape)

    # Adopt a single t0 for all of the stars.
    t0 = t['t'][np.isfinite(t['t'])].mean()

    # Model undistorted positions
    xm_t = x0 + vx * (t - t0)
    ym_t = y0 + vy * (t - t0)

    # Model distorted positions


    return


def project_gaia(gaia, epoch, ra, dec=None, projection='gnomonic', include_parallax=False, correct_bright_pm=False):
    """
    Take the Gaia measurements, forward them in time, and then project them
    onto the plane about the field center.

    Each star is moved on the sky from its ``ref_epoch`` to ``epoch``
    (Eq. prop of docs/plans/tangent_plane: proper motion along its own east
    and north, plus the barycentric parallax shift if requested) and then
    projected, so the result does not depend on the order of the two steps.
    Errors are propagated with Gaia's correlation coefficients (when the
    table has them) and mapped through the projection's Jacobian:
    C(t) = F J A C A^T J^T F, with A the derivative of the position at
    ``epoch`` with respect to (ra*, dec, pmra, pmdec[, parallax]).

    Inputs
    ----------
    gaia : astropy table
        Gaia catalog from query_gaia().

    epoch : float (year)
        The decimal year to project the measurement to. Note that we use
        365.25 days per year. With include_parallax=True it is treated as an
        exact (UTC) decimal year.

    ra : SkyCoord, str, Quantity, or float
        Projection center (tangent point), ICRS. A plain number is decimal
        degrees; a string is hourangle (as in prepare_gaia_for_flystar); a
        SkyCoord needs dec=None.

    dec : str, Quantity, or float, optional
        Declination of the center (ICRS, degrees).

    Optional Inputs
    ---------------
    projection : {'gnomonic', 'flat', 'flat_legacy'}, optional
        As in prepare_gaia_for_flystar. 'flat_legacy' is TEMPORARY and
        REMOVED BEFORE RELEASE: the old code path bit for bit (stars moved in
        the plane by unrotated proper motions, no correlations, no parallax).

    include_parallax : bool, optional
        Add each star's barycentric parallax displacement at ``epoch``
        (observer at the Earth). Default False.

    correct_bright_pm : bool, optional
        Apply the Cantat-Gaudin & Brandt (2021) bright-star proper-motion
        correction (Gaia EDR3/DR3 only). Default False.

    Returns
    -------
    gaia_lis : StarList
        x, y (arcsec, x west-positive), xe, ye, m, me and x_avg, y_avg,
        m_avg; except for 'flat_legacy', also each star's ra, dec (deg, ICRS)
        at ``epoch``. ``meta`` holds the same frame metadata as
        prepare_gaia_for_flystar plus ``epoch`` and ``include_parallax``.
    """
    proj._check_projection(projection)
    ra, dec = _parse_center(ra, dec)   # center in decimal degrees, ICRS

    if projection == 'flat_legacy':
        # TEMPORARY -- REMOVED BEFORE RELEASE. The old code path, bit for bit.
        if include_parallax or correct_bright_pm:
            raise ValueError("include_parallax and correct_bright_pm are not supported "
                             "with projection='flat_legacy'.")

        t0 = gaia['ref_epoch']
        x0, y0 = proj.sky_to_plane(gaia['ra'], gaia['dec'], ra, dec, projection='flat_legacy')  # Arcsec
        x0e = gaia['ra_error']  / 1.0e3       # arcsec, already in alpha* (multiplied by cos(delta))
        y0e = gaia['dec_error'] / 1.0e3       # arcsec

        vx = gaia['pmra'] / 1.0e3            # arcsec / yr
        vy = gaia['pmdec'] / 1.0e3
        vxe = gaia['pmra_error'] / 1.0e3     # arcsec / yr
        vye = gaia['pmdec_error'] / 1.0e3

        # Modify any vx/vy, etc. that are zero and make a regular (unmasked) numpy array.
        vx[vx.mask] = 0.0
        vy[vy.mask] = 0.0
        vxe[vxe.mask] = 0.0
        vye[vye.mask] = 0.0
        vx = np.array(vx)
        vy = np.array(vy)
        vxe = np.array(vxe)
        vye = np.array(vye)

        dt = epoch - t0
        x_now = (x0 + (vx * dt)) * -1.0  # Switch to a left-handed coordinate system, like detector pixels.
        y_now = (y0 + (vy * dt))
        xe_now = np.hypot(x0e, vxe*dt)
        ye_now = np.hypot(y0e, vye*dt)
        extra_cols = {}
    else:
        ra_star, _ = _filled(gaia['ra'], np.nan)
        dec_star, _ = _filled(gaia['dec'], np.nan)
        t0, _ = _filled(gaia['ref_epoch'], np.nan)
        dt = epoch - t0

        pmra, pmdec, pm_ok = _gaia_proper_motions(gaia, correct_bright_pm)

        plx, plx_mask = _filled(gaia['parallax'], 0.0)
        plx_ok = ~plx_mask & np.isfinite(plx)
        plx[~plx_ok] = 0.0

        # Parallax shift vector s(t) and factors (P_e, P_n) at each star.
        n_star = len(gaia)
        p_e = np.zeros(n_star)
        p_n = np.zeros(n_star)
        s = None
        if include_parallax:
            t_epoch = Time(epoch, format='decimalyear', scale='utc').tdb
            obs_pos = parallax.get_observer_barycentric('earth', t_epoch)
            obs_pos = obs_pos.xyz.to_value(u.au)
            s = proj.parallax_shift_vector(ra_star, dec_star, obs_pos)
            _, e_hat, n_hat = proj.unit_vectors(ra_star, dec_star)
            p_e = np.sum(s * e_hat, axis=-1)
            p_n = np.sum(s * n_hat, axis=-1)

        # Move on the sky, then project (Eq. prop).
        ra_t, dec_t = proj.propagate_on_sky(ra_star, dec_star, pmra, pmdec, dt,
                                            parallax=plx if include_parallax else None, s=s)
        xi, eta = proj.sky_to_plane(ra_t, dec_t, ra, dec, projection=projection)

        # Errors: A maps (ra*, dec, pmra, pmdec, parallax) to the sky position at epoch.
        cov5 = _gaia_covariance(gaia, pm_ok, plx_ok & include_parallax)
        amat = np.zeros((n_star, 2, 5))
        amat[:, 0, 0] = 1.0
        amat[:, 1, 1] = 1.0
        amat[:, 0, 2] = dt
        amat[:, 1, 3] = dt
        amat[:, 0, 4] = p_e
        amat[:, 1, 4] = p_n
        fj = _flip_jacobian(proj.jacobian(ra_t, dec_t, ra, dec, projection=projection)) @ amat
        cov = fj @ cov5 @ np.swapaxes(fj, -1, -2)

        x_now = -xi
        y_now = eta
        xe_now = np.sqrt(cov[:, 0, 0]) / 1e3     # arcsec
        ye_now = np.sqrt(cov[:, 1, 1]) / 1e3
        extra_cols = {'ra': ra_t, 'dec': dec_t}

    # Format as a starlist
    gaia_lis = starlists.StarList(name=gaia['source_id'],
                                  x=x_now, y=y_now, m=gaia['phot_g_mean_mag'],
                                  xe=xe_now, ye=ye_now, me=1.0/gaia['phot_g_mean_flux_over_error'])

    # Duplicate columns to 'x_avg', etc. Needed for initial guessing.
    gaia_lis['x_avg'] = gaia_lis['x']
    gaia_lis['y_avg'] = gaia_lis['y']
    gaia_lis['m_avg'] = gaia_lis['m']

    for col, val in extra_cols.items():
        gaia_lis[col] = val

    gaia_lis.meta.update(_gaia_meta(gaia, ra, dec, projection, correct_bright_pm))
    gaia_lis.meta['epoch'] = float(epoch)
    gaia_lis.meta['include_parallax'] = bool(include_parallax)

    return gaia_lis


def rename_after_flystar(star_tab, label_dat_file, new_copy=True, dr_tol=0.05, dm_tol=0.3, verbose=False):
    """
    Take a StarTable output from FlyStar MosaicToRef that has been
    aligned into R.A. and Dec. (usually by way of Gaia). Align
    the output to a label.dat file for this source and rename
    everything.
    """
    label_tab = starlists.read_label(label_dat_file, flipX=True)

    # Propogate the label.dat into the epoch of the star table.
    dt = label_tab['t0'] - star_tab['t0'][0]
    x_lab = label_tab['x0'] + (label_tab['vx'] * 1e-3 * dt)
    y_lab = label_tab['y0'] + (label_tab['vy'] * 1e-3 * dt)
    m_lab = label_tab['m']

    # Find the sets of named stars that are in common for
    # zeropoint correction and shift transformations.
    ndx_lab, ndx_star, n_matched = starlists.restrict_by_name(label_tab, star_tab)

    dm = star_tab['m0'][ndx_star] - m_lab[ndx_lab]
    dx = star_tab['x0'][ndx_star] - x_lab[ndx_lab]
    dy = star_tab['y0'][ndx_star] - y_lab[ndx_lab]

    if verbose:
        fmt = '{0:20s} {1:20s} {2:8.4f} {3:8.4f} {4:8.4f} {5:8.4f} {6:8.4f} {7:8.4f}'
        for ii in range(n_matched):
            print(fmt.format(label_tab['name'][ndx_lab[ii]], star_tab['name'][ndx_star[ii]],
                             x_lab[ndx_lab[ii]], star_tab['x0'][ndx_star[ii]],
                             y_lab[ndx_lab[ii]], star_tab['y0'][ndx_star[ii]],
                             m_lab[ndx_lab[ii]], star_tab['m0'][ndx_star[ii]]))


        print('Temporary shift transformations: ')
        print('    dm = {0:8.4f} +/- {1:8.4f}'.format(dm.mean(), dm.std()))
        print('    dx = {0:8.4f} +/- {1:8.4f}'.format(dx.mean(), dx.std()))
        print('    dy = {0:8.4f} +/- {1:8.4f}'.format(dy.mean(), dy.std()))

    m_lab = label_tab['m'] + dm.mean()
    x_lab += dx.mean()
    y_lab += dy.mean()

    # Now that we are in a common coordinate and magnitude
    # system, lets match the whole lists by coordinates.
    idx_lab, idx_star, dr, dm = match.match(x_lab, y_lab, m_lab,
                                            star_tab['x0'], star_tab['y0'], star_tab['m0'],
                                            dr_tol=dr_tol, dm_tol=dm_tol, verbose=verbose)
    #print('idx_lab:')
    #for iii in range(len(idx_lab)):
    #    print(label_tab["name"][idx_lab[iii]], star_tab["name"][idx_star[iii]])

    print('Renaming {0:d} out of {1:d} stars'.format(len(idx_lab), len(star_tab)))

    # Make a copy of the table, UNLESS, the user specifies.
    if new_copy:
        new_tab = copy.deepcopy(star_tab)
    else:
        new_tab = star_tab

    # copy over the original names... don't overwrite (this could mean data loss)
    if 'name_orig' not in new_tab.colnames:
        new_tab.add_column(Column(star_tab['name'].data, name='name_orig'))

    new_tab['name'][idx_star] = label_tab[idx_lab]['name']

    return new_tab

def pick_good_ref_stars(star_tab, r_cut=None, m_cut=None, p_err_cut=None, pm_err_cut=None, name_cut=None, reset=True):
    """
    Set the 'use_in_trans' flag according to a set of cuts.
    """
    # Start off assuming we will use all of them (or use what was previously specified).
    if reset:
        use = np.ones(len(star_tab), dtype=bool)
    else:
        use = star_tab['use_in_trans']
    print('pick_good_ref_stars: Starting with {0:d} stars.'.format(use.sum()))

    if r_cut is not None:
        r = np.hypot(star_tab['x0'], star_tab['y0'])
        use = use & (r < r_cut)
        print('pick_good_ref_stars: Use {0:d} stars after r<{1:.2f}.'.format(use.sum(), r_cut))

    if m_cut is not None:
        use = use & (star_tab['m0'] < m_cut)
        print('pick_good_ref_stars: Use {0:d} stars after m<{1:.2f}.'.format(use.sum(), m_cut))

    if p_err_cut is not None:
        p_err = np.mean((star_tab['x0_err'], star_tab['y0_err']), axis=0)
        use = use & (p_err < p_err_cut)
        print('pick_good_ref_stars: Use {0:d} stars after p_err<{1:.5f}.'.format(use.sum(), p_err_cut))

    if pm_err_cut is not None:
        pm_err = np.mean((star_tab['vx_err'], star_tab['vy_err']), axis=0)
        use = use & (pm_err < pm_err_cut)
        print('pick_good_ref_stars: Use {0:d} stars after pm_err<{1:.5f}.'.format(use.sum(), pm_err_cut))

    if name_cut is not None:
        for ii in range(len(name_cut)):
            use = use & (star_tab['name'] != name_cut[ii])
        print('pick_good_ref_stars: Use {0:d} stars after name cut.'.format(use.sum()))

    idx = np.where(use == True)[0]

    return idx


def startable_subset(tab, idx, mag_trans=True, mag_trans_orig=False):
    """
    Input is MosaicToRef table from alignment of multiple filters,
    such that the astrometry is combined but the photometry is not.
    This function is used to separate out a selected filter from the
    combined astrometry + uncombined photometry table.
    """
    # Multiples: ['x', 'y', 'm', 'name_in_list', 'xe', 'ye', 'me', 't',
    #     'x_orig', 'y_orig', 'm_orig', 'xe_orig', 'ye_orig', 'me_orig', 'used_in_trans',
    #     'xe_boot','ye_boot','me_boot']
    # Single: ['name', 'm0', 'm0_err', 'use_in_trans', 'ref_orig', 'n_detect',
    #     'x0', 'vx', 'y0', 'vy', 'x0_err', 'vx_err', 'y0_err', 'vy_err', 't0']
    # Don't include n_vfit

    new_tab = copy.deepcopy(tab)
    #new_tab.remove_column('n_fit')
    new_tab.remove_column('n_detect')
    for col in ['x','y','m','idx_in_list','name_in_list','xe','ye','me','t',
                'x_orig','y_orig','m_orig','xe_orig','ye_orig','me_orig',
                'used_in_trans','xe_boot','ye_boot','me_boot']:
        # 'idx_in_list'/'name_in_list' are alternatives (the latter only on
        # tables written before the switch to indices), and several of the
        # rest are optional, so slice whichever are actually present.
        if col in tab.colnames:
            new_tab[col] = tab[col][:,idx]

    new_tab.combine_lists('m', weights_col='me', sigma=3, ismag=True)

    if mag_trans:
        use = np.where(new_tab['used_in_trans'].mean(axis=1) == 1)[0]
        for ii in np.arange(len(new_tab['m'][0])):
            m_resid = new_tab['m0'][use] - new_tab['m'][use,ii]
            threshold = 3 * np.nanstd(m_resid)
            keepers = np.where(np.absolute(m_resid - np.nanmean(m_resid)) < threshold)[0]
            mag_offset = np.nanmean(m_resid[keepers])
            new_tab['m'][:,ii] += mag_offset
            print(f'mag_offset: {ii:d} dm = {mag_offset:7.4f}')

            # Update the original table.
            if mag_trans_orig:
                tab['m'][:,idx[ii]] += mag_offset

    return new_tab


##################################################
# Old codes.
##################################################

def calc_F(red_chi2_1, red_chi2_2, v1, v2):
    """
    compare two different models to get the proper polynomial fitting order

    Parameters
    ----------
    red_chi2_1 : float
        reduced chi2 for the first model
    red_chi2_2 : float
        reduced chi2 for the second model
    v1 : int
        degree of freedom for the first model,
        ``= 2*(N_star_matched) - model_parameters``
    v2 : int
        degree of freedom for the second mdoel

    Returns
    -------
    P : float
        The probability that the first model is better

    Notes
    -----
    - make sure the first model is the simple model
      and the second model is the more complicated model
    - the return value represents the probability that
      the first model is better than the second model, in other words,
      the small P means the more colicated model is needed.
      the large P means the simple model is good enough.
    - normally, the P value will increase from model1->model2, to
      model2->model3, to model3->model4. The user can decide a
      critical value (eg, 0.7) to find the proper model.

    Examples
    --------
    For 1st order polynomial fitting::

        x' = a0 + a1*x + a2*y
        y' = b0 + b1*x + b2*y
        v1 = 2*N1 - 2*3 (2*: because x and y direction)
        red_chi2_1 = chi2/v1

    For 2nd order polynomial fitting::

        x' = a0 + a1*x + a2*y + a3*x**2 + a4*y**2 + a5*x*y
        y' = b0 + b1*x + b2*y + b3*x**2 + b4*y**2 + b5*x*y
        v1 = 2*N1 - 2*6
        red_chi2_2 = chi2/v2

    then::

        calc_F(red_chi2_1, red_chi2_2, v1, v2)
    """

    f_value = red_chi2_1/red_chi2_2
    p = 1-f.cdf(f_value, v1, v2)
    return p
