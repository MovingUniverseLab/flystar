import numpy as np
import pylab as plt
from flystar import starlists
from flystar import startables
from flystar import align
from flystar import match
from flystar import transforms
from astropy import table
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from astroquery.mast import Observations, Catalogs
import pdb, copy
import math
from scipy.stats import f

##################################################
# New codes for velocity support in FlyStar and using
# the new StarTable and StarList format. 
##################################################

def query_gaia(ra, dec, search_radius=30.0, table_name='gaiadr2'):
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
    target_coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    ra = target_coords.ra.degree
    dec = target_coords.dec.degree

    search_radius *= u.arcsec

    Gaia.ROW_LIMIT = 50000
    gaia_job = Gaia.cone_search_async(target_coords, search_radius, table_name = table_name + '.gaia_source')
    gaia = gaia_job.get_results()

    #Change new 'SOURCE_ID' column header back to lowercase 'source_id' so all subsequent functions still work:
    gaia['SOURCE_ID'].name = 'source_id'

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
    

def prepare_gaia_for_flystar(gaia, ra, dec, targets_dict=None, match_dr_max=0.2):
    """
    Take a Gaia table (from astroquery) and produce a new table with a tangential projection
    and shift such that the origin is centered on the target of interest. 
    Convert everything into arcseconds and name columns such that they are 
    ready for FlyStar input.

    Inputs
    ------------
    gaia : astropy table
        Gaia Catalog from query_gaia()

    ra : float
        Units = hourangle

    dec : float
        Units = degree
    """
    target_coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    ra = target_coords.ra.degree     # in decimal degrees
    dec = target_coords.dec.degree   # in decimal degrees
    
    cos_dec = np.cos(np.radians(dec))
    x = (gaia['ra'] - ra) * cos_dec * 3600.0   # arcsec
    y = (gaia['dec'] - dec) * 3600.0           # arcsec
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
    gaia_new['parallax'] = gaia['parallax']
    gaia_new['parallax_error'] = gaia['parallax_error']

    # Set the velocities (and uncertainties) to zero if they aren't measured.
    idx = np.where(np.isnan(gaia_new['vx']) == True)[0]
    gaia_new['vx'][idx] = 0.0
    gaia_new['vx_err'][idx] = 0.0
    gaia_new['vy'][idx] = 0.0
    gaia_new['vy_err'][idx] = 0.0

    #macy additions to try to fix wild magnitude values
    #gaia_new['ruwe'] = gaia['ruwe']
    #try:
    #    gaia_new = gaia_new[~gaia_new['m'].mask]
    #except:
    #    print('no invalig mags')

    gaia_new = gaia_new.filled()  #convert masked colunms to regular columns

    if targets_dict != None:
        for targ_name, targ_coo in targets_dict.items():
            dx = gaia_new['x0'] - (targ_coo[0] * -1.0)
            dy = gaia_new['y0'] - targ_coo[1]
            dr = np.hypot(dx, dy)

            idx = dr.argmin()

            if dr[idx] < match_dr_max:
                gaia_new['name'][idx] = targ_name
                print('Found match for: ', targ_name, ' - ',gaia_new['source_id'][idx])

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


def project_gaia(gaia, epoch, ra, dec):
    """
    Take the Gaia measurements, forward them in time, and then convert them into a tangential projection.
    
    Inputs
    ----------
    epoch : float (year)
        The decimal year to project the measurement to. Note that we use 365.25 days per year.
        
    ra : float (deg)
        The right ascension (J2000) in decimal degrees of the center of the field.
        
    dec : float (deg)
        The declination (J2000) in decimal degrees of the center of the field.
        
    """
    t0 = gaia['ref_epoch']
    x0 = (gaia['ra'] - ra) * np.cos(np.radians(dec)) * 3600.0   # Arcsec
    y0 = (gaia['dec'] - dec) * 3600.0
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
    
    # Format as a starlist
    gaia_lis = starlists.StarList(name=gaia['source_id'], 
                                  x=x_now, y=y_now, m=gaia['phot_g_mean_mag'],
                                  xe=xe_now, ye=ye_now, me=1.0/gaia['phot_g_mean_flux_over_error'])
    
    # Duplicate columns to 'x_avg', etc. Needed for initial guessing.
    gaia_lis['x_avg'] = gaia_lis['x']
    gaia_lis['y_avg'] = gaia_lis['y']
    gaia_lis['m_avg'] = gaia_lis['m']    
    
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
    #     'x_orig', 'y_orig', 'm_orig', 'xe_orig', 'ye_orig', 'me_orig', 'used_in_trans']
    # Single: ['name', 'm0', 'm0_err', 'use_in_trans', 'ref_orig', 'n_detect',
    #     'x0', 'vx', 'y0', 'vy', 'x0_err', 'vx_err', 'y0_err', 'vy_err', 't0']
    # Don't include n_vfit

    new_tab = startables.StarTable(name=tab['name'].data, 
                                   x=tab['x'][:,idx].data,
                                   y=tab['y'][:,idx].data,
                                   m=tab['m'][:,idx].data,
                                   xe=tab['xe'][:,idx].data,
                                   ye=tab['ye'][:,idx].data,
                                   me=tab['me'][:,idx].data,
                                   t=tab['t'][:,idx].data,                                
                                   x_orig=tab['x_orig'][:,idx].data,                                
                                   y_orig=tab['y_orig'][:,idx].data,                                
                                   m_orig=tab['m_orig'][:,idx].data,                                
                                   xe_orig=tab['xe_orig'][:,idx].data,                                
                                   ye_orig=tab['ye_orig'][:,idx].data,                                
                                   me_orig=tab['me_orig'][:,idx].data,                                  
                                   used_in_trans=tab['used_in_trans'][:,idx].data,                                
                                   m0=tab['m0'].data,
                                   m0e=tab['m0_err'].data,
                                   use_in_trans=tab['use_in_trans'].data,         
                                   x0=tab['x0'].data,
                                   vx=tab['vx'].data,
                                   y0=tab['y0'].data,
                                   vy=tab['vy'].data,   
                                   x0e=tab['x0_err'].data,
                                   vxe=tab['vx_err'].data,
                                   y0e=tab['y0_err'].data,
                                   vye=tab['vy_err'].data,
                                   t0=tab['t0'].data)

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

def calc_chi2(ref_mat, starlist_mat, transform, errs='both'):
    """
    calculate the chi2 and reduced chi2 of the position 
    between two matched starlists.
    Input:
    ref_mat: astropy table
        Reference starlist only containing matched stars that were used in the
        transformation. Standard column headers are assumed.
        
    starlist_mat: astropy table
        Transformed starlist only containing the matched stars used in
        the transformation. Standard column headers are assumed.

    transform: transformation object
        Transformation object of final transform. Used in chi-square
        determination

    errs: string; 'both', 'reference', or 'starlist'
        If both, add starlist errors in quadrature with reference errors.

        If reference, only consider reference errors. This should be used if the starlist
        does not have valid errors

        If starlist, only consider starlist errors. This should be used if the reference
        does not have valid errors

    Output:
    chi_sq: float
        chi2 = sum (diff_x**2 / xerr**2 + diff_y**2 /yerr**2)
    chi_sq_red: float
        reduced chi2 = chi2/ degree of freedom
    deg_freedom: int
        degree of freedom

    """
    diff_x = ref_mat['x'] - starlist_mat['x']
    diff_y = ref_mat['y'] - starlist_mat['y']

    # Set errors as per user input
    if errs == 'both':
        xerr = np.hypot(ref_mat['xe'], starlist_mat['xe'])
        yerr = np.hypot(ref_mat['ye'], starlist_mat['ye'])
    elif errs == 'reference':
        xerr = ref_mat['xe']
        yerr = ref_mat['ye']
    elif errs == 'starlist':
        xerr = starlist_mat['xe']
        yerr = starlist_mat['ye']
          

    # For both X and Y, calculate chi-square. Combine arrays to get combined
    # chi-square
    chi_sq_x = diff_x**2. / xerr**2.
    chi_sq_y = diff_y**2. / yerr**2.

    chi_sq = np.append(chi_sq_x, chi_sq_y)
    
    # Calculate degrees of freedom in transformation
    num_mod_params = calc_nparam(transform)
    deg_freedom = len(chi_sq) - num_mod_params
    
    # Calculate reduced chi-square
    chi_sq = np.sum(chi_sq)
    chi_sq_red = chi_sq / deg_freedom

    return chi_sq, chi_sq_red, deg_freedom


def calc_nparam(transformation):
    """
    calculate the degree of freedom for a transformation
    """
    # Read transformation: Extract X, Y coefficients from transform
    if transformation.__class__.__name__ == 'four_paramNW':
        nparam = 4
    elif transformation.__class__.__name__ == 'PolyTransform':
        order = transformation.order
        nparam = (order+1) * (order+2) 
    return nparam

def calc_F(red_chi2_1, red_chi2_2, v1, v2):
    """
    compare two different models to get the proper polynomial fitting order

    Input:
    red_chi2_1: reduced chi2 for the first model
    red_chi2_2: reduced chi2 for the second model
    v1 = degree of freedom for the first model
       = 2*(N_star_matched) - model_parameters
    v2 = degree of freedom for the second mdoel

    Output:
    P: The probability that the first model is better

    Example:
    for 1st order polynomial fitting:
        x' = a0 + a1*x + a2*y
        y' = b0 + b1*x + b2*y
        v1 = 2*N1 - 2*3 (2*: because x and y direction) 
        red_chi2_1 = chi2/v1
    for 2nd order polynomial fitting:
        x' = a0 + a1*x + a2*y + a3*x**2 + a4*y**2 + a5*x*y
        y' = b0 + b1*x + b2*y + b3*x**2 + b4*y**2 + b5*x*y
        v1 = 2*N1 - 2*6 
        red_chi2_2 = chi2/v2
    calc_F(red_chi2_1, red_chi2_2, v1, v2)
    
    ***Note***
    * make sure the first model is the simple model 
      and the second model is the more complicated model
    * the return value represents the probability that 
      the first model is better than the second model, in other words,
      the small P means the more colicated model is needed.
      the large P means the simple model is good enough.
    * normally, the P value will increase from model1->model2, to 
      model2->model3, to model3->model4. The user can decide a 
      critical value (eg, 0.7) to find the proper model.
    """

    f_value = red_chi2_1/red_chi2_2
    p = 1-f.cdf(f_value, v1, v2)
    return p
