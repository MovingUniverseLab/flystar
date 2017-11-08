import numpy as np
from flystar import match
from flystar import transforms
from astropy.table import Table, Column
import datetime
import copy
import os
import pdb

def mosaic_lists(list_of_starlists, ref_index=0, iters=2,
                     trans_input=None, trans_class=transforms.PolyTransform, trans_args={'order': 2}):
    """
    Parameters
    ----------
    list_of_starlists : array of StarList objects
        An array or list of flystar.starlists.StarList objects (which are Astropy Tables).
        There should be one for each starlist and they must contain 'x', 'y', and 'm' columns.

    ref_index : int
        The index of the reference epoch. (default = 0). Note that this is the reference
        list only for the first iteration. Subsequent iterations will utilize the sigma-clipped 
        mean of the positions from all the starlists. 

    iters : int
        The number of iterations used in the matching and transformation.  TO DO: INNER/OUTER? 

    trans_input : array or list of transform objects
        def = None. If not None, then this should contain an array or list of transform
        objects that will be used as the initial guess in the alignment and matching. 

    trans_class : transforms.Transform2D object (or subclass)
        The transform class that will be used to when deriving the optimal
        transformation parameters between each list and the reference list. 

    trans_args : dictionary
        A dictionary containing any extra keywords that are needed in the transformation object.
        For instance, "order".

    """
    # Shorthand:
    star_lists = list_of_starlists
    
    # Start with the reference list.... this will change and grow
    # over time, so make a copy.
    ref_table = Table(star_lists[ref_index])

    # Calculate N_lists and N_stars, where N_stars starts as the number in the
    # reference epoch. This will grow as we find new stars in the new epochs.
    N_lists = len(star_lists)
    N_stars = len(ref_table)

    # Save the reference index and number of epochs to the meta data on the
    # reference list for now. 
    ref_table.meta['L_REF'] = ref
    ref_table.meta['N_epochs'] = N_epochs

    # Keep a list of the transformation objects for each epochs.
    # Load up previous transformations, if they exist.
    trans_list = [None for ii in range(N_lists)]
    if input_transforms != None:
        trans_list = [trans_input[ii] for ii in range(N_lists)]
    
    for ii in range(len(star_lists)):
        star_list = star_lists[ii]
        trans = trans_list[ii]

        ref_list = ref_table.get_list(0)

        # Preliminary match and calculate a 1st order transformation. (if we haven't already).
        if trans == None:
            trans = align.initial_align(star_list, ref_list, briteN=50,
                                            transformModel=transforms.PolyTransform, order=1)
            
        # Repeat transform + match several times.
        for nn in range(iters):
            # Apply the transformation to the starlist.
            star_list_T = align.transform_from_object(star_list, trans)

            # Match stars between the lists.
            idx1, idx2, dm, dr = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                            ref_list['x'], ref_list['y'], ref_list['m'],
                                            dr_tol=5, dm_tol=2)
            print( 'In Loop ', nn, ' found ', len(idx1), ' matches' )
            
    
            # Calculate transform based on the matched stars    
            trans = trans_class.derive_transform(star_list['x'][idx1], star_list['y'][idx1],
                                                 ref_list['x'][idx2], ref_list['y'][idx2],
                                                 order=1)

        trans_list.append(trans)
        
        # The point is matching... lets do one final match.
        star_list_T = align.transform_from_object(star_list, trans)
        idx1, idx2, dm, dr = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                            ref_list['x'], ref_list['y'], ref_list['m'],
                                            dr_tol=3, dm_tol=2)

        # Add the matched stars to the reference table. These will be un-transformed positions.
        suffix = '_{0:03d}'.format(ii)
        ref_table.add_list(star_list[idx1])
        ref_table.meta['L' + suffix] = os.path.split(star_lists[ii])[-1]

        # Add the unmatched stars and grow the size of the table.
        

    # Clean up th extra copy of the reference star list:
    ref_list.remove_columns(['name', 't', 'x', 'y', 'm', 'xe', 'ye'])
    ref_list.remove_columns(['snr', 'corr', 'N_frames', 'flux'])

    ref_list.write(lis_dir + output_name, overwrite=True)

    return ref_list

def run_align_iter(catalog, trans_order=1, poly_deg=1, ref_mag_lim=19, ref_radius_lim=300):
    # Load up data with matched stars.
    d = Table.read(catalog)

    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # Determine how many stars there are. 
    N_stars = len(d)

    # Determine the reference epoch
    ref = d.meta['L_REF']

    # Figure out the number of free parameters for the specified
    # poly2d order.
    poly2d = models.Polynomial2D(trans_order)
    N_par_trans_per_epoch = 2.0 * poly2d.get_num_coeff(2)  # one poly2d for each dimension (X, Y)
    N_par_trans = N_par_trans_per_epoch * N_epochs

    ##########
    # First iteration -- align everything to REF epoch with zero velocities. 
    ##########
    print('ALIGN_EPOCHS: run_align_iter() -- PASS 1')
    ee_ref = d.meta['L_REF']

    target_name = 'OB120169'

    trans1, used1 = calc_transform_ref_epoch(d, target_name, ee_ref, ref_mag_lim, ref_radius_lim)

    ##########
    # Derive the velocity of each stars using the round 1 transforms. 
    ##########
    calc_polyfit_all_stars(d, poly_deg, init_fig_idx=0)

    calc_mag_avg_all_stars(d)
    
    tdx = np.where((d['name_0'] == 'OB120169') | (d['name_0'] == 'OB120169_L'))[0]
    print(d[tdx]['name_0', 't0', 'mag', 'x0', 'vx', 'x0e', 'vxe', 'chi2x', 'y0', 'vy', 'y0e', 'vye', 'chi2y', 'dof'])

    ##########
    # Second iteration -- align everything to reference positions derived from iteration 1
    ##########
    print('ALIGN_EPOCHS: run_align_iter() -- PASS 2')
    target_name = 'OB120169'

    trans2, used2 = calc_transform_ref_poly(d, target_name, poly_deg, ref_mag_lim, ref_radius_lim)
    
    ##########
    # Derive the velocity of each stars using the round 1 transforms. 
    ##########
    calc_polyfit_all_stars(d, poly_deg, init_fig_idx=4)

    ##########
    # Save output
    ##########
    d.write(catalog.replace('.fits', '_aln.fits'), overwrite=True)
    
    return

def calc_transform_ref_epoch(d, target_name, ee_ref, ref_mag_lim, ref_radius_lim):
    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # output array
    trans = []
    used = []

    # Find the target
    tdx = np.where(d['name_0'] == 'OB120169')[0][0]

    # Reference values
    t_ref = d['t_{0:d}'.format(ee_ref)]
    m_ref = d['m_{0:d}'.format(ee_ref)]
    x_ref = d['x_{0:d}'.format(ee_ref)]
    y_ref = d['y_{0:d}'.format(ee_ref)]
    xe_ref = d['xe_{0:d}'.format(ee_ref)]
    ye_ref = d['ye_{0:d}'.format(ee_ref)]
    
    # Calculate some quanitites we use for selecting reference stars.
    r_ref = np.hypot(x_ref - x_ref[tdx], y_ref - y_ref[tdx])

    # Loop through and align each epoch to the reference epoch.
    for ee in range(N_epochs):
        # Pull out the X, Y positions (and errors) for the two
        # starlists we are going to align.
        x_epo = d['x_{0:d}'.format(ee)]
        y_epo = d['y_{0:d}'.format(ee)]
        t_epo = d['t_{0:d}'.format(ee)]
        xe_epo = d['xe_{0:d}'.format(ee)]
        ye_epo = d['ye_{0:d}'.format(ee)]

        # Figure out the set of stars detected in both epochs.
        idx = np.where((t_ref != 0) & (t_epo != 0) & (xe_ref != 0) & (xe_epo != 0))[0]

        # Find those in both epochs AND reference stars. This is [idx][rdx]
        rdx = np.where((r_ref[idx] < ref_radius_lim) & (m_ref[idx] < ref_mag_lim))[0]
        
        # Average the positional errors together to get one weight per star.
        xye_ref = (xe_ref + ye_ref) / 2.0
        xye_epo = (xe_epo + ye_epo) / 2.0
        xye_wgt = (xye_ref**2 + xye_epo**2)**0.5
        
        # Calculate transform based on the matched stars    
        trans_tmp = transforms.PolyTransform(x_epo[idx][rdx], y_epo[idx][rdx], x_ref[idx][rdx], y_ref[idx][rdx],
                                                 weights=xye_wgt[idx][rdx], order=2)

        trans.append(trans_tmp)


        # Apply thte transformation to the stars positions and errors:
        xt_epo = np.zeros(len(d), dtype=float)
        yt_epo = np.zeros(len(d), dtype=float)
        xet_epo = np.zeros(len(d), dtype=float)
        yet_epo = np.zeros(len(d), dtype=float)
        
        xt_epo[idx], xet_epo[idx], yt_epo[idx], yet_epo[idx] = trans_tmp.evaluate_errors(x_epo[idx], xe_epo[idx],
                                                                                         y_epo[idx], ye_epo[idx],
                                                                                         nsim=100)

        d['xt_{0:d}'.format(ee)] = xt_epo
        d['yt_{0:d}'.format(ee)] = yt_epo
        d['xet_{0:d}'.format(ee)] = xet_epo
        d['yet_{0:d}'.format(ee)] = yet_epo

        # Record which stars we used in the transform.
        used_tmp = np.zeros(len(d), dtype=bool)
        used_tmp[idx[rdx]] = True

        used.append(used_tmp)

        if True:
            plot_quiver_residuals(xt_epo, yt_epo, x_ref, y_ref, idx, rdx, 'Epoch: ' + str(ee))
            
    used = np.array(used)
    
    return trans, used


def calc_transform_ref_poly(d, target_name, poly_deg, ref_mag_lim, ref_radius_lim):
    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # output array
    trans = []
    used = []

    # Find the target
    tdx = np.where(d['name_0'] == 'OB120169')[0][0]

    # Temporary Reference values
    t_ref = d['t0']
    m_ref = d['mag']
    x_ref = d['x0']
    y_ref = d['y0']
    xe_ref = d['x0e']
    ye_ref = d['y0e']    
    
    # Calculate some quanitites we use for selecting reference stars.
    r_ref = np.hypot(x_ref - x_ref[tdx], y_ref - y_ref[tdx])

    for ee in range(N_epochs):
        # Pull out the X, Y positions (and errors) for the two
        # starlists we are going to align.
        x_epo = d['x_{0:d}'.format(ee)]
        y_epo = d['y_{0:d}'.format(ee)]
        t_epo = d['t_{0:d}'.format(ee)]
        xe_epo = d['xe_{0:d}'.format(ee)]
        ye_epo = d['ye_{0:d}'.format(ee)]

        # Shift the reference position by the polyfit for each star.
        dt = t_epo - t_ref
        if poly_deg >= 0:
            x_ref_ee = x_ref
            y_ref_ee = y_ref
            xe_ref_ee = x_ref
            ye_ref_ee = y_ref
            
        if poly_deg >= 1:
            x_ref_ee += d['vx'] * dt
            y_ref_ee += d['vy'] * dt
            xe_ref_ee = np.hypot(xe_ref_ee, d['vxe'] * dt)
            ye_ref_ee = np.hypot(ye_ref_ee, d['vye'] * dt)

        if poly_deg >= 2:
            x_ref_ee += d['ax'] * dt
            y_ref_ee += d['ay'] * dt
            xe_ref_ee = np.hypot(xe_ref_ee, d['axe'] * dt)
            ye_ref_ee = np.hypot(ye_ref_ee, d['aye'] * dt)
            
        # Figure out the set of stars detected in both.
        idx = np.where((t_ref != 0) & (t_epo != 0) & (xe_ref != 0) & (xe_epo != 0))[0]

        # Find those in both AND reference stars. This is [idx][rdx]
        rdx = np.where((r_ref[idx] < ref_radius_lim) & (m_ref[idx] < ref_mag_lim))[0]
        
        # Average the positional errors together to get one weight per star.
        xye_ref = (xe_ref_ee + ye_ref_ee) / 2.0
        xye_epo = (xe_epo + ye_epo) / 2.0
        xye_wgt = (xye_ref**2 + xye_epo**2)**0.5
        
        # Calculate transform based on the matched stars    
        trans_tmp = transforms.PolyTransform(x_epo[idx][rdx], y_epo[idx][rdx], x_ref_ee[idx][rdx], y_ref_ee[idx][rdx],
                                                 weights=xye_wgt[idx][rdx], order=2)
        trans.append(trans_tmp)

        # Apply thte transformation to the stars positions and errors:
        xt_epo = np.zeros(len(d), dtype=float)
        yt_epo = np.zeros(len(d), dtype=float)
        xet_epo = np.zeros(len(d), dtype=float)
        yet_epo = np.zeros(len(d), dtype=float)
        
        xt_epo[idx], xet_epo[idx], yt_epo[idx], yet_epo[idx] = trans_tmp.evaluate_errors(x_epo[idx], xe_epo[idx],
                                                                                         y_epo[idx], ye_epo[idx],
                                                                                         nsim=100)
        d['xt_{0:d}'.format(ee)] = xt_epo
        d['yt_{0:d}'.format(ee)] = yt_epo
        d['xet_{0:d}'.format(ee)] = xet_epo
        d['yet_{0:d}'.format(ee)] = yet_epo

        # Record which stars we used in the transform.
        used_tmp = np.zeros(len(d), dtype=bool)
        used_tmp[idx[rdx]] = True

        used.append(used_tmp)

        if True:
            plot_quiver_residuals(xt_epo, yt_epo, x_ref_ee, y_ref_ee, idx, rdx, 'Epoch: ' + str(ee))

    used = np.array(used)
    
    return trans, used

def calc_polyfit_all_stars(d, poly_deg, init_fig_idx=0):
    # Determine how many stars there are. 
    N_stars = len(d)

    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])
    
    # Setup some variables to save the results
    t0_all = []
    px_all = []
    py_all = []
    pxe_all = []
    pye_all = []
    chi2x_all = []
    chi2y_all = []
    dof_all = []

    # Get the time array, which is the same for all stars.
    # Also, sort the time indices.
    t = np.array([d['t_{0:d}'.format(ee)][0] for ee in range(N_epochs)])
    tdx = t.argsort()
    t_sorted = t[tdx]
    
    # Run polyfit on each star.
    for ss in range(N_stars):
        # Get the x, y, xe, ye, and t arrays for this star.
        xt = np.array([d['xt_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        yt = np.array([d['yt_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        xet = np.array([d['xet_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        yet = np.array([d['yet_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        t_tmp = np.array([d['t_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])

        # Sort these arrays.
        xt_sorted = xt[tdx]
        yt_sorted = yt[tdx]
        xet_sorted = xet[tdx]
        yet_sorted = yet[tdx]
        t_tmp_sorted = t_tmp[tdx]

        # Get only the detected epochs.
        edx = np.where(t_tmp_sorted != 0)[0]

        # Calculate the weighted t0 (using the transformed errors).
        weight_for_t0 = 1.0 / np.hypot(xet_sorted, yet_sorted)
        t0 = np.average(t_sorted[edx], weights=weight_for_t0[edx])

        # for ee in edx:
        #     print('{0:8.3f}  {1:10.5f}  {2:10.5f}  {3:8.5f}  {4:8.5f}'.format(t[ee], xt[ee], yt[ee], xet[ee], yet[ee]))
        # pdb.set_trace()

        # Run polyfit
        dt = t_sorted - t0
        px, covx = np.polyfit(dt[edx], xt_sorted[edx], poly_deg, w=1./xet_sorted[edx], cov=True)
        py, covy = np.polyfit(dt[edx], yt_sorted[edx], poly_deg, w=1./yet_sorted[edx], cov=True)

        pxe = np.sqrt(np.diag(covx))
        pye = np.sqrt(np.diag(covy))


        x_mod = np.polyval(px, dt[edx])
        y_mod = np.polyval(py, dt[edx])
        chi2x = np.sum( ((x_mod - xt_sorted[edx]) / xet_sorted[edx])**2 )
        chi2y = np.sum( ((y_mod - yt_sorted[edx]) / yet_sorted[edx])**2 )
        dof = len(edx) - (poly_deg + 1)

        # Save results:
        t0_all.append(t0)
        px_all.append(px)
        py_all.append(py)
        pxe_all.append(pxe)
        pye_all.append(pye)
        chi2x_all.append(chi2x)
        chi2y_all.append(chi2y)
        dof_all.append(dof)

        if d[ss]['name_0'] in ['OB120169', 'OB120169_L']:
            gs = GridSpec(3, 2) # 3 rows, 1 column
            fig = plt.figure(ss + 1 + init_fig_idx, figsize=(12, 8))
            a0 = fig.add_subplot(gs[0:2, 0])
            a1 = fig.add_subplot(gs[2, 0])
            a2 = fig.add_subplot(gs[0:2, 1])
            a3 = fig.add_subplot(gs[2, 1])
            
            a0.errorbar(t_sorted[edx], xt_sorted[edx], yerr=xet_sorted[edx], fmt='ro')
            a0.plot(t_sorted[edx], x_mod, 'k-')
            a0.set_title(d[ss]['name_0'] + ' X')
            a1.errorbar(t_sorted[edx], xt_sorted[edx] - x_mod, yerr=xet_sorted[edx], fmt='ro')
            a1.axhline(0, linestyle='--')
            a1.set_xlabel('Time (yrs)')
            a2.errorbar(t_sorted[edx], yt_sorted[edx], yerr=yet_sorted[edx], fmt='ro')
            a2.plot(t_sorted[edx], y_mod, 'k-')
            a2.set_title(d[ss]['name_0'] + ' Y')
            a3.errorbar(t_sorted[edx], yt_sorted[edx] - y_mod, yerr=yet_sorted[edx], fmt='ro')
            a3.axhline(0, linestyle='--')
            a3.set_xlabel('Time (yrs)')

            

    t0_all = np.array(t0_all)
    px_all = np.array(px_all)
    py_all = np.array(py_all)
    pxe_all = np.array(pxe_all)
    pye_all = np.array(pye_all)
    chi2x_all = np.array(chi2x_all)
    chi2y_all = np.array(chi2y_all)
    dof_all = np.array(dof_all)
        
    # Done with all the stars... recast as numpy arrays and save to output table.
    d['t0'] = t0_all
    d['chi2x'] = chi2x_all
    d['chi2y'] = chi2y_all
    d['dof'] = dof_all
    if poly_deg >= 0:
        d['x0'] = px_all[:, -1]
        d['y0'] = py_all[:, -1]
        d['x0e'] = pxe_all[:, -1]
        d['y0e'] = pye_all[:, -1]
        
    if poly_deg >= 1:
        d['vx'] = px_all[:, -2]
        d['vy'] = py_all[:, -2]
        d['vxe'] = pxe_all[:, -2]
        d['vye'] = pye_all[:, -2]

    if poly_deg >= 2:
        d['ax'] = px_all[:, -3]
        d['ay'] = py_all[:, -3]
        d['axe'] = pxe_all[:, -3]
        d['aye'] = pye_all[:, -3]

    pdb.set_trace()
        
    return

def calc_mag_avg_all_stars(d):
    # Determine how many stars there are. 
    N_stars = len(d)

    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # 2D mag array
    mag_all = np.zeros((N_epochs, N_stars), dtype=float)

    for ee in range(N_epochs):
        mag_all[ee, :] = d['m_{0:d}'.format(ee)]

    mag_all_masked = np.ma.masked_equal(mag_all, 0)
    flux_all_masked = 10**(-mag_all_masked / 2.5)

    flux_avg = flux_all_masked.mean(axis=0)
    mag_avg = -2.5 * np.log10(flux_avg)

    d['mag'] = mag_avg

    return



def initial_align(table1, table2, briteN=100,
                      transformModel=transforms.PolyTransform, order=1, req_match=5):
    """
    Calculates an initial (unweighted) transformation from table1 starlist into
    table2 starlist (i.e., table2 is the reference starlist). Matching is done using
    a blind triangle-matching algorithm of the brightest briteN stars in both starlists.
    Transformation is done using the transformModel in the input parameter.

    Starlists must be astropy tables with standard column headers. All
    positions must be at the same epoch, and +x must be in the same
    direction.

    Standard column headers:
    name: name
    x: x position
    y: y position
    xe: error in x position
    ye: error in y position
    
    vx: proper motion in x direction
    vy proper motion in y direction
    vxe: error in x proper motion
    vye: error in y proper motion

    m: magnitude
    me: magnitude error
    
    t0: linear motion time zero point

    use: specify use in transformation
    

    Parameters:
    ----------
    -table1: astropy.table
        contains name,m,x,y,xe,ye,vx,vy,vxe,vye,t0.

    -table2: astropy.table
        contains name,m,x,y,xe,ye.
        this is the reference template

    -briteN: int
        The number of brightest stars used to match two starlists.

    -transformModel:  transformation model object (class)
        The transformation model class that will be instantiated to find the
        best-fit transformation parameters between matched table1 and table2.
        eg: transforms.four_paramNW, transforms.PolyTransform

    -order: int
         Order of the transformation. Not relevant for 4 parameter or spline fit

    -req_match: int
         Number of required matches of the input catalog to the total reference

    Output:
    ------
    Transformation object
    
    """
    # Extract necessary information from tables (x, y, m)
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']
    x2 = table2['x']
    y2 = table2['y']
    m2 = table2['m']

    # Run the blind triangle-matching algorithm to find the matches between the starlists
    print(( 'Attempting match with {0} and {1} stars from starlist1 and starlist2'.format(len(table1), len(table2))))
    print( 'Begin initial match')

    N, x1m, y1m, m1m, x2m, y2m, m2m = match.miracle_match_briteN(x1, y1, m1, x2, y2, m2, briteN)
    assert len(x1m) > req_match, 'Failed to find at least '+str(req_match)+' (only ' + str(len(x1m)) + ') matches, giving up'
    print(( '{0} stars matched between starlist1 and starlist2'.format(N)))

    # Calculate transformation based on matches
    t = transformModel.derive_transform(x1m, y1m ,x2m, y2m, order=order, weights=None)

    print( 'End initial match \n')
    return t



def transform_and_match(table1, table2, transform, dr_tol=1.0, dm_tol=None):
    """
    apply transformation to starlist1 and
    match stars to given radius and magnitude tolerance.

    Starlists must be astropy tables with standard columns names as specified
    in initial_align.

    Parameters:
   -----------
    -table1: astropy.table
        contains name,m,x,y,xe,ye,vx,vy,vxe,vye,t0.

    -table2: astropy.table
        contains name,m,x,y,xe,ye.
        this is the reference template

    -dr_tol: float (default=1.0)
        The search radius for the matching algorithm, in the same units as the
        starlist file positions.

    -transform: transformation object


    Output:
    -------
    -idx1: indicies of matched stars from table1
    -idx2: indicies of matched stars from tabel2
    """

    # Extract necessary information from tables (x, y, m)
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']
    x2 = table2['x']
    y2 = table2['y']
    m2 = table2['m']


    # Transform x, y coordinates from starlist 1 into starlist 2
    x1t, y1t = transform.evaluate(x1, y1)

    # Match starlist 1 and 2
    idx1, idx2, dr, dm = match.match(x1t, y1t, m1, x2, y2, m2, dr_tol, dm_tol)

    print(( '{0} of {1} stars matched'.format(len(idx1), len(x1t))))

    return idx1, idx2


def find_transform(table1, table1_trans, table2, transModel=transforms.PolyTransform, order=1,
                weights=None):
    """
    Given a matched starlist, derive a new transform. This transformation is
    calculated for starlist 1 into starlist 2

    Parameters:
    -----------
    table1: astropy table
        Table which we have calculated the transformation for, trimmed to only
        stars which match with table2. Original coords, not transformed into
        reference frame.

    table1_trans: astropy table
        Table which we calculated the transformation fo, trimmed to only
        stars which match with table2. Contains transformed coords. Only
        used when calculating weights.

    table2: astropy table
        Table with the reference starlist. Trimmed to only stars which
        match table1.

    trans: transformation object
        Transformation used to transform table1 coords in transform_and_match
        in order to do the star matching.

    transModel: transformation class (default: transform.four_paramNW)
        Desired transform to apply to matched stars, e.g. four_paramNW or PolyTransform.
        If PolyTransform is selected, order defines the order of polynomial used

    order: int (default=1)
        Order of polynomial to use in the transformation. Only active if
        PolyTransform is selected

    weights: string (default=None)
        if weights=='both', we use both position error in transformed starlist and
        reference starlist as uncertanty. And weights is the reciprocal of this uncertanty.
        if weights=='starlist', we only use postion error in transformed starlist.
        if weights=='reference', we only use position error in reference starlist.
        if weights==None, we don't use weights.

    Output:
    ------
    -transformation object
    -number of stars used in transform
    """
    # First, check that desired transform is supported
    if ( (transModel != transforms.four_paramNW) &
         (transModel != transforms.PolyTransform) &
         (transModel != transforms.Shift) &
         (transModel != transforms.LegTransform) ):
        print(( '{0} not supported yet!'.format(transModel)))
        return
    
    # Extract *untransformed* coordinates from starlist 1
    # and the matching coordinates from starlist 2
    x1 = table1['x']
    y1 = table1['y']
    x2 = table2['x']
    y2 = table2['y']

    # calculate weights from *transformed* coords. This is where we use the
    # transformation object
    if ('xe' in table1_trans.colnames):
        x1e = table1_trans['xe']
        y1e = table1_trans['ye']

    if 'xe' in table2.colnames:
        x2e = table2['xe']
        y2e = table2['ye']

    # Calculate weights as to user specification
    if weights=='both':
        weight = 1/np.sqrt( x1e**2 + y1e**2 + x2e**2 + y2e**2)
    elif weights=='starlist':
        weight = 1/np.sqrt( x1e**2 + y1e**2 )
    elif weights=='reference':
        weight = 1/np.sqrt( x2e**2 +  y2e**2)
    else:
        weight = None

    # Calculate transform based on the matched stars
    t = transModel.derive_transform(x1, y1, x2, y2, order)

    N_trans = len(x1)
    print(( '{0} stars used in transform\n'.format(N_trans)))

    # Ret3urn transformation object and number of stars used in transform
    return t, N_trans


def find_transform_new(table1_mat, table2_mat,
                       transModel=transforms.four_paramNW, order=1,
                       weights=None, transInit=None):
    """
    Given a matched starlist, derive a new transform. This transformation is
    calculated for starlist 1 into starlist 2

    Parameters:
    -----------
    table1_mat: astropy table
        Table with matched stars from starlist 1, with original positions
        (not transformed into starlist 2 frame)

    table2_mat: astropy table
        Table with matched stars from starlist 2, in starlist 2 frame.

    transModel: transformation class (default: transform.four_paramNW)
        Specify desired transform, e.g. four_paramNW or PolyTransform. If
        PolyTransform is selected, order defines the order of polynomial used

    order: int (default=1)
        Order of polynomial to use in the transformation. Only active if
        PolyTransform is selected

    weights: string (default=None)
        if weights=='both', we use position error  in transformed
        starlist and reference starlist as uncertanties. And weights is the reciprocal
            of this uncertanty.
        if weights=='starlist', we only use postion error and velocity error in transformed
        starlist as uncertainty.
        if weights=='reference', we only use position error in reference starlist as uncertainty.
        if weights==None, we don't use weights.

    transInit: Transform Object (default=None)
        if weights = 'both' or 'starlist' then the positions in table 1 are first transformed
        using the transInit object. This is necessary if the plate scales are very different
        between the table 1 and the reference list.

    Output:
    ------
    -transformation object
    -number of stars used in transform
    """
    # First, check that desired transform is supported
    if ( (transModel != transforms.four_paramNW) & (transModel != transforms.PolyTransform) ):
        print(( '{0} not supported yet!'.format(transModel)))
        return
    
    # Extract *untransformed* coordinates from starlist 1
    # and the matching coordinates from starlist 2
    x1 = table1_mat['x']
    y1 = table1_mat['y']
    x2 = table2_mat['x']
    y2 = table2_mat['y']

    # Get the uncertainties (if needed) and calculate the weights.
    if weights != None:
        x1e = table1_mat['xe']
        y1e = table1_mat['ye']
        x2e = table2_mat['xe']
        y2e = table2_mat['ye']

        if transInit != None:
            table1T_mat = table1_mat.copy()
            table1T_mat = transform_by_object(table1T_mat, transInit)

            x1e = table1T_mag['xe']
            y1e = table1T_mag['ye']

        # Calculate weights as to user specification
        if weights == 'both':
            weight = 1.0 / ( x1e**2 + x2e**2 + y1e**2 + y2e**2 )
        elif weights == 'starlist':
            weight = 1.0 / ( x1e**2 + y1e**2 )
        elif weights == 'reference':
            weight = 1.0 / ( x2e**2 +  y2e**2 )
        else:
            weight = None

    # Calculate transform based on the matched stars
    t = transModel(x1, y1, x2, y2, order=order, weights=weight)

    N_trans = len(x1)
    print(( '{0} stars used in transform\n'.format(N_trans)))

    # Return transformation object and number of stars used in transform
    return t, N_trans


def write_transform(transform, starlist, reference, N_trans, deltaMag=0, restrict=False, weights=None,
                    outFile='outTrans.txt'):
    """
    Given a transformation object, write out the coefficients in a java align
    readable format. Outfile name is specified by user.

    Coefficients are output in file in the following way:
    x' = a0 + a1*x + a2*y + a3*x**2. + a4*x*y  + a5*y**2. + ...
    y' = b0 + b1*x + b2*y + b3*x**2. + b4*x*y  + b5*y**2. + ...

    Parameters:
    ----------
    transform: transformation object
        Transformation object we want to feed into java align

    starlist: string
        File name of starlist; this is the starlist the transformation should
        be applied to. For output purposes only

    reference: string
        File name of reference; this is what the starlist is transformed to.
        For output purposes only

    N_trans: int
        Number of stars used in the transformation

    deltaMag: float (default = 0)
        Average magnitude difference between reference and starlist
        (reference - starlist)

    restrict: boolean (default=False)
        Set to True if transformation restricted to stars with use > 2. Purely
        for output purposes

    weights: string (default=None)
        if weights=='both', we use both position error and velocity error in transformed
        starlist and reference starlist as uncertanties. And weights is the reciprocal
            of this uncertanty.
        if weights=='starlist', we only use postion error and velocity error in transformed
        starlist as uncertainty.
        if weights=='reference', we only use position error in reference starlist as uncertainty
        if weights==None, we don't use weights.

    outFile: string (default: 'outTrans.txt')
        Name of output text file
        
    Output:
    ------
    txt file with the file name outFile
    """
    # Extract info about transformation
    trans_name = transform.__class__.__name__
    trans_order = transform.order
    
    # Extract X, Y coefficients from transform
    if trans_name == 'four_paramNW':
        Xcoeff = transform.px
        Ycoeff = transform.py
    elif trans_name == 'PolyTransform':
        Xcoeff = transform.px.parameters
        Ycoeff = transform.py.parameters
    else:
        print(( '{0} not yet supported!'.format(transType)))
        return
        
    # Write output
    _out = open(outFile, 'w')
    
    # Write the header. DO NOT CHANGE, HARDCODED IN JAVA ALIGN
    _out.write('## Date: {0}\n'.format(datetime.date.today()) )
    _out.write('## File: {0}, Reference: {1}\n'.format(starlist, reference) )
    _out.write('## Directory: {0}\n'.format(os.getcwd()) )
    _out.write('## Transform Class: {0}\n'.format(transform.__class__.__name__))
    _out.write('## Order: {0}\n'.format(transform.order))
    _out.write('## Restrict: {0}\n'.format(restrict))
    _out.write('## Weights: {0}\n'.format(weights))
    _out.write('## N_coeff: {0}\n'.format(len(Xcoeff)))
    _out.write('## N_trans: {0}\n'.format(N_trans))
    _out.write('## Delta Mag: {0}\n'.format(deltaMag))
    _out.write('{0:16s} {1:16s}\n'.format('# Xcoeff', 'Ycoeff'))
    
    # Write the coefficients such that the orders are together as defined in
    # documentation. This is a pain because PolyTransform output is weird.
    # (see astropy Polynomial2D documentation)
    if (trans_name == 'four_paramNW'):
        for i in range(len(Xcoeff)):
            _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[i], Ycoeff[i]) )
    elif (trans_name == 'PolyTransform'):
        # CODE TO GET INDICIES
        N = trans_order - 1
        idx_list = list()
        
        # when trans_order=1, N=0
        idx_list.append(0)
        idx_list.append(1)
        idx_list.append(N+2)
        
        if trans_order >= 2:
            for k in range(2, N+2):
                idx_list.append(k)
                for j in range(1, k):
                    i = k-j
                    idx_list.append(int(2*N +2 +j + (2*N+2-i)*(i-1)/2.))
                idx_list.append(N+1+k)

        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[0], Ycoeff[0]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[1], Ycoeff[1]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[3], Ycoeff[3]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[2], Ycoeff[2]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[5], Ycoeff[5]) )
        #_out.write('{0:16.6e}  {1:16.6e}'.format(Xcoeff[4], Ycoeff[4]) )

        for i in idx_list:
            _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[i], Ycoeff[i]) )


    _out.close()
    
    return


def transform_from_file(starlist, transFile):
    """
    Apply transformation from transFile to starlist. Returns astropy table with
    added columns with the transformed coordinates. NOTE: Transforms
    positions/position errors, plus velocities and velocity errors if they
    are present in starlist.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS
    
    Parameters:
    ----------
    starlist: astropy table
         Starlist we want to apply the transformation too. Must already
         have standard column headers

    transFile: ascii file
        File with the transformation coefficients. Assumed to be output of
        write_transform, with coefficients specified as code documents

    Output:
    ------
    Copy of starlist astropy table with transformed coordinates.
    """
    # Make a copy of starlist. This is what we will eventually modify with
    # the transformed coordinates
    starlist_f = copy.deepcopy(starlist)
    
    # Check to see if velocities are present in starlist. If so, we will
    # need to transform these as well as positions
    vel = False
    keys = list(starlist.keys())
    if 'vx' in keys:
        vel = True
    
    # Extract needed information from starlist
    x_orig = starlist['x']
    y_orig = starlist['y']
    xe_orig = starlist['xe']
    ye_orig = starlist['ye']

    if vel:
        x0_orig = starlist['x0']
        y0_orig = starlist['y0']
        x0e_orig = starlist['x0e']
        y0e_orig = starlist['y0e']
        
        vx_orig = starlist['vx']
        vy_orig = starlist['vy']
        vxe_orig = starlist['vxe']
        vye_orig = starlist['vye']
    
    # Read transFile
    trans = Table.read(transFile, format='ascii.commented_header', header_start=-1)
    Xcoeff = trans['Xcoeff']
    Ycoeff = trans['Ycoeff']

    #-----------------------------------------------#
    # General equation for applying the transform
    #-----------------------------------------------#
    #"""
    # First determine the order based on the number of terms
    # Comes from Nterms = (N+1)*(N+2) / 2.
    order = (np.sqrt(1 + 8*len(Xcoeff)) - 3) / 2.

    if order%1 != 0:
        print( 'Incorrect number of coefficients for polynomial')
        print( 'Stopping')
        return
    order = int(order)

    # Position transformation
    x_new, y_new = transform_pos_from_file(Xcoeff, Ycoeff, order, x_orig,
                                           y_orig)
    
    if vel:
        x0_new, y0_new = transform_pos_from_file(Xcoeff, Ycoeff, order, x0_orig,
                                             y0_orig)

    # Position error transformation
    xe_new, ye_new = transform_poserr_from_file(Xcoeff, Ycoeff, order, xe_orig,
                                                ye_orig, x_orig, y_orig)

    if vel:
        x0e_new, y0e_new = transform_poserr_from_file(Xcoeff, Ycoeff, order, x0e_orig,
                                                y0e_orig, x0_orig, y0_orig)

    if vel:
        # Velocity transformation
        vx_new, vy_new = transform_vel_from_file(Xcoeff, Ycoeff, order, vx_orig,
                                                 vy_orig, x_orig, y_orig)
            
        # Velocity error transformation
        vxe_new, vye_new = transform_velerr_from_file(Xcoeff, Ycoeff, order,
                                                      vxe_orig, vye_orig,
                                                      vx_orig, vy_orig,
                                                      xe_orig, ye_orig,
                                                      x_orig, y_orig)

    #----------------------------------------#
    # Hard coded example: old but functional
    #----------------------------------------#
    """
    # How the transformation is applied depends on the type of transform.
    # This can be determined by the length of Xcoeff, Ycoeff
    if len(Xcoeff) == 3:
        x_new = Xcoeff[0] + Xcoeff[1] * x_orig + Xcoeff[2] * y_orig
        y_new = Ycoeff[0] + Ycoeff[1] * x_orig + Ycoeff[2] * y_orig
        xe_new = np.sqrt( (Xcoeff[1] * xe_orig)**2 + (Xcoeff[2] * ye_orig)**2 )
        ye_new = np.sqrt( (Ycoeff[1] * xe_orig)**2 + (Ycoeff[2] * ye_orig)**2 )

        if vel:
            vx_new = Xcoeff[1] * vx_orig + Xcoeff[2] * vy_orig
            vy_new = Ycoeff[1] * vx_orig + Ycoeff[2] * vy_orig
            vxe_new = np.sqrt( (Xcoeff[1] * vxe_orig)**2 + (Xcoeff[2] * vye_orig)**2 )
            vye_new = np.sqrt( (Ycoeff[1] * vxe_orig)**2 + (Ycoeff[2] * vye_orig)**2 )

    elif len(Xcoeff) == 6:
        x_new = Xcoeff[0] + Xcoeff[1]*x_orig + Xcoeff[3]*x_orig**2 + Xcoeff[2]*y_orig + \
                Xcoeff[5]*y_orig**2. + Xcoeff[4]*x_orig*y_orig
          
        y_new = Ycoeff[0] + Ycoeff[1]*x_orig + Ycoeff[3]*x_orig**2 + Ycoeff[2]*y_orig + \
                Ycoeff[5]*y_orig**2. + Ycoeff[4]*x_orig*y_orig
          
        xe_new = np.sqrt( (Xcoeff[1] + 2*Xcoeff[3]*x_orig + Xcoeff[4]*y_orig)**2 * xe_orig**2 + \
                          (Xcoeff[2] + 2*Xcoeff[5]*y_orig + Xcoeff[4]*x_orig)**2 * ye_orig**2 )
          
        ye_new = np.sqrt( (Ycoeff[1] + 2*Ycoeff[3]*x_orig + Ycoeff[4]*y_orig)**2 * xe_orig**2 + \
                          (Ycoeff[2] + 2*Ycoeff[5]*y_orig + Ycoeff[4]*x_orig)**2 * ye_orig**2 )

        if vel:
            vx_new = Xcoeff[1]*vx_orig + 2*Xcoeff[3]*x_orig*vx_orig + Xcoeff[2]*vy_orig + \
                    2.*Xcoeff[5]*y_orig*vy_orig + Xcoeff[4]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vy_new = Ycoeff[1]*vx_orig + 2*Ycoeff[3]*x_orig*vx_orig + Ycoeff[2]*vy_orig + \
                    2.*Ycoeff[5]*y_orig*vy_orig + Ycoeff[4]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vxe_new = np.sqrt( (Xcoeff[1] + 2*Xcoeff[3]*x_orig + Xcoeff[4]*y_orig)**2 * vxe_orig**2 + \
                               (Xcoeff[2] + 2*Xcoeff[5]*y_orig + Xcoeff[4]*x_orig)**2 * vye_orig**2 + \
                               (2*Xcoeff[3]*vx_orig + Xcoeff[4]*vy_orig)**2 * xe_orig**2 + \
                               (2*Xcoeff[5]*vy_orig + Xcoeff[4]*vx_orig)**2 * ye_orig**2 )
                               
            vye_new = np.sqrt( (Ycoeff[1] + 2*Ycoeff[3]*x_orig + Ycoeff[4]*y_orig)**2 * vxe_orig**2 + \
                               (Ycoeff[2] + 2*Ycoeff[5]*y_orig + Ycoeff[4]*x_orig)**2 * vye_orig**2 + \
                               (2*Ycoeff[3]*vx_orig + Ycoeff[4]*vy_orig)**2 * xe_orig**2 + \
                               (2*Ycoeff[5]*vy_orig + Ycoeff[4]*vx_orig)**2 * ye_orig**2 )
    """
    #Update transformed coords to copy of astropy table
    starlist_f['x'] = x_new
    starlist_f['y'] = y_new
    starlist_f['xe'] = xe_new
    starlist_f['ye'] = ye_new
    
    if vel:
        starlist_f['x0'] = x0_new
        starlist_f['y0'] = y0_new
        starlist_f['x0e'] = x0e_new
        starlist_f['y0e'] = y0e_new
        starlist_f['vx'] = vx_new
        starlist_f['vy'] = vy_new
        starlist_f['vxe'] = vxe_new
        starlist_f['vye'] = vye_new

    return starlist_f


def transform_from_object(starlist, transform):
    """
    Apply transformation to starlist. Returns astropy table with
    transformed positions/position errors, velocities and velocity errors
    if they are present in starlits
    
    Parameters:
    ----------
    starlist: astropy table
         Starlist we want to apply the transformation too. Must already
         have standard column headers
         x0, y0, x0e, y0e, vx, vy, vxe, vye, x, y, xe, ye

    transform: transformation object

    Output:
    ------
    Copy of starlist astropy table with transformed x0, y0, x0e, y0e,
    vx, vy, vxe, vye, x, y, xe, ye

    """
    # Make a copy of starlist. This is what we will eventually modify with
    # the transformed coordinates
    starlist_f = copy.deepcopy(starlist)

    # Check to see if velocities are present in starlist. If so, we will
    # need to transform these as well as positions
    vel = False
    keys = list(starlist.keys())
    if 'vx' in keys:
        vel = True
    
    # Extract needed information from starlist
    x = starlist_f['x']
    y = starlist_f['y']
    xe = starlist_f['xe']
    ye = starlist_f['ye']

    if vel:
        x0 = starlist_f['x0']
        y0 = starlist_f['y0']
        x0e = starlist_f['x0e']
        y0e = starlist_f['y0e']
        vx = starlist_f['vx']
        vy = starlist_f['vy']
        vxe = starlist_f['vxe']
        vye = starlist_f['vye']
    
    # calculate the transformed position and velocity

    # (x_new, y_new, xe_new, ye_new) in (x,y)
    x_new, y_new, xe_new, ye_new = position_transform_from_object(x, y, xe, ye, transform)

    
    if vel:
        # (x0_new,  y0_new, x0e_new, y0e_new) in (x0, y0, x0e, y0e)
        x0_new, y0_new, x0e_new, y0e_new = position_transform_from_object(x0, y0, x0e, y0e, transform)
        # (vx_new, vy_new, vxe_new, vye_new) in (x0, y0, x0e, y0e, vx, vy, vxe, vye)
        vx_new, vy_new, vxe_new, vye_new = velocity_transform_from_object(x0, y0, x0e, y0e, vx, vy, vxe, vye, transform)

    # update transformed coords to copy of astropy table
    starlist_f['x'] = x_new
    starlist_f['y'] = y_new
    starlist_f['xe'] = xe_new
    starlist_f['ye'] = ye_new
    
    if vel:
        starlist_f['x0'] = x0_new
        starlist_f['y0'] = y0_new
        starlist_f['x0e'] = x0e_new
        starlist_f['y0e'] = y0e_new
        starlist_f['vx'] = vx_new
        starlist_f['vy'] = vy_new
        starlist_f['vxe'] = vxe_new
        starlist_f['vye'] = vye_new
        
    return starlist_f





def position_transform_from_object(x, y, xe, ye, transform):
    """
    given the orginal position and position error, calculate the transformed
    position and position error based on transformation object from
    astropy.modeling.models.polynomial2D.
    Input:
        - x, y: original position
        - xe, ye: original position error
        - transform: transformation object from astropy.modeling.models.polynomial2D
        
    Outpus:
        - x_new, y_new: transformed position
        - xe_new, ye_new: transformed position error
    """

    # Read transformation: Extract X, Y coefficients from transform
    if transform.__class__.__name__ == 'four_paramNW':
        Xcoeff = transform.px
        Ycoeff = transform.py
        order = 1
    elif (transform.__class__.__name__ == 'PolyTransform'):
        Xcoeff = transform.px.parameters
        Ycoeff = transform.py.parameters
        order = transform.order
    else:
        txt = 'Transform not yet supported by position_transform_from_object'
        raise StandardError(txt)
        
    # How the transformation is applied depends on the type of transform.
    # This can be determined by the length of Xcoeff, Ycoeff
    N = order - 1

    x_new = 0
    for i in range(0, N+2):
        x_new += Xcoeff[i] * (x**i)
    for j in range(1, N+2):
        x_new += Xcoeff[N+1+j] * (y**j)
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            x_new += Xcoeff[sub] * (x**i) * (y**j)

    y_new = 0
    for i in range(0, N+2):
        y_new += Ycoeff[i] * (x**i)
    for j in range(1, N+2):
        y_new += Ycoeff[N+1+j] * (y**j)
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            y_new += Ycoeff[sub] * (x**i) * (y**j)

    # xe_new & ye_new in (x,y,xe,ye)
    xe_new = 0
    temp1 = 0
    temp2 = 0
    for i in range(1, N+2):
        temp1 += i * Xcoeff[i] * (x**(i-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            temp1 += i * Xcoeff[sub] * (x**(i-1)) * (y**j)
    for j in range(1, N+2):
        temp2 += j * Xcoeff[N+1+j] * (y**(j-1))
    for i in range(1, N+1):
        for j in range(N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            temp2 += j * Xcoeff[sub] * (x**i) * (y**(j-1))
    xe_new = np.sqrt((temp1*xe)**2 + (temp2*ye)**2)

    ye_new = 0
    temp1 = 0
    temp2 = 0
    for i in range(1, N+2):
        temp1 += i * Ycoeff[i] * (x**(i-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            temp1 += i * Ycoeff[sub] * (x**(i-1)) * (y**j)
    for j in range(1, N+2):
        temp2 += j * Ycoeff[N+1+j] * (y**(j-1))
    for i in range(1, N+1):
        for j in range(N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            temp2 += j * Ycoeff[sub] * (x**i) * (y**(j-1))
    ye_new = np.sqrt((temp1*xe)**2 + (temp2*ye)**2)

    return x_new, y_new, xe_new, ye_new


def velocity_transform_from_object(x0, y0, x0e, y0e, vx, vy, vxe, vye, transform):
    """
    given the orginal position & position error & velocity & veolicty error,
    calculat the transformed velocity and velocity error based on transformation
    from astropy.modling.models.polynomial2D.
    Input:
        - x0, y0, x0e, y0e: original position and position error
        - vx, vy, vxe, vye: original velocity and velocity error
        - transform: transformation object from astropy.modeling.models.polynomial2D
        
    Outpus:
        - vx_new, vy_new, vxe_new, vye_new: transformed velocity and velocity error
    """
 
    # Read transformation: Extract X, Y coefficients from transform
    if transform.__class__.__name__ == 'four_paramNW':
        Xcoeff = transform.px
        Ycoeff = transform.py
        order = 1
    elif transform.__class__.__name__ == 'PolyTransform':
        Xcoeff = transform.px.parameters
        Ycoeff = transform.py.parameters
        order = transform.order
    else:
        txt = 'Transform not yet supported by velocity_transform_from_object'
        raise StandardError(txt)
        
    # How the transformation is applied depends on the type of transform.
    # This can be determined by the length of Xcoeff, Ycoeff
    N = order - 1

    # vx_new & vy_new
    vx_new = 0
    for i in range(1, N+2):
        vx_new += i * Xcoeff[i] * (x0**(i-1)) * vx
    for j in range(1, N+2):
        vx_new += j * Xcoeff[N+1+j] * (y0**(j-1)) * vy
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            vx_new += i * Xcoeff[sub] * (x0**(i-1)) * (y0**j) * vx
            vx_new += j * Xcoeff[sub] * (x0**i) * (y0**(j-1)) * vy

    vy_new = 0
    for i in range(1, N+2):
        vy_new += i * Ycoeff[i] * (x0**(i-1)) * vx
    for j in range(1, N+2):
        vy_new += j * Ycoeff[N+1+j] * (y0**(j-1)) * vy
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            vy_new += i * Ycoeff[sub] * (x0**(i-1)) * (y0**j) * vx
            vy_new += j * Ycoeff[sub] * (x0**i) * (y0**(j-1)) * vy

    # vxe_new & vye_new
    vxe_new = 0
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i in range(2, N+2):
        temp1 += i * (i-1) * Xcoeff[i] * (x0**(i-2)) * vx
    for i in range(2, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp1 += i * (i-1) * Xcoeff[sub] * (x0**(i-2)) * (y0**j) * vx
    for i in range(1,N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp1 += i * j * Xcoeff[sub] * (x0**(i-1)) * (y0**(j-1)) * vy

    for j in range(2, N+2):
        temp2 += j * (j-1) * Xcoeff[N+1+j] * (y0**(j-2)) * vy
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp2 += i * j * Xcoeff[sub] * (x0**(i-1)) * (y0**(j-1)) * vx
    for i in range(1, N+1):
        for j in range(2, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp2 += j * (j-1) * Xcoeff[sub] * (x0**i) * (y0**(j-2)) * vy

    for i in range(1, N+2):
        temp3 += i * Xcoeff[i] * (x0**(i-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp3 += i * Xcoeff[sub] * (x0**(i-1)) * (y0**j)

    for j in range(1, N+2):
        temp4 += j * Xcoeff[N+1+j] * (y0**(j-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp4 += j * Xcoeff[sub] * (x0**i) * (y0**(j-1))

    vxe_new = np.sqrt((temp1*x0e)**2 + (temp2*y0e)**2 + (temp3*vxe)**2 + (temp4*vye)**2)


    vye_new = 0
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i in range(2, N+2):
        temp1 += i * (i-1) * Ycoeff[i] * (x0**(i-2)) * vx
    for i in range(2, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp1 += i * (i-1) * Ycoeff[sub] * (x0**(i-2)) * (y0**j) * vx
    for i in range(1,N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp1 += j * i * Ycoeff[sub] * (x0**(i-1)) * (y0**(j-1)) * vy

    for j in range(2, N+2):
        temp2 += j * (j-1) * Ycoeff[N+1+j] * (y0**(j-2)) * vy
    for i in range(1, N+1):
        for j in range(2, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp2 += j * (j-1) * Ycoeff[sub] * (x0**i) * (y0**(j-2)) * vy
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp2 += i * j * Ycoeff[sub] * (x0**(i-1)) * (y0**(j-1)) * vx

    for i in range(1, N+2):
        temp3 += i * Ycoeff[i] * (x0**(i-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp3 += i * Ycoeff[sub] * (x0**(i-1)) * (y0**j)

    for j in range(1, N+2):
        temp4 += j * Ycoeff[N+1+j] * (y0**(j-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp4 += j * Ycoeff[sub] * (x0**i) * (y0**(j-1))

    vye_new = np.sqrt((temp1*x0e)**2 + (temp2*y0e)**2 + (temp3*vxe)**2 + (temp4*vye)**2)

    return vx_new, vy_new, vxe_new, vye_new


def transform_pos_from_file(Xcoeff, Ycoeff, order, x_orig, y_orig):
    """
    Given the read-in coefficients from transform_from_file, apply the
    transformation to the observed positions. This is generalized to
    work with any order polynomial transform.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS

    Parameters:
    ----------
    Xcoeff: Array
        Array with the coefficients of the X pos transformation

    Ycoeff: Array
        Array with the coefficients of the Y pos transformation

    order: int
        Order of transformation

    x_orig: array
        Array with the original X positions

    y_orig: array
        Array with the original Y positions

    Output:
    ------
    x_new: array
       Transformed X positions

    y_new: array
        Transformed Y positions

    """
    idx = 0 # coeff index
    x_new = 0.0
    y_new = 0.0
    for i in range(order+1):
        for j in range(i+1):
            x_new += Xcoeff[idx] * x_orig**(i-j) * y_orig**j
            y_new += Ycoeff[idx] * x_orig**(i-j) * y_orig**j

            idx += 1

    return x_new, y_new

def transform_poserr_from_file(Xcoeff, Ycoeff, order, xe_orig, ye_orig, x_orig, y_orig):
    """
    Given the read-in coefficients from transform_from_file, apply the
    transformation to the observed position errors. This is generalized to
    work with any order transform.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS

    Parameters:
    ----------
    Xcoeff: Array
        Array with the coefficients of the X pos transformation

    Ycoeff: Array
        Array with the coefficients of the Y pos transformation

    order: int
        Order of transformation

    xe_orig: array
        Array with the original X position errs

    ye_orig: array
        Array with the original Y position errs
        
    x_orig: array
        Array with the original X positions

    y_orig: array
        Array with the original Y positions

    Output:
    ------
    xe_new: array
       Transformed X position errs

    ye_new: array
        Transformed Y position errs
    """
    idx = 0 # coeff index
    xe_new_sq = 0.0
    ye_new_sq = 0.0
    # First loop: dx'/dx
    for i in range(order+1):
        for j in range(i+1):
            xe_new_sq += (Xcoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j)**2. * xe_orig**2.
            ye_new_sq += (Ycoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j)**2. * xe_orig**2.

            idx += 1
            
    # Second loop: dy'/dy
    idx = 0 # coeff index
    for i in range(order+1):
        for j in range(i+1):
            xe_new_sq += (Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1))**2. * ye_orig**2.
            ye_new_sq += (Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1))**2. * ye_orig**2.

            idx += 1
    # Take square root for xe/ye_new
    xe_new = np.sqrt(xe_new_sq)
    ye_new = np.sqrt(ye_new_sq)

    return xe_new, ye_new

def transform_vel_from_file(Xcoeff, Ycoeff, order, vx_orig, vy_orig, x_orig, y_orig):
    """
    Given the read-in coefficients from transform_from_file, apply the
    transformation to the observed proper motions. This is generalized to
    work with any order transform.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS
    
    Parameters:
    ----------
    Xcoeff: Array
        Array with the coefficients of the X pos transformation

    Ycoeff: Array
        Array with the coefficients of the Y pos transformation

    order: int
        Order of transformation

    vx_orig: array
        Array with the original X proper motions

    vy_orig: array
        Array with the original Y proper motions
        
    x_orig: array
        Array with the original X positions

    y_orig: array
        Array with the original Y positions

    Output:
    ------
    vx_new: array
       Transformed X proper motions

    vy_new: array
        Transformed Y proper motions
    """
    idx = 0 # coeff index
    vx_new = 0.0
    vy_new = 0.0
    # First loop: dx'/dx
    for i in range(order+1):
        for j in range(i+1):
            vx_new += Xcoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j * vx_orig
            vy_new += Ycoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j * vx_orig
            
            idx += 1
    # Second loop: dy'/dy
    idx = 0 # coeff index
    for i in range(order+1):
        for j in range(i+1):
            vx_new += Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1) * vy_orig
            vy_new += Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1) * vy_orig

            idx += 1

    return vx_new, vy_new

def transform_velerr_from_file(Xcoeff, Ycoeff, order, vxe_orig, vye_orig, vx_orig,
                                vy_orig, xe_orig, ye_orig, x_orig, y_orig):
    """
    Given the read-in coefficients from transform_from_file, apply the
    transformation to the observed proper motion errors. This is generalized to
    work with any order transform.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS
    
    Parameters:
    ----------
    Xcoeff: Array
        Array with the coefficients of the X pos transformation

    Ycoeff: Array
        Array with the coefficients of the Y pos transformation

    order: int
        Order of transformation

    vxe_orig: array
        Array with the original X proper motion errs

    vye_orig: array
        Array with the original Y proper motion errs
        
    vx_orig: array
        Array with the original X proper motions

    vy_orig: array
        Array with the original Y proper motions

    xe_orig: array
        Array with the original X position errs

    ye_orig: array
        Array with the original Y position errs
        
    x_orig: array
        Array with the original X positions

    y_orig: array
        Array with the original Y positions

    Output:
    ------
    vxe_new: array
       Transformed X proper motion errs

    vye_new: array
        Transformed Y proper motion errs
    """
    idx = 0
    vxe_new_sq = 0.0
    vye_new_sq = 0.0
    # First loop: dvx' / dx
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_sq += (Xcoeff[idx] * (i-j) * (i-j-1) * x_orig**(i-j-2) * y_orig**j * vx_orig)**2. * xe_orig**2. + \
                  (Xcoeff[idx] * (j) * (i-j) * x_orig**(i-j-1) * y_orig**(j-1) * vy_orig)**2. * xe_orig**2.
            vye_new_sq += (Ycoeff[idx] * (i-j) * (i-j-1) * x_orig**(i-j-2) * y_orig**j * vx_orig)**2. * xe_orig**2. + \
                  (Xcoeff[idx] * (j) * (i-j) * x_orig**(i-j-1) * y_orig**(j-1) * vy_orig)**2. * xe_orig**2.

            idx += 1

    # Second loop: dvx' / dy
    idx = 0
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_sq += (Xcoeff[idx] * (i-j) * (j) * x_orig**(i-j-1) * y_orig**(j-1) * vx_orig)**2. * ye_orig**2. + \
                  (Xcoeff[idx] * (j) * (j-1) * x_orig**(i-j-1) * y_orig**(j-2) * vy_orig)**2. * ye_orig**2.
            vye_new_sq += (Ycoeff[idx] * (i-j) * (j) * x_orig**(i-j-1) * y_orig**(j-1) * vx_orig)**2. * ye_orig**2. + \
                  (Xcoeff[idx] * (j) * (j-1) * x_orig**(i-j-1) * y_orig**(j-2) * vy_orig)**2. * ye_orig**2.

            idx += 1

    # Third loop: dvx' / dvx
    idx = 0
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_sq += (Xcoeff[idx] * (i-j) * x_orig**(i-j-1) * y_orig**j)**2. * vxe_orig**2.
            vye_new_sq += (Ycoeff[idx] * (i-j) * x_orig**(i-j-1) * y_orig**j)**2. * vxe_orig**2.

            idx += 1

    # Fourth loop: dvx' / dvy
    idx = 0
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_sq += (Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1))**2. * vye_orig**2.
            vye_new_sq += (Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1))**2. * vye_orig**2.

            idx += 1

    vxe_new = np.sqrt(vxe_new_sq)
    vye_new = np.sqrt(vye_new_sq)

    return vxe_new, vye_new
