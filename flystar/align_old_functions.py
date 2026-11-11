"""
Old functions that are only referenced in examples and template
"""
def transform_from_file(starlist, transFile):
    """
    Apply transformation from transFile to starlist. Returns astropy table with
    added columns with the transformed coordinates. NOTE: Transforms
    positions/position errors, plus velocities and velocity errors if they
    are present in starlist.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS
    
    Parameters
    ----------
    starlist: astropy table
         Starlist we want to apply the transformation too. Must already
         have standard column headers
    transFile: ascii file
        File with the transformation coefficients. Assumed to be output of
        write_transform, with coefficients specified as code documents

    Output
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
        x0e_orig = starlist['x0_err']
        y0e_orig = starlist['y0_err']
        
        vx_orig = starlist['vx']
        vy_orig = starlist['vy']
        vxe_orig = starlist['vx_err']
        vye_orig = starlist['vy_err']
    
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
        starlist_f['x0_err'] = x0e_new
        starlist_f['y0_err'] = y0e_new
        starlist_f['vx'] = vx_new
        starlist_f['vy'] = vy_new
        starlist_f['vx_err'] = vxe_new
        starlist_f['vy_err'] = vye_new

    return starlist_f

def transform_pos_from_file(Xcoeff, Ycoeff, order, x_orig, y_orig):
    """
    Given the read-in coefficients from transform_from_file, apply the
    transformation to the observed positions. This is generalized to
    work with any order polynomial transform.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS

    Parameters
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

    Output
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

    Parameters
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

    Output
    ------
    xe_new: array
       Transformed X position errs
    ye_new: array
        Transformed Y position errs
    """
    idx = 0 # coeff index
    xe_new_tmp1 = 0.0
    ye_new_tmp1 = 0.0
    xe_new_tmp2 = 0.0
    ye_new_tmp2 = 0.0
    
    # First loop: dx'/dx
    for i in range(order+1):
        for j in range(i+1):
            xe_new_tmp1 += Xcoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j
            ye_new_tmp1 += Ycoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j

            idx += 1
            
    # Second loop: dy'/dy
    idx = 0 # coeff index
    for i in range(order+1):
        for j in range(i+1):
            xe_new_tmp2 += Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1)
            ye_new_tmp2 += Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1)

            idx += 1
    # Take square root for xe/ye_new
    xe_new = np.sqrt((xe_new_tmp1 * xe_orig)**2 + (xe_new_tmp2 * ye_orig)**2)
    ye_new = np.sqrt((ye_new_tmp1 * ye_orig)**2 + (ye_new_tmp2 * ye_orig)**2)

    return xe_new, ye_new

def transform_vel_from_file(Xcoeff, Ycoeff, order, vx_orig, vy_orig, x_orig, y_orig):
    """
    Given the read-in coefficients from transform_from_file, apply the
    transformation to the observed proper motions. This is generalized to
    work with any order transform.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS
    
    Parameters
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

    Output
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
    
    Parameters
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

    Output
    ------
    vxe_new: array
       Transformed X proper motion errs
    vye_new: array
        Transformed Y proper motion errs
    """
    idx = 0
    vxe_new_tmp1 = 0.0
    vye_new_tmp1 = 0.0
    vxe_new_tmp2 = 0.0
    vye_new_tmp2 = 0.0
    vxe_new_tmp3 = 0.0
    vye_new_tmp3 = 0.0
    vxe_new_tmp4 = 0.0
    vye_new_tmp4 = 0.0

    
    # First loop: dvx' / dx
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_tmp1 += Xcoeff[idx] * (i-j) * (i-j-1) * x_orig**(i-j-2) * y_orig**j * vx_orig
            vxe_new_tmp1 += Xcoeff[idx] * (j) * (i-j) * x_orig**(i-j-1) * y_orig**(j-1) * vy_orig
            vye_new_tmp1 += Ycoeff[idx] * (i-j) * (i-j-1) * x_orig**(i-j-2) * y_orig**j * vx_orig
            vye_new_tmp1 += Ycoeff[idx] * (j) * (i-j) * x_orig**(i-j-1) * y_orig**(j-1) * vy_orig

            idx += 1

    # Second loop: dvx' / dy
    idx = 0
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_tmp2 += Xcoeff[idx] * (i-j) * (j) * x_orig**(i-j-1) * y_orig**(j-1) * vx_orig
            vxe_new_tmp2 += Xcoeff[idx] * (j) * (j-1) * x_orig**(i-j-1) * y_orig**(j-2) * vy_orig
            vye_new_tmp2 += Ycoeff[idx] * (i-j) * (j) * x_orig**(i-j-1) * y_orig**(j-1) * vx_orig
            vye_new_tmp2 += Ycoeff[idx] * (j) * (j-1) * x_orig**(i-j-1) * y_orig**(j-2) * vy_orig

            idx += 1

    # Third loop: dvx' / dvx
    idx = 0
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_tmp3 += Xcoeff[idx] * (i-j) * x_orig**(i-j-1) * y_orig**j
            vye_new_tmp3 += Ycoeff[idx] * (i-j) * x_orig**(i-j-1) * y_orig**j

            idx += 1

    # Fourth loop: dvx' / dvy
    idx = 0
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_tmp4 += Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1)
            vye_new_tmp4 += Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1)

            idx += 1

    vxe_new = np.sqrt((vxe_new_tmp1 * xe_orig)**2 + (vxe_new_tmp2 * ye_orig)**2 + \
                      (vxe_new_tmp3 * vxe_orig)**2 + (vxe_new_tmp4 * vye_orig)**2)
    vye_new = np.sqrt((vye_new_tmp1 * xe_orig)**2 + (vye_new_tmp2 * ye_orig)**2 + \
                      (vye_new_tmp3 * vxe_orig)**2 + (vye_new_tmp4 * vye_orig)**2)

    return vxe_new, vye_new






"""
Old functions with things hard-coded for OB120169
"""

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
    print(d[tdx]['name_0', 't0', 'mag', 'x0', 'vx', 'x0_err', 'vx_err', 'chi2x', 'y0', 'vy', 'y0_err', 'vy_err', 'chi2y', 'dof'])

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
    xe_ref = d['x0_err']
    ye_ref = d['y0_err']
    
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
            xe_ref_ee = np.hypot(xe_ref_ee, d['vx_err'] * dt)
            ye_ref_ee = np.hypot(ye_ref_ee, d['vy_err'] * dt)

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
        d['x0_err'] = pxe_all[:, -1]
        d['y0_err'] = pye_all[:, -1]
        
    if poly_deg >= 1:
        d['vx'] = px_all[:, -2]
        d['vy'] = py_all[:, -2]
        d['vx_err'] = pxe_all[:, -2]
        d['vy_err'] = pye_all[:, -2]

    if poly_deg >= 2:
        d['ax'] = px_all[:, -3]
        d['ay'] = py_all[:, -3]
        d['axe'] = pxe_all[:, -3]
        d['aye'] = pye_all[:, -3]

    pdb.set_trace()
        
    return


# Needs gcwork's starset and the Java-align align/polyfit/points outputs.
# plots.plot_stars makes the same plots from a StarTable.
def plotStar(starNames, rootDir='./', align='align/align_d_rms_1000_abs_t',
             poly='polyfit_d/fit', points='points_d/', radial=False, NcolMax=3, figsize=(6, 6)):

    print( 'Creating residuals plots for star(s):' )
    print( starNames )

    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    Nstars = len(starNames)
    Ncols = 2 * np.min([Nstars, NcolMax])
    if Nstars <= Ncols/2:
        Nrows = 3
    else:
        Nrows = math.ceil(Nstars / (Ncols / 2)) * 3

    plt.close('all')
    plt.figure(2, figsize=figsize)
    names = s.getArray('name')
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x,y)

    for i in range(Nstars):

        starName = starNames[i]

        ii = names.index(starName)
        star = s.stars[ii]

        pointsTab = Table.read(rootDir + points + starName + '.points', format='ascii')

        time = pointsTab[pointsTab.colnames[0]]
        x = pointsTab[pointsTab.colnames[1]]
        y = pointsTab[pointsTab.colnames[2]]
        xerr = pointsTab[pointsTab.colnames[3]]
        yerr = pointsTab[pointsTab.colnames[4]]

        fitx = star.fitXv
        fity = star.fitYv
        dt = time - fitx.t0
        fitLineX = fitx.p + (fitx.v * dt)
        fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

        fitLineY = fity.p + (fity.v * dt)
        fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

        if (radial == True):
            # Lets also do radial/tangential
            x0 = fitx.p
            y0 = fity.p
            vx = fitx.v
            vy = fity.v
            x0e = fitx.perr
            y0e = fity.perr
            vxe = fitx.verr
            vye = fity.verr

            r0 = np.sqrt(x0**2 + y0**2)

            vr = ((vx*x0) + (vy*y0)) / r0
            vt = ((vx*y0) - (vy*x0)) / r0
            vre =  (vxe*x0/r0)**2 + (vye*y0/r0)**2
            vre += (y0*x0e*vt/r0**2)**2 + (x0*y0e*vt/r0**2)**2
            vre =  np.sqrt(vre)
            vte =  (vxe*y0/r0)**2 + (vye*x0/r0)**2
            vte += (y0*x0e*vr/r0**2)**2 + (x0*y0e*vr/r0**2)**2
            vte =  np.sqrt(vte)

            r = ((x*x0) + (y*y0)) / r0
            t = ((x*y0) - (y*x0)) / r0
            rerr = (xerr*x0/r0)**2 + (yerr*y0/r0)**2
            rerr += (y0*x0e*t/r0**2)**2 + (x0*y0e*t/r0**2)**2
            rerr =  np.sqrt(rerr)
            terr =  (xerr*y0/r0)**2 + (yerr*x0/r0)**2
            terr += (y0*x0e*r/r0**2)**2 + (x0*y0e*r/r0**2)**2
            terr =  np.sqrt(terr)

            fitLineR = ((fitLineX*x0) + (fitLineY*y0)) / r0
            fitLineT = ((fitLineX*y0) - (fitLineY*x0)) / r0
            fitSigR = ((fitSigX*x0) + (fitSigY*y0)) / r0
            fitSigT = ((fitSigX*y0) - (fitSigY*x0)) / r0

            diffR = r - fitLineR
            diffT = t - fitLineT
            sigR = diffR / rerr
            sigT = diffT / terr

            idxR = np.where(abs(sigR) > 4)
            idxT = np.where(abs(sigT) > 4)


        diffX = x - fitLineX
        diffY = y - fitLineY
        diff = np.hypot(diffX, diffY)
        rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
        sigX = diffX / xerr
        sigY = diffY / yerr
        sig = diff / rerr


        # Determine if there are points that are more than 5 sigma off
        idxX = np.where(abs(sigX) > 4)
        idxY = np.where(abs(sigY) > 4)
        idx = np.where(abs(sig) > 4)

        print( 'Star:        ', starName )
        print( '\tX Chi^2 = %5.2f (%6.2f for %2d dof)' %
              (fitx.chi2red, fitx.chi2, fitx.dof))
        print( '\tY Chi^2 = %5.2f (%6.2f for %2d dof)' %
              (fity.chi2red, fity.chi2, fity.dof))
        # print( 'X  Outliers: ', time[idxX] )
        # print( 'Y  Outliers: ', time[idxY] )
        # if (radial):
        #     print( 'R  Outliers: ', time[idxX] )
        #     print( 'T  Outliers: ', time[idxY] )
        # print( 'XY Outliers: ', time[idx] )

        # close(2)
        #             figure(2, figsize=(7, 8))
        #             clf()

        t0 = int(np.floor(np.min(time)))
        tO = int(np.ceil(np.max(time)))

        dateTicLoc = plt.MultipleLocator(3)
        dateTicRng = [t0-1, tO+1]
        dateTics = np.arange(t0, tO+1)
        DateTicsLabel = dateTics-2000

        # See if we are using MJD instead.
        if time[0] > 50000:
            print('MJD')
            dateTicLoc = plt.MultipleLocator(1000)
            t0 = int(np.round(np.min(time), 50))
            tO = int(np.round(np.max(time), 50))
            dateTicRng = [t0-200, tO+200]
            dateTics = np.arange(dateTicRng[0], dateTicRng[-1]+500, 1000)
            DateTicsLabel = dateTics


        maxErr = np.array([xerr, yerr]).max()
        resTicRng = [-1.1*maxErr, 1.1*maxErr]

        fmtX = FormatStrFormatter('%5i')
        fmtY = FormatStrFormatter('%6.2f')
        fontsize1 = 10

        if i < (Ncols/2):
            col = (2*i)+1
            row = 1
        else:
            col = 1 + 2*(i % (Ncols/2))
            row = 1 + 3*(i//(Ncols/2))

        ind = (row-1)*Ncols + col

        paxes = plt.subplot(Nrows, Ncols, ind)
        plt.plot(time, fitLineX, 'b-')
        plt.plot(time, fitLineX + fitSigX, 'b--')
        plt.plot(time, fitLineX - fitSigX, 'b--')
        plt.errorbar(time, x, yerr=xerr, fmt='k.')
        rng = plt.axis()
        plt.ylim(np.min(x-xerr-0.1),np.max(x+xerr+0.1))
        plt.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            plt.xlabel('Date (MJD)', fontsize=fontsize1)
        plt.ylabel('X (pix)', fontsize=fontsize1)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        plt.yticks(np.arange(np.min(x-xerr-0.1), np.max(x+xerr+0.1), 0.2))
        plt.xticks(dateTics, DateTicsLabel)
        plt.xlim(np.min(dateTics), np.max(dateTics))
        plt.annotate(starName,xy=(1.0,1.1), xycoords='axes fraction', fontsize=12, color='red')


        col = col + 1
        ind = (row-1)*Ncols + col

        paxes = plt.subplot(Nrows, Ncols, ind)
        plt.plot(time, fitLineY, 'b-')
        plt.plot(time, fitLineY + fitSigY, 'b--')
        plt.plot(time, fitLineY - fitSigY, 'b--')
        plt.errorbar(time, y, yerr=yerr, fmt='k.')
        rng = plt.axis()
        plt.axis(dateTicRng + [rng[2], rng[3]], fontsize=fontsize1)
        plt.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            plt.xlabel('Date (MJD)', fontsize=fontsize1)
        plt.ylabel('Y (pix)', fontsize=fontsize1)
        #paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        paxes.tick_params(axis='both', which='major', labelsize=12)
        plt.ylim(np.min(y-yerr-0.1),np.max(y+yerr+0.1))
        plt.yticks(np.arange(np.min(y-yerr-0.1), np.max(y+yerr+0.1), 0.2))
        plt.xticks(dateTics, DateTicsLabel)
        plt.xlim(np.min(dateTics), np.max(dateTics))

        row = row + 1
        col = col - 1
        ind = (row-1)*Ncols + col

        paxes = plt.subplot(Nrows, Ncols, ind)
        plt.plot(time, np.zeros(len(time)), 'b-')
        plt.plot(time, fitSigX, 'b--')
        plt.plot(time, -fitSigX, 'b--')
        plt.errorbar(time, x - fitLineX, yerr=xerr, fmt='k.')
        plt.axis(dateTicRng + resTicRng, fontsize=fontsize1)
        plt.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            plt.xlabel('Date (MJD)', fontsize=fontsize1)
        plt.ylabel('X Residuals (pix)', fontsize=fontsize1)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        plt.xticks(dateTics, DateTicsLabel)
        plt.xlim(np.min(dateTics), np.max(dateTics))

        col = col + 1
        ind = (row-1)*Ncols + col

        paxes = plt.subplot(Nrows, Ncols, ind)
        plt.plot(time, np.zeros(len(time)), 'b-')
        plt.plot(time, fitSigY, 'b--')
        plt.plot(time, -fitSigY, 'b--')
        plt.errorbar(time, y - fitLineY, yerr=yerr, fmt='k.')
        plt.axis(dateTicRng + resTicRng, fontsize=fontsize1)
        plt.xlabel('Date -2000 (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            plt.xlabel('Date (MJD)', fontsize=fontsize1)
        plt.ylabel('Y Residuals (pix)', fontsize=fontsize1)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        plt.xticks(dateTics, DateTicsLabel)
        plt.xlim(np.min(dateTics), np.max(dateTics))

        row = row + 1
        col = col - 1
        ind = (row-1)*Ncols + col


        paxes = plt.subplot(Nrows, Ncols, ind)
        plt.errorbar(x,y, xerr=xerr, yerr=yerr, fmt='k.')
        plt.yticks(np.arange(np.min(y-yerr-0.1), np.max(y+yerr+0.1), 0.2))
        plt.xticks(np.arange(np.min(x-xerr-0.1), np.max(x+xerr+0.1), 0.2), rotation = 270)
        plt.axis('equal')
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        paxes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        paxes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlabel('X (pix)', fontsize=fontsize1)
        plt.ylabel('Y (pix)', fontsize=fontsize1)
        plt.plot(fitLineX, fitLineY, 'b-')

        col = col + 1
        ind = (row-1)*Ncols + col

        bins = np.arange(-7.5, 7.5, 1)
        paxes = plt.subplot(Nrows, Ncols, ind)
        id = np.where(diffY < 0)[0]
        sig[id] = -1.*sig[id]
        (n, b, p) = plt.hist(sigX, bins, histtype='stepfilled', color='b', label='X')
        plt.setp(p, 'facecolor', 'b')
        (n, b, p) = plt.hist(sigY, bins, histtype='step', color='r', label='Y')
        plt.axis([-7, 7, 0, 8], fontsize=10)
        plt.legend()
        plt.xlabel('Residuals (sigma)', fontsize=fontsize1)
        plt.ylabel('Number of Epochs', fontsize=fontsize1)

        ##########
        #
        # Also plot radial/tangential
        #
        ##########
        if (radial == True):
            plt.clf()

            dateTicLoc = plt.MultipleLocator(3)

            maxErr = np.array([rerr, terr]).max()
            resTicRng = [-3*maxErr, 3*maxErr]

            fmtX = FormatStrFormatter('%5i')
            fmtY = FormatStrFormatter('%6.2f')

            paxes = plt.subplot(3,2,1)
            plt.plot(time, fitLineR, 'b-')
            plt.plot(time, fitLineR + fitSigR, 'b--')
            plt.plot(time, fitLineR - fitSigR, 'b--')
            plt.errorbar(time, r, yerr=rerr, fmt='k.')
            rng = plt.axis()
            plt.axis(dateTicRng + [rng[2], rng[3]])
            plt.xlabel('Date (yrs)')
            plt.ylabel('R (pix)')
            paxes.xaxis.set_major_formatter(fmtX)
            paxes.get_xaxis().set_major_locator(dateTicLoc)
            paxes.yaxis.set_major_formatter(fmtY)

            paxes = plt.subplot(3, 2, 2)
            plt.plot(time, fitLineT, 'b-')
            plt.plot(time, fitLineT + fitSigT, 'b--')
            plt.plot(time, fitLineT - fitSigT, 'b--')
            plt.errorbar(time, t, yerr=terr, fmt='k.')
            rng = plt.axis()
            plt.axis(dateTicRng + [rng[2], rng[3]])
            plt.xlabel('Date (yrs)')
            plt.ylabel('T (pix)')
            paxes.xaxis.set_major_formatter(fmtX)
            paxes.get_xaxis().set_major_locator(dateTicLoc)
            paxes.yaxis.set_major_formatter(fmtY)

            paxes = plt.subplot(3, 2, 3)
            plt.plot(time, np.zeros(len(time)), 'b-')
            plt.plot(time, fitSigR, 'b--')
            plt.plot(time, -fitSigR, 'b--')
            plt.errorbar(time, r - fitLineR, yerr=rerr, fmt='k.')
            plt.axis(dateTicRng + resTicRng)
            plt.xlabel('Date (yrs)')
            plt.ylabel('R Residuals (pix)')
            paxes.get_xaxis().set_major_locator(dateTicLoc)

            paxes = plt.subplot(3, 2, 4)
            plt.plot(time, np.zeros(len(time)), 'b-')
            plt.plot(time, fitSigT, 'b--')
            plt.plot(time, -fitSigT, 'b--')
            plt.errorbar(time, t - fitLineT, yerr=terr, fmt='k.')
            plt.axis(dateTicRng + resTicRng)
            plt.xlabel('Date (yrs)')
            plt.ylabel('T Residuals (pix)')
            paxes.get_xaxis().set_major_locator(dateTicLoc)

            bins = np.arange(-7, 7, 1)
            plt.subplot(3, 2, 5)
            (n, b, p) = plt.hist(sigR, bins)
            plt.setp(p, 'facecolor', 'k')
            plt.axis([-5, 5, 0, 20])
            plt.xlabel('T Residuals (sigma)')
            plt.ylabel('Number of Epochs')

            plt.subplot(3, 2, 6)
            (n, b, p) = plt.hist(sigT, bins)
            plt.axis([-5, 5, 0, 20])
            plt.setp(p, 'facecolor', 'k')
            plt.xlabel('Y Residuals (sigma)')
            plt.ylabel('Number of Epochs')

            plt.subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
            plt.savefig(rootDir+'plots/plotStarRadial_' + starName + '.png', dpi=300)
            plt.show()

    title = rootDir.split('/')[-2]
    plt.suptitle(title, x=0.5, y=0.97)

    if Nstars == 1:
        plt.subplots_adjust(wspace=0.4, hspace=0.4, left = 0.15, bottom = 0.1, right=0.9, top=0.9)
        plt.savefig(rootDir+'plots/plotStar_' + starName + '.png', dpi=300)
    else:
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left = 0.08, bottom = 0.05, right=0.95, top=0.90)
        plt.savefig(rootDir+'plots/plotStar_all.png', dpi=300)
        plt.show()

    plt.show()
    print('Fubar')
