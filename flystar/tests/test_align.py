from flystar import align
from flystar import starlists
from flystar import startables
from flystar import transforms
from astropy.table import Table
import numpy as np
import pylab as plt
import pdb
import datetime


def test_MosaicSelfRef():
    """
    Cross-match and align 4 starlists using the OO version of mosaic lists.
    """
    list_files = ['A.lis', 'B.lis', 'C.lis', 'D.lis']
    lists = [starlists.StarList.from_lis_file(lf) for lf in list_files]

    ##########
    # Test instantiation and basic fitting.
    ##########
    msc = align.MosaicSelfRef(lists, ref_index=0, iters=2,
                              dr_tol=[3, 3], dm_tol=[1, 1],
                              trans_class=transforms.PolyTransform,
                              verbose=False,
                              trans_args={'order': 2})

    msc.fit()
    
    # Check some of the output quantities on the final table.
    assert 'x0' in msc.ref_table.colnames
    assert 'x0e' in msc.ref_table.colnames
    assert 'y0' in msc.ref_table.colnames
    assert 'y0e' in msc.ref_table.colnames
    assert 'm0' in msc.ref_table.colnames
    assert 'm0e' in msc.ref_table.colnames
    assert 'use_in_trans' in msc.ref_table.colnames
    assert 'used_in_trans' in msc.ref_table.colnames
    assert 'ref_orig' in msc.ref_table.colnames

    assert msc.ref_table['use_in_trans'].shape == msc.ref_table['x0'].shape
    assert msc.ref_table['used_in_trans'].shape == msc.ref_table['x'].shape
    

    # Check that we have some matched stars... should be at least 35 stars
    # that are detected in all 4 starlists.
    idx = np.where(msc.ref_table['n_detect'] == 4)[0]
    assert len(idx) > 35 

    # Check that the transformation error isn't too big
    assert (msc.ref_table['x0e'] < 3.0).all() # less than 1 pix
    assert (msc.ref_table['y0e'] < 3.0).all()
    #assert (msc.ref_table['m0e'] < 1.0).all() # less than 0.5 mag
    assert (msc.ref_table['m0e'] < 1.5).all() # less than 0.5 mag
    
    # Check that the transformation lists aren't too wacky
    for ii in range(4):
        np.testing.assert_almost_equal(msc.trans_list[ii].px.c1_0, 1.0, 2)
        np.testing.assert_almost_equal(msc.trans_list[ii].py.c0_1, 1.0, 2)
    
    # We didn't do any velocity fitting, so make sure nothing got created.
    assert 'vx' not in msc.ref_table.colnames
    assert 'vy' not in msc.ref_table.colnames
    assert 'vxe' not in msc.ref_table.colnames
    assert 'vye' not in msc.ref_table.colnames

    plt.clf()
    plt.plot(msc.ref_table['x'][:, 0],
             msc.ref_table['y'][:, 0],
             'k+', color='red', mec='red', mfc='none')
    plt.plot(msc.ref_table['x'][:, 1],
             msc.ref_table['y'][:, 1],
             'kx', color='blue', mec='blue', mfc='none')
    plt.plot(msc.ref_table['x'][:, 2],
             msc.ref_table['y'][:, 2],
             'ko', color='cyan', mec='cyan', mfc='none')
    plt.plot(msc.ref_table['x'][:, 3],
             msc.ref_table['y'][:, 3],
             'k^', color='green', mec='green', mfc='none')
    plt.plot(msc.ref_table['x0'],
             msc.ref_table['y0'],
             'k.', color='black', alpha=0.2)

    return

def test_MosaicSelfRef_vel_tconst():
    """
    Cross-match and align 4 starlists using the OO version of mosaic lists.
    The 4 lists are all taken at the same time (so 0 velocities should result).
    
    """
    list_files = ['A.lis', 'B.lis', 'C.lis', 'D.lis']
    lists = [starlists.StarList.from_lis_file(lf) for lf in list_files]

    ##########
    # Test instantiation and basic fitting.
    # Note these star lists are ALL at the same date.
    ##########
    msc = align.MosaicSelfRef(lists, ref_index=0, iters=2,
                              dr_tol=[3, 3], dm_tol=[1, 1],
                              trans_class=transforms.PolyTransform,
                              trans_args={'order': 2}, use_vel=True,
                              verbose=False)

    msc.fit()
    
    # Check some of the output quantities on the final table.
    assert 'x0' in msc.ref_table.colnames
    assert 'x0e' in msc.ref_table.colnames
    assert 'y0' in msc.ref_table.colnames
    assert 'y0e' in msc.ref_table.colnames
    assert 'm0' in msc.ref_table.colnames
    assert 'm0e' in msc.ref_table.colnames
    assert 'vx' in msc.ref_table.colnames
    assert 'vxe' in msc.ref_table.colnames
    assert 'vy' in msc.ref_table.colnames
    assert 'vye' in msc.ref_table.colnames
    assert 't0' in msc.ref_table.colnames

    # Check that we have some matched stars... should be at least 35 stars
    # that are detected in all 4 starlists.
    idx = np.where(msc.ref_table['n_detect'] == 4)[0]
    assert len(idx) > 35 

    # Check that the transformation error isn't too big
    assert (msc.ref_table['x0e'] < 3.0).all() # less than 1 pix
    assert (msc.ref_table['y0e'] < 3.0).all()
    assert (msc.ref_table['m0e'] < 1.0).all() # less than 0.5 mag
    
    # Check that the transformation lists aren't too wacky
    for ii in range(4):
        np.testing.assert_almost_equal(msc.trans_list[ii].px.c1_0, 1.0, 2)
        np.testing.assert_almost_equal(msc.trans_list[ii].py.c0_1, 1.0, 2)

    # Check that the velocities aren't crazy...
    # they should be zero (since there is no time difference)
    np.testing.assert_almost_equal(msc.ref_table['vx'], 0, 1)
    np.testing.assert_almost_equal(msc.ref_table['vy'], 0, 1)

    assert (msc.ref_table['vx'] == 0).all()
    assert (msc.ref_table['vy'] == 0).all()
    assert (msc.ref_table['vxe'] == 0).all()
    assert (msc.ref_table['vye'] == 0).all()

    return

def test_MosaicSelfRef_vel():
    """
    Cross-match and align 4 starlists using the OO version of mosaic lists.
    
    """
    list_files = ['A.lis', 'B.lis', 'C.lis', 'D.lis']
    lists = [starlists.StarList.from_lis_file(lf) for lf in list_files]

    # Modify the times so that we get velocities out.
    lists[0].meta['list_time'] = 2001.4
    lists[0]['t'] = 2001.4
    
    lists[1].meta['list_time'] = 2002.4
    lists[1]['t'] = 2002.4
    
    lists[2].meta['list_time'] = 2003.4
    lists[2]['t'] = 2003.4
    
    lists[3].meta['list_time'] = 2004.4
    lists[3]['t'] = 2004.4


    ##########
    # Test instantiation and basic fitting.
    ##########
    msc = align.MosaicSelfRef(lists, ref_index=0, iters=3,
                              dr_tol=[5, 3, 3], dm_tol=[1, 1, 0.5], outlier_tol=None,
                              trans_class=transforms.PolyTransform,
                              trans_args={'order': 2}, use_vel=True,
                              verbose=False)

    msc.fit()
    
    # Check some of the output quantities on the final table.
    assert 'x0' in msc.ref_table.colnames
    assert 'x0e' in msc.ref_table.colnames
    assert 'y0' in msc.ref_table.colnames
    assert 'y0e' in msc.ref_table.colnames
    assert 'm0' in msc.ref_table.colnames
    assert 'm0e' in msc.ref_table.colnames
    assert 'vx' in msc.ref_table.colnames
    assert 'vxe' in msc.ref_table.colnames
    assert 'vy' in msc.ref_table.colnames
    assert 'vye' in msc.ref_table.colnames
    assert 't0' in msc.ref_table.colnames

    # Check that we have some matched stars... should be at least 35 stars
    # that are detected in all 4 starlists.
    idx = np.where(msc.ref_table['n_detect'] == 4)[0]
    assert len(idx) > 35 

    # Check that the transformation error isn't too big
    assert (msc.ref_table['x0e'] < 3.0).all() # less than 1 pix
    assert (msc.ref_table['y0e'] < 3.0).all()
    assert (msc.ref_table['m0e'] < 1.0).all() # less than 0.5 mag
    
    # Check that the transformation lists aren't too wacky
    for ii in range(4):
        np.testing.assert_almost_equal(msc.trans_list[ii].px.c1_0, 1.0, 2)
        np.testing.assert_almost_equal(msc.trans_list[ii].py.c0_1, 1.0, 2)

    
    plt.clf()
    plt.plot(msc.ref_table['vx'],
             msc.ref_table['vy'],
             'k.', color='black', alpha=0.2)

    return

def test_MosaicToRef():
    make_fake_starlists_poly1_vel(seed=42)
    
    ref_file = 'random_vel_ref.fits'
    list_files = ['random_vel_0.fits',
                  'random_vel_1.fits',
                  'random_vel_2.fits',
                  'random_vel_3.fits']

    ref_list = Table.read(ref_file)

    # Convert velocities to arcsec/yr
    ref_list['vx'] *= 1e-3
    ref_list['vy'] *= 1e-3
    ref_list['vxe'] *= 1e-3
    ref_list['vye'] *= 1e-3

    # Switch our list to a "increasing to the West" list.
    ref_list['x0'] *= -1.0
    ref_list['vx'] *= -1.0
        
    lists = [starlists.StarList.read(lf) for lf in list_files]

    msc = align.MosaicToRef(ref_list, lists, iters=2,
                              dr_tol=[0.2, 0.1], dm_tol=[1, 0.5],
                              trans_class=transforms.PolyTransform,
                              trans_args={'order': 2}, use_vel=True,
                              update_ref_orig=False, verbose=False)

    msc.fit()

    # Check our status columns
    assert 'use_in_trans' in msc.ref_table.colnames
    assert 'used_in_trans' in msc.ref_table.colnames
    assert 'ref_orig' in msc.ref_table.colnames
    assert msc.ref_table['use_in_trans'].shape == msc.ref_table['x0'].shape
    assert msc.ref_table['used_in_trans'].shape == msc.ref_table['x'].shape

    # The velocities should be almost the same as the input 
    # velocities since update_ref_orig == False.
    np.testing.assert_almost_equal(msc.ref_table['vx'], ref_list['vx'], 5)
    np.testing.assert_almost_equal(msc.ref_table['vy'], ref_list['vy'], 5)


    ##########
    # Align and let velocities be free. 
    ##########
    msc.update_ref_orig = True
    msc.fit()

    # The velocities should be almost the same (but not as close as before)
    # as the input velocities since update_ref == False.
    np.testing.assert_almost_equal(msc.ref_table['vx'], ref_list['vx'], 1)
    np.testing.assert_almost_equal(msc.ref_table['vy'], ref_list['vy'], 1)

    # Also double check that they aren't exactly the same for the reference stars.
    assert np.any(np.not_equal(msc.ref_table['vx'], ref_list['vx']))
    
    return msc
    

def make_fake_starlists_shifts():
    N_stars = 200
    x = np.random.rand(N_stars) * 1000
    y = np.random.rand(N_stars) * 1000
    m = (np.random.rand(N_stars) * 8) + 9
        
    sdx = np.argsort(m)
    x = x[sdx]
    y = y[sdx]
    m = m[sdx]
    
    name = ['star_{0:03d}'.format(ii) for ii in range(N_stars)]

    # Save original positions as reference (1st) list.
    fmt = '{0:10s}  {1:5.2f} 2015.0 {2:9.4f}  {3:9.4f} 0 0 0 0\n'
    _out = open('random_0.lis', 'w')
    for ii in range(N_stars):
        _out.write(fmt.format(name[ii], m[ii], x[ii], y[ii]))
    _out.close()


    ##########
    # Shifts
    ##########
    # Make 4 new starlists with different shifts.
    shifts = [[  6.5,  10.1],
              [100.3,  50.5],
              [-30.0,-100.7],
              [250.0,-250.0]]

    for ss in range(len(shifts)):
        xnew = x - shifts[ss][0]
        ynew = y - shifts[ss][1]

        # Perturb with small errors (0.1 pix)
        xnew += np.random.randn(N_stars) * 0.1
        ynew += np.random.randn(N_stars) * 0.1

        mnew = m + np.random.randn(N_stars) * 0.05

        _out = open('random_shift_{0:d}.lis'.format(ss+1), 'w')
        for ii in range(N_stars):
            _out.write(fmt.format(name[ii], mnew[ii], xnew[ii], ynew[ii]))
        _out.close()

    return shifts

def make_fake_starlists_poly1(seed=-1):
    # If seed >=0, then set random seed to that value
    if seed >= 0:
        np.random.seed(seed=seed)
        
    N_stars = 200
    x = np.random.rand(N_stars) * 1000
    y = np.random.rand(N_stars) * 1000
    m = (np.random.rand(N_stars) * 8) + 9
        
    sdx = np.argsort(m)
    x = x[sdx]
    y = y[sdx]
    m = m[sdx]
    
    name = ['star_{0:03d}'.format(ii) for ii in range(N_stars)]

    # Save original positions as reference (1st) list.
    fmt = '{0:10s}  {1:5.2f} 2015.0 {2:9.4f}  {3:9.4f} 0 0 0 0\n'
    _out = open('random_0.lis', 'w')
    for ii in range(N_stars):
        _out.write(fmt.format(name[ii], m[ii], x[ii], y[ii]))
    _out.close()


    ##########
    # Shifts
    ##########
    # Make 4 new starlists with different shifts.
    transforms = [[[  6.5, 0.99, 1e-5], [  10.1, 1e-5, 0.99]],
                  [[100.3, 0.98, 1e-5], [  50.5, 9e-6, 1.001]],
                  [[-30.0, 1.00, 1e-5], [-100.7, 2e-5, 0.999]],
                  [[250.0, 0.97, 2e-5], [-250.0, 1e-5, 1.001]]]

    for ss in range(len(shifts)):
        #transforms.PolyTransform2D(1, transforms[ss])
        xnew = x - shifts[ss][0]
        ynew = y - shifts[ss][1]

        # Perturb with small errors (0.1 pix)
        xnew += np.random.randn(N_stars) * 0.1
        ynew += np.random.randn(N_stars) * 0.1

        mnew = m + np.random.randn(N_stars) * 0.05

        _out = open('random_shift_{0:d}.lis'.format(ss+1), 'w')
        for ii in range(N_stars):
            _out.write(fmt.format(name[ii], mnew[ii], xnew[ii], ynew[ii]))
        _out.close()

    return shifts


def make_fake_starlists_poly1_vel(seed=-1):
    # If seed >=0, then set random seed to that value
    if seed >= 0:
        np.random.seed(seed=seed)
        
    N_stars = 200

    x0  = np.random.rand(N_stars) * 10.0     # arcsec (increasing to East)
    y0  = np.random.rand(N_stars) * 10.0     # arcsec
    x0e = np.random.randn(N_stars) * 5.0e-4  # arcsec
    y0e = np.random.randn(N_stars) * 5.0e-4  # arcsec
    vx  = np.random.randn(N_stars) * 5.0     # mas / yr
    vy  = np.random.randn(N_stars) * 5.0     # mas / yr
    vxe = np.random.randn(N_stars) * 0.1    # mas / yr
    vye = np.random.randn(N_stars) * 0.1    # mas / yr
    m0  = (np.random.rand(N_stars) * 8) + 9  # mag
    m0e = np.random.randn(N_stars) * 0.05    # mag
    t0 = np.ones(N_stars) * 2019.5

    # Make all the errors positive
    x0e = np.abs(x0e)
    y0e = np.abs(y0e)
    m0e = np.abs(m0e)
    vxe = np.abs(vxe)
    vye = np.abs(vye)
    
    name = ['star_{0:03d}'.format(ii) for ii in range(N_stars)]

    # Make an StarList
    lis = starlists.StarList([name, m0, m0e, x0, x0e, y0, y0e, vx, vxe, vy, vye, t0],
                             names = ('name', 'm0', 'm0e', 'x0', 'x0e', 'y0', 'y0e',
                                          'vx', 'vxe', 'vy', 'vye', 't0'))
    
    sdx = np.argsort(m0)
    lis = lis[sdx]

    # Save original positions as reference (1st) list
    # in a StarList format (with velocities).
    lis.write('random_vel_ref.fits', overwrite=True)
    
    ##########
    # Propogate to new times and distort.
    ##########
    # Make 4 new starlists with different epochs and transformations.
    times = [2018.5, 2019.5, 2020.5, 2021.5]
    xy_trans = [[[ 6.5, 0.99, 1e-5], [  10.1, 1e-5, 0.99]],
               [[100.3, 0.98, 1e-5], [  50.5, 9e-6, 1.001]],
               [[  0.0, 1.00,  0.0], [   0.0,  0.0, 1.0]],
               [[250.0, 0.97, 2e-5], [-250.0, 1e-5, 1.001]]]
    mag_trans = [0.1, 0.4, 0.0, -0.3]

    # Convert into pixels (undistorted) with the following info.
    scale = 0.01  # arcsec / pix
    shift = [1.0, 1.0]  # pix
    
    for ss in range(len(times)):
        dt = times[ss] - lis['t0']
        
        x = lis['x0'] + (lis['vx']/1e3) * dt
        y = lis['y0'] + (lis['vy']/1e3) * dt
        t = np.ones(N_stars) * times[ss]

        # Convert into pixels
        xp = (x / -scale) + shift[0]  # -1 from switching to increasing to West (right)
        yp = (y /  scale) + shift[1]
        xpe = lis['x0e'] / scale
        ype = lis['y0e'] / scale

        # Distort the positions
        trans = transforms.PolyTransform(1, xy_trans[ss][0], xy_trans[ss][1], mag_offset=mag_trans[ss])
        xd, yd = trans.evaluate(xp, yp)
        md = trans.evaluate_mag(lis['m0'])

        # Perturb with small errors (0.1 pix)
        xd += np.random.randn(N_stars) * 0.1
        yd += np.random.randn(N_stars) * 0.1
        md += np.random.randn(N_stars) * 0.02
        xde = xpe
        yde = ype
        mde = lis['m0e']

        # Save the new list as a starlist.
        new_lis = starlists.StarList([lis['name'], md, mde, xd, xde, yd, yde, t],
                                     names=('name', 'm', 'me', 'x', 'xe', 'y', 'ye', 't'))

        new_lis.write('random_vel_{0:d}.fits'.format(ss), overwrite=True)

    return (xy_trans, mag_trans)
                     
def test_MosaicToRef_hst_me():
    """
    Test Casey's issue with 'me' not getting propogated 
    from the input starlists to the output table.

    Use data from MB10-364 microlensing target for the test. 
    """
    # Target RA and Dec (MOA data download)
    ra = '17:57:05.401'
    dec = '-34:27:05.01'
    
    # Load up a Gaia catalog (queried around the RA/Dec above)
    my_gaia = Table.read('mb10364_data/my_gaia.fits')
    my_gaia['me'] = 0.01
    
    # Gather the list of starlists. For first pass, don't modify the starlists.
    # Loop through the observations and read them in, in prep for alignment with Gaia
    epochs = [2011.83, 2012.73, 2013.81]
    starlist_names = ['mb10364_data/2011_10_31_F606W_MATCHUP_XYMEEE_final.calib',
                      'mb10364_data/2012_09_25_F606W_MATCHUP_XYMEEE_final.calib',
                      'mb10364_data/2013_10_24_F606W_MATCHUP_XYMEEE_final.calib']
        
    list_of_starlists = []
    
    # Just using the F606W filters first.
    for ee in range(len(starlist_names)):
        lis = starlists.StarList.from_lis_file(starlist_names[ee])
        
        # # Add additive error term. MAYBE YOU DON'T NEED THIS
        # lis['xe'] = np.hypot(lis['xe'], 0.01)  # Adding 0.01 pix (0.1 mas) in quadrature.
        # lis['ye'] = np.hypot(lis['ye'], 0.01)
        
        lis['t'] = epochs[ee]

        # Lets dump the faint stars.
        idx = np.where(lis['m'] < 20.0)[0]
        lis = lis[idx]

        list_of_starlists.append(lis)
        
    msc = align.MosaicToRef(my_gaia, list_of_starlists, iters=1,
                        dr_tol=[0.1], dm_tol=[5],
                        outlier_tol=[None], mag_lim=[13, 21],
                        trans_class=transforms.PolyTransform,
                        trans_args=[{'order': 1}],
                        use_vel=False,
                        use_ref_new=False,
                        update_ref_orig=False,
                        mag_trans=False,
                        weights='both,std',
                        init_guess_mode='miracle', verbose=False)
    msc.fit()
    tab = msc.ref_table

    assert 'me' in tab.colnames

    return

def test_bootstrap():
    """
    Test to make sure calc_bootstrap_error() call is working 
    properly (e.g., only called when user calls calc_bootstrap_error,
    n_boot param for calc_bootstrap_error only, boot_epochs_min working,
    etc.)
    """
    # Read in starlists for MosaicToRef
    ref = Table.read('ref_vel.lis', format='ascii')
    list1 = Table.read('E.lis', format='ascii')
    list2 = Table.read('F.lis', format='ascii')

    list1 = starlists.StarList.from_table(list1)
    list2 = starlists.StarList.from_table(list2)
        
    # Set parameters for alignment
    transModel = transforms.PolyTransform
    trans_args = {'order':2}
    N_loop = 1
    dr_tol = 0.08
    dm_tol = 99
    outlier_tol = None
    mag_lim = None
    ref_mag_lim = None
    weights = 'both,var'
    mag_trans = False

    n_boot = 15
    boot_epochs_min=-1

    # Run FLYSTAR, no bootstraps yet!
    match1 = align.MosaicToRef(ref, [list1, list2], iters=N_loop, dr_tol=dr_tol,
                                  dm_tol=dm_tol, outlier_tol=outlier_tol,
                                  trans_class=transModel,
                                  trans_args=trans_args,
                                  mag_trans=mag_trans,
                                  mag_lim=mag_lim,
                                  ref_mag_lim=ref_mag_lim,
                                  weights=weights,
                                  use_vel=True,
                                  use_ref_new=False,
                                  update_ref_orig=False,
                                  init_guess_mode='name',
                                  verbose=False)
    match1.fit()

    # Make sure no bootstrap columns exist
    assert 'xe_boot' not in match1.ref_table.keys()
    assert 'ye_boot' not in match1.ref_table.keys()
    assert 'vxe_boot' not in match1.ref_table.keys()
    assert 'vye_boot' not in match1.ref_table.keys()

    # Run bootstrap: no boot_epochs_min
    match1.calc_bootstrap_errors(n_boot=n_boot, boot_epochs_min=boot_epochs_min)

    # Make sure columns exist, and none of them are nan values
    assert np.sum(np.isnan(match1.ref_table['xe_boot'])) == 0
    assert np.sum(np.isnan(match1.ref_table['ye_boot'])) == 0
    assert np.sum(np.isnan(match1.ref_table['vxe_boot'])) == 0
    assert np.sum(np.isnan(match1.ref_table['vye_boot'])) == 0

    # Test 2: make sure boot_epochs_min is working
    # Eliminate some rows to list2, so some stars are only in 1 epoch.
    # Rerun align. Some stars should only be detected in 1 epoch
    list3 = list2[0:60]

    match2 = align.MosaicToRef(ref, [list1, list3], iters=N_loop, dr_tol=dr_tol,
                                  dm_tol=dm_tol, outlier_tol=outlier_tol,
                                  trans_class=transModel,
                                  trans_args=trans_args,
                                  mag_trans=mag_trans,
                                  mag_lim=mag_lim,
                                  ref_mag_lim=ref_mag_lim,
                                  weights=weights,
                                  use_vel=True,
                                  use_ref_new=False,
                                  update_ref_orig=False,
                                  init_guess_mode='name',
                                  verbose=False)
    match2.fit()

    # Now run_calc_bootstrap_error, with boot_epochs_min engaged
    boot_epochs_min2 = 2
    match2.calc_bootstrap_errors(n_boot=n_boot, boot_epochs_min=boot_epochs_min2)

    # Make sure boot_epochs_min cut worked as intended
    out = match2.ref_table
    bad = np.where( (out['n_detect'] == 1) & (out['use_in_trans'] == False) )
    good = np.where(out['n_detect'] == 2)

    # Some stars must exist in both "good" and "bad" criteria,
    # otherwise this test isn't as useful as intended.
    assert len(bad[0]) > 0
    assert len(good[0]) > 0

    # For "good" stars: all bootstrap vals should be present
    assert np.sum(np.isnan(out['xe_boot'][good])) == 0
    assert np.sum(np.isnan(out['ye_boot'][good])) == 0
    assert np.sum(np.isnan(out['vxe_boot'][good])) == 0
    assert np.sum(np.isnan(out['vye_boot'][good])) == 0

    # For "bad" stars, all bootstrap vals should be nans
    assert np.sum(np.isfinite(out['xe_boot'][bad])) == 0
    assert np.sum(np.isfinite(out['ye_boot'][bad])) == 0
    assert np.sum(np.isfinite(out['vxe_boot'][bad])) == 0
    assert np.sum(np.isfinite(out['vye_boot'][bad])) == 0

    return

def test_use_vel():
    """
    Make sure PMs are calculated for use_vel=False option
    in MosaicToRef. use_vel only refers to whether or not you use proper motions
    in reference list to propagate reference stars. So,
    should still calculate proper motions across the starlists here.
    """
    # Define match parameters
    ref_no_vel = Table.read('ref.lis', format='ascii')

    list1 = Table.read('E.lis', format='ascii')
    list2 = Table.read('F.lis', format='ascii')

    list1 = starlists.StarList.from_table(list1)
    list2 = starlists.StarList.from_table(list2)
        
    # Set parameters for alignment
    transModel = transforms.PolyTransform
    trans_args = {'order':2}
    N_loop = 1
    dr_tol = 0.08
    dm_tol = 99
    outlier_tol = None
    mag_lim = None
    ref_mag_lim = None
    weights = 'both,var'
    mag_trans = False

    n_boot = 15
    boot_epochs_min=-1

    # Run align with use_vel=False.
    match = align.MosaicToRef(ref_no_vel, [list1, list2], iters=N_loop, dr_tol=dr_tol,
                                  dm_tol=dm_tol, outlier_tol=outlier_tol,
                                  trans_class=transModel,
                                  trans_args=trans_args,
                                  mag_trans=mag_trans,
                                  mag_lim=mag_lim,
                                  ref_mag_lim=ref_mag_lim,
                                  weights=weights,
                                  use_vel=False,
                                  use_ref_new=False,
                                  update_ref_orig=False,
                                  init_guess_mode='name',
                                  verbose=False)
    match.fit()

    # Make sure velocities are calculated



    # Make sure calc_boostrap still works with this use case

    pdb.set_trace()

    return

def test_transform_xym():
    """
    Test to make sure transforms are being done to mags only
    if mag_trans = True. This can cause subtle bugs 
    otherwise
    """
    #---Align 1: self.mag_Trans = False---#
    ref = Table.read('ref_vel.lis', format='ascii')
    list1 = Table.read('E.lis', format='ascii')
    list2 = Table.read('F.lis', format='ascii')

    list1 = starlists.StarList.from_table(list1)
    list2 = starlists.StarList.from_table(list2)
    
    # Set parameters for alignment
    transModel = transforms.PolyTransform
    trans_args = {'order':2}
    N_loop = 1
    dr_tol = 0.08
    dm_tol = 99
    outlier_tol = None
    mag_lim = None
    ref_mag_lim = None
    weights = 'both,var'
    n_boot = 15

    mag_trans = False

    # Run FLYSTAR, with bootstraps
    match1 = align.MosaicToRef(ref, [list1, list2], iters=N_loop, dr_tol=dr_tol,
                                  dm_tol=dm_tol, outlier_tol=outlier_tol,
                                  trans_class=transModel,
                                  trans_args=trans_args,
                                  mag_trans=mag_trans,
                                  mag_lim=mag_lim,
                                  ref_mag_lim=ref_mag_lim,
                                  weights=weights,
                                  use_vel=False,
                                  use_ref_new=False,
                                  update_ref_orig=False,
                                  init_guess_mode='name',
                                  verbose=False)

    match1.fit()
    match1.calc_bootstrap_errors(n_boot=n_boot)

    # Make sure all transformations have mag_offset = 0 
    trans_list = match1.trans_list

    for ii in trans_list:
        assert ii.mag_offset == 0

    # Check that no mag transformation has been applied to m col in ref_table
    tab1 = match1.ref_table
    assert np.all(tab1['m'] == tab1['m_orig'])
    
    # Check me_boost == 0 or really small (should be the case
    # since we don't transform mags)
    assert np.isclose(np.max(tab1['me_boot']), 0, rtol=10**-5)
    print('Done mag_trans = False case')

    #---Align 2: self.mag_Trans = True---#
    # Repeat, this time with mag_trans = False
    mag_trans = True
    match2 = align.MosaicToRef(ref, [list1, list2], iters=N_loop, dr_tol=dr_tol,
                                  dm_tol=dm_tol, outlier_tol=outlier_tol,
                                  trans_class=transModel,
                                  trans_args=trans_args,
                                  mag_trans=mag_trans,
                                  mag_lim=mag_lim,
                                  ref_mag_lim=ref_mag_lim,
                                  weights=weights,
                                  use_vel=False,
                                  use_ref_new=False,
                                  update_ref_orig=False,
                                  init_guess_mode='name',
                                  verbose=False)

    match2.fit()
    match2.calc_bootstrap_errors(n_boot=n_boot)


    # Make sure all transformations have correct mag offset
    trans_list2 = match2.trans_list

    for ii in trans_list2:
        assert ii.mag_offset > 20

    # Make sure final table mags have transform applied (i.e, 
    tab2 = match2.ref_table
    assert np.all(tab2['m'] != tab2['m_orig'])
    
    # Check me_boost > 0
    assert np.min(tab2['me_boot']) > 10**-3

    print('Done mag_trans = True case')
   
    return

def test_MosaicToRef_mag_bug():
    """
    Bug found by Tuan Do on 2020-04-12.
    """
    make_fake_starlists_poly1_vel()

    ref_list = starlists.StarList.from_lis_file('random_0.lis', error=False)
    lists = [ref_list]

    msc = align.MosaicToRef(ref_list, lists, 
                              mag_trans=True,
                              iters=1,                              
                              dr_tol=[0.2], dm_tol=[1],
                              outlier_tol=None,
                              trans_class=transforms.PolyTransform,
                              trans_args=[{'order': 1}],
                              use_vel=False,
                              use_ref_new=False,
                              update_ref_orig=False,
                              verbose=True)

    msc.fit()

    out_tab = msc.ref_table

    # The issue is that in the initial guess with
    #   mag_trans = True
    # somehow the transformed magnitudes are nan.
    # This causes zero matches to occur.
    assert len(out_tab) == len(ref_list)

    return
