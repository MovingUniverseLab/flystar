from flystar import align
from flystar import starlists
from flystar import startables
from flystar import transforms
from astropy.table import Table
import numpy as np
import pylab as plt
import pdb

def test_mosaic_lists_shifts():
    """
    Cross-match and align 4 starlists.
    """
    input_shifts = make_fake_starlists_shifts()
    
    list_files = ['random_0.lis',
                      'random_shift_1.lis',
                      'random_shift_2.lis',
                      'random_shift_3.lis',
                      'random_shift_4.lis']
    lists = [starlists.StarList.from_lis_file(lf, error=False) for lf in list_files]

    star_table, trans_table = align.mosaic_lists(lists, ref_index=0, iters=2,
                                                 trans_class=transforms.PolyTransform,
                                                 trans_args={'order': 1})

    assert len(star_table) == 200
    assert star_table['x'].shape == (200, 5)
    assert star_table['name_in_list'][0, 0] == 'star_000'
    assert star_table['name_in_list'][0, 1] == 'star_000'
    assert star_table['name_in_list'][0, 2] == 'star_000'
    assert star_table['name_in_list'][0, 3] == 'star_000'
    assert star_table['name_in_list'][0, 4] == 'star_000'

    # The 'x' column should contain the transformed coordinates for each epoch.
    # These should be the nearly same within ~0.2 pixels.
    assert star_table['x'][0].std() < 0.2
    assert star_table['x_orig'][0, 0] == lists[0]['x'][0]

    return trans_table[0]
    
    

def test_mosaic_lists():
    """
    Cross-match and align 4 starlists.
    """
    list_files = ['A.lis', 'B.lis', 'C.lis', 'D.lis']
    lists = [starlists.StarList.from_lis_file(lf) for lf in list_files]

    star_table, trans_table = align.mosaic_lists(lists, ref_index=0, iters=2,
                                                 dr_tol=[3, 3], dm_tol=[1, 1],
                                                 trans_class=transforms.PolyTransform,
                                                 trans_args={'order': 2})

    return
    

def test_mosaic_lists_vel():
    xy_trans_in, m_trans_in = make_fake_starlists_poly1_vel()

    ref_file = 'random_vel_ref.fits'
    list_files = ['random_vel_0.fits',
                  'random_vel_1.fits',
                  'random_vel_2.fits',
                  'random_vel_3.fits']
        
    lists = [starlists.StarList.read(lf) for lf in list_files]

    star_table, trans_table = align.mosaic_lists(lists, ref_index=2, iters=2,
                                                 dr_tol=[3, 3], dm_tol=[1, 1],
                                                 trans_class=transforms.PolyTransform,
                                                 trans_args={'order': 1})

    return
    

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
    assert (msc.ref_table['m0e'] < 1.0).all() # less than 0.5 mag
    
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

    # Check our status columsn
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

def make_fake_starlists_poly1():
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


def make_fake_starlists_poly1_vel():
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

def test_mosaic_lists_keck_hst():
    from microlens.jlu import pm_tools as p
    
    ##############################
    # Prepare starlists
    ##############################
    
    ############### GAIA ###############
    gaia_file = "/u/fatima/Desktop/Haynes/astrometry/MB98_36as_box.csv"
    gaia_data = Table.read(gaia_file, format='csv')
    gaia_list = p.gaia_starlist(gaia_file, 'mb980006')  # Make starlist from gaia

    # Name two stars for matching purposes 
    gaia_list, Target_G   = p.rename_star(gaia_list,  0.0,  0.0, 'MB980006', gaia=True)
    gaia_list, RefStar1_G = p.rename_star(gaia_list,  0.1,  4.4, 'RefStar1', gaia=True)
    gaia_list, RefStar2_G = p.rename_star(gaia_list, -3.5,  2.4, 'RefStar2', gaia=True)
    gaia_list, RefStar3_G = p.rename_star(gaia_list, -3.6, -2.3, 'RefStar3', gaia=True)
    gaia_list, RefStar3_G = p.rename_star(gaia_list,  7.0,  1.5, 'RefStar4', gaia=True)
    gaia_list, RefStar3_G = p.rename_star(gaia_list, -4.3, -9.4, 'RefStar5', gaia=True)


    gaia_list = Table(gaia_list, masked=False)
    gaia_list = starlists.StarList.from_table(gaia_list)
    gaia_list.rename_column('t', 't0')

    gaia_list = gaia_list.filled() #all stars
    match_list = gaia_list[p.match_star_ind(gaia_list)] #4 named stars

    ############### KECK ###############
    keck_file_1 = '/u/fatima/Desktop/Haynes/astrometry/mag16jul14_mb980006_kp_rms_named.lis'
    keck_file_2 = '/u/fatima/Desktop/Haynes/astrometry/mag17jun08_mb980006_kp_rms_named.lis'
    kLis1 = p.starlists.StarList.from_lis_file(keck_file_1, error=True)
    kLis2 = p.starlists.StarList.from_lis_file(keck_file_2, error=True)

    # Name a few stars for matching purposes
    kLis1 = p.rename_star(kLis1, 558, 930, 'RefStar1')
    kLis1 = p.rename_star(kLis1, 196, 729, 'RefStar2')
    kLis1 = p.rename_star(kLis1, 194, 268, 'RefStar3')
    kLis1 = p.rename_star(kLis1, 553, 486, 'MB980006')

    kLis2 = p.rename_star(kLis2, 537, 930, 'RefStar1')
    kLis2 = p.rename_star(kLis2, 179, 727, 'RefStar2')
    kLis2 = p.rename_star(kLis2, 171, 262, 'RefStar3')
    kLis2 = p.rename_star(kLis2, 532, 488, 'MB980006')
    
    keck_lists = [kLis1, kLis2]

    ############### HST ###############
    hst_F439W_file = '/u/fatima/Desktop/Haynes/hst_data/MB980006/F439W/01.XYM/u65c030dr_c0f.xym' #not using for now
    hst_F555W_file = '/u/fatima/Desktop/Haynes/hst_data/MB980006/F555W/01.XYM/u65c0308m_c0f.xym'
    hst_F675W_file = '/u/fatima/Desktop/Haynes/hst_data/MB980006/F675W/01.XYM/u65c0306r_c0f.xym'
    hst_F814W_file = '/u/fatima/Desktop/Haynes/hst_data/MB980006/F814W/01.XYM/u65c0302r_c0f.xym'

    hst_files = [hst_F439W_file, hst_F555W_file, hst_F675W_file, hst_F814W_file]
    hst_lists = []

    for file in hst_files:
        tab = Table.read(file, format='ascii')
        chip_ind = np.where((tab['col1']<800)&(tab['col2']<800))
        x = tab['col1'][chip_ind]
        y = tab['col2'][chip_ind]
        m = tab['col3'][chip_ind]
        name = ['s1_{0:04d}'.format(ii) for ii in range(len(x))]
        t = [2000+(174/365.25)]*len(x)
        new_tab = Table([name, t, x, y, m], names=['name', 't', 'x', 'y', 'm'])
        starlist = starlists.StarList.from_table(new_tab)
        hst_lists.append(starlist)

    # Reference stars for 555, 675, and 814 (439 is so different that it needs unique matches to gaia)
    x1_guesses = [661, 666, 661]
    y1_guesses = [433, 444, 433]
    x2_guesses = [597, 602, 597]
    y2_guesses = [496, 508, 496]
    x3_guesses = [500, 505, 501]
    y3_guesses = [472, 484, 472]
    xt_guesses = [566, 571, 566]
    yt_guesses = [407, 419, 407]

    for ii in range(len(hst_lists)):
        if ii==0:
            pass
        else:
            hst_lists[ii] = p.rename_star(hst_lists[ii], x1_guesses[ii-1], y1_guesses[ii-1], 'RefStar1')
            hst_lists[ii] = p.rename_star(hst_lists[ii], x2_guesses[ii-1], y2_guesses[ii-1], 'RefStar2')
            hst_lists[ii] = p.rename_star(hst_lists[ii], x3_guesses[ii-1], y3_guesses[ii-1], 'RefStar3')
            hst_lists[ii] = p.rename_star(hst_lists[ii], xt_guesses[ii-1], yt_guesses[ii-1], 'MB980006')

    hst_lists[0] = p.rename_star(hst_lists[0], 566, 407, 'MB980006')
    hst_lists[0] = p.rename_star(hst_lists[0], 643, 266, 'RefStar4')
    hst_lists[0] = p.rename_star(hst_lists[0], 346, 443, 'RefStar5')

    ##############################
    # Alignment
    ##############################
    iters = 3
    dr_tol = [1.0, 0.5, 0.2]
    dm_tol = [3.0, 3.0, 3.0]
    msc = align.MosaicToRef(gaia_list, keck_lists + hst_lists, iters=iters,
                                 dr_tol=dr_tol, dm_tol=40,
                                 outlier_tol=None, mag_lim=None,
                                 trans_class=transforms.PolyTransform,
                                 trans_input=None,
                                 trans_args=[{'order': 1}]*iters, 
                                 use_vel=True, weights=None,
                                 update_ref_orig=False, use_ref_new=False, mag_trans=True,
                                 init_guess_mode='name', verbose=False)

    msc.fit()
    tab = msc.ref_table

    assert len(tab) > 1000

    return tab
              

        
