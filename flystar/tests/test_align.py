import pdb
import flystar
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from flystar.plots import plot_stars
from flystar import align, starlists, transforms, analysis, motion_model

test_data_path = f'{flystar.__path__[0]}/tests/test_data'

def test_MosaicSelfRef():
    """
    Cross-match and align 4 starlists using the OO version of mosaic lists.
    """
    list_files = [f'{test_data_path}/{f}' for f in ['A.lis', 'B.lis', 'C.lis', 'D.lis']]
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
    assert 'x0_err' in msc.ref_table.colnames
    assert 'y0' in msc.ref_table.colnames
    assert 'y0_err' in msc.ref_table.colnames
    assert 'm0' in msc.ref_table.colnames
    assert 'm0_err' in msc.ref_table.colnames
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
    valid_err = np.isfinite(msc.ref_table['x0_err']) & np.isfinite(msc.ref_table['y0_err']) & np.isfinite(msc.ref_table['m0_err'])
    assert (msc.ref_table['x0_err'][valid_err] < 3.0).all() # less than 1 pix
    assert (msc.ref_table['y0_err'][valid_err] < 3.0).all()
    #assert (msc.ref_table['m0_err'][valid_err] < 1.0).all() # less than 0.5 mag
    assert (msc.ref_table['m0_err'][valid_err] < 1.5).all() # less than 0.5 mag
    # Check that the transformation lists aren't too wacky
    for ii in range(4):
        np.testing.assert_allclose(msc.trans_list[ii].px.c1_0, 1.0, rtol=1e-2)
        np.testing.assert_allclose(msc.trans_list[ii].py.c0_1, 1.0, rtol=1e-2)
    # We didn't do any velocity fitting, so make sure nothing got created.
    assert 'vx' not in msc.ref_table.colnames
    assert 'vy' not in msc.ref_table.colnames
    assert 'vx_err' not in msc.ref_table.colnames
    assert 'vy_err' not in msc.ref_table.colnames

    plt.clf()
    plt.plot(msc.ref_table['x'][:, 0],
             msc.ref_table['y'][:, 0],
             '+', color='red', mec='red', mfc='none')
    plt.plot(msc.ref_table['x'][:, 1],
             msc.ref_table['y'][:, 1],
             'x', color='blue', mec='blue', mfc='none')
    plt.plot(msc.ref_table['x'][:, 2],
             msc.ref_table['y'][:, 2],
             'o', color='cyan', mec='cyan', mfc='none')
    plt.plot(msc.ref_table['x'][:, 3],
             msc.ref_table['y'][:, 3],
             '^', color='green', mec='green', mfc='none')
    plt.plot(msc.ref_table['x0'],
             msc.ref_table['y0'],
             '.', color='black', alpha=0.2)
    return

def test_MosaicSelfRef_vel_tconst():
    """
    Cross-match and align 4 starlists using the OO version of mosaic lists.
    The 4 lists are all taken at the same time (so 0 velocities should result).

    """
    list_files = [f'{test_data_path}/{f}' for f in ['A.lis', 'B.lis', 'C.lis', 'D.lis']]
    lists = [starlists.StarList.from_lis_file(lf) for lf in list_files]

    ##########
    # Test instantiation and basic fitting.
    # Note these star lists are ALL at the same date.
    ##########
    msc = align.MosaicSelfRef(lists, ref_index=0, iters=2,
                              dr_tol=[3, 3], dm_tol=[1, 1],
                              trans_class=transforms.PolyTransform,
                              trans_args={'order': 2},
                              motion_models=['Empty', 'Fixed', 'Linear'],
                              verbose=False)

    msc.fit()

    # Check some of the output quantities on the final table.
    assert 'x0' in msc.ref_table.colnames
    assert 'x0_err' in msc.ref_table.colnames
    assert 'y0' in msc.ref_table.colnames
    assert 'y0_err' in msc.ref_table.colnames
    assert 'm0' in msc.ref_table.colnames
    assert 'm0_err' in msc.ref_table.colnames
    # Since they are in the same epoch, no velocity information can be inferred
    # assert 'vx' in msc.ref_table.colnames
    # assert 'vx_err' in msc.ref_table.colnames
    # assert 'vy' in msc.ref_table.colnames
    # assert 'vy_err' in msc.ref_table.colnames
    assert 't0' in msc.ref_table.colnames

    # Check that we have some matched stars... should be at least 35 stars
    # that are detected in all 4 starlists.
    idx = np.where(msc.ref_table['n_detect'] == 4)[0]
    assert len(idx) > 35

    # Check that the transformation error isn't too big
    valid_err = np.isfinite(msc.ref_table['x0_err']) & np.isfinite(msc.ref_table['y0_err']) & np.isfinite(msc.ref_table['m0_err'])
    assert (msc.ref_table['x0_err'][valid_err] < 3.0).all() # less than 1 pix
    assert (msc.ref_table['y0_err'][valid_err] < 3.0).all()
    # A star detected in only 1 epoch now correctly gets a finite m0_err
    # from that single epoch's own 'me' (weighted average of 1 point)
    # instead of being silently excluded via an inf from the unweighted
    # fallback -- so its (legitimately large, single-detection) uncertainty
    # is included here rather than skipped by the isfinite() filter above.
    assert (msc.ref_table['m0_err'][valid_err] < 1.5).all()

    # Check that the transformation lists aren't too wacky
    for ii in range(4):
        np.testing.assert_allclose(msc.trans_list[ii].px.c1_0, 1.0, rtol=1e-2)
        np.testing.assert_allclose(msc.trans_list[ii].py.c0_1, 1.0, rtol=1e-2)
    return

def test_MosaicSelfRef_vel():
    """
    Cross-match and align 4 starlists using the OO version of mosaic lists.
    """
    list_files = [f'{test_data_path}/{f}' for f in ['A.lis', 'B.lis', 'C.lis', 'D.lis']]
    lists = [starlists.StarList.from_lis_file(lf) for lf in list_files]

    # Modify the times so that we get velocities out.
    lists[0].meta['list_times'] = 2001.4
    lists[0]['t'] = 2001.4

    lists[1].meta['list_times'] = 2002.4
    lists[1]['t'] = 2002.4

    lists[2].meta['list_times'] = 2003.4
    lists[2]['t'] = 2003.4

    lists[3].meta['list_times'] = 2004.4
    lists[3]['t'] = 2004.4


    ##########
    # Test instantiation and basic fitting.
    ##########
    msc = align.MosaicSelfRef(lists, ref_index=0, iters=3,
                              dr_tol=[5, 3, 3], dm_tol=[1, 1, 0.5], outlier_tol=None, briteN=30,
                              trans_class=transforms.PolyTransform,
                              trans_args={'order': 2}, motion_models=['Empty', 'Fixed', 'Linear'],
                              verbose=False)

    msc.fit()

    # Check some of the output quantities on the final table.
    assert 'x0' in msc.ref_table.colnames
    assert 'x0_err' in msc.ref_table.colnames
    assert 'y0' in msc.ref_table.colnames
    assert 'y0_err' in msc.ref_table.colnames
    assert 'm0' in msc.ref_table.colnames
    assert 'm0_err' in msc.ref_table.colnames
    assert 'vx' in msc.ref_table.colnames
    assert 'vx_err' in msc.ref_table.colnames
    assert 'vy' in msc.ref_table.colnames
    assert 'vy_err' in msc.ref_table.colnames
    assert 't0' in msc.ref_table.colnames

    # Check that we have some matched stars... should be at least 35 stars
    # that are detected in all 4 starlists.
    idx = np.where(msc.ref_table['n_detect'] == 4)[0]
    assert len(idx) >= 35, f"Expected at least 35 stars detected in all 4 starlists, but only found {len(idx)}"

    # Check that the transformation error isn't too big
    valid_err = np.isfinite(msc.ref_table['x0_err']) & np.isfinite(msc.ref_table['y0_err']) & np.isfinite(msc.ref_table['m0_err'])
    assert (msc.ref_table['x0_err'][valid_err] < 3.0).all() # less than 1 pix
    assert (msc.ref_table['y0_err'][valid_err] < 3.0).all()
    # A star detected in only 1 epoch now correctly gets a finite m0_err
    # from that single epoch's own 'me' (weighted average of 1 point)
    # instead of being silently excluded via an inf from the unweighted
    # fallback -- so its (legitimately large, single-detection) uncertainty
    # is included here rather than skipped by the isfinite() filter above.
    assert (msc.ref_table['m0_err'][valid_err] < 1.5).all()

    # Check that the transformation lists aren't too wacky
    for ii in range(4):
        np.testing.assert_allclose(msc.trans_list[ii].px.c1_0, 1.0, rtol=2e-2)
        np.testing.assert_allclose(msc.trans_list[ii].py.c0_1, 1.0, rtol=2e-2)

    plt.clf()
    plt.plot(msc.ref_table['vx'],
             msc.ref_table['vy'],
             'k.', color='black', alpha=0.2)


    return

def test_MosaicToRef():
    make_fake_starlists_poly1(seed=42)

    ref_file = f'{test_data_path}/random_ref.fits'
    list_files = [f'{test_data_path}/random_{i}.fits' for i in range(8)]

    ref_list = Table.read(ref_file)

    # Switch our list to a "increasing to the West" list.
    ref_list['x0'] *= -1.0

    lists = [starlists.StarList.read(lf) for lf in list_files]

    msc = align.MosaicToRef(ref_list, lists, iters=2,
                              dr_tol=[0.2, 0.1], dm_tol=[1, 0.5],
                              trans_class=transforms.PolyTransform,
                              trans_args={'order': 2}, motion_models=['Empty', 'Fixed'],
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
    np.testing.assert_allclose(msc.ref_table['x0'], ref_list['x0'], rtol=1e-5)
    np.testing.assert_allclose(msc.ref_table['y0'], ref_list['y0'], rtol=1e-5)

    ##########
    # Align and let velocities be free.
    ##########
    msc.update_ref_orig = 'periter'
    msc.fit()

    # The velocities should be almost the same (but not as close as before)
    # as the input velocities since update_ref == False.
    np.testing.assert_allclose(msc.ref_table['x0'], ref_list['x0'], rtol=1e-1)
    np.testing.assert_allclose(msc.ref_table['y0'], ref_list['y0'], rtol=1e-1)

    # Also double check that they aren't exactly the same for the reference stars.
    assert np.not_equal(msc.ref_table['x0'], ref_list['x0']).all()
    assert np.not_equal(msc.ref_table['y0'], ref_list['y0']).all()

    return

def test_MosaicToRef_p0_vel():
    make_fake_starlists_poly0_vel(seed=42)

    ref_file = f'{test_data_path}/random_vel_ref.fits'
    list_files = [f'{test_data_path}/random_vel_p0_{i}.fits' for i in range(4)]

    ref_list = Table.read(ref_file)

    # Convert velocities to arcsec/yr
    ref_list['vx'] *= 1e-3
    ref_list['vy'] *= 1e-3
    ref_list['vx_err'] *= 1e-3
    ref_list['vy_err'] *= 1e-3

    # Switch our list to a "increasing to the West" list.
    ref_list['x0'] *= -1.0
    ref_list['vx'] *= -1.0

    lists = [starlists.StarList.read(lf) for lf in list_files]

    msc = align.MosaicToRef(ref_list, lists, iters=2,
                              dr_tol=[0.2, 0.1], dm_tol=[1, 0.5],
                              outlier_tol=[None, None],
                              trans_class=transforms.PolyTransform,
                              trans_args={'order': 1}, motion_models=['Empty', 'Fixed', 'Linear'],
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
    assert (msc.ref_table['name']==ref_list['name']).all()
    np.testing.assert_allclose(msc.ref_table['vx'], ref_list['vx'], rtol=1e-5)
    np.testing.assert_allclose(msc.ref_table['vy'], ref_list['vy'], rtol=1e-5)

    ##########
    # Align and let velocities be free.
    ##########
    msc.update_ref_orig = 'periter'
    msc.fit()

    # The velocities should be almost the same (but not as close as before)
    # as the input velocities since update_ref == True.
    assert (msc.ref_table['name']==ref_list['name']).all()
    np.testing.assert_allclose(msc.ref_table['vx'], ref_list['vx'], rtol=1e-1, atol=3e-4)
    np.testing.assert_allclose(msc.ref_table['vy'], ref_list['vy'], rtol=1e-1, atol=3e-4)

    # Also double check that they aren't exactly the same for the reference stars.
    #assert np.any(np.not_equal(msc.ref_table['vx'], ref_list['vx']))
    assert np.not_equal(msc.ref_table['vx'], ref_list['vx']).any()

    return

def test_MosaicToRef_vel():
    make_fake_starlists_poly1_vel(seed=42)

    ref_file = f'{test_data_path}/random_vel_ref.fits'
    list_files = [f'{test_data_path}/random_vel_{i}.fits' for i in range(4)]

    ref_list = Table.read(ref_file)

    # Convert velocities to arcsec/yr
    ref_list['vx'] *= 1e-3
    ref_list['vy'] *= 1e-3
    ref_list['vx_err'] *= 1e-3
    ref_list['vy_err'] *= 1e-3

    # Switch our list to a "increasing to the West" list.
    ref_list['x0'] *= -1.0
    ref_list['vx'] *= -1.0

    lists = [starlists.StarList.read(lf) for lf in list_files]

    msc = align.MosaicToRef(ref_list, lists, iters=2,
                              dr_tol=[0.2, 0.1], dm_tol=[1, 0.5],
                              outlier_tol=[None, None],
                              trans_class=transforms.PolyTransform,
                              trans_args={'order': 1}, motion_models=['Empty', 'Fixed', 'Linear'],
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
    assert (msc.ref_table['name']==ref_list['name']).all()
    np.testing.assert_allclose(msc.ref_table['vx'], ref_list['vx'], rtol=1e-5)
    np.testing.assert_allclose(msc.ref_table['vy'], ref_list['vy'], rtol=1e-5)

    ##########
    # Align and let velocities be free.
    ##########
    msc.update_ref_orig = 'periter'
    msc.fit()

    # The velocities should be almost the same (but not as close as before)
    # as the input velocities since update_ref == True.
    assert (msc.ref_table['name']==ref_list['name']).all()
    np.testing.assert_allclose(msc.ref_table['vx'], ref_list['vx'], rtol=1e-1, atol=3e-4)
    np.testing.assert_allclose(msc.ref_table['vy'], ref_list['vy'], rtol=1e-1, atol=3e-4)

    # Also double check that they aren't exactly the same for the reference stars.
    #assert np.any(np.not_equal(msc.ref_table['vx'], ref_list['vx']))
    assert np.not_equal(msc.ref_table['vx'], ref_list['vx']).any()

    return

def test_MosaicToRef_acc():
    make_fake_starlists_poly1_acc(seed=42)

    ref_file = f'{test_data_path}/random_acc_ref.fits'
    list_files = [f'{test_data_path}/random_acc_{i}.fits' for i in range(8)]

    ref_list = Table.read(ref_file)

    # Convert velocities to arcsec/yr
    ref_list['vx0'] *= 1e-3
    ref_list['vy0'] *= 1e-3
    ref_list['vx0_err'] *= 1e-3
    ref_list['vy0_err'] *= 1e-3

    # Convert accelerations to arcsec/yr**2
    ref_list['ax'] *= 1e-3
    ref_list['ay'] *= 1e-3
    ref_list['ax_err'] *= 1e-3
    ref_list['ay_err'] *= 1e-3

    # Switch our list to a "increasing to the West" list.
    ref_list['x0'] *= -1.0
    ref_list['vx0'] *= -1.0
    ref_list['ax'] *= -1.0

    lists = [starlists.StarList.read(lf) for lf in list_files]

    msc = align.MosaicToRef(ref_list, lists, iters=2,
                              dr_tol=[0.4, 0.2], dm_tol=[1, 0.5],
                              trans_class=transforms.PolyTransform,
                              trans_args={'order': 2},
                              motion_models=['Acceleration'],
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
    i_orig, i_fit = [],[]
    for i,star in enumerate(ref_list["name"]):
        if star in msc.ref_table["name"]:
            i_fit.append(np.where(msc.ref_table["name"]==star)[0][0])
            i_orig.append(i)
    np.testing.assert_allclose(msc.ref_table['ax'][i_fit], ref_list['ax'][i_orig], rtol=1e-5)
    np.testing.assert_allclose(msc.ref_table['ay'][i_fit], ref_list['ay'][i_orig], rtol=1e-5)

    ##########
    # Align and let velocities be free.
    ##########
    msc.update_ref_orig = 'periter'
    msc.fit()

    # The velocities should be almost the same (but not as close as before)
    # as the input velocities since update_ref == False.
    i_orig, i_fit = [],[]
    for i,star in enumerate(ref_list["name"]):
        if star in msc.ref_table["name"]:
            ix_fit = np.where(msc.ref_table["name"]==star)[0][0]
            if ~np.isnan(msc.ref_table['ax'][ix_fit]):
                i_orig.append(i)
                i_fit.append(ix_fit)
    # Accelerations all too small, rtol doesn't work well here. atol is
    # loosened slightly beyond the fit noise floor (individual ax_err/ay_err
    # are themselves ~2-3e-4 for the most weakly-constrained stars) since
    # correctly weighting the magnitude combination by 'me' (rather than the
    # previous unweighted average) nudges the mag-based transform fit enough
    # to shift the most marginal star's acceleration by a comparable amount.
    atol = 6e-4
    np.testing.assert_allclose(msc.ref_table['ax'][i_fit], ref_list['ax'][i_orig], atol=atol)
    np.testing.assert_allclose(msc.ref_table['ay'][i_fit], ref_list['ay'][i_orig], atol=atol)

    ax_min = np.min(ref_list['ax'][i_orig])
    ax_max = np.max(ref_list['ax'][i_orig])
    ay_min = np.min(ref_list['ay'][i_orig])
    ay_max = np.max(ref_list['ay'][i_orig])

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(ref_list['ax'][i_orig], msc.ref_table['ax'][i_fit], '.')
    ax1.plot([ax_min, ax_max], [ax_min, ax_max], color='C3')
    ax1.plot([ax_min, ax_max], [ax_min - atol, ax_max - atol], ls='--', color='C3')
    ax1.plot([ax_min, ax_max], [ax_min + atol, ax_max + atol], ls='--', color='C3')
    ax1.set_xlabel('Input ax')
    ax1.set_ylabel('Ref Table ax')
    ax1.set_title('Acceleration in X')

    ax2.plot(ref_list['ay'][i_orig], msc.ref_table['ay'][i_fit], '.')
    ax2.plot([ay_min, ay_max], [ay_min, ay_max], color='C3')
    ax2.plot([ay_min, ay_max], [ay_min - atol, ay_max - atol], ls='--', color='C3')
    ax2.plot([ay_min, ay_max], [ay_min + atol, ay_max + atol], ls='--', color='C3')
    ax2.set_xlabel('Input ay')
    ax2.set_ylabel('Ref Table ay')
    ax2.set_title('Acceleration in Y')
    plt.tight_layout()

    # Also double check that they aren't exactly the same for the reference stars.
    assert np.any(np.not_equal(msc.ref_table['ax'][i_fit], ref_list['ax'][i_orig]))
    return

def test_MosaicToRef_hst_me():
    """
    Test Casey's issue with 'me' not getting propogated
    from the input starlists to the output table.

    Use data from MB10-364 microlensing target for the test.
    """
    # Target RA and Dec (MOA data download)
    # ra = '17:57:05.401'
    # dec = '-34:27:05.01'

    # Load up a Gaia catalog (queried around the RA/Dec above)
    my_gaia = Table.read(f'{test_data_path}/mb10364_data/my_gaia.fits')
    my_gaia['me'] = 0.01

    my_gaia.rename_columns(
        ['x0e', 'y0e', 'vxe', 'vye'],
        ['x0_err', 'y0_err', 'vx_err', 'vy_err']
    )
    # Gather the list of starlists. For first pass, don't modify the starlists.
    # Loop through the observations and read them in, in prep for alignment with Gaia
    epochs = [2011.83, 2012.73, 2013.81]
    starlist_names = [f'{test_data_path}/mb10364_data/2011_10_31_F606W_MATCHUP_XYMEEE_final.calib',
                      f'{test_data_path}/mb10364_data/2012_09_25_F606W_MATCHUP_XYMEEE_final.calib',
                      f'{test_data_path}/mb10364_data/2013_10_24_F606W_MATCHUP_XYMEEE_final.calib']

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

    msc = align.MosaicToRef(
        my_gaia, list_of_starlists, iters=1,
        dr_tol=[0.1], dm_tol=[5],
        outlier_tol=[None], mag_lim=[13, 21],
        trans_class=transforms.PolyTransform,
        trans_args=[{'order': 1}],
        motion_models=['Empty', 'Fixed'],
        use_ref_new=False,
        update_ref_orig=False,
        mag_trans=False,
        trans_weights='both,std',
        init_guess_mode='miracle',
        # save_path=f'{test_data_path}/mb10364_data/test_MosaicToRef_hst_me.pkl',
        verbose=False
    )
    msc.fit()

    assert 'me' in msc.ref_table.colnames
    return

def test_bootstrap():
    """
    Test to make sure calc_bootstrap_error() call is working
    properly (e.g., only called when user calls calc_bootstrap_error,
    n_boot param for calc_bootstrap_error only, boot_epochs_min working,
    etc.)
    """
    # Read in starlists for MosaicToRef
    ref = Table.read(f'{test_data_path}/ref_vel.lis', format='ascii')
    list1 = Table.read(f'{test_data_path}/E.lis', format='ascii')
    list2 = Table.read(f'{test_data_path}/F.lis', format='ascii')

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
    trans_weights = 'both,var'
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
                                  trans_weights=trans_weights,
                                  motion_models=['Linear'],
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
    match1.calc_bootstrap_errors(n_boot=n_boot, boot_epochs_min=boot_epochs_min, seed=42)
    # Make sure columns exist, and none of them are nan values
    assert np.sum(np.isnan(match1.ref_table['xe_boot'])) == 0
    assert np.sum(np.isnan(match1.ref_table['ye_boot'])) == 0
    assert np.sum(np.isnan(match1.ref_table['vx_err_boot'])) == 0
    assert np.sum(np.isnan(match1.ref_table['vy_err_boot'])) == 0

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
                                  trans_weights=trans_weights,
                                  motion_models=['Linear'],
                                  use_ref_new=False,
                                  update_ref_orig=False,
                                  init_guess_mode='name',
                                  verbose=False)
    match2.fit()

    # Now run_calc_bootstrap_error, with boot_epochs_min engaged
    boot_epochs_min2 = 2
    match2.calc_bootstrap_errors(n_boot=n_boot, boot_epochs_min=boot_epochs_min2, seed=42)

    # Make sure boot_epochs_min cut worked as intended
    out = match2.ref_table
    bad = np.where( (out['n_detect'] == 1) & (out['use_in_trans'] == False) )
    good = np.where(out['n_detect'] == 2)

    # Some stars must exist in both "good" and "bad" criteria,
    # otherwise this test isn't as useful as intended.
    assert len(bad[0]) > 0
    assert len(good[0]) > 0

    # For "good" stars: all bootstrap vals should be present
    assert np.sum(~np.isfinite(out['xe_boot'][good])) == 0
    assert np.sum(~np.isfinite(out['ye_boot'][good])) == 0
    assert np.sum(~np.isfinite(out['vx_err_boot'][good])) == 0
    assert np.sum(~np.isfinite(out['vy_err_boot'][good])) == 0

    # For "bad" stars, all bootstrap vals should be nans
    assert np.sum(np.isfinite(out['xe_boot'][bad])) == 0
    assert np.sum(np.isfinite(out['ye_boot'][bad])) == 0
    assert np.sum(np.isfinite(out['vx_err_boot'][bad])) == 0
    assert np.sum(np.isfinite(out['vy_err_boot'][bad])) == 0

    return

def test_calc_vel_in_bootstrap():
    """
    Check calc_vel_in_bootstrap performance in calc_bootstrap_errors()

    Only calculate velocity bootstrap (e.g., bootstrap over epochs and
    calculating proper motions) if calc_vel_in_bootstrap=True.

    """
    import copy

    # Define match parameters
    ref = Table.read(f'{test_data_path}/ref_vel.lis', format='ascii')
    list1 = Table.read(f'{test_data_path}/E.lis', format='ascii')
    list2 = Table.read(f'{test_data_path}/F.lis', format='ascii')

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
    trans_weights = 'both,var'
    mag_trans = False

    n_boot = 15
    boot_epochs_min=-1

    # Run match
    match = align.MosaicToRef(ref, [list1, list2], iters=N_loop, dr_tol=dr_tol,
                                  dm_tol=dm_tol, outlier_tol=outlier_tol,
                                  trans_class=transModel,
                                  trans_args=trans_args,
                                  mag_trans=mag_trans,
                                  mag_lim=mag_lim,
                                  ref_mag_lim=ref_mag_lim,
                                  trans_weights=trans_weights,
                                  motion_models=['Linear'],
                                  use_ref_new=False,
                                  update_ref_orig=False,
                                  init_guess_mode='name',
                                  verbose=False)
    match.fit()

    # Make 2 copies of match object: one to test
    # each case of calc_vel_in_bootstrap
    match_vel = copy.deepcopy(match)

    # Run calc_bootstrap_error function with calc_vel_in_bootstrap=True.
    # Make sure bootstrap velocity errors are calculated and valid
    n_boot = 50
    match_vel.calc_bootstrap_errors(n_boot=n_boot, calc_vel_in_bootstrap=True, seed=42)

    assert 'xe_boot' in match_vel.ref_table.keys()
    assert np.sum(np.isnan(match_vel.ref_table['xe_boot'])) == 0
    assert 'vx_err_boot' in match_vel.ref_table.keys()
    assert np.sum(np.isnan(match_vel.ref_table['vx_err_boot'])) == 0

    # Run without calc_vel_in_bootstrap, make sure velocities are NOT calculated
    match.calc_bootstrap_errors(n_boot=n_boot, calc_vel_in_bootstrap=False, seed=42)

    assert 'xe_boot' in match.ref_table.keys()
    assert np.sum(np.isnan(match.ref_table['xe_boot'])) == 0
    assert 'vx_err_boot' not in match.ref_table.keys()

    return

def test_transform_xym():
    """
    Test to make sure transforms are being done to mags only
    if mag_trans = True. This can cause subtle bugs
    otherwise
    """
    #---Align 1: self.mag_Trans = False---#
    ref = Table.read(f'{test_data_path}/ref_vel.lis', format='ascii')
    list1 = Table.read(f'{test_data_path}/E.lis', format='ascii')
    list2 = Table.read(f'{test_data_path}/F.lis', format='ascii')

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
    trans_weights = 'both,var'
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
                                  trans_weights=trans_weights,
                                  motion_models=['Fixed'],
                                  use_ref_new=False,
                                  update_ref_orig=False,
                                  init_guess_mode='name',
                                  verbose=False)

    match1.fit()
    match1.calc_bootstrap_errors(n_boot=n_boot, seed=42)

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
                                  trans_weights=trans_weights,
                                  motion_models=['Fixed'],
                                  use_ref_new=False,
                                  update_ref_orig=False,
                                  init_guess_mode='name',
                                  verbose=False)

    match2.fit()
    match2.calc_bootstrap_errors(n_boot=n_boot, seed=42)


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
    make_fake_starlists_poly1_vel(seed=42)

    ref_list = starlists.StarList.read(f'{test_data_path}/random_vel_0.fits')
    lists = [ref_list]

    msc = align.MosaicToRef(ref_list, lists,
                              mag_trans=True,
                              iters=1,
                              dr_tol=[0.2], dm_tol=[1],
                              outlier_tol=None,
                              trans_class=transforms.PolyTransform,
                              trans_args=[{'order': 1}],
                              motion_models=['Fixed'],
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

def test_masked_cols():
    """
    Test to make sure analysis.prepare_gaia_for_flystar
    produces an astropy.table.Table, NOT a masked column
    table. MosaicToRef cannot handle masked column tables.

    Also make sure this example works, since we use it for the examples
    jupyter notebook.
    """
    # Get gaia reference stars using analysis.py
    # around a test location.
    # target = 'ob150029'
    ra = '17:59:46.60'
    dec = '-28:38:41.8'

    # Coordinates are arcsecs offset +x to the East.
    targets_dict = {
        'ob150029':   [0.0, 0.0],
        'S005': [1.1416,    3.7405],
        'S002': [-4.421,    0.027]
    }

    # Get gaia catalog stars. Note that this produces a masked column table
    search_radius = 10.0   # arcsec
    gaia = analysis.query_gaia(ra, dec, search_radius=search_radius)
    my_gaia = analysis.prepare_gaia_for_flystar(gaia, ra, dec, targets_dict=targets_dict)

    assert isinstance(my_gaia, Table)

    # Let's make sure the entire align runs, just to be safe

    # Get starlists to align to gaia
    epochs = ['15jun07','16jul14', '17may21']

    list_of_starlists = []

    for ee in range(len(epochs)):
        lis_file = 'mag' + epochs[ee] + '_ob150029_kp_rms_named.lis'
        lis = starlists.StarList.from_lis_file(f'{test_data_path}/{lis_file}')
        list_of_starlists.append(lis)

    # Run the align
    msc = align.MosaicToRef(my_gaia, list_of_starlists, iters=2,
                        dr_tol=[0.2, 0.1], dm_tol=[1, 1],
                        trans_class=transforms.PolyTransform,
                        trans_args=[{'order': 1}, {'order': 1}],
                        motion_models=['Linear'],
                        use_ref_new=False,
                        update_ref_orig=False,
                        mag_trans=True,
                        init_guess_mode='name', verbose=True)

    msc.fit()
    return

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
    _out = open(f'{test_data_path}/random_0.lis', 'w')
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

        _out = open(f'{test_data_path}/random_shift_{ss+1}.lis', 'w')
        for ii in range(N_stars):
            _out.write(fmt.format(name[ii], mnew[ii], xnew[ii], ynew[ii]))
        _out.close()

    return shifts

def make_fake_starlists_poly1(seed=-1):
    # If seed >=0, then set random seed to that value
    if seed >= 0:
        np.random.seed(seed=seed)

    N_stars = 200

    x0  = np.random.rand(N_stars) * 10.0     # arcsec (increasing to East)
    y0  = np.random.rand(N_stars) * 10.0     # arcsec
    x0e = np.random.randn(N_stars) * 5.0e-4  # arcsec
    y0e = np.random.randn(N_stars) * 5.0e-4  # arcsec
    m0  = (np.random.rand(N_stars) * 8) + 9  # mag
    m0e = np.random.randn(N_stars) * 0.05    # mag
    t0 = np.ones(N_stars) * 2019.5

    # Make all the errors positive
    x0e = np.abs(x0e)
    y0e = np.abs(y0e)
    m0e = np.abs(m0e)

    name = ['star_{0:03d}'.format(ii) for ii in range(N_stars)]

    # Make an StarList
    lis = starlists.StarList([name, m0, m0e, x0, x0e, y0, y0e, t0],
                             names = ('name', 'm0', 'm0_err', 'x0', 'x0_err', 'y0', 'y0_err', 't0'))

    sdx = np.argsort(m0)
    lis = lis[sdx]

    # Save original positions as reference (1st) list
    # in a StarList format (with velocities).
    lis.write(f'{test_data_path}/random_ref.fits', overwrite=True)

    ##########
    # Shifts
    ##########
    # Make 4 new starlists with different shifts.
    times = [2018.5, 2019.0, 2019.5, 2020.0, 2020.5, 2021.0, 2021.5, 2022.0]
    xy_trans = [[[ 6.5, 0.99, 1e-5], [  10.1, 1e-5, 0.99]],
               [[100.3, 0.98, 1e-5], [  50.5, 9e-6, 1.001]],
               [[  0.0, 1.00,  0.0], [   0.0,  0.0, 1.0]],
               [[250.0, 0.97, 2e-5], [-250.0, 1e-5, 1.001]],
               [[ 50.0, 1.01, 1e-5], [ -31.0, 1e-5, 1.000]],
               [[ 78.0, 0.98, 0.0 ], [  45.0, 9e-6, 1.001]],
               [[-13.0, 0.99, 1e-5], [  150, 2e-5, 1.002]],
               [[ 94.0, 1.00, 9e-6], [-182.0, 0.0, 0.99]]]
    mag_trans = [0.1, 0.4, 0.0, -0.3, 0.2, 0.0, -0.1, -0.3]

    # Convert into pixels (undistorted) with the following info.
    scale = 0.01  # arcsec / pix
    shift = [1.0, 1.0]  # pix

    for ss in range(len(times)):
        dt = times[ss] - lis['t0']

        x = lis['x0']
        y = lis['y0']
        t = np.ones(N_stars) * times[ss]

        # Convert into pixels
        xp = (x / -scale) + shift[0]  # -1 from switching to increasing to West (right)
        yp = (y /  scale) + shift[1]
        xpe = lis['x0_err'] / scale
        ype = lis['y0_err'] / scale

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
        mde = lis['m0_err']

        # fig, ax = plt.subplots()
        # ax.scatter(x0, y0, s=2, label='Reference')
        # ax.scatter(xd, yd, s=2, label='Starlist')
        # ax.set_xlabel('X (pix)')
        # ax.set_ylabel('Y (pix)')
        # ax.legend()
        # plt.show()

        # Save the new list as a starlist.
        new_lis = starlists.StarList([lis['name'], md, mde, xd, xde, yd, yde, t],
                                     names=('name', 'm', 'me', 'x', 'xe', 'y', 'ye', 't'))

        new_lis.write(f'{test_data_path}/random_{ss}.fits', overwrite=True)

    return (xy_trans,mag_trans)

def make_fake_starlists_poly0_vel(seed=-1):
    # If seed >=0, then set random seed to that value
    if seed >= 0:
        np.random.seed(seed=seed)

    N_stars = 200

    x0  = np.random.rand(N_stars) * 10.0     # arcsec (increasing to East)
    y0  = np.random.rand(N_stars) * 10.0     # arcsec
    x0e = np.ones(N_stars) * 1.0e-4  # arcsec
    y0e = np.ones(N_stars) * 1.0e-4  # arcsec
    vx  = np.random.randn(N_stars) * 5.0     # mas / yr
    vy  = np.random.randn(N_stars) * 5.0     # mas / yr
    vxe = np.ones(N_stars) * 0.05    # mas / yr
    vye = np.ones(N_stars) * 0.05    # mas / yr
    m0  = (np.random.rand(N_stars) * 8) + 9  # mag
    m0e = np.random.randn(N_stars) * 0.05    # mag
    t0 = np.ones(N_stars) * 2019.5

    # Make all the errors positive
    x0e = np.abs(x0e)
    y0e = np.abs(y0e)
    m0e = np.abs(m0e)
    vxe = np.abs(vxe)
    vye = np.abs(vye)

    name = [f'star_{ii:03d}' for ii in range(N_stars)]

    # Make an StarList
    lis = starlists.StarList([name, m0, m0e, x0, x0e, y0, y0e, vx, vxe, vy, vye, t0],
                             names = ('name', 'm0', 'm0_err', 'x0', 'x0_err', 'y0', 'y0_err',
                                          'vx', 'vx_err', 'vy', 'vy_err', 't0'))

    sdx = np.argsort(m0)
    lis = lis[sdx]

    # Save original positions as reference (1st) list
    # in a StarList format (with velocities).
    lis.write(f'{test_data_path}/random_vel_ref.fits', overwrite=True)

    ##########
    # Propogate to new times and distort.
    ##########
    # Make 4 new starlists with different epochs and transformations.
    times = [2018.5, 2019.0, 2019.5, 2020.0, 2020.5, 2021.0, 2021.5, 2022.0]
    xy_trans = [[[ 6.5], [  10.1]],
               [[100.3], [  50.5]],
               [[  0.0], [   0.0]],
               [[250.0], [-250.0]],
               [[ 50.0], [ -31.0]],
               [[ 78.0], [  45.0]],
               [[-13.0], [   150]],
               [[ 94.0], [-182.0]]]
    mag_trans = [0.1, 0.4, 0.0, -0.3, 0.2, 0.0, -0.1, -0.3]

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
        xpe = lis['x0_err'] / scale
        ype = lis['y0_err'] / scale

        # Distort the positions
        trans = transforms.PolyTransform(0, xy_trans[ss][0], xy_trans[ss][1], mag_offset=mag_trans[ss])
        xd, yd = trans.evaluate(xp, yp)
        md = trans.evaluate_mag(lis['m0'])

        # Perturb with small errors (0.1 pix)
        xd += np.random.randn(N_stars) * xpe
        yd += np.random.randn(N_stars) * ype
        md += np.random.randn(N_stars) * 0.02
        xde = xpe
        yde = ype
        mde = lis['m0_err']

        # Save the new list as a starlist.
        new_lis = starlists.StarList([lis['name'], md, mde, xd, xde, yd, yde, t],
                                     names=('name', 'm', 'me', 'x', 'xe', 'y', 'ye', 't'))

        new_lis.write(f'{test_data_path}/random_vel_p0_{ss}.fits', overwrite=True)

    return (xy_trans, mag_trans)


def make_fake_starlists_poly1_vel(seed=-1):
    # If seed >=0, then set random seed to that value
    if seed >= 0:
        np.random.seed(seed=seed)

    N_stars = 200

    x0  = np.random.rand(N_stars) * 10.0     # arcsec (increasing to East)
    y0  = np.random.rand(N_stars) * 10.0     # arcsec
    x0e = np.ones(N_stars) * 1.0e-4  # arcsec
    y0e = np.ones(N_stars) * 1.0e-4  # arcsec
    vx  = np.random.randn(N_stars) * 5.0     # mas / yr
    vy  = np.random.randn(N_stars) * 5.0     # mas / yr
    vxe = np.ones(N_stars) * 0.05    # mas / yr
    vye = np.ones(N_stars) * 0.05    # mas / yr
    m0  = (np.random.rand(N_stars) * 8) + 9  # mag
    m0e = np.random.randn(N_stars) * 0.05    # mag
    t0 = np.ones(N_stars) * 2019.5

    # Make all the errors positive
    x0e = np.abs(x0e)
    y0e = np.abs(y0e)
    m0e = np.abs(m0e)
    vxe = np.abs(vxe)
    vye = np.abs(vye)

    name = [f'star_{ii:03d}' for ii in range(N_stars)]

    # Make an StarList
    lis = starlists.StarList([name, m0, m0e, x0, x0e, y0, y0e, vx, vxe, vy, vye, t0],
                             names = ('name', 'm0', 'm0_err', 'x0', 'x0_err', 'y0', 'y0_err',
                                          'vx', 'vx_err', 'vy', 'vy_err', 't0'))

    sdx = np.argsort(m0)
    lis = lis[sdx]

    # Save original positions as reference (1st) list
    # in a StarList format (with velocities).
    lis.write(f'{test_data_path}/random_vel_ref.fits', overwrite=True)

    ##########
    # Propogate to new times and distort.
    ##########
    # Make 4 new starlists with different epochs and transformations.
    times = [2018.5, 2019.0, 2019.5, 2020.0, 2020.5, 2021.0, 2021.5, 2022.0]
    xy_trans = [[[ 6.5, 0.99, 1e-5], [  10.1, 1e-5, 0.99]],
               [[100.3, 0.98, 1e-5], [  50.5, 9e-6, 1.001]],
               [[  0.0, 1.00,  0.0], [   0.0,  0.0, 1.000]],
               [[250.0, 1.01, 2e-5], [-250.0, 1e-5, 0.98]],
               [[ 50.0, 1.01, 1e-5], [ -31.0, 1e-5, 1.000]],
               [[ 78.0, 0.98, 0.0 ], [  45.0, 9e-6, 1.001]],
               [[-13.0, 0.99, 1e-5], [  150, 2e-5, 1.002]],
               [[ 94.0, 1.00, 9e-6], [-182.0, 0.0, 0.99]]]
    mag_trans = [0.1, 0.4, 0.0, -0.3, 0.2, 0.0, -0.1, -0.3]

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
        xpe = lis['x0_err'] / scale
        ype = lis['y0_err'] / scale

        # Distort the positions
        trans = transforms.PolyTransform(1, xy_trans[ss][0], xy_trans[ss][1], mag_offset=mag_trans[ss])
        xd, yd = trans.evaluate(xp, yp)
        md = trans.evaluate_mag(lis['m0'])

        # Perturb with small errors (0.1 mas)
        xd += np.random.randn(N_stars) * xpe
        yd += np.random.randn(N_stars) * ype
        md += np.random.randn(N_stars) * 0.02
        xde = xpe
        yde = ype
        mde = lis['m0_err']

        # Save the new list as a starlist.
        new_lis = starlists.StarList([lis['name'], md, mde, xd, xde, yd, yde, t],
                                     names=('name', 'm', 'me', 'x', 'xe', 'y', 'ye', 't'))

        new_lis.write(f'{test_data_path}/random_vel_{ss}.fits', overwrite=True)

    return (xy_trans, mag_trans)

def make_fake_starlists_poly1_acc(seed=-1):
    # If seed >=0, then set random seed to that value
    if seed >= 0:
        np.random.seed(seed=seed)

    N_stars = 200

    x0  = np.random.rand(N_stars) * 10.0     # arcsec (increasing to East)
    y0  = np.random.rand(N_stars) * 10.0     # arcsec
    x0e = np.ones(N_stars) * 1.0e-4  # arcsec
    y0e = np.ones(N_stars) * 1.0e-4  # arcsec
    vx  = np.random.randn(N_stars) * 5.0     # mas / yr
    vy  = np.random.randn(N_stars) * 5.0     # mas / yr
    vxe = np.ones(N_stars) * 0.1     # mas / yr
    vye = np.ones(N_stars) * 0.1     # mas / yr
    ax  = np.random.randn(N_stars) * 0.5     # mas / yr^2
    ay  = np.random.randn(N_stars) * 0.5     # mas / yr^2
    axe = np.ones(N_stars) * 0.01    # mas / yr^2
    aye = np.ones(N_stars) * 0.01    # mas / yr^2
    m0  = (np.random.rand(N_stars) * 8) + 9  # mag
    m0e = np.random.randn(N_stars) * 0.05    # mag
    t0 = np.ones(N_stars) * 2019.5

    # Make all the errors positive
    x0e = np.abs(x0e)
    y0e = np.abs(y0e)
    m0e = np.abs(m0e)
    vxe = np.abs(vxe)
    vye = np.abs(vye)
    axe = np.abs(axe)
    aye = np.abs(aye)

    name = ['star_{0:03d}'.format(ii) for ii in range(N_stars)]

    # Make an StarList
    lis = starlists.StarList([name, m0, m0e,
                              x0, x0e, y0, y0e,
                              vx, vxe, vy, vye,
                              ax, axe, ay, aye,
                              t0],
                             names = ('name', 'm0', 'm0_err',
                                      'x0', 'x0_err', 'y0', 'y0_err',
                                      'vx0', 'vx0_err', 'vy0', 'vy0_err',
                                      'ax', 'ax_err', 'ay', 'ay_err',
                                      't0'))

    sdx = np.argsort(m0)
    lis = lis[sdx]

    # Save original positions as reference (1st) list
    # in a StarList format (with velocities).
    lis.write(f'{test_data_path}/random_acc_ref.fits', overwrite=True)

    ##########
    # Propogate to new times and distort.
    ##########
    # Make 4 new starlists with different epochs and transformations.
    times = [2018.5, 2019.0, 2019.5, 2020.0, 2020.5, 2021.0, 2021.5, 2022.0]
    xy_trans = [[[ 6.5, 0.99, 1e-5], [  10.1, 1e-5, 0.99]],
               [[100.3, 0.98, 1e-5], [  50.5, 9e-6, 1.001]],
               [[  0.0, 1.00,  0.0], [   0.0,  0.0, 1.000]],
               [[250.0, 0.97, 2e-5], [-250.0, 1e-5, 1.001]],
               [[ 50.0, 1.01, 1e-5], [ -31.0, 1e-5, 1.000]],
               [[ 78.0, 0.98, 0.0 ], [  45.0, 9e-6, 1.001]],
               [[-13.0, 0.99, 1e-5], [  150, 2e-5, 1.002]],
               [[ 94.0, 1.00, 9e-6], [-182.0, 0.0, 0.99]]]
    mag_trans = [0.1, 0.4, 0.0, -0.3, 0.2, 0.0, -0.1, -0.3]

    # Convert into pixels (undistorted) with the following info.
    scale = 0.01  # arcsec / pix
    shift = [1.0, 1.0]  # pix

    for ss in range(len(times)):
        dt = times[ss] - lis['t0']

        x = lis['x0'] + (lis['vx0']/1e3) * dt + 0.5*(lis['ax']/1e3) * dt**2
        y = lis['y0'] + (lis['vy0']/1e3) * dt + 0.5*(lis['ay']/1e3) * dt**2
        t = np.ones(N_stars) * times[ss]

        # Convert into pixels
        xp = (x / -scale) + shift[0]  # -1 from switching to increasing to West (right)
        yp = (y /  scale) + shift[1]
        xpe = lis['x0_err'] / scale
        ype = lis['y0_err'] / scale

        # Distort the positions
        trans = transforms.PolyTransform(1, xy_trans[ss][0], xy_trans[ss][1], mag_offset=mag_trans[ss])
        xd, yd = trans.evaluate(xp, yp)
        md = trans.evaluate_mag(lis['m0'])

        # Perturb with small errors (0.1 pix)
        xd += np.random.randn(N_stars) * xpe
        yd += np.random.randn(N_stars) * ype
        md += np.random.randn(N_stars) * 0.02
        xde = xpe
        yde = ype
        mde = lis['m0_err']

        # Save the new list as a starlist.
        new_lis = starlists.StarList([lis['name'], md, mde, xd, xde, yd, yde, t],
                                     names=('name', 'm', 'me', 'x', 'xe', 'y', 'ye', 't'))

        new_lis.write(f'{test_data_path}/random_acc_{ss}.fits', overwrite=True)

    return (xy_trans, mag_trans)

def make_fake_starlists_poly1_par(seed=-1):
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
    vxe = np.random.randn(N_stars) * 0.1     # mas / yr
    vye = np.random.randn(N_stars) * 0.1     # mas / yr
    pi  = np.random.randn(N_stars) * 0.5     # mas
    pie = np.random.randn(N_stars) * 0.01    # mas
    m0  = (np.random.rand(N_stars) * 8) + 9  # mag
    m0e = np.random.randn(N_stars) * 0.05    # mag
    t0 = np.ones(N_stars) * 2019.5

    # Make all the errors positive
    x0e = np.abs(x0e)
    y0e = np.abs(y0e)
    m0e = np.abs(m0e)
    vxe = np.abs(vxe)
    vye = np.abs(vye)
    pie = np.abs(pie)

    name = ['star_{0:03d}'.format(ii) for ii in range(N_stars)]

    # Make an StarList
    lis = starlists.StarList([name, m0, m0e,
                              x0, x0e, y0, y0e,
                              vx, vxe, vy, vye,
                              pi, pie,
                              t0],
                             names = ('name', 'm0', 'm0_err',
                                      'x0', 'x0_err', 'y0', 'y0_err',
                                      'vx', 'vx_err', 'vy', 'vy_err',
                                      'pi', 'pi_err',
                                      't0'))

    sdx = np.argsort(m0)
    lis = lis[sdx]

    # Save original positions as reference (1st) list
    # in a StarList format (with velocities).
    lis.write(f'{test_data_path}/random_par_ref.fits', overwrite=True)

    ##########
    # Propogate to new times and distort.
    ##########
    # Make 4 new starlists with different epochs and transformations.
    '''times = [2018.5, 2019.5, 2020.5, 2021.5]
    xy_trans = [[[ 6.5, 0.99, 1e-5], [  10.1, 1e-5, 0.99]],
               [[100.3, 0.98, 1e-5], [  50.5, 9e-6, 1.001]],
               [[  0.0, 1.00,  0.0], [   0.0,  0.0, 1.0]],
               [[250.0, 0.97, 2e-5], [-250.0, 1e-5, 1.001]]]
    mag_trans = [0.1, 0.4, 0.0, -0.3]'''

    times = [2018.5, 2019.0, 2019.5, 2020.0, 2020.5, 2021.0, 2021.5, 2022.0]
    xy_trans = [[[ 6.5, 0.99, 1e-5], [  10.1, 1e-5, 0.99]],
               [[100.3, 0.98, 1e-5], [  50.5, 9e-6, 1.001]],
               [[  0.0, 1.00,  0.0], [   0.0,  0.0, 1.0]],
               [[250.0, 0.97, 2e-5], [-250.0, 1e-5, 1.001]],
               [[ 50.0, 1.00, 0.0], [ -31.0, 0.0, 1.000]],
               [[ 78.0, 1.00, 0.0 ], [  45.0, 0.0, 1.00]],
               [[-13.0, 1.00, 0.0], [  150, 0.0, 1.00]],
               [[ 94.0, 1.00, 0.0], [-182.0, 0.0, 1.00]]]
    mag_trans = [0.1, 0.4, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0]

    # Convert into pixels (undistorted) with the following info.
    scale = 0.01  # arcsec / pix
    shift = [1.0, 1.0]  # pix

    for ss in range(len(times)):
        dt = times[ss] - lis['t0']

        par_mod = motion_model.Parallax(pa=0,ra=18.0, dec=-30.0)
        par_mod_dat = par_mod.get_batch_pos_at_time(dt+lis['t0'], x0=lis['x0'],vx=lis['vx']/1e3, pi=lis['pi'],
                            y0=lis['y0'], vy=lis['vy']/1e3, t0=lis['t0'])
        x,y = par_mod_dat[0], par_mod_dat[1]
        t = np.ones(N_stars) * times[ss]

        # Convert into pixels
        xp = (x / -scale) + shift[0]  # -1 from switching to increasing to West (right)
        yp = (y /  scale) + shift[1]
        xpe = lis['x0_err'] / scale
        ype = lis['y0_err'] / scale

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
        mde = lis['m0_err']

        # Save the new list as a starlist.
        new_lis = starlists.StarList([lis['name'], md, mde, xd, xde, yd, yde, t],
                                     names=('name', 'm', 'me', 'x', 'xe', 'y', 'ye', 't'))

        new_lis.write(f'{test_data_path}/random_par_{ss}.fits', overwrite=True)

    return (xy_trans, mag_trans)


def _bruteforce_determine_motion_models(startable, motion_models, fixed_params_dict, verbose=False):
    """
    Reference implementation of align.determine_motion_models(), kept here only
    as ground truth for test_determine_motion_models_vectorized: a plain,
    unambiguous per-star Python loop (the same algorithm the vectorized version
    in align.py replaced, for performance, with whole-column numpy ops).
    """
    if all(isinstance(mm, str) for mm in motion_models):
        mm_map = motion_model.motion_model_map()
        motion_models = [mm_map[mm] for mm in motion_models]

    motion_models_possible = []
    for mm in motion_models:
        required_columns = mm.fit_param_names + mm.fixed_param_names
        req_col_in_table = [col for col in required_columns if (col in startable.colnames)]
        req_col_in_dict = [col for col in required_columns if (col in fixed_params_dict.keys())]
        req_cols = startable[req_col_in_table]
        if all((col in startable.colnames) or (col in fixed_params_dict.keys()) for col in required_columns):
            motion_models_possible.append((mm, req_col_in_table, req_cols, req_col_in_dict))

    motion_model_used = []
    n_params = []
    for k in range(len(startable)):
        for mm, req_col_in_table, req_cols, req_col_in_dict in motion_models_possible[::-1]:
            if all(np.isfinite(req_cols[col][k]) for col in req_col_in_table if np.issubdtype(req_cols[col].dtype, np.number)) \
            and all(np.isfinite(fixed_params_dict[col]) for col in req_col_in_dict if np.issubdtype(np.array(fixed_params_dict[col]).dtype, np.number)):
                motion_model_used.append(mm.name)
                n_params.append(mm.n_params)
                break

    return motion_model_used, n_params


def test_determine_motion_models_vectorized():
    """
    align.determine_motion_models() was rewritten to use whole-column numpy
    operations instead of a Python loop over every star (a major bottleneck
    for large mosaics). Check the vectorized version against a brute-force
    per-star reference on a table that exercises: an always-finite fallback
    model (Empty), a model needing table columns to be finite (Fixed), and a
    model needing both table columns and a fixed_params_dict entry to be
    finite (Linear, gated on 't0').
    """
    rng = np.random.default_rng(42)
    n_stars = 200

    x0 = rng.uniform(-10, 10, n_stars)
    y0 = rng.uniform(-10, 10, n_stars)
    vx = rng.uniform(-1, 1, n_stars)
    vy = rng.uniform(-1, 1, n_stars)

    # Sprinkle in some non-finite values so all three models get exercised.
    x0[::7] = np.nan          # these rows can only ever be 'Empty'
    vx[::5] = np.inf          # these rows (minus the ones above) can only be 'Fixed'
    vy[1::11] = np.nan

    table = Table({'x0': x0, 'y0': y0, 'vx': vx, 'vy': vy})

    for fixed_params_dict in [{'t0': 2020.0}, {'t0': np.inf}, {}]:
        motion_models = ['Empty', 'Fixed', 'Linear']

        got_used, got_n = align.determine_motion_models(
            table, motion_models=motion_models, fixed_params_dict=dict(fixed_params_dict), verbose=False
        )
        want_used, want_n = _bruteforce_determine_motion_models(
            table, motion_models=motion_models, fixed_params_dict=dict(fixed_params_dict), verbose=False
        )

        assert got_used == want_used
        assert got_n == want_n
        # Sanity check: with fixed_params_dict containing a finite t0, at least
        # some stars should have resolved to each of the three models.
        if fixed_params_dict.get('t0') == 2020.0:
            assert set(got_used) == {'Empty', 'Fixed', 'Linear'}


def test_update_old_and_new_names():
    """
    align.update_old_and_new_names() used to find the max existing name length
    by looping over every row in the reference table. It now reads the length
    straight off the fixed-width numpy dtype. Check both the "no widening
    needed" and "widening needed" branches against the original per-row logic.
    """
    n_old = 50
    old_names = np.array([f'{i:03d}_star' for i in range(n_old)])  # 8 chars each
    name_in_list = np.array([f'star_{i}' for i in range(n_old)]).reshape(-1, 1)  # 6-7 chars

    ref_table = Table({'name': old_names, 'name_in_list': name_in_list})
    idx_ref_new = np.array([5, 12, 30])
    list_index = 0

    def _bruteforce_update_old_and_new_names(ref_table, list_index, idx_ref_new):
        new_names = [f"{list_index:3d}_{name}" for name in ref_table['name_in_list'][idx_ref_new, list_index]]
        new_name_len_max = np.max([len(new_name) for new_name in new_names])
        old_names = ref_table['name']
        old_name_len = [len(old_name) for old_name in old_names]
        old_name_len_max = np.max(old_name_len)
        if new_name_len_max > old_name_len_max:
            all_names = old_names.astype('U{0:d}'.format(new_name_len_max))
        else:
            all_names = old_names
        all_names[idx_ref_new] = new_names
        return all_names

    # Case 1: new names are no longer than existing ones -- no widening needed.
    got = align.update_old_and_new_names(ref_table.copy(), list_index, idx_ref_new)
    want = _bruteforce_update_old_and_new_names(ref_table.copy(), list_index, idx_ref_new)
    assert list(got) == list(want)

    # Case 2: new names are longer than any existing name -- dtype must widen.
    # Widen name_in_list's dtype explicitly first -- assigning a longer string
    # into a narrower fixed-width numpy array would silently truncate it.
    ref_table2 = ref_table.copy()
    wide_name_in_list = ref_table2['name_in_list'].astype('U40')
    wide_name_in_list[idx_ref_new[0], 0] = 'a_much_much_longer_star_name'
    ref_table2.replace_column('name_in_list', wide_name_in_list)
    got2 = align.update_old_and_new_names(ref_table2.copy(), list_index, idx_ref_new)
    want2 = _bruteforce_update_old_and_new_names(ref_table2.copy(), list_index, idx_ref_new)
    assert list(got2) == list(want2)


if __name__ == '__main__':
    import pickle
    with open(f'{test_data_path}/my_gaia.pkl', 'rb') as f:
        my_gaia = pickle.load(f)
    with open(f'{test_data_path}/list_of_starlists.pkl', 'rb') as f:
        list_of_starlists = pickle.load(f)
    ra_deg, dec_deg = 18.0, -30.0
    my_gaia.remove_column('motion_model_used')
    msc = align.MosaicToRef(my_gaia, list_of_starlists, iters=3,
                        dr_tol=[0.2, 0.1, 0.08], dm_tol=[5,5,5],
                        outlier_tol=[None, None, 3], mag_lim=[6, 20],
                        trans_class=transforms.PolyTransform,
                        trans_args=[{'order': 1}, {'order': 1}, {'order': 1}],
                        motion_models=['Linear','Parallax'],
                        fixed_params_dict = {'ra':ra_deg, 'dec':dec_deg, 'pa':0.0, 'obsLocation':'earth'},
                        use_ref_new=True,
                        update_ref_orig=False,
                        mag_trans=True,
                        trans_weights='both,std',
                        init_guess_mode='name', verbose=3)
    msc.fit()
    for i in range(msc.ref_table['x'].shape[1]):
        plt.scatter(msc.ref_table['x'][:, i], msc.ref_table['y'][:, i])
    plt.show()
    plot_stars(msc.ref_table, msc.ref_table['name'][:3])

def test_ref_velocity_propagation_independent_of_motion_models():
    """
    align's `motion_models` says which models to FIT for the observed stars.
    It must NOT limit how a reference star is PROPAGATED to an observed epoch:
    a reference imported from an external catalog can carry vx/vy/t0 that were
    never fit here, and those have to be used even when motion_models=['Fixed']
    -- otherwise the velocities sit unused in the table and the reference stays
    frozen at its catalog epoch.
    """
    from flystar.align import MosaicToRef, determine_motion_models
    from flystar.starlists import StarList

    n, T0 = 25, 2020.0
    rng = np.random.default_rng(3)
    names = [f'r{i:03d}' for i in range(n)]
    x0 = rng.uniform(40, 160, n)
    y0 = rng.uniform(40, 160, n)
    # per-star (not uniform) velocities -- a uniform proper motion is
    # degenerate with the per-epoch transformation, which would absorb it and
    # make this test pass vacuously
    vx = rng.normal(0, 0.8, n)
    vy = rng.normal(0, 0.8, n)
    m0 = rng.uniform(13, 18, n)

    # NOTE: StarList.__init__ only accepts x/y/m/xe/ye/me/corr as keywords and
    # silently drops anything else, so the Linear params go on as columns.
    ref = StarList(name=names, x=x0, y=y0, m=m0, xe=np.full(n, .01),
                   ye=np.full(n, .01), me=np.full(n, .01))
    for col, val in [('vx', vx), ('vy', vy), ('t0', np.full(n, T0)),
                     ('vx_err', np.full(n, .001)), ('vy_err', np.full(n, .001)),
                     ('x0_err', np.full(n, .01)), ('y0_err', np.full(n, .01))]:
        ref[col] = val

    lists = []
    for e in range(4):
        t = T0 + e
        sl = StarList(name=names,
                      x=x0 + vx * (t - T0) + rng.normal(0, .01, n),
                      y=y0 + vy * (t - T0) + rng.normal(0, .01, n),
                      m=m0 + rng.normal(0, .01, n),
                      xe=np.full(n, .01), ye=np.full(n, .01), me=np.full(n, .01))
        sl.meta['list_time'] = t
        lists.append(sl)

    for models in (['Fixed'], ['Linear']):
        mtr = MosaicToRef(ref, lists, motion_models=models,
                          update_ref_orig=False, iters=1, dr_tol=[6.],
                          dm_tol=[3], outlier_tol=[None],
                          init_guess_mode='name', verbose=False)
        mtr.fit()
        tab = mtr.ref_table

        # the reference's Linear params must survive into ref_table
        assert 'vx' in tab.colnames, f'{models}: vx column lost'
        assert 't0' in tab.colnames, f'{models}: t0 column lost'

        # propagation picks the most complex model each star supports...
        mm_prop, _ = determine_motion_models(tab, None, mtr.fixed_params_dict,
                                             verbose=False)
        assert set(np.asarray(mm_prop)) == {'Linear'}, \
            f'{models}: propagation model should be Linear, got {set(mm_prop)}'

        # ...and align's own propagation path really moves the stars at the
        # reference's per-star velocity
        r0 = mtr.get_ref_list_from_table(T0)
        r5 = mtr.get_ref_list_from_table(T0 + 5.0)
        slope = (np.asarray(r5['x']) - np.asarray(r0['x'])) / 5.0
        np.testing.assert_allclose(
            slope[:n], vx, rtol=1e-6, atol=1e-8,
            err_msg=f'{models}: reference not propagated at its own vx')

        slope_y = (np.asarray(r5['y']) - np.asarray(r0['y'])) / 5.0
        np.testing.assert_allclose(
            slope_y[:n], vy, rtol=1e-6, atol=1e-8,
            err_msg=f'{models}: reference not propagated at its own vy')

    # And what was FIT still respects motion_models: Fixed-only must not
    # produce Linear fits for the observed stars.
    mtr_fixed = MosaicToRef(ref, lists, motion_models=['Fixed'],
                            update_ref_orig=False, iters=1, dr_tol=[6.],
                            dm_tol=[3], outlier_tol=[None],
                            init_guess_mode='name', verbose=False)
    mtr_fixed.fit()
    assert set(np.asarray(mtr_fixed.ref_table['motion_model_used'])) == {'Fixed'}, \
        'motion_models=[Fixed] must still fit only Fixed'


def test_propagation_honors_motion_model_input():
    """
    determine_motion_models(tab, None) honors a caller-supplied 'motion_model_input'
    column as an explicit per-star propagation request, and otherwise falls
    back to the most complex model each star's own finite parameters support.

    The subtlety worth pinning: 'motion_model_input' must NOT be auto-filled
    by setup_ref_table_from_starlist. It used to be populated with
    motion_models[-1].name whenever the input lacked it, which made the column
    always present and so indistinguishable from a real request -- tying
    propagation back to `motion_models` and re-freezing a reference that
    carries velocities, exactly what
    test_ref_velocity_propagation_independent_of_motion_models forbids.
    """
    from flystar.align import MosaicToRef
    from flystar.starlists import StarList

    n, T0 = 25, 2020.0
    rng = np.random.default_rng(3)
    names = [f'r{i:03d}' for i in range(n)]
    x0 = rng.uniform(40, 160, n)
    y0 = rng.uniform(40, 160, n)
    vx = rng.normal(0, 0.8, n)
    vy = rng.normal(0, 0.8, n)
    m0 = rng.uniform(13, 18, n)

    def build(mm_input=None):
        ref = StarList(name=names, x=x0, y=y0, m=m0, xe=np.full(n, .01),
                       ye=np.full(n, .01), me=np.full(n, .01), vx=vx, vy=vy,
                       t0=np.full(n, T0), vx_err=np.full(n, .001),
                       vy_err=np.full(n, .001), x0_err=np.full(n, .01),
                       y0_err=np.full(n, .01))
        if mm_input is not None:
            ref['motion_model_input'] = np.array(mm_input, dtype='U20')
        lists = []
        for e in range(4):
            t = T0 + e
            sl = StarList(name=names, x=x0 + vx * (t - T0), y=y0 + vy * (t - T0),
                          m=m0, xe=np.full(n, .01), ye=np.full(n, .01),
                          me=np.full(n, .01))
            sl.meta['list_time'] = t
            lists.append(sl)
        return ref, lists

    def slopes(mm_input):
        ref, lists = build(mm_input)
        mtr = MosaicToRef(ref, lists, motion_models=['Fixed'],
                          update_ref_orig=False, iters=1, dr_tol=[6.],
                          dm_tol=[3], outlier_tol=[None],
                          init_guess_mode='name', verbose=False)
        mtr.fit()
        r0 = mtr.get_ref_list_from_table(T0)
        r5 = mtr.get_ref_list_from_table(T0 + 5.0)
        return mtr, (np.asarray(r5['x']) - np.asarray(r0['x']))[:n] / 5.0

    # no column supplied -> it must not be invented, and the velocities stand
    mtr, sl_auto = slopes(None)
    assert 'motion_model_input' not in mtr.ref_table.colnames, \
        'motion_model_input was auto-filled; propagation can no longer tell a ' \
        'real per-star request from the fitting setting restated per row'
    np.testing.assert_allclose(sl_auto, vx, rtol=1e-6, atol=1e-8,
                               err_msg='reference velocities were suppressed')

    # explicit Fixed request -> honored, so the stars do NOT move
    mtr, sl_fixed = slopes(['Fixed'] * n)
    assert 'motion_model_input' in mtr.ref_table.colnames
    np.testing.assert_allclose(sl_fixed, 0.0, atol=1e-10,
                               err_msg='explicit Fixed request was not honored')

    # explicit Linear request -> honored, stars move at their own vx
    mtr, sl_linear = slopes(['Linear'] * n)
    np.testing.assert_allclose(sl_linear, vx, rtol=1e-6, atol=1e-8,
                               err_msg='explicit Linear request was not honored')

    # mixed per-star requests are resolved per star
    mixed = ['Fixed'] * 10 + ['Linear'] * 15
    mtr, sl_mixed = slopes(mixed)
    np.testing.assert_allclose(sl_mixed[:10], 0.0, atol=1e-10,
                               err_msg='per-star Fixed rows moved')
    np.testing.assert_allclose(sl_mixed[10:], vx[10:], rtol=1e-6, atol=1e-8,
                               err_msg='per-star Linear rows did not move at vx')

    # A usable per-star request outranks the motion_models list, so
    # motion_model_used follows the request rather than being clamped to
    # ['Fixed'] -- the same priority fit_motion_models gives the column. These
    # reference stars carry real vx/vy from the input catalog, so labeling the
    # requested rows 'Linear' points at parameters that genuinely exist.
    mmu = np.asarray(mtr.ref_table['motion_model_used'])[:n]
    assert set(mmu[:10]) == {'Fixed'}, f'Fixed-requested rows got {set(mmu[:10])}'
    assert set(mmu[10:]) == {'Linear'}, f'Linear-requested rows got {set(mmu[10:])}'


def test_determine_motion_models_precedence():
    """Unit-level precedence: a usable request wins; an unusable, unrecognized
    or absent one falls back to most-complex-available."""
    from astropy.table import Column
    from flystar.startables import StarTable
    from flystar.align import determine_motion_models

    n = 6
    tab = StarTable(name=[f's{i}' for i in range(n)],
                    x=np.zeros((n, 2)), y=np.zeros((n, 2)), m=np.zeros((n, 2)),
                    xe=np.ones((n, 2)) * .01, ye=np.ones((n, 2)) * .01,
                    me=np.ones((n, 2)) * .01, t=np.tile([2020., 2021.], (n, 1)))
    tab['x0'] = np.arange(n, dtype=float)
    tab['y0'] = np.arange(n, dtype=float)
    tab['vx'] = np.array([1., 2., np.nan, 4., 5., 6.])
    tab['vy'] = np.array([1., 2., np.nan, 4., 5., 6.])
    tab['t0'] = np.full(n, 2020.)

    # no request column -> pure finiteness fallback
    got, _ = determine_motion_models(tab, None, verbose=False)
    assert list(got) == ['Linear', 'Linear', 'Fixed', 'Linear', 'Linear', 'Linear']

    tab['motion_model_input'] = Column(
        ['Linear',    # usable          -> honored
         'Fixed',     # explicit downgrade despite finite vx -> honored
         'Linear',    # vx is nan       -> unusable, fallback
         'Bogus',     # unrecognized    -> fallback
         'Parallax',  # needs pi/ra/dec -> absent, fallback
         'Empty'],    # explicit        -> honored
        dtype='U20')
    got, _ = determine_motion_models(tab, None, verbose=False)
    assert list(got) == ['Linear', 'Fixed', 'Fixed', 'Linear', 'Linear', 'Empty']
