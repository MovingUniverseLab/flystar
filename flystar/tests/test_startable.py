import os
import pdb
import pytest
import flystar
import numpy as np
from astropy.table import Table
from astropy import table
from flystar import motion_model
from flystar.startables import StarTable
from flystar.starlists import StarList

test_data_path = f'{flystar.__path__[0]}/tests/test_data'

def test_StarTable_init1():
    """
    Test creation of new StarTable.
    """
    # User input
    cat_file = f'{test_data_path}/test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)

    N_stars = len(cat_tab)
    N_lists = cat_tab['x'].shape[1]
    print(N_stars, N_lists)
    print(cat_tab['x'].shape)

    # Make a fake 2D array of names per epoch. We will call them "id".
    # Note that all of these inputs will be numpy arrays.
    x_in = cat_tab['x'].data
    y_in = cat_tab['y'].data
    m_in = cat_tab['m'].data
    xe_in = cat_tab['xe'].data
    ye_in = cat_tab['ye'].data
    me_in = cat_tab['me'].data

    # Name is a unique name for each star and is a 1D array.
    name_in = cat_tab['name'].data
    starlist_times = np.array([2001.0, 2002.1, 2003.0, 2004., 2005., 2006., 2007., 2008.])
    starlist_names = np.array(['file1', 'file2', 'file3', 'file4', 'file5', 'file6', 'file7', 'file8'])

    # Generate the startable
    startable = StarTable(
        name=name_in, 
        x=x_in, y=y_in, m=m_in, 
        xe=xe_in, ye=ye_in, me=me_in,
        ref_list=1,
        list_times=starlist_times, 
        list_names=starlist_names
    )

    # Now put in some assertions to make sure all our startable columns
    # have the right dimensions.
    assert len(startable) == N_stars
    assert startable['x'].shape == (N_stars, N_lists)
    assert startable['y'].shape == (N_stars, N_lists)
    assert startable['m'].shape == (N_stars, N_lists)
    assert startable['xe'].shape == (N_stars, N_lists)
    assert startable['ye'].shape == (N_stars, N_lists)
    assert startable['me'].shape == (N_stars, N_lists)
    assert len(startable['name']) == N_stars
    assert startable.meta['list_times'][0] == starlist_times[0]
    assert type(startable) == StarTable

    return

def test_StarTable_init2():
    """
    Initialize a StarTable with a StarList... this should 
    work (and just add a few meta keywords) if everything is working correctly. 
    Also double check that we can add a second list to it using add_starlist and
    we can get_starlist() as well.
    """
    list_file1 = f'{test_data_path}/A.lis'
    list_file2 = f'{test_data_path}/B.lis'
    list1 = StarList.from_lis_file(list_file1)
    list2 = StarList.from_lis_file(list_file2)

    # Test initializer
    tab = StarTable(list1)

    assert len(tab) == len(list1)

    return
    
def test_combine_lists():
    """
    Test the startables.combine_lists() functionality.
    """
    t = make_star_table()
    tt = make_tiny_star_table()

    # Test 1: call on a non-existent column.
    with pytest.raises(KeyError):
        t.combine_lists('foo')

    # Test 2: average x an check the first entry manually. Unweighted.
    x_avg_0 = t['x'][0, :].mean()
    t.combine_lists('x', mask_val=-100000)
    assert t['x0'][0] == x_avg_0
    np.testing.assert_allclose(t['x0'][-1], 2108.855, rtol=1e-3)

    # Test 3: Trying calling the same thing a second time and make sure the
    # answers don't change and we didn't break anything.
    t.combine_lists('x', mask_val=-100000)
    assert t['x0'][0] == x_avg_0
    np.testing.assert_allclose(t['x0'][-1], 2108.855, rtol=1e-3)

    # Test 4: weighted average of x.
    x_wgt_0 = 1.0 / t['xe'][0, :]**2
    x_avg_0 = np.average(t['x'][0, :], weights=x_wgt_0)
    t.combine_lists('x', mask_val=-100000, weights_col='xe')
    # A weighted-mean reduction over a 2D array's axis=1 (as combine_lists
    # does internally) doesn't reproduce a 1D np.average() call bit-for-bit
    # -- that's a numpy summation-order quirk (also true of the original
    # numpy.ma-based implementation for a plain np.average, just not for
    # np.ma.average specifically), not a precision issue worth chasing.
    np.testing.assert_allclose(t['x0'][0], x_avg_0)

    x_wgt_last = 1.0 / t['xe'][-1, :]**2
    x_avg_last = np.average(t['x'][-1, [2,7]], weights=x_wgt_last[[2,7]])
    assert t['x0'][-1] == pytest.approx(x_avg_last)

    ##########
    # Test 5: make sure mask_list is working.
    ##########
    # Test 5ai: Non-masked, weighted_m=False
    tt.combine_lists_xym(weighted_xy=True, weighted_m=False, mask_lists=None)
    assert np.arange(1.8, 38, 4) == pytest.approx(tt['x0'].data)
    assert np.arange(1.8, 38, 4) == pytest.approx(tt['y0'].data)
    avg_m = -2.5 * np.log10((4 * 10**-0.4 + 1)/5)
    assert avg_m * np.ones(10) == pytest.approx(tt['m0'].data)

    # Test 5aii: Non-masked, weighted_m=True
    tt.combine_lists_xym(weighted_xy=True, weighted_m=True, mask_lists=None)
    assert np.arange(1.8, 38, 4) == pytest.approx(tt['x0'].data)
    assert np.arange(1.8, 38, 4) == pytest.approx(tt['y0'].data)
    avg_m_weight = 0.9391744564422395
    assert avg_m_weight * np.ones(10) == pytest.approx(tt['m0'].data)

    # Test 5bii: Masked, weighted_m=False
    tt.combine_lists_xym(weighted_xy=True, weighted_m=False, mask_lists=[1])
    assert np.arange(2.25, 48, 5) == pytest.approx(tt['x0'].data)
    assert np.arange(2.25, 48, 5) == pytest.approx(tt['y0'].data)
    assert np.ones(10) == pytest.approx(tt['m0'].data)

    # Test 5bii: Masked, weighted_m=True (should be identical to 5bi)
    tt.combine_lists_xym(weighted_xy=True, weighted_m=True, mask_lists=[1])
    assert np.arange(2.25, 48, 5) == pytest.approx(tt['x0'].data)
    assert np.arange(2.25, 48, 5) == pytest.approx(tt['y0'].data)
    assert np.ones(10) == pytest.approx(tt['m0'].data)

    # Test 5c: Things that should break the code.
    # with pytest.raises(RuntimeError):
    #     This would not break the code anymore
    #     t.combine_lists_xym(weighted_xy=True, weighted_m=True, mask_lists=np.arange(2))
    with pytest.raises(AssertionError):
        t.combine_lists_xym(weighted_xy=True, weighted_m=True, mask_lists=True)

    return

def test_combine_lists_select_stars():
    """
    combine_lists()/combine_lists_xym() gained a select_stars parameter so
    align.update_ref_table_aggregates() can recompute averages only for the
    rows that changed, instead of the whole (potentially huge, ever-growing)
    ref_table every time it's called. Check that:
      - computing with select_stars over a subset gives the same numbers as
        a full recompute, for that subset.
      - rows outside select_stars are left completely untouched, even if
        their underlying per-epoch data changed after the last full
        recompute.
    """
    t = make_star_table()

    # Seed x0/x0_err and m0/m0_err with a full computation first.
    t.combine_lists('x', weights_col='xe', mask_val=-100000)
    t.combine_lists('m', weights_col='me', mask_val=-100000, ismag=True)
    x0_before = t['x0'].copy()
    x0_err_before = t['x0_err'].copy()
    m0_before = t['m0'].copy()

    # Mutate the underlying per-epoch data for every star...
    rng = np.random.default_rng(0)
    t['x'] = t['x'] + rng.uniform(-5, 5, t['x'].shape)
    t['m'] = t['m'] + rng.uniform(-0.5, 0.5, t['m'].shape)

    # ...but only recompute a subset of rows.
    select = np.zeros(len(t), dtype=bool)
    select[[1, 3, 5, 7]] = True

    t.combine_lists('x', weights_col='xe', mask_val=-100000, select_stars=select)
    t.combine_lists('m', weights_col='me', mask_val=-100000, ismag=True, select_stars=select)

    # A fresh full recompute on the same (mutated) data is ground truth.
    t_full = make_star_table()
    t_full['x'] = t['x']
    t_full['m'] = t['m']
    t_full.combine_lists('x', weights_col='xe', mask_val=-100000)
    t_full.combine_lists('m', weights_col='me', mask_val=-100000, ismag=True)

    # Selected rows should match the fresh full recompute (allowing for
    # floating-point reduction-order noise between a sliced vs. full array).
    np.testing.assert_allclose(t['x0'][select], t_full['x0'][select], rtol=1e-12)
    np.testing.assert_allclose(t['x0_err'][select], t_full['x0_err'][select], rtol=1e-12)
    np.testing.assert_allclose(t['m0'][select], t_full['m0'][select], rtol=1e-12)

    # Unselected rows should be untouched -- still equal to the pre-mutation
    # values, not the (different) values the new data would produce.
    np.testing.assert_array_equal(t['x0'][~select], x0_before[~select])
    np.testing.assert_array_equal(t['x0_err'][~select], x0_err_before[~select])
    np.testing.assert_array_equal(t['m0'][~select], m0_before[~select])

    # combine_lists_xym should thread select_stars through consistently too.
    tt = make_tiny_star_table()
    tt.combine_lists_xym(weighted_xy=True, weighted_m=True)
    x0_before_tt = tt['x0'].copy()
    tt['x'] = tt['x'] + 100.0  # move every star
    select_tt = np.array([True, False] * 5)
    tt.combine_lists_xym(weighted_xy=True, weighted_m=True, select_stars=select_tt)
    assert not np.allclose(tt['x0'][select_tt], x0_before_tt[select_tt])   # these moved
    np.testing.assert_array_equal(tt['x0'][~select_tt], x0_before_tt[~select_tt])  # these didn't

    # Edge case: an all-False selection should be a safe no-op.
    t2 = make_star_table()
    t2.combine_lists('x', weights_col='xe', mask_val=-100000)
    x0_snapshot = t2['x0'].copy()
    none_selected = np.zeros(len(t2), dtype=bool)
    t2.combine_lists('x', weights_col='xe', mask_val=-100000, select_stars=none_selected)
    np.testing.assert_array_equal(t2['x0'], x0_snapshot)

    return


def _bruteforce_combine_lists(startable, col_name_in, weights_col=None, mask_val=None,
                               mask_lists=None, ismag=False, sigma=3):
    """
    Reference implementation of StarTable.combine_lists(), kept here only as
    ground truth for test_combine_lists_vectorized: the original numpy.ma
    -based implementation that the vectorized (plain-numpy) version replaced,
    for performance (numpy.ma carries heavy per-operation overhead compared
    to explicit boolean-mask arithmetic on plain arrays).
    """
    from astropy.stats import sigma_clip as _sigma_clip

    if mask_lists is not None:
        mask_lists = np.atleast_1d(mask_lists)
        list_indices = np.array([i for i in np.arange(startable[col_name_in].data.shape[1]) if i not in mask_lists])
    else:
        list_indices = np.arange(startable[col_name_in].data.shape[1])

    val_2d = np.ma.masked_invalid(startable[col_name_in].data[:, list_indices])

    if ismag:
        val_2d = 10**(-0.4 * val_2d)

    if mask_val:
        val_2d = np.ma.masked_values(val_2d, mask_val)

    if sigma:
        val_2d_clip = _sigma_clip(val_2d, sigma=sigma, maxiters=5, axis=1)
    else:
        val_2d_clip = val_2d

    if weights_col in startable.colnames:
        err_2d = np.ma.masked_invalid(startable[weights_col].data[:, list_indices])
        if ismag:
            err_2d = 0.4 * np.log(10) * val_2d * err_2d
        unified_mask = val_2d_clip.mask | err_2d.mask
        val_2d_clip.mask = unified_mask
        err_2d.mask = unified_mask
        wgt_2d = np.ma.masked_invalid(1. / err_2d**2)
        avg = np.ma.average(val_2d_clip, weights=wgt_2d, axis=1)
        std = np.ma.sqrt(1. / np.ma.sum(wgt_2d, axis=1))
    else:
        avg = np.ma.mean(val_2d_clip, axis=1)
        std = np.ma.std(val_2d_clip, axis=1)

    std = np.ma.masked_where(std == 0., std)

    if ismag:
        std = 2.5 / np.log(10) * std / avg
        avg = -2.5 * np.ma.log10(avg)

    avg = avg.filled(np.nan)
    std = std.filled(np.inf)
    return avg, std


def test_combine_lists_vectorized():
    """
    StarTable.combine_lists() was rewritten to use plain numpy arithmetic
    with explicit boolean masks instead of numpy.ma (which carries heavy
    per-operation overhead -- mask bookkeeping and generic dispatch on every
    arithmetic op -- and was a measurable chunk of align.py's runtime for
    large mosaics). Check the vectorized version against the original
    numpy.ma-based reference across a battery of randomized tables that
    exercise: weighted/unweighted, magnitude conversion, mask_lists,
    mask_val, sigma clipping, all-invalid rows, and rows with exactly one
    valid epoch.
    """
    rng = np.random.default_rng(7)

    for trial in range(20):
        n_stars = 60
        n_epochs = 5

        x = rng.normal(100, 5, size=(n_stars, n_epochs))
        xe = rng.uniform(0.001, 0.05, size=(n_stars, n_epochs))

        # Sprinkle in missing epochs (NaN), a sentinel mask value, and some
        # gross outliers for sigma clipping to catch.
        x[rng.random((n_stars, n_epochs)) < 0.25] = np.nan
        xe[np.isnan(x)] = np.nan
        sentinel_mask = rng.random((n_stars, n_epochs)) < 0.05
        x[sentinel_mask] = -100000
        outlier_mask = rng.random((n_stars, n_epochs)) < 0.05
        x[outlier_mask] += rng.choice([-1, 1], size=outlier_mask.sum()) * rng.uniform(50, 200, size=outlier_mask.sum())

        # A few rows with zero, or exactly one, valid epoch -- edge cases for
        # "no data" and "std of a single point."
        x[0, :] = np.nan
        xe[0, :] = np.nan
        x[1, 1:] = np.nan
        xe[1, 1:] = np.nan

        t_weighted = Table({'x': x.copy(), 'xe': xe.copy()})
        t_unweighted = Table({'x': x.copy()})

        for use_weights, ismag, mask_lists, sigma in [
            (True, False, None, 3),
            (False, False, None, 3),
            (True, True, None, 3),
            (True, False, [2], 3),
            (True, False, None, None),
        ]:
            t = t_weighted if use_weights else t_unweighted
            kwargs = dict(mask_val=-100000, mask_lists=mask_lists, ismag=ismag, sigma=sigma)
            if use_weights:
                kwargs['weights_col'] = 'xe'

            want_avg, want_std = _bruteforce_combine_lists(t, 'x', **kwargs)

            t_copy = Table({k: t[k].copy() for k in t.colnames})
            t_copy.__class__ = StarTable  # combine_lists is a StarTable method
            t_copy.combine_lists('x', **kwargs)
            got_avg = np.asarray(t_copy['x0'])
            got_std = np.asarray(t_copy['x0_err'])

            np.testing.assert_allclose(got_avg, want_avg, rtol=1e-10, atol=1e-10, equal_nan=True,
                                        err_msg=f"trial={trial} use_weights={use_weights} ismag={ismag} mask_lists={mask_lists} sigma={sigma}: avg mismatch")
            np.testing.assert_allclose(got_std, want_std, rtol=1e-10, atol=1e-10, equal_nan=True,
                                        err_msg=f"trial={trial} use_weights={use_weights} ismag={ismag} mask_lists={mask_lists} sigma={sigma}: std mismatch")


def test_combine_lists_weight_fallback():
    """
    Regression test for StarTable.combine_lists()'s handling of stars whose
    weighting column (e.g. 'xe'/'ye'/'me') is entirely invalid (inf) in
    every epoch. The contract:
      - if a star has at least one epoch with a real, finite weight, only
        those epoch(s) are used (epochs with an invalid weight are simply
        dropped, even if their raw value is finite) -- the reported error
        is a real, finite propagated uncertainty.
      - if a star has NO usable weight anywhere but does have at least one
        finite raw value, the mean falls back to an (unweighted-in-spirit)
        average of the valid value(s) -- but the reported error MUST be
        exactly np.inf, never a fabricated finite number, since the true
        uncertainty was never actually known. This is the exact bug that
        motivated this refactor: a fake weight=1 fallback elsewhere in this
        codebase once leaked a finite x0_err=1.0 into real output for
        months, undetected.
      - if a star has no valid raw value at all, the mean is nan and the
        error is inf (nothing to fall back to).
      - a column with no weights_col at all (the plain unweighted branch,
        untouched by this refactor) still handles inf/nan correctly.

    All expected numbers below were derived by hand (or, for the weighted
    cases, via the same textbook inverse-variance formula the production
    code implements: wgt = 1/err**2, avg = weighted mean, std =
    sqrt(1/sum(wgt))) and cross-checked against the implementation before
    being hardcoded here, so a future refactor that silently changes the
    fallback's arithmetic (and not just its inf/nan-ness) will also be caught.
    """
    nan, inf = np.nan, np.inf

    ##########
    # Non-magnitude column ('x'/'xe'), one star per case, sigma clipping
    # disabled so every number below is exact (not subject to outlier
    # rejection on tiny synthetic rows).
    ##########
    names = np.array(['case1_baseline', 'case2_partial', 'case3a_fallback_single',
                       'case3b_fallback_multi', 'case4_no_data', 'case5_composite'])

    # case1_baseline: every epoch has a valid value AND a valid weight --
    # sanity check that normal weighted averaging is unaffected.
    # case2_partial: epoch 1 has a finite raw value (999) but an inf weight
    # -- it must be excluded, leaving only epochs 0, 2, 3 to average, with a
    # real (finite) propagated error.
    # case3a_fallback_single: only epoch 0 has a finite value; every weight
    # is inf. Falls back to that single value; error must be exactly inf.
    # case3b_fallback_multi: epochs 0, 1 have finite values (5, 9); every
    # weight is inf. Falls back to the unweighted mean of the two valid
    # values (7.0); error must be exactly inf.
    # case4_no_data: no valid value anywhere and no usable weight -- nothing
    # to fall back to, so mean is nan and error is inf.
    # case5_composite: combines four different conditions in one row --
    # epoch 0 is an invalid (nan) value with a valid-looking weight, epoch 1
    # is a valid value with an inf (unusable) weight, epoch 2 is a valid
    # value with a real, usable weight, epoch 3 is an invalid (inf) value
    # with a valid-looking weight. Since epoch 2 gives this star a real,
    # non-zero weight sum, this is NOT a fallback star -- it should reduce
    # to the ordinary weighted case using only epoch 2.
    x = np.array([
        [10., 20., 30., 40.],
        [10., 999., 20., 30.],
        [7., nan, nan, nan],
        [5., 9., nan, nan],
        [nan, nan, nan, nan],
        [nan, 50., 60., inf],
    ])
    xe = np.array([
        [1., 2., 3., 4.],
        [1., inf, 2., 3.],
        [inf, inf, inf, inf],
        [inf, inf, inf, inf],
        [inf, inf, inf, inf],
        [0.5, inf, 1.0, nan],
    ])

    t = StarTable(name=names, x=x.copy(), y=x.copy(), m=np.ones_like(x),
                  xe=xe.copy(), ye=xe.copy())
    t.combine_lists('x', weights_col='xe', sigma=None)

    i1, i2, i3a, i3b, i4, i5 = range(6)

    # Case 1: baseline, all weights valid -- ordinary weighted average.
    wgt1 = 1. / xe[i1]**2
    avg1 = np.average(x[i1], weights=wgt1)
    std1 = np.sqrt(1. / wgt1.sum())
    np.testing.assert_allclose(t['x0'][i1], avg1, rtol=1e-12)
    np.testing.assert_allclose(t['x0_err'][i1], std1, rtol=1e-12)
    assert np.isfinite(t['x0_err'][i1])

    # Case 2: epoch 1 (value 999, weight inf) must be excluded -- average
    # matches using only the epochs with a real, finite weight (0, 2, 3),
    # and the error is finite (real weight info exists), not inf.
    idx2 = [0, 2, 3]
    wgt2 = 1. / xe[i2][idx2]**2
    avg2 = np.average(x[i2][idx2], weights=wgt2)
    std2 = np.sqrt(1. / wgt2.sum())
    np.testing.assert_allclose(t['x0'][i2], avg2, rtol=1e-12)
    np.testing.assert_allclose(t['x0_err'][i2], std2, rtol=1e-12)
    assert np.isfinite(t['x0_err'][i2])
    assert not np.isinf(t['x0_err'][i2])

    # Case 3a: single valid value, all weights inf -- fallback mean is just
    # that one value; error must be EXACTLY inf (not merely large).
    assert t['x0'][i3a] == 7.0
    assert t['x0_err'][i3a] == np.inf
    assert np.isinf(t['x0_err'][i3a])

    # Case 3b: two valid values (5, 9), all weights inf -- fallback mean is
    # their plain (unweighted) average, 7.0; error must be EXACTLY inf.
    # This is the core regression case for today's fix.
    assert t['x0'][i3b] == pytest.approx(7.0)
    assert t['x0_err'][i3b] == np.inf
    assert np.isinf(t['x0_err'][i3b])

    # Case 4: no valid value anywhere -- mean is nan, error is inf.
    assert np.isnan(t['x0'][i4])
    assert t['x0_err'][i4] == np.inf

    # Case 5: composite row -- only epoch 2 (value 60, weight 1.0) carries
    # real weight, so the star reduces to an ordinary weighted case using
    # only that epoch, exactly as if epochs 0, 1, 3 didn't exist.
    assert t['x0'][i5] == pytest.approx(60.0)
    assert t['x0_err'][i5] == pytest.approx(1.0)
    assert np.isfinite(t['x0_err'][i5])

    ##########
    # Magnitude column ('m'/'me', ismag=True) -- same fallback contract, but
    # exercised through the flux-space conversion pipeline.
    ##########
    m_names = np.array(['mag_baseline', 'mag_fallback_single', 'mag_fallback_multi'])
    m_vals = np.array([
        [10., 12., 14.],
        [15., nan, nan],
        [12.0, 14.0, nan],
    ])
    m_errs = np.array([
        [0.05, 0.1, 0.2],
        [inf, inf, inf],
        [inf, inf, inf],
    ])
    tm = StarTable(name=m_names, x=np.ones_like(m_vals), y=np.ones_like(m_vals),
                   m=m_vals.copy(), me=m_errs.copy())
    tm.combine_lists('m', weights_col='me', ismag=True, sigma=None)

    # mag_baseline: every epoch has a valid value and a valid error -- the
    # refactor must not have changed ordinary weighted-in-flux averaging.
    val_flux = 10**(-0.4 * m_vals[0])
    err_flux = 0.4 * np.log(10) * val_flux * m_errs[0]
    wgt = 1. / err_flux**2
    avg_flux = np.average(val_flux, weights=wgt)
    std_flux = np.sqrt(1. / wgt.sum())
    avg_mag = -2.5 * np.log10(avg_flux)
    std_mag = 2.5 / np.log(10) * std_flux / avg_flux
    np.testing.assert_allclose(tm['m0'][0], avg_mag, rtol=1e-10)
    np.testing.assert_allclose(tm['m0_err'][0], std_mag, rtol=1e-10)
    assert np.isfinite(tm['m0_err'][0])

    # mag_fallback_single: one valid magnitude (15.0), every weight inf --
    # fallback mean is that value; error must be EXACTLY inf.
    np.testing.assert_allclose(tm['m0'][1], 15.0, rtol=1e-10)
    assert tm['m0_err'][1] == np.inf

    # mag_fallback_multi: two DIFFERENT valid magnitudes (12.0, 14.0), every
    # weight inf. Averaging magnitudes is physically a flux-space average,
    # not a plain arithmetic mean of the mag values themselves -- so
    # independently reproduce that here (a plain, equally-weighted mean of
    # the *flux* values, since that's the space val_2d is already in when
    # ismag=True) and require flystar's result to match it, rather than
    # asserting some simpler (and wrong) unweighted-in-mag-space expectation.
    flux2 = 10**(-0.4 * m_vals[2, :2])
    avg_flux2 = flux2.mean()
    avg_mag2 = -2.5 * np.log10(avg_flux2)
    np.testing.assert_allclose(tm['m0'][2], avg_mag2, rtol=1e-10)
    assert tm['m0_err'][2] == np.inf

    ##########
    # No weights_col at all -- the plain unweighted branch, untouched by
    # this refactor, but still deserving direct inf/nan regression coverage.
    ##########
    names_uw = np.array(['one_nan_two_valid', 'all_nan'])
    x_uw = np.array([
        [10., nan, 30.],
        [nan, nan, nan],
    ])
    t_uw = StarTable(name=names_uw, x=x_uw.copy(), y=x_uw.copy(), m=np.ones_like(x_uw))
    t_uw.combine_lists('x', sigma=None)

    # One nan among three epochs -- mean and std computed from the two
    # valid values only (10, 30): mean 20, population std of residuals 10.
    assert t_uw['x0'][0] == pytest.approx(20.0)
    assert t_uw['x0_err'][0] == pytest.approx(10.0)
    assert t_uw.meta['x0'] == 'not_weighted'

    # All epochs nan -- nothing to average; mean nan, error inf.
    assert np.isnan(t_uw['x0'][1])
    assert t_uw['x0_err'][1] == np.inf

    return


def test_add_starlist():
    """
    Test the startables.combine_lists() functionality.
    """
    t = make_star_table()
    t_orig = Table(t)

    # Make some new data for a new "list".
    x_new = t['x'][:, 0] + 0.1
    y_new = t['y'][:, 0] + 0.1
    m_new = t['m'][:, 0] + 0.1
    xe_new = t['xe'][:, 0] + 0.01
    ye_new = t['ye'][:, 0] + 0.01
    me_new = t['me'][:, 0] + 0.01
    t_new = 2008.0

    # Test 1: Add new list to the end with complete data: Keyword format
    t.add_starlist(x=x_new, y=y_new, m=m_new, xe=xe_new, ye=ye_new, me=me_new,
                   meta={'list_times': t_new})

    np.testing.assert_equal(len(t), len(t_orig))

    expected_shape = np.array(t_orig['x'].shape)
    expected_shape[1] += 1

    np.testing.assert_equal(len(t['x'].shape), len(expected_shape))
    np.testing.assert_equal(t['x'].shape[0], expected_shape[0])
    assert t['x'].shape[1] == expected_shape[1]

    np.testing.assert_equal(len(t['y'].shape), len(expected_shape))
    np.testing.assert_equal(t['y'].shape[0], expected_shape[0])
    assert t['y'].shape[1] == expected_shape[1]

    np.testing.assert_equal(len(t['m'].shape), len(expected_shape))
    np.testing.assert_equal(t['m'].shape[0], expected_shape[0])
    assert t['m'].shape[1] == expected_shape[1]

    np.testing.assert_equal(len(t['xe'].shape), len(expected_shape))
    np.testing.assert_equal(t['xe'].shape[0], expected_shape[0])
    np.testing.assert_equal(t['xe'].shape[1], expected_shape[1])

    np.testing.assert_equal(len(t['ye'].shape), len(expected_shape))
    np.testing.assert_equal(t['ye'].shape[0], expected_shape[0])
    np.testing.assert_equal(t['ye'].shape[1], expected_shape[1])
    np.testing.assert_equal(len(t['me'].shape), len(expected_shape))
    np.testing.assert_equal(t['me'].shape[0], expected_shape[0])
    np.testing.assert_equal(t['me'].shape[1], expected_shape[1])

    np.testing.assert_equal(len(t['name']), len(t_orig['name']))
    np.testing.assert_equal(len(t.meta['list_times']), expected_shape[1])
    np.testing.assert_equal(t.meta['n_lists'], 9)
    # Test 2: Add as starlist rather than with keywords.
    starlist = StarList(
        name=t_orig['name'], 
        x=x_new, y=y_new, m=m_new,
        xe=xe_new, ye=ye_new, me=me_new, 
        list_time=2001.0, list_name='A.lis'
    )
    
    t = make_star_table()
    t.add_starlist(starlist=starlist)

    np.testing.assert_equal(len(t), len(t_orig))

    expected_shape = np.array(t_orig['x'].shape)
    expected_shape[1] += 1

    np.testing.assert_equal(len(t['x'].shape), len(expected_shape))
    np.testing.assert_equal(t['x'].shape[0], expected_shape[0])
    np.testing.assert_equal(t['x'].shape[1], expected_shape[1])

    np.testing.assert_equal(len(t['y'].shape), len(expected_shape))
    np.testing.assert_equal(t['y'].shape[0], expected_shape[0])
    np.testing.assert_equal(t['y'].shape[1], expected_shape[1])

    np.testing.assert_equal(len(t['m'].shape), len(expected_shape))
    np.testing.assert_equal(t['m'].shape[0], expected_shape[0])
    np.testing.assert_equal(t['m'].shape[1], expected_shape[1])

    np.testing.assert_equal(len(t['xe'].shape), len(expected_shape))
    np.testing.assert_equal(t['xe'].shape[0], expected_shape[0])
    np.testing.assert_equal(t['xe'].shape[1], expected_shape[1])
    np.testing.assert_equal(len(t['ye'].shape), len(expected_shape))
    np.testing.assert_equal(t['ye'].shape[0], expected_shape[0])
    np.testing.assert_equal(t['ye'].shape[1], expected_shape[1])

    np.testing.assert_equal(len(t['me'].shape), len(expected_shape))
    np.testing.assert_equal(t['me'].shape[0], expected_shape[0])
    np.testing.assert_equal(t['me'].shape[1], expected_shape[1])

    np.testing.assert_equal(len(t['name']), len(t_orig['name']))
    np.testing.assert_equal(len(t.meta['list_times']), expected_shape[1])
    np.testing.assert_equal(t.meta['n_lists'], 9)

    return

def test_get_starlist():
    """
    Make a StarTable and have it return a StarList for just one 
    of the epochs.
    """
    t = make_star_table()

    t_list = t.get_starlist(2)

    assert t['x'][0,2] == t_list['x'][0]
    assert type(t_list) == StarList
    assert len(t_list['x'].shape) == 1

    return


def test_combine_1col():
    # User input
    cat_file = f'{test_data_path}/test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)
    
    # Make a fake 2D array of names per epoch. We will call them "id".
    # Note that all of these inputs will be numpy arrays.
    x_in = cat_tab['x'].data[:, [0]]
    y_in = cat_tab['y'].data[:, [0]]
    m_in = cat_tab['m'].data[:, [0]]
    xe_in = cat_tab['xe'].data[:, [0]]
    ye_in = cat_tab['ye'].data[:, [0]]
    me_in = cat_tab['me'].data[:, [0]]

    # Name is a unique name for each star and is a 1D array.
    name_in = cat_tab['name'].data
    starlist_times = np.array([2001.0])
    starlist_names = np.array(['file1'])

    # Generate the startable
    t = StarTable(name=name_in, x=x_in, y=y_in, m=m_in, xe=xe_in, ye=ye_in, me=me_in,
                  ref_list=0,
                  list_times=starlist_times, list_names=starlist_names)

    t.combine_lists('x', weights_col='xe')

    np.testing.assert_equal(t['x0'][0], t['x'][0])

    return

def test_fit_motion_models():
    tab = make_star_table()
    tt = make_tiny_star_table()

    # We don't need the entire table... lets just
    # pull a small subset for faster testing.
    tab1 = tab[0:100]
    tab2 = tab[10000:10100]
    tab3 = tab[-100:]
    tab = table.vstack((tab1, tab2, tab3))
    tab.meta = tab1.meta

    tab.fit_motion_models(verbose=True, mask_value=-100000.)

    # Test creation of new variables
    np.testing.assert_equal(len(tab['vx']), len(tab))
    np.testing.assert_equal(len(tab['vy']), len(tab))
    np.testing.assert_equal(len(tab['vx_err']), len(tab))
    np.testing.assert_equal(len(tab['vy_err']), len(tab))
    np.testing.assert_equal(len(tab['n_fit']), len(tab))
    np.testing.assert_equal(tab.meta['n_bootstrap'], 0)

    # Test no-fit for stars with N<2 epochs.
    n_epochs = (tab['x'] >= 0).sum(axis=1)
    idx = np.where(n_epochs < 2)[0]
    np.testing.assert_equal((tab['vx'][idx] == 0).all(), True)
    np.testing.assert_equal((tab['vx_err'][idx] == 0).all(), True)
    np.testing.assert_equal((tab['n_fit'][idx] == 2).all(), True)

    # Test that the velocity errors were calculated.
    np.testing.assert_equal((~(tab['vx_err'][0:100] < 0)).all(), True)
    np.testing.assert_equal((~(tab['x0_err'][0:100] < 0)).all(), True)
    np.testing.assert_equal((~(tab['vy_err'][0:100] < 0)).all(), True)
    np.testing.assert_equal((~(tab['y0_err'][0:100] < 0)).all(), True)

    ##########
    # Test running a second time. We should get the same results.
    ##########
    vx_orig = tab['vx']
    x0_orig = tab['x0']
    vxe_orig = tab['vx_err']
    x0e_orig = tab['x0_err']
    tab.fit_motion_models(verbose=False, mask_value=-100000.)

    np.testing.assert_allclose(tab['vx'], vx_orig)
    np.testing.assert_allclose(tab['x0'], x0_orig)
    np.testing.assert_allclose(tab['vx_err'], vxe_orig)
    np.testing.assert_allclose(tab['x0_err'], x0e_orig)

    ##########
    # Test fixed_t0 functionality
    ##########
    fixed_t0 = tab['t0'] + np.random.normal(size=len(tab))
    tab.fit_motion_models(verbose=False, mask_value=-100000., fixed_params_dict={'t0': fixed_t0})
    np.testing.assert_allclose(tab['t0'], fixed_t0)

    ##########
    # Test bootstrap
    ##########
    tab_b = table.vstack((tab1, tab2, tab3))
    tab_b.meta = tab1.meta
    tab_b.fit_motion_models(verbose=True, bootstrap=50)

    np.testing.assert_equal(tab_b.meta['n_bootstrap'], 50)
    np.testing.assert_array_less(tab['x0_err'][0], tab_b['x0_err'][0])
    np.testing.assert_array_less(tab['vx_err'][0], tab_b['vx_err'][0])
    np.testing.assert_array_less(tab['y0_err'][0], tab_b['y0_err'][0])
    np.testing.assert_array_less(tab['vy_err'][0], tab_b['vy_err'][0])

    ##########
    # Test what happens with no velocity errors
    ##########
    tab.remove_columns(['xe', 'ye', 'x0', 'y0', 'x0_err', 'y0_err', 'vx', 'vy', 'vx_err', 'vy_err', 'n_fit'])
    tab.fit_motion_models(verbose=False)

    np.testing.assert_equal(len(tab['vx']), len(tab))
    np.testing.assert_equal(len(tab['vy']), len(tab))
    np.testing.assert_equal(len(tab['vx_err']), len(tab))
    np.testing.assert_equal(len(tab['vy_err']), len(tab))
    np.testing.assert_equal(len(tab['n_fit']), len(tab))
    np.testing.assert_equal((~(tab['vx_err'][0:100] < 0)).all(), True)
    np.testing.assert_equal((~(tab['x0_err'][0:100] < 0)).all(), True)
    np.testing.assert_equal((~(tab['vy_err'][0:100] < 0)).all(), True)
    np.testing.assert_equal((~(tab['y0_err'][0:100] < 0)).all(), True)

    #########
    # Test mask_list
    #########
    # Test 5a: Masked
    print("Testing Masked List")
    tt.fit_motion_models(verbose=False, mask_lists=[1])
    np.testing.assert_allclose(np.arange(2.25, 48, 5), tt['x0'].data)
    np.testing.assert_allclose(np.arange(2.25, 48, 5), tt['y0'].data)
    np.testing.assert_allclose(np.full(10, 0.05), tt['x0_err'].data)
    np.testing.assert_allclose(np.full(10, 0.05), tt['y0_err'].data)
    np.testing.assert_allclose(np.ones(10), tt['vx'].data)
    np.testing.assert_allclose(np.ones(10), tt['vy'].data)
    np.testing.assert_allclose(np.full(10, 0.03380617), tt['vx_err'].data)
    np.testing.assert_allclose(np.full(10, 0.03380617), tt['vy_err'].data)
    np.testing.assert_allclose(2017.25 * np.ones(10), tt['t0'].data)

    return


def test_fit_motion_model_2epoch():
    ##########
    # Test: only 2 epoch2
    ##########
    tab = make_star_table_2epoch()

    # We don't need the entire table... lets just
    # pull a small subset for faster testing.
    tab1 = tab[0:100]
    tab2 = tab[10000:10100]
    tab3 = tab[-100:]
    tab_2 = table.vstack((tab1, tab2, tab3))
    tab_2.meta=tab1.meta

    tab_2.fit_motion_models(verbose=False, mask_value=-100000.)

    assert all([_ in tab_2.colnames for _ in ['n_fit', 't0', 'x0', 'y0', 'vx', 'vy', 'x0_err', 'y0_err', 'vx_err', 'vy_err']])

    # 2 detections
    print(tab1.meta)
    np.testing.assert_almost_equal(tab_2['x0'][0], tab_2['x'][0,0], 1)
    np.testing.assert_equal(tab_2['n_fit'][0], 2)

    # 1 detection
    np.testing.assert_equal(tab_2['x0'][100], tab_2['x'][100, 0])
    np.testing.assert_equal(tab_2['n_fit'][100], 1)

    # 0 detections
    np.testing.assert_equal(np.isnan(tab_2['x0'][-1]), True)
    np.testing.assert_equal(tab_2['n_fit'][-1], 0)

    return


def test_multiprocessing():
    rng = np.random.default_rng(42)
    N = 10000
    x = rng.random((N, 5))
    y = rng.random((N, 5))
    m = rng.random((N, 5))
    xe = rng.random((N, 5))
    ye = rng.random((N, 5))
    t = np.arange(5) + 2026
    fixed_params_dict = [None for _ in range(N)]
    weighting = 'var'
    fill_value = np.nan
    verbose = True

    st1 = StarTable(
        name=np.arange(N),
        x=x,
        y=y,
        m=m,
        xe=xe,
        ye=ye
    )
    st1.meta['list_times'] = t

    st2 = StarTable(
        name=np.arange(N),
        x=x,
        y=y,
        m=m,
        xe=xe,
        ye=ye
    )
    st2.meta['list_times'] = t

    st1.fit_motion_models(
        motion_models=['Linear'],
        weighting=weighting,
        use_scipy=True,
        absolute_sigma=True,
        bootstrap=0,
        fill_value=fill_value,
        verbose=verbose
    )

    st2.fit_motion_models(
        motion_models=['Linear'],
        weighting=weighting,
        use_scipy=True,
        absolute_sigma=True,
        bootstrap=0,
        fill_value=fill_value,
        processes=10,
        verbose=verbose
    )

    for key in ['x0', 'x0_err', 'y0', 'y0_err', 'vx', 'vx_err', 'vy', 'vy_err', 'chi2_x', 'chi2_y', 'n_params', 't0']:
        np.testing.assert_array_equal(st1[key], st2[key], err_msg=f"Mismatch in {key} between single and multi-processing runs.")
    return


def make_star_table():
    # User input
    cat_file = f'{test_data_path}/test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)
    
    # Make a fake 2D array of names per epoch. We will call them "id".
    # Note that all of these inputs will be numpy arrays.
    x_in = cat_tab['x'].data
    y_in = cat_tab['y'].data
    m_in = cat_tab['m'].data
    xe_in = cat_tab['xe'].data
    ye_in = cat_tab['ye'].data
    me_in = cat_tab['me'].data
    n_in = cat_tab['n'].data

    # Name is a unique name for each star and is a 1D array.
    name_in = cat_tab['name'].data
    starlist_times = np.array([2001.0, 2002.1, 2003.0, 2004., 2005., 2006., 2007., 2008.])
    starlist_names = np.array(['file1', 'file2', 'file3', 'file4', 'file5', 'file6', 'file7', 'file8'])

    # Generate the startable
    startable = StarTable(
        name=name_in, 
        x=x_in, y=y_in, m=m_in, 
        xe=xe_in, ye=ye_in, me=me_in, 
        n=n_in,
        ref_list=1
    )
    startable.meta['list_times'] = starlist_times
    startable.meta['list_names'] = starlist_names

    return startable

def make_star_table_1epoch():
    # User input
    cat_file = f'{test_data_path}/test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)
    
    # Make a fake 2D array of names per epoch. We will call them "id".
    # Note that all of these inputs will be numpy arrays.
    x_in = cat_tab['x'].data[:, [0]]
    y_in = cat_tab['y'].data[:, [0]]
    m_in = cat_tab['m'].data[:, [0]]
    xe_in = cat_tab['xe'].data[:, [0]]
    ye_in = cat_tab['ye'].data[:, [0]]
    me_in = cat_tab['me'].data[:, [0]]
    n_in = cat_tab['n'].data[:, [0]]

    # Name is a unique name for each star and is a 1D array.
    name_in = cat_tab['name'].data
    starlist_times = np.array([2001.0])
    starlist_names = np.array(['file1'])

    # Generate the startable
    startable = StarTable(name=name_in, x=x_in, y=y_in, m=m_in, xe=xe_in, ye=ye_in, me=me_in, n=n_in,
                              ref_list=0,
                              list_times=starlist_times, list_names=starlist_names)

    return startable

def make_star_table_2epoch():
    # User input
    cat_file = f'{test_data_path}/test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)
    
    # Make a fake 2D array of names per epoch. We will call them "id".
    # Note that all of these inputs will be numpy arrays.
    x_in = cat_tab['x'].data[:, 0:2]
    y_in = cat_tab['y'].data[:, 0:2]
    m_in = cat_tab['m'].data[:, 0:2]
    xe_in = cat_tab['xe'].data[:, 0:2]
    ye_in = cat_tab['ye'].data[:, 0:2]
    me_in = cat_tab['me'].data[:, 0:2]
    n_in = cat_tab['n'].data[:, 0:2]

    # Name is a unique name for each star and is a 1D array.
    name_in = cat_tab['name'].data
    starlist_times = np.array([2001.0, 2002.1])
    starlist_names = np.array(['file1', 'file2'])

    # Generate the startable
    startable = StarTable(name=name_in, x=x_in, y=y_in, m=m_in, xe=xe_in, ye=ye_in, me=me_in, n=n_in,
                              ref_list=0,
                              list_times=starlist_times, list_names=starlist_names)

    return startable


def make_tiny_star_table():
    """
    A small (10 stars, 5 epoch) startable for testing masks.
    """
    
    name_in = np.array(['N00', 'N01', 'N02', 'N03', 'N04',
                        'N05', 'N06', 'N07', 'N08', 'N09'])
    x_in = np.arange(50).reshape((10,5))
    y_in = np.arange(50).reshape((10,5))
    m_in = np.ones((10,5))
    t_in = np.arange(2015,2020) * np.ones((10,5))
    xe_in = 0.1 * np.ones((10,5))
    ye_in = 0.1 * np.ones((10,5))
    me_in = 0.1 * np.ones((10,5))

    # Modify one epoch to have different values.
    x_in[:,1] = 0
    y_in[:,1] = 0
    m_in[:,1] = 0
    
    # Generate the startable
    startable = StarTable(name=name_in, t=t_in,
                          x=x_in, y=y_in, m=m_in, 
                          xe=xe_in, ye=ye_in, me=me_in)

    return startable


if __name__ == "__main__":
    test_combine_lists()