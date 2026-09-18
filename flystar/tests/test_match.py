from flystar import align, match, starlists, transforms
import numpy as np
import pdb
from astropy.table import Table

def test_match_duplicates():
    x1 = np.array([1618.4, 1619.5, 1346.8, 1533.6, 1541.7,
                    232.5, 2165.4, 2354.1, 1584.5, 1697.8,
                   1028.9])
    y1 = np.array([  39.9,   41.3,   97.3,  130.8,  351.9,
                    769.6,  938.5, 1013.5, 1679.6, 1893.1,
                   1916.8])
    
    m1 = np.array([-5.94, -5.98, -1.96, -2.09, -2.50,
                   -3.43, -2.23, -3.72, -5.77, -2.97,
                   -3.35])

    x2 = np.array([1619.2, 1347.1, 1542.0, 2165.7, 2354.6,
                    734.1,  820.0, 2092.4, 1029.3])
    y2 = np.array([  41.8,   98.5,  353.1,  940.0, 1015.0,
                   1763.1, 1783.9, 1806.8, 1918.0])
    m2 = np.array([-6.05, -2.00, -2.65, -2.30, -3.79,
                   -2.06, -2.10, -2.38, -3.30])

    #####
    # Test 1:
    # First two stars in x1 are "duplicates"; but the 2nd one
    # should match x2[0] because it is closest in distance and mag.
    #####
    idx1, idx2, dr, dm = match.match(x1, y1, m1, x2, y2, m2, dr_tol=5, dm_tol=None)

    # Correct indices:
    idx1_good = [1, 2, 4, 6, 7, 10]
    idx2_good = [0, 1, 2, 3, 4, 8]

    assert set(idx1) == set(idx1_good)
    assert set(idx2) == set(idx2_good)
    assert len(dr) == len(dm)
    assert len(idx1) == len(idx2)
    assert len(dr) == len(idx2)


    #####
    # Test 2:
    # Reverse of test1
    #####
    idx2, idx1, dr, dm = match.match(x2, y2, m2, x1, y1, m1, dr_tol=5, dm_tol=None)

    # Correct indices:
    idx1_good = [1, 2, 4, 6, 7, 10]
    idx2_good = [0, 1, 2, 3, 4, 8]

    assert set(idx1) == set(idx1_good)
    assert set(idx2) == set(idx2_good)
    assert len(dr) == len(dm)
    assert len(idx1) == len(idx2)
    assert len(dr) == len(idx2)


    #####
    # Test 3:
    # Test confused case.
    #####
    m2[0] = -5.9
    idx1, idx2, dr, dm = match.match(x1, y1, m1, x2, y2, m2, dr_tol=5, dm_tol=None)

    # Correct indices:
    idx1_good = [2, 4, 6, 7, 10]
    idx2_good = [1, 2, 3, 4, 8]

    assert set(idx1) == set(idx1_good)
    assert set(idx2) == set(idx2_good)
    assert len(dr) == len(dm)
    assert len(idx1) == len(idx2)
    assert len(dr) == len(idx2)

    #####
    # Test 4:
    # Reverse test 3.
    #####
    m2[0] = -5.9
    idx2, idx1, dr, dm = match.match(x2, y2, m2, x1, y1, m1, dr_tol=5, dm_tol=None)

    # Correct indices:
    idx1_good = [2, 4, 6, 7, 10]
    idx2_good = [1, 2, 3, 4, 8]

    assert set(idx1) == set(idx1_good)
    assert set(idx2) == set(idx2_good)
    assert len(dr) == len(dm)
    assert len(idx1) == len(idx2)
    assert len(dr) == len(idx2)
    

    return

    

def test_miracle_match_briteN():
    x1 = np.array([1619.5, 1346.8, 1533.6, 1541.7,
                    232.5, 2165.4, 2354.1, 1584.5, 1697.8,
                   1028.9])
    y1 = np.array([  41.3,   97.3,  130.8,  351.9,
                    769.6,  938.5, 1013.5, 1679.6, 1893.1,
                   1916.8])
    
    m1 = np.array([-5.98, -1.96, -2.09, -2.50,
                   -3.43, -2.23, -3.72, -5.77, -2.97,
                   -3.35])

    x2 = np.array([1619.2, 1347.1, 1542.0, 2165.7, 2354.6,
                    734.1,  820.0, 2092.4, 1029.3])
    y2 = np.array([  41.8,   98.5,  353.1,  940.0, 1015.0,
                   1763.1, 1783.9, 1806.8, 1918.0])
    m2 = np.array([-6.05, -2.00, -2.65, -2.30, -3.79,
                   -2.06, -2.10, -2.38, -3.30])

    #####
    # Test 1:
    # First two stars in x1 are "duplicates"; but the 2nd one
    # should match x2[0] because it is closest in distance and mag.
    #####
    Nbrite = 8
    N, x1m, y1m, m1m, x2m, y2m, m2m = match.miracle_match_briteN(x1, y1, m1, x2, y2, m2, Nbrite,
                                                                     Nbins_vmax=300,
                                                                     Nbins_angle = 460,
                                                                     verbose=True)
    # Correct indices:
    idx1_good = [0, 1, 5, 6, 9]
    idx2_good = [0, 1, 3, 4, 8]

    assert N == len(idx1_good)

    # Check that all the positional differences for the matches are within a couple of pixels.
    for ii in range(N):
        assert abs(x1m[ii] - x2m[ii]) < 2
        assert abs(y1m[ii] - y2m[ii]) < 2
    
    return


def test_generic_match():
    # copied and slightly modified from test_match_duplicates:
    x1 = np.array([1618.4, 1619.5, 1346.8, 1533.6, 1541.7,
                    232.5, 2165.4, 2354.1, 1584.5, 1697.8,
                   1028.9])
    y1 = np.array([  39.9,   41.3,   97.3,  130.8,  351.9,
                    769.6,  938.5, 1013.5, 1679.6, 1893.1,
                   1916.8])
    
    m1 = np.array([-5.94, -5.98, -1.96, -2.09, -2.50,
                   -3.43, -2.23, -3.72, -5.77, -2.97,
                   -3.35])
    n1 = np.array(['S00', 'S01', 'S02', 'S03', 'S04',
                   'S04', 'S06', 'S07', 'S08', 'S09',
                   'S10'])

    x2 = np.array([1619.2, 1347.1, 1542.0, 2165.7, 2354.6,
                    734.1,  820.0, 2092.4, 1029.3])
    y2 = np.array([  41.8,   98.5,  353.1,  940.0, 1015.0,
                   1763.1, 1783.9, 1806.8, 1918.0])
    m2 = np.array([-6.05, -2.00, -2.65, -2.30, -3.79,
                   -2.06, -2.10, -2.38, -3.30])
    n2 = np.array(['S11', 'S12', 'S13', 'S14', 'S15',
                   'S16', 'S17', 'S18', 'S19'])


    list1 = Table([n1, x1, y1, m1],
                  names=('name', 'x', 'y', 'm'))
    list2 = Table([n2, x2, y2, m2],
                  names=('name', 'x', 'y', 'm'))

    starlist1 = starlists.StarList.from_table(list1)
    starlist2 = starlists.StarList.from_table(list2)

    # These two lists are deliberately perturbed copies of each other: after the
    # blind-triangle initial align, the six real counterparts sit 1.3-3.2 pixels
    # from their partners, and the three sl2 stars with no counterpart sit
    # hundreds of pixels away. dr_tol has to straddle that gap -- the original
    # 1.0 matched nothing at all, so the refinement fit 3 free parameters to 0
    # stars and returned a NaN transformation.
    transf, st = align.generic_match(starlist1, starlist2, init_mode='triangle',
                              model=transforms.PolyTransform, order_dr=[[1, 4.0]],
                              dr_final=4.0,
                              xy_match=(None, None, None, None, None, None, None, None),
                              m_match=(None, None, None, None), sigma_match=None,
                              n_bright=8, verbose=True)

    assert np.isfinite(transf.px.parameters).all(), \
        f'NaN in the x transformation parameters: {transf.px.parameters}'
    assert np.isfinite(transf.py.parameters).all(), \
        f'NaN in the y transformation parameters: {transf.py.parameters}'

    # The six sl2 stars that have a counterpart in sl1, and only those.
    assert len(st) == 6, f'expected 6 matches, got {len(st)}'
    matched_sl1 = set(np.asarray(st['ep_name'][:, 0]))
    assert matched_sl1 == {'S01', 'S02', 'S04', 'S06', 'S07', 'S10'}, \
        f'matched the wrong sl1 stars: {sorted(matched_sl1)}'

    # Every matched pair lands inside the final search radius.
    resid = np.hypot(st['x'][:, 0] - st['x'][:, 1], st['y'][:, 0] - st['y'][:, 1])
    assert (resid < 4.0).all(), f'matched pairs beyond dr_final: {resid}'


def test_generic_match_no_matches_raises():
    """
    A refinement pass that matches nothing must say so, not return NaN.

    find_transform runs an underdetermined least-squares solve and hands back
    NaN coefficients without raising. Those NaNs used to travel one loop
    iteration further and surface as the thoroughly unhelpful

        ValueError: x1 does not contain any finite values!

    from inside match.match. The error should name the fit that failed instead.
    """
    import pytest

    n = 12
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 2000, n)
    y = rng.uniform(0, 2000, n)
    m = rng.uniform(-6, -2, n)
    names = [f'S{i:02d}' for i in range(n)]

    sl1 = starlists.StarList.from_table(
        Table([names, x, y, m], names=('name', 'x', 'y', 'm')))
    # The same stars under a strongly quadratic distortion. Name-matching gives
    # an initial guess from all 12 stars, but an order=1 refinement cannot
    # absorb the quadratic term, so its residuals are tens of pixels -- nothing
    # falls inside dr_tol=0.001 and the refit gets 0 stars.
    sl2 = starlists.StarList.from_table(
        Table([names,
               x + 50.0 + 1e-4 * (x - 1000.0)**2,
               y + 50.0 + 1e-4 * (y - 1000.0)**2,
               m], names=('name', 'x', 'y', 'm')))

    with pytest.raises(ValueError, match='non-finite parameters'):
        align.generic_match(sl1, sl2, init_mode='match_name',
                            model=transforms.PolyTransform,
                            order_dr=[[1, 0.001]], dr_final=0.001,
                            m_match=(None, None, None, None),
                            sigma_match=None, verbose=False)


def test_chi2_matching_decisive_vs_ambiguous():
    """
    matching='chi2' must keep a decisively closer candidate and still refuse a
    genuine positional tie.

    The legacy rule required a star's nearest candidate in position to also be
    its nearest in magnitude, with no regard for how lopsided the positional
    evidence was. Real case from a NIRCam pointing: a counterpart 3.6 mas away
    was thrown out because an unrelated star 82.9 mas away -- 23x farther --
    happened to be 0.06 mag closer in brightness.
    """
    import pytest

    # One star, two candidates. Distances in arcsec.
    x1, y1, m1 = np.array([0.0]), np.array([0.0]), np.array([20.23])

    decisive_x, decisive_m = np.array([0.0036, 0.0829]), np.array([19.97, 20.43])
    tie_x, tie_m = np.array([0.011, 0.013]), np.array([20.65, 20.28])

    def run(x2, m2, mode):
        return match.match(x1, y1, m1, x2, np.zeros(2), m2,
                           dr_tol=0.1, dm_tol=0.5, matching=mode, verbose=0)

    # Decisive: legacy drops it, chi2 takes the near one.
    assert len(run(decisive_x, decisive_m, 'legacy')[0]) == 0, \
        'legacy behaviour changed: the 3.6 mas match is no longer dropped'
    i1, i2, dr, dm = run(decisive_x, decisive_m, 'chi2')
    assert len(i1) == 1 and i2[0] == 0, \
        f'chi2 did not match the decisively closer candidate: {i2}'
    np.testing.assert_allclose(dr[0], 0.0036, rtol=1e-6)

    # Genuine tie, 11 vs 13 mas: neither mode should guess.
    assert len(run(tie_x, tie_m, 'legacy')[0]) == 0
    assert len(run(tie_x, tie_m, 'chi2')[0]) == 0, \
        'chi2 matched a pair separated by less than the scatter from its rival'

    with pytest.raises(ValueError, match="matching must be"):
        run(decisive_x, decisive_m, 'nonsense')


def test_chi2_matching_is_reciprocal():
    """
    Two stars competing for one reference star: only the pair that both sides
    prefer survives, and only if it wins decisively.

    Legacy enforced one-to-one by arbitrating duplicates after the fact with
    the same both-must-agree test. Reciprocity is symmetric by construction --
    the result cannot depend on which catalog is passed first.
    """
    # Catalog 1: a star right on top of ref 0, and an interloper further away.
    x1 = np.array([0.000, 0.030])
    y1 = np.array([0.0, 0.0])
    m1 = np.array([18.00, 18.10])
    # Catalog 2: a single reference star.
    x2, y2, m2 = np.array([0.0]), np.array([0.0]), np.array([18.02])

    i1, i2, dr, dm = match.match(x1, y1, m1, x2, y2, m2, dr_tol=0.05, dm_tol=0.5,
                                 matching='chi2', verbose=0)
    assert len(i1) == 1 and i1[0] == 0, \
        f'the contested reference star went to the wrong suitor: {i1}'

    # Symmetry: swapping the catalogs must pair the same two stars.
    j1, j2, _, _ = match.match(x2, y2, m2, x1, y1, m1, dr_tol=0.05, dm_tol=0.5,
                               matching='chi2', verbose=0)
    assert len(j1) == 1 and j2[0] == 0, \
        f'match is not symmetric under catalog order: {j2}'

    # Equidistant suitors of equal brightness are unresolvable, so neither wins.
    x1_sym = np.array([-0.010, 0.010])
    m1_sym = np.array([18.02, 18.02])
    k1, _, _, _ = match.match(x1_sym, y1, m1_sym, x2, y2, m2, dr_tol=0.05,
                              dm_tol=0.5, matching='chi2', verbose=0)
    assert len(k1) == 0, 'chi2 picked a winner between two identical suitors'


def test_chi2_scale_calibration_without_errors():
    """
    The chi^2 scales are measured from the starlists, with no error columns.

    Tier 1 uses unambiguous pairs; the estimate must track the injected
    scatter. A catalog matched against itself has zero scatter, which must not
    become a zero divisor -- the same class of degeneracy that made outlier
    rejection reject everything.
    """
    from flystar.match import calibrate_match_scales, robust_sigma

    rng = np.random.default_rng(5)
    n = 500
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    m = rng.uniform(14, 20, n)

    for truth in (0.002, 0.010):
        x2 = x + rng.normal(0, truth, n)
        y2 = y + rng.normal(0, truth, n)
        m2 = m + rng.normal(0, 0.03, n)
        i1, i2, _, _ = match.match(x2, y2, m2, x, y, m, dr_tol=0.05, dm_tol=0.5,
                                   matching='chi2', verbose=0)
        # Recover the scale the matcher would have measured for this pairing.
        pair_i = np.arange(len(i1))
        s_pos, s_mag = calibrate_match_scales(
            pair_i, x2[i1] - x[i2], y2[i1] - y[i2], m2[i1] - m[i2],
            len(x2), 0.05, 0.5, verbose=0)
        # sqrt(2) because both catalogs carry the injected scatter.
        expect = truth * np.sqrt(2)
        assert 0.6 * expect < s_pos < 1.4 * expect, \
            f'measured sigma_pos={s_pos:.5f}, expected about {expect:.5f}'
        assert i1.size > 0.9 * n, f'only {i1.size} of {n} matched at scatter {truth}'

    # Zero scatter: a list against itself. Must not divide by zero, and must
    # match every star to itself.
    i1, i2, _, _ = match.match(x, y, m, x, y, m, dr_tol=0.05, dm_tol=0.5,
                               matching='chi2', verbose=0)
    assert len(i1) > 0, 'self-match produced nothing -- zero scatter divided by zero'
    np.testing.assert_array_equal(i1, i2)

    assert np.isnan(robust_sigma([]))
    np.testing.assert_allclose(robust_sigma([1., 1., 1.]), 0.0)
