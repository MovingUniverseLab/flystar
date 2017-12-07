from flystar import match
import numpy as np
import pdb

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
