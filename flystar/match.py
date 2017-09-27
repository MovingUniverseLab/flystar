
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree as KDT
import itertools
import pylab as py
import scipy.signal
from skimage import transform
from scipy.spatial import distance
import math
import pdb

def miracle_match_briteN(xin1, yin1, min1, xin2, yin2, min2, Nbrite,
                         Nbins_vmax=200, Nbins_angle=360,verbose=False):
    """
    Take two input starlists and select the <Nbrite> brightest stars from
    each. Then performa a triangle matching algorithm along the lines of
    Groth 1986.

    For every possible triangle (combination of 3 stars) in a starlist,
    compute the ratio of two sides and the angle between those sides.
    These quantities are invariant under scale and rotation transformations.
    Use a histogram of these quantities to vote for possible matching
    triangles between the two star lists. Then take the vote winners
    as actual matches.

    There may be some sensitivity to the bin sizes used in the histogram
    for vmax (ratio of two sides) and the angles. The larger the positional
    and brightness uncertainties, the more bigger the bin sizes should really
    be. But this isn't well tested.
    """
    print( '')
    print( '  miracle_match_briteN: mirmat50: use brightest 50')
    print( '  miracle_match_briteN:  ')
    print( '  miracle_match_briteN:  ')

    # Get/check the lengths of the two starlists
    nin1 = len(xin1)
    nin2 = len(xin2)

    if (nin1 < Nbrite) or (nin2 < Nbrite):
        print(( 'You need at least {0} to '.format(Nbrite)))
        print( 'find the matches...')
        print(( 'NIN1: ', nin1))
        print(( 'NIN2: ', nin2))
        return (0, None, None, None, None, None, None)

    # Take the Nbrite brightest stars from each list and order by brightness.
    print( '  miracle_match_briteN: ')
    print( '  miracle_match_briteN: ORD_BRITE: ')
    print( '  miracle_match_briteN: ')
    x1, y1, m1 = order_by_brite(xin1, yin1, min1, Nbrite, verbose=verbose)
    x2, y2, m2 = order_by_brite(xin2, yin2, min2, Nbrite, verbose=verbose)

    ####################
    #
    # Triangle Matching
    #
    ####################
    print( '  miracle_match_briteN: ')
    print( '  miracle_match_briteN: DO Matching Triangles search...')
    print( '  miracle_match_briteN: ')

    # These are the bins for the 2D (vmax, angle) array we will be making later.
    bins_vmax = np.arange(-1.0, 1.01, 2.0 / Nbins_vmax)
    bins_angle = np.arange(0, 360+1, 360.0 / Nbins_angle)


    ##########
    # List 1
    ##########
    # Make triangles for all combinations within the first starlist.
    stars_in_tri1, vmax1, angle1 = calc_triangles_vmax_angle(x1, y1)

    # Over 2D (vmax, angle) space, decide where everything goes.
    # We only care about the first instance in any bin.
    idx1_vmax_hist = np.digitize(vmax1, bins_vmax) - 1  # indices into the 2D arraya
    idx1_angl_hist = np.digitize(angle1, bins_angle) - 1

    # Make a 2D array and every position is a 3 element vector containing the indicies
    # for the points in this triangle. At each 2D position, we only have a single
    # triangle recorded. We want the first insance. So we just go in reverse order
    # of the triangles and the last updates are the first entries in the original
    # array.
    stars1_at_hist = np.ones((len(bins_vmax)-1, len(bins_angle)-1, 3), dtype=np.int16) * -1
    stars1_at_hist[idx1_vmax_hist[::-1], idx1_angl_hist[::-1], :] = stars_in_tri1[::-1]

    ##########
    # List 2
    ##########
    # Make triangles for all combinations within the second starlist.
    stars_in_tri2, vmax2, angle2 = calc_triangles_vmax_angle(x2, y2)

    # Over 2D (vmax, angle) space, decide where everything goes.
    # We only care about the first instance in any bin.
    idx2_vmax_hist = np.digitize(vmax2, bins_vmax) - 1  # indices into the 2D arraya
    idx2_angl_hist = np.digitize(angle2, bins_angle) - 1

    ##########
    # Possible Matches
    ##########
    # Find the triangles that have the same vmax and angle in list 1 and list 2.
    stars_in1_matches2 = stars1_at_hist[idx2_vmax_hist, idx2_angl_hist, :]

    ##########
    # Tally Votes
    ##########
    # Now vote for all stars in the triangles that have possible matches (same vmax, angle)
    # between the first and second lists.
    votes = np.zeros((Nbrite, Nbrite))
    
    matches = np.where(stars_in1_matches2[:,0] >= 0)[0]
    match_stars1 = stars_in1_matches2[matches,:]
    match_stars2 = stars_in_tri2[matches,:]
    # Ideally I would like to do:
    #votes[match_stars1[:,0], match_stars2[:,0]] += 1   # vote for 1st star in triangle
    #votes[match_stars1[:,1], match_stars2[:,1]] += 1   # vote for 2nd star in triangle
    #votes[match_stars1[:,2], match_stars2[:,2]] += 1   # vote for 3rd star in triangle
    # But python doesn't handle this properly... repeat occurences don't respond to +1

    add_votes(votes, match_stars1[:,0], match_stars2[:,0])
    add_votes(votes, match_stars1[:,1], match_stars2[:,1])
    add_votes(votes, match_stars1[:,2], match_stars2[:,2])
    
    ##########
    # Find matching triangles with most votes (and that pass threshold)
    ##########
    # Reverse sort along the columns. Each column is a star in list #2.
    # For each star in list #2, sort the votes over all the different stars in list #1.
    votes_sdx = votes.argsort(axis=0)[::-1]
    tmp = votes[votes_sdx, list(range(votes.shape[1]))]

    # For each star in list #2, figure out if the number of matches exceeds our threshold.
    # The threshold is that for each star in list #2, the highest voted list #1 stars has
    # votes that are 2 * higher than the second highest voted list #1 star.
    good = np.where(tmp[0, :] > (2 * tmp[1, :]))[0]  # good #2 stars

    ##########
    # Return the good matches
    ##########
    print( '  miracle_match_briteN: ')
    print(( '  miracle_match_briteN: found {0} matches '.format(len(good))))
    print( '  miracle_match_briteN: ')

    x2_mat = x2[good]
    y2_mat = y2[good]
    m2_mat = m2[good]
    x1_mat = x1[votes_sdx[0, good]]
    y1_mat = y1[votes_sdx[0, good]]
    m1_mat = m1[votes_sdx[0, good]]


    return len(x1_mat), x1_mat, y1_mat, m1_mat, x2_mat, y2_mat, m2_mat

def order_by_brite(xi, yi, mi, Nout, verbose=True):
    # Length of the input starlists.
    Nin = len(xi)
    if verbose:
        print(( 'order_by_brite: nstars in =', Nin))
        print(( 'order_by_brite: desired nstars out =', Nout))

    if Nout > Nin:
        Nout = Nin

    if verbose:
        print(( 'order_by_brite: return nstars out =', Nout))

    sdx = mi.argsort()
    brite = sdx[:Nout]

    if verbose:
        print(( 'order_by_brite: faintest star m =', mi[brite[-1]]))

    xo = xi[brite]
    yo = yi[brite]
    mo = mi[brite]

    return xo, yo, mo


def match(x1, y1, m1, x2, y2, m2, dr_tol, dm_tol=None):
    """
    Finds matches between two different catalogs. No transformations are done and it
    is assumed that the two catalogs are already on the same coordinate system
    and magnitude system.

    For two stars to be matched, they must be within a specified radius (dr_tol) and
    delta-magnitude (dm_tol). For stars with more than 1 neighbor (within the tolerances),
    if one is found that is the best match in both brightness and positional offsets
    (closest in both), then the match is made. Otherwise,
    their is a conflict and no match is returned for the star.
    
 
    Parameters
    x1 : array-like
        X coordinate in the first catalog
    y1 : array-like
        Y coordinate in the first catalog (shape of array must match `x1`)
    m1 : array-like
        Magnitude in the first catalog. Must have the same shape as x1.
    x2 : array-like
        X coordinate in the second catalog
    y2 : array-like
        Y coordinate in the second catalog (shape of array must match `x2`)
    m2 : array-like
        Magnitude in the second catalog. Must have the same shape as x2.
    dr_tol : float
        How close (in units of the first catalog) a match has to be to count as a match.
        For stars with more than one nearest neighbor, the delta-magnitude is checked
        and the closest in delta-mag is chosen.
    dm_tol : float or None, optional
        How close in delta-magnitude a match has to be to count as a match.
        If None, then any delta-magnitude is allowed.
 
    Returns
    -------
    idx1 : int array
        Indicies into the first catalog of the matches. Will never be
        larger than `x1`/`y1`.
    idx2 : int array
        Indicies into the second catalog of the matches. Will never be
        larger than `x1`/`y1`.
    dr : float array
        Distance between the matches.
    dm : float array
        Delta-mag between the matches. (m1 - m2)
 
    """
 
    x1 = np.array(x1, copy=False)
    y1 = np.array(y1, copy=False)
    m1 = np.array(m1, copy=False)
    x2 = np.array(x2, copy=False)
    y2 = np.array(y2, copy=False)
    m2 = np.array(m2, copy=False)
 
    if x1.shape != y1.shape:
        raise ValueError('x1 and y1 do not match!')
    if x2.shape != y2.shape:
        raise ValueError('x2 and y2 do not match!')
 
    # Setup coords1 pairs and coords 2 pairs
    # this is equivalent to, but faster than just doing np.array([x1, y1])
    coords1 = np.empty((x1.size, 2))
    coords1[:, 0] = x1
    coords1[:, 1] = y1
 
    # this is equivalent to, but faster than just doing np.array([x1, y1])
    coords2 = np.empty((x2.size, 2))
    coords2[:, 0] = x2
    coords2[:, 1] = y2

    # Utimately we will generate arrays of indices.
    # idxs1 is the indices for matches into catalog 1. This
    # is just a place holder for which stars actually
    # have matches.
    idxs1 = np.ones(x1.size, dtype=int) * -1
    idxs2 = np.ones(x1.size, dtype=int) * -1

    # The matching will be done using a KDTree.
    kdt = KDT(coords2)

    # This returns the number of neighbors within the specified
    # radius. We will use this to find those stars that have no or one
    # match and deal with them easily. The more complicated conflict
    # cases will be dealt with afterward.
    i2_match = kdt.query_ball_point(coords1, dr_tol)
    Nmatch = np.array([len(idxs) for idxs in i2_match])

    # What is the largest number of matches we have for a given star?
    Nmatch_max = Nmatch.max()


    # Loop through and handle all the different numbers of matches.
    # This turns out to be the most efficient so we can use numpy
    # array operations. Remember, skip the Nmatch=0 objects... they
    # already have indices set to -1.
    for nn in range(1, Nmatch_max+1):
        i1_nn = np.where(Nmatch == nn)[0]

        if len(i1_nn) == 0:
            continue

        if nn == 1:
            i2_nn = np.array([i2_match[mm][0] for mm in i1_nn])
            if dm_tol != None:
                dm = np.abs(m1[i1_nn] - m2[i2_nn])
                keep = dm < dm_tol
                idxs1[i1_nn[keep]] = i1_nn[keep]
                idxs2[i1_nn[keep]] = i2_nn[keep]
            else:
                idxs1[i1_nn] = i1_nn
                idxs2[i1_nn] = i2_nn
        else:
            i2_tmp = np.array([i2_match[mm] for mm in i1_nn])

            # Repeat star list 1 positions and magnitudes
            # for nn times (tile then transpose) 
            x1_nn = np.tile(x1[i1_nn], (nn, 1)).T
            y1_nn = np.tile(y1[i1_nn], (nn, 1)).T
            m1_nn = np.tile(m1[i1_nn], (nn, 1)).T

            # Get out star list 2 positions and magnitudes
            x2_nn = x2[i2_tmp]
            y2_nn = y2[i2_tmp]
            m2_nn = m2[i2_tmp]
            dr = np.abs(x1_nn - x2_nn, y1_nn - y2_nn)
            dm = np.abs(m1_nn - m2_nn)

            if dm_tol != None:
                # Don't even consider stars that exceed our
                # delta-mag threshold. 
                dr_msk = np.ma.masked_where(dm > dm_tol, dr)
                dm_msk = np.ma.masked_where(dm > dm_tol, dm)

                # Remember that argmin on masked arrays can find
                # one of the masked array elements if ALL are masked.
                # But our subsequent "keep" check should get rid of all
                # of these.
                dm_min = dm_msk.argmin(axis=1)
                dr_min = dr_msk.argmin(axis=1)

                # Double check that "min" choice is still within our
                # detla-mag tolerence.
                dm_tmp = np.choose(dm_min, dm.T)

                keep = (dm_min == dr_min) & (dm_tmp < dm_tol)
            else:
                dm_min = dm.argmin(axis=1)
                dr_min = dr.argmin(axis=1)

                keep = (dm_min == dr_min)

            i2_keep_2D = i2_tmp[keep]
            dr_keep = dr_min[keep]  # which i2 star for a given i1 star
            ii_keep = np.arange(len(dr_keep))  # a running index for the i2 keeper stars.

            idxs1[i1_nn[keep]] = i1_nn[keep]
            idxs2[i1_nn[keep]] = i2_keep_2D[ii_keep, dr_keep]

    idxs1 = idxs1[idxs1 >= 0]
    idxs2 = idxs2[idxs2 >= 0]        

    dr = np.hypot(x1[idxs1] - x2[idxs2], y1[idxs1] - y2[idxs2])
    dm = m1[idxs1] - m2[idxs2]

    # Deal with duplicates
    duplicates = [item for item, count in list(Counter(idxs2).items()) if count > 1]
    print(( 'Found {0:d} duplicates out of {1:d} matches'.format(len(duplicates), len(dm))))
    keep = np.ones(len(idxs1), dtype=bool)
    for dd in range(len(duplicates)):
        # Index into the idxs1, idxs2 array of this duplicate.
        dups = np.where(idxs2 == duplicates[dd])[0]

        # Assume the duplicates are confused first... see if we
        # can resolve the confusion below.
        keep[dups] = False
        
        dm_dups = m1[idxs1[dups]] - m2[idxs2[dups]]
        dr_dups = np.hypot(x1[idxs1[dups]] - x2[idxs2[dups]], y1[idxs1[dups]] - y2[idxs2[dups]])

        dm_min = np.abs(dm_dups).argmin()
        dr_min = np.abs(dr_dups).argmin()

        # If there is a clearly preferred match (closest in distance and brightness), then
        # keep it and dump the other duplicates.
        if dm_min == dr_min:
            keep[dups[dm_min]] = True
        else:
            print('    confused, dropping')


    # Clean up the duplicates
    idxs1 = idxs1[keep]
    idxs2 = idxs2[keep]
    dr = dr[keep]
    dm = dm[keep]
 
    return idxs1, idxs2, dr, dm

def calc_triangles_vmax_angle(x, y):
    idx = np.arange(len(x), dtype=np.int16)
    
    # Option 1 -- this takes 0.217 seconds for 50 objects
    # t1 = time.time()
    # combo_iter1 = itertools.combinations(idx1, 3)
    # combo_idx1_1 = np.array(list(combo_iter1), dtype=np.int16)
    # t2 = time.time()
    # print( 'Finished Option 1: ', t2 - t1)
    # print( combo_idx1_1.shape)
    # print( combo_idx1_1)
    
    # Option 2 -- this takes 0.016 seconds for 50 objects
    combo_iter = itertools.combinations(idx, 3)
    combo_dt = np.dtype('i2,i2,i2')
    combo_idx_tmp = np.fromiter(combo_iter, dtype=combo_dt)
    combo_idx = combo_idx_tmp.view(np.int16).reshape(-1, 3)
    
    ii0 = combo_idx[:,0]
    ii1 = combo_idx[:,1]
    ii2 = combo_idx[:,2]
    
    dxab = x[ii1] - x[ii0]
    dyab = y[ii1] - y[ii0]
    dxac = x[ii2] - x[ii0]
    dyac = y[ii2] - y[ii0]
    
    dab = np.hypot(dxab, dyab)
    dac = np.hypot(dxac, dyac)
    
    dmax = np.max([dab, dac], axis=0)
    dmin = np.min([dab, dac], axis=0)
    
    vmax = dmin ** 2 / dmax ** 2
    vmax[dab < dac] *= -1
    
    vdprod = dxab * dxac + dyab * dyac
    vcprod = dxab * dyac - dyab * dxac
    
    angle = np.degrees( np.arctan2( vdprod, vcprod) )
    angle[angle < 0] += 360.0
    angle[angle > 360] -= 360.0
    
    return combo_idx, vmax, angle

def add_votes(votes, match1, match2):
    # Construct a histogram of how often a bin is matched... then add the delta
    flat_idx = np.ravel_multi_index((match1, match2), dims=votes.shape)
    
    # extract the unique indices and their position
    unique_idx, idx_idx = np.unique(flat_idx, return_inverse=True)
    
    # aggregate the repeated indices
    deltas = np.bincount(idx_idx)
    
    # Sum them to the array
    votes.flat[unique_idx] += deltas
    
    return
