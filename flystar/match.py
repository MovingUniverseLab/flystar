import itertools
import numpy as np
from collections import Counter
from scipy.spatial import KDTree as KDT


def miracle_match_briteN(xin1, yin1, min1, xin2, yin2, min2, Nbrite,
                         polygon1=None, polygon2=None, buffer=0, Nbins_vmax=200, Nbins_angle=360,verbose=False):
    """
    Take two input starlists and select the <Nbrite> brightest stars from
    each. Then perform a triangle matching algorithm along the lines of
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

    if verbose:
        print( '')
        print( '  miracle_match_briteN: use brightest {0}'.format(Nbrite))
        print( '  miracle_match_briteN:  ')
        print( '  miracle_match_briteN:  ')

    xin1 = np.array(xin1)
    yin1 = np.array(yin1)
    min1 = np.array(min1)
    xin2 = np.array(xin2)
    yin2 = np.array(yin2)
    min2 = np.array(min2)

    if (polygon1 is not None) and (polygon2 is not None):
        import shapely
        points1 = shapely.points(xin1, yin1)
        points2 = shapely.points(xin2, yin2)
        overlap = polygon1.intersection(polygon2).buffer(buffer)
        in_poly1 = shapely.contains(overlap, points1)
        in_poly2 = shapely.contains(overlap, points2)
        xin1 = xin1[in_poly1]
        yin1 = yin1[in_poly1]
        min1 = min1[in_poly1]
        xin2 = xin2[in_poly2]
        yin2 = yin2[in_poly2]
        min2 = min2[in_poly2]
    # else:
    #     # Only look for matches within overlapping minimum-bounding-boxes of the 2 lists
    #     valid1 = (np.isfinite(xin1)) & (np.isfinite(yin1)) & (np.isfinite(min1))
    #     valid2 = (np.isfinite(xin2)) & (np.isfinite(yin2)) & (np.isfinite(min2))
    #     if (sum(valid1) < Nbrite) or (sum(valid2) < Nbrite):
    #         raise ValueError(
    #             f'Not enough valid stars to find matches! Need at least {Nbrite} valid stars.\n' +
    #             f'Valid stars in list 1: {sum(valid1)}\n' +
    #             f'Valid stars in list 2: {sum(valid2)}\n'
    #         )

    #     xin1 = xin1[valid1]
    #     yin1 = yin1[valid1]
    #     min1 = min1[valid1]
    #     xin2 = xin2[valid2]
    #     yin2 = yin2[valid2]
    #     min2 = min2[valid2]

    #     xmin1, xmax1 = np.min(xin1), np.max(xin1)
    #     ymin1, ymax1 = np.min(yin1), np.max(yin1)
    #     xmin2, xmax2 = np.min(xin2), np.max(xin2)
    #     ymin2, ymax2 = np.min(yin2), np.max(yin2)

    #     # Find the overlapping minimum bounding box
    #     x_overlap = (max(xmin1, xmin2), min(xmax1, xmax2))
    #     y_overlap = (max(ymin1, ymin2), min(ymax1, ymax2))
    #     if x_overlap[0] >= x_overlap[1] or y_overlap[0] >= y_overlap[1]:
    #         fig, ax = plt.subplots()
    #         ax.scatter(xin1, yin1, s=1, label='List 1')
    #         ax.scatter(xin2, yin2, s=1, label='List 2')
    #         ax.set_aspect('equal')
    #         ax.legend()
    #         plt.show()
    #         raise ValueError('The two star lists do not have an overlapping region!')

    #     # Select overlapping regions
    #     in_overlap1 = (xin1 >= x_overlap[0]) & (xin1 <= x_overlap[1]) & (yin1 >= y_overlap[0]) & (yin1 <= y_overlap[1])
    #     in_overlap2 = (xin2 >= x_overlap[0]) & (xin2 <= x_overlap[1]) & (yin2 >= y_overlap[0]) & (yin2 <= y_overlap[1])
    #     if sum(in_overlap1) < Nbrite or sum(in_overlap2) < Nbrite:
    #         raise ValueError(
    #             'Not enough stars in the overlapping region to find matches!\n' +
    #             f'Stars in overlap for list 1: {sum(in_overlap1)}\n' +
    #             f'Stars in overlap for list 2: {sum(in_overlap2)}\n'
    #         )

    #     from matplotlib.patches import Rectangle
    #     fig, ax = plt.subplots()
    #     polygon1 = Rectangle((xmin1, ymin1), xmax1-xmin1, ymax1-ymin1, fill=True, edgecolor='C0', facecolor='C0', alpha=0.5, label='MBB List 1')
    #     polygon2 = Rectangle((xmin2, ymin2), xmax2-xmin2, ymax2-ymin2, fill=True, edgecolor='C2', facecolor='C2', alpha=0.5, label='MBB List 2')
    #     polygon_overlap = Rectangle((x_overlap[0], y_overlap[0]), x_overlap[1]-x_overlap[0], y_overlap[1]-y_overlap[0], fill=True, edgecolor='red', facecolor='C3', alpha=0.5, label='Overlap Region')
    #     ax.scatter(xin1, yin1, s=1, label='List 1')
    #     ax.scatter(xin2, yin2, s=1, label='List 2')
    #     ax.add_patch(polygon1)
    #     ax.add_patch(polygon2)
    #     ax.add_patch(polygon_overlap)
    #     ax.set_aspect('equal')
    #     ax.legend()
    #     plt.show()

    #     xin1 = xin1[in_overlap1]
    #     yin1 = yin1[in_overlap1]
    #     min1 = min1[in_overlap1]
    #     xin2 = xin2[in_overlap2]
    #     yin2 = yin2[in_overlap2]
    #     min2 = min2[in_overlap2]

    # Get/check the lengths of the two starlists
    nin1 = len(xin1)
    nin2 = len(xin2)

    if (nin1 < Nbrite) or (nin2 < Nbrite):
        raise ValueError(
            f'Not enough stars in the overlapping region to find matches! Need at least {Nbrite} valid stars.\n' +
            f'Stars in overlap for list 1: {nin1}\n' +
            f'Stars in overlap for list 2: {nin2}\n'
        )
        # print(f'WARNING: You need at least {Nbrite} to find the matches...')
        # print(f'NIN1: {nin1}')
        # print(f'NIN2: {nin2}')
        # # Nbrite = min(nin1, nin2)
        # # print(f'Updating Nbrite to {Nbrite}...')
        # return (0, None, None, None, None, None, None)

    # Take the Nbrite brightest stars from each list and order by brightness.
    if verbose:
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
    if verbose:
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
    idx1_vmax_hist = np.digitize(vmax1, bins_vmax)  # indices into the 2D array
    idx1_angl_hist = np.digitize(angle1, bins_angle)

    # Make a 2D array and every position is a 3 element vector containing the indicies
    # for the points in this triangle. At each 2D position, we only have a single
    # triangle recorded. We want the first insance. So we just go in reverse order
    # of the triangles and the last updates are the first entries in the original
    # array.
    stars1_at_hist = np.ones((len(bins_vmax) + 1, len(bins_angle) + 1, 3), dtype=np.int16) * -1
    stars1_at_hist[idx1_vmax_hist[::-1], idx1_angl_hist[::-1], :] = stars_in_tri1[::-1]

    ##########
    # List 2
    ##########
    # Make triangles for all combinations within the second starlist.
    stars_in_tri2, vmax2, angle2 = calc_triangles_vmax_angle(x2, y2)

    # Over 2D (vmax, angle) space, decide where everything goes.
    # We only care about the first instance in any bin.
    idx2_vmax_hist = np.digitize(vmax2, bins_vmax)  # indices into the 2D arraya
    idx2_angl_hist = np.digitize(angle2, bins_angle)

    ##########
    # Trim out stars outside our histogram. This only occurs for nan cases.
    # Note, this is a result of digitize not rejecting values outside the specified
    # range. Instead it drops them into bin ii=0 or bin ii=len(bins).
    ##########
    good_idx2 = np.where((idx2_vmax_hist > 0) & (idx2_vmax_hist < len(bins_vmax)) &
                         (idx2_angl_hist > 0) & (idx2_angl_hist < len(bins_angle)))[0]
    idx2_vmax_hist = idx2_vmax_hist[good_idx2]
    idx2_angl_hist = idx2_angl_hist[good_idx2]

    ##########
    # Possible Matches
    ##########
    # Find the triangles that have the same vmax and angle in list 1 and list 2.
    # Make sure to trim out the stuff that fell outside the range (typically nan).
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
    if verbose:
        print( '  miracle_match_briteN: ')
        print( '  miracle_match_briteN: found {0} matches '.format(len(good)))
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


def match(x1, y1, m1, x2, y2, m2, dr_tol, dm_tol=None, verbose=True):
    """
    Finds matches between two different catalogs. No transformations are done and it
    is assumed that the two catalogs are already on the same coordinate system
    and magnitude system.

    For two stars to be matched, they must be within a specified radius (dr_tol) and
    delta-magnitude (dm_tol). For stars with more than 1 neighbor (within the tolerances),
    if one is found that is the best match in both brightness and positional offsets
    (closest in both), then the match is made.
    Otherwise, their is a conflict and no match is returned for the star.


    Parameters
    ----------
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
    verbose : bool or int, optional
        Prints on screen information on the matching. Higher verbose values
        (up to 9) provide more detail.

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

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape or if they do not contain any finite values.
        Or when no match is found between the two catalogs.
    """

    x1 = np.array(x1, copy=False)
    y1 = np.array(y1, copy=False)
    m1 = np.array(m1, copy=False)
    x2 = np.array(x2, copy=False)
    y2 = np.array(y2, copy=False)
    m2 = np.array(m2, copy=False)

    for val, name in zip([x1, y1, m1, x2, y2, m2], ['x1', 'y1', 'm1', 'x2', 'y2', 'm2']):
        if not np.isfinite(val).any():
            raise ValueError(f'{name} does not contain any finite values!')

    assert x1.shape == y1.shape, 'x1 and y1 do not match!'
    assert x2.shape == y2.shape, 'x2 and y2 do not match!'

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
    #kdt = KDT(coords2, balanced_tree=False)
    #KDTree handling of NaNs throws error in scipy v1.10.1 and newer.
    #Replace NaNs in coords2 with zero (0). -SKT
    kdt = KDT(np.where(np.isfinite(coords2), coords2, 0), balanced_tree=False)

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
            if dm_tol is not None:
                dm = np.abs(m1[i1_nn] - m2[i2_nn])
                keep = dm < dm_tol
                idxs1[i1_nn[keep]] = i1_nn[keep]
                idxs2[i1_nn[keep]] = i2_nn[keep]
            else:
                idxs1[i1_nn] = i1_nn
                idxs2[i1_nn] = i2_nn
        else:
            i2_tmp = np.array([i2_match[mm] for mm in i1_nn])

            x1_nn = x1[i1_nn]
            y1_nn = y1[i1_nn]
            m1_nn = m1[i1_nn]

            # Get out star list 2 positions and magnitudes
            x2_nn = x2[i2_tmp]
            y2_nn = y2[i2_tmp]
            m2_nn = m2[i2_tmp]
            dr = np.hypot(x2_nn - x1_nn[:, np.newaxis], y2_nn - y1_nn[:, np.newaxis])
            dm = np.abs(m2_nn - m1_nn[:, np.newaxis])

            if dm_tol is not None:
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
                dm_tmp = np.array([dm.T[dm_min[I]][I] for I in np.ndindex(dm_min.shape)])

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

    dr = np.hypot(x2[idxs2] - x1[idxs1], y2[idxs2] - y1[idxs1])
    dm = m2[idxs2] - m1[idxs1]

    # Deal with duplicates
    duplicates = [item for item, count in list(Counter(idxs2).items()) if count > 1]
    if verbose > 2:
        print(( '    Found {0:d} duplicates out of {1:d} matches'.format(len(duplicates), len(dm))))
    keep = np.ones(len(idxs1), dtype=bool)
    for dd in range(len(duplicates)):
        # Index into the idxs1, idxs2 array of this duplicate.
        dups = np.where(idxs2 == duplicates[dd])[0]

        # Assume the duplicates are confused first... see if we can resolve the confusion below.
        keep[dups] = False
        best_dm = np.abs(m2[idxs2[dups]] - m1[idxs1[dups]]).argmin()
        best_dr = np.hypot(x2[idxs2[dups]] - x1[idxs1[dups]], y2[idxs2[dups]] - y1[idxs1[dups]]).argmin()

        # If there is a clearly preferred match (closest in distance and brightness), then
        # keep it and dump the other duplicates. Otherwise, drop the match as confused.
        if best_dm == best_dr:
            keep[dups[best_dm]] = True
        elif verbose > 3:
            print('    confused, dropping star at',x2[idxs2[dups]][0],y2[idxs2[dups]][0])

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
