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


def robust_sigma(values):
    """
    Gaussian-consistent robust scatter, 1.4826 * median absolute deviation.

    Used to measure the real offset scatter between two matched catalogs
    without trusting any per-star error columns. NaNs are ignored; returns NaN
    if nothing finite is left.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return np.nan

    return 1.4826 * np.median(np.abs(v - np.median(v)))


def best_and_runner_up(keys, score, n_keys):
    """
    For each key, find its lowest-scoring pair and the score of the runner-up.

    Parameters
    ----------
    keys : int array
        Group label of every candidate pair -- the catalog-1 index when ranking
        each list star's candidates, the catalog-2 index when ranking each
        reference star's suitors.
    score : float array
        The score (chi^2) of every candidate pair. Same length as keys.
    n_keys : int
        Size of the catalog the keys index into, so the returned arrays can be
        addressed directly by catalog index.

    Returns
    ----------
    best_pair : int array, length n_keys
        Index into keys/score of each key's best pair, or -1 if the key has no
        candidate pairs at all.
    delta : float array, length n_keys
        score(runner-up) - score(best) for each key: how decisively the best
        pair wins. inf when the key has exactly one candidate (nothing to be
        confused with), and 0.0 for keys with no candidates.
    """
    best_pair = np.full(n_keys, -1, dtype=int)
    delta = np.zeros(n_keys, dtype=float)

    if len(keys) == 0:
        return best_pair, delta

    # Sort by key, then by score within each key, so each group's best pair is
    # its first element and the runner-up is the one right after it.
    order = np.lexsort((score, keys))
    k_sorted = keys[order]
    s_sorted = score[order]

    is_first = np.ones(len(k_sorted), dtype=bool)
    is_first[1:] = k_sorted[1:] != k_sorted[:-1]
    i_first = np.flatnonzero(is_first)

    best_pair[k_sorted[i_first]] = order[i_first]

    # The runner-up exists only if the next entry belongs to the same key.
    i_next = np.minimum(i_first + 1, len(k_sorted) - 1)
    has_runner_up = (i_first + 1 < len(k_sorted)) & (k_sorted[i_next] == k_sorted[i_first])
    runner_up = np.where(has_runner_up, s_sorted[i_next], np.inf)

    delta[k_sorted[i_first]] = runner_up - s_sorted[i_first]

    return best_pair, delta


def calibrate_match_scales(pair_i, dx, dy, dm, n_stars, dr_tol, dm_tol,
                           verbose=True, min_pairs=10):
    """
    Measure the position and magnitude scatter between two catalogs from the
    catalogs themselves, so the chi^2 needs no per-star error columns.

    A star with exactly one candidate inside the tolerances needs no
    tie-breaking, so the spread of those offsets is a clean estimate of how far
    apart the same star lands in the two catalogs -- centroiding error,
    transformation error and any systematic, all folded in. That is precisely
    the scale needed to judge whether one candidate is decisively closer than
    another.

    Three tiers, in order of preference:

    1. Unambiguous (single-candidate) pairs. Cleanest, and what any real
       catalog pairing supplies in bulk.
    2. Each star's nearest candidate. Contains some wrong pairs, which inflates
       the scale and so errs toward calling things ambiguous.
    3. dr_tol / 10, with the magnitude term switched off. Reached only when
       there are barely any candidates to learn from. dr_tol is a search
       radius, chosen with room to spare, so the true scatter sits well inside
       it; anchoring the scale AT the tolerance would make a candidate 20x
       closer than its rival look like a coin toss.

    The magnitude term is used only if its scale was actually measured (tier 1
    or 2). Without a measured scale there is no defensible exchange rate
    between arcseconds and magnitudes, and inventing one from the ratio of the
    two tolerances is the very mistake this is meant to remove -- dm_tol keeps
    working as a hard gate regardless.

    Parameters
    ----------
    pair_i : int array
        Catalog-1 index of every candidate pair that passed the tolerances.

    dx, dy, dm : float array
        Offsets of those pairs.
    n_stars : int
        Length of catalog 1.
    dr_tol : float
        Match radius, for the tier-3 fallback.
    dm_tol : float or None
        Magnitude tolerance. None means magnitudes are not compared at all.
    min_pairs : int, optional
        Fewest pairs a tier needs before its scatter is trusted, by default 10.

    Returns
    ----------
    sigma_pos : float
        Per-axis position scatter.
    sigma_mag : float or None
        Magnitude scatter, or None to score on position alone.
    """
    n_cand = np.bincount(pair_i, minlength=n_stars)

    def scales_from(mask):
        if int(mask.sum()) < min_pairs:
            return np.nan, np.nan
        sx = robust_sigma(dx[mask])
        sy = robust_sigma(dy[mask])
        s_pos = np.sqrt(0.5 * (sx**2 + sy**2)) if np.isfinite(sx) and np.isfinite(sy) else np.nan
        s_mag = robust_sigma(dm[mask]) if dm_tol is not None else np.nan
        return s_pos, s_mag

    # Tier 1: pairs belonging to a star that had exactly one candidate.
    tier = 'unambiguous pairs'
    sigma_pos, sigma_mag = scales_from(n_cand[pair_i] == 1)

    # Tier 2: each star's nearest candidate, ambiguous stars included.
    if not np.isfinite(sigma_pos) or sigma_pos <= 0:
        dr = np.hypot(dx, dy)
        nearest = np.zeros(len(pair_i), dtype=bool)
        order = np.lexsort((dr, pair_i))
        i_sorted = pair_i[order]
        is_first = np.ones(len(i_sorted), dtype=bool)
        is_first[1:] = i_sorted[1:] != i_sorted[:-1]
        nearest[order[is_first]] = True
        tier = 'nearest candidates'
        sigma_pos, sigma_mag = scales_from(nearest)

    # Tier 3: nothing to learn from.
    if not np.isfinite(sigma_pos) or sigma_pos <= 0:
        tier = 'dr_tol/10 fallback'
        sigma_pos = dr_tol / 10.0
        sigma_mag = np.nan

    if dm_tol is None or not np.isfinite(sigma_mag) or sigma_mag <= 0:
        sigma_mag = None

    if verbose > 2:
        mag_msg = 'position only' if sigma_mag is None else f'sigma_mag={sigma_mag:.4f}'
        print(f'    Match scales from {tier}: sigma_pos={sigma_pos:.6f}, {mag_msg}')

    return sigma_pos, sigma_mag


def match_chi2(x1, y1, m1, x2, y2, m2, i2_match, dr_tol, dm_tol,
               dchi2_tol=9.0, sigma_pos=None, sigma_mag=None, verbose=True):
    r"""
    Resolve candidate matches by chi^2, keeping only reciprocal best pairs.

    Scores every candidate pair as

    .. math::

        \chi^2 = \frac{\Delta x^2 + \Delta y^2}{\sigma_\mathrm{pos}^2}
                 + \frac{\Delta m^2}{\sigma_\mathrm{mag}^2}

    and matches a pair when it is BOTH stars' lowest-chi^2 candidate and wins
    by at least dchi2_tol over each star's runner-up.

    This replaces two rules that were losing good matches in crowded fields.
    The first required a star's nearest candidate in position to also be its
    nearest in magnitude, which discarded a 3.6 mas match because a star 83 mas
    away happened to be 0.06 mag closer -- position and magnitude were compared
    as equals no matter how lopsided the evidence. Scoring in units of the
    measured scatter lets each carry the weight it has earned: a 20x closer
    candidate produces a chi^2 difference in the hundreds, while a fraction of
    a magnitude produces a few, so magnitude only decides when the positions
    are genuinely coincident. Second, one-to-one was enforced by resolving
    duplicates after the fact with the same both-must-agree rule; requiring the
    match to be reciprocal is symmetric by construction and needs no
    arbitration.

    Parameters
    ----------
    x1, y1, m1, x2, y2, m2 : float array
        The two catalogs, already on a common system.
    i2_match : list of lists
        Candidate catalog-2 indices within dr_tol of each catalog-1 star, as
        returned by the KD-tree radius query.

    dr_tol, dm_tol : float, float or None
        Hard search tolerances, already applied to i2_match for dr.
    dchi2_tol : float, optional
        How much better the best candidate must be than the runner-up, in
        chi^2. The default 9 is a 3-sigma margin. Below it the pair is treated
        as genuinely ambiguous and left unmatched.

    sigma_pos, sigma_mag : float or None
        Scales for the chi^2. None (the default) measures them from the
        unambiguous pairs of these two catalogs -- no error columns needed, by default 9.0.

    Returns
    ----------
    idxs1, idxs2, dr, dm : arrays
        As match().
    """
    n_cand = np.array([len(c) for c in i2_match])
    n_pairs_total = int(n_cand.sum())

    if n_pairs_total == 0:
        empty_i = np.zeros(0, dtype=int)
        empty_f = np.zeros(0, dtype=float)
        return empty_i, empty_i, empty_f, empty_f

    pair_i = np.repeat(np.arange(len(x1)), n_cand)
    pair_j = np.fromiter(itertools.chain.from_iterable(i2_match), dtype=int,
                         count=n_pairs_total)

    dx = x2[pair_j] - x1[pair_i]
    dy = y2[pair_j] - y1[pair_i]
    dm = m2[pair_j] - m1[pair_i]

    # Apply the hard gates. A non-finite offset can never be a match, and the
    # KD-tree was built with non-finite catalog-2 coordinates replaced by 0, so
    # those rows must be dropped here rather than scored.
    good = np.isfinite(dx) & np.isfinite(dy)
    if dm_tol is not None:
        good &= np.isfinite(dm) & (np.abs(dm) < dm_tol)

    pair_i, pair_j = pair_i[good], pair_j[good]
    dx, dy, dm = dx[good], dy[good], dm[good]

    if len(pair_i) == 0:
        empty_i = np.zeros(0, dtype=int)
        empty_f = np.zeros(0, dtype=float)
        return empty_i, empty_i, empty_f, empty_f

    if sigma_pos is None or sigma_mag is None:
        auto_pos, auto_mag = calibrate_match_scales(
            pair_i, dx, dy, dm, len(x1), dr_tol, dm_tol, verbose=verbose
        )
        if sigma_pos is None:
            sigma_pos = auto_pos
        if sigma_mag is None:
            sigma_mag = auto_mag

    chi2 = (dx**2 + dy**2) / sigma_pos**2
    if dm_tol is not None and sigma_mag is not None:
        chi2 = chi2 + dm**2 / sigma_mag**2

    best_of_i, delta_i = best_and_runner_up(pair_i, chi2, len(x1))
    best_of_j, delta_j = best_and_runner_up(pair_j, chi2, len(x2))

    # Keep a pair only if each star prefers the other, and each prefers it
    # decisively. The reciprocity makes the result independent of which catalog
    # is which; the margin is what used to be called "confused".
    p = np.arange(len(pair_i))
    keep = ((best_of_i[pair_i] == p) & (best_of_j[pair_j] == p) &
            (delta_i[pair_i] >= dchi2_tol) & (delta_j[pair_j] >= dchi2_tol))

    if verbose > 2:
        n_contested = int(((best_of_i[pair_i] == p) & (best_of_j[pair_j] == p)).sum())
        mag_msg = 'off' if sigma_mag is None else f'{sigma_mag:.3f}'
        print(f'    chi2 matching: sigma_pos={sigma_pos:.5f}, sigma_mag={mag_msg}, '
              f'dchi2_tol={dchi2_tol}')
        print(f'    {int(keep.sum())} matched; {n_contested - int(keep.sum())} '
              f'reciprocal pairs dropped as ambiguous')

    idxs1 = pair_i[keep]
    idxs2 = pair_j[keep]

    return idxs1, idxs2, np.hypot(dx[keep], dy[keep]), dm[keep]


def match(x1, y1, m1, x2, y2, m2, dr_tol, dm_tol=None, workers=1, verbose=True,
          matching='legacy', dchi2_tol=9.0, sigma_pos=None, sigma_mag=None):
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
        If None, then any delta-magnitude is allowed, by default None.
    workers : int, optional
        Number of jobs to schedule for parallel processing. If -1 is given all processors are used. Default: 1.
        By default 1.
    verbose : bool or int, optional
        Prints on screen information on the matching. Higher verbose values
        (up to 9) provide more detail, by default True.
    matching : {'legacy', 'chi2'}, optional
        How to resolve a star with more than one candidate, and how to enforce
        one-to-one.

        'legacy' (default) keeps the historical rules: a multi-candidate star
        is matched only if its nearest candidate in position is also its
        nearest in magnitude, and duplicates are arbitrated afterwards by the
        same both-must-agree test. In a crowded field this discards good
        matches -- a candidate 20x closer loses to one a few hundredths of a
        magnitude nearer in brightness -- and every discarded star then becomes
        a duplicate reference entry that makes the next catalog ambiguous too.

        'chi2' scores each candidate as (dr/sigma_pos)^2 + (dm/sigma_mag)^2 and
        keeps reciprocal best pairs that win by dchi2_tol. See match_chi2().
        By default 'legacy'.
    dchi2_tol : float, optional
        matching='chi2' only. Required chi^2 margin over the runner-up.
        Default 9.0, a 3-sigma margin, by default 9.0.
    sigma_pos : float or None, optional
        matching='chi2' only. Position scale for the chi^2, in the units of
        x1/y1. None (default) measures it from the unambiguous pairs of these
        two catalogs, so no per-star error columns are needed, by default None.
    sigma_mag : float or None, optional
        matching='chi2' only. Magnitude scale for the chi^2. None (default)
        measures it the same way, by default None.

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
    i2_match = kdt.query_ball_point(coords1, dr_tol, workers=workers)

    if matching == 'chi2':
        return match_chi2(x1, y1, m1, x2, y2, m2, i2_match, dr_tol, dm_tol,
                          dchi2_tol=dchi2_tol, sigma_pos=sigma_pos,
                          sigma_mag=sigma_mag, verbose=verbose)
    elif matching != 'legacy':
        raise ValueError(f"matching must be 'legacy' or 'chi2', got {matching!r}")

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
