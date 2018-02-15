from flystar import transforms
from flystar import match
from flystar import align
from flystar import starlists
from flystar import plots
import numpy as np
import copy
import pdb


def align_example(labelFile, reference, transModel=transforms.four_paramNW, order=1, N_loop=2,
                  dr_tol=1.0, dm_tol=None, briteN=100, weights=False, outFile='outTrans.txt'):
    """
    Base example of how to use the flystar code. Assumes we are transforming a label.dat into
    a reference starlist.

    Parameters:
    -----------
    labelFile: ascii file
        Starlist we would like to transform into the reference frame. For this
        code, we expect a label.dat file

    reference: ascii file
        Starlist that defines the reference frame

    transModel: transformation class (default: transform.four_paramNW)
        Defines which transformation model to use. Only the four-parameter and
        1st/2nd order polynomial transformations are supported

    order: int (default=1)
        Order of the polynomial transformation. Only used for polynomial transform

    N_loop: int (default=2)
        How many times to iterate on the transformation calculation. Ideally,
        each iteration adds more stars and thus a better transform, to some
        limit.

    dr_tol: float (default = 1.0)
        The search radius for the matching algorithm, in the same units as the
        starlist file positions.
        
    dm_tol: float or None
        If float, sets the maximum magnitude difference allowed in matching
        between label.dat and starlist. Note that this should be set to
        None if the label.dat and refStarlist are in different filters.

    briteN: int (default = 100)
        Number of bright stars in both starlists to run the blind matching
        algorithm on. britN must be less than the length of either starlist

    weights: boolean (default = False)
        If true, use weights to calculate transformation. These weights are
        based on position and velocity errors

    outFile: string (default = 'outTrans.txt')
        Name of output ascii file which contains the transform parameters.
    
    Output:
    ------
        
    """
    # Read in label.dat file and reference starlist, changing columns to their
    # standard column headers/epochs/orientations
    starlist = starlists.read_starlist(reference)
    t0 = starlist['t'][0]

    label = starlists.read_label(labelFile, t0)

    # Perform blind matching of briteN brightest stars and calculate initial transform
    trans = align.initial_align(label, starlist, briteN, transformModel=transModel,
                                order=order)

    # Apply intial transformation to label.dat (for error weighting purposes below)
    label_trans = align.transform_from_object(label, trans)
    
    # Use transformation to match starlists, then recalculate transformation.
    # Iterate on this as many times as desired
    for i in range(N_loop):
        idx_label, idx_starlist = align.transform_and_match(label, starlist,
                                                            trans,
                                                            dr_tol=dr_tol,
                                                            dm_tol=dm_tol)
        
        trans, N_trans = align.find_transform(label[idx_label],
                                              label_trans[idx_label],
                                              starlist_mat[idx_starlist],
                                              transModel=transModel,
                                              order=order, weights=weights)


    # Write final transform in java align format
    print('Write transform to {0}'.format(outFile))
    align.write_transform(trans, labelFile, reference, N_trans, outFile=outFile)
    
    # Test transform: apply final transformation to label.dat
    label_trans2 = align.transform(label, outFile)

    # Make diagnostic plots
    
    return
    

def align_Arches(labelFile, reference, transModel=transforms.four_paramNW, order=1, N_loop=2,
                 dr_tol=1.0, dm_tol=None, briteN=100, weights=None, restrict=False,
                 outFile='outTrans.txt'):
    """
    Application of flystar code to align Arches label.dat and reference starlist.
    Transforming the label file into the frame of the reference starlist.

    Parameters:
    -----------
    labelFile: ascii file
        Starlist we would like to transform into the reference frame. For this
        code, we expect a label.dat file.

    reference: ascii file
        Starlist that defines the reference frame

    transModel: transformation class (default: transform.four_paramNW)
        Defines which transformation model to use. Only the four-parameter and
        1st/2nd order polynomial transformations are supported

    order: int (default=1)
        Order of the polynomial transformation. Only used for polynomial transform

    N_loop: int (default=2)
        How many times to iterate on the transformation calculation. Ideally,
        each iteration adds more stars and thus a better transform, to some
        limit.

    dr_tol: float (default = 1.0)
        The search radius for the matching algorithm, in the same units as the
        starlist file positions.
        
    dm_tol: float or None (default = None)
        If float, sets the maximum magnitude difference allowed in matching
        between label.dat and starlist. Note that this should be set to
        None if the label.dat and refStarlist are in different filters.

    briteN: int (default = 100)
        Number of bright stars in both starlists to run the blind matching
        algorithm on. britN must be less than the length of either starlist

    weights: string (default=None)
        if weights=='both', we use both position error and velocity error in transformed
        starlist and reference starlist as uncertanties. And weights is the reciprocal 
            of this uncertanty.
        if weights=='starlist', we only use postion error and velocity error in transformed
        starlist as uncertainty.
        if weights=='reference', we only use position error in reference starlist as uncertainty.
        if weights==None, we don't use weights.

    restrict: boolean (default = False)
        If true, restrict to only stars with use > 2 in the label.dat file.
        This behaves same way as java align -restrict flag.

    outFile: string (default = 'outTrans.txt')
        Name of output ascii file which contains the transform parameters.
    
    Output:
    ------
    outFile is written containing the tranformation coefficients

    Diagnostic plots included from plots.py
    -Transformed_positons.png: Shows positions of matched stars (both reference
        starlist and label.dat) in reference starlist coordinates. The label.dat
        coordinates are the ones after the derived transformation has been applied.

    -Positions_hist.png: Histogram of the difference between the reference list
        positions and the label.dat positions after transformation.

    -Positions_quiver.png: Quiver plot showing the difference between reference
        positions and transformed label.dat positions as a function of location. 
        
    -Magnitude_hist.png: Histogram of the difference between the reference list
        magnitude and label.dat magnitude for matched stars.
        
    """
    # Read in label.dat file and reference starlist, changing columns to their
    # standard column headers/epochs/orientations
    starlist = starlists.read_starlist(reference)
    t0 = starlist['t'][0]

    label = starlists.read_label(labelFile, t0)

    # Restrict label.dat file to all stars within 15 arcseconds of central star
    # This is the area covered by the reference starlist
    area = [[-15, 15], [-15,15]]
    idx_area = starlists.restrict_by_area(label, area)
    label_r = label[idx_area]

    # Perform blind matching of 100 brightest stars and calculate initial transform
    trans = align.initial_align(label_r, starlist, briteN, transformModel=transModel,
                                order=order)
    
    # Apply transformation to label.dat file, for weighting purposes.
    label_trans = align.transform_from_object(label, trans)
    
    # Use transformation to match starlists, then recalculate transformation.
    # Iterate on this as many times as desired
    for i in range(N_loop):
        idx_label, idx_starlist = align.transform_and_match(label, starlist,
                                                            trans, dr_tol=dr_tol,
                                                            dm_tol=dm_tol)
        # Restrict to use > 2, if desired
        if restrict:
            idx_label, idx_starlist = starlists.restrict_by_use(label[idx_label],
                                                                starlist[idx_starlist],
                                                                idx_label,
                                                                idx_starlist)

        trans, N_trans = align.find_transform(label[idx_label],
                                              label_trans[idx_label],
                                              starlist[idx_starlist],
                                              transModel=transModel,
                                              order=order,
                                              weights=weights)

    # Calculate delta mag (reference - starlist) for matching stars
    delta_m = np.mean(starlist[idx_starlist]['m'] - label[idx_label]['m'])

    # Write final transform in java align format
    print('Write transform to {0}'.format(outFile))
    align.write_transform(trans, labelFile, reference, N_trans, deltaMag=delta_m,
                          restrict=restrict, weights=weights, outFile=outFile)
    
    # Test transform: apply to label.dat, make diagnostic plots
    label_trans2 = align.transform_from_file(label, outFile)

    #--------------------#
    # Diagnostic plots
    #--------------------#
    print('Making test plots')
    xlim = [0, 1200]
    ylim = [0, 1200]
    #vxlim = [-5/121.625, 8/121.625]
    #vylim = [-8/121.625, 5/121.625]

    # Transformed positions compared to reference positions
    plots.trans_positions(starlist, starlist[idx_starlist], label_trans2,
                          label_trans2[idx_label], xlim=xlim, ylim=ylim)

    # Histogram of difference in transformed and reference positions for
    # matched stars 
    plots.pos_diff_hist(starlist[idx_starlist], label_trans2[idx_label])

    # Histogram of difference in transformed and reference positions for
    # matched stars, divided by the astrometric errors. Also calculates
    # chi-squared
    plots.pos_diff_err_hist(starlist[idx_starlist], label_trans2[idx_label],
                            trans, errs='both', bin_width=0.5, xlim=[-6,6])

    # Histogram of difference in the magnitudes for the matched stars    
    plots.mag_diff_hist(starlist[idx_starlist], label_trans2[idx_label])

    # Quiver plot showing difference between transformed and reference
    # positions for matched stars
    plots.pos_diff_quiver(starlist[idx_starlist], label_trans2[idx_label],
                          keyLength=0.2, qscale=10, xlim=xlim,
                          ylim=ylim, outlier_reject=None)

    print('Done with plots')

    return



def align_gc(starFile, refFile, transModel=transforms.PolyTransform, order=1, N_loop=2,
          dr_tol=1.0, briteN=100, weights='both', restrict=False, outFile='outTrans.txt'):
    """
    Transforming a starlist(label.dat) into a reference frame.

    Parameters:
    -----------
    starFile: string
        Starlist we would like to transform into the reference frame, eg:label.dat 

    refFile: string
        Starlist that defines the reference frame.

    transModel: transformation class (default: transforms.polyTransform)
        Defines which transformation model to use. Both the four-parameter and
        polynomial transformations are supported

    order: int (default=1)
        Order of the polynomial transformation. Only used for polynomial transform

    N_loop: int (default=2)
        How many times to iterate on the transformation calculation. Ideally,
        each iteration adds more stars and thus a better transform, to some
        limit.

    dr_tol: float(default=1.0)
        the distance tolerance for matching two stars in align.transform_and_match

    briteN: int (default=100)
        the number of stars used in blind matching

    weights: string (default='both')
        if weights=='both', we use both position error in transformed starlist and
           reference starlist as uncertanty. And weights is the reciprocal of this uncertanty.
        if weights=='starlist', we only use postion error in transformed starlist.
        if weights=='reference', we only use position error in reference starlist.
        if weights==None, we don't use weights.

    restrict: boolean (default=False)
        Set to True if transformation restricted to stars with use > 2.

    outFile: string('outTrans.txt')
        the name of the output transformation file
    """

    #----------------------------------------------
    # Read in starlist and reference 
    #----------------------------------------------
    # starlist has postion &  postion err
    ref = starlists.read_starlist(refFile, error=True)
    t_ref = ref['t'][0]
    # label.dat has position & position err and velocity & velocity error
    starlist = starlists.read_label(starFile, prop_to_time=t_ref, flipX=True)


    #--------------------------------------------------
    # Initial transformation with brightest briteN stars
    #--------------------------------------------------
    # define stars that are used to calculate initial transformation.
    idx_ini_ref, idx_ini_starlist, briteN = starlists.restrict_by_name(ref, starlist)

    # Perform a blind trangle-matching of the brightest briteN stars
    # and calculate initial transform
    ref_ini = ref[idx_ini_ref]
    starlist_ini = starlist[idx_ini_starlist]

    trans = align.initial_align(starlist_ini, ref_ini, briteN=briteN,
            transformModel=transModel, order=order)

    # apply the initial transform to label.dat
    # this is used for future weights calculation
    starlist_trans_ini = align.transform_from_object(starlist, trans)

    #------------------------------------------------------------------------
    # Use transformation to match starlists, then recalculate transformation.
    #------------------------------------------------------------------------
    # Iterate on this as many times as desired
    for i in range(N_loop):
        # apply the transformation to label.dat and
        # matched the transformed label with starlist.
        idx_starlist, idx_ref = align.transform_and_match(starlist, ref, trans, dr_tol=dr_tol)

        # use the matched stars to calculate new transformation
        ref_match = ref[idx_ref]
        starlist_match = starlist[idx_starlist]
        starlist_ini_match = starlist_trans_ini[idx_starlist]

        trans, N_trans = align.find_transform(starlist_match, starlist_ini_match, ref_match,
                            transModel=transModel, order=order, weights = weights)


    #---------------------------------------------
    # Write final transform in java align format
    #---------------------------------------------
    delta_m = np.mean(ref_match['m'] - starlist_match['m'])
    align.write_transform(trans, starFile, refFile, N_trans, deltaMag=delta_m,
                          restrict=restrict, weights=weights, outFile=outFile)


    #-----------------------------------------------------------
    # Test transform: apply to label.dat, make diagnostic plots
    #-----------------------------------------------------------
    # apply the final transformation to label.dat
    starlist_trans = align.transform_from_object(starlist, trans)
    starlist_trans_match = starlist_trans[idx_starlist]

    # postion map with every star in starlist and transformed label.
    # both matched and unmatched stars.
    plots.trans_positions(ref, ref_match, starlist_trans, starlist_trans_match)

    # position difference histogram for matched stars.
    plots.pos_diff_hist(ref_match, starlist_trans_match)

    # position diff/ astrometric err histogrm for matched stars.
    # chi2, reduced_chi2, degree of freedom are showed on the plots.
    plots.pos_diff_err_hist(ref_match, starlist_trans_match, trans)

    # magnitude difference histogram for matched stars.
    plots.mag_diff_hist(ref_match, starlist_trans_match)

    # quiver plot of postion residules
    plots.pos_diff_quiver(ref_match, starlist_trans_match)

    return


def align_starlists(starlist, ref, transModel=transforms.PolyTransform, order=2, N_loop=2,
          dr_tol=1.0, briteN=None, weights='both', outFile='outTrans.txt'):
    """
    Transforming a starlist(label.dat) into a reference frame.

    Parameters:
    -----------
    starlist: Table
        Starlist we would like to transform into the reference frame, eg:label.dat 

    ref: Table
        Starlist that defines the reference frame.

    transModel: transformation class (default: transforms.polyTransform)
        Defines which transformation model to use. Both the four-parameter and
        polynomial transformations are supported

    order: int (default=1)
        Order of the polynomial transformation. Only used for polynomial transform

    N_loop: int (default=2)
        How many times to iterate on the transformation calculation. Ideally,
        each iteration adds more stars and thus a better transform, to some
        limit.

    dr_tol: float(default=1.0)
        the distance tolerance for matching two stars in align.transform_and_match

    briteN: int (default=100)
        the number of stars used in blind matching

    weights: string (default='both')
        if weights=='both', we use both position error in transformed starlist and
           reference starlist as uncertanty. And weights is the reciprocal of this uncertanty.
        if weights=='starlist', we only use postion error in transformed starlist.
        if weights=='reference', we only use position error in reference starlist.
        if weights==None, we don't use weights.

    outFile: string('outTrans.txt')
        the name of the output transformation file
    """
    
    #--------------------------------------------------
    # Initial transformation with brightest briteN stars
    #--------------------------------------------------
    # define stars that are used to calculate initial transformation.
    #pdb.set_trace()
    idx_ini_ref, idx_ini_starlist, briteN_ini = starlists.restrict_by_name(ref, starlist)

    # Perform a blind trangle-matching of the brightest briteN stars
    # and calculate initial transform
    ref_ini = ref[idx_ini_ref]
    starlist_ini = starlist[idx_ini_starlist]

    if briteN == None:
        briteN = briteN_ini
    #trans = align.initial_align(starlist_ini, ref_ini, briteN=briteN,
    #        transformModel=transModel, order=order)
    #pdb.set_trace()
    trans = transModel(starlist_ini['x'], starlist_ini['y'], ref_ini['x'], ref_ini['y'],order=order, weights=None)

    # apply the initial transform to label.dat
    # this is used for future weights calculation
    starlist_trans_ini = align.transform_from_object(starlist, trans)

    #------------------------------------------------------------------------
    # Use transformation to match starlists, then recalculate transformation.
    #------------------------------------------------------------------------
    # Iterate on this as many times as desired
    for i in range(N_loop):
        # apply the transformation to label.dat and
        # matched the transformed label with starlist.
        idx_starlist, idx_ref = align.transform_and_match(starlist, ref, trans, dr_tol=dr_tol)

        # use the matched stars to calculate new transformation
        ref_match = ref[idx_ref]
        starlist_match = starlist[idx_starlist]
        starlist_ini_match = starlist_trans_ini[idx_starlist]

        trans, N_trans = align.find_transform(starlist_match, starlist_ini_match, ref_match,
                            transModel=transModel, order=order, weights = weights)


    #---------------------------------------------
    # Write final transform in java align format
    #---------------------------------------------
    delta_m = np.mean(ref_match['m'] - starlist_match['m'])
    align.write_transform(trans, 'starFile', 'refFile', N_trans, deltaMag=delta_m,
                          restrict=False, weights=weights, outFile=outFile)

    starlist_trans = align.transform_from_object(starlist, trans)
    starlist_trans_match = starlist_trans[idx_starlist]
    return idx_starlist, idx_ref, starlist_trans

def apply_trans(starlist, ref, trans_file, dr_tol=1.0, dm_tol=None):
    """
    apply transformation to starlist with trans_file
    match transformed starslist with ref
    return"""
    starlist_trans = align.transform_from_file(starlist, trans_file)
    x1t = starlist_trans['x']
    y1t = starlist_trans['y']
    m1 = starlist_trans['m']
    x2 = ref['x']
    y2 = ref['y']
    m2 = ref['m']
    idx1, idx2, dr, dm = match.match(x1t, y1t, m1, x2, y2, m2, dr_tol, dm_tol)
    return idx1, idx2, starlist_trans
