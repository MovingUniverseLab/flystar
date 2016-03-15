import transforms
import match
import align
import starlists
import plots
import numpy as np
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
    starlist = align.readStarlist(reference)
    t0 = starlist['t'][0]

    label = align.readLabel(labelFile, t0)

    # Perform blind matching of briteN brightest stars and calculate initial transform
    trans = align.initial_align(label, starlist, briteN, transformModel=transModel,
                                order=order)

    # Use transformation to match starlists, then recalculate transformation.
    # Iterate on this as many times as desired
    for i in range(N_loop):
        label_mat_orig, label_mat, starlist_mat = align.transform_and_match(label,
                                                                            starlist,
                                                                            trans,
                                                                            dr_tol=dr_tol,
                                                                            dm_tol=dm_tol)

        trans, N_trans = align.find_transform(label_mat_orig, starlist_mat, transModel=transModel,
                                     order=order, weights=weights)


    # Write final transform in java align format
    print 'Write transform to {0}'.format(outFile)
    align.write_transform(trans, labelFile, reference, N_trans, outFile=outFile)
    
    # Test transform: apply to label.dat, make diagnostic plots
    label_trans = align.transform(label, outFile)
    
    return
    

def align_Arches(labelFile, reference, transModel=transforms.four_paramNW, order=1, N_loop=2,
                 dr_tol=1.0, dm_tol=None, briteN=100, weights=None, restrict=False,
                 outFile='outTrans.txt'):
    """
    Application of flystar code to align Arches label.dat and reference starlist..

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
        
    dm_tol: float or None (default = None)
        If float, sets the maximum magnitude difference allowed in matching
        between label.dat and starlist. Note that this should be set to
        None if the label.dat and refStarlist are in different filters.

    briteN: int (default = 100)
        Number of bright stars in both starlists to run the blind matching
        algorithm on. britN must be less than the length of either starlist

    weights: None or function
        If None, do not use weights in transformation. If function, use that
        function to calculate weights to be used in the transformation.

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
    starlist = align.readStarlist(reference)
    t0 = starlist['t'][0]

    label = align.readLabel(labelFile, t0)

    # Restrict label.dat file to all stars within 15 arcseconds of central star
    # This is the area covered by the reference starlist
    area = [[-15, 15], [-15,15]]
    label_r = starlists.restrict_by_area(label, area)

    # Perform blind matching of 100 brightest stars and calculate initial transform
    trans = align.initial_align(label_r, starlist, briteN, transformModel=transModel,
                                order=order)

    # Use transformation to match starlists, then recalculate transformation.
    # Iterate on this as many times as desired
    for i in range(N_loop):
        label_mat_orig, label_mat, starlist_mat = align.transform_and_match(label,
                                                                            starlist,
                                                                            trans,
                                                                            dr_tol=dr_tol,
                                                                            dm_tol=dm_tol)

        # Restrict to use > 2, if desired
        if restrict:
            label_mat_orig, label_mat, starlist_mat = starlists.restrict_by_use(label_mat_orig,
                                                                                label_mat,
                                                                                starlist_mat)

        trans, N_trans = align.find_transform(label_mat_orig, starlist_mat,
                                              transModel=transModel, order=order,
                                              weights=weights)

    # Calculate delta mag (reference - starlist) for matching stars
    delta_m = np.mean(starlist_mat['m'] - label_mat['m'])
    
    # Write final transform in java align format
    print 'Write transform to {0}'.format(outFile)
    align.write_transform(trans, labelFile, reference, N_trans, delta_m,
                          restrict=restrict, outFile=outFile)
    
    # Test transform: apply to label.dat, make diagnostic plots
    label_trans = align.transform_by_file(label, outFile)

    #--------------------#
    # Diagnostic plots
    #--------------------#
    print 'Making test plots'

    plots.trans_positions(starlist, starlist_mat, label_trans, label_mat)
    plots.posDiff_hist(starlist_mat, label_mat)
    plots.magDiff_hist(starlist_mat, label_mat)
    plots.posDiff_quiver(starlist_mat, label_mat)
    
    print 'Done with plots'        

    return
    
def align_gc(labelFile, reference, transModel=transforms.four_paramNW, order=1, N_loop=2, 
                dr_tol=1.0, dm_tol=None, weights='both', outFile='outTrans.txt'):
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

    transModel: transformation class (default: transforms.four_paramNW)
        Defines which transformation model to use. Only the four-parameter and
        1st/2nd order polynomial transformations are supported

    order: int (default=1)
        Order of the polynomial transformation. Only used for polynomial transform

    N_loop: int (default=2)
        How many times to iterate on the transformation calculation. Ideally,
        each iteration adds more stars and thus a better transform, to some
        limit.

    weights: string (default='both')
        which kind of weights are used in calculating transformation.

    Output:
    ------
        
    """
    # Read in label.dat file and reference starlist, changing columns to their
    # standard column headers/epochs/orientations
    starlist = align.readStarlist(reference)
    tref = starlist['t'][0]
    label = align.readLabel(labelFile, tref)

    
    # find the stars with common name  
    label_initial, ref_initial, briteN = starlists.restrict_by_name(label, starlist)
    # Perform blind matching and calculate initial transform
    trans = align.initial_align(label_initial, ref_initial, briteN=briteN, transformModel=transModel,
                                order=order)


    # Use transformation to match starlists, then recalculate transformation.
    # Iterate on this as many times as desired
    for i in range(N_loop):
        label_matched, labelT_matched, ref_matched = align.transform_and_match(label, starlist, trans,
                                                            dr_tol=dr_tol, dm_tol=dm_tol)
        trans, N_trans = align.find_transform(label_matched, labelT_matched, ref_matched, transModel=transModel,
                                                order=order, weights = weights)


    # Write final transform in java align format
    align.write_transform(trans, labelFile, reference, N_trans, outFile=outFile)

    # Test transform: apply to label.dat, make diagnostic plots
    label_trans = align.transform(label, outFile)

    # Diagnostic plots
    pdb.set_trace()

    print 'Making test plots'

    plots.trans_positions(starlist, ref_matched, label_trans, labelT_matched)
    plots.posDiff_hist(ref_matched, labelT_matched)
    plots.magDiff_hist(ref_matched, labelT_matched)
    plots.posDiff_quiver(ref_matched, labelT_matched)

    
    return
 
