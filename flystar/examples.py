import transforms
import match
import align
import starlists



def align_example(labelFile, reference, transModel=transform.four_paramNW, order=1, N_loop=2, dr_tol=1.0, dm_tol=None, weights=False):
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

    weights: boolean
        If true, use weights to calculate transformation. These weights are
        based on position and velocity errors
    
    Output:
    ------
        
    """
    # Read in label.dat file and reference starlist, changing columns to their
    # standard column headers/epochs/orientations
    starlist = align.readStarlist(reference)
    t0 = starlist['t'][0]

    label = align.readLabel(labelFile, t0)

    # Perform blind matching of 100 brightest stars and calculate initial transform
    briteN = 100
    trans = align.initial_align(label, starlist, briteN, transformModel=transModel,
                                order=order)

    # Use transformation to match starlists, then recalculate transformation.
    # Iterate on this as many times as desired
    for i in range(N_loop):
        dr_tol = 1.0 # in pixels 
        label_mat_orig, label_mat, starlist_mat = align.transform_and_match(label,
                                                                            starlist,
                                                                            trans,
                                                                            dr_tol=dr_tol,
                                                                            dm_tol=dm_tol)

        trans, N_trans = align.find_transform(label_mat_orig, starlist_mat, transModel=transModel,
                                     order=order, weights=weights)


    # Write final transform in java align format
    align.write_transform(trans, labelFile, reference, N_trans)

    
    # Test transform: apply to label.dat, make diagnostic plots
    label_trans = align.transform(label, 'outTrans.txt')
    

    # Diagnostic plots


    
    return
    

def align_Arches(labelFile, reference, transModel=transform.four_paramNW, order=1, N_loop=2, dr_tol=1.0, dm_tol=None, weights=False):
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

    weights: boolean
        If true, use weights to calculate transformation. These weights are
        based on position and velocity errors
    
    Output:
    ------
        
    """
    # Read in label.dat file and reference starlist, changing columns to their
    # standard column headers/epochs/orientations
    starlist = align.readStarlist(reference)
    t0 = starlist['t'][0]

    label = align.readLabel(labelFile, t0)

    # Perform blind matching of 100 brightest stars and calculate initial transform
    briteN = 100
    trans = align.initial_align(label, starlist, briteN, transformModel=transModel,
                                order=order)

    # Use transformation to match starlists, then recalculate transformation.
    # Iterate on this as many times as desired
    for i in range(N_loop):
        dr_tol = 1.0 # in pixels 
        label_mat_orig, label_mat, starlist_mat = align.transform_and_match(label,
                                                                            starlist,
                                                                            trans,
                                                                            dr_tol=dr_tol,
                                                                            dm_tol=dm_tol)

        trans, N_trans = align.find_transform(label_mat_orig, starlist_mat, transModel=transModel,
                                     order=order, weights=weights)


    # Write final transform in java align format
    align.write_transform(trans, labelFile, reference, N_trans)

    
    # Test transform: apply to label.dat, make diagnostic plots
    label_trans = align.transform(label, 'outTrans.txt')
    

    # Diagnostic plots


    
    return
    
def align_example_gc(labelFile, reference, transModel=transforms.four_paramNW, order=1, N_loop=2):
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

    Output:
    ------
        
    """
    # Read in label.dat file and reference starlist, changing columns to their
    # standard column headers/epochs/orientations
    starlist = align.readStarlist(reference)
    t0 = starlist['t'][0]
    label = align.readLabel(labelFile, t0)

    #  
    # Perform blind matching and calculate initial transform
    trans = initial_align()


    # Use transformation to match starlists, then recalculate transformation.
    # Iterate on this as many times as desired
    for i in range(N_loop):
        matched = align.transform_and_match()

        trans = align.find_transform()


    # Write final transform in java align format
    align.write_transform()

    
    # Test transform: apply to label.dat, make diagnostic plots
    label_trans = align.transform(label, 'outTrans.txt')
    

    # Diagnostic plots


    
    return
 
