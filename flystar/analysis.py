import numpy as np
from scipy.stats import f

def calc_chi2(ref_mat, starlist_mat, transform, errs='both'):
    """
    calculate the chi2 and reduced chi2 of the position 
    between two matched starlists.
    Input:
    ref_mat: astropy table
        Reference starlist only containing matched stars that were used in the
        transformation. Standard column headers are assumed.
        
    starlist_mat: astropy table
        Transformed starlist only containing the matched stars used in
        the transformation. Standard column headers are assumed.

    transform: transformation object
        Transformation object of final transform. Used in chi-square
        determination

    errs: string; 'both', 'reference', or 'starlist'
        If both, add starlist errors in quadrature with reference errors.

        If reference, only consider reference errors. This should be used if the starlist
        does not have valid errors

        If starlist, only consider starlist errors. This should be used if the reference
        does not have valid errors

    Output:
    chi_sq: float
        chi2 = sum (diff_x**2 / xerr**2 + diff_y**2 /yerr**2)
    chi_sq_red: float
        reduced chi2 = chi2/ degree of freedom
    deg_freedom: int
        degree of freedom

    """
    diff_x = ref_mat['x'] - starlist_mat['x']
    diff_y = ref_mat['y'] - starlist_mat['y']

    # Set errors as per user input
    if errs == 'both':
        xerr = np.hypot(ref_mat['xe'], starlist_mat['xe'])
        yerr = np.hypot(ref_mat['ye'], starlist_mat['ye'])
    elif errs == 'reference':
        xerr = ref_mat['xe']
        yerr = ref_mat['ye']
    elif errs == 'starlist':
        xerr = starlist_mat['xe']
        yerr = starlist_mat['ye']
          

    # For both X and Y, calculate chi-square. Combine arrays to get combined
    # chi-square
    chi_sq_x = diff_x**2. / xerr**2.
    chi_sq_y = diff_y**2. / yerr**2.

    chi_sq = np.append(chi_sq_x, chi_sq_y)
    
    # Calculate degrees of freedom in transformation
    num_mod_params = calc_nparam(transform)
    deg_freedom = len(chi_sq) - num_mod_params
    
    # Calculate reduced chi-square
    chi_sq = np.sum(chi_sq)
    chi_sq_red = chi_sq / deg_freedom

    return chi_sq, chi_sq_red, deg_freedom


def calc_nparam(transformation):
    """
    calculate the degree of freedom for a transformation
    """
    # Read transformation: Extract X, Y coefficients from transform
    if transformation.__class__.__name__ == 'four_paramNW':
        nparam = 4
    elif transformation.__class__.__name__ == 'PolyTransform':
        order = transformation.order
        nparam = (order+1) * (order+2) 
    return nparam

def calc_F(red_chi2_1, red_chi2_2, v1, v2):
    """
    compare two different models to get the proper polynomial fitting order

    Input:
    red_chi2_1: reduced chi2 for the first model
    red_chi2_2: reduced chi2 for the second model
    v1 = degree of freedom for the first model
       = 2*(N_star_matched) - model_parameters
    v2 = degree of freedom for the second mdoel

    Output:
    P: The probability that the first model is better

    Example:
    for 1st order polynomial fitting:
        x' = a0 + a1*x + a2*y
        y' = b0 + b1*x + b2*y
        v1 = 2*N1 - 2*3 (2*: because x and y direction) 
        red_chi2_1 = chi2/v1
    for 2nd order polynomial fitting:
        x' = a0 + a1*x + a2*y + a3*x**2 + a4*y**2 + a5*x*y
        y' = b0 + b1*x + b2*y + b3*x**2 + b4*y**2 + b5*x*y
        v1 = 2*N1 - 2*6 
        red_chi2_2 = chi2/v2
    calc_F(red_chi2_1, red_chi2_2, v1, v2)
    
    ***Note***
    * make sure the first model is the simple model 
      and the second model is the more complicated model
    * the return value represents the probability that 
      the first model is better than the second model, in other words,
      the small P means the more colicated model is needed.
      the large P means the simple model is good enough.
    * normally, the P value will increase from model1->model2, to 
      model2->model3, to model3->model4. The user can decide a 
      critical value (eg, 0.7) to find the proper model.
    """

    f_value = red_chi2_1/red_chi2_2
    p = 1-f.cdf(f_value, v1, v2)
    return p
