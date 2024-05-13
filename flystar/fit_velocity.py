from tqdm import tqdm
import numpy as np
import pandas as pd

def linear(x, k, b):
    return k*x + b

def linear_fit(x, y, sigma=None, absolute_sigma=True):
    """Weighted linear regression (See https://en.wikipedia.org/wiki/Weighted_least_squares#Solution). Recommended for low-dimension, non-degenerate data. Otherwise, please use scipy.curve_fit.

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data
    sigma : array-like, optional
        Weighted by 1/sigma**2. If not provided, weight = 1, by default None
    absolute_sigma : bool, optional
        If True (default), sigma is used in an absolute sense and the estimated parameter uncertainty reflects these absolute values. If False, only the relative magnitudes of the sigma values matter, by default True
    
    Returns
    -------
    result : dictionary
        Dictionary with keys 'slope', 'e_slope', 'intercept', 'e_intercept', and 'chi2' if return_chi2=True.
    """
    x = np.array(x)
    y = np.array(y)
    if sigma is None:
        sigma = np.ones_like(x)
    else:
        sigma = np.array(sigma)
    
    X = np.vander(x, 2)
    W = np.diag(1/sigma**2)
    XTWX = X.T @ W @ X
    pcov = np.linalg.inv(XTWX)  # Covariance Matrix
    popt = pcov @ X.T @ W @ y   # Linear Solution
    perr = np.sqrt(np.diag(pcov))   # Uncertainty of Linear Solution
    
    residual = y - X @ popt
    chi2 = residual.T @ W @ residual
    
    if not absolute_sigma:
        reduced_chi2 = chi2/(len(x) - 2)
        perr *= reduced_chi2**0.5
    
    result = {
        'slope': popt[0],
        'intercept': popt[1],
        'e_slope': perr[0],
        'e_intercept': perr[1],
        'chi2': chi2
    }
    
    return result


def calc_chi2(x, y, sigma, slope, intercept):
    popt = np.array([slope, intercept])
    X = np.vander(x, 2)
    W = np.diag(1/sigma**2)
    residual = y - X @ popt
    return residual.T @ W @ residual

    
def fit_velocity(startable, weighting='var', use_scipy=False, absolute_sigma=True, epoch_cols='all', art_star=False):
    """Fit proper motion with weighted linear regression equations (see https://en.wikipedia.org/wiki/Weighted_least_squares#Solution).
    Assumes that all data are valid.

    Parameters
    ----------
    startable : StarTable
        StarTable object
    weighting : str, optional
        Weighting by variance (1/ye**2) or standard deviation (1/ye), by default 'var'
    use_scipy : bool, optional
        Use scipy.curve_fit or flystar.fit_velocity.linear_fit, by default False
    epoch_cols : str or list of intergers, optional
        List of indicies of columns to use. If 'all', use all columns, by default 'all'
    art_star : bool, optional
        Artificial star catalog or not. If True, use startable['x'][:, epoch_ols, 1] as the location, by default False.
    
    Returns
    -------
    result : pd.DataFrame
        Proper motion dataframe with keys vx, vxe, vy, vye, x0, x0e, y0, y0e

    Raises
    ------
    ValueError
        If weighting is neither 'std' nor 'var'
    """
    if weighting not in ['std', 'var']:
        raise ValueError(f"Weighting must be either 'std' or 'var', not '{weighting}'.")
    if epoch_cols is None:
        epoch_cols = np.arange(len(startable.meta['YEARS'])) # use all cols if not specified
    
    N = len(startable)
    vx  = np.zeros(N)
    vy  = np.zeros(N)
    vxe = np.zeros(N)
    vye = np.zeros(N)
    x0  = np.zeros(N)
    y0  = np.zeros(N)
    x0e = np.zeros(N)
    y0e = np.zeros(N)
    chi2_vx = np.zeros(N)
    chi2_vy = np.zeros(N)
    t0 = np.zeros(N)

    time = np.array(startable.meta['YEARS'])[epoch_cols]
    
    if not art_star:
        x_arr = startable['x'][:, epoch_cols]
        y_arr = startable['y'][:, epoch_cols]
    else:
        x_arr = startable['x'][:, epoch_cols, 1]
        y_arr = startable['y'][:, epoch_cols, 1]
    
    xe_arr = startable['xe'][:, epoch_cols]
    ye_arr = startable['ye'][:, epoch_cols]
    
    if weighting=='std':
        sigma_x_arr = np.abs(xe_arr)**0.5
        sigma_y_arr = np.abs(ye_arr)**0.5
    elif weighting=='var':
        sigma_x_arr = xe_arr
        sigma_y_arr = ye_arr

    # For each star
    for i in tqdm(range(len(startable))):
        x = x_arr[i]
        y = y_arr[i]
        xe = xe_arr[i]
        ye = ye_arr[i]
        sigma_x = sigma_x_arr[i]
        sigma_y = sigma_y_arr[i]
        
        t_weight = 1. / np.hypot(xe, ye)
        t0[i] = np.average(time, weights=t_weight)
        dt = time - t0[i]
        
        if use_scipy:
            p0x = np.array([0., x.mean()])
            p0y = np.array([0., y.mean()])
            
            # Use scipy.curve_fit to fit for velocity
            vx_opt, vx_cov = curve_fit(linear, dt, x, p0=p0x, sigma=sigma_x, absolute_sigma=absolute_sigma)
            vy_opt, vy_cov = curve_fit(linear, dt, y, p0=p0y, sigma=sigma_y, absolute_sigma=absolute_sigma)
            
            vx[i] = vx_opt[0]
            vy[i] = vy_opt[0]
            x0[i] = vx_opt[1]
            y0[i] = vy_opt[1]
            vxe[i], x0e[i] = np.sqrt(vx_cov.diagonal())
            vye[i], y0e[i] = np.sqrt(vy_cov.diagonal())
            chi2_vx[i] = calc_chi2(dt, x, sigma_x, *vx_opt)
            chi2_vy[i] = calc_chi2(dt, y, sigma_y, *vy_opt)
        
        else:
            vx_result = linear_fit(dt, x, sigma=sigma_x, absolute_sigma=absolute_sigma)
            vy_result = linear_fit(dt, y, sigma=sigma_y, absolute_sigma=absolute_sigma)
            
            vx[i]   = vx_result['slope']
            vxe[i]  = vx_result['e_slope']
            x0[i]   = vx_result['intercept']
            x0e[i]  = vx_result['e_intercept']
            chi2_vx[i]  = vx_result['chi2']
            
            vy[i]   = vy_result['slope']
            vye[i]  = vy_result['e_slope']
            y0[i]   = vy_result['intercept']
            y0e[i]  = vy_result['e_intercept']
            chi2_vy[i] = vy_result['chi2']
    
    result = pd.DataFrame({
        'vx': vx,   'vy': vy, 
        'vxe': vxe, 'vye': vye, 
        'x0': x0,   'y0': y0, 
        'x0e': x0e, 'y0e': y0e, 
        'chi2_vx': chi2_vx, 
        'chi2_vy': chi2_vy, 
        't0': t0
    })
    return result


# Test
if __name__=='__main__':
    from scipy.optimize import curve_fit
    
    x = np.array([1,2,3,4])
    y = np.array([1,2,5,6])
    sigma = np.array([.4,.2,.1,.3])
        
    for absolute_sigma in [True, False]:
        result = linear_fit(x, y, sigma=sigma, absolute_sigma=absolute_sigma)
        popt, pcov = curve_fit(linear, x, y, sigma=sigma, absolute_sigma=absolute_sigma)
        perr = np.sqrt(np.diag(pcov))
        print(f'Absolute Sigma = {absolute_sigma}')
        print(f"linear_fit: slope = {result['slope']:.3f} ± {result['e_slope']:.3f}, intercept = {result['intercept']:.3f} ± {result['e_intercept']:.3f}, chi2={result['chi2']:.3f}")
        print(f'curve_fit:  slope = {popt[0]:.3f} ± {perr[0]:.3f}, intercept = {popt[1]:.3f} ± {perr[1]:.3f}, chi2={calc_chi2(x, y, sigma, *popt):.3f}\n')