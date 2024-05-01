import tqdm
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

    
def fit_velocity(startable, weighting='var', absolute_sigma=True, epoch_cols=None):
    """Fit proper motion with weighted linear regression equations (see https://en.wikipedia.org/wiki/Weighted_least_squares#Solution).

    Parameters
    ----------
    startable : StarTable
        StarTable object
    weighting : str, optional
        Weighting by variance (1/ye**2) or standard deviation (1/ye), by default 'var'
    epoch_cols : list, optional
        Choose which columns to use. If None, use all columns, by default None

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
    x0  = np.zeros(N)
    x0e = np.zeros(N)
    y0  = np.zeros(N)
    y0e = np.zeros(N)
    
    vx  = np.zeros(N)
    vxe = np.zeros(N)
    vy  = np.zeros(N)
    vye = np.zeros(N)

    time = np.array(startable.meta['YEARS'])[epoch_cols]
    
    for i in tqdm(range(len(startable))):
        x = startable['x'][i, epoch_cols]
        y = startable['y'][i, epoch_cols]
        xe = startable['xe'][i, epoch_cols]
        ye = startable['ye'][i, epoch_cols]
        
        t_weight = 1. / np.hypot(xe, ye)
        t0 = np.average(time, weights=t_weight)
        dt = time - t0
        
        if weighting == 'var':
            vx_result = linear_fit(dt, x, xe, absolute_sigma=absolute_sigma)
        elif weighting == 'std':
            vx_result = linear_fit(dt, x, np.abs(xe)**0.5)
        vx[i]   = vx_result['slope']
        vxe[i]  = vx_result['e_slope']
        x0[i]   = vx_result['intercept']
        x0e[i]  = vx_result['e_intercept']
        
        if weighting == 'var':
            vy_result = linear_fit(dt, y, ye)
        elif weighting == 'std':
            vy_result = linear_fit(dt, y, np.abs(ye)**0.5, absolute_sigma=absolute_sigma)
        vy[i]   = vy_result['slope']
        vye[i]  = vy_result['e_slope']
        y0[i]   = vy_result['intercept']
        y0e[i]  = vy_result['e_intercept']
    
    result = pd.DataFrame({
        'vx': vx, 'vxe': vxe,
        'vy': vy, 'vye': vye,
        'x0': x0, 'x0e': x0e,
        'y0': y0, 'y0e': y0e
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