import pylab as py
import numpy as np
import matplotlib.mlab as mlab
import pdb

####################################################
# Code for making diagnostic plots for astrometry
# alignment
####################################################


def trans_positions(ref, ref_mat, starlist, starlist_mat, xlim=None, ylim=None):
    """
    Plot positions of stars in reference list and the transformed starlist,
    in reference list coordinates. Stars used in the transformation are
    highlighted.
s
    Parameters:
    ----------
    ref: astropy table
        Reference starlist, with standard column headers

    ref_mat: astropy table
        Reference starlist only containing matched stars that were used in the
        transformation. Standard column headers are assumed.

    starlist: astropy table
        Transformed starist with the reference starlist coordinates.
        Standard column headers are assumed

    starlist_mat: astropy table
        Transformed starlist only containing the matched stars used in
        the transformation. Standard column headers are assumed.

    xlim: None or list/array [xmin, xmax]
        If not None, sets the xmin and xmax limit of the plot

    ylim: None or list/array [ymin, ymax]
        If not None, sets the ymin and ymax limit of the plot    
    
    """
    py.figure(figsize=(10,10))
    py.clf()
    py.plot(ref['x'], ref['y'], 'g.', ms=5, label='Reference')
    py.plot(starlist['x_trans'], starlist['y_trans'], 'r.', ms=5, label='Label.dat')
    py.plot(ref_mat['x'], ref_mat['y'], color='skyblue', marker='s', ms=10, linestyle='None', label='Matched Reference')
    py.plot(starlist_mat['x_trans'], starlist_mat['y_trans'], color='darkblue', marker='s', ms=5, linestyle='None', label='Matched label.dat')
    py.xlabel('X position (Reference Coords)')
    py.ylabel('Y position (Reference Coords)')
    py.legend(numpoints=1)
    py.title('Label.dat Positions After Transformation')
    if xlim != None:
        py.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
    py.savefig('Transformed_positions.png')

    return


def pos_diff_hist(ref_mat, starlist_mat, nbins=25, bin_width=None, xlim=None):
    """
    Plot histogram of position differences for the matched
    stars: reference - starlist

    Parameters:
    -----------
    ref_mat: astropy table
        Reference starlist only containing matched stars that were used in the
        transformation. Standard column headers are assumed.
        
    starlist_mat: astropy table
        Transformed starlist only containing the matched stars used in
        the transformation. Standard column headers are assumed. 

    nbins: int
        Number of bins used in histogram, regardless of data range. This is
        ignored if bin_width != None

    bin_width: None or float
        If float, sets the width of the bins used in the histogram. Will override
        nbins

    xlim: None or [xmin, xmax]
         If not none, set the X range of the plot
        
    """
    diff_x = ref_mat['x'] - starlist_mat['x_trans']
    diff_y = ref_mat['y'] - starlist_mat['y_trans']

    # Set the binning as per user input
    bins = nbins
    if bin_width != None:
        min_range = min([min(diff_x), min(diff_y)])
        max_range = max([max(diff_x), max(diff_y)])

        bins = np.arange(min_range, max_range+bin_width, bin_width)
    
    py.figure(figsize=(10,10))
    py.clf()
    py.hist(diff_x, histtype='step', bins=bins, color='blue', label='X')
    py.hist(diff_y, histtype='step', bins=bins, color='red', label='Y')
    py.xlabel('Reference Position - label.dat Position (reference coords)')
    py.ylabel('N stars')
    py.title('Position Differences for matched stars')
    if xlim != None:
        py.xlim([xlim[0], xlim[1]])
    py.legend()
    py.savefig('Positions_hist.png')

    return

def pos_diff_err_hist(ref_mat, starlist_mat, nbins=25, bin_width=None, errs='both', xlim=None):
    """
    Plot histogram of position residuals / astrometric error for the matched
    stars: reference - starlist.

    Computes reduced chi-squared between this distribution nad a Gaussian
    distribution, which ideally would be a match.

    Parameters:
    -----------
    ref_mat: astropy table
        Reference starlist only containing matched stars that were used in the
        transformation. Standard column headers are assumed.
        
    starlist_mat: astropy table
        Transformed starlist only containing the matched stars used in
        the transformation. Standard column headers are assumed. 

    nbins: int
        Number of bins used in histogram, regardless of data range. This is
        ignored if bin_width != None

    bin_width: None or float
        If float, sets the width of the bins used in the histogram. Will override
        nbins

    errs: string; 'both', 'reference', or 'starlist'
        If both, add starlist errors in quadrature with reference errors.

        If reference, only consider reference errors. This should be used if the starlist
        does not have valid errors

        If starlist, only consider starlist errors. This should be used if the reference
        does not have valid errors

    xlim: None or [xmin, xmax] (default=None)
        If not None, set the min and max value of the X axis
        
    """
    diff_x = ref_mat['x'] - starlist_mat['x_trans']
    diff_y = ref_mat['y'] - starlist_mat['y_trans']

    # Set errors as per user input
    if errs == 'both':
        xerr = np.hypot(ref_mat['xe'], starlist_mat['xe_trans'])
        yerr = np.hypot(ref_mat['ye'], starlist_mat['ye_trans'])
    elif errs == 'reference':
        xerr = ref_mat['xe']
        yerr = ref_mat['ye']
    elif errs == 'starlist':
        xerr = starlist_mat['xe_trans']
        yerr = starlist_mat['ye_trans']
          
    # Calculate ratio between differences and the combined error. This is
    # what we will plot
    ratio_x = diff_x / xerr
    ratio_y = diff_y / yerr
    
    # Set the binning as per user input
    bins = nbins
    if bin_width != None:
        min_range = min([min(ratio_x), min(ratio_y)])
        max_range = max([max(ratio_x), max(ratio_y)])

        bins = np.arange(min_range, max_range+bin_width, bin_width)
    
    py.figure(figsize=(10,10))
    py.clf()
    n_x, bins_x, p = py.hist(ratio_x, histtype='step', bins=bins, color='blue',
                             label='X', normed=True, linewidth=2)
    n_y, bins_y, p = py.hist(ratio_y, histtype='step', bins=bins, color='red',
                             label='Y', normed=True, linewidth=2)
    # Overplot a Gaussian, as well
    mean = 0
    sigma = 1
    x = np.arange(-6, 6, 0.1)
    py.plot(x, mlab.normpdf(x,mean,sigma), 'g-', linewidth=2)

    # For both distributions, calculate reduced chi-square when compared to
    # the Gaussian. Only consider bins where n > 0
    good_x = np.where(n_x > 0)
    good_y = np.where(n_y > 0)
    
    bincenterX = (bins_x[:-1] + bins_x[1:]) / 2.0
    bincenterY = (bins_y[:-1] + bins_y[1:]) / 2.0

    # We'll consider Poisson errors for each hist bin. Remember that the bins
    # have been normalized, so we have to multiply each value by the total number
    # of stars to get the stars in each bin...and then we have to convert the error
    # back to normalized units for the chi-square calculation
    binX_err = np.sqrt(n_x * len(ratio_x))
    binY_err = np.sqrt(n_y * len(ratio_y))

    binX_normErr = (binX_err / (n_x * len(ratio_x))) * n_x
    binY_normErr = (binY_err / (n_y * len(ratio_y))) * n_y

    gauss_x = mlab.normpdf(bincenterX, mean, sigma)
    gauss_y = mlab.normpdf(bincenterY, mean, sigma)
    
    chi_x = np.sum( (n_x[good_x] - gauss_x[good_x])**2. / binX_normErr[good_x]**2. )
    chi_y = np.sum( (n_y[good_y] - gauss_y[good_y])**2. / binY_normErr[good_y]**2. )

    # Will have len(n) - 2 degrees of freedom (since 2 parameters in Gaussian)
    chi_x_red = chi_x / (len(good_x[0]) - 2)
    chi_y_red = chi_y / (len(good_y[0]) - 2)

    # Annotate reduced chi-sqared values in plot
    xstr = '$\chi^2_r$ = {0}'.format(np.round(chi_x_red, decimals=3))
    ystr = r'$\chi^2_r$ = {0}'.format(np.round(chi_y_red, decimals=3))
    
    py.annotate(xstr, xy=(0.3, 0.8), xycoords='figure fraction', color='blue')
    py.annotate(ystr, xy=(0.3, 0.75), xycoords='figure fraction', color='red')

    py.xlabel('(Ref Pos - label.dat Pos) / Ast. Error')
    py.ylabel('N stars')
    py.title('Position Diff/Errs for matched stars')
    if xlim != None:
        py.xlim([xlim[0], xlim[1]])
    py.legend()
    py.savefig('Positions_err_ratio_hist.png')

    return


def mag_diff_hist(ref_mat, starlist_mat, bins=25):
    """
    Plot histogram of mag differences for the matched
    stars: reference - starlist

    Parameters:
    -----------
    ref_mat: astropy table
        Reference starlist only containing matched stars that were used in the
        transformation. Standard column headers are assumed.
        
    starlist_mat: astropy table
        Transformed starlist only containing the matched stars used in
        the transformation. Standard column headers are assumed.         

    """
    diff_m = ref_mat['m'] - starlist_mat['m']

    py.figure(figsize=(10,10))
    py.clf()
    py.hist(diff_m, bins=bins)
    py.xlabel('Reference Mag - label.dat Mag')
    py.ylabel('N stars')
    py.title('Magnitude Difference for matched stars')
    py.savefig('Magnitude_hist.png')

    return

def pos_diff_quiver(ref_mat, starlist_mat, qscale=10, keyLength=0.2, xlim=None, ylim=None,
                    outlier_reject=None):
    """
    Plot histogram of position differences for the matched
    stars: reference - starlist

    Parameters:
    -----------
    ref_mat: astropy table
        Reference starlist only containing matched stars that were used in the
        transformation. Standard column headers are assumed.
        
    starlist_mat: astropy table
        Transformed starlist only containing the matched stars used in
        the transformation. Standard column headers are assumed.

    qscale: int (default=10)
        Scale parameter for the quiver plot. Lower the number, bigger the scale

    keyLength: float (default=0.2)
        Key length parameter for quiver plot, in reference units.

    xlim: None or list/array [xmin, xmax]
        If not None, sets the xmin and xmax limit of the plot

    ylim: None or list/array [ymin, ymax]
        If not None, sets the ymin and ymax limit of the plot

    outlier_reject: None or float
        If float, ignore any star with a combined position difference larger than
        the float. difference = np.hypot(diff_x, diff_y). This value needs to be
        in reference units

    """
    diff_x = ref_mat['x'] - starlist_mat['x_trans']
    diff_y = ref_mat['y'] - starlist_mat['y_trans']

    # Add own reference quiver arrow to end of array, since actual one is broken
    # This will be in lower left portion of plot
    diff_x = np.array(diff_x)
    diff_y = np.array(diff_y)
    xpos = np.array(ref_mat['x'])
    ypos = np.array(ref_mat['y'])

    # Apply outlier_reject criteria, if desired
    if outlier_reject != None:
        difference = np.hypot(diff_x, diff_y)
        good = np.where(difference < outlier_reject)

        diff_x = diff_x[good]
        diff_y = diff_y[good]
        xpos = xpos[good]
        ypos = ypos[good]
        
        
    # Due to quiver silliness I need to add this twice
    xpos = np.append(xpos, max(xpos))
    xpos = np.append(xpos, max(xpos))
    ypos = np.append(ypos, min(ypos))
    ypos = np.append(ypos, min(ypos))
    diff_x = np.append(diff_x, keyLength*-1.0)
    diff_x = np.append(diff_x, keyLength*-1.0)
    diff_y = np.append(diff_y, 0)
    diff_y = np.append(diff_y, 0)

    s = len(xpos)
    
    py.figure(figsize=(10,10))
    py.clf()
    q = py.quiver(xpos, ypos, diff_x, diff_y, scale=qscale)
    fmt = '{0} ref units'.format(keyLength)
    py.quiverkey(q, 0.2, 0.92, keyLength, fmt, coordinates='figure', color='black')
    # Make our reference arrow a different color
    q2 = py.quiver(xpos[s-2:s], ypos[s-2:s], diff_x[s-2:s], diff_y[s-2:s], scale=qscale, color='red')
    # Annotate our reference quiver arrow
    py.annotate(fmt, xy=(xpos[-1]-2, ypos[-1]+0.5), color='red')
    py.xlabel('X Position (Reference coords)')
    py.ylabel('Y Position (Reference coords)')
    py.title('Reference - Transformed label.dat positions')
    if xlim != None:
        py.axis([xlim[0], ylim[1], ylim[0], ylim[1]])
    py.savefig('Positions_quiver.png')

    return

def vpd(ref, starlist_trans, vxlim, vylim):
    """
    Plot the VPD of the reference starlist and the transformed starlist.
    If all went well, both should be in the same frame.

    Note: we need velocities in both starlists in order for this
    to work.

    Parameters:
    ----------
    ref: astropy table
        Reference starlist which contains velocity info. Standard column
        names are assumed

    starlist_trans: astropy table
        Transformed starlist which also contains velocity info. Standard
        column names are assumed.

    vxlim: None or list/array [vxmin, vxmax]
        If not None, sets the vxmin and vxmax limit of the plot

    vylim: None or list/array [vymin, vymax]
        If not None, sets the vymin and vymax limit of the plot  
    """
    # Extract velocities
    ref_vx = ref['vx']
    ref_vy = ref['vy']
    trans_vx = starlist_trans['vx_trans']
    trans_vy = starlist_trans['vy_trans']

    py.figure(figsize=(10,10))
    py.clf()
    py.plot(trans_vx, trans_vy, 'k.', ms=8, label='Transformed', alpha=0.4)
    py.plot(ref_vx, ref_vy, 'r.', ms=8, label='Reference', alpha=0.4)
    py.xlabel('Vx (Reference units)')
    py.ylabel('Vy (Reference units)')
    if vxlim != None:
        py.axis([vxlim[0], vylim[1], vylim[0], vylim[1]])
    py.title('Reference and Transformed Proper Motions')
    py.legend()
    py.savefig('Transformed_velocities.png')

    return

def vel_hist(ref_mat, starlist_mat, nbins=25, bin_width=None, vxlim=None, vylim=None):
    """
    Plot the distributions of the velocity residuals in the reference list to
    the transformed starlist, realtive to the velocity errors. We assume that
    both lists have velocities and velocity errors

    Paramters:
    ----------
    ref_mat: astropy table
         Reference starlist, with velocities

    starlist_mat: astropy table
         Transformed starlist, with velocities

    nbins: int
        Number of bins used in histogram, regardless of data range. This is
        ignored if bin_width != None

    bin_width: None or float
        If float, sets the width of the bins used in the histograms. Will override
        nbins
        
    vxlim: None or [vx_min, vx_max]
        If not none, set the X axis of the Vx plot by defining the minimum
        and maximum values

    vylim: None or [vy_min, vy_max]
        If not none, set the Y axis of the Vy plot by defining the minimum
        and maximum values
    """
    # Will produce 2-panel plot: Vx resid and Vy resid
    diff_vx = ref_mat['vx'] - starlist_mat['vx_trans']
    diff_vy = ref_mat['vy'] - starlist_mat['vy_trans']
    
    vx_err = np.hypot(ref_mat['vxe'], starlist_mat['vxe_trans'])
    vy_err = np.hypot(ref_mat['vye'], starlist_mat['vye_trans'])

    ratio_vx = diff_vx / vx_err
    ratio_vy = diff_vy / vy_err

    # Set the binning as per user input
    xbins = nbins
    ybins = nbins
    if bin_width != None:
        xbins = np.arange(min(ratio_vx), max(ratio_vx)+bin_width, bin_width)
        ybins = np.arange(min(ratio_vy), max(ratio_vy)+bin_width, bin_width)

    # Parameters for a Gaussian to be overplotted on each
    mean = 0
    sigma = 1
    x = np.arange(-6, 6, 0.1)
        
    py.figure(figsize=(20,10))
    py.subplot(121)
    py.subplots_adjust(left=0.1)
    py.hist(ratio_vx, bins=xbins, histtype='step', color='black', normed=True,
            linewidth=2)
    py.plot(x, mlab.normpdf(x,mean,sigma), 'r-', linewidth=2)
    py.xlabel('(Ref Vx - Trans Vx) / Vxe')
    py.ylabel('N_stars')
    py.title('Vx Residuals, Matched')
    if vxlim != None:
        py.xlim([vxlim[0], vxlim[1]])
    py.subplot(122)
    py.hist(ratio_vy, bins=ybins, histtype='step', color='black', normed=True,
            linewidth=2)
    py.plot(x, mlab.normpdf(x,mean,sigma), 'r-', linewidth=2)
    py.xlabel('(Ref Vy - Trans Vy) / Vye')
    py.ylabel('N_stars')
    py.title('Vy Residuals, Matched')
    if vylim != None:
        py.xlim([vylim[0], vylim[1]])
    py.savefig('Vel_err_ratio_dist.png')

    return
