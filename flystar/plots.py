from . import analysis
import pylab as py
import pylab as plt
import numpy as np
import matplotlib.mlab as mlab
from matplotlib import colors
import matplotlib.cm as cm
from scipy.stats import chi2
from scipy.optimize import curve_fit
import pdb
import math
import astropy
from astropy.io import ascii

####################################################
# Code for making diagnostic plots for astrometry
# alignment
####################################################


def trans_positions(ref, ref_mat, starlist, starlist_mat, xlim=None, ylim=None, fileName=None, 
                    root='./'):
    """
    Plot positions of stars in reference list and the transformed starlist,
    in reference list coordinates. Stars used in the transformation are
    highlighted.

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
    py.plot(ref['x'], ref['y'], 'g+', ms=5, label='Reference')
    py.plot(starlist['x'], starlist['y'], 'rx', ms=5, label='starlist')
    py.plot(ref_mat['x'], ref_mat['y'], color='skyblue', marker='s', ms=10, alpha=0.3,
                linestyle='None', label='Matched Reference')
    py.plot(starlist_mat['x'], starlist_mat['y'], color='darkblue', marker='s', ms=5, alpha=0.3,
                linestyle='None', label='Matched starlist')
    py.xlabel('X position (Reference Coords)')
    py.ylabel('Y position (Reference Coords)')
    py.legend(numpoints=1)
    py.title('Label.dat Positions After Transformation')
    if xlim != None:
        py.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
    py.axis('equal')
    if fileName!=None:
        #py.savefig(root + fileName[3:8] + 'Transformed_positions_' + '.png')
        py.savefig(root + 'Transformed_positions_{0}'.format(fileName) + '.png')
    else:
        py.savefig(root + 'Transformed_positions.png')
    py.close()
    return


def pos_diff_hist(ref_mat, starlist_mat, nbins=25, bin_width=None, xlim=None, fileName=None, root='./'):
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
    diff_x = ref_mat['x'] - starlist_mat['x']
    diff_y = ref_mat['y'] - starlist_mat['y']

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
    py.xlabel('Reference Position - starlist Position')
    py.ylabel('N stars')
    py.title('Position Differences for matched stars')
    if xlim != None:
        py.xlim([xlim[0], xlim[1]])
    py.legend()
    if fileName != None:
        py.savefig(root + fileName[3:8] + 'Positions_hist_' + '.png')
    else:
        py.savefig(root + 'Positions_hist.png')

    py.close()
    return

def pos_diff_err_hist(ref_mat, starlist_mat, transform, nbins=25, bin_width=None, errs='both', xlim=None,
                      outlier=10, fileName=None, root='./'):
    """
    Plot histogram of position residuals / astrometric error for the matched
    stars: reference - starlist.

    Also calculates the reduced chi-square of the fit, annotates value on plot.
    Calculate this reduced chi-square both including and excluding outliers,
    which we identify as > +/- <outlier> sigma away from 0

    Parameters:
    -----------
    ref_mat: astropy table
        Reference starlist only containing matched stars that were used in the
        transformation. Standard column headers are assumed.
        
    starlist_mat: astropy table
        Transformed starlist only containing the matched stars used in
        the transformation. Standard column headers are assumed.

    transform: transformation object
        Transformation object of final transform. Used in chi-square
        determination

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

    outlier: float (default = 10)
        Defines how many sigma away from 0 a star must be in order to be considered
        an outlier. 
        
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
          
    # Calculate ratio between differences and the combined error. This is
    # what we will plot
    ratio_x = diff_x / xerr
    ratio_y = diff_y / yerr

    # Identify non-outliers, within +/- <outlier> sigma away from 0
    good = np.where( (np.abs(ratio_x) < outlier) & (np.abs(ratio_y) < outlier) )
    
    """
    # For both X and Y, calculate chi-square. Combine arrays to get combined
    # chi-square
    chi_sq_x = diff_x**2. / xerr**2.
    chi_sq_y = diff_y**2. / yerr**2.

    chi_sq = np.append(chi_sq_x, chi_sq_y)
    
    # Calculate degrees of freedom in transformation
    num_mod_params = calc_nparam(transform)
    deg_freedom = len(chi_sq) - num_mod_params
    
    # Calculate reduced chi-square
    chi_sq_red = np.sum(chi_sq) / deg_freedom
    """
    # Chi-square analysis for all stars, including outliers
    chi_sq, chi_sq_red, deg_freedom = analysis.calc_chi2(ref_mat, starlist_mat,
                                                        transform, errs=errs)
    # Chi-square analysis for only non-outlier stars
    chi_sq_good, chi_sq_red_good, deg_freedom_good = analysis.calc_chi2(ref_mat[good],
                                                                   starlist_mat[good],
                                                                   transform,
                                                                   errs=errs)
    
    num_mod_params = analysis.calc_nparam(transform)

    #-------------------------------------------#
    # Plotting
    #-------------------------------------------#
    
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
    
    # Annotate reduced chi-sqared values in plot: with outliers
    xstr = '$\chi^2_r$ = {0}'.format(np.round(chi_sq_red, decimals=3))
    py.annotate(xstr, xy=(0.3, 0.77), xycoords='figure fraction', color='black')
    txt = r'$\nu$ = 2*{0} - {1} = {2}'.format(len(diff_x), num_mod_params,
                                                 deg_freedom)
    py.annotate(txt, xy=(0.25,0.74), xycoords='figure fraction', color='black')
    xstr2 = 'With Outliers'
    xstr3 = '{0} with +/- {1}+ sigma'.format(len(ratio_x) - len(good[0]), outlier)
    py.annotate(xstr2, xy=(0.29, 0.83), xycoords='figure fraction', color='black')
    py.annotate(xstr3, xy=(0.25, 0.80), xycoords='figure fraction', color='black')
    
    # Annotate reduced chi-sqared values in plot: without outliers
    xstr = '$\chi^2_r$ = {0}'.format(np.round(chi_sq_red_good, decimals=3))
    py.annotate(xstr, xy=(0.7, 0.8), xycoords='figure fraction', color='black')
    txt = r'$\nu$ = 2*{0} - {1} = {2}'.format(len(good[0]), num_mod_params,
                                                 deg_freedom_good)
    py.annotate(txt, xy=(0.65,0.77), xycoords='figure fraction', color='black')
    xstr2 = 'Without Outliers'
    py.annotate(xstr2, xy=(0.67, 0.83), xycoords='figure fraction', color='black')
    
    py.xlabel('(Ref Pos - TransStarlist Pos) / Ast. Error')
    py.ylabel('N stars (normalized)')
    py.title('Position Residuals for Matched Stars')
    if xlim != None:
        py.xlim([xlim[0], xlim[1]])
    py.legend()
    if fileName != None:
        py.savefig(root + fileName[3:8] + 'Positions_err_ratio_hist_' + '.png')
    else:
        py.savefig(root + 'Positions_err_ratio_hist.png')

    py.close()
    return


def mag_diff_hist(ref_mat, starlist_mat, bins=25, fileName=None, root='./'):
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

    # Deal with nans so it doesn't crash the code. Set to -99
    bad = np.isnan(diff_m)
    bad2 = np.where(bad == True)
    diff_m = np.delete(diff_m, bad2)
 
    py.figure(figsize=(10,10))
    py.clf()
    py.hist(diff_m, bins=bins)
    py.xlabel('Reference Mag - TransStarlist Mag')
    py.ylabel('N stars')
    py.title('Magnitude Difference for matched stars')
    if fileName != None:
        py.savefig(root + fileName[3:8] + 'Magnitude_hist_' + '.png')
    else:
        py.savefig(root + 'Magnitude_hist.png')

    py.close()
    return

def pos_diff_quiver(ref_mat, starlist_mat, qscale=10, keyLength=0.2, xlim=None, ylim=None,
                    outlier_reject=None, sigma=False, fileName=None, root='./'):
    """
    Quiver plot of position differences for the matched
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

    sigma = boolean
        If true, plot position differences divided by reference position error,
        rather than just the position difference
    """
    diff_x = ref_mat['x'] - starlist_mat['x']
    diff_y = ref_mat['y'] - starlist_mat['y']

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
        

    # Divide differences by reference error, if desired
    if sigma:
        xerr = ref_mat['xe']
        yerr = ref_mat['ye']

        diff_x /= xerr
        diff_y /= yerr

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
    #py.quiverkey(q, 0.2, 0.92, keyLength, fmt, coordinates='figure', color='black')
    # Make our reference arrow a different color
    q2 = py.quiver(xpos[s-2:s], ypos[s-2:s], diff_x[s-2:s], diff_y[s-2:s], scale=qscale, color='red')
    # Annotate our reference quiver arrow
    py.annotate(fmt, xy=(xpos[-1]-2, ypos[-1]+0.5), color='red')
    py.xlabel('X Position (Reference coords)')
    py.ylabel('Y Position (Reference coords)')
    if xlim != None:
        py.axis([xlim[0], ylim[1], ylim[0], ylim[1]])
    if sigma:
        if fileName != None:
            py.title('(Reference - Transformed Starlist positions) / sigma')
            py.savefig(root + fileName[3:8] + 'Positions_quiver_sigma_' + '.png')
        else:
            py.title('(Reference - Transformed Starlist positions) / sigma')
            py.savefig(root + 'Positions_quiver_sigma.png')
    else:
        if fileName != None:
            py.title('Reference - Transformed Starlist positions')
            py.savefig(root + fileName[3:8] + 'Positions_quiver_' + '.png')
        else:
            py.title('Reference - Transformed Starlist positions')
            py.savefig(root + 'Positions_quiver.png')

    py.close()
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
    trans_vx = starlist_trans['vx']
    trans_vy = starlist_trans['vy']

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

def vel_diff_err_hist(ref_mat, starlist_mat, nbins=25, bin_width=None, vxlim=None, vylim=None):
    """
    Plot the distributions of the velocity differences between the reference list
    and the transformed starlist, realtive to the velocity errors. We assume that
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
    diff_vx = ref_mat['vx'] - starlist_mat['vx']
    diff_vy = ref_mat['vy'] - starlist_mat['vy']
    
    vx_err = np.hypot(ref_mat['vxe'], starlist_mat['vxe'])
    vy_err = np.hypot(ref_mat['vye'], starlist_mat['vye'])

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

def residual_vpd(ref_mat, starlist_trans_mat, pscale=None):
    """
    Make VPD diagram of the residuals between the reference proper motions
    and the transformed proper motions.

    Parameters:
    -----------
    ref_mat: astropy table
        Table with matched stars from the reference starlist. Assumes
        standard headers

    starlist_trans: astropy table
        Table with matched stars from the transformed starlist. Assumes
        standard headers

    pscale: None or float
        If float, convert all values to mas/yr using pscale as the plate scale.
        Assumes pscale is conversion from pixels to milliarcsecs

    Output:
    ------
    Creates (reference - transformed) VPD
    """
    # Calculate the residual
    diff_x = ref_mat['vx'] - starlist_trans_mat['vx']
    diff_y = ref_mat['vy'] - starlist_trans_mat['vy']

    # Error calculation depends on if we are converting to mas/yr
    if pscale != None:
        xerr_frac = np.hypot((ref_mat['vxe'] / ref_mat['vx']),
                             (starlist_trans_mat['vxe'] / starlist_trans_mat['vx']))
        yerr_frac = np.hypot((ref_mat['vye'] / ref_mat['vy']),
                             (starlist_trans_mat['vye'] / starlist_trans_mat['vy']))

        # Now apply the plate scale to convert to mas/yr
        diff_x *= pscale
        diff_y *= pscale
        xerr = diff_x * xerr_frac
        yerr = diff_y * yerr_frac
    else:
        xerr = np.hypot(ref_mat['vxe'], starlist_trans_mat['vxe'])
        yerr = np.hypot(ref_mat['vye'], starlist_trans_mat['vye'])

    # Plotting
    py.figure(figsize=(10,10))
    py.clf()
    py.errorbar(diff_x, diff_y, xerr=xerr, yerr=yerr, fmt='k.', ms=8, alpha=0.5)
    if pscale != None:
        py.xlabel('Reference_vx - Transformed_vx (mas/yr)')
        py.ylabel('Reference_vy - Transformed_vy (mas/yr)')
    else:
        py.xlabel('Reference_vx - Transformed_vx (reference coords)')
        py.ylabel('Reference_vy - Transformed_vy (reference coords)')
    py.title('Proper Motion Residuals')
    py.savefig('resid_vpd.png')

    return


def plotStar(starNames, rootDir='./', align='align/align_d_rms_1000_abs_t',
             poly='polyfit_d/fit', points='points_d/', radial=False, NcolMax=3, figsize=(15,15)):

    print( 'Creating residuals plots for star(s):' )
    print( starNames )
    
    s = starset.StarSet(rootDir + align)
    s.loadPolyfit(rootDir + poly, accel=0, arcsec=0)
    Nstars = len(starNames)
    Ncols = 2 * np.min([Nstars, NcolMax])
    if Nstars <= Ncols/2:
        Nrows = 3
    else:
        Nrows = math.ceil(Nstars / (Ncols / 2)) * 3

    py.close('all')
    py.figure(2, figsize=figsize)
    names = s.getArray('name')
    mag = s.getArray('mag')
    x = s.getArray('x')
    y = s.getArray('y')
    r = np.hypot(x,y)
    
    for i in range(Nstars):
    
        starName = starNames[i]
        
        ii = names.index(starName)
        star = s.stars[ii]

        pointsTab = Table.read(rootDir + points + starName + '.points', format='ascii')

        time = pointsTab[pointsTab.colnames[0]]
        x = pointsTab[pointsTab.colnames[1]]
        y = pointsTab[pointsTab.colnames[2]]
        xerr = pointsTab[pointsTab.colnames[3]]
        yerr = pointsTab[pointsTab.colnames[4]]

        fitx = star.fitXv
        fity = star.fitYv
        dt = time - fitx.t0
        fitLineX = fitx.p + (fitx.v * dt)
        fitSigX = np.sqrt( fitx.perr**2 + (dt * fitx.verr)**2 )

        fitLineY = fity.p + (fity.v * dt)
        fitSigY = np.sqrt( fity.perr**2 + (dt * fity.verr)**2 )

        if (radial == True):
            # Lets also do radial/tangential
            x0 = fitx.p
            y0 = fity.p
            vx = fitx.v
            vy = fity.v
            x0e = fitx.perr
            y0e = fity.perr
            vxe = fitx.verr
            vye = fity.verr

            r0 = np.sqrt(x0**2 + y0**2)

            vr = ((vx*x0) + (vy*y0)) / r0
            vt = ((vx*y0) - (vy*x0)) / r0
            vre =  (vxe*x0/r0)**2 + (vye*y0/r0)**2
            vre += (y0*x0e*vt/r0**2)**2 + (x0*y0e*vt/r0**2)**2
            vre =  np.sqrt(vre)
            vte =  (vxe*y0/r0)**2 + (vye*x0/r0)**2
            vte += (y0*x0e*vr/r0**2)**2 + (x0*y0e*vr/r0**2)**2
            vte =  np.sqrt(vte)

            r = ((x*x0) + (y*y0)) / r0
            t = ((x*y0) - (y*x0)) / r0
            rerr = (xerr*x0/r0)**2 + (yerr*y0/r0)**2
            rerr += (y0*x0e*t/r0**2)**2 + (x0*y0e*t/r0**2)**2
            rerr =  np.sqrt(rerr)
            terr =  (xerr*y0/r0)**2 + (yerr*x0/r0)**2
            terr += (y0*x0e*r/r0**2)**2 + (x0*y0e*r/r0**2)**2
            terr =  np.sqrt(terr)

            fitLineR = ((fitLineX*x0) + (fitLineY*y0)) / r0
            fitLineT = ((fitLineX*y0) - (fitLineY*x0)) / r0
            fitSigR = ((fitSigX*x0) + (fitSigY*y0)) / r0
            fitSigT = ((fitSigX*y0) - (fitSigY*x0)) / r0

            diffR = r - fitLineR
            diffT = t - fitLineT
            sigR = diffR / rerr
            sigT = diffT / terr

            idxR = np.where(abs(sigR) > 4)
            idxT = np.where(abs(sigT) > 4)


        diffX = x - fitLineX
        diffY = y - fitLineY
        diff = np.hypot(diffX, diffY)
        rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
        sigX = diffX / xerr
        sigY = diffY / yerr
        sig = diff / rerr


        # Determine if there are points that are more than 5 sigma off
        idxX = np.where(abs(sigX) > 4)
        idxY = np.where(abs(sigY) > 4)
        idx = np.where(abs(sig) > 4)

        print( 'Star:        ', starName )
        print( '\tX Chi^2 = %5.2f (%6.2f for %2d dof)' % 
              (fitx.chi2red, fitx.chi2, fitx.dof))
        print( '\tY Chi^2 = %5.2f (%6.2f for %2d dof)' % 
              (fity.chi2red, fity.chi2, fity.dof))
        # print( 'X  Outliers: ', time[idxX] )
        # print( 'Y  Outliers: ', time[idxY] )
        # if (radial):
        #     print( 'R  Outliers: ', time[idxX] )
        #     print( 'T  Outliers: ', time[idxY] )
        # print( 'XY Outliers: ', time[idx] )

        # close(2)
        #             figure(2, figsize=(7, 8))
        #             clf()

        t0 = int(np.floor(np.min(time)))
        tO = int(np.ceil(np.max(time)))
        
        dateTicLoc = py.MultipleLocator(3)
        dateTicRng = [t0-1, tO+1]
        dateTics = np.arange(t0, tO+1)
        DateTicsLabel = dateTics-2000

        # See if we are using MJD instead.
        if time[0] > 50000:
            print('MJD')
            dateTicLoc = py.MultipleLocator(1000)
            t0 = int(np.round(np.min(time), 50))
            tO = int(np.round(np.max(time), 50))
            dateTicRng = [t0-200, tO+200]
            dateTics = np.arange(dateTicRng[0], dateTicRng[-1]+500, 1000)
            DateTicsLabel = dateTics


        maxErr = np.array([xerr, yerr]).max()
        resTicRng = [-1.1*maxErr, 1.1*maxErr]

        from matplotlib.ticker import FormatStrFormatter
        fmtX = FormatStrFormatter('%5i')
        fmtY = FormatStrFormatter('%6.2f')
        fontsize1 = 10

        if i < (Ncols/2):
            col = (2*i)+1
            row = 1
        else:
            col = 1 + 2*(i % (Ncols/2))
            row = 1 + 3*(i//(Ncols/2)) 
            
        ind = (row-1)*Ncols + col

        paxes = py.subplot(Nrows, Ncols, ind)
        py.plot(time, fitLineX, 'b-')
        py.plot(time, fitLineX + fitSigX, 'b--')
        py.plot(time, fitLineX - fitSigX, 'b--')
        py.errorbar(time, x, yerr=xerr, fmt='k.')
        rng = py.axis()
        py.ylim(np.min(x-xerr-0.1),np.max(x+xerr+0.1)) 
        py.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            py.xlabel('Date (MJD)', fontsize=fontsize1)
        py.ylabel('X (pix)', fontsize=fontsize1)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        py.yticks(np.arange(np.min(x-xerr-0.1), np.max(x+xerr+0.1), 0.2))
        py.xticks(dateTics, DateTicsLabel)
        py.xlim(np.min(dateTics), np.max(dateTics))
        py.annotate(starName,xy=(1.0,1.1), xycoords='axes fraction', fontsize=12, color='red')


        col = col + 1
        ind = (row-1)*Ncols + col

        paxes = py.subplot(Nrows, Ncols, ind)
        py.plot(time, fitLineY, 'b-')
        py.plot(time, fitLineY + fitSigY, 'b--')
        py.plot(time, fitLineY - fitSigY, 'b--')
        py.errorbar(time, y, yerr=yerr, fmt='k.')
        rng = py.axis()
        py.axis(dateTicRng + [rng[2], rng[3]], fontsize=fontsize1)
        py.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            py.xlabel('Date (MJD)', fontsize=fontsize1)
        py.ylabel('Y (pix)', fontsize=fontsize1)
        #paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.yaxis.set_major_formatter(fmtY)
        paxes.tick_params(axis='both', which='major', labelsize=12)
        py.ylim(np.min(y-yerr-0.1),np.max(y+yerr+0.1))
        py.yticks(np.arange(np.min(y-yerr-0.1), np.max(y+yerr+0.1), 0.2))
        py.xticks(dateTics, DateTicsLabel)
        py.xlim(np.min(dateTics), np.max(dateTics))

        row = row + 1
        col = col - 1
        ind = (row-1)*Ncols + col

        paxes = py.subplot(Nrows, Ncols, ind)
        py.plot(time, np.zeros(len(time)), 'b-')
        py.plot(time, fitSigX, 'b--')
        py.plot(time, -fitSigX, 'b--')
        py.errorbar(time, x - fitLineX, yerr=xerr, fmt='k.')
        py.axis(dateTicRng + resTicRng, fontsize=fontsize1)
        py.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            py.xlabel('Date (MJD)', fontsize=fontsize1)
        py.ylabel('X Residuals (pix)', fontsize=fontsize1)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        py.xticks(dateTics, DateTicsLabel)
        py.xlim(np.min(dateTics), np.max(dateTics))

        col = col + 1
        ind = (row-1)*Ncols + col

        paxes = py.subplot(Nrows, Ncols, ind)
        py.plot(time, np.zeros(len(time)), 'b-')
        py.plot(time, fitSigY, 'b--')
        py.plot(time, -fitSigY, 'b--')
        py.errorbar(time, y - fitLineY, yerr=yerr, fmt='k.')
        py.axis(dateTicRng + resTicRng, fontsize=fontsize1)
        py.xlabel('Date -2000 (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            py.xlabel('Date (MJD)', fontsize=fontsize1)
        py.ylabel('Y Residuals (pix)', fontsize=fontsize1)
        paxes.get_xaxis().set_major_locator(dateTicLoc)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        py.xticks(dateTics, DateTicsLabel)
        py.xlim(np.min(dateTics), np.max(dateTics))

        row = row + 1
        col = col - 1
        ind = (row-1)*Ncols + col


        paxes = py.subplot(Nrows, Ncols, ind)
        py.errorbar(x,y, xerr=xerr, yerr=yerr, fmt='k.')
        py.yticks(np.arange(np.min(y-yerr-0.1), np.max(y+yerr+0.1), 0.2))
        py.xticks(np.arange(np.min(x-xerr-0.1), np.max(x+xerr+0.1), 0.2), rotation = 270)
        py.axis('equal')
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        paxes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        paxes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        py.xlabel('X (pix)', fontsize=fontsize1)
        py.ylabel('Y (pix)', fontsize=fontsize1)
        py.plot(fitLineX, fitLineY, 'b-')    

        col = col + 1
        ind = (row-1)*Ncols + col

        bins = np.arange(-7.5, 7.5, 1)
        paxes = py.subplot(Nrows, Ncols, ind)
        id = np.where(diffY < 0)[0]
        sig[id] = -1.*sig[id] 
        (n, b, p) = py.hist(sigX, bins, histtype='stepfilled', color='b', label='X')
        py.setp(p, 'facecolor', 'b')
        (n, b, p) = py.hist(sigY, bins, histtype='step', color='r', label='Y')
        py.axis([-7, 7, 0, 8], fontsize=10)
        py.legend()
        py.xlabel('Residuals (sigma)', fontsize=fontsize1)
        py.ylabel('Number of Epochs', fontsize=fontsize1)

        ##########
        #
        # Also plot radial/tangential
        #
        ##########
        if (radial == True):
            py.clf()

            dateTicLoc = py.MultipleLocator(3)

            maxErr = np.array([rerr, terr]).max()
            resTicRng = [-3*maxErr, 3*maxErr]

            from matplotlib.ticker import FormatStrFormatter
            fmtX = FormatStrFormatter('%5i')
            fmtY = FormatStrFormatter('%6.2f')

            paxes = py.subplot(3,2,1)
            py.plot(time, fitLineR, 'b-')
            py.plot(time, fitLineR + fitSigR, 'b--')
            py.plot(time, fitLineR - fitSigR, 'b--')
            py.errorbar(time, r, yerr=rerr, fmt='k.')
            rng = py.axis()
            py.axis(dateTicRng + [rng[2], rng[3]])
            py.xlabel('Date (yrs)')
            py.ylabel('R (pix)')
            paxes.xaxis.set_major_formatter(fmtX)
            paxes.get_xaxis().set_major_locator(dateTicLoc)
            paxes.yaxis.set_major_formatter(fmtY)

            paxes = py.subplot(3, 2, 2)
            py.plot(time, fitLineT, 'b-')
            py.plot(time, fitLineT + fitSigT, 'b--')
            py.plot(time, fitLineT - fitSigT, 'b--')
            py.errorbar(time, t, yerr=terr, fmt='k.')
            rng = py.axis()
            py.axis(dateTicRng + [rng[2], rng[3]])
            py.xlabel('Date (yrs)')
            py.ylabel('T (pix)')
            paxes.xaxis.set_major_formatter(fmtX)
            paxes.get_xaxis().set_major_locator(dateTicLoc)
            paxes.yaxis.set_major_formatter(fmtY)

            paxes = py.subplot(3, 2, 3)
            py.plot(time, np.zeros(len(time)), 'b-')
            py.plot(time, fitSigR, 'b--')
            py.plot(time, -fitSigR, 'b--')
            py.errorbar(time, r - fitLineR, yerr=rerr, fmt='k.')
            py.axis(dateTicRng + resTicRng)
            py.xlabel('Date (yrs)')
            py.ylabel('R Residuals (pix)')
            paxes.get_xaxis().set_major_locator(dateTicLoc)

            paxes = py.subplot(3, 2, 4)
            py.plot(time, np.zeros(len(time)), 'b-')
            py.plot(time, fitSigT, 'b--')
            py.plot(time, -fitSigT, 'b--')
            py.errorbar(time, t - fitLineT, yerr=terr, fmt='k.')
            py.axis(dateTicRng + resTicRng)
            py.xlabel('Date (yrs)')
            py.ylabel('T Residuals (pix)')
            paxes.get_xaxis().set_major_locator(dateTicLoc)

            bins = np.arange(-7, 7, 1)
            py.subplot(3, 2, 5)
            (n, b, p) = py.hist(sigR, bins)
            py.setp(p, 'facecolor', 'k')
            py.axis([-5, 5, 0, 20])
            py.xlabel('T Residuals (sigma)')
            py.ylabel('Number of Epochs')

            py.subplot(3, 2, 6)
            (n, b, p) = py.hist(sigT, bins)
            py.axis([-5, 5, 0, 20])
            py.setp(p, 'facecolor', 'k')
            py.xlabel('Y Residuals (sigma)')
            py.ylabel('Number of Epochs')

            py.subplots_adjust(wspace=0.4, hspace=0.4, right=0.95, top=0.95)
            py.savefig(rootDir+'plots/plotStarRadial_' + starName + '.png')
            py.show()

    title = rootDir.split('/')[-2]
    py.suptitle(title, x=0.5, y=0.97)

    if Nstars == 1:
        py.subplots_adjust(wspace=0.4, hspace=0.4, left = 0.15, bottom = 0.1, right=0.9, top=0.9) 
        py.savefig(rootDir+'plots/plotStar_' + starName + '.png')
    else:
        py.subplots_adjust(wspace=0.6, hspace=0.6, left = 0.08, bottom = 0.05, right=0.95, top=0.90)
        py.savefig(rootDir+'plots/plotStar_all.png')
        py.show()

    py.show()
    print('Fubar')
        


##################################################
# New codes for velocity support in FlyStar and using
# the new StarTable and StarList format. 
##################################################

def plot_pm(tab):
    plt.figure(figsize=(6,6))
    plt.clf()
    plt.subplots_adjust(top=0.85)
    q = plt.quiver(tab['x0'], tab['y0'],
                   tab['vx']*1e3, tab['vy']*1e3,
                   scale=1e2, angles='xy')
    plt.quiverkey(q, 0.5, 0.8, 10, '10 mas/yr', color='red', 
                    coordinates='figure', labelpos='E')
    plt.xlabel(r'$\Delta \alpha$ (")')
    plt.ylabel(r'$\Delta \delta$ (")')

    return


def plot_gaia(gaia):
    ra_tan = gaia['ra']
    de_tan = gaia['dec']
    ra_tan_mean = ra_tan.mean()
    de_tan_mean = de_tan.mean()
    cos_dec = np.cos(np.radians(de_tan_mean))

    d_ra_tan = (ra_tan - ra_tan_mean) * cos_dec * 3600.0
    d_de_tan = (de_tan - de_tan_mean) * 3600.0
    
    pmra = gaia['pmra']
    pmdec = gaia['pmdec']
    
    plt.figure(figsize=(6,6))
    plt.clf()
    plt.subplots_adjust(top=0.85)
    q = plt.quiver(d_ra_tan, d_de_tan,
                   pmra, pmdec,
                   scale=1e2, angles='xy')
    plt.quiverkey(q, 0.5, 0.8, 10, '10 mas/yr', color='red', 
                    coordinates='figure', labelpos='E')
    plt.xlabel(r'$\Delta \alpha \cos \delta$ ('')')
    plt.ylabel(r'$\Delta \delta$ ('')')

    fmt = r'[$\alpha$, $\delta$] = [{0:8.3f}$^\circ$, {1:8.3f}$^\circ$]'
    plt.title(fmt.format(ra_tan_mean, de_tan_mean))
    plt.gca().invert_xaxis()
    

    return

def plot_pm_error(tab):
    plt.figure(figsize=(6,6))
    plt.clf()
    plt.semilogy(tab['m0'], tab['vxe']*1e3, 'r.', label=r'$\sigma_{\mu_{\alpha *}}$', alpha=0.4)
    plt.semilogy(tab['m0'], tab['vye']*1e3, 'b.', label=r'$\sigma_{\mu_{\delta}}$', alpha=0.4)
    plt.legend()
    plt.xlabel('Mag')
    plt.ylabel('PM Error (mas/yr)')

    return

def plot_mag_error(tab):
    plt.figure(figsize=(6,6))
    plt.clf()
    plt.semilogy(tab['m0'], tab['m0e'], 'r.', alpha=0.4)
    plt.legend()
    plt.xlabel('Mag')
    plt.ylabel('Mag Error (mag)')

    return

def plot_quiver_residuals_all_epochs(tab, unit='arcsec', scale=None, plotlim=None):

    # Keep track of the residuals for averaging.
    dr_good = np.zeros(len(tab), dtype=float)
    n_good = np.zeros(len(tab), dtype=int)
    dr_ref = np.zeros(len(tab), dtype=float)
    n_ref = np.zeros(len(tab), dtype=int)

    for ee in range(tab['x'].shape[1]):
        dt = tab['t'][:, ee] - tab['t0']
        xt_mod = tab['x0'] + tab['vx'] * dt
        yt_mod = tab['y0'] + tab['vy'] * dt
        
        good_idx = np.where(np.isfinite(tab['x'][:, ee]) == True)[0]
        ref_idx = np.where(tab[good_idx]['used_in_trans'][:, ee] == True)[0]

        dx, dy = plot_quiver_residuals(tab['x'][:, ee], tab['y'][:, ee], 
                                       xt_mod, yt_mod, 
                                       good_idx, ref_idx,
                                       'Epoch {0:d}'.format(ee), 
                                       unit=unit, scale=scale, plotlim=plotlim)

        # Building up average dr for a set of stars.
        dr = np.hypot(dx, dy)

        dr_good[good_idx] += dr[good_idx]
        dr_ref[good_idx[ref_idx]] += dr[good_idx[ref_idx]]

        n_good[good_idx] += 1
        n_ref[good_idx[ref_idx]] += 1

    dr_good_avg = np.zeros(len(tab), dtype=float)
    idx = np.where(n_good > 0)[0]
    dr_good_avg[idx] = dr_good[idx] / n_good[idx]
    
    dr_ref_avg = np.zeros(len(tab), dtype=float)
    idx = np.where(n_ref > 0)[0]
    dr_ref_avg[idx] = dr_ref[idx] / n_ref[idx]

    hdr = '{name:>16s} {mag:>5s} {dr:>6s} {x:>6s} {y:>6s} {r:>6s}'
    fmt = '{name:16s} {mag:5.2f} {dr:6.4f} {x:6.3f} {y:6.3f} {r:6.3f}'

    # print()
    # print('##########')
    # print('# GOOD stars')
    # print('##########')
    # for rr in range(len(tab)):
    #     if dr_good_avg[rr] > 0:
    #         print(fmt.format(name=tab['name'][rr], mag=tab['m0'][rr], dr=dr_good_avg[rr],
    #                          x=tab['x0'][rr], y=tab['y0'][rr], r=np.hypot(tab['x0'][rr], tab['y0'][rr])))

#    print()
#    print('##########')
#    print('# REF stars')
#    print('##########')
#    print(hdr.format(name='Name', mag='Mag', dr='dr', x='x', y='y', r='r'))
#    for rr in range(len(tab)):
#        if (dr_ref_avg[rr] > 0):
#            print(fmt.format(name=tab['name'][rr], mag=tab['m0'][rr], dr=dr_ref_avg[rr],
#                             x=tab['x0'][rr], y=tab['y0'][rr], r=np.hypot(tab['x0'][rr], tab['y0'][rr])))
            
    return


def plot_quiver_residuals_with_orig_all_epochs(tab, trans_list, unit='arcsec', scale=None, plotlim=None, scale_orig=None, cte_fit=None, mlim=15):

    # Keep track of the residuals for averaging.
    dr_good = np.zeros(len(tab), dtype=float)
    n_good = np.zeros(len(tab), dtype=int)
    dr_ref = np.zeros(len(tab), dtype=float)
    n_ref = np.zeros(len(tab), dtype=int)

    for ee in range(tab['x'].shape[1]):
        dt = tab['t'][:, ee] - tab['t0']
        xt_mod = tab['x0'] + tab['vx'] * dt
        yt_mod = tab['y0'] + tab['vy'] * dt
        
        good_idx = np.where(np.isfinite(tab['x'][:, ee]) == True)[0]
        ref_idx = np.where(tab[good_idx]['used_in_trans'][:, ee] == True)[0]

        da = calc_da(trans_list[ee])

        dx, dy = plot_quiver_residuals(tab['x'][:, ee], tab['y'][:, ee], 
                                       xt_mod, yt_mod, 
                                       good_idx, ref_idx,
                                       'Epoch {0:d}'.format(ee), 
                                       unit=unit, scale=scale, plotlim=plotlim)

        plot_quiver_residuals_orig(tab['x'][:, ee], tab['y'][:, ee], 
                                   xt_mod, yt_mod, 
                                   good_idx, ref_idx,
                                   tab['x_orig'][:, ee], tab['y_orig'][:, ee], da,
                                   'Epoch {0:d}'.format(ee), 
                                   scale=scale_orig, plotlim=plotlim)

        plot_mag_scatter(tab['m'][:, ee], 
                         tab['x'][:, ee], tab['y'][:, ee], 
                         tab['xe'][:, ee], tab['ye'][:, ee],
                         xt_mod, yt_mod, 
                         good_idx, ref_idx,
                         'Epoch {0:d}'.format(ee), da=da,
                         xorig=tab['x_orig'][:, ee], yorig=tab['y_orig'][:, ee],
                         cte_fit=cte_fit, mlim=mlim)

#        plot_quiver_residuals_orig_angle_xy(tab['x'][:, ee], tab['y'][:, ee],
#                                            xt_mod, yt_mod, 
#                                            good_idx, ref_idx,
#                                            tab['x_orig'][:, ee], tab['y_orig'][:, ee], da,
#                                            'Epoch {0:d}'.format(ee))
#
#        plot_quiver_residuals_vs_pos_err(dx, dy, good_idx, ref_idx,
#                                         1e3 * tab['xe'][:, ee], 1e3 * tab['ye'][:, ee], 
#                                         'positional err (mas)', 'Epoch {0:d}'.format(ee), da=da)
    
        # Building up average dr for a set of stars.
        dr = np.hypot(dx, dy)

        dr_good[good_idx] += dr[good_idx]
        dr_ref[good_idx[ref_idx]] += dr[good_idx[ref_idx]]

        n_good[good_idx] += 1
        n_ref[good_idx[ref_idx]] += 1

    dr_good_avg = np.zeros(len(tab), dtype=float)
    idx = np.where(n_good > 0)[0]
    dr_good_avg[idx] = dr_good[idx] / n_good[idx]
    
    dr_ref_avg = np.zeros(len(tab), dtype=float)
    idx = np.where(n_ref > 0)[0]
    dr_ref_avg[idx] = dr_ref[idx] / n_ref[idx]

    hdr = '{name:>16s} {mag:>5s} {dr:>6s} {x:>6s} {y:>6s} {r:>6s}'
    fmt = '{name:16s} {mag:5.2f} {dr:6.4f} {x:6.3f} {y:6.3f} {r:6.3f}'

    # print()
    # print('##########')
    # print('# GOOD stars')
    # print('##########')
    # for rr in range(len(tab)):
    #     if dr_good_avg[rr] > 0:
    #         print(fmt.format(name=tab['name'][rr], mag=tab['m0'][rr], dr=dr_good_avg[rr],
    #                          x=tab['x0'][rr], y=tab['y0'][rr], r=np.hypot(tab['x0'][rr], tab['y0'][rr])))

#    print()
#    print('##########')
#    print('# REF stars')
#    print('##########')
#    print(hdr.format(name='Name', mag='Mag', dr='dr', x='x', y='y', r='r'))
#    for rr in range(len(tab)):
#        if (dr_ref_avg[rr] > 0):
#            print(fmt.format(name=tab['name'][rr], mag=tab['m0'][rr], dr=dr_ref_avg[rr],
#                             x=tab['x0'][rr], y=tab['y0'][rr], r=np.hypot(tab['x0'][rr], tab['y0'][rr])))
            
    return



def angle_from_xy(x, y):
    """
    Given x and y, calculate the angle theta in degrees
    """
    angles = np.arctan2(y, x) * 180 / np.pi

    return angles


def calc_da(trans_list):
    """
    Rotation angle.
    """
    c01_idx = trans_list.px.param_names.index('c0_1')
    c10_idx = trans_list.px.param_names.index('c1_0')
    c01 = trans_list.px.parameters[c01_idx]
    c10 = trans_list.px.parameters[c10_idx]
    da = np.degrees(np.arctan2(-c01, c10))
    
    return da


def plot_mag_scatter(m_t, x_t, y_t, xe_t, ye_t, x_ref, y_ref, good_idx, ref_idx, title, da=0, xorig=None, yorig=None, cte_fit=None, mlim=15):
    # Residual
    dx = (x_t - x_ref)
    dy = (y_t - y_ref)
    
    # Magnitude
    mgood = m_t[good_idx]
    mref = m_t[good_idx][ref_idx]

    # Residual angle
    agood = angle_from_xy(dx[good_idx], dy[good_idx])
    aref = angle_from_xy(dx[good_idx][ref_idx], dy[good_idx][ref_idx])
    # Subtract off some angle IN DEGREES (e.g. if going from Gaia to HST camera frame)
    agood -= da
    aref -= da

    # Keep everything within 0 to 360
    agood = agood % 360
    aref = aref % 360

    # Residual magnitude
    rgood = np.hypot(dx[good_idx], dy[good_idx])
    rref = np.hypot(dx[good_idx][ref_idx], dy[good_idx][ref_idx])

    # Errors in x and y position
    xegood = xe_t[good_idx]
    xeref = xe_t[good_idx][ref_idx]
    yegood = ye_t[good_idx]
    yeref = ye_t[good_idx][ref_idx]

    ygood = np.sin(np.radians(agood)) * rgood
    yref = np.sin(np.radians(aref)) * rref

    xgood = np.cos(np.radians(agood)) * rgood
    xref = np.cos(np.radians(aref)) * rref

    fig, ax = plt.subplots(6, 1, figsize=(6,18), sharex=True, num=103)
#    plt.clf()
    plt.subplots_adjust(hspace=0.01)
    ax[0].scatter(mgood, agood, color='black', alpha=0.3, s=2)
    ax[0].scatter(mref, aref, color='red', alpha=0.3, s=2)
    ax[0].set_ylabel('Angle (deg)')

    ax[1].scatter(mgood, rgood, color='black', alpha=0.3, s=2)
    ax[1].scatter(mref, rref, color='red', alpha=0.3, s=2)
    ax[1].set_xlabel('mag')
    ax[1].set_ylabel('Modulus (arcsec)')
    ax[1].set_yscale('log')
    if type(rgood) == astropy.table.column.MaskedColumn:
        ax[1].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood.data, rref.data])))
    else:
        ax[1].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood, rref])))

    ax[2].scatter(mgood, xgood, color='black', alpha=0.3, s=2)
    ax[2].scatter(mref, xref, color='red', alpha=0.3, s=2)
    ax[2].set_xlabel('mag')
    ax[2].set_ylabel('Res, x (arcsec)')
    ax[2].set_ylim(-0.01, 0.01)
    ax[2].axhline(y=0)

    ax[3].scatter(mgood, ygood, color='black', alpha=0.3, s=2)
    ax[3].scatter(mref, yref, color='red', alpha=0.3, s=2)
    ax[3].set_xlabel('mag')
    ax[3].set_ylabel('Res, y (arcsec)')
    ax[3].set_ylim(-0.01, 0.01)
    ax[3].axhline(y=0)

    ax[4].scatter(mgood, xgood/xegood, color='black', alpha=0.3, s=2)
    ax[4].scatter(mref, xref/xeref, color='red', alpha=0.3, s=2)
    ax[4].set_xlabel('mag')
    ax[4].set_ylabel('Res/Pos Err, x')
    ax[4].axhline(y=0)

    ax[5].scatter(mgood, ygood/yegood, color='black', alpha=0.3, s=2)
    ax[5].scatter(mref, yref/yeref, color='red', alpha=0.3, s=2)
    ax[5].set_xlabel('mag')
    ax[5].set_ylabel('Res/Pos Err, y')
    ax[5].axhline(y=0)

    ax[0].set_title(title)
    plt.show()
    plt.pause(1)


def plot_y_scatter(m_t, x_t, y_t, xe_t, ye_t, x_ref, y_ref, good_idx, ref_idx, title, da=0, xorig=None, yorig=None, cte_fit=None, mlim=15):
    # Residual
    dx = (x_t - x_ref)
    dy = (y_t - y_ref)
    
    # Magnitude
    mgood = m_t[good_idx]
    mref = m_t[good_idx][ref_idx]

    # Residual angle
    agood = angle_from_xy(dx[good_idx], dy[good_idx])
    aref = angle_from_xy(dx[good_idx][ref_idx], dy[good_idx][ref_idx])
    # Subtract off some angle IN DEGREES (e.g. if going from Gaia to HST camera frame)
    agood -= da
    aref -= da

    # Keep everything within 0 to 360
    agood = agood % 360
    aref = aref % 360

    # Residual magnitude
    rgood = np.hypot(dx[good_idx], dy[good_idx])
    rref = np.hypot(dx[good_idx][ref_idx], dy[good_idx][ref_idx])

    # Errors in x and y position
    xegood = xe_t[good_idx]
    xeref = xe_t[good_idx][ref_idx]
    yegood = ye_t[good_idx]
    yeref = ye_t[good_idx][ref_idx]

    ygood = np.sin(np.radians(agood)) * rgood
    yref = np.sin(np.radians(aref)) * rref

    xgood = np.cos(np.radians(agood)) * rgood
    xref = np.cos(np.radians(aref)) * rref

    fig, ax = plt.subplots(6, 1, figsize=(6,18), sharex=True, num=103)
#    plt.clf()
    plt.subplots_adjust(hspace=0.01)
    ax[0].scatter(yorig[good_idx], agood, color='black', alpha=0.3, s=2)
    ax[0].scatter(yorig[good_idx][ref_idx], aref, color='red', alpha=0.3, s=2)
    ax[0].set_ylabel('Angle (deg)')

    ax[1].scatter(yorig[good_idx], rgood, color='black', alpha=0.3, s=2)
    ax[1].scatter(yorig[good_idx][ref_idx], rref, color='red', alpha=0.3, s=2)
    ax[1].set_xlabel('y orig(pix)')
    ax[1].set_ylabel('Modulus (arcsec)')
    ax[1].set_yscale('log')
    if type(rgood) == astropy.table.column.MaskedColumn:
        ax[1].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood.data, rref.data])))
    else:
        ax[1].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood, rref])))

    ax[2].scatter(yorig[good_idx], xgood, color='black', alpha=0.3, s=2)
    ax[2].scatter(yorig[good_idx][ref_idx], xref, color='red', alpha=0.3, s=2)
    ax[2].set_xlabel('y orig (pix)')
    ax[2].set_ylabel('Res, x (arcsec)')
    ax[2].set_ylim(-0.01, 0.01)
    ax[2].axhline(y=0)

    ax[3].scatter(yorig[good_idx], ygood, color='black', alpha=0.3, s=2)
    ax[3].scatter(yorig[good_idx][ref_idx], yref, color='red', alpha=0.3, s=2)
    ax[3].set_xlabel('y orig (pix)')
    ax[3].set_ylabel('Res, y (arcsec)')
    ax[3].set_ylim(-0.01, 0.01)
    ax[3].axhline(y=0)

    ax[4].scatter(yorig[good_idx], xgood/xegood, color='black', alpha=0.3, s=2)
    ax[4].scatter(yorig[good_idx][ref_idx], xref/xeref, color='red', alpha=0.3, s=2)
    ax[4].set_xlabel('y orig (pix)')
    ax[4].set_ylabel('Res/Pos Err, x')
    ax[4].axhline(y=0)

    ax[5].scatter(yorig[good_idx], ygood/yegood, color='black', alpha=0.3, s=2)
    ax[5].scatter(yorig[good_idx][ref_idx], yref/yeref, color='red', alpha=0.3, s=2)
    ax[5].set_xlabel('y orig (pix)')
    ax[5].set_ylabel('Res/Pos Err, y')
    ax[5].axhline(y=0)

    ax[0].set_title(title)
    plt.show()
    plt.pause(1)


##############################################
# New testing stuff below
##############################################

    #####
    # Fit the y-residuals and subtract them away.
    # Then remake the plots.
    #####
    # Don't fit the saturated stars
    # (Other ideas: 1. Sigma clip and don't fit the outliers on bright end
    # that pull fit around. 2. Fit bright and faint ends separately.
    # 3. Include weights?

    if cte_fit=='power':
        idx = np.where(mgood > mlim)[0]
        gpopt, gpcov = curve_fit(T_cte_y, mgood[idx], ygood[idx], maxfev=100000)
     
        marr = np.linspace(13, 24, 1000)
    
        # Corrected values
        ygood_new = ygood - T_cte_y(mgood, *gpopt)
        yref_new = yref - T_cte_y(mref, *gpopt)
    
        agood = angle_from_xy(xgood, ygood) % 360
        rgood = np.hypot(xgood, ygood)
        aref = angle_from_xy(xref, yref) % 360
        rref = np.hypot(xref, yref)
    
        agood_new = angle_from_xy(xgood, ygood_new) % 360
        rgood_new = np.hypot(xgood, ygood_new)
        aref_new = angle_from_xy(xref, yref_new) % 360
        rref_new = np.hypot(xref, yref_new)
    
        fig, ax = plt.subplots(4, 2, figsize=(12,12), sharex=True, sharey='row', num=105)
        plt.subplots_adjust(hspace=0.01, wspace=0.01)
        ax[0,0].scatter(mgood, ygood, color='black', alpha=0.3, s=2)
        ax[0,0].scatter(mref, yref, color='red', alpha=0.3, s=2)
        ax[0,0].set_ylabel('Res, y (arcsec)')
        ax[0,0].set_ylim(-0.01, 0.01)
        ax[0,0].axhline(y=0)
        ax[0,0].plot(marr, T_cte_y(marr, *gpopt), 'k-')
        ax[0,0].set_title('No correction')
    
        ax[0,1].scatter(mgood, ygood_new, color='black', alpha=0.3, s=2)
        ax[0,1].scatter(mref, yref_new, color='red', alpha=0.3, s=2)
        ax[0,1].set_ylim(-0.01, 0.01)
        ax[0,1].axhline(y=0)
        ax[0,1].set_title('Corrected')
    
        ax[1,0].scatter(mgood, ygood/yegood, color='black', alpha=0.3, s=2)
        ax[1,0].scatter(mref, yref/yeref, color='red', alpha=0.3, s=2)
        ax[1,0].set_ylabel('Res/Pos Err, y')
        ax[1,0].set_ylim(-10, 10)
        ax[1,0].axhline(y=0)
    
        ax[1,1].scatter(mgood, ygood_new/yegood, color='black', alpha=0.3, s=2)
        ax[1,1].scatter(mref, yref_new/yeref, color='red', alpha=0.3, s=2)
        ax[1,1].set_ylim(-10, 10)
        ax[1,1].axhline(y=0)
    
        ax[2,0].scatter(mgood, rgood, color='black', alpha=0.3, s=2)
        ax[2,0].scatter(mref, rref, color='red', alpha=0.3, s=2)
        ax[2,0].set_ylabel('Modulus (arcsec)')
        ax[2,0].set_yscale('log')
        if type(rgood) == astropy.table.column.MaskedColumn:
            ax[2,0].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood.data, rref.data])))
        else:
            ax[2,0].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood, rref])))
    
        ax[2,1].scatter(mgood, rgood_new, color='black', alpha=0.3, s=2)
        ax[2,1].scatter(mref, rref_new, color='red', alpha=0.3, s=2)
        ax[2,1].set_yscale('log')
        if type(rgood_new) == astropy.table.column.MaskedColumn:
            ax[2,1].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood_new.data, rref_new.data])))
        else:
            ax[2,1].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood_new, rref_new])))
    
        ax[3,0].scatter(mgood, agood, color='black', alpha=0.3, s=2)
        ax[3,0].scatter(mref, aref, color='red', alpha=0.3, s=2)
        ax[3,0].set_ylabel('Angle (deg)')
        ax[3,0].set_xlabel('mag')
    
        ax[3,1].scatter(mgood, agood_new, color='black', alpha=0.3, s=2)
        ax[3,1].scatter(mref, aref_new, color='red', alpha=0.3, s=2)
        ax[3,1].set_xlabel('mag')

    if cte_fit=='power_line':
        idx1 = np.where((mgood > 15) & (mgood < 18.5))[0]
        idx2 = np.where(mgood > 18.5)[0]

        idx1r = np.where((mref > 15) & (mref < 18.5))[0]
        idx2r = np.where(mref > 18.5)[0]

        gpopt1, gpcov1 = curve_fit(T_line, mgood[idx1], ygood[idx1], maxfev=100000)
        gpopt2, gpcov2 = curve_fit(T_cte_y, mgood[idx2], ygood[idx2], maxfev=100000)
     
        marr1 = np.linspace(13, 18.5, 1000)
        marr2 = np.linspace(18.5, 24, 1000)

        mgood1 = mgood[idx1]
        mgood2 = mgood[idx2]
        mref1 = mref[idx1r]
        mref2 = mref[idx2r]

        xgood1 = xgood[idx1]
        xgood2 = xgood[idx2]
        ygood1 = ygood[idx1]
        ygood2 = ygood[idx2]

        xref1 = xref[idx1r]
        xref2 = xref[idx2r]
        yref1 = yref[idx1r]
        yref2 = yref[idx2r]

        xegood1 = xegood[idx1]
        xegood2 = xegood[idx2]
        yegood1 = yegood[idx1]
        yegood2 = yegood[idx2]

        xeref1 = xeref[idx1r]
        xeref2 = xeref[idx2r]
        yeref1 = yeref[idx1r]
        yeref2 = yeref[idx2r]
    
        # Corrected values
        ygood_new1 = ygood1 - T_line(mgood1, *gpopt1)
        yref_new1 = yref1 - T_line(mref1, *gpopt1)
        ygood_new2 = ygood2 - T_cte_y(mgood2, *gpopt2)
        yref_new2 = yref2 - T_cte_y(mref2, *gpopt2)

        agood1 = angle_from_xy(xgood1, ygood1) % 360
        rgood1 = np.hypot(xgood1, ygood1)
        aref1 = angle_from_xy(xref1, yref1) % 360
        rref1 = np.hypot(xref1, yref1)

        agood_new1 = angle_from_xy(xgood1, ygood_new1) % 360
        rgood_new1 = np.hypot(xgood1, ygood_new1)
        aref_new1 = angle_from_xy(xref1, yref_new1) % 360
        rref_new1 = np.hypot(xref1, yref_new1)

        agood2 = angle_from_xy(xgood2, ygood2) % 360
        rgood2 = np.hypot(xgood2, ygood2)
        aref2 = angle_from_xy(xref2, yref2) % 360
        rref2 = np.hypot(xref2, yref2)

        agood_new2 = angle_from_xy(xgood2, ygood_new2) % 360
        rgood_new2 = np.hypot(xgood2, ygood_new2)
        aref_new2 = angle_from_xy(xref2, yref_new2) % 360
        rref_new2 = np.hypot(xref2, yref_new2)
    
        fig, ax = plt.subplots(4, 2, figsize=(12,12), sharex=True, sharey='row', num=105)
        plt.subplots_adjust(hspace=0.01, wspace=0.01)
        ax[0,0].scatter(mgood, ygood, color='black', alpha=0.3, s=2)
        ax[0,0].scatter(mref, yref, color='red', alpha=0.3, s=2)
        ax[0,0].set_ylabel('Res, y (arcsec)')
        ax[0,0].set_ylim(-0.01, 0.01)
        ax[0,0].axhline(y=0)
        ax[0,0].plot(marr1, T_line(marr1, *gpopt1), 'b-')
        ax[0,0].plot(marr2, T_cte_y(marr2, *gpopt2), 'b-')
        ax[0,0].set_title('No correction')
    
        ax[0,1].scatter(mgood1, ygood_new1, color='black', alpha=0.3, s=2)
        ax[0,1].scatter(mref1, yref_new1, color='red', alpha=0.3, s=2)
        ax[0,1].scatter(mgood2, ygood_new2, color='black', alpha=0.3, s=2)
        ax[0,1].scatter(mref2, yref_new2, color='red', alpha=0.3, s=2)
        ax[0,1].set_ylim(-0.01, 0.01)
        ax[0,1].axhline(y=0)
        ax[0,1].set_title('Corrected')
    
        ax[1,0].scatter(mgood, ygood/yegood, color='black', alpha=0.3, s=2)
        ax[1,0].scatter(mref, yref/yeref, color='red', alpha=0.3, s=2)
        ax[1,0].set_ylabel('Res/Pos Err, y')
        ax[1,0].set_ylim(-10, 10)
        ax[1,0].axhline(y=0)
    
        ax[1,1].scatter(mgood1, ygood_new1/yegood1, color='black', alpha=0.3, s=2)
        ax[1,1].scatter(mref1, yref_new1/yeref1, color='red', alpha=0.3, s=2)
        ax[1,1].scatter(mgood2, ygood_new2/yegood2, color='black', alpha=0.3, s=2)
        ax[1,1].scatter(mref2, yref_new2/yeref2, color='red', alpha=0.3, s=2)
        ax[1,1].set_ylim(-10, 10)
        ax[1,1].axhline(y=0)
    
        ax[2,0].scatter(mgood, rgood, color='black', alpha=0.3, s=2)
        ax[2,0].scatter(mref, rref, color='red', alpha=0.3, s=2)
        ax[2,0].set_ylabel('Modulus (arcsec)')
        ax[2,0].set_yscale('log')
        if type(rgood) == astropy.table.column.MaskedColumn:
            ax[2,0].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood.data, rref.data])))
        else:
            ax[2,0].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood, rref])))
    
        ax[2,1].scatter(mgood1, rgood_new1, color='black', alpha=0.3, s=2)
        ax[2,1].scatter(mref1, rref_new1, color='red', alpha=0.3, s=2)
        ax[2,1].scatter(mgood2, rgood_new2, color='black', alpha=0.3, s=2)
        ax[2,1].scatter(mref2, rref_new2, color='red', alpha=0.3, s=2)
        ax[2,1].set_yscale('log')
        if type(rgood_new2) == astropy.table.column.MaskedColumn:
            ax[2,1].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood_new.data2, rref_new.data2])))
        else:
            ax[2,1].set_ylim(1e-6, 1.1 * np.max(np.concatenate([rgood_new2, rref_new2])))
    
        ax[3,0].scatter(mgood, agood, color='black', alpha=0.3, s=2)
        ax[3,0].scatter(mref, aref, color='red', alpha=0.3, s=2)
        ax[3,0].set_ylabel('Angle (deg)')
        ax[3,0].set_xlabel('mag')
    
        ax[3,1].scatter(mgood1, agood_new1, color='black', alpha=0.3, s=2)
        ax[3,1].scatter(mref1, aref_new1, color='red', alpha=0.3, s=2)
        ax[3,1].scatter(mgood2, agood_new2, color='black', alpha=0.3, s=2)
        ax[3,1].scatter(mref2, aref_new2, color='red', alpha=0.3, s=2)
        ax[3,1].set_xlabel('mag')
    
def T_cte_y(m, A, m0, alpha, m1):
    base = m/m0

    return m1 + A * np.sign(base) * np.abs(base)**alpha

def T_line(m, a, b):
    return a + m*b


def plot_quiver_residuals(x_t, y_t, x_ref, y_ref, good_idx, ref_idx, title, 
                          unit='pixel', scale=None, plotlim=None):
    """
    unit : str
        'pixel' or 'arcsec'
        The pixel units of the input values. Note, if arcsec, then the values will be
        converted to milli-arcsec for plotting when appropriate. 

    scale : float
        The quiver scale. If none, then default units will be used appropriate to the unit. 

    plotlim : float (positive)
        Sets the size of the plotted figure. If None, then default is used.
        Otherwise plots figure of range [-plotlim, plotlim] x [-plotlim, plotlim].
    """
    dx = (x_t - x_ref)
    dy = (y_t - y_ref)

    if unit == 'pixel':
        if scale == None:
            quiv_scale = 5
        else:
            quiv_scale = scale
        quiv_label = '0.3 pix'
        quiv_label_val = 0.3
        unit2 = 'pix'
    else:
        dx *= 1e3
        dy *= 1e3
        if scale == None:
            quiv_scale = 50
        else:
            quiv_scale = scale
        quiv_label = '1 mas'
        quiv_label_val = 1.0
        unit2 = 'mas'


    plt.figure(101, figsize=(6,6))
    plt.clf()
    q = plt.quiver(x_ref[good_idx], y_ref[good_idx], dx[good_idx], dy[good_idx],
                   color='black', scale=quiv_scale, angles='xy', alpha=0.5)
    plt.quiver(x_ref[good_idx][ref_idx], y_ref[good_idx][ref_idx], dx[good_idx][ref_idx], dy[good_idx][ref_idx],
               color='red', scale=quiv_scale, angles='xy')
    plt.quiverkey(q, 0.5, 0.85, quiv_label_val, quiv_label,
                      coordinates='figure', labelpos='E', color='green')
    plt.xlabel('X (ref ' + unit + ')')
    plt.ylabel('Y (ref ' + unit + ')')
    plt.title(title)
    plt.axis('equal')
    if plotlim is not None:
        plt.xlim(-1 * plotlim, plotlim)
        plt.ylim(-1 * plotlim, plotlim)
    plt.show()
    plt.pause(1)

    str_fmt = 'Residuals (mean, std): dx = {0:7.3f} +/- {1:7.3f} {5:s}  dy = {2:7.3f} +/- {3:7.3f} {5:s} for {4:s} stars'
    if len(ref_idx) > 1:
        print(str_fmt.format(dx[good_idx][ref_idx].mean(), dx[good_idx][ref_idx].std(),
                             dy[good_idx][ref_idx].mean(), dy[good_idx][ref_idx].std(), 'REF', unit2))
    else:
        print(str_fmt.format(dx[good_idx][ref_idx].mean(), 0.0,
                             dy[good_idx][ref_idx].mean(), 0.0, 'REF', unit2))
        
    print(str_fmt.format(dx[good_idx].mean(), dx[good_idx].std(),
                         dy[good_idx].mean(), dy[good_idx].std(), 'GOOD', unit2))


    return (dx, dy)


def plot_quiver_residuals_orig(x_t, y_t, x_ref, y_ref, good_idx, ref_idx, 
                               x_orig, y_orig, da, title, 
                               scale=None, plotlim=None):
    """
    unit : str
        'pixel' or 'arcsec'
        The pixel units of the input values. Note, if arcsec, then the values will be
        converted to milli-arcsec for plotting when appropriate. 

    scale : float
        The quiver scale. If none, then default units will be used appropriate to the unit. 

    plotlim : float (positive)
        Sets the size of the plotted figure. If None, then default is used.
        Otherwise plots figure of range [-plotlim, plotlim] x [-plotlim, plotlim].
    """
    # This is in arcsec...
    dx = (x_t - x_ref)
    dy = (y_t - y_ref)

    # ...convert to HST pixels, 0.04 arcsec/pix
    dx /= 0.04
    dy /= 0.04

    # Residual modulus
    r_good = np.hypot(dx[good_idx], dy[good_idx])
    r_ref = np.hypot(dx[good_idx][ref_idx], dy[good_idx][ref_idx])

    # Residual angle
    agood = angle_from_xy(dx[good_idx], dy[good_idx])
    aref = angle_from_xy(dx[good_idx][ref_idx], dy[good_idx][ref_idx])
    # Subtract off rotation angle IN DEGREES (e.g. if going from Gaia to HST camera frame)
    agood -= da
    aref -= da
    # Keep everything within 0 to 360
    agood = agood % 360
    aref = aref % 360

    dx_good_new, dy_good_new = rotate(dx[good_idx], dy[good_idx], -da)
    dx_ref_new, dy_ref_new = rotate(dx[good_idx][ref_idx], dy[good_idx][ref_idx], -da)
    print('Rotation angle between HST and Gaia (deg): ', da)

    plt.figure(102, figsize=(6,6))
    plt.clf()
    q = plt.quiver(x_orig[good_idx], y_orig[good_idx], dx_good_new, dy_good_new,
                   color='black', scale=scale, angles='xy', alpha=0.5)
    plt.quiver(x_orig[good_idx][ref_idx], y_orig[good_idx][ref_idx], dx_ref_new, dy_ref_new,
               color='red', scale=scale, angles='xy')
    plt.quiverkey(q, 0.5, 0.85, 0.3, '0.3 pix',
                      coordinates='figure', labelpos='E', color='green')
    plt.xlabel('X (ref pix)')
    plt.ylabel('Y (ref pix)')
    plt.title(title)
    plt.axis('equal')
    if plotlim is not None:
        plt.xlim(-1 * plotlim, plotlim)
        plt.ylim(-1 * plotlim, plotlim)
    plt.show()
    plt.pause(1)

    return (dx, dy)


def rotate(x, y, theta):
    """
    theta: in degrees
    """
    theta = np.radians(theta)

    xnew = x*np.cos(theta) - y*np.sin(theta)
    ynew = x*np.sin(theta) + y*np.cos(theta)

    return xnew, ynew


def plot_chi2_dist(tab, Ndetect):
    """
    tab = flystar table
    Ndetect = Number of epochs star detected in
    """
    chi2_x_list = []
    chi2_y_list = []
    fnd_list = [] # Number of non-NaN error measurements
    
    for ii in range(len(tab['xe'])):
        # Ignore the NaNs 
        fnd = np.where(tab['xe'][ii, :] > 0)[0]
        fnd_list.append(len(fnd))
        
        time = tab['t'][ii, fnd]
        x = tab['x'][ii, fnd]
        y = tab['y'][ii, fnd]
        xerr = tab['xe'][ii, fnd]
        yerr = tab['ye'][ii, fnd]

        dt = tab['t'][ii, fnd] - tab['t0'][ii]
        fitLineX = tab['x0'][ii] + (tab['vx'][ii] * dt)
        fitLineY = tab['y0'][ii] + (tab['vy'][ii] * dt)

        diffX = x - fitLineX
        diffY = y - fitLineY
        sigX = diffX / xerr
        sigY = diffY / yerr
        
        chi2_x = np.sum(sigX**2)
        chi2_y = np.sum(sigY**2)
        chi2_x_list.append(chi2_x)
        chi2_y_list.append(chi2_y)

    x = np.array(chi2_x_list)
    y = np.array(chi2_y_list)
    fnd = np.array(fnd_list)
    
    idx = np.where(fnd == Ndetect)[0]
    # Fitting position and velocity... so subtract 2 to get Ndof
    Ndof = Ndetect - 2 
    chi2_xaxis = np.linspace(0, 40, 100)

    plt.figure(figsize=(6,4))
    plt.clf()
    plt.hist(x[idx], bins=np.arange(400), histtype='step', label='X', density=True)
    plt.hist(y[idx], bins=np.arange(400), histtype='step', label='Y', density=True)
    plt.plot(chi2_xaxis, chi2.pdf(chi2_xaxis, Ndof), 'r-', alpha=0.6, 
             label='$\chi^2$ ' + str(Ndof) + ' dof')
    plt.title('$N_{epoch} = $' + str(Ndetect) + ', $N_{dof} = $' + str(Ndof))
    plt.xlim(0, 40)
    plt.legend()

    return

def plot_stars(tab, star_names, NcolMax=3, epoch_array = None, figsize=(15,15), color_time=False):
    """
    Plot a set of stars positions and residuals over time. 

    epoch_array : None, array
        Array of the epoch indicies to plot. If None, plots all epochs.
    """
    print( 'Creating residuals plots for star(s):' )
    print( star_names )
    
    Nstars = len(star_names)
    Ncols = 2 * np.min([Nstars, NcolMax])
    if Nstars <= Ncols/2:
        Nrows = 3
    else:
        Nrows = math.ceil(Nstars / (Ncols / 2)) * 3

    plt.close('all')
    plt.figure(2, figsize=figsize)
    names = tab['name']
    mag = tab['m0']
    x = tab['x0']
    y = tab['y0']
    r = np.hypot(x, y)
    
    for i in range(Nstars):
        starName = star_names[i]
        
        try:
            ii = np.where(tab['name'] == starName)[0][0]
            print(ii, tab[ii]['name'])
        except IndexError:
            print("!! %s is not in this list"%starName)

        fnd = np.where(tab['xe'][ii, :] > 0)[0]

        if epoch_array is not None:
            fnd = np.intersect1d(fnd, epoch_array)

        time = tab['t'][ii, fnd]
        dtime = time.data % 1 
        x = tab['x'][ii, fnd]
        y = tab['y'][ii, fnd]
        xerr = tab['xe'][ii, fnd]
        yerr = tab['ye'][ii, fnd]

        dt = tab['t'][ii, fnd] - tab['t0'][ii]
        fitLineX = tab['x0'][ii] + (tab['vx'][ii] * dt)
        fitLineY = tab['y0'][ii] + (tab['vy'][ii] * dt)

        fitSigX = np.hypot(tab['x0e'][ii], tab['vxe'][ii]*dt)
        fitSigY = np.hypot(tab['y0e'][ii], tab['vye'][ii]*dt)
        
        diffX = x - fitLineX
        diffY = y - fitLineY
        diff = np.hypot(diffX, diffY)
        rerr = np.sqrt((diffX*xerr)**2 + (diffY*yerr)**2) / diff
        sigX = diffX / xerr
        sigY = diffY / yerr
        sig = diff / rerr

        # Determine if there are points that are more than 5 sigma off
        idxX = np.where(abs(sigX) > 4)
        idxY = np.where(abs(sigY) > 4)
        idx = np.where(abs(sig) > 4)

        # Calculate chi^2 metrics
        chi2_x = np.sum(sigX**2)
        chi2_y = np.sum(sigY**2)

        dof = len(x) - 2

        chi2_red_x = chi2_x / dof
        chi2_red_y = chi2_y / dof
        

        print( 'Star:        ', starName )
        print( '\tX Chi^2 = %5.2f (%6.2f for %2d dof)' % 
              (chi2_red_x, chi2_x, dof))
        print( '\tY Chi^2 = %5.2f (%6.2f for %2d dof)' % 
              (chi2_red_y, chi2_y, dof))

        tmin = time.min()
        tmax = time.max()

        dateTicLoc = plt.MultipleLocator(3)
        dateTicRng = [np.floor(tmin), np.ceil(tmax)]
        dateTics = np.arange(np.floor(tmin), np.ceil(tmax)+0.1)
        DateTicsLabel = dateTics

        # See if we are using MJD instead.
        if time[0] > 50000:
            print('MJD')
            dateTicLoc = plt.MultipleLocator(1000)
            t0 = int(np.round(np.min(time), 50))
            tO = int(np.round(np.max(time), 50))
            dateTicRng = [tmin-200, tmax+200]
            dateTics = np.arange(dateTicRng[0], dateTicRng[-1]+500, 1000)
            DateTicsLabel = dateTics


        maxErr = np.array([(diffX-xerr)*1e3, (diffX+xerr)*1e3,
                           (diffY-yerr)*1e3, (diffY+yerr)*1e3]).max()

        if maxErr > 2:
            maxErr = 2.0
        resTicRng = [-1.1*maxErr, 1.1*maxErr]

        from matplotlib.ticker import FormatStrFormatter
        fmtX = FormatStrFormatter('%5i')
        fmtY = FormatStrFormatter('%6.3f')
        fontsize1 = 10

        if i < (Ncols/2):
            col = (2*i)+1
            row = 1
        else:
            col = 1 + 2*(i % (Ncols/2))
            row = 1 + 3*(i//(Ncols/2)) 
            
        ind = (row-1)*Ncols + col
        
        paxes = plt.subplot(Nrows, Ncols, ind)
        plt.plot(time, fitLineX, 'b-')
        plt.plot(time, fitLineX + fitSigX, 'b--')
        plt.plot(time, fitLineX - fitSigX, 'b--')
        if not color_time:
            plt.errorbar(time, x, yerr=xerr, fmt='k.')
        else:
            norm = colors.Normalize(vmin=0, vmax=1, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
            time_color = np.array([(mapper.to_rgba(v)) for v in dtime])
            for xx, yy, ee, color in zip(time, x, xerr, time_color):
                plt.plot(xx, yy, '.', color=color)
                plt.errorbar(xx, yy, ee, color=color)
        rng = plt.axis()
        #plt.ylim(np.min(x-xerr)*0.99, np.max(x+xerr)*1.01) 
        plt.xlabel('Date (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            plt.xlabel('Date (MJD)', fontsize=fontsize1)
        plt.ylabel('X (asec)', fontsize=fontsize1)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.yaxis.set_major_formatter(fmtY)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        plt.annotate(starName,xy=(1.0,1.1), xycoords='axes fraction', fontsize=12, color='red')


        col = col + 1
        ind = (row-1)*Ncols + col

        paxes = plt.subplot(Nrows, Ncols, ind)
        plt.plot(time, fitLineY, 'b-')
        plt.plot(time, fitLineY + fitSigY, 'b--')
        plt.plot(time, fitLineY - fitSigY, 'b--')
        if not color_time:
            plt.errorbar(time, y, yerr=yerr, fmt='k.')
        else:
            norm = colors.Normalize(vmin=0, vmax=1, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
            time_color = np.array([(mapper.to_rgba(v)) for v in dtime])
            for xx, yy, ee, color in zip(time, y, yerr, time_color):
                plt.plot(xx, yy, '.', color=color)
                plt.errorbar(xx, yy, ee, color=color)
        rng = plt.axis()
        plt.axis(dateTicRng + [rng[2], rng[3]], fontsize=fontsize1)
        plt.xlabel('Date - 2000 (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            plt.xlabel('Date (MJD)', fontsize=fontsize1)
        plt.ylabel('Y (asec)', fontsize=fontsize1)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.yaxis.set_major_formatter(fmtY)
        paxes.tick_params(axis='both', which='major', labelsize=12)

        row = row + 1
        col = col - 1
        ind = (row-1)*Ncols + col

        paxes = plt.subplot(Nrows, Ncols, ind)
        plt.plot(time, np.zeros(len(time)), 'b-')
        plt.plot(time,  fitSigX*1e3, 'b--')
        plt.plot(time, -fitSigX*1e3, 'b--')
        if not color_time:
            plt.errorbar(time, (x - fitLineX)*1e3, yerr=xerr*1e3, fmt='k.')
        else:
            norm = colors.Normalize(vmin=0, vmax=1, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
            time_color = np.array([(mapper.to_rgba(v)) for v in dtime])
            for xx, yy, ee, color in zip(time, (x - fitLineX)*1e3, xerr*1e3, time_color):
                plt.plot(xx, yy, '.', color=color)
                plt.errorbar(xx, yy, ee, color=color)
        plt.axis(dateTicRng + resTicRng, fontsize=fontsize1)
        plt.xlabel('Date (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            plt.xlabel('Date (MJD)', fontsize=fontsize1)
        plt.ylabel('X Residuals (mas)', fontsize=fontsize1)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)

        col = col + 1
        ind = (row-1)*Ncols + col

        paxes = plt.subplot(Nrows, Ncols, ind)
        plt.plot(time, np.zeros(len(time)), 'b-')
        plt.plot(time,  fitSigY*1e3, 'b--')
        plt.plot(time, -fitSigY*1e3, 'b--')
        if not color_time:
            plt.errorbar(time, (y - fitLineY)*1e3, yerr=yerr*1e3, fmt='k.')
        else:
            norm = colors.Normalize(vmin=0, vmax=1, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
            time_color = np.array([(mapper.to_rgba(v)) for v in dtime])
            for xx, yy, ee, color in zip(time, (y - fitLineY)*1e3, yerr*1e3, time_color):
                plt.plot(xx, yy, '.', color=color)
                plt.errorbar(xx, yy, ee, color=color)
        plt.axis(dateTicRng + resTicRng, fontsize=fontsize1)
        plt.xlabel('Date (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            plt.xlabel('Date (MJD)', fontsize=fontsize1)
        plt.ylabel('Y Residuals (mas)', fontsize=fontsize1)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)

        row = row + 1
        col = col - 1
        ind = (row-1)*Ncols + col

        paxes = plt.subplot(Nrows, Ncols, ind)
        if not color_time:
            plt.errorbar(x,y, xerr=xerr, yerr=yerr, fmt='k.')
        else:
            sc = plt.scatter(x, y, s=0, c=dtime, vmin=0, vmax=1, cmap='hsv')
            clb = plt.colorbar(sc)
            clb.ax.tick_params(labelsize=fontsize1)
            norm = colors.Normalize(vmin=0, vmax=1, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
            time_color = np.array([(mapper.to_rgba(v)) for v in dtime])
            for xx, yy, eexx, eeyy, color in zip(x, y, xerr, yerr, time_color):
                plt.plot(xx, yy, '.', color=color)
                plt.errorbar(xx, yy, eexx, eeyy, color=color)
        plt.axis('equal')
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
        paxes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        paxes.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        plt.xlabel('X (asec)', fontsize=fontsize1)
        plt.ylabel('Y (asec)', fontsize=fontsize1)
        plt.plot(fitLineX, fitLineY, 'b-')    

        col = col + 1
        ind = (row-1)*Ncols + col

        bins = np.arange(-7.5, 7.5, 1)
        paxes = plt.subplot(Nrows, Ncols, ind)
        id = np.where(diffY < 0)[0]
        sig[id] = -1.*sig[id] 
        (n, b, p) = plt.hist(sigX, bins, histtype='stepfilled', color='b', label='X')
        plt.setp(p, 'facecolor', 'b')
        (n, b, p) = plt.hist(sigY, bins, histtype='step', color='r', label='Y')
        plt.axis([-7, 7, 0, 8], fontsize=10)
        plt.legend(fontsize=10)
        plt.xlabel('Residuals (sigma)', fontsize=fontsize1)
        plt.ylabel('Number of Epochs', fontsize=fontsize1)

    if Nstars == 1:
        plt.subplots_adjust(wspace=0.4, hspace=0.4, left = 0.15, bottom = 0.1, right=0.9, top=0.9) 
        # plt.savefig(rootDir+'plots/plotStar_' + starName + '.png')
    else:
        plt.subplots_adjust(wspace=0.6, hspace=0.6, left = 0.08, bottom = 0.05, right=0.95, top=0.90)
        # plt.savefig(rootDir+'plots/plotStar_all.png')
        plt.show()

    plt.show()

    return


def plot_stars_mag(tab, star_names, NcolMax=4, epoch_array = None, figsize=(12,12), color_time=False):
    """
    Plot a set of stars magnitude + error bars over time. 

    epoch_array : None, array
        Array of the epoch indicies to plot. If None, plots all epochs.
    """
    print( 'Creating magnitude plots for star(s):' )
    print( star_names )

    Nstars = len(star_names)
    Ncols = np.min([Nstars, NcolMax])
    if Nstars <= Ncols:
        Nrows = 2
    else:
        Nrows = math.ceil(Nstars/Ncols) * 2
    
    plt.close('all')
    plt.figure(2, figsize=figsize)
    names = tab['name']
    
    for i in range(Nstars):
        starName = star_names[i]

        try:
            ii = np.where(tab['name'] == starName)[0][0]
            print(ii, tab[ii]['name'])
        except IndexError:
            print("!! %s is not in this list"%starName)

        fnd = np.where(tab['xe'][ii, :] > 0)[0]

        if epoch_array is not None:
            fnd = np.intersect1d(fnd, epoch_array)

        time = tab['t'][ii, fnd]
        dtime = time.data % 1 
        m = tab['m'][ii, fnd]
        merr = tab['me'][ii, fnd]
        m0 = tab['m0'][ii]
        m0e = tab['m0e'][ii]

        diff = m0 - m
        sig = diff/merr
        
        chi2 = np.sum(sig)
        dof = len(m) - 1 # horizontal line has only 1 dof right?
        chi2_red = chi2/dof

        print( 'Star:        ', starName )
#        print( 'Average mag (unweighted) = {:.3f}'.format(np.average(m)))
#        print( 'St dev mag (unweighted, millimag) = {:.1f}'.format(1000 * np.std(m)))
        print( 'Average mag (weighted) = {:.3f}'.format(m0))
        print( 'Std dev mag (weighted, millimag) = {:.1f}'.format(1000 * m0e))

        tmin = time.min()
        tmax = time.max()

        dateTicLoc = plt.MultipleLocator(3)
        dateTicRng = [np.floor(tmin), np.ceil(tmax)]
        dateTics = np.arange(np.floor(tmin), np.ceil(tmax)+0.1)
        DateTicsLabel = dateTics

        # See if we are using MJD instead.
        if time[0] > 50000:
            print('MJD')
            dateTicLoc = plt.MultipleLocator(1000)
            t0 = int(np.round(np.min(time), 50))
            tO = int(np.round(np.max(time), 50))
            dateTicRng = [tmin-200, tmax+200]
            dateTics = np.arange(dateTicRng[0], dateTicRng[-1]+500, 1000)
            DateTicsLabel = dateTics

        from matplotlib.ticker import FormatStrFormatter
        fmtX = FormatStrFormatter('%5i')
        fmtY = FormatStrFormatter('%6.3f')
        fontsize1 = 10

        paxes = plt.subplot(2*Nrows, Ncols, 2*i+1)
        plt.plot(time, m0 * np.ones(len(time)), label='m0')
        if not color_time:
            plt.errorbar(time, m, yerr=merr, fmt='k.')
        else:
            norm = colors.Normalize(vmin=0, vmax=1, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
            time_color = np.array([(mapper.to_rgba(v)) for v in dtime])
            for xx, yy, ee, color in zip(time, m, merr, time_color):
                plt.plot(xx, yy, '.', color=color)
                plt.errorbar(xx, yy, ee, color=color)
        rng = plt.axis()
        plt.xlabel('Date (yrs)', fontsize=fontsize1)
        if time[0] > 50000:
            plt.xlabel('Date (MJD)', fontsize=fontsize1)
        plt.ylabel('mag', fontsize=fontsize1)
        paxes.xaxis.set_major_formatter(fmtX)
        paxes.yaxis.set_major_formatter(fmtY)
        paxes.tick_params(axis='both', which='major', labelsize=fontsize1)
#        plt.title(starName, fontsize=12, color='red')
        plt.gca().invert_yaxis()
        if i == 0:
            plt.legend()
        plt.annotate(starName,xy=(1.0, 1.1), xycoords='axes fraction', fontsize=12, color='red')

        bins = np.arange(-7.5, 7.5, 1)
        paxes = plt.subplot(2*Nrows, Ncols, 2*i+2)
        id = np.where(diff < 0)[0]
        sig[id] = -1.*sig[id] 
        (n, b, p) = plt.hist(sig, bins, histtype='stepfilled', color='b')
        plt.setp(p, 'facecolor', 'b')
        plt.axis([-7, 7, 0, 8], fontsize=10)
        plt.xlabel('Residuals (sigma)', fontsize=fontsize1)
        plt.ylabel('N epochs', fontsize=fontsize1)
       
    if Nstars == 1:
        plt.subplots_adjust(wspace=0.4, hspace=0.4, left = 0.15, bottom = 0.1, right=0.9, top=0.9) 
        # plt.savefig(rootDir+'plots/plotStar_' + starName + '.png')
    else:
        plt.subplots_adjust(wspace=0.5, hspace=0.7, left = 0.08, bottom = 0.05, right=0.95, top=0.90)
        # plt.savefig(rootDir+'plots/plotStar_all.png')
        plt.show()

    plt.show()


def plot_errors_vs_r_m(star_tab):
    """
    Plot the positional errors and the proper motion errors as a function of radius 
    and magnitude. The positional an proper motion errors will be the mean in the 
    two axis (as is used in pick_good_ref_stars()). 
    """
    r = np.hypot(star_tab['x0'], star_tab['y0'])
    p_err = np.mean((star_tab['x0e'], star_tab['y0e']), axis=0) * 1e3
    pm_err = np.mean((star_tab['vxe'], star_tab['vye']), axis=0) * 1e3

    plt.figure(figsize=(12, 6))
    plt.clf()
    plt.subplots_adjust(wspace=0.4)
    
    plt.subplot(1, 2, 1)
    plt.scatter(star_tab['m0'], r, c=p_err, s=8, vmin=0, vmax=0.75)
    plt.colorbar(label='Pos Err (mas)')
    plt.xlabel('Mag')
    plt.ylabel('Radius (")')

    plt.subplot(1, 2, 2)
    plt.scatter(star_tab['m0'], r, c=pm_err, s=8, vmin=0, vmax=0.75)
    plt.colorbar(label='PM Err (mas/yr)')
    plt.xlabel('Mag')
    plt.ylabel('Radius (")')

    return
 
       
def plot_sky(stars_tab,
            plot_errors=False, center_star=None, range=0.4,
            xcenter=0, ycenter=0, show_names=False, saveplot=False,
            mag_range=None, with_orbits=True,
            orbits_file=None, manual_print=False):
    """
    Plot all the stars at their positions over time with each star having a different
    symbol and each epoch having a different color.
    """
    """
    Parameters
    ----------
    stars_tab : flystar.startables.StarTable
        The StarTable containining 'x', 'y', 't', 'xe', 'ye', columns etc. 
        for plotting, where each of these columns is a 2D array of 
        [star_index, epoch_index].


    Optional Keywords
    -----------------
    plot_errors : bool
        (def=False) Plot error bars on all the points (quad sum of positional
        and alignment errors).
    center_star : str
        (def=None) Named star to center initial plot on.
    show_names : bool
        (def=False) label the name of the star in the first epoch
    range : float
        (def=0.4) Sets the half width of the X and Y axis in
        arcseconds from xcenter and ycenter.
    xcenter : float
        (def=0.0) The X center point of the plot in arcseconds
        offset from Sgr A*.
    ycenter : float
        (def=0.0) The Y center point of the plot in arcseconds
        offset from Sgr A*.
    saveplot : bool
        (def=False) Save plot as .png and .pdf.
    mag_range: intervals
        (def=None) Magnitude cuts performed using the given interval
    """

    nEpochs = stars_tab['x'].shape[1]
    nStars = stars_tab['x'].shape[0]

    if (center_star != None):
        idx = np.where(stars_tab['name'] == center_star)[0]

        if (len(idx) > 0):
            xcenter = stars_tab['x0'][idx[0]]
            ycenter = stars_tab['y0'][idx[0]]
        else:
            print('Could not find star to center, %s. Reverting to default.' % \
                  (center_star))

    # Determine the range of years in integer units.
    good_t = np.isfinite(stars_tab['t'])
    epochs = np.unique(stars_tab['t'][good_t])
    assert len(epochs) == stars_tab['t'].shape[1]
    
    yearsInt = np.floor(epochs).astype('int')

    # Set up a color scheme
    cnorm = colors.Normalize(stars_tab['t'][0, :].min(), stars_tab['t'][0, :].max() + 1)
    cmap = plt.cm.gist_ncar

    colorList = []
    for ee in np.arange(nEpochs):
        foo = cnorm(yearsInt[ee])
        colorList.append( cmap(cnorm(yearsInt[ee])) )

    py.close(2)
    fig = py.figure(2, figsize=(13,10))

    previousYear = 0.0

    point_labels = {}
    epochs_legend=[]

    for ee in np.arange(nEpochs):
        x = stars_tab['x'][:, ee]
        y = stars_tab['y'][:, ee]

        xe = stars_tab['xe'][:, ee]
        ye = stars_tab['ye'][:, ee]

        mag = stars_tab['m'][:, ee]
        name_epoch  = stars_tab['name_in_list'][:, ee]

        if mag_range is None:
            idx = np.where((x > -1000) & (y > -1000))[0]
        else:
            idx = np.where((x > -1000) & (y > -1000) & (mag <= np.max(mag_range)) & (mag >= np.min(mag_range)))[0]

        x = x[idx]
        y = y[idx]
        xe = xe[idx]
        ye = ye[idx]
        mag = mag[idx]
        name_epoch = name_epoch[idx]

        tmpNames = stars_tab['name'][idx]

        if yearsInt[ee] != previousYear:
            previousYear = yearsInt[ee]
            label = '%d' % yearsInt[ee]
        else:
            label = '_nolegend_'

        if plot_errors:
            (line, foo1, foo2) = py.errorbar(x, y, xerr=xe, yerr=ye,
                                            color=colorList[ee], fmt='k^',
                                            markeredgecolor=colorList[ee],
                                            markerfacecolor=colorList[ee],
                                            label=label, picker=4)
        else:
            (line, foo1, foo2) = py.errorbar(x, y, xerr=None, yerr=None,
                                            color=colorList[ee], fmt='k^',
                                            markeredgecolor=colorList[ee],
                                            markerfacecolor=colorList[ee],
                                            label=label, picker=4)

        #for legend
        if label is not '_nolegend_':
            line.set_label(str(label))
            epochs_legend.append(line)


        points_info = {'year': stars_tab['t'][0, ee],
                       'name': tmpNames, 'x': x, 'y': y, 'xe': xe, 'ye': ye, 'mag': mag,'name_epoch':name_epoch}

        point_labels[line] = points_info

    foo = PrintSelected(point_labels, fig, stars_tab, mag_range, manual_print=manual_print)
    py.connect('pick_event', foo)

    xlo = xcenter + (range)
    xhi = xcenter - (range)
    ylo = ycenter - (range)
    yhi = ycenter + (range)

    py.axis('equal')
    py.axis([xlo, xhi, ylo, yhi])
    py.xlabel('R.A. Offset from Sgr A* (arcsec)')
    py.ylabel('Dec. Offset from Sgr A* (arcsec)')

    py.legend(handles=epochs_legend, numpoints=1, loc='lower left', fontsize=12)

    if show_names:
        xpos = stars_tab['x0']
        ypos = stars_tab['y0']
        goodind = np.where((xpos <= xlo) & (xpos >= xhi) &
                           (ypos >= ylo) & (ypos <= yhi))[0]
        for ind in goodind:
            py.text(xpos[ind], ypos[ind], stars_tab['name'][ind], size=10)

    if saveplot:
        py.show(block=0)
        if (center_star != None):
            py.savefig('plot_sky_' + center_star + '.png')
        else:
            py.savefig('plot_sky.png')
    else:
        py.show()

    return
    
    
class PrintSelected(object):
    def __init__(self, points_info, fig, tab, mag_range, manual_print=False):
        self.points_info = points_info
        self.selected, = fig.gca().plot([0.],[0.], 'o', ms=12,
                                 markerfacecolor='none', markeredgecolor='red', visible=False)
        self.selected_same_year, = fig.gca().plot([0.],[0.], '*', ms=15,
                                markerfacecolor='red', markeredgecolor='red', visible=False)
        self.fig = fig
        self.tab = tab
        self.manual_print=manual_print
        self.mag_range=mag_range
        return

    def __call__(self, event):
        if event.mouseevent.button == 1:
            indices = event.ind

            data = self.points_info[event.artist]

            if self.manual_print:
                fmt = 'align_name="{:s}",epoch={:f},align_mag={:4.2f},align_x={:10.4f},align_xerr={:7.4f},align_y={:10.4f},align_yerr={:7.4f},name_epoch="{:s}"'
            else:
                fmt = '{:15s}  t={:10.6f}  m={:5.2f}  x={:10.4f} +/- {:7.4f}  y={:10.4f} +/- {:7.4f}  Epoch name: {:15s}'

            for ii in indices:
                print(fmt.format(data['name'][ii], data['year'], data['mag'][ii],
                            data['x'][ii], data['xe'][ii],
                            data['y'][ii], data['ye'][ii],data['name_epoch'][ii]))

            idx = self.tab['name'].index(data['name'][indices[0]])
            xs = self.tab['x'][idx, :]
            ys = self.tab['y'][idx, :]
            self.selected.set_visible(True)
            self.selected.set_data(xs, ys)
            self.fig.canvas.draw()
        elif event.mouseevent.button == 3:
            indices = event.ind
            data = self.points_info[event.artist]

            if self.manual_print:
                fmt = 'align_name="{:s}",epoch={:f},align_mag={:4.2f},align_x={:10.4f},align_xerr={:7.4f},align_y={:10.4f},align_yerr={:7.4f},name_epoch="{:s}"'
            else:
                fmt = '{:15s}  t={:10.6f}  m={:5.2f}  x={:10.4f} +/- {:7.4f}  y={:10.4f} +/- {:7.4f}  Epoch name: {:15s}'

            ii =indices[0]
            print(fmt.format(data['name'][ii], data['year'], data['mag'][ii],
                             data['x'][ii], data['xe'][ii],
                             data['y'][ii], data['ye'][ii],data['name_epoch'][ii]))

            idxEpoch = np.where(self.tab['t'][0, :] == data['year'])[0][0]
            x = self.tab['x'][:, idxEpoch]
            y = self.tab['y'][:, idxEpoch]
            mag = self.tab['m'][:, idxEpoch]
            if self.mag_range is None:
                idx = np.where((x > -1000) & (y > -1000))[0]
            else:
                idx = np.where((x > -1000) & (y > -1000) &
                               (mag <= np.max(self.mag_range)) &
                               (mag >= np.min(self.mag_range)))[0]

            x = x[idx]
            y = y[idx]
            self.selected_same_year.set_visible(True)
            self.selected_same_year.set_data(x, y)
            self.fig.canvas.draw()

        return


def compare_correction(good, ref, good_new, ref_new, fit_titles, bp_arr, plot_regime=None):
    """
    region : 'nonlin', 'nonsat', 'lin'
    """
    # Length of fit_titles shoudl equal len(good)
    # FIXME figsize=(20,18), num=103
    n = len(good)
    if len(fit_titles) != len(good):
        print('The number of titles needs to be the same length as the arrays')
        pdb.set_trace()
    # Loop over epochs
    for ee in range(len(good[0][0])):
        fig, ax = plt.subplots(6, n+1, figsize=((n+1)*3, 10), sharex='col', sharey='row', num=100+ee)
        plt.subplots_adjust(hspace=0.01, wspace=0.01)
        
        for ii in range(len(good)):
            mgood = good[ii][0][ee]
            mref = ref[ii][0][ee]
            yegood = good[ii][6][ee]
            yeref = ref[ii][6][ee]
            
            if ii == 0:
                agood = good[ii][1][ee]
                rgood = good[ii][2][ee]
                dxgood = good[ii][3][ee]
                dygood = good[ii][4][ee]
                xegood = good[ii][5][ee]
                
                aref = ref[ii][1][ee]
                rref = ref[ii][2][ee]
                dxref = ref[ii][3][ee]
                dyref = ref[ii][4][ee]
                xeref = ref[ii][5][ee]

                ax[0,ii].set_title('Uncorrected')
                
                ax[0,ii].scatter(mgood, dxgood, color='black', alpha=0.3, s=1)
                ax[0,ii].scatter(mref, dxref, color='red', alpha=0.3, s=1)
                ax[0,ii].set_ylabel('Res, x (mas)')
                ax[0,ii].set_ylim(-5, 5)
                ax[0,ii].axhline(y=0)
                
                ax[1,ii].scatter(mgood, dxgood/xegood, color='black', alpha=0.3, s=1)
                ax[1,ii].scatter(mref, dxref/xeref, color='red', alpha=0.3, s=1)
                ax[1,ii].set_ylabel('Res/Pos Err, x')
                ax[1,ii].set_ylim(-10, 10)
                ax[1,ii].axhline(y=0)
                
                ax[2,ii].scatter(mgood, dygood, color='black', alpha=0.3, s=1)
                ax[2,ii].scatter(mref, dyref, color='red', alpha=0.3, s=1)
                ax[2,ii].set_ylabel('Res, y (mas)')
                ax[2,ii].set_ylim(-5, 5)
                ax[2,ii].axhline(y=0)
                
                ax[3,ii].scatter(mgood, dygood/yegood, color='black', alpha=0.3, s=1)
                ax[3,ii].scatter(mref, dyref/yeref, color='red', alpha=0.3, s=1)
                ax[3,ii].set_ylabel('Res/Pos Err, y')
                ax[3,ii].set_ylim(-10, 10)
                ax[3,ii].axhline(y=0)

                ax[4,ii].scatter(mgood, agood, color='black', alpha=0.3, s=1)
                ax[4,ii].scatter(mref, aref, color='red', alpha=0.3, s=1)
                ax[4,ii].set_ylabel('Angle (deg)')
                
                ax[5,ii].scatter(mgood, rgood, color='black', alpha=0.3, s=1)
                ax[5,ii].scatter(mref, rref, color='red', alpha=0.3, s=1)
                ax[5,ii].set_ylabel('Modulus (mas)')
                ax[5,ii].set_yscale('log')
                if type(rgood) == astropy.table.column.MaskedColumn:
                    ax[5,ii].set_ylim(1e-3, 1.1 * np.max(np.concatenate([rgood.data, rref.data])))
                else:
                    ax[5,ii].set_ylim(1e-3, 1.1 * np.max(np.concatenate([rgood, rref])))
                ax[5,ii].set_xlabel('mag')
            
                # Denote the saturated, nonlinear, linear regimes  
                dmag = -3.5

                ax[0,ii].axvline(x=bp_arr[ee], ls = ':')
                ax[1,ii].axvline(x=bp_arr[ee], ls = ':')
                ax[2,ii].axvline(x=bp_arr[ee], ls = ':')
                ax[3,ii].axvline(x=bp_arr[ee], ls = ':')
                ax[4,ii].axvline(x=bp_arr[ee], ls = ':')
                ax[5,ii].axvline(x=bp_arr[ee], ls = ':')

                ax[0,ii].axvline(x=bp_arr[ee] + dmag, ls = ':')
                ax[1,ii].axvline(x=bp_arr[ee] + dmag, ls = ':')
                ax[2,ii].axvline(x=bp_arr[ee] + dmag, ls = ':')
                ax[3,ii].axvline(x=bp_arr[ee] + dmag, ls = ':')
                ax[4,ii].axvline(x=bp_arr[ee] + dmag, ls = ':')
                ax[5,ii].axvline(x=bp_arr[ee] + dmag, ls = ':')

                if plot_regime == 'nonlin':
                    # Axis limits to only show the nonlinear regime
                    ax[0,ii].set_xlim(bp_arr[ee] + dmag, bp_arr[ee])

                if plot_regime == 'nonsat':
                    # Axis limits to only show the nonsaturated regime
                    ax[0,ii].set_xlim(bp_arr[ee] + dmag, None)

                if plot_regime == 'lin':
                    # Axis limits to only show the linear regime
                    ax[0,ii].set_xlim(bp_arr[ee], None)

            ax[0,ii+1].set_title(fit_titles[ii])
                
            agood_new = good_new[ii][0][ee]
            rgood_new = good_new[ii][1][ee]
            dygood_new = good_new[ii][2][ee]

            aref_new = ref_new[ii][0][ee]
            rref_new = ref_new[ii][1][ee]
            dyref_new = ref_new[ii][2][ee]
            
            ax[2,ii+1].scatter(mgood, dygood_new, color='black', alpha=0.3, s=1)
            ax[2,ii+1].scatter(mref, dyref_new, color='red', alpha=0.3, s=1)
#            ax[2,ii+1].set_ylim(-5, 5)
            ax[2,ii+1].axhline(y=0)
            
            ax[3,ii+1].scatter(mgood, dygood_new/yegood, color='black', alpha=0.3, s=1)
            ax[3,ii+1].scatter(mref, dyref_new/yeref, color='red', alpha=0.3, s=1)
            ax[3,ii+1].axhline(y=0)

            ax[4,ii+1].scatter(mgood, agood_new, color='black', alpha=0.3, s=1)
            ax[4,ii+1].scatter(mref, aref_new, color='red', alpha=0.3, s=1)
            
            ax[5,ii+1].scatter(mgood, rgood_new, color='black', alpha=0.3, s=1)
            ax[5,ii+1].scatter(mref, rref_new, color='red', alpha=0.3, s=1)
            ax[5,ii+1].set_yscale('log')
            if type(rgood) == astropy.table.column.MaskedColumn:
                ax[5,ii].set_ylim(1e-3, 1.1 * np.max(np.concatenate([rgood_new.data, rref_new.data])))
            else:
                ax[5,ii].set_ylim(1e-3, 1.1 * np.max(np.concatenate([rgood_new, rref_new])))
            ax[5,ii+1].set_xlabel('mag')
        
            ax[0,ii+1].axvline(x=bp_arr[ee], ls = ':')
            ax[1,ii+1].axvline(x=bp_arr[ee], ls = ':')
            ax[2,ii+1].axvline(x=bp_arr[ee], ls = ':')
            ax[3,ii+1].axvline(x=bp_arr[ee], ls = ':')
            ax[4,ii+1].axvline(x=bp_arr[ee], ls = ':')
            ax[5,ii+1].axvline(x=bp_arr[ee], ls = ':')
            
            ax[0,ii+1].axvline(x=bp_arr[ee] + dmag, ls = ':')
            ax[1,ii+1].axvline(x=bp_arr[ee] + dmag, ls = ':')
            ax[2,ii+1].axvline(x=bp_arr[ee] + dmag, ls = ':')
            ax[3,ii+1].axvline(x=bp_arr[ee] + dmag, ls = ':')
            ax[4,ii+1].axvline(x=bp_arr[ee] + dmag, ls = ':')
            ax[5,ii+1].axvline(x=bp_arr[ee] + dmag, ls = ':')

            if plot_regime == 'nonlin':
                # Axis limits to only show the nonlinear regime
                ax[0,ii+1].set_xlim(bp_arr[ee] + dmag, bp_arr[ee])

            if plot_regime == 'nonsat':
                # Axis limits to only show the nonsaturated regime
                ax[0,ii+1].set_xlim(bp_arr[ee] + dmag, None)

            if plot_regime == 'lin':
                # Axis limits to only show the linear regime
                ax[0,ii+1].set_xlim(bp_arr[ee], None)
                
        for ax_i in ax:
            for ax_j in ax_i:
                for item in ([ax_j.title, ax_j.xaxis.label, ax_j.yaxis.label] +
                             ax_j.get_xticklabels() + ax_j.get_yticklabels()):
                    item.set_fontsize(10)

        plt.suptitle('Epoch ' + str(ee))
        plt.show()
        plt.pause(1)
