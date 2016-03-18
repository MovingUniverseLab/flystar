import pylab as py
import numpy as np
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
    py.plot(starlist_mat['x'], starlist_mat['y'], color='darkblue', marker='s', ms=5, linestyle='None', label='Matched label.dat')
    py.xlabel('X position (Reference Coords)')
    py.ylabel('Y position (Reference Coords)')
    py.legend(numpoints=1)
    py.title('Label.dat Positions After Transformation')
    if xlim != None:
        py.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
    py.savefig('Transformed_positions.png')

    return


def posDiff_hist(ref_mat, starlist_mat, bins=25):
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

    bins: int
        number of bins used in histogram
    
    """
    diff_x = ref_mat['x'] - starlist_mat['x']
    diff_y = ref_mat['y'] - starlist_mat['y']

    py.figure(figsize=(10,10))
    py.clf()
    py.hist(diff_x, histtype='step', bins=bins, color='blue', label='X')
    py.hist(diff_y, histtype='step', bins=bins, color='red', label='Y')
    py.xlabel('Reference Position - label.dat Position (reference coords)')
    py.ylabel('N stars')
    py.title('Position Differences for matched stars')
    py.legend()
    py.savefig('Positions_hist.png')

    return

def magDiff_hist(ref_mat, starlist_mat, bins=25):
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

def posDiff_quiver(ref_mat, starlist_mat, qscale=10, keyLength=0.2):
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
        Key length parameter for quiver plot

    
    """
    diff_x = ref_mat['x'] - starlist_mat['x']
    diff_y = ref_mat['y'] - starlist_mat['y']

    # Add own reference quiver arrow to end of array, since actual one is broken
    # This will be in lower left portion of plot
    diff_x = np.array(diff_x)
    diff_y = np.array(diff_y)
    xpos = np.array(ref_mat['x'])
    ypos = np.array(ref_mat['y'])

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
    trans_vx = starlist_trans['vx']
    trans_vy = starlist_trans['vy']

    py.figure(figsize=(10,10))
    py.clf()
    py.plot(trans_vx, trans_vy, 'k.', ms=8, label='Transformed', alpha=0.1)
    py.plot(ref_vx, ref_vy, 'r.', ms=8, label='Reference', alpha=0.4)
    py.xlabel('Vx (Reference units)')
    py.ylabel('Vy (Reference units)')
    if vxlim != None:
        py.axis([vxlim[0], vylim[1], vylim[0], vylim[1]])
    py.title('Reference and Transformed Proper Motions')
    py.savefig('Transformed_velocities.png')

    return

def vel_hist(ref_mat, starlist_mat, bins=25):
    """
    Plot the distributions of the velocities in the reference list to
    the transformed starlist. Obviously, we assume that both list have
    velocities.

    Paramters:
    ----------
    ref_mat: astropy table
         Reference starlist, with velocities

    starlist_mat: astropy table
         Transformed starlist, with velocities

    bins: int (default=25)
         Number of bins in the histogram
    """
    # Will produce 2-panel plot: Vx, and Vy
    ref_vx = ref_mat['vx']
    ref_vy = ref_mat['vy']
    trans_vx = starlist_mat['vx']
    trans_vy = starlist_mat['vy']
    
    py.figure(figsize=(20,10))
    py.subplot(121)
    py.subplots_adjust(left=0.1)
    py.hist(ref_vx, bins=bins, histtype='step', color='red', label='Reference')
    py.hist(trans_vx, bins=bins, histtype='step', color='blue', label='Transformed')
    py.xlabel('Vx (reference units)')
    py.ylabel('N_stars')
    py.title('Vx Distribution, Matched')
    py.legend()
    py.subplot(122)
    py.hist(ref_vy, bins=bins, histtype='step', color='red', label='Reference')
    py.hist(trans_vy, bins=bins, histtype='step', color='blue', label='Transformed')
    py.xlabel('Vy (reference units)')
    py.ylabel('N_stars')
    py.title('Vy Distribution, Matched')
    py.legend()
    py.savefig('Vel_dist.png')

    return
