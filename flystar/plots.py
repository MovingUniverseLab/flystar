import pylab as py

####################################################
# Code for making diagnostic plots for astrometry
# alignment
####################################################


def trans_positions(ref, ref_mat, starlist, starlist_mat):
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
    
    """
    py.figure(figsize=(10,10))
    py.clf()
    py.plot(ref['x'], ref['y'], 'g+', ms=5, label='Reference')
    py.plot(starlist['x_trans'], starlist['y_trans'], 'rx', ms=5, label='Label.dat')
    py.plot(ref_mat['x'], ref_mat['y'], color='skyblue', marker='s', ms=10, linestyle='None', label='Matched Reference')
    py.plot(starlist_mat['x'], starlist_mat['y'], color='darkblue', marker='s', ms=5, linestyle='None', label='Matched label.dat')
    py.xlabel('X position (Reference Coords)')
    py.ylabel('Y position (Reference Coords)')
    py.legend(numpoints=1)
    py.title('Label.dat Positions After Transformation')
    py.axis([-100, 1300, -100, 1500])
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
    py.xlabel('Reference Position - label.dat Position (starlist pix)')
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

def posDiff_quiver(ref_mat, starlist_mat, qscale=10):
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
        Scale parameter for the quiver plot
    
    """
    diff_x = ref_mat['x'] - starlist_mat['x']
    diff_y = ref_mat['y'] - starlist_mat['y']

    
    py.figure(figsize=(10,10))
    py.clf()
    q = py.quiver(ref_mat['x'], ref_mat['y'], diff_x, diff_y, scale=qscale)
    py.quiverkey(q, 0.2, 0.92, 0.2, '0.2 pix', coordinates='figure', color='black')
    py.xlabel('X Position (Reference, pix)')
    py.ylabel('Y Position (Reference, pix)')
    py.title('Reference - Transformed label.dat positions')
    py.savefig('Positions_quiver.png')

    return
