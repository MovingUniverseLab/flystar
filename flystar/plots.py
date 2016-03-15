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
        transformation

    starlist: astropy table
        Transformed starist with the reference starlist coordinates.
        Standard column headers are assumed

    starlist_mat: astropy table
        Transformed starlist only containing the matched stars used in
        the transformation. Standard column headers are assumed. 
    
    """
    py.figure(figsize=(10,10))
    py.clf()
    py.plot(ref['x'], ref['y'], 'k.', ms=10, label='Reference')
    py.plot(starlist['x_trans'], starlist['y_trans'], 'r.', ms=5, label='Label.dat')
    py.plot(ref_mat['x'], ref_mat['y'], 'gs', ms=10, label='Matched Reference')
    py.plot(starlist_mat['x'], starlist_mat['y'], 'bs', ms=5, label='Matched label.dat')
    py.xlabel('X position (Reference Coords)')
    py.ylabel('Y position (Reference Coords)')
    py.legend(numpoints=1)
    py.title('Label.dat Positions After Transformation')
    py.axis([-100, 1300, -100, 1500])
    py.savefig('Transformed_positions.png')

    return




    # Histogram of position differences for the matched stars:
    # reference - label.dat
    diff_x = starlist_mat['x'] - label_mat['x']
    diff_y = starlist_mat['y'] - label_mat['y']

    py.figure(2, figsize=(10,10))
    py.clf()
    py.hist(diff_x, bins=25, color='blue', label='X')
    py.hist(diff_y, bins=25, color='red', label='Y')
    py.xlabel('Reference Position - label.dat Position (starlist pix)')
    py.ylabel('N stars')
    py.title('Position Differences after Transformation')
    py.legend()
    py.savefig('Positions_hist.png')

    # Looking at quiverplot of differences relative to position in image.
    # These are all the stars that were used in the transformation
    py.figure(3, figsize=(10,10))
    py.clf()
    q = py.quiver(starlist_mat['x'], starlist_mat['y'], diff_x, diff_y, scale=10)
    py.quiverkey(q, 0.2, 0.92, 0.2, '0.2 pix', coordinates='figure', color='black')
    py.xlabel('X Position (Reference, pix)')
    py.ylabel('Y Position (Reference, pix)')
    py.title('Reference - Transformed label.dat positions')
    py.savefig('Positions_quiver.png')
    
    # Look at the histogram of mag diff for matched stars, as well
    diff_m = starlist_mat['m'] - label_mat['m']
    
    py.figure(4, figsize=(10,10))
    py.clf()
    py.hist(diff_m, bins=25)
    py.xlabel('Reference Mag - label.dat Mag')
    py.ylabel('N stars')
    py.title('Magnitude Match')
    py.savefig('Magnitude_hist.png')
