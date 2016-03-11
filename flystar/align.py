import match
import transforms

def initial_align(table1, table2, transformModel, briteN):
    """
    Calculates an initial (unweighted) transformation between two sets of
    starlists. Matching is done using a blind triangle-matching algorithm
    of the brightest briteN stars in both starlists.

    Starlists must be astropy tables with standard column headers. All
    positions must be at the same epoch, and +x must be in the same
    direction.

    Standard column headers:
    name: name
    x: x position
    y: y position
    xe: error in x position
    ye: error in y position
    
    vx: proper motion in x direction
    vy proper motion in y direction
    vxe: error in x proper motion
    vye: error in y proper motion

    m: magnitude
    me: magnitude error
    
    t: epoch of observation

    use?: specify use in transformation 
    

    Parameters:
    ----------





    Output:
    ------
    Transformation object
    
    """
    # Extract necessary information from tables (x, y, m)
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']




    # Check to make sure epoch match
    if table1['t'][0] != table2['t'][0]:
        print 'Warning: starlist epochs do not match'
        
    # Run the blind triangle-matching algorithm to find the matches between
    # the starlists


    # Calculate transformation based on matches


    return transform



def transform_and_match(table1, table2, transform, dr_tol, dm_tol=None):
    """
    Transform starlist 2 into starlist 1 frame and match stars to given radius
    and magnitude tolerance.

    Starlists must be astropy tables with standard columns names as specified
    in initial_align. 

    Parameters:
    -----------
    table1: astropy table

    transform: transformation object



    Output:
    -------
    
    """
    # Transform x, y coordinates from starlist 2 into starlist 1


    # Match starlist 1 and 2


    
    # Output matched starlists


    return




def find_transform(matchedStarlist, transModel, weights=False)
    """
    Given a matched starlist, derive a new transform

    """
    # Read matched starlist

    # Calculate transformation based on matched stars


    # Output transformation

    return





def readLabel(labelFile, t0):
    """
    Read in a label.dat file, rename columns with standard
    names. Use velocities to convert positions into epoch
    t0, and then flip x positions and velocities such that
    +x is to the west.

    Update values in columns as necessary

    Parameters:
    ----------





    Output:
    ------
    
    """



    return



def readStarlist(starlistFile):
    """
    Read in a starlist file, rename columns with standard
    names

    Parameter:
    ---------


    Output:
    ------
    """



    return 



def align_example():
    """"
    Example of how to use the flystar code.
    """
    # Read in label.dat, reference starlists






    
    return
    
