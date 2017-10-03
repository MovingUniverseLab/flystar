from astropy.table import Table
from flystar.startables import StarTable
import numpy as np
import os

test_dir = os.path.dirname(__file__)

def test_StarTable_init1():
    """
    Test creation of new StarTable.
    """
    # User input
    cat_file = test_dir + '/test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)

    N_stars = len(cat_tab)
    N_lists = cat_tab['x'].shape[1]
    print(N_stars, N_lists)
    print(cat_tab['x'].shape)

    # Make a fake 2D array of names per epoch. We will call them "id".
    # Note that all of these inputs will be numpy arrays.
    x_in = cat_tab['x'].data
    y_in = cat_tab['y'].data
    m_in = cat_tab['m'].data
    xe_in = cat_tab['xe'].data
    ye_in = cat_tab['ye'].data
    me_in = cat_tab['me'].data

    # Name is a unique name for each star and is a 1D array. 
    name_in = cat_tab['name'].data
    starlist_times = np.array([2001.0, 2002.1, 2003.0, 2004., 2005., 2006., 2007., 2008.])
    starlist_names = np.array(['file1', 'file2', 'file3', 'file4', 'file5', 'file6', 'file7', 'file8'])

    # Generate the startable
    startable = StarTable(name=name_in, x=x_in, y=y_in, m=m_in, xe=xe_in, ye=ye_in, me=me_in,
                              ref_list=1,
                              list_times=starlist_times, list_names=starlist_names)

    # Now put in some assertions to make sure all our startable columns
    # have the right dimensions.
    assert len(startable) == N_stars
    assert startable['x'].shape == (N_stars, N_lists)
    assert startable['y'].shape == (N_stars, N_lists)
    assert startable['m'].shape == (N_stars, N_lists)
    assert startable['xe'].shape == (N_stars, N_lists)
    assert startable['ye'].shape == (N_stars, N_lists)
    assert startable['me'].shape == (N_stars, N_lists)
    assert len(startable['name']) == N_stars
    assert startable.meta['list_times'][0] == starlist_times[0]
    
    return

def test_StarTable_init2():
    """
    Test creation of new StarTable.
    """
    # User input
    cat_file = 'test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)
    
    name = cat_tab['name']
    x = cat_tab['x']
    y = cat_tab['y']
    m = cat_tab['m']

    xe = cat_tab['xe']
    ye = cat_tab['ye']
    me = cat_tab['me']
    
    time_lists = [2001, 2002.1, 2003.0]
    name_lists = ['file1', 'file2', 'file3']

    # Generate the startable
    startable = StarTable(x=x, y=y_, m=m, xe=xe, ye=ye, me=me, name=name,
                              ref=1, time_lists=time_lists,
                              name_lists=name_lists)

    assert len(startable) == x.shape[0]

    return

def test_StarTable_init3():
    """
    Test creation of new StarTable with NO extra information.
    """
    # User input
    cat_file = 'test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)
    
    name = cat_tab['name']
    x = cat_tab['x']
    y = cat_tab['y']
    m = cat_tab['m']

    time_lists = [2001, 2002.1, 2003.0]
    name_lists = ['file1', 'file2', 'file3']

    # Generate the startable
    startable = StarTable(x=x, y=y, m=m, name=name,
                              ref=1, time_lists=time_lists,
                              name_lists=name_lists)

    assert len(startable) == x.shape[0]

    return

def test_StarTable_init4():
    """
    Test creation of new StarTable with NO name or NO x column. Make sure
    the correct errors are passed.
    """
    # User input
    cat_file = 'test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)
    
    name = cat_tab['name']
    x = cat_tab['x']
    y = cat_tab['y']
    m = cat_tab['m']

    time_lists = [2001, 2002.1, 2003.0]
    name_lists = ['file1', 'file2', 'file3']

    # Generate the startable with no names
    startable = StarTable(x=x, y=y, m=m, 
                              ref=1, time_lists=time_lists,
                              name_lists=name_lists)
    
    # TO DO Add assert or try catch to make sure the proper error is thrown.

    # Generate the startable with no x
    startable = StarTable(y=y, m=m, name=name,
                              ref=1, time_lists=time_lists,
                              name_lists=name_lists)
    
    # TO DO Add assert or try catch to make sure the proper error is thrown.
    
    return
