from astropy.table import Table
from flystar.startables import StarTable
import numpy as np
import pytest
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
    assert type(startable) == StarTable
    
    return

def test_combine_lists():
    """
    Test the startables.combine_lists() functionality.
    """
    t = make_star_table()

    # Test 1: call on a non-existant column.
    with pytest.raises(KeyError):
        t.combine_lists('foo')

    # Test 2: average x an check the first entry manually. Unweighted.
    x_avg_0 = t['x'][0, :].mean()
    t.combine_lists('x', mask_val=-100000)
    assert t['x_avg'][0] == x_avg_0
    assert t['x_avg'][-1] == pytest.approx(2108.855, 0.001)

    # Test 3: Trying calling the same thing a second time and make sure the
    # answers don't change and we didn't break anything.
    t.combine_lists('x', mask_val=-100000)
    assert t['x_avg'][0] == x_avg_0
    assert t['x_avg'][-1] == pytest.approx(2108.855, 0.001)
    
    # Test 4: weighted average of x.
    x_wgt_0 = 1.0 / t['xe'][0, :]**2
    x_avg_0 = np.average(t['x'][0, :], weights=x_wgt_0)
    t.combine_lists('x', mask_val=-100000, weights_col='xe')
    assert t['x_avg'][0] == x_avg_0
    
    x_wgt_last = 1.0 / t['xe'][-1, :]**2
    x_avg_last = np.average(t['x'][-1, [2,7]], weights=x_wgt_last[[2,7]])
    assert t['x_avg'][-1] == pytest.approx(x_avg_last)

    return

def test_add_list():
    """
    Test the startables.combine_lists() functionality.
    """
    t = make_star_table()
    t_orig = t.copy()

    # Make some new data for a new "list".
    x_new = t['x'][0] + 0.1
    y_new = t['y'][0] + 0.1
    m_new = t['m'][0] + 0.1
    xe_new = t['xe'][0] + 0.01
    ye_new = t['ye'][0] + 0.01
    me_new = t['me'][0] + 0.01
    t_new = 2008.0

    # Test 1: Add new list to the end with complete data
    t.add_list(x=x_new, y=y_new, m=m_new, xe=xe_new, ye=ye_new, me=me_new,
                   meta={'list_times': t_new})

    assert len(t) == len(t_orig)
    assert t['x'].shape == (t_orig['x'].shape + [0, 1])

    return

def make_star_table():
    # User input
    cat_file = test_dir + '/test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)
    
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

    return startable
