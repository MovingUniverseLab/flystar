from astropy.table import Table
from astropy import table
from flystar.startables import StarTable
from flystar.starlists import StarList
import numpy as np
import pytest
import os
import pdb

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

def test_StarTable_init2():
    """
    Initialize a StarTable with a StarList... this should 
    work (and just add a few meta keywords) if everything is working correctly. 
    Also double check that we can add a second list to it using add_starlist and
    we can get_starlist() as well.
    """
    list_file1 = 'A.lis'
    list_file2 = 'B.lis'
    list1 = StarList.from_lis_file(list_file1)
    list2 = StarList.from_lis_file(list_file2)

    # Test initializer
    tab = StarTable(list1)

    assert len(tab) == len(list1)

    
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
    assert t['x0'][0] == x_avg_0
    assert t['x0'][-1] == pytest.approx(2108.855, 0.001)

    # Test 3: Trying calling the same thing a second time and make sure the
    # answers don't change and we didn't break anything.
    t.combine_lists('x', mask_val=-100000)
    assert t['x0'][0] == x_avg_0
    assert t['x0'][-1] == pytest.approx(2108.855, 0.001)
    
    # Test 4: weighted average of x.
    x_wgt_0 = 1.0 / t['xe'][0, :]**2
    x_avg_0 = np.average(t['x'][0, :], weights=x_wgt_0)
    t.combine_lists('x', mask_val=-100000, weights_col='xe')
    assert t['x0'][0] == x_avg_0
    
    x_wgt_last = 1.0 / t['xe'][-1, :]**2
    x_avg_last = np.average(t['x'][-1, [2,7]], weights=x_wgt_last[[2,7]])
    assert t['x0'][-1] == pytest.approx(x_avg_last)

    return

def test_add_starlist():
    """
    Test the startables.combine_lists() functionality.
    """
    t = make_star_table()
    t_orig = Table(t)

    # Make some new data for a new "list".
    x_new = t['x'][:, 0] + 0.1
    y_new = t['y'][:, 0] + 0.1
    m_new = t['m'][:, 0] + 0.1
    xe_new = t['xe'][:, 0] + 0.01
    ye_new = t['ye'][:, 0] + 0.01
    me_new = t['me'][:, 0] + 0.01
    t_new = 2008.0

    # Test 1: Add new list to the end with complete data: Keyword format
    t.add_starlist(x=x_new, y=y_new, m=m_new, xe=xe_new, ye=ye_new, me=me_new,
                   meta={'list_times': t_new})

    assert len(t) == len(t_orig)

    expected_shape = np.array(t_orig['x'].shape)
    expected_shape[1] += 1
    
    assert len(t['x'].shape) == len(expected_shape)
    assert t['x'].shape[0] == expected_shape[0]
    assert t['x'].shape[1] == expected_shape[1]

    assert len(t['y'].shape) == len(expected_shape)
    assert t['y'].shape[0] == expected_shape[0]
    assert t['y'].shape[1] == expected_shape[1]

    assert len(t['m'].shape) == len(expected_shape)
    assert t['m'].shape[0] == expected_shape[0]
    assert t['m'].shape[1] == expected_shape[1]

    assert len(t['xe'].shape) == len(expected_shape)
    assert t['xe'].shape[0] == expected_shape[0]
    assert t['xe'].shape[1] == expected_shape[1]

    assert len(t['ye'].shape) == len(expected_shape)
    assert t['ye'].shape[0] == expected_shape[0]
    assert t['ye'].shape[1] == expected_shape[1]

    assert len(t['me'].shape) == len(expected_shape)
    assert t['me'].shape[0] == expected_shape[0]
    assert t['me'].shape[1] == expected_shape[1]

    assert len(t['name']) == len(t_orig['name'])
    assert len(t.meta['list_times']) == expected_shape[1]
    assert t.meta['n_lists'] == 9

    # Test 2: Add as starlist rather than with keywords.
    starlist = StarList(name=t_orig['name'], x=x_new, y=y_new, m=m_new,
                            xe=xe_new, ye=ye_new, me=me_new, list_time=2001.0, list_name='A.lis')
    
    t = make_star_table()
    t.add_starlist(starlist=starlist)

    assert len(t) == len(t_orig)

    expected_shape = np.array(t_orig['x'].shape)
    expected_shape[1] += 1
    
    assert len(t['x'].shape) == len(expected_shape)
    assert t['x'].shape[0] == expected_shape[0]
    assert t['x'].shape[1] == expected_shape[1]

    assert len(t['y'].shape) == len(expected_shape)
    assert t['y'].shape[0] == expected_shape[0]
    assert t['y'].shape[1] == expected_shape[1]

    assert len(t['m'].shape) == len(expected_shape)
    assert t['m'].shape[0] == expected_shape[0]
    assert t['m'].shape[1] == expected_shape[1]

    assert len(t['xe'].shape) == len(expected_shape)
    assert t['xe'].shape[0] == expected_shape[0]
    assert t['xe'].shape[1] == expected_shape[1]

    assert len(t['ye'].shape) == len(expected_shape)
    assert t['ye'].shape[0] == expected_shape[0]
    assert t['ye'].shape[1] == expected_shape[1]

    assert len(t['me'].shape) == len(expected_shape)
    assert t['me'].shape[0] == expected_shape[0]
    assert t['me'].shape[1] == expected_shape[1]

    assert len(t['name']) == len(t_orig['name'])
    assert len(t.meta['list_times']) == expected_shape[1]
    assert t.meta['n_lists'] == 9

    return

def test_get_starlist():
    """
    Make a StarTable and have it return a StarList for just one 
    of the epochs.
    """
    t = make_star_table()

    t_list = t.get_starlist(2)

    assert t['x'][0,2] == t_list['x'][0]
    assert type(t_list) == StarList
    assert len(t_list['x'].shape) == 1
    
    return


def test_combine_1col():
    # User input
    cat_file = test_dir + '/test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)
    
    # Make a fake 2D array of names per epoch. We will call them "id".
    # Note that all of these inputs will be numpy arrays.
    x_in = cat_tab['x'].data[:, [0]]
    y_in = cat_tab['y'].data[:, [0]]
    m_in = cat_tab['m'].data[:, [0]]
    xe_in = cat_tab['xe'].data[:, [0]]
    ye_in = cat_tab['ye'].data[:, [0]]
    me_in = cat_tab['me'].data[:, [0]]

    # Name is a unique name for each star and is a 1D array.
    name_in = cat_tab['name'].data
    starlist_times = np.array([2001.0])
    starlist_names = np.array(['file1'])

    # Generate the startable
    t = StarTable(name=name_in, x=x_in, y=y_in, m=m_in, xe=xe_in, ye=ye_in, me=me_in,
                  ref_list=0,
                  list_times=starlist_times, list_names=starlist_names)

    t.combine_lists('x', weights_col='xe')

    assert t['x0'][0] == t['x'][0]

    return

def test_fit_velocities():
    tab = make_star_table()

    # We don't need the entire table... lets just
    # pull a small subset for faster testing.
    tab1 = tab[0:100]
    tab2 = tab[10000:10100]
    tab3 = tab[-100:]
    tab = table.vstack((tab1, tab2, tab3))

    tab.fit_velocities(verbose=True)

    # Test creation of new variables
    assert len(tab['vx']) == len(tab)
    assert len(tab['vy']) == len(tab)
    assert len(tab['vxe']) == len(tab)
    assert len(tab['vye']) == len(tab)
    assert len(tab['n_vfit']) == len(tab)
    assert tab.meta['n_vfit_bootstrap'] == 0

    # Test no-fit for stars with N<2 epochs.
    n_epochs = (tab['x'] >= 0).sum(axis=1)
    idx = np.where(n_epochs <= 2)[0]
    assert (tab['vx'][idx] == 0).all()
    assert (tab['vxe'][idx] == 0).all()
    assert (tab['n_vfit'][idx] == 2).all()

    # Test that the velocity errors were calculated.
    assert (tab['vxe'][0:100] > 0).all()
    assert (tab['x0e'][0:100] > 0).all()
    assert (tab['vye'][0:100] > 0).all()
    assert (tab['y0e'][0:100] > 0).all()
    assert np.isfinite(tab['x0']).all()
    assert np.isfinite(tab['vx']).all()
    assert np.isfinite(tab['y0']).all()
    assert np.isfinite(tab['vy']).all()
    assert np.isfinite(tab['x0e']).all()
    assert np.isfinite(tab['vxe']).all()
    assert np.isfinite(tab['y0e']).all()
    assert np.isfinite(tab['vye']).all()

    ##########
    # Test running a second time. We should get the same results.
    ##########
    vx_orig = tab['vx']
    x0_orig = tab['x0']
    vxe_orig = tab['vxe']
    x0e_orig = tab['x0e']
    tab.fit_velocities(verbose=False)

    assert (vx_orig == tab['vx']).all()
    assert (x0_orig == tab['x0']).all()
    assert (vxe_orig == tab['vxe']).all()
    assert (x0e_orig == tab['x0e']).all()
    
    ##########
    # Test bootstrap
    ##########
    tab_b = table.vstack((tab1, tab2, tab3))
    tab_b.fit_velocities(verbose=True, bootstrap=50)
    
    assert tab_b.meta['n_vfit_bootstrap'] == 50
    assert tab_b['x0e'][0] > tab['x0e'][0]
    assert tab_b['vxe'][0] > tab['vxe'][0]
    assert tab_b['y0e'][0] > tab['y0e'][0]
    assert tab_b['vye'][0] > tab['vye'][0]
    
    ##########
    # Test what happens with no velocity errors
    ##########
    tab.remove_columns(['xe', 'ye', 'x0', 'y0', 'x0e', 'y0e', 'vx', 'vy', 'vxe', 'vye', 'n_vfit'])
    tab.fit_velocities(verbose=False)

    assert len(tab['vx']) == len(tab)
    assert len(tab['vy']) == len(tab)
    assert len(tab['vxe']) == len(tab)
    assert len(tab['vye']) == len(tab)
    assert len(tab['n_vfit']) == len(tab)
    assert (tab['vxe'][0:100] > 0).all()
    assert (tab['x0e'][0:100] > 0).all()
    assert (tab['vye'][0:100] > 0).all()
    assert (tab['y0e'][0:100] > 0).all()

    
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
    n_in = cat_tab['n'].data

    # Name is a unique name for each star and is a 1D array.
    name_in = cat_tab['name'].data
    starlist_times = np.array([2001.0, 2002.1, 2003.0, 2004., 2005., 2006., 2007., 2008.])
    starlist_names = np.array(['file1', 'file2', 'file3', 'file4', 'file5', 'file6', 'file7', 'file8'])

    # Generate the startable
    startable = StarTable(name=name_in, x=x_in, y=y_in, m=m_in, xe=xe_in, ye=ye_in, me=me_in, n=n_in,
                              ref_list=1,
                              list_times=starlist_times, list_names=starlist_names)

    return startable

    
