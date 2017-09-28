from astropy.table import Table
from flystar.startables import StarTable


def test_StarTable_init1():
    """
    Test creation of new StarTable.
    """
    # User input
    cat_file = 'test_catalog.fits'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file)
    
    name_tab = Table([cat_tab['name'], cat_tab['name'], cat_tab['name']],
                    names=('1', '2', '3'))
    x_tab = Table([cat_tab['x'][:, 1], cat_tab['x'][:, 2], cat_tab['x'][:, 3]],
                      names=('1', '2', '3'))
    y_tab = Table([cat_tab['y'][:, 1], cat_tab['y'][:, 2], cat_tab['y'][:, 3]],
                names=('1', '2', '3'))
    m_tab = Table([cat_tab['m'][:, 1], cat_tab['m'][:, 2], cat_tab['m'][:, 3]],
                names=('1', '2', '3'))
    xe_tab = Table([cat_tab['xe'][:, 1], cat_tab['xe'][:, 2], cat_tab['xe'][:, 3]],
                names=('1', '2', '3'))
    ye_tab = Table([cat_tab['ye'][:, 1], cat_tab['ye'][:, 2], cat_tab['ye'][:, 3]],
                names=('1', '2', '3'))
    me_tab = Table([cat_tab['me'][:, 1], cat_tab['me'][:, 2], cat_tab['me'][:, 3]],
                names=('1', '2', '3'))
    time_lists = [2001, 2002.1, 2003.0]
    name_lists = ['file1', 'file2', 'file3']

    # Generate the startable
    startable = StarTable(x=x_tab, y=y_tab, m=m_tab, xe=xe_tab, ye=ye_tab,
                        me=me_tab, name=name_tab, ref=1, time_lists=time_lists,
                        name_lists=name_lists)

    print('')  # Test line for inserting a debug breakpoint

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
