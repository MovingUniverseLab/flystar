from astropy.table import Table
from flystar.starlists import StarList
import os

test_dir = os.path.dirname(__file__)


def make_star_list():
    # User input
    cat_file = test_dir + '/A.lis'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file, format='ascii', delimiter='\s')
    
    # Copy columns from the input file.
    # Note that all of these inputs will be numpy arrays.
    x_in = cat_tab['x'].data
    y_in = cat_tab['y'].data
    m_in = cat_tab['m'].data
    xe_in = cat_tab['xe'].data
    ye_in = cat_tab['ye'].data
    snr_in = cat_tab['snr'].data
    name_in = cat_tab['name'].data

    # Name is a unique name for each star and is a 1D array.
    
    starlist_time = 2011.1
    starlist_name = 'A.lis'

    # Generate the starlist
    starlist = StarList(name=name_in, x=x_in, y=y_in, m=m_in, xe=xe_in,
                        ye=ye_in, snr=snr_in, list_time=starlist_time,
                        list_name=starlist_name)

    return starlist


def test_copy():
    lis = make_star_list()

    lis2 = StarList(lis)

    assert len(lis) == len(lis2)

    return
