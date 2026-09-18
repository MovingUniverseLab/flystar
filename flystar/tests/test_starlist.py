from astropy.table import Table
from flystar.starlists import StarList
import os,  pdb
import flystar

test_data_path = f'{flystar.__path__[0]}/tests/test_data'


def make_star_list():
    # User input
    cat_file = f'{test_data_path}/A.lis'

    # Read and arrange the test input
    cat_tab = Table.read(cat_file, format='ascii', delimiter=r'\s')
    
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
    starlist_name = f'{test_data_path}/A.lis'

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

def test_restrict_by_value():
    liso = make_star_list()
    
    # Test 1
    lis = make_star_list()
    lis.restrict_by_value(m_min=10, m_max=13)

    assert lis['m'].min() > 10
    assert lis['m'].max() < 13
    assert len(lis) < len(liso)

    # Test 2
    lis = make_star_list()
    lis.restrict_by_value(m_min=10, x_max=500)

    assert lis['m'].min() > 10
    assert lis['x'].max() < 500
    assert len(lis) < len(liso)

    return


def test_starlist_keeps_extra_columns():
    """
    StarList's keyword constructor used to accept only
    ('x','y','m','xe','ye','me','corr') and silently DROP every other keyword.
    Building a reference list for a Linear motion model the obvious way --
    StarList(name=.., x=.., y=.., m=.., vx=.., vy=.., t0=..) -- therefore
    produced a list with no velocities at all, and no error to say so. That
    surfaced much later as an align reference that refused to propagate.
    Extra keywords are kept as columns now.
    """
    import numpy as np
    from astropy.table import MaskedColumn
    from flystar.starlists import StarList

    n = 5
    base = dict(name=[f's{i}' for i in range(n)],
                x=np.arange(n, dtype=float), y=np.arange(n, dtype=float),
                m=np.full(n, 15.0), xe=np.full(n, .01), ye=np.full(n, .01),
                me=np.full(n, .01))

    sl = StarList(vx=np.full(n, 1.5), vy=np.full(n, -0.5),
                  t0=np.full(n, 2020.0), vx_err=np.full(n, .001), **base)
    for col in ('vx', 'vy', 't0', 'vx_err'):
        assert col in sl.colnames, f'extra column {col} was dropped'
    np.testing.assert_allclose(np.asarray(sl['vx']), 1.5)
    np.testing.assert_allclose(np.asarray(sl['t0']), 2020.0)

    # the recognized columns and meta still behave
    assert sl.colnames[:4] == ['name', 'x', 'y', 'm']
    sl_meta = StarList(list_time=2020.5, list_name='ep1', meta={'extra': 'kept'}, **base)
    assert sl_meta.meta['list_time'] == 2020.5
    assert sl_meta.meta['list_name'] == 'ep1'
    assert sl_meta.meta['extra'] == 'kept'
    assert sl_meta.meta['n_stars'] == n

    # a wrongly-shaped extra fails loudly rather than becoming a bad column
    for bad in (np.zeros(3), np.zeros((n, 2)), np.ma.masked_array(np.zeros(3))):
        try:
            StarList(vx=bad, **base)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted bad extra of shape {np.shape(bad)}')

    # masks on an extra column survive (MaskedColumn subclasses MaskedArray,
    # so a plain np.ma.masked_array must be handled too)
    for maker in (lambda d, mk: np.ma.masked_array(d, mask=mk),
                  lambda d, mk: MaskedColumn(data=d, mask=mk)):
        mv = maker(np.arange(n, dtype=float), [0, 1, 0, 0, 1])
        sl_masked = StarList(vx=mv, **base)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(sl_masked['vx']), [False, True, False, False, True])

    # copy-construction from an existing StarList is a different code path and
    # must keep every column, extras included
    sl_copy = StarList(sl, copy=True)
    assert sl_copy.colnames == sl.colnames
