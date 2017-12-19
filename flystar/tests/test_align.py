from flystar import align
from flystar import starlists
from flystar import startables
from flystar import transforms
import numpy as np
import pdb

def test_mosaic_lists_shifts():
    """
    Cross-match and align 4 starlists.
    """
    input_shifts = make_fake_starlists_shifts()
    
    list_files = ['random_0.lis',
                      'random_shift_1.lis',
                      'random_shift_2.lis',
                      'random_shift_3.lis',
                      'random_shift_4.lis']
    lists = [starlists.StarList.from_lis_file(lf, error=False) for lf in list_files]

    star_table, trans_table = align.mosaic_lists(lists, ref_index=0, iters=2,
                                                 trans_class=transforms.PolyTransform,
                                                 trans_args={'order': 1})

    assert len(star_table) == 200
    assert star_table['x'].shape == (200, 5)
    assert star_table['name_in_list'][0, 0] == 'star_000'
    assert star_table['name_in_list'][0, 1] == 'star_000'
    assert star_table['name_in_list'][0, 2] == 'star_000'
    assert star_table['name_in_list'][0, 3] == 'star_000'
    assert star_table['name_in_list'][0, 4] == 'star_000'

    # The 'x' column should contain the transformed coordinates for each epoch.
    # These should be the nearly same within ~0.2 pixels.
    assert star_table['x'][0].std() < 0.2
    assert star_table['x_orig'][0, 0] == lists[0]['x'][0]

    return trans_table[0]
    
    

def test_mosaic_lists():
    """
    Cross-match and align 4 starlists.
    """
    list_files = ['A.lis', 'B.lis', 'C.lis', 'D.lis']
    lists = [starlists.StarList.from_lis_file(lf) for lf in list_files]

    star_table, trans_table = align.mosaic_lists(lists, ref_index=0, iters=5,
                                                 trans_class=transforms.PolyTransform,
                                                 trans_args={'order': 2})

    return
    
    
def make_fake_starlists_shifts():
    N_stars = 200
    x = np.random.rand(N_stars) * 1000
    y = np.random.rand(N_stars) * 1000
    m = (np.random.rand(N_stars) * 8) + 9
        
    sdx = np.argsort(m)
    x = x[sdx]
    y = y[sdx]
    m = m[sdx]
    
    name = ['star_{0:03d}'.format(ii) for ii in range(N_stars)]

    # Save original positions as reference (1st) list.
    fmt = '{0:10s}  {1:5.2f} 2015.0 {2:9.4f}  {3:9.4f} 0 0 0 0\n'
    _out = open('random_0.lis', 'w')
    for ii in range(N_stars):
        _out.write(fmt.format(name[ii], m[ii], x[ii], y[ii]))
    _out.close()


    ##########
    # Shifts
    ##########
    # Make 4 new starlists with different shifts.
    shifts = [[  6.5,  10.1],
              [100.3,  50.5],
              [-30.0,-100.7],
              [250.0,-250.0]]

    for ss in range(len(shifts)):
        xnew = x - shifts[ss][0]
        ynew = y - shifts[ss][1]

        # Perturb with small errors (0.1 pix)
        xnew += np.random.randn(N_stars) * 0.1
        ynew += np.random.randn(N_stars) * 0.1

        mnew = m + np.random.randn(N_stars) * 0.05

        _out = open('random_shift_{0:d}.lis'.format(ss+1), 'w')
        for ii in range(N_stars):
            _out.write(fmt.format(name[ii], mnew[ii], xnew[ii], ynew[ii]))
        _out.close()

    return shifts

def make_fake_starlists_poly1():
    N_stars = 200
    x = np.random.rand(N_stars) * 1000
    y = np.random.rand(N_stars) * 1000
    m = (np.random.rand(N_stars) * 8) + 9
        
    sdx = np.argsort(m)
    x = x[sdx]
    y = y[sdx]
    m = m[sdx]
    
    name = ['star_{0:03d}'.format(ii) for ii in range(N_stars)]

    # Save original positions as reference (1st) list.
    fmt = '{0:10s}  {1:5.2f} 2015.0 {2:9.4f}  {3:9.4f} 0 0 0 0\n'
    _out = open('random_0.lis', 'w')
    for ii in range(N_stars):
        _out.write(fmt.format(name[ii], m[ii], x[ii], y[ii]))
    _out.close()


    ##########
    # Shifts
    ##########
    # Make 4 new starlists with different shifts.
    transforms = [[[  6.5, 0.99, 1e-5], [  10.1, 1e-5, 0.99]],
                  [[100.3, 0.98, 1e-5], [  50.5, 9e-6, 1.001]],
                  [[-30.0, 1.00, 1e-5], [-100.7, 2e-5, 0.999]],
                  [[250.0, 0.97, 2e-5], [-250.0, 1e-5, 1.001]]]

    for ss in range(len(shifts)):
        #transforms.PolyTransform2D(1, transforms[ss])
        xnew = x - shifts[ss][0]
        ynew = y - shifts[ss][1]

        # Perturb with small errors (0.1 pix)
        xnew += np.random.randn(N_stars) * 0.1
        ynew += np.random.randn(N_stars) * 0.1

        mnew = m + np.random.randn(N_stars) * 0.05

        _out = open('random_shift_{0:d}.lis'.format(ss+1), 'w')
        for ii in range(N_stars):
            _out.write(fmt.format(name[ii], mnew[ii], xnew[ii], ynew[ii]))
        _out.close()

    return shifts

        
              

        
