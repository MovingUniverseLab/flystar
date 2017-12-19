from flystar import transforms
import numpy as np
import pdb

def compare_evaluate_errors():
    px_init = [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    py_init = [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    
    foo = transforms.PolyTransform(2, px_init, py_init)

    x = np.random.rand(100) * 1000
    y = np.random.rand(100) * 1000
    xe = np.abs(np.random.randn(100) * 0.1)
    ye = np.abs(np.random.randn(100) * 0.1)

    xe_new1 = foo._evaluate_error2(x, y, xe, ye, foo.px.parameters)

    print()
    print()
    xe_new2, ye_new2 = foo._evaluate_error(x, y, xe, ye)

    # BROKEN

    return

def test_evaluate_errors():
    px_init = [ 100.0, 0.99, 1e-6]
    py_init = [ 100.0, 1e-6, 0.99]
    
    foo = transforms.PolyTransform(1, px_init, py_init)

    x = np.random.rand(10) * 1000
    y = np.random.rand(10) * 1000
    xe = np.abs(np.random.randn(10) * 0.1)
    ye = np.abs(np.random.randn(10) * 0.1)

    xe_new, ye_new = foo.evaluate_error(x, y, xe, ye)

    assert np.abs(xe[0] - xe_new[0]) < 0.01
    assert np.abs(ye[0] - ye_new[0]) < 0.01

    assert np.abs(xe[1] - xe_new[1]) < 0.01
    assert np.abs(ye[1] - ye_new[1]) < 0.01

    assert np.abs(xe[2] - xe_new[2]) < 0.01
    assert np.abs(ye[2] - ye_new[2]) < 0.01

def test_evaluate_velocities():
    px_init = [ 0.0, 0.99, 1e-6]
    py_init = [ 0.0, 1e-6, 0.99]
    
    foo = transforms.PolyTransform(1, px_init, py_init)

    x = np.random.rand(10) * 1000
    y = np.random.rand(10) * 1000
    xe = np.abs(np.random.randn(10) * 0.2)
    ye = np.abs(np.random.randn(10) * 0.2)
    vx = np.random.randn(10)
    vy = np.random.randn(10)
    vxe = np.abs(np.random.randn(10) * 0.1)
    vye = np.abs(np.random.randn(10) * 0.1)

    vx_new, vy_new = foo.evaluate_vel(x, y, vx, vy)
    vxe_new, vye_new = foo.evaluate_vel_err(x, y, vx, vy, xe, ye, vxe, vye)
    

    assert np.abs(vx[0] - vx_new[0]) < 0.05
    assert np.abs(vy[0] - vy_new[0]) < 0.05

    assert np.abs(vx[1] - vx_new[1]) < 0.05
    assert np.abs(vy[1] - vy_new[1]) < 0.05

    assert np.abs(vx[2] - vx_new[2]) < 0.05
    assert np.abs(vy[2] - vy_new[2]) < 0.05

    assert np.abs(vxe[0] - vxe_new[0]) < 0.05
    assert np.abs(vye[0] - vye_new[0]) < 0.05

    assert np.abs(vxe[1] - vxe_new[1]) < 0.05
    assert np.abs(vye[1] - vye_new[1]) < 0.05

    assert np.abs(vxe[2] - vxe_new[2]) < 0.05
    assert np.abs(vye[2] - vye_new[2]) < 0.05
    
def try_out_equation(order):
    x = 1
    y = 1
    fmt = '{0:1s}  {1:1s}  {2:3s} {3:5s} {4:5s} {5:3s} {6:3s} {7:5s} {8:5s}'
    print(fmt.format('i', 'j', 'i-j', 'i-j-1', 'i-j-2', 'j-1', 'j-2', 'term1', 'term2'))


    # From dvx'/dx
    for i in range(order+1):
        for j in range(i+1):
            term1 = (i-j) * (i-j-1) * x**(i-j-2) * y**j
            term2 = (j) * (i-j) * x**(i-j-1) * y**(j-1)
            
            fmt = '{0:1d}  {1:1d}  {2:3d} {3:5d} {4:5d} {5:3d} {6:3d} {7:5.1f} {8:5.1f}'
            print(fmt.format(i, j, i-j, i-j-1, i-j-2, j-1, j-2, term1, term2))

            
            
    return
