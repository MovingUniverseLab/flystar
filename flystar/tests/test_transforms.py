from flystar import transforms
import numpy as np
import pdb
from astropy.modeling import models, fitting

def test_PolyTransform_init_no_guess():
    px_init = [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    py_init = [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    
    t = transforms.PolyTransform(2, px_init, py_init)

    return

def compare_evaluate_errors():
    px_init = [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    py_init = [ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    
    foo = transforms.PolyTransform(2, px_init, py_init)

    x = np.random.rand(100) * 1000
    y = np.random.rand(100) * 1000
    xe = np.abs(np.random.randn(100) * 0.1)
    ye = np.abs(np.random.randn(100) * 0.1)

    xe_new1 = foo._evaluate_error2(x, y, xe, ye, foo.px.parameters)

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

    return

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

    return

def test_0th_order_poly():
    # Test out a 0th order polynomial implementation.
    # Note that this isn't a true 0th order; but rather a 1st order
    # with the c1_* coefficieints fixed. But this is the way we
    # ALWAYS need it.

    px_init = [0.1]
    py_init = [25.0]

    ##########
    # Check PolyTransform initialization
    ##########
    trans = transforms.PolyTransform(0, px_init, py_init)

    assert len(trans.px.parameters) == 3
    assert len(trans.py.parameters) == 3
    assert trans.order == 0

    x_in = np.array([5.0, 10.0, 100.0])
    y_in = np.array([50.0, 100.0, 1000.0])

    # Test evaluation
    x_out, y_out = trans.evaluate(x_in, y_in)
    
    np.testing.assert_almost_equal(x_in + px_init[0], x_out, 4)
    np.testing.assert_almost_equal(y_in + py_init[0], y_out, 4)

    xe_in = np.array([0.1, 0.2, 0.1])
    ye_in = np.array([0.05, 0.1, 0.3])

    # Test errors
    xe_out, ye_out = trans.evaluate_error(x_in, y_in, xe_in, ye_in)
    
    np.testing.assert_almost_equal(xe_in, xe_out, 4)
    np.testing.assert_almost_equal(ye_in, ye_out, 4)

    # Test velocities. (almost the same code as errors
    vx_in = xe_in
    vy_in = ye_in
    vx_out, vy_out = trans.evaluate_vel(x_in, y_in, vx_in, vy_in)
    
    np.testing.assert_almost_equal(vx_in, vx_out, 4)
    np.testing.assert_almost_equal(vy_in, vy_out, 4)

    # Test velocity errors. 
    vxe_in = xe_in
    vye_in = ye_in
    vxe_out, vye_out = trans.evaluate_vel_err(x_in, y_in, vx_in, vy_in,
                                              xe_in, ye_in, vxe_in, vye_in)
    
    np.testing.assert_almost_equal(vx_in, vx_out, 4)
    np.testing.assert_almost_equal(vy_in, vy_out, 4)
    

    ##########
    # Check derive transform
    ##########
    x1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y1 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    x2 = x1 + 0.1
    y2 = y1 - 0.3

    x2[3] += 0.005
    x2[4] -= 0.004
    y2[3] += 0.006
    y2[4] -= 0.007

    trans = transforms.PolyTransform.derive_transform(x2, y2, x1, y1, 0)

    assert trans.order == 0
    assert len(trans.px.parameters) == 3
    assert len(trans.py.parameters) == 3

    x2_new, y2_new = trans.evaluate(x2, y2)

    np.testing.assert_almost_equal(x1, x2_new, 2)
    np.testing.assert_almost_equal(y1, y2_new, 2)
    
    return
    
# def test_legendre_conditioning():
#     x_old = np.arange(0, 900, 50)
#     y_old = np.arange(1, 901, 50)
#     domain = [0, 1000]

#     # 0.1 translation, 1.0 x-scale, 0.0 y-scale, no cross term (x*y)
#     px_init_d = {'c0_0':0.1, 'c1_0':1.0, 'c0_1':0.0, 'c1_1':0.0}
#     py_init_d = {'c0_0':0.5, 'c1_0':0.0, 'c0_1':1.0, 'c1_1':0.0}
#     px_init = list(px_init_d.values())
#     py_init = list(py_init_d.values())

#     # Make the model and setup the domain of the input data.
#     # This should be mapped onto a window of [-1, 1]; but I would expect
#     # this mapping to only be internal to the model.
#     foo = transforms.LegTransform(1, px_init, py_init, domain, domain, astropy_order=True)

#     # Test #1: Check evaluation 
#     x_new, y_new = foo.evaluate(x_old, y_old)
    
#     x_off, x_scl, y_off, y_scl = foo.convert_domain_to_scale_offset()
#     x_new_exp = x_old + (px_init_d['c0_0'] / x_scl)
#     y_new_exp = y_old + (py_init_d['c0_0'] / y_scl)
    
#     np.testing.assert_almost_equal(x_new, x_new_exp, 4)
#     np.testing.assert_almost_equal(y_new, y_new_exp, 4)
    
#     return

def test_legendre_errors():
    
    # Test #2: Check evaluate errors. They shouldn't change at all.
    # xe_old = np.abs( np.random.randn(len(x_old)) * 0.1 )
    # ye_old = np.abs( np.random.randn(len(x_old)) * 0.1 )

    # xe_new, ye_new = foo.evaluate_error(x_old, y_old, xe_old, ye_old)

    # np.testing.assert_almost_equal(xe_old, xe_new, 5)
    # np.testing.assert_almost_equal(ye_old, ye_new, 5)
    # pdb.set_trace()
    
    return


def test_PolyTransform():
    # test the transformation
    x = np.random.uniform(low=0,high=1000,size=1000)
    y = np.random.uniform(low=0,high=1000,size=1000)
    x_err = 0.1*np.random.normal(size=1000)
    y_err = 0.1*np.random.normal(size=1000)

    xref = (x + 100.0)*.5
    yref = (y + 100.0)*.5

    x+= x_err
    y+= y_err
    
    t = transforms.PolyTransform.derive_transform(x,y,xref,yref,2)
    x_trans_err,y_trans_err = t.evaluate_error(x,x_err,y,y_err)
    x_trans, y_trans = t.evaluate(x,y)

    np.testing.assert_allclose(xref,x_trans,atol=0.3,rtol=1e-2)
    np.testing.assert_allclose(yref,y_trans,atol=0.3,rtol=1e-2)
    
    for i in np.arange(len(x_trans)):
        print(( '%5.4f %5.4f %5.4f %5.4f %5.4f %5.4f' %
                    (x[i],x_trans[i],x_trans_err[i],y[i],y_trans[i],y_trans_err[i])))
        
    return

def test_LegTransform():
    # test the transformation
    # x = np.random.uniform(low=0, high=1000, size=1000)
    # y = np.random.uniform(low=0, high=1000, size=1000)
    x = np.arange(1, 1000, 1)
    y = np.arange(1000, 1, -1)
    x_err = np.random.normal(loc=0.0, scale=0.1, size=len(x))
    y_err = np.random.normal(loc=0.0, scale=0.1, size=len(x))
    xin = x + x_err
    yin = y + y_err

    # Test #1, simple 0th order shift
    xref = 100.0 + x
    yref = 100.0 + y
    
    t = transforms.LegTransform.derive_transform(xin, yin, xref, yref, 1)
    x_trans, y_trans = t.evaluate(xin, yin)

    np.testing.assert_allclose(xref, x_trans, atol=0.1, rtol=1e-2)
    np.testing.assert_allclose(yref, y_trans, atol=0.1, rtol=1e-2)
    
    # Test #2, 1st order 
    xref = 100.0 + x*0.92 + y*0.001
    yref = 100.0 + y*0.87 + x*0.023
    
    t = transforms.LegTransform.derive_transform(xin, yin, xref, yref, 1)
    x_trans, y_trans = t.evaluate(xin, yin)

    np.testing.assert_allclose(xref, x_trans, atol=0.1, rtol=1e-2)
    np.testing.assert_allclose(yref, y_trans, atol=0.1, rtol=1e-2)

    # Test #3, 1st order but fit with 2nd order
    t = transforms.LegTransform.derive_transform(xin, yin, xref, yref, 2)
    x_trans, y_trans = t.evaluate(xin, yin)

    np.testing.assert_allclose(xref, x_trans, atol=0.1, rtol=1e-2)
    np.testing.assert_allclose(yref, y_trans, atol=0.1, rtol=1e-2)
    
    return

    

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
