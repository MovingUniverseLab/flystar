from flystar import motion_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def within_error(true_val, fit_val, fit_err, n_sigma=3):
    #print('True', true_val, 'Fit', fit_val, 'Fit err', fit_err)
    # return (true_val < (fit_val + fit_err*n_sigma)) & (true_val > (fit_val - fit_err*n_sigma))
    return np.abs(true_val - fit_val) <= n_sigma*fit_err

def test_Fixed():
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1}
    mod = motion_model.Fixed()
    param_list = mod.fit_param_names
    fixed_param_list = mod.fixed_param_names
    # Confirm return of proper values for single t and array t
    # x_t, y_t = mod.get_pos_at_time([true_params[p] for p in param_list],
    #                                     [true_params[p] for p in fixed_param_list], 0.0)
    x_t, y_t = mod.model(
        0.0,
        fit_params=np.array([true_params['x0'], true_params['y0']]).T
    )
    assert x_t==true_params['x0']
    assert y_t==true_params['y0']
    # x_t, y_t = mod.get_pos_at_time([true_params[p] for p in param_list],
    #                                     [true_params[p] for p in fixed_param_list],
    #                                     [0.0,2025.0,10000])
    x_t, y_t = mod.model(
        [0.0,2025.0,10000],
        fit_params=np.array([true_params['x0'], true_params['y0']]).T
    )
    assert (x_t==true_params['x0']).all()
    assert (y_t==true_params['y0']).all()
    
    # Check behavior of get_batch_pos_at_time
    x0_batch = np.random.uniform(-2.0,2.0, 50)
    y0_batch = np.random.uniform(-2.0,2.0, 50)
    x0_err_batch = np.repeat(0.1, 50)
    y0_err_batch = np.repeat(0.1, 50)
    # Single epoch
    t_batch=2020.0
    # x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.get_batch_pos_at_time(t_batch,
    #                         x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch)
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.model(
        t_batch,
        fit_params=np.array([x0_batch, y0_batch]).T,
        fit_param_errs=np.array([x0_err_batch, y0_err_batch]).T
    )
    assert (x_t_batch==x0_batch).all()
    assert (y_t_batch==y0_batch).all()
    assert (x_err_t_batch==x0_err_batch).all()
    assert (y_err_t_batch==y0_err_batch).all()
    # Multiple times
    t_batch = np.arange(2015.0,2025.0, 0.5)
    # x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.get_batch_pos_at_time(t_batch,
    #                         x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch)
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.model(
        t_batch,
        fit_params=np.array([x0_batch, y0_batch]).T,
        fit_param_errs=np.array([x0_err_batch, y0_err_batch]).T
    )
    assert (x_t_batch==np.array([np.repeat(x0_batch_i, len(t_batch)) for x0_batch_i in x0_batch])).all()
    assert (y_t_batch==np.array([np.repeat(y0_batch_i, len(t_batch)) for y0_batch_i in y0_batch])).all()
    assert (x_err_t_batch==np.array([np.repeat(x0_err_batch_i, len(t_batch)) for x0_err_batch_i in x0_err_batch])).all()
    assert (y_err_t_batch==np.array([np.repeat(y0_err_batch_i, len(t_batch)) for y0_err_batch_i in y0_err_batch])).all()
    
    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    # x_true, y_true = mod.get_pos_at_time([true_params[p] for p in param_list],
    #                         [true_params[p] for p in fixed_param_list], t)
    x_true, y_true = mod.model(
        t, 
        fit_params=np.array([true_params['x0'], true_params['y0']])
    )
    x_sim = np.random.normal(x_true, true_params['x0_err'])
    y_sim = np.random.normal(y_true, true_params['y0_err'])
    # Run fit
    params, param_errs, _ , _ = mod.fit(
        t, 
        x_sim,y_sim,
        xe=np.ones(len(t))*true_params['x0_err'],
        ye=np.ones(len(t))*true_params['y0_err'],
        t0=np.nan
    )
    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])


def test_Linear():
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1,
                    'vx':0.2, 'vy':0.5, 'vx_err':0.05, 'vy_err':0.05,
                    't0':2025.0}
    mod = motion_model.Linear()
    param_list = mod.fit_param_names
    fixed_param_list = mod.fixed_param_names
    # Confirm return of proper values for single t=t0 and array t
    x_t, y_t = mod.model(
        t=true_params['t0'],
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params=np.array([true_params[p] for p in fixed_param_list]).T
    )
    assert x_t==true_params['x0']
    assert y_t==true_params['y0']
    t_arr = np.array([2010.0,true_params['t0'],2030.0])
    x_t, y_t = mod.model(
        t=t_arr,
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params=np.array([true_params[p] for p in fixed_param_list]).T
    )
    assert (x_t==(true_params['x0'] + (t_arr-true_params['t0'])*true_params['vx'])).all()
    assert (y_t==(true_params['y0'] + (t_arr-true_params['t0'])*true_params['vy'])).all()

    # Check behavior of get_batch_pos_at_time
    x0_batch = np.random.uniform(-2.0,2.0, 50)
    y0_batch = np.random.uniform(-2.0,2.0, 50)
    vx_batch = np.random.uniform(-2.0,2.0, 50)
    vy_batch = np.random.uniform(-2.0,2.0, 50)
    x0_err_batch = np.repeat(0.1, 50)
    y0_err_batch = np.repeat(0.1, 50)
    vx_err_batch = np.repeat(0.05, 50)
    vy_err_batch = np.repeat(0.05, 50)
    t0_batch = np.repeat(2025.0,50)
    # Single epoch
    t_batch=2020.0
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.model(
        t=t_batch,
        fit_params=np.array([x0_batch, vx_batch, y0_batch, vy_batch]).T,
        fit_param_errs=np.array([x0_err_batch, vx_err_batch, y0_err_batch, vy_err_batch]).T,
        fixed_params=t0_batch
    )
    assert (x_t_batch==(x0_batch+(t_batch-t0_batch)*vx_batch)).all()
    assert (y_t_batch==(y0_batch+(t_batch-t0_batch)*vy_batch)).all()
    assert (x_err_t_batch==np.hypot(x0_err_batch, (t_batch-t0_batch)*vx_err_batch)).all()
    assert (y_err_t_batch==np.hypot(y0_err_batch, (t_batch-t0_batch)*vy_err_batch)).all()
    # Multiple times
    t_batch = np.arange(2015.0,2025.0, 0.5)
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.model(
        t=t_batch,
        fit_params=np.array([x0_batch, vx_batch, y0_batch, vy_batch]).T,
        fit_param_errs=np.array([x0_err_batch, vx_err_batch, y0_err_batch, vy_err_batch]).T,
        fixed_params=t0_batch
    )
    assert (x_t_batch==np.array([x0_batch[i] + (t_batch-t0_batch[i])*vx_batch[i] for i in range(len(x0_batch))])).all()
    assert (y_t_batch==np.array([y0_batch[i] + (t_batch-t0_batch[i])*vy_batch[i] for i in range(len(x0_batch))])).all()
    assert (x_err_t_batch==np.array([np.hypot(x0_err_batch[i], (t_batch-t0_batch[i])*vx_err_batch[i]) for i in range(len(x0_batch))])).all()
    assert (y_err_t_batch==np.array([np.hypot(y0_err_batch[i], (t_batch-t0_batch[i])*vy_err_batch[i]) for i in range(len(x0_batch))])).all()
    
    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    # x_true, y_true = mod.get_pos_at_time([true_params[p] for p in param_list],
    #                                     [true_params[p] for p in fixed_param_list],t)
    x_true, y_true = mod.model(
        t=t,
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params=np.array([true_params[p] for p in fixed_param_list]).T
    )
    x_sim = np.random.normal(x_true, 0.05)
    y_sim = np.random.normal(y_true, 0.05)
    # Run fit
    params, param_errs, _, _ = mod.fit(
        t=t, 
        x=x_sim,
        y=y_sim,
        xe=np.repeat(0.05, len(t)), 
        ye=np.repeat(0.05,len(t)), 
        t0=true_params['t0']
    )
    print(param_errs)
    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])
    
    # Test fitter with bootstrap
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod.get_pos_at_time([true_params[p] for p in param_list],
                                        [true_params[p] for p in fixed_param_list],t)
    x_true_err, y_true_err = np.repeat(0.05,len(t)), np.repeat(0.05,len(t))
    x_sim = np.random.normal(x_true, x_true_err)
    y_sim = np.random.normal(y_true, y_true_err)
    # Run fit
    params, param_errs = mod.fit(t, x_sim,y_sim, x_true_err, y_true_err, true_params['t0'],bootstrap=10)
    print(param_errs)
    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])

#    # Test fitter for 2 pts
#    t = np.array([2015.0,2025.0])
#    # Get values from model and add scatter
#    x_true, y_true = mod.get_pos_at_time([true_params[p] for p in param_list],
#                                        [true_params[p] for p in fixed_param_list],t)
#    x_true_err, y_true_err = np.repeat(0.05,len(t)), np.repeat(0.05,len(t))
#    x_sim = np.random.normal(x_true, x_true_err)
#    y_sim = np.random.normal(y_true, y_true_err)
#    # Run fit
#    mod_fit = motion_model.Linear(t0=true_params['t0'])
#    params, param_errs = mod_fit.fit_motion_model(t, x_sim,y_sim, x_true_err, y_true_err, true_params['t0'])
#    print("DJSKBGJ",param_list)
#    print([true_params[p] for p in param_list])
#    print(params)
#    print(param_errs)
#    # Confirm true value is within error bar of fit value
#    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params),2)])
    
    
def test_Acceleration():
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1,
                    'vx0':0.2, 'vy0':0.5, 'vx0_err':0.05, 'vy0_err':0.05,
                    'ax':0.1, 'ay':-0.1, 'ax_err':0.02, 'ay_err':0.02,
                    't0':2025.0}
    mod = motion_model.Acceleration()
    param_list = mod.fit_param_names
    fixed_param_list = mod.fixed_param_names
    # Confirm return of proper values for single t=t0 and array t
    x_t, y_t = mod.model(
        t=true_params['t0'],
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params=np.array([true_params[p] for p in fixed_param_list]).T
    )
    np.testing.assert_allclose(x_t, true_params['x0'])
    np.testing.assert_allclose(y_t, true_params['y0'])
    t_arr = np.array([2010.0, true_params['t0'], 2030.0])
    x_t, y_t = mod.model(
        t=t_arr,
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params=np.array([true_params[p] for p in fixed_param_list]).T
    )
    np.testing.assert_allclose(x_t, true_params['x0'] + (t_arr-true_params['t0'])*true_params['vx0'] + 0.5*(t_arr-true_params['t0'])**2*true_params['ax'])
    np.testing.assert_allclose(y_t, true_params['y0'] + (t_arr-true_params['t0'])*true_params['vy0'] + 0.5*(t_arr-true_params['t0'])**2*true_params['ay'])
    
    # Check behavior of get_batch_pos_at_time
    x0_batch = np.random.uniform(-2.0,2.0, 50)
    y0_batch = np.random.uniform(-2.0,2.0, 50)
    vx0_batch = np.random.uniform(-2.0,2.0, 50)
    vy0_batch = np.random.uniform(-2.0,2.0, 50)
    ax_batch = np.random.uniform(-1.0,1.0, 50)
    ay_batch = np.random.uniform(-1.0,1.0, 50)
    x0_err_batch = np.repeat(0.1, 50)
    y0_err_batch = np.repeat(0.1, 50)
    vx0_err_batch = np.repeat(0.05, 50)
    vy0_err_batch = np.repeat(0.05, 50)
    ax_err_batch = np.repeat(0.02, 50)
    ay_err_batch = np.repeat(0.02, 50)
    t0_batch = np.repeat(2025.0,50)
    # Single epoch
    t_batch=2020.0
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.model(
        t=t_batch,
        fit_params=np.array([x0_batch, vx0_batch, ax_batch, y0_batch, vy0_batch, ay_batch]).T,
        fit_param_errs=np.array([x0_err_batch, vx0_err_batch, ax_err_batch, y0_err_batch, vy0_err_batch, ay_err_batch]).T,
        fixed_params=t0_batch
    )
    np.testing.assert_allclose(x_t_batch, x0_batch + (t_batch-t0_batch)*vx0_batch + 0.5*(t_batch-t0_batch)**2*ax_batch)
    np.testing.assert_allclose(y_t_batch, y0_batch + (t_batch-t0_batch)*vy0_batch + 0.5*(t_batch-t0_batch)**2*ay_batch)
    np.testing.assert_allclose(x_err_t_batch, np.sqrt(x0_err_batch**2 + ((t_batch-t0_batch)*vx0_err_batch)**2 +
                                    (0.5*(t_batch-t0_batch)**2*ax_err_batch)**2))
    np.testing.assert_allclose(y_err_t_batch, np.sqrt(y0_err_batch**2 + ((t_batch-t0_batch)*vy0_err_batch)**2 +
                                    (0.5*(t_batch-t0_batch)**2*ay_err_batch)**2))

    # Multiple times
    t_batch = np.arange(2015.0,2025.0, 0.5)
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.model(
        t=t_batch,
        fit_params=np.array([x0_batch, vx0_batch, ax_batch, y0_batch, vy0_batch, ay_batch]).T,
        fit_param_errs=np.array([x0_err_batch, vx0_err_batch, ax_err_batch, y0_err_batch, vy0_err_batch, ay_err_batch]).T,
        fixed_params=t0_batch
    )
    np.testing.assert_allclose(x_t_batch, np.array([x0_batch[i] + (t_batch-t0_batch[i])*vx0_batch[i] + 0.5*(t_batch-t0_batch[i])**2*ax_batch[i] for i in range(len(x0_batch))]))
    np.testing.assert_allclose(y_t_batch, np.array([y0_batch[i] + (t_batch-t0_batch[i])*vy0_batch[i] + 0.5*(t_batch-t0_batch[i])**2*ay_batch[i] for i in range(len(x0_batch))]))
    np.testing.assert_allclose(x_err_t_batch, np.array([np.sqrt(x0_err_batch[i]**2 + ((t_batch-t0_batch[i])*vx0_err_batch[i])**2 + (0.5*(t_batch-t0_batch[i])**2*ax_err_batch[i])**2) for i in range(len(x0_batch))]))
    np.testing.assert_allclose(y_err_t_batch, np.array([np.sqrt(y0_err_batch[i]**2 + ((t_batch-t0_batch[i])*vy0_err_batch[i])**2 + (0.5*(t_batch-t0_batch[i])**2*ay_err_batch[i])**2) for i in range(len(x0_batch))]))

    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod.model(
        t=t,
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params=np.array([true_params[p] for p in fixed_param_list]).T
    )
    x_true_err = np.sqrt(true_params['x0_err']**2 + ((t - true_params['t0']) * true_params['vx0_err'])**2 +
                            (0.5*(t - true_params['t0'])**2 * true_params['ax_err'])**2)
    y_true_err = np.sqrt(true_params['y0_err']**2 + ((t - true_params['t0']) * true_params['vy0_err'])**2 +
                            (0.5*(t - true_params['t0'])**2 * true_params['ay_err'])**2)
    x_sim = np.random.normal(x_true, x_true_err)
    y_sim = np.random.normal(y_true, y_true_err)
    # Run fit
    mod_fit = motion_model.Acceleration()
    params, param_errs, _, _ = mod_fit.fit(
        t=t, 
        x=x_sim,
        y=y_sim,
        xe=x_true_err, 
        ye=y_true_err, 
        t0=true_params['t0']
    )
    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])
    
#@pytest.mark.skip(reason="not written")
def test_Parallax():
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':-0.5, 'x0_err':0.1, 'y0_err':0.1,
                    'vx':-0.2, 'vy':0.5, 'vx_err':0.05, 'vy_err':0.05,
                    'pi':0.5, 'RA':17.76, 'Dec':-28.933, 'PA':0,
                    't0':2020.0}
    mod = motion_model.Parallax(**{'RA':17.76, 'Dec':-28.933, 'PA':0})
    param_list = mod.fit_param_names
    fixed_param_list = mod.fixed_param_names
    print(param_list)
    
    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod.get_pos_at_time([true_params[p] for p in param_list],
                                        [true_params[p] for p in fixed_param_list],t)
    x_true_err, y_true_err = np.repeat(0.1,len(t)), np.repeat(0.1,len(t))
    x_sim = np.random.normal(x_true, x_true_err)
    y_sim = np.random.normal(y_true, y_true_err)
    # Run fit
    params, param_errs = mod.fit(t, x_sim,y_sim, x_true_err, y_true_err, true_params['t0'])
    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])


def test_Parallax_PA():
    # Set PA=0 model
    x0, y0 = 2.0, -1.0
    vx, vy = 0.2, 0.5
    ra, dec = 17.76, -28.933
    pi = 0.5
    mod_pa0 = motion_model.Parallax(ra=ra, dec=dec, pa=0)
    # Set PA=90 model with equivalent parameters in that frame
    mod_pa90 = motion_model.Parallax(ra=ra, dec=dec, pa=90)
    
    t_set = np.arange(2018, 2024, 0.01)
    dat_pa0 = mod_pa0.model(
        t = t_set,
        fit_params = np.array([x0, vx, y0, vy, pi]).T,
        fixed_params = [2020.0]
    )
    dat_pa90 = mod_pa90.model(
        t = t_set,
        fit_params = np.array([y0, vy, -x0, -vx, pi]).T,
        fixed_params = [2020.0]
    )
    np.testing.assert_allclose(dat_pa0[0], -dat_pa90[1], atol=1e-10)
    np.testing.assert_allclose(dat_pa0[1], dat_pa90[0], atol=1e-10)


def test_Linear_fit_vs_scipy():
    # Compare Linear fit results to scipy curve_fit results
    t = np.array([0, 1., 2.2, 3.5, 5.])

    x = np.array([
        [0., 0.5, 2.1, 3.2, 6.0],                   # Increasing 5 Epochs
        [10.0, 8.9, 9.2, 7.4, 7.0],                 # Decreasing 5 Epochs
        [2.5, np.nan, 5.2, np.nan, 5.0],            # 3 Epochs
        [np.nan, 6.2, np.nan, np.nan, 9.2],         # 2 Epochs
        # [np.nan, 2.0, np.nan, np.nan, np.nan],      # 1 Epoch
        # [np.nan, np.nan, np.nan, np.nan, np.nan]   # All NaNs
    ])

    y = np.array([
        [10.2, 8.5, 9.1, 12.2, 13.0],                   # Increasing 5 Epochs
        [8.0, 9.9, 8.2, 7.4, 7.0],                 # Decreasing 5 Epochs
        [5.2, np.nan, 4.7, np.nan, 6.0],            # 3 Epochs
        [np.nan, 1.2, np.nan, np.nan, 3.2],         # 2 Epochs
        # [np.nan, 2.0, np.nan, np.nan, np.nan],      # 1 Epoch
        # [np.nan, np.nan, np.nan, np.nan, np.nan]   # All NaNs
    ])

    xe = np.array([
        [0.2, 0.5, 0.3, 0.4, 0.6],
        [0.5, 0.2, 0.7, 0.3, 0.2],
        [0.5, np.nan, 0.6, np.nan, 0.3],
        [np.nan, 0.6, np.nan, np.nan, 0.3],
        # [np.nan, 0.4, np.nan, np.nan, np.nan],
        # [np.nan, np.nan, np.nan, np.nan, np.nan]
    ])

    ye = np.array([
        [0.3, 0.2, 0.5, 0.2, 0.4],
        [0.2, 0.5, 0.6, 0.4, 0.2],
        [0.7, np.nan, 0.5, np.nan, 0.2],
        [np.nan, 0.4, np.nan, np.nan, 0.5],
        # [np.nan, 0.5, np.nan, np.nan, np.nan],
        # [np.nan, np.nan, np.nan, np.nan, np.nan]
    ])

    x = np.ma.masked_invalid(x)
    y = np.ma.masked_invalid(y)
    xe = np.ma.masked_invalid(xe)
    ye = np.ma.masked_invalid(ye)
    mask = np.ma.getmaskarray(x) | np.ma.getmaskarray(y) | np.ma.getmaskarray(xe) | np.ma.getmaskarray(ye)

    # tab = StarTable({
    #     'x': x,
    #     'y': y,
    #     'xe': xe,
    #     'ye': ye
    # })
    # tab.meta['LIST_TIMES'] = t
    # tab.fit_velocities(use_scipy=True, absolute_sigma=True)

    # Plot data
    N = x.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for i in range(N):
        line_mask = ~np.isnan(x[i]) & ~mask[i]
        ax1.errorbar(t[line_mask], x[i][line_mask], yerr=xe[i][line_mask], fmt='o-', label=f'Line {i}')
        ax2.errorbar(t[line_mask], y[i][line_mask], yerr=ye[i][line_mask], fmt='o-', label=f'Line {i}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position')
    ax1.legend()
    ax1.set_title('X vs Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position')
    ax2.legend()
    ax2.set_title('Y vs Time')
    plt.show()

    N = len(x)
    t0 = np.average(np.broadcast_to(t, x.shape), weights=1./np.hypot(xe, ye), axis=1)
    dt = np.zeros_like(x)

    # velfit
    # vx_velfit = np.zeros(N)
    # vxe_velfit = np.zeros(N)
    # vy_velfit = np.zeros(N)
    # vye_velfit = np.zeros(N)
    # x0_velfit = np.zeros(N)
    # x0e_velfit = np.zeros(N)
    # y0_velfit = np.zeros(N)
    # y0e_velfit = np.zeros(N)

    # scipy
    vx_scipy = np.zeros(N)
    vxe_scipy = np.zeros(N)
    vy_scipy = np.zeros(N)
    vye_scipy = np.zeros(N)
    x0_scipy = np.zeros(N)
    x0e_scipy = np.zeros(N)
    y0_scipy = np.zeros(N)
    y0e_scipy = np.zeros(N)

    # motion_model
    mm = motion_model.Linear()

    vx_mm_scipy = np.zeros(N)
    vxe_mm_scipy = np.zeros(N)
    vy_mm_scipy = np.zeros(N)
    vye_mm_scipy = np.zeros(N)
    x0_mm_scipy = np.zeros(N)
    x0e_mm_scipy = np.zeros(N)
    y0_mm_scipy = np.zeros(N)
    y0e_mm_scipy = np.zeros(N)

    vx_mm = np.zeros(N)
    vxe_mm = np.zeros(N)
    vy_mm = np.zeros(N)
    vye_mm = np.zeros(N)
    x0_mm = np.zeros(N)
    x0e_mm = np.zeros(N)
    y0_mm = np.zeros(N)
    y0e_mm = np.zeros(N)

    def linear(t, c0, c1):
        return c0 + c1*t

    # Absolute sigma
    for absolute_sigma in [True, False]:
        for i in range(N):
            dt[i] = t - t0[i]

            # # velfit.linear_fit
            # vx_velfit_results = linear_fit(dt[i][~mask[i]], x[i][~mask[i]], sigma=xe[i][~mask[i]], absolute_sigma=absolute_sigma)
            # vy_velfit_results = linear_fit(dt[i][~mask[i]], y[i][~mask[i]], sigma=ye[i][~mask[i]], absolute_sigma=absolute_sigma)

            # vx_velfit[i] = vx_velfit_results['slope']
            # vxe_velfit[i] = vx_velfit_results['e_slope']
            # vy_velfit[i] = vy_velfit_results['slope']
            # vye_velfit[i] = vy_velfit_results['e_slope']
            # x0_velfit[i] = vx_velfit_results['intercept']
            # x0e_velfit[i] = vx_velfit_results['e_intercept']
            # y0_velfit[i] = vy_velfit_results['intercept']
            # y0e_velfit[i] = vy_velfit_results['e_intercept']
            
            # scipy.curve_fit
            p0x = np.array([0., x[i][~mask[i]].mean()])
            p0y = np.array([0., y[i][~mask[i]].mean()])
            popt_x, pcov_x = curve_fit(linear, dt[i][~mask[i]], x[i][~mask[i]], p0=p0x, sigma=xe[i][~mask[i]], absolute_sigma=absolute_sigma)
            vx_scipy[i], vxe_scipy[i] = popt_x[1], np.sqrt(pcov_x[1, 1])
            x0_scipy[i], x0e_scipy[i] = popt_x[0], np.sqrt(pcov_x[0, 0])
            popt_y, pcov_y = curve_fit(linear, dt[i][~mask[i]], y[i][~mask[i]], p0=p0y, sigma=ye[i][~mask[i]], absolute_sigma=absolute_sigma)
            vy_scipy[i], vye_scipy[i] = popt_y[1], np.sqrt(pcov_y[1, 1])
            y0_scipy[i], y0e_scipy[i] = popt_y[0], np.sqrt(pcov_y[0, 0])

            # motion_model without scipy
            params, param_errs = mm.fit(
                t[~mask[i]], x[i][~mask[i]], y[i][~mask[i]], 
                xe[i][~mask[i]], ye[i][~mask[i]], t0[i], 
                weighting='var', 
                use_scipy=False, 
                absolute_sigma=absolute_sigma
            )
            vx_mm[i] = params[mm.fit_param_names.index('vx')]
            vy_mm[i] = params[mm.fit_param_names.index('vy')]
            vxe_mm[i] = param_errs[mm.fit_param_names.index('vx')]
            vye_mm[i] = param_errs[mm.fit_param_names.index('vy')]
            x0_mm[i] = params[mm.fit_param_names.index('x0')]
            y0_mm[i] = params[mm.fit_param_names.index('y0')]
            x0e_mm[i] = param_errs[mm.fit_param_names.index('x0')]
            y0e_mm[i] = param_errs[mm.fit_param_names.index('y0')]

            # motion_model with scipy
            params, param_errs = mm.fit(
                t[~mask[i]], x[i][~mask[i]], y[i][~mask[i]], 
                xe[i][~mask[i]], ye[i][~mask[i]], t0[i], 
                weighting='var', 
                use_scipy=True, 
                absolute_sigma=absolute_sigma
            )
            vx_mm_scipy[i] = params[mm.fit_param_names.index('vx')]
            vy_mm_scipy[i] = params[mm.fit_param_names.index('vy')]
            vxe_mm_scipy[i] = param_errs[mm.fit_param_names.index('vx')]
            vye_mm_scipy[i] = param_errs[mm.fit_param_names.index('vy')]
            x0_mm_scipy[i] = params[mm.fit_param_names.index('x0')]
            y0_mm_scipy[i] = params[mm.fit_param_names.index('y0')]
            x0e_mm_scipy[i] = param_errs[mm.fit_param_names.index('x0')]
            y0e_mm_scipy[i] = param_errs[mm.fit_param_names.index('y0')]

        rtol = 1e-5
        # np.testing.assert_allclose(vx_velfit, vx_scipy, rtol=rtol)
        # np.testing.assert_allclose(vxe_velfit, vxe_scipy, rtol=rtol)
        # np.testing.assert_allclose(vy_velfit, vy_scipy, rtol=rtol)
        # np.testing.assert_allclose(vye_velfit, vye_scipy, rtol=rtol)
        # np.testing.assert_allclose(x0_velfit, x0_scipy, rtol=rtol)
        # np.testing.assert_allclose(x0e_velfit, x0e_scipy, rtol=rtol)
        # np.testing.assert_allclose(y0_velfit, y0_scipy, rtol=rtol)
        # np.testing.assert_allclose(y0e_velfit, y0e_scipy, rtol=rtol)
        # np.testing.assert_allclose(vx_velfit, vx_mm, rtol=rtol)
        # np.testing.assert_allclose(vxe_velfit, vxe_mm, rtol=rtol)
        # np.testing.assert_allclose(vy_velfit, vy_mm, rtol=rtol)
        # np.testing.assert_allclose(vye_velfit, vye_mm, rtol=rtol)
        # np.testing.assert_allclose(x0_velfit, x0_mm, rtol=rtol)
        # np.testing.assert_allclose(x0e_velfit, x0e_mm, rtol=rtol)
        # np.testing.assert_allclose(y0_velfit, y0_mm, rtol=rtol)
        # np.testing.assert_allclose(y0e_velfit, y0e_mm, rtol=rtol)
        np.testing.assert_allclose(vx_scipy, vx_mm, rtol=rtol)
        np.testing.assert_allclose(vxe_scipy, vxe_mm, rtol=rtol)
        np.testing.assert_allclose(vy_scipy, vy_mm, rtol=rtol)
        np.testing.assert_allclose(vye_scipy, vye_mm, rtol=rtol)
        np.testing.assert_allclose(x0_scipy, x0_mm, rtol=rtol)
        np.testing.assert_allclose(x0e_scipy, x0e_mm, rtol=rtol)
        np.testing.assert_allclose(y0_scipy, y0_mm, rtol=rtol)
        np.testing.assert_allclose(y0e_scipy, y0e_mm, rtol=rtol)
        np.testing.assert_allclose(vx_scipy, vx_mm_scipy, rtol=rtol)
        np.testing.assert_allclose(vxe_scipy, vxe_mm_scipy, rtol=rtol)
        np.testing.assert_allclose(vy_scipy, vy_mm_scipy, rtol=rtol)
        np.testing.assert_allclose(vye_scipy, vye_mm_scipy, rtol=rtol)
        np.testing.assert_allclose(x0_scipy, x0_mm_scipy, rtol=rtol)
        np.testing.assert_allclose(x0e_scipy, x0e_mm_scipy, rtol=rtol)
        np.testing.assert_allclose(y0_scipy, y0_mm_scipy, rtol=rtol)
        np.testing.assert_allclose(y0e_scipy, y0e_mm_scipy, rtol=rtol)
