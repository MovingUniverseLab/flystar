from flystar import motion_model
import numpy as np
import pytest

def within_error(true_val, fit_val, fit_err, n_sigma=3):
    #print('True', true_val, 'Fit', fit_val, 'Fit err', fit_err)
    return (true_val < (fit_val+fit_err*n_sigma)) & (true_val> (fit_val-fit_err*n_sigma))

def test_Fixed():
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1}
    mod = motion_model.Fixed()
    param_list = mod.fitter_param_names
    fixed_param_list = mod.fixed_param_names
    # Confirm return of proper values for single t and array t
    x_t, y_t = mod.get_pos_at_time([true_params[p] for p in param_list],
                                        [true_params[p] for p in fixed_param_list], 0.0)
    assert x_t==true_params['x0']
    assert y_t==true_params['y0']
    x_t, y_t = mod.get_pos_at_time([true_params[p] for p in param_list],
                                        [true_params[p] for p in fixed_param_list],
                                        [0.0,2025.0,10000])
    assert (x_t==true_params['x0']).all()
    assert (y_t==true_params['y0']).all()
    
    # Check behavior of get_batch_pos_at_time
    x0_batch = np.random.uniform(-2.0,2.0, 50)
    y0_batch = np.random.uniform(-2.0,2.0, 50)
    x0_err_batch = np.repeat(0.1, 50)
    y0_err_batch = np.repeat(0.1, 50)
    # Single epoch
    t_batch=2020.0
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.get_batch_pos_at_time(t_batch,
                            x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch)
    assert (x_t_batch==x0_batch).all()
    assert (y_t_batch==y0_batch).all()
    assert (x_err_t_batch==x0_err_batch).all()
    assert (y_err_t_batch==y0_err_batch).all()
    # Multiple times
    t_batch = np.arange(2015.0,2025.0, 0.5)
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.get_batch_pos_at_time(t_batch,
                            x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch)
    assert (x_t_batch==np.array([np.repeat(x0_batch_i, len(t_batch)) for x0_batch_i in x0_batch])).all()
    assert (y_t_batch==np.array([np.repeat(y0_batch_i, len(t_batch)) for y0_batch_i in y0_batch])).all()
    assert (x_err_t_batch==np.array([np.repeat(x0_err_batch_i, len(t_batch)) for x0_err_batch_i in x0_err_batch])).all()
    assert (y_err_t_batch==np.array([np.repeat(y0_err_batch_i, len(t_batch)) for y0_err_batch_i in y0_err_batch])).all()
    
    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod.get_pos_at_time([true_params[p] for p in param_list],
                            [true_params[p] for p in fixed_param_list], t)
    x_sim = np.random.normal(x_true, true_params['x0_err'])
    y_sim = np.random.normal(y_true, true_params['y0_err'])
    # Run fit
    params, param_errs = mod.fit_motion_model(t, x_sim,y_sim,
                np.ones(len(t))*true_params['x0_err'],
                np.ones(len(t))*true_params['y0_err'],
                np.nan)
    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])


def test_Linear():
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1,
                    'vx':0.2, 'vy':0.5, 'vx_err':0.05, 'vy_err':0.05,
                    't0':2025.0}
    mod = motion_model.Linear()
    param_list = mod.fitter_param_names
    fixed_param_list = mod.fixed_param_names
    # Confirm return of proper values for single t=t0 and array t
    x_t, y_t = mod.get_pos_at_time([true_params[p] for p in param_list],
                                        [true_params[p] for p in fixed_param_list],
                                        true_params['t0'])
    assert x_t==true_params['x0']
    assert y_t==true_params['y0']
    t_arr = np.array([2010.0,true_params['t0'],2030.0])
    x_t, y_t = mod.get_pos_at_time([true_params[p] for p in param_list],
                                        [true_params[p] for p in fixed_param_list],
                                        t_arr)
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
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.get_batch_pos_at_time(t_batch,
                            x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch,
                            vx=vx_batch, vy=vy_batch, vx_err=vx_err_batch, vy_err=vy_err_batch,
                            t0=t0_batch)
    assert (x_t_batch==(x0_batch+(t_batch-t0_batch)*vx_batch)).all()
    assert (y_t_batch==(y0_batch+(t_batch-t0_batch)*vy_batch)).all()
    assert (x_err_t_batch==np.hypot(x0_err_batch, (t_batch-t0_batch)*vx_err_batch)).all()
    assert (y_err_t_batch==np.hypot(y0_err_batch, (t_batch-t0_batch)*vy_err_batch)).all()
    # Multiple times
    t_batch = np.arange(2015.0,2025.0, 0.5)
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.get_batch_pos_at_time(t_batch,
                            x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch,
                            vx=vx_batch, vy=vy_batch, vx_err=vx_err_batch, vy_err=vy_err_batch,
                            t0=t0_batch)
    assert (x_t_batch==np.array([x0_batch[i] + (t_batch-t0_batch[i])*vx_batch[i] for i in range(len(x0_batch))])).all()
    assert (y_t_batch==np.array([y0_batch[i] + (t_batch-t0_batch[i])*vy_batch[i] for i in range(len(x0_batch))])).all()
    assert (x_err_t_batch==np.array([np.hypot(x0_err_batch[i], (t_batch-t0_batch[i])*vx_err_batch[i]) for i in range(len(x0_batch))])).all()
    assert (y_err_t_batch==np.array([np.hypot(y0_err_batch[i], (t_batch-t0_batch[i])*vy_err_batch[i]) for i in range(len(x0_batch))])).all()
    
    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod.get_pos_at_time([true_params[p] for p in param_list],
                                        [true_params[p] for p in fixed_param_list],t)
    x_sim = np.random.normal(x_true, 0.05)
    y_sim = np.random.normal(y_true, 0.05)
    # Run fit
    params, param_errs = mod.fit_motion_model(t, x_sim,y_sim,
            np.repeat(0.05, len(t)), np.repeat(0.05,len(t)), true_params['t0'])
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
    params, param_errs = mod.fit_motion_model(t, x_sim,y_sim, x_true_err, y_true_err, true_params['t0'],bootstrap=10)
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
    param_list = mod.fitter_param_names
    fixed_param_list = mod.fixed_param_names
    # Confirm return of proper values for single t=t0 and array t
    x_t, y_t = mod.get_pos_at_time([true_params[p] for p in param_list],
                                        [true_params[p] for p in fixed_param_list],
                                        true_params['t0'])
    assert x_t==true_params['x0']
    assert y_t==true_params['y0']
    t_arr = np.array([2010.0,true_params['t0'],2030.0])
    x_t, y_t = mod.get_pos_at_time([true_params[p] for p in param_list],
                                        [true_params[p] for p in fixed_param_list],
                                        t_arr)
    assert (x_t==(true_params['x0'] + (t_arr-true_params['t0'])*true_params['vx0'] + 0.5*(t_arr-true_params['t0'])**2*true_params['ax'])).all()
    assert (y_t==(true_params['y0'] + (t_arr-true_params['t0'])*true_params['vy0'] + 0.5*(t_arr-true_params['t0'])**2*true_params['ay'])).all()
    
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
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.get_batch_pos_at_time(t_batch,
                            x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch,
                            vx0=vx0_batch, vy0=vy0_batch, vx0_err=vx0_err_batch, vy0_err=vy0_err_batch,
                            ax=ax_batch, ay=ay_batch, ax_err=ax_err_batch, ay_err=ay_err_batch,
                            t0=t0_batch)
    assert (x_t_batch==(x0_batch + (t_batch-t0_batch)*vx0_batch + 0.5*(t_batch-t0_batch)**2*ax_batch)).all()
    assert (y_t_batch==(y0_batch + (t_batch-t0_batch)*vy0_batch + 0.5*(t_batch-t0_batch)**2*ay_batch)).all()
    assert (x_err_t_batch==np.sqrt(x0_err_batch**2 + ((t_batch-t0_batch)*vx0_err_batch)**2 +
                                    (0.5*(t_batch-t0_batch)**2*ax_err_batch)**2)).all()
    assert (y_err_t_batch==np.sqrt(y0_err_batch**2 + ((t_batch-t0_batch)*vy0_err_batch)**2 +
                                    (0.5*(t_batch-t0_batch)**2*ay_err_batch)**2)).all()
    # Multiple times
    t_batch = np.arange(2015.0,2025.0, 0.5)
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.get_batch_pos_at_time(t_batch,
                            x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch,
                            vx0=vx0_batch, vy0=vy0_batch, vx0_err=vx0_err_batch, vy0_err=vy0_err_batch,
                            ax=ax_batch, ay=ay_batch, ax_err=ax_err_batch, ay_err=ay_err_batch,
                            t0=t0_batch)
    assert (x_t_batch==np.array([x0_batch[i] + (t_batch-t0_batch[i])*vx0_batch[i] + 0.5*(t_batch-t0_batch[i])**2*ax_batch[i] for i in range(len(x0_batch))])).all()
    assert (y_t_batch==np.array([y0_batch[i] + (t_batch-t0_batch[i])*vy0_batch[i] + 0.5*(t_batch-t0_batch[i])**2*ay_batch[i] for i in range(len(x0_batch))])).all()
    assert (x_err_t_batch==np.array([np.sqrt(x0_err_batch[i]**2 + ((t_batch-t0_batch[i])*vx0_err_batch[i])**2 + (0.5*(t_batch-t0_batch[i])**2*ax_err_batch[i])**2) for i in range(len(x0_batch))])).all()
    assert (y_err_t_batch==np.array([np.sqrt(y0_err_batch[i]**2 + ((t_batch-t0_batch[i])*vy0_err_batch[i])**2 + (0.5*(t_batch-t0_batch[i])**2*ay_err_batch[i])**2) for i in range(len(x0_batch))])).all()
    
    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod.get_pos_at_time([true_params[p] for p in param_list],
                                        [true_params[p] for p in fixed_param_list],t)
    x_true_err = np.sqrt(true_params['x0_err']**2 + ((t-true_params['t0'])*true_params['vx0_err'])**2 +
                            (0.5*(t-true_params['t0'])**2*true_params['ax_err'])**2)
    y_true_err = np.sqrt(true_params['y0_err']**2 + ((t-true_params['t0'])*true_params['vy0_err'])**2 +
                            (0.5*(t-true_params['t0'])**2*true_params['ay_err'])**2)
    x_sim = np.random.normal(x_true, x_true_err)
    y_sim = np.random.normal(y_true, y_true_err)
    # Run fit
    mod_fit = motion_model.Acceleration(t0=true_params['t0'])
    params, param_errs = mod_fit.fit_motion_model(t, x_sim,y_sim, x_true_err, y_true_err, true_params['t0'])
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
    param_list = mod.fitter_param_names
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
    params, param_errs = mod.fit_motion_model(t, x_sim,y_sim, x_true_err, y_true_err, true_params['t0'])
    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])


def test_Parallax_PA():
    # Set PA=0 model
    x0, y0 = 2.0, -1.0
    vx, vy = 0.2, 0.5
    RA, Dec = 17.76, -28.933
    pi = 0.5
    mod_pa0 = motion_model.Parallax(RA=RA,Dec=Dec, PA=0)
    # Set PA=90 model with equivalent parameters in that frame
    mod_pa90 = motion_model.Parallax(RA=RA,Dec=Dec,t0=2020.0, PA=90)
    t_set = np.arange(2018,2024,0.01)
    dat_pa0 = mod_pa0.get_pos_at_time([x0,vx,y0,vy,pi],[2020.0],t_set)
    dat_pa90 = mod_pa90.get_pos_at_time([y0,vy,-x0,-vx,pi],[2020.0],t_set)
    assert (np.abs(dat_pa0[0]-(-dat_pa90[1]))<1e-10).all()
    assert (np.abs(dat_pa0[1]-(dat_pa90[0]))<1e-10).all()
