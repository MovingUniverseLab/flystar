from flystar import motion_model
import numpy as np
import pytest

def within_error(true_val, fit_val, fit_err, n_sigma=1):
    return (true_val < (fit_val+fit_err)) & (true_val> (fit_val-fit_err))

def test_Fixed():
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1}
    mod_true = motion_model.Fixed(**true_params)
    param_list = mod_true.fitter_param_names
    # Confirm return of proper values for single t and array t
    x_t, y_t = mod_true.get_pos_at_time(0.0)
    assert x_t==true_params['x0']
    assert y_t==true_params['y0']
    x_t, y_t = mod_true.get_pos_at_time([0.0,mod_true.t0,10000])
    assert (x_t==true_params['x0']).all()
    assert (y_t==true_params['y0']).all()
    x_err_t, y_err_t = mod_true.get_pos_err_at_time(0.0)
    assert x_err_t==true_params['x0_err']
    assert y_err_t==true_params['y0_err']
    x_err_t, y_err_t = mod_true.get_pos_err_at_time([0.0,mod_true.t0,10000])
    assert (x_err_t==true_params['x0_err']).all()
    assert (y_err_t==true_params['y0_err']).all()
    
    # Check behavior of get_batch_pos_at_time
    x0_batch = np.random.uniform(-2.0,2.0, 50)
    y0_batch = np.random.uniform(-2.0,2.0, 50)
    x0_err_batch = np.repeat(0.1, 50)
    y0_err_batch = np.repeat(0.1, 50)
    # Single epoch
    t_batch=2020.0
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod_true.get_batch_pos_at_time(t_batch,
                            x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch)
    assert (x_t_batch==x0_batch).all()
    assert (y_t_batch==y0_batch).all()
    assert (x_err_t_batch==x0_err_batch).all()
    assert (y_err_t_batch==y0_err_batch).all()
    # Multiple times
    t_batch = np.arange(2015.0,2025.0, 0.5)
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod_true.get_batch_pos_at_time(t_batch,
                            x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch)
    assert (x_t_batch==np.array([np.repeat(x0_batch_i, len(t_batch)) for x0_batch_i in x0_batch])).all()
    assert (y_t_batch==np.array([np.repeat(y0_batch_i, len(t_batch)) for y0_batch_i in y0_batch])).all()
    assert (x_err_t_batch==np.array([np.repeat(x0_err_batch_i, len(t_batch)) for x0_err_batch_i in x0_err_batch])).all()
    assert (y_err_t_batch==np.array([np.repeat(y0_err_batch_i, len(t_batch)) for y0_err_batch_i in y0_err_batch])).all()
    
    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod_true.get_pos_at_time(t)
    x_true_err, y_true_err = mod_true.get_pos_err_at_time(t)
    x_sim = np.random.normal(x_true, x_true_err)
    y_sim = np.random.normal(y_true, y_true_err)
    # Run fit
    mod_fit = motion_model.Fixed()
    params, param_errs = mod_fit.fit_motion_model(t, x_sim,y_sim, x_true_err, y_true_err)
    # Confirm true value is within error bar of fit value
    assert [within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))]


def test_Linear():
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1,
                    'vx':0.2, 'vy':0.5, 'vx_err':0.05, 'vy_err':0.05}
    mod_true = motion_model.Linear(**true_params)
    param_list = mod_true.fitter_param_names
    # Confirm return of proper values for single t=t0 and array t
    x_t, y_t = mod_true.get_pos_at_time(mod_true.t0)
    assert x_t==true_params['x0']
    assert y_t==true_params['y0']
    t_arr = np.array([2010.0,mod_true.t0,2030.0])
    x_t, y_t = mod_true.get_pos_at_time(t_arr)
    assert (x_t==(true_params['x0'] + (t_arr-mod_true.t0)*true_params['vx'])).all()
    assert (y_t==(true_params['y0'] + (t_arr-mod_true.t0)*true_params['vy'])).all()
    x_err_t, y_err_t = mod_true.get_pos_err_at_time(mod_true.t0)
    assert x_err_t==true_params['x0_err']
    assert y_err_t==true_params['y0_err']
    x_err_t, y_err_t = mod_true.get_pos_err_at_time(t_arr)
    assert (x_err_t==np.hypot(true_params['x0_err'],(t_arr-mod_true.t0)*true_params['vx_err'])).all()
    assert (y_err_t==np.hypot(true_params['y0_err'],(t_arr-mod_true.t0)*true_params['vy_err'])).all()
    
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
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod_true.get_batch_pos_at_time(t_batch,
                            x0=x0_batch, y0=y0_batch, x0_err=x0_err_batch, y0_err=y0_err_batch,
                            vx=vx_batch, vy=vy_batch, vx_err=vx_err_batch, vy_err=vy_err_batch,
                            t0=t0_batch)
    assert (x_t_batch==(x0_batch+(t_batch-t0_batch)*vx_batch)).all()
    assert (y_t_batch==(y0_batch+(t_batch-t0_batch)*vy_batch)).all()
    assert (x_err_t_batch==np.hypot(x0_err_batch, (t_batch-t0_batch)*vx_err_batch)).all()
    assert (y_err_t_batch==np.hypot(y0_err_batch, (t_batch-t0_batch)*vy_err_batch)).all()
    # Multiple times
    t_batch = np.arange(2015.0,2025.0, 0.5)
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod_true.get_batch_pos_at_time(t_batch,
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
    x_true, y_true = mod_true.get_pos_at_time(t)
    x_true_err, y_true_err = mod_true.get_pos_err_at_time(t)
    x_sim = np.random.normal(x_true, x_true_err)
    y_sim = np.random.normal(y_true, y_true_err)
    # Run fit
    mod_fit = motion_model.Linear()
    params, param_errs = mod_fit.fit_motion_model(t, x_sim,y_sim, x_true_err, y_true_err)
    print(param_errs)
    # Confirm true value is within error bar of fit value
    assert [within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))]
    
    # Test fitter with bootstrap
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod_true.get_pos_at_time(t)
    x_true_err, y_true_err = mod_true.get_pos_err_at_time(t)
    x_sim = np.random.normal(x_true, x_true_err)
    y_sim = np.random.normal(y_true, y_true_err)
    # Run fit
    mod_fit = motion_model.Linear()
    params, param_errs = mod_fit.fit_motion_model(t, x_sim,y_sim, x_true_err, y_true_err,bootstrap=10)
    print(param_errs)
    # Confirm true value is within error bar of fit value
    assert [within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))]
    
    # Test fitter for 2 pts
    t = np.array([2015.0,2025.0])
    # Get values from model and add scatter
    x_true, y_true = mod_true.get_pos_at_time(t)
    x_true_err, y_true_err = mod_true.get_pos_err_at_time(t)
    x_sim = np.random.normal(x_true, x_true_err)
    y_sim = np.random.normal(y_true, y_true_err)
    # Run fit
    mod_fit = motion_model.Linear()
    params, param_errs = mod_fit.fit_motion_model(t, x_sim,y_sim, x_true_err, y_true_err)
    print(param_errs)
    # Confirm true value is within error bar of fit value
    assert [within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))]
    
    
def test_Acceleration():
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1,
                    'vx0':0.2, 'vy0':0.5, 'vx0_err':0.05, 'vy0_err':0.05,
                    'ax':0.1, 'ay':-0.1, 'ax_err':0.02, 'ay_err':0.02}
    mod_true = motion_model.Acceleration(**true_params)
    param_list = mod_true.fitter_param_names
    # Confirm return of proper values for single t=t0 and array t
    x_t, y_t = mod_true.get_pos_at_time(mod_true.t0)
    assert x_t==true_params['x0']
    assert y_t==true_params['y0']
    t_arr = np.array([2010.0,mod_true.t0,2030.0])
    x_t, y_t = mod_true.get_pos_at_time(t_arr)
    assert (x_t==(true_params['x0'] + (t_arr-mod_true.t0)*true_params['vx0'] + 0.5*(t_arr-mod_true.t0)**2*true_params['ax'])).all()
    assert (y_t==(true_params['y0'] + (t_arr-mod_true.t0)*true_params['vy0'] + 0.5*(t_arr-mod_true.t0)**2*true_params['ay'])).all()
    x_err_t, y_err_t = mod_true.get_pos_err_at_time(mod_true.t0)
    assert x_err_t==true_params['x0_err']
    assert y_err_t==true_params['y0_err']
    x_err_t, y_err_t = mod_true.get_pos_err_at_time(t_arr)
    assert (x_err_t==np.sqrt(true_params['x0_err']**2 + ((t_arr-mod_true.t0)*true_params['vx0_err'])**2 +
                            (0.5*(t_arr-mod_true.t0)**2*true_params['ax_err'])**2)).all()
    assert (y_err_t==np.sqrt(true_params['y0_err']**2 + ((t_arr-mod_true.t0)*true_params['vy0_err'])**2 +
                            (0.5*(t_arr-mod_true.t0)**2*true_params['ay_err'])**2)).all()
    
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
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod_true.get_batch_pos_at_time(t_batch,
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
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod_true.get_batch_pos_at_time(t_batch,
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
    x_true, y_true = mod_true.get_pos_at_time(t)
    x_true_err, y_true_err = mod_true.get_pos_err_at_time(t)
    x_sim = np.random.normal(x_true, x_true_err)
    y_sim = np.random.normal(y_true, y_true_err)
    # Run fit
    mod_fit = motion_model.Acceleration()
    params, param_errs = mod_fit.fit_motion_model(t, x_sim,y_sim, x_true_err, y_true_err)
    # Confirm true value is within error bar of fit value
    assert [within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))]
    
#@pytest.mark.skip(reason="not written")
def test_Parallax():
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':-0.5, 'x0_err':0.1, 'y0_err':0.1,
                    'vx':-0.2, 'vy':0.5, 'vx_err':0.05, 'vy_err':0.05,
                    'pi':0.5, 'RA':17.76, 'Dec':-28.933, 'PA':0}
    mod_true = motion_model.Parallax(**true_params)
    param_list = mod_true.fitter_param_names
    
    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod_true.get_pos_at_time(t)
    x_true_err, y_true_err = mod_true.get_pos_err_at_time(t)
    x_sim = np.random.normal(x_true, x_true_err)
    y_sim = np.random.normal(y_true, y_true_err)
    # Run fit
    mod_fit = motion_model.Parallax(RA=17.76, Dec=-28.933, PA=0)
    params, param_errs = mod_fit.fit_motion_model(t, x_sim,y_sim, x_true_err, y_true_err)
    # Confirm true value is within error bar of fit value
    assert [within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))]


def test_Parallax_PA():
    # Set PA=0 model
    x0, y0 = 2.0, -1.0
    vx, vy = 0.2, 0.5
    RA, Dec = 17.76, -28.933
    pi = 0.5
    mod_pa0 = motion_model.Parallax(x0=x0,vx=vx,y0=y0,vy=vy,pi=pi,
                                    RA=RA,Dec=Dec,t0=2020.0, PA=0)
    # Set PA=90 model with equivalent parameters in that frame
    mod_pa90 = motion_model.Parallax(x0=y0,vx=vy,y0=-x0,vy=-vx,pi=pi,
                                    RA=RA,Dec=Dec,t0=2020.0, PA=90)
    t_set = np.arange(2018,2024,0.01)
    dat_pa0 = mod_pa0.get_pos_at_time(t_set)
    dat_pa90 = mod_pa90.get_pos_at_time(t_set)
    assert (np.abs(dat_pa0[0]-(-dat_pa90[1]))<1e-10).all()
    assert (np.abs(dat_pa0[1]-(dat_pa90[0]))<1e-10).all()
