from flystar import motion_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def within_error(true_val, fit_val, fit_err, n_sigma=3):
    return np.abs(true_val - fit_val) <= n_sigma*fit_err

def test_Fixed():
    rng = np.random.default_rng(0)
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1}
    mod = motion_model.Fixed()
    param_list = mod.fit_param_names
    # Confirm return of proper values for single t and array t
    x_t, y_t = mod.model(
        0.0,
        fit_params=np.array([true_params['x0'], true_params['y0']]).T
    )
    assert x_t==true_params['x0']
    assert y_t==true_params['y0']
    x_t, y_t = mod.model(
        [0.0,2025.0,10000],
        fit_params=np.array([true_params['x0'], true_params['y0']]).T
    )
    assert (x_t==true_params['x0']).all()
    assert (y_t==true_params['y0']).all()
    
    # Check behavior of model
    x0_batch = np.random.uniform(-2.0,2.0, 50)
    y0_batch = np.random.uniform(-2.0,2.0, 50)
    x0_err_batch = np.repeat(0.1, 50)
    y0_err_batch = np.repeat(0.1, 50)
    # Single epoch
    t_batch=2020.0
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
    x_true, y_true = mod.model(
        t, 
        fit_params=np.array([true_params['x0'], true_params['y0']])
    )
    x_sim = rng.normal(x_true, true_params['x0_err'])
    y_sim = rng.normal(y_true, true_params['y0_err'])
    xe = np.ones_like(t)*true_params['x0_err']
    ye = np.ones_like(t)*true_params['y0_err']
    # Run fit
    params, param_errs = mod.fit(
        t, 
        x_sim,y_sim,
        xe=xe,
        ye=ye
    )
    
    x_wt = 1. / xe**2
    y_wt = 1. / ye**2
    x_wt_norm = x_wt / np.sum(x_wt)
    y_wt_norm = y_wt / np.sum(y_wt)
    x_mean = np.average(x_sim, weights=x_wt)
    y_mean = np.average(y_sim, weights=y_wt)
    x_std = (np.sum(x_wt_norm**2 * xe**2))**0.5
    y_std = (np.sum(y_wt_norm**2 * ye**2))**0.5
    
    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])
    np.testing.assert_allclose(params[0], x_mean, atol=1e-5)
    np.testing.assert_allclose(params[1], y_mean, atol=1e-5)
    np.testing.assert_allclose(param_errs[0], x_std, atol=1e-5)
    np.testing.assert_allclose(param_errs[1], y_std, atol=1e-5)


def test_Linear():
    rng = np.random.default_rng(1)
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1,
                    'vx':0.2, 'vy':0.5, 'vx_err':0.05, 'vy_err':0.05,
                    't0':2025.0}
    mod = motion_model.Linear()
    param_list = mod.fit_param_names
    # Confirm return of proper values for single t=t0 and array t
    x_t, y_t = mod.model(
        t=true_params['t0'],
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params_dict={'t0': true_params['t0']}
    )
    assert x_t==true_params['x0']
    assert y_t==true_params['y0']
    t_arr = np.array([2010.0,true_params['t0'],2030.0])
    x_t, y_t = mod.model(
        t=t_arr,
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params_dict={'t0': true_params['t0']}
    )
    assert (x_t==(true_params['x0'] + (t_arr-true_params['t0'])*true_params['vx'])).all()
    assert (y_t==(true_params['y0'] + (t_arr-true_params['t0'])*true_params['vy'])).all()

    # Check behavior of model
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
        fixed_params_dict={'t0': t0_batch}
    )

    np.testing.assert_allclose(x_t_batch, (x0_batch+(t_batch-t0_batch)*vx_batch), atol=1e-5)
    np.testing.assert_allclose(y_t_batch, (y0_batch+(t_batch-t0_batch)*vy_batch), atol=1e-5)
    np.testing.assert_allclose(x_err_t_batch, np.hypot(x0_err_batch, (t_batch-t0_batch)*vx_err_batch), atol=1e-5)
    np.testing.assert_allclose(y_err_t_batch, np.hypot(y0_err_batch, (t_batch-t0_batch)*vy_err_batch), atol=1e-5)

    # Multiple times
    t_batch = np.arange(2015.0,2025.0, 0.5)
    x_t_batch, y_t_batch, x_err_t_batch, y_err_t_batch = mod.model(
        t=t_batch,
        fit_params=np.array([x0_batch, vx_batch, y0_batch, vy_batch]).T,
        fit_param_errs=np.array([x0_err_batch, vx_err_batch, y0_err_batch, vy_err_batch]).T,
        fixed_params_dict={'t0': t0_batch}
    )
    np.testing.assert_allclose(x_t_batch, np.array([x0_batch[i] + (t_batch-t0_batch[i])*vx_batch[i] for i in range(len(x0_batch))]), atol=1e-5)
    np.testing.assert_allclose(y_t_batch, np.array([y0_batch[i] + (t_batch-t0_batch[i])*vy_batch[i] for i in range(len(x0_batch))]), atol=1e-5)
    np.testing.assert_allclose(x_err_t_batch, np.array([np.hypot(x0_err_batch[i], (t_batch-t0_batch[i])*vx_err_batch[i]) for i in range(len(x0_batch))]), atol=1e-5)
    np.testing.assert_allclose(y_err_t_batch, np.array([np.hypot(y0_err_batch[i], (t_batch-t0_batch[i])*vy_err_batch[i]) for i in range(len(x0_batch))]), atol=1e-5)

    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod.model(
        t=t,
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params_dict={'t0': true_params['t0']}
    )
    x_sim = rng.normal(x_true, 0.05)
    y_sim = rng.normal(y_true, 0.05)
    # Run fit
    xe = np.ones_like(t)*0.05
    ye = np.ones_like(t)*0.05
     
    def linear(t, x0, vx):
        return x0 + vx * t

    for absolute_sigma in [True, False]:
        for weighting in ['std', 'var']:
            params, param_errs = mod.fit(
                t=t,
                x=x_sim,
                y=y_sim,
                xe=xe,
                ye=ye,
                fixed_params_dict={'t0': true_params['t0']},
                weighting=weighting,
                absolute_sigma=absolute_sigma
            )

            # Scipy (independent ground truth to check our closed-form fit against)
            xe_scipy = xe**0.5 if weighting=='std' else xe
            ye_scipy = ye**0.5 if weighting=='std' else ye
            x_popt, x_pcov = curve_fit(
                linear,
                t - true_params['t0'],
                x_sim,
                sigma=xe_scipy,
                absolute_sigma=absolute_sigma,
                p0=[np.mean(x_sim), 0.0]
            )
            y_popt, y_pcov = curve_fit(
                linear,
                t - true_params['t0'],
                y_sim,
                sigma=ye_scipy,
                absolute_sigma=absolute_sigma,
                p0=[np.mean(y_sim), 0.0]
            )
            np.testing.assert_allclose(params[:2], x_popt, atol=1e-5)
            np.testing.assert_allclose(param_errs[:2], np.sqrt(np.diag(x_pcov)), atol=1e-5)
            np.testing.assert_allclose(params[2:], y_popt, atol=1e-5)
            np.testing.assert_allclose(param_errs[2:], np.sqrt(np.diag(y_pcov)), atol=1e-5)

    # Test fitter with bootstrap
    t = np.arange(2015.0, 2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod.model(
        t=t,
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params_dict={'t0': true_params['t0']}
    )
    x_true_err, y_true_err = np.ones_like(t)*0.05, np.ones_like(t)*0.05
    x_sim = rng.normal(x_true, x_true_err)
    y_sim = rng.normal(y_true, y_true_err)
    # Run fit
    params, param_errs = mod.fit(t, x_sim, y_sim, x_true_err, y_true_err, fixed_params_dict={'t0': true_params['t0']}, bootstrap=10, seed=42)
    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])

    
def test_Acceleration():
    rng = np.random.default_rng(2)
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':0.5, 'x0_err':0.1, 'y0_err':0.1,
                    'vx0':0.2, 'vy0':0.5, 'vx0_err':0.05, 'vy0_err':0.05,
                    'ax':0.1, 'ay':-0.1, 'ax_err':0.02, 'ay_err':0.02,
                    't0':2025.0}
    mod = motion_model.Acceleration()
    param_list = mod.fit_param_names
    # Confirm return of proper values for single t=t0 and array t
    x_t, y_t = mod.model(
        t=true_params['t0'],
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params_dict={'t0': true_params['t0']}
    )
    np.testing.assert_allclose(x_t, true_params['x0'])
    np.testing.assert_allclose(y_t, true_params['y0'])
    t_arr = np.array([2010.0, true_params['t0'], 2030.0])
    x_t, y_t = mod.model(
        t=t_arr,
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params_dict={'t0': true_params['t0']}
    )
    np.testing.assert_allclose(x_t, true_params['x0'] + (t_arr-true_params['t0'])*true_params['vx0'] + 0.5*(t_arr-true_params['t0'])**2*true_params['ax'])
    np.testing.assert_allclose(y_t, true_params['y0'] + (t_arr-true_params['t0'])*true_params['vy0'] + 0.5*(t_arr-true_params['t0'])**2*true_params['ay'])
    
    # Check behavior of model
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
        fixed_params_dict={'t0': t0_batch}
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
        fixed_params_dict={'t0': t0_batch}
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
        fixed_params_dict={'t0': true_params['t0']}
    )
    x_true_err = np.sqrt(true_params['x0_err']**2 + ((t - true_params['t0']) * true_params['vx0_err'])**2 +
                            (0.5*(t - true_params['t0'])**2 * true_params['ax_err'])**2)
    y_true_err = np.sqrt(true_params['y0_err']**2 + ((t - true_params['t0']) * true_params['vy0_err'])**2 +
                            (0.5*(t - true_params['t0'])**2 * true_params['ay_err'])**2)
    x_sim = rng.normal(x_true, x_true_err)
    y_sim = rng.normal(y_true, y_true_err)
    # Run fit
    mod_fit = motion_model.Acceleration()
    params, param_errs = mod_fit.fit(
        t=t, 
        x=x_sim,
        y=y_sim,
        xe=x_true_err, 
        ye=y_true_err, 
        fixed_params_dict={'t0': true_params['t0']}
    )
    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])

#@pytest.mark.skip(reason="not written")
def test_Parallax():
    rng = np.random.default_rng(4)
    # Test handling of a single star
    true_params = {'x0': 1.0, 'y0':-0.5, 'x0_err':0.1, 'y0_err':0.1,
                    'vx':-0.2, 'vy':0.5, 'vx_err':0.05, 'vy_err':0.05,
                    'pi':0.5, 'ra':17.76, 'dec':-28.933, 'pa':0,
                    't0':2020.0, 'obsLocation': 'earth'}
    mod = motion_model.Parallax()
    param_list = mod.fit_param_names
    fixed_params_dict = {
        't0': true_params['t0'],
        'ra': true_params['ra'],
        'dec': true_params['dec'],
        'pa': true_params['pa'],
        'obsLocation': true_params['obsLocation']
    }
    
    # Test fitter
    t = np.arange(2015.0,2025.0, 0.5)
    # Get values from model and add scatter
    x_true, y_true = mod.model(
        t=t,
        fit_params=np.array([true_params[p] for p in param_list]).T,
        fixed_params_dict=fixed_params_dict
    )
    x_true_err, y_true_err = np.ones_like(t)*true_params['x0_err'], np.ones_like(t)*true_params['y0_err']
    x_sim = rng.normal(x_true, x_true_err)
    y_sim = rng.normal(y_true, y_true_err)
    # Run fit
    params, param_errs = mod.fit(t, x_sim,y_sim, x_true_err, y_true_err, fixed_params_dict=fixed_params_dict)

    x_model, y_model = mod.model(
        t=t,
        fit_params=params,
        fixed_params_dict=fixed_params_dict
    )
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(t, x_model, color='C3', lw=2, label='Model x')
    ax1.plot(t, x_true, color='C0', ls='--', label='True x')
    ax1.errorbar(t, x_sim, yerr=x_true_err, fmt='o', color='C0', label='Sim x')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')
    ax1.legend()
    ax2.plot(t, y_model, color='C3', lw=2, label='Model y')
    ax2.plot(t, y_true, color='C0', ls='--', label='True y')
    ax2.errorbar(t, y_sim, yerr=y_true_err, fmt='o', color='C0', label='Sim y')
    ax2.set_xlabel('t')
    ax2.set_ylabel('y')
    ax2.legend()
    plt.tight_layout()

    # Confirm true value is within error bar of fit value
    assert np.all([within_error(true_params[param_list[i]], params[i], param_errs[i]) for i in range(len(params))])

def test_Parallax_PA():
    # Set PA=0 model
    x0, y0 = 2.0, -1.0
    vx, vy = 0.2, 0.5
    ra, dec = 17.76, -28.933
    pi = 0.5
    mod_pa0 = motion_model.Parallax()
    # Set PA=90 model with equivalent parameters in that frame
    mod_pa90 = motion_model.Parallax()
    
    t_set = np.arange(2018, 2024, 0.01)
    t0 = 2020.0
    dat_pa0 = mod_pa0.model(
        t = t_set,
        fit_params = np.array([x0, vx, y0, vy, pi]).T,
        fixed_params_dict = {'t0': t0, 'ra': ra, 'dec': dec, 'pa': 0}
    )
    dat_pa90 = mod_pa90.model(
        t = t_set,
        fit_params = np.array([y0, vy, -x0, -vx, pi]).T,
        fixed_params_dict = {'t0': t0, 'ra': ra, 'dec': dec, 'pa': 90}
    )
    np.testing.assert_allclose(dat_pa0[0], -dat_pa90[1], atol=1e-10)
    np.testing.assert_allclose(dat_pa0[1], dat_pa90[0], atol=1e-10)


def test_motion_model_param_names_dedup():
    """
    motion_model_param_names() used to re-expand fit_param_names/fixed_param_names
    once per input entry even when the same motion model repeated thousands of
    times (e.g. align.py passing one entry per star). It now dedups the input
    first. Check a heavily-duplicated input still gives the same result as the
    plain unique input.
    """
    repeated_names = ['Fixed', 'Linear'] * 5000
    got = motion_model.motion_model_param_names(repeated_names, with_errors=True, with_fixed=True)
    want = motion_model.motion_model_param_names(['Fixed', 'Linear'], with_errors=True, with_fixed=True)
    assert got == want

    # Order of first appearance should still control the output order.
    reordered = ['Linear', 'Fixed'] * 3000
    got_reordered = motion_model.motion_model_param_names(reordered, with_errors=True, with_fixed=True)
    want_reordered = motion_model.motion_model_param_names(['Linear', 'Fixed'], with_errors=True, with_fixed=True)
    assert got_reordered == want_reordered
    assert got_reordered != got  # different first-seen order -> different param order

    # Mixing model classes with their string names should still be correct.
    mixed = [motion_model.Fixed, 'Fixed', motion_model.Linear, 'Linear'] * 100
    got_mixed = motion_model.motion_model_param_names(mixed, with_errors=True, with_fixed=True)
    assert got_mixed == want

    # with_errors=False / with_fixed=False should still behave as before.
    got_no_extras = motion_model.motion_model_param_names(repeated_names, with_errors=False, with_fixed=False)
    want_no_extras = motion_model.motion_model_param_names(['Fixed', 'Linear'], with_errors=False, with_fixed=False)
    assert got_no_extras == want_no_extras


def test_Fixed_run_fit():
    """
    Fixed.run_fit() fits many stars at once in a single vectorized call
    (closed-form weighted average, no iterative optimizer needed). Check it
    against a per-star loop calling fit() directly (which wraps a single
    star into a batch of one row and calls run_fit() itself), across a
    battery of randomized cases: full epochs, ragged (different numbers of
    valid epochs per star), a star with exactly one valid epoch
    (degree_of_freedom == 0), a star with zero valid epochs (not enough
    data), var/std weighting, and absolute_sigma True/False.
    """
    rng = np.random.default_rng(3)
    n_stars = 40
    n_epochs = 6

    t = np.tile(np.arange(n_epochs) + 2020.0, (n_stars, 1))
    x = rng.normal(100, 5, size=(n_stars, n_epochs))
    y = rng.normal(-50, 5, size=(n_stars, n_epochs))
    xe = rng.uniform(0.01, 0.5, size=(n_stars, n_epochs))
    ye = rng.uniform(0.01, 0.5, size=(n_stars, n_epochs))

    valid = rng.random((n_stars, n_epochs)) > 0.3
    valid[0, :] = False          # zero valid epochs -- not enough data
    valid[1, :] = False
    valid[1, 2] = True           # exactly one valid epoch -- degree_of_freedom == 0
    valid[2, :] = True           # fully detected, for a clean baseline case

    for weighting, absolute_sigma in [('var', True), ('std', True), ('var', False)]:
        mod = motion_model.Fixed()
        got_params, got_errs, got_chi2x, got_chi2y = mod.run_fit(
            t, x, y, xe, ye, valid, weighting=weighting, absolute_sigma=absolute_sigma,
            fill_value=np.nan, verbose=False
        )

        want_params = np.full((n_stars, 2), np.nan)
        want_errs = np.full((n_stars, 2), np.inf)
        want_chi2x = np.full(n_stars, np.nan)
        want_chi2y = np.full(n_stars, np.nan)
        for i in range(n_stars):
            idx = np.flatnonzero(valid[i])
            params, errs, chi2x, chi2y = mod.fit(
                t=t[i][idx], x=x[i][idx], y=y[i][idx], xe=xe[i][idx], ye=ye[i][idx],
                weighting=weighting, absolute_sigma=absolute_sigma,
                fill_value=np.nan, return_chi2=True, bootstrap=0, verbose=False
            )
            want_params[i] = params
            want_errs[i] = errs
            want_chi2x[i] = chi2x
            want_chi2y[i] = chi2y

        np.testing.assert_allclose(got_params, want_params, rtol=1e-10, atol=1e-10, equal_nan=True,
                                    err_msg=f"weighting={weighting} absolute_sigma={absolute_sigma}: params mismatch")
        np.testing.assert_allclose(got_errs, want_errs, rtol=1e-10, atol=1e-10, equal_nan=True,
                                    err_msg=f"weighting={weighting} absolute_sigma={absolute_sigma}: errs mismatch")
        np.testing.assert_allclose(got_chi2x, want_chi2x, rtol=1e-10, atol=1e-10, equal_nan=True,
                                    err_msg=f"weighting={weighting} absolute_sigma={absolute_sigma}: chi2x mismatch")
        np.testing.assert_allclose(got_chi2y, want_chi2y, rtol=1e-10, atol=1e-10, equal_nan=True,
                                    err_msg=f"weighting={weighting} absolute_sigma={absolute_sigma}: chi2y mismatch")