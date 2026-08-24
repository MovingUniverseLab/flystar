"""
Time the motion-model fits on both branches and draw the figure used by the
Performance section of docs/motion_models.rst.

The comparison spans two branches, so it needs two checkouts of flystar. Add
one as a worktree, time each, then plot:

    git worktree add /tmp/wt_mmrework mm_rework

    python docs/benchmark_motion_models.py time /tmp/wt_mmrework \\
        mm_rework /tmp/bench_mmrework.json
    python docs/benchmark_motion_models.py time . \\
        mm_rework_lingfeng /tmp/bench_lingfeng.json

    python docs/benchmark_motion_models.py plot \\
        /tmp/bench_lingfeng.json /tmp/bench_mmrework.json

Timing runs are separate processes so that each imports the flystar it is
timing, and are run one after another rather than concurrently so that they do
not compete for cores. The figure is committed, so building the documentation
never runs any of this.
"""
import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np

N_STARS = 10_000
EPOCHS = list(range(3, 21, 2))
MODELS = ['Fixed', 'Linear', 'Acceleration', 'Parallax']
FIXED_PARAMS = {'ra': 18.0, 'dec': -30.0, 'pa': 0.0, 'obsLocation': 'earth'}


def make_table(n_stars, n_epochs, seed=1):
    """Build a StarTable of linearly moving stars, observed at every epoch."""
    from flystar.startables import StarTable

    rng = np.random.default_rng(seed)
    t = np.tile(np.linspace(2015., 2015. + 2*(n_epochs - 1), n_epochs), (n_stars, 1))
    x0 = rng.uniform(0, 1000, (n_stars, 1))
    y0 = rng.uniform(0, 1000, (n_stars, 1))
    vx = rng.normal(0, 0.3, (n_stars, 1))
    vy = rng.normal(0, 0.3, (n_stars, 1))
    dt = t - t.mean()
    x = x0 + vx*dt + rng.normal(0, 0.05, (n_stars, n_epochs))
    y = y0 + vy*dt + rng.normal(0, 0.05, (n_stars, n_epochs))
    m = np.tile(rng.uniform(12, 19, (n_stars, 1)), (1, n_epochs))
    e = np.full((n_stars, n_epochs), 0.05)

    return StarTable(name=np.array([f'S{i:06d}' for i in range(n_stars)]),
                     x=x, y=y, m=m, xe=e, ye=e, me=e, t=t)


def time_one(branch, model, n_epochs, n_stars=N_STARS):
    """Time a single fit, and report what fraction of stars got `model`."""
    from flystar import motion_model as MM

    tab = make_table(n_stars, n_epochs)

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        if branch == 'mm_rework_lingfeng':
            tab.fit_motion_models(motion_models=[model],
                                  fixed_params_dict=FIXED_PARAMS, verbose=False)
        else:
            # The old API: one model for everything, and Parallax needs an
            # instance carrying the parameters that are fixed_params_dict now.
            mmd = ({'Parallax': MM.Parallax(RA=FIXED_PARAMS['ra'],
                                            Dec=FIXED_PARAMS['dec'],
                                            PA=FIXED_PARAMS['pa'],
                                            obsLocation=FIXED_PARAMS['obsLocation'])}
                   if model == 'Parallax' else {})
            tab.fit_velocities(default_motion_model=model, motion_model_dict=mmd,
                               show_progress=False, verbose=False)
    elapsed = time.perf_counter() - t0

    used = np.asarray(tab['motion_model_used']).astype(str)
    return elapsed, float(np.mean(used == model))


def run_timings(flystar_dir, branch, out_json):
    warnings.filterwarnings('ignore')
    flystar_dir = os.path.abspath(flystar_dir)
    sys.path.insert(0, flystar_dir)
    import flystar
    assert flystar.__path__[0].startswith(flystar_dir), flystar.__path__

    print(f'branch={branch}  flystar={flystar.__path__[0]}', flush=True)
    results = {}
    for model in MODELS:
        # One throwaway fit on a tiny table first. The first fit of a model in
        # a process pays a one-time set-up the rest do not -- ~0.03 s for
        # Parallax, which is 40% of a batched 10,000-star fit and would land
        # entirely on whichever cell happened to be timed first.
        time_one(branch, model, EPOCHS[0], n_stars=50)
        for n_epochs in EPOCHS:
            elapsed, frac = time_one(branch, model, n_epochs)
            results[f'{model}|{n_epochs}'] = {'sec': elapsed, 'frac_used': frac}
            print(f'  {model:13} {n_epochs:2d} epochs: {elapsed:8.3f}s '
                  f'({frac*100:.0f}% got {model})', flush=True)

    json.dump({'branch': branch, 'n_stars': N_STARS, 'results': results},
              open(out_json, 'w'), indent=1)
    print('wrote', out_json, flush=True)


def plot(batched_json, per_star_json, out_png=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if out_png is None:
        out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '_static', 'motion_model_performance.png')
    batched = json.load(open(batched_json))['results']
    per_star = json.load(open(per_star_json))['results']

    def series(res, model, key='sec'):
        return np.array([res[f'{model}|{n}'][key] for n in EPOCHS])

    colors = dict(zip(MODELS, plt.get_cmap('tab10').colors))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for model in MODELS:
        c = colors[model]
        per, bat = series(per_star, model), series(batched, model)
        # A cell where the requested model was not what the stars actually got
        # timed a fallback instead, so it is not a like-for-like comparison:
        # draw it hollow, and break the speed-up line rather than quote a ratio
        # between two different models.
        same = ((series(per_star, model, 'frac_used') == 1)
                & (series(batched, model, 'frac_used') == 1))

        ax1.plot(EPOCHS, per, '--', color=c, lw=1.4, label=f'{model}, per-star')
        ax1.plot(EPOCHS, bat, '-', color=c, lw=1.8, label=f'{model}, batched')
        for y in (per, bat):
            ax1.plot(np.array(EPOCHS)[same], y[same], 'o', color=c, ms=4)
            ax1.plot(np.array(EPOCHS)[~same], y[~same], 'o', ms=5,
                     mfc='white', mec=c, mew=1.2)
        ax2.plot(EPOCHS, np.where(same, per/bat, np.nan), 'o-', color=c,
                 ms=4, lw=1.8, label=model)

    ax1.set(xlabel='Number of epochs', ylabel='Seconds for one fit',
            yscale='log', title=f'Fitting {N_STARS:,} stars')
    ax1.legend(fontsize=7.5, ncol=2, loc='center right')

    ax2.set(xlabel='Number of epochs', ylabel='Speed-up (per-star / batched)',
            title='Batched speed-up')
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xticks(EPOCHS)
        ax.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print('wrote', out_png)


if __name__ == '__main__':
    if sys.argv[1] == 'time':
        run_timings(*sys.argv[2:5])
    else:
        plot(*sys.argv[2:])
