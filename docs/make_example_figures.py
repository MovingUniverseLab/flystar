"""
Regenerate the figures used by docs/index.rst and docs/examples/alignment_example.rst.

Run from anywhere with flystar importable:

    python docs/make_example_figures.py

The figures are committed so that the documentation build needs no science
stack (sphinx-autoapi reads the source statically and never imports flystar).
This script is the record of how they were produced -- keep it in step with the
code blocks on that page, which are meant to be identical.
"""
import os
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flystar import align, starlists, transforms

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_static')
rng = np.random.default_rng(42)
N, YEARS, ERR = 250, np.array([2015.0, 2017.0, 2019.0, 2021.0]), 0.05
t0 = YEARS.mean()

x0 = rng.uniform(0, 1000, N); y0 = rng.uniform(0, 1000, N)
vx = rng.normal(0, 0.3, N);   vy = rng.normal(0, 0.3, N)
mag = rng.uniform(12, 19, N)
names = np.array([f'S{j:03d}' for j in range(N)])

shift_x = np.array([0., 18., -25., 40.]); shift_y = np.array([0., -12., 30., -20.])
angle = np.deg2rad(np.array([0., 0.3, -0.5, 0.7]))

lists, raw = [], []
for i, yr in enumerate(YEARS):
    dt = yr - t0
    xt, yt = x0 + vx*dt, y0 + vy*dt
    xc, yc = xt - 500., yt - 500.
    c, s = np.cos(angle[i]), np.sin(angle[i])
    xo = (c*xc - s*yc) + 500. + shift_x[i] + rng.normal(0, ERR, N)
    yo = (s*xc + c*yc) + 500. + shift_y[i] + rng.normal(0, ERR, N)
    seen = rng.random(N) > 0.12
    raw.append((xo, yo, seen))
    sl = starlists.StarList(name=names[seen], x=xo[seen], y=yo[seen], m=mag[seen],
                            xe=np.full(seen.sum(), ERR), ye=np.full(seen.sum(), ERR),
                            me=np.full(seen.sum(), 0.05))
    sl.meta['list_time'] = yr
    lists.append(sl)

msc = align.MosaicSelfRef(lists, dr_tol=[60., 10., 5.], dm_tol=[1., 1., 1.],
                          trans_class=transforms.PolyTransform,
                          trans_args=[{'order': 1}]*3,
                          motion_models=['Linear'], init_guess_mode='miracle', verbose=False)
msc.fit()
ref = msc.ref_table
print(f"ref_table: {len(ref)} rows; {int((ref['n_detect']==4).sum())} stars in all 4 epochs")

COL = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

# ---- Figure 1: before vs after, zoomed so the frame offsets are visible -----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2))
lo, hi = 100, 260
# Join the four measurements of one star, so the spread reads as one star seen
# in four misaligned frames rather than as four different stars.
in_box = np.all([(raw[i][0] > lo) & (raw[i][0] < hi) &
                 (raw[i][1] > lo) & (raw[i][1] < hi) & raw[i][2]
                 for i in range(len(YEARS))], axis=0)
for j in np.flatnonzero(in_box):
    a1.plot([raw[i][0][j] for i in range(len(YEARS))],
            [raw[i][1][j] for i in range(len(YEARS))],
            '-', color='0.75', lw=.8, zorder=1)
for i, yr in enumerate(YEARS):
    xo, yo, seen = raw[i]
    k = seen & (xo > lo) & (xo < hi) & (yo > lo) & (yo < hi)
    a1.scatter(xo[k], yo[k], s=26, color=COL[i], alpha=.85, label=f'{yr:.0f}', zorder=2)
a1.set_title('Before: one star, four frames (grey joins the same star)')
a1.legend(title='epoch', fontsize=8); a1.set_xlabel('x (pixels)'); a1.set_ylabel('y (pixels)')

xt_all, yt_all = np.asarray(ref['x']), np.asarray(ref['y'])
sel = (np.asarray(ref['x0']) > lo) & (np.asarray(ref['x0']) < hi) & \
      (np.asarray(ref['y0']) > lo) & (np.asarray(ref['y0']) < hi)
for i, yr in enumerate(YEARS):
    a2.scatter(xt_all[sel, i], yt_all[sel, i], s=26, color=COL[i], alpha=.85, label=f'{yr:.0f}')
a2.set_title('After: all epochs in the common frame')
a2.legend(title='epoch', fontsize=8); a2.set_xlabel('x (pixels)'); a2.set_ylabel('y (pixels)')
for a in (a1, a2):
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect('equal')
fig.tight_layout(); fig.savefig(f'{OUT}/align_before_after.png', dpi=130); plt.close(fig)

# ---- Figure 2: residuals per epoch about the fitted model -------------------
fig, ax = plt.subplots(figsize=(6.4, 4.4))
good = np.asarray(ref['n_detect']) == len(YEARS)
xm, ym, _, _ = ref.infer_positions(YEARS)
dx = (xt_all - xm)[good].ravel() * 1000
dy = (yt_all - ym)[good].ravel() * 1000
ax.hist(dx[np.isfinite(dx)], bins=60, alpha=.7, label='x', color=COL[0])
ax.hist(dy[np.isfinite(dy)], bins=60, alpha=.7, label='y', color=COL[1])
ax.axvline(0, color='k', lw=.8, ls='--')
ax.set_xlabel('residual from fitted motion (milli-pixels)'); ax.set_ylabel('count')
ax.set_title(f'Post-alignment residuals  (injected noise {ERR*1000:.0f} mpix)')
ax.legend(); fig.tight_layout(); fig.savefig(f'{OUT}/align_residuals.png', dpi=130); plt.close(fig)
print(f"residual scatter: x {np.nanstd(dx):.1f} mpix, y {np.nanstd(dy):.1f} mpix (injected {ERR*1000:.0f})")

# ---- Figure 3: recovered proper motion vs truth -----------------------------
name_to_i = {n: j for j, n in enumerate(names)}
rn = np.asarray(ref['name']).astype(str)
idx = np.array([name_to_i.get(n, -1) for n in rn])
ok = good & (idx >= 0)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.6))
for a, rec, tru, lab in ((b1, np.asarray(ref['vx'])[ok], vx[idx[ok]], 'v_x'),
                         (b2, np.asarray(ref['vy'])[ok], vy[idx[ok]], 'v_y')):
    a.scatter(tru, rec, s=16, alpha=.7, color=COL[0])
    lim = [min(tru.min(), rec.min()) - .1, max(tru.max(), rec.max()) + .1]
    a.plot(lim, lim, 'k--', lw=.9)
    a.set_xlim(lim); a.set_ylim(lim); a.set_aspect('equal')
    a.set_xlabel(f'true {lab} (pix/yr)'); a.set_ylabel(f'recovered {lab} (pix/yr)')
    a.set_title(f'{lab}:  scatter = {np.std(rec - tru)*1000:.1f} mpix/yr')
fig.tight_layout(); fig.savefig(f'{OUT}/align_proper_motion.png', dpi=130); plt.close(fig)
print(f"proper motion recovery: vx rms {np.std(np.asarray(ref['vx'])[ok]-vx[idx[ok]])*1000:.1f} mpix/yr, "
      f"n={ok.sum()}")
