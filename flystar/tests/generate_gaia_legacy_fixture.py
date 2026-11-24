"""Write the legacy Gaia-ingestion snapshot used by test_gaia_projection.py.

TEMPORARY: removed together with the 'flat_legacy' projection mode, before
release.

The checked-in ``test_data/gaia_legacy_outputs.npz`` was written by this
script running against FlyStar's ``analysis.prepare_gaia_for_flystar`` and
``analysis.project_gaia`` as they were BEFORE the projection keyword was
added (commit 9c2aa09 on branch tangent-plane-impl), on the synthetic table
from ``gaia_synthetic.make_synthetic_gaia``. The tests check that
``projection='flat_legacy'`` still reproduces it bit for bit.

Rerunning it against the new code with ``LEGACY_KWARGS`` set to
``{'projection': 'flat_legacy'}`` must reproduce the same file.

Run from the repository root:

    python flystar/tests/generate_gaia_legacy_fixture.py
"""
import pathlib

import numpy as np

from flystar import analysis
from flystar.tests.gaia_synthetic import make_synthetic_gaia, CENTER_RA_STR, CENTER_DEC_STR

OUT = pathlib.Path(__file__).parent / 'test_data' / 'gaia_legacy_outputs.npz'

# Keyword arguments added to every call. Empty for the old code.
LEGACY_KWARGS = {}

# The cases stored in the snapshot.
PREPARE_CASES = {
    'prep_linear': dict(),
    'prep_parallax': dict(default_motion_model='Parallax', pi_err_limit=0.5),
    'prep_targets': dict(targets_dict={'targA': [10.0, -20.0]}, match_dr_max=30.0),
}
PROJECT_EPOCHS = {'proj_2010': 2010.3, 'proj_2024': 2024.7}


def legacy_outputs(extra_kwargs=None):
    """
    Run the legacy cases and return {key: array} with one entry per column.

    Parameters
    ----------
    extra_kwargs : dict, optional
        Keyword arguments added to every call (e.g. projection='flat_legacy').

    Returns
    -------
    out : dict
        Keys are '<case>/<column>'; values are numpy arrays.
    """
    extra_kwargs = {} if extra_kwargs is None else extra_kwargs
    out = {}

    for case, kw in PREPARE_CASES.items():
        tab = analysis.prepare_gaia_for_flystar(make_synthetic_gaia(), CENTER_RA_STR,
                                                CENTER_DEC_STR, **kw, **extra_kwargs)
        for col in tab.colnames:
            out[f'{case}/{col}'] = np.asarray(tab[col])

    # project_gaia takes the center in degrees.
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    c = SkyCoord(CENTER_RA_STR, CENTER_DEC_STR, unit=(u.hourangle, u.deg))
    for case, epoch in PROJECT_EPOCHS.items():
        lis = analysis.project_gaia(make_synthetic_gaia(), epoch, c.ra.deg, c.dec.deg,
                                    **extra_kwargs)
        for col in lis.colnames:
            out[f'{case}/{col}'] = np.asarray(lis[col])

    return out


if __name__ == '__main__':
    out = legacy_outputs(LEGACY_KWARGS)
    np.savez(OUT, **out)
    print(f'Wrote {len(out)} arrays to {OUT}')
