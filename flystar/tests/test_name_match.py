"""Unit tests for name matching (override and tiebreak modes)."""
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

from flystar import align, match

KW = dict(verbose=0, sigma_pos=0.1, sigma_mag=0.2)


def pairs(i1, i2):
    return dict(zip(np.asarray(i1).tolist(), np.asarray(i2).tolist()))


def ambiguous_setup():
    # List star 0 at x=0.32: nearest in position is ref 1 (dr 0.28), nearest
    # in magnitude is ref 0 (dm 0.0 vs 0.1). Legacy: ambiguous. chi2 with
    # these scales: chi2 = 10.24 vs 7.84 + 0.25 -> margin < 9, ambiguous.
    x2 = np.array([0.0, 0.6, 10.0])
    y2 = np.zeros(3)
    m2 = np.array([15.0, 15.1, 12.0])
    x1 = np.array([0.32, 10.01])
    y1 = np.zeros(2)
    m1 = np.array([15.0, 12.0])
    return x1, y1, m1, x2, y2, m2


@pytest.mark.parametrize("mode", ["legacy", "chi2"])
def test_tiebreak_ambiguous_star_is_tie_broken_by_name(mode):
    x1, y1, m1, x2, y2, m2 = ambiguous_setup()
    n2 = np.array(["S0-35", "S0-36", "irs16C"])
    n1 = np.array(["S0-35", "irs16C"])
    i1, i2, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, matching=mode, **KW)
    assert 0 not in i1
    i1, i2, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, matching=mode,
                               names1=n1, names2=n2, name_dr_tol=1.0, name_mode="tiebreak", **KW)
    assert pairs(i1, i2) == {0: 0, 1: 2}


@pytest.mark.parametrize("mode", ["legacy", "chi2"])
def test_tiebreak_unambiguous_position_beats_name(mode):
    # Ref 1 is nearest in position AND magnitude; the list star's name says
    # ref 0. java rule: the positional match stands.
    x2 = np.array([0.0, 0.6]); y2 = np.zeros(2); m2 = np.array([16.0, 15.0])
    x1 = np.array([0.58]); y1 = np.zeros(1); m1 = np.array([15.0])
    i1, i2, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, matching=mode,
                               names1=np.array(["S0-35"]),
                               names2=np.array(["S0-35", "S0-36"]),
                               name_dr_tol=1.0, name_mode="tiebreak", **KW)
    assert pairs(i1, i2) == {0: 1}


@pytest.mark.parametrize("mode", ["legacy", "chi2"])
def test_tiebreak_single_candidate_with_other_name_is_kept(mode):
    x2 = np.array([0.0, 5.0]); y2 = np.zeros(2); m2 = np.array([15.0, 15.0])
    x1 = np.array([0.1]); y1 = np.zeros(1); m1 = np.array([15.0])
    i1, i2, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, matching=mode,
                               names1=np.array(["S1-1"]),
                               names2=np.array(["S1-2", "S1-1"]),
                               name_dr_tol=10.0, name_mode="tiebreak", **KW)
    assert pairs(i1, i2) == {0: 0}


@pytest.mark.parametrize("mode", ["legacy", "chi2"])
def test_tiebreak_name_gate(mode):
    # Same ambiguous star, but its namesake (ref 0, dr 0.32) is beyond the gate.
    x1, y1, m1, x2, y2, m2 = ambiguous_setup()
    i1, i2, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, matching=mode,
                               names1=np.array(["S0-35", "irs16C"]),
                               names2=np.array(["S0-35", "S0-36", "irs16C"]),
                               name_dr_tol=0.3, name_mode="tiebreak", **KW)
    assert 0 not in i1


@pytest.mark.parametrize("mode", ["legacy", "chi2"])
def test_tiebreak_startswith_exclusion(mode):
    x1, y1, m1, x2, y2, m2 = ambiguous_setup()
    n2 = np.array(["star_7", "S0-36", "irs16C"])
    # 'star_7' is anonymous: no tie-break.
    i1, _, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, matching=mode,
                              names1=np.array(["star_7", "irs16C"]), names2=n2,
                              name_dr_tol=1.0, name_mode="tiebreak", **KW)
    assert 0 not in i1
    # A name that merely CONTAINS 'star' still counts (startswith rule).
    n2b = np.array(["S0-35star", "S0-36", "irs16C"])
    i1, i2, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, matching=mode,
                               names1=np.array(["S0-35star", "irs16C"]), names2=n2b,
                               name_dr_tol=1.0, name_mode="tiebreak", **KW)
    assert pairs(i1, i2)[0] == 0
    # Custom exclusion list.
    n2c = np.array(["unc_S0-35", "S0-36", "irs16C"])
    i1, _, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, matching=mode,
                              names1=np.array(["unc_S0-35", "irs16C"]), names2=n2c,
                              name_dr_tol=1.0, name_exclude=("star", "unc_"), name_mode="tiebreak", **KW)
    assert 0 not in i1


def test_is_named():
    out = match.is_named(np.array(["star_1", " star_2", "S0-2", "S0-2star", "", "irs16C"]))
    assert out.tolist() == [False, False, True, True, False, True]


def test_tiebreak_legacy_duplicate_claim_resolved_by_name():
    # Two list stars, each with the single candidate ref 0. Star 0 is nearer,
    # star 1 is closer in magnitude -> confused. Star 1 carries ref 0's name.
    x2 = np.array([0.0]); y2 = np.zeros(1); m2 = np.array([15.0])
    x1 = np.array([0.1, -0.2]); y1 = np.zeros(2); m1 = np.array([15.5, 15.05])
    i1, _, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, verbose=0)
    assert len(i1) == 0
    i1, i2, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, verbose=0,
                               names1=np.array(["star_9", "S2-5"]),
                               names2=np.array(["S2-5"]), name_dr_tol=1.0,
                               name_mode="tiebreak")
    assert pairs(i1, i2) == {1: 0}


def test_prefer_original_row_and_availability():
    x2 = np.array([0.0, 0.3]); y2 = np.zeros(2)
    n2 = np.array(["S0-2", "S0-2"]); pref = np.array([True, False])
    j = match.same_name_candidates([0], np.array(["S0-2"]), n2, np.array([0.25]),
                                   np.array([0.0]), x2, y2, 1.0, prefer2=pref)
    assert j.tolist() == [0]
    j = match.same_name_candidates([0], np.array(["S0-2"]), n2, np.array([0.25]),
                                   np.array([0.0]), x2, y2, 1.0, prefer2=pref,
                                   avail2=np.array([False, True]))
    assert j.tolist() == [1]


def test_duplicate_list_names_not_used():
    j = match.same_name_candidates([0, 1], np.array(["S0-2", "S0-2"]),
                                   np.array(["S0-2"]), np.array([0.0, 0.1]),
                                   np.zeros(2), np.array([0.0]), np.zeros(1), 1.0)
    assert j.tolist() == [-1, -1]


def test_names_require_gate():
    with pytest.raises(ValueError):
        match.match(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
                    np.zeros(1), np.zeros(1), 1.0, names1=np.array(["a"]),
                    names2=np.array(["a"]), verbose=0)


def test_name_keys():
    t = Table({"name": [" S0-6", " 21_S0-35", "  3_star_9", "1_ab"],
               "ref_orig": [True, False, False, True]})
    assert align.ref_name_keys(t).tolist() == ["S0-6", "S0-35", "star_9", "1_ab"]
    keys = align.set_name_keys(np.array(["ab", "cd"]), [1], np.array(["S0-1234567"]))
    assert keys.tolist() == ["ab", "S0-1234567"]


def test_fix_name_match_dr_tol():
    fix = align.MosaicSelfRef.fix_name_match_dr_tol
    ns = SimpleNamespace(name_match=True, name_match_dr_tol=0.04, iters=3)
    fix(ns)
    assert ns.name_match_dr_tol == [0.04] * 3
    ns = SimpleNamespace(name_match=True, name_match_dr_tol=[0.04, 0.02], iters=3)
    with pytest.raises(ValueError):
        fix(ns)
    ns = SimpleNamespace(name_match=True, name_match_dr_tol=None, iters=3)
    with pytest.raises(ValueError):
        fix(ns)
    ns = SimpleNamespace(name_match=False, name_match_dr_tol=None, iters=3)
    fix(ns)
    assert ns.name_match_dr_tol is None


# --------------------------------------------------------------------------
# name_mode=override (default).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["legacy", "chi2"])
def test_override_name_beats_nearer_positional_candidate(mode):
    # Ref 1 is nearest in position AND magnitude (unambiguous), but the list
    # star is named after ref 0, 0.58 away, inside the gate: name wins.
    x2 = np.array([0.0, 0.6]); y2 = np.zeros(2); m2 = np.array([16.0, 15.0])
    x1 = np.array([0.58]); y1 = np.zeros(1); m1 = np.array([15.0])
    i1, i2, _, _ = match.match(x1, y1, m1, x2, y2, m2, 1.0, matching=mode,
                               names1=np.array(["S0-35"]),
                               names2=np.array(["S0-35", "S0-36"]),
                               name_dr_tol=1.0, **KW)
    assert pairs(i1, i2) == {0: 0}


@pytest.mark.parametrize("mode", ["legacy", "chi2"])
def test_override_beyond_dr_tol_inside_gate(mode):
    x2 = np.array([0.0, 10.0]); y2 = np.zeros(2); m2 = np.array([15.0, 12.0])
    x1 = np.array([0.03, 10.0]); y1 = np.zeros(2); m1 = np.array([15.0, 12.0])
    n1 = np.array(["S0-2", "irs16C"]); n2 = np.array(["S0-2", "irs16C"])
    i1, _, _, _ = match.match(x1, y1, m1, x2, y2, m2, 0.02, matching=mode, **KW)
    assert 0 not in i1
    i1, i2, dr, _ = match.match(x1, y1, m1, x2, y2, m2, 0.02, matching=mode,
                                names1=n1, names2=n2, name_dr_tol=0.04, **KW)
    assert pairs(i1, i2) == {0: 0, 1: 1}
    assert dr[np.flatnonzero(i1 == 0)[0]] == pytest.approx(0.03)


@pytest.mark.parametrize("mode", ["legacy", "chi2"])
def test_override_ignores_dm(mode):
    x2 = np.array([0.0]); y2 = np.zeros(1); m2 = np.array([15.0])
    x1 = np.array([0.01]); y1 = np.zeros(1); m1 = np.array([17.5])
    i1, _, _, _ = match.match(x1, y1, m1, x2, y2, m2, 0.1, dm_tol=0.5,
                              matching=mode, **KW)
    assert len(i1) == 0
    i1, i2, _, dm = match.match(x1, y1, m1, x2, y2, m2, 0.1, dm_tol=0.5,
                                matching=mode, names1=np.array(["S1-1"]),
                                names2=np.array(["S1-1"]), name_dr_tol=0.1, **KW)
    assert pairs(i1, i2) == {0: 0} and dm[0] == pytest.approx(-2.5)


@pytest.mark.parametrize("mode", ["legacy", "chi2"])
def test_override_rejected_outside_gate_falls_back_to_position(mode):
    # Namesake (ref 0) is 0.5 away, gate 0.3: no name pair. Positional
    # matching then takes the unnamed-but-nearest ref 1.
    x2 = np.array([0.0, 0.52]); y2 = np.zeros(2); m2 = np.array([15.0, 15.0])
    x1 = np.array([0.5]); y1 = np.zeros(1); m1 = np.array([15.0])
    i1, i2, _, _ = match.match(x1, y1, m1, x2, y2, m2, 0.1, matching=mode,
                               names1=np.array(["S0-35"]),
                               names2=np.array(["S0-35", "star_4"]),
                               name_dr_tol=0.3, **KW)
    assert pairs(i1, i2) == {0: 1}


@pytest.mark.parametrize("mode", ["legacy", "chi2"])
def test_override_removes_pairs_from_positional_pool(mode):
    # Ref 0 is taken by name; anonymous list star 1 sits on ref 0 too, so
    # positionally it can only get ref 1 (inside dr_tol) -- not ref 0.
    x2 = np.array([0.0, 0.08]); y2 = np.zeros(2); m2 = np.array([15.0, 15.0])
    x1 = np.array([0.05, 0.01]); y1 = np.zeros(2); m1 = np.array([15.0, 15.0])
    i1, i2, _, _ = match.match(x1, y1, m1, x2, y2, m2, 0.1, matching=mode,
                               names1=np.array(["S0-35", "star_1"]),
                               names2=np.array(["S0-35", "star_9"]),
                               name_dr_tol=0.1, **KW)
    assert pairs(i1, i2) == {0: 0, 1: 1}


def test_override_exclusion_and_duplicates():
    x2 = np.array([0.0, 1.0]); y2 = np.zeros(2); m2 = np.array([15.0, 15.0])
    x1 = np.array([0.5, 0.6]); y1 = np.zeros(2); m1 = np.array([15.0, 15.0])
    # Anonymous names are never paired, even with a far gate.
    i1, _, _, _ = match.match(x1, y1, m1, x2, y2, m2, 0.01,
                              names1=np.array(["star_1", "star_2"]),
                              names2=np.array(["star_1", "star_2"]),
                              name_dr_tol=2.0, verbose=0)
    assert len(i1) == 0
    # Names duplicated in the list are not paired.
    i1, _, _, _ = match.match(x1, y1, m1, x2, y2, m2, 0.01,
                              names1=np.array(["S0-2", "S0-2"]),
                              names2=np.array(["S0-2", "S0-3"]),
                              name_dr_tol=2.0, verbose=0)
    assert len(i1) == 0
    # Duplicate ref rows: the original (preferred) row wins over the nearer.
    i1, i2, _, _ = match.match(x1[:1], y1[:1], m1[:1], x2, y2, m2, 0.01,
                               names1=np.array(["S0-2"]),
                               names2=np.array(["S0-2", "S0-2"]),
                               name_prefer2=np.array([False, True]),
                               name_dr_tol=2.0, verbose=0)
    assert pairs(i1, i2) == {0: 1}


def test_bad_name_mode():
    with pytest.raises(ValueError):
        match.match(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
                    np.zeros(1), np.zeros(1), 1.0, names1=np.array(["a"]),
                    names2=np.array(["a"]), name_dr_tol=1.0, name_mode="x",
                    verbose=0)
