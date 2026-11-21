"""Independent tall-design checks for the joint, separately parameterized fit."""
from copy import deepcopy

import numpy as np
import pytest

import analysis_search as s
from scripts import joint_force_fit as j


NAMES = ("intercept", "flow", "rate")


def cases():
    rng = np.random.default_rng(4415)
    result = []
    for i in range(8):
        n = 11 + i
        xr = np.r_[1., rng.normal(size=2)]
        dx = np.c_[np.zeros(n), rng.normal(size=(n, 2))]
        # Independent A and B; offsets of unanchored runs are deliberately huge.
        dy = dx @ np.array([0., -.4, .9]) + rng.normal(size=n)*.03
        y = xr @ np.array([.2, .7, -.2]) + dy + (0. if i % 3 else 1000.)
        weights = np.linspace(.5, 1.5, n)/n
        result.append(dict(run=str(i), condition=str(i//2), anchored=bool(i % 3),
            xr=xr, dx=dx, y=y, train_dx=dx.copy(), train_dy=dy.copy(), weights=weights))
    return result


def test_joint_statistics_match_explicit_design_all_fields():
    cc, norm = cases(), (.3, .02)
    explicit = j.joint_matrix(cc, NAMES, NAMES, NAMES, norm)
    assert explicit.names == j.label_names(NAMES)
    expected = s.statistics(explicit.x, explicit.y)
    actual = j.joint_statistics(cc, NAMES, norm)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-13, atol=2e-13)
    selected = ("A::rate", "intercept", "B::flow")
    subset = j.subset_stats(actual, explicit.names, selected)
    indices = [explicit.names.index(k) for k in selected]
    np.testing.assert_allclose(subset["g"], explicit.x[:, indices].T @ explicit.x[:, indices])


def test_component_losses_are_exact_condition_mean_mses():
    cc = cases()
    sa, sd = j.component_statistics(cc, NAMES)
    coef = np.array([.2, -.7, .3, -.1, .8])
    errors_a, errors_d = {}, {}
    for c in cc:
        delta = c["train_dx"][:, 1:] @ coef[3:] - c["train_dy"]
        errors_d.setdefault(c["condition"], []).append(float(c["weights"] @ delta**2))
        if c["anchored"]:
            absolute = c["xr"] @ coef[:3] + c["dx"][:, 1:] @ coef[3:] - c["y"]
            errors_a.setdefault(c["condition"], []).append(float(np.mean(absolute**2)))
    for stat, errors in ((sa, errors_a), (sd, errors_d)):
        expected = np.mean([np.mean(v) for v in errors.values()])
        actual = coef @ stat["g"] @ coef - 2*coef @ stat["h"] + stat["yy"]
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("ridge_a,ridge_d", [(0., 0.), (.01, 1.), (1., .0001)])
def test_fit_matches_independent_augmented_svd(ridge_a, ridge_d):
    cc, norm = cases(), (.7, .06)
    matrix = j.joint_matrix(cc, NAMES, NAMES, NAMES, norm)
    stat = j.joint_statistics(cc, NAMES, norm)
    ids = [c["run"] for c in cc]
    anchor_ids = [c["run"] for c in cc if c["anchored"]]
    ma, md = j.fit_joint(stat, matrix.names, matrix.names, ridge_a, ridge_d, ids, anchor_ids)
    scale = np.linalg.norm(matrix.x, axis=0)
    penalties = [0., ridge_a, ridge_a, ridge_d, ridge_d]
    design = np.vstack([matrix.x/scale, np.diag(np.sqrt(penalties))])
    target = np.r_[matrix.y, np.zeros(len(scale))]
    expected = np.linalg.lstsq(design, target, rcond=None)[0]/scale
    actual = np.r_[ma["coefficients"], md["coefficients"]]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert ma["training_runs"] == md["training_runs"] == ids
    assert ma["family_supervision_runs"] == anchor_ids
    assert md["family_supervision_runs"] == ids
    assert ma["coefficients"][1] != md["coefficients"][0]


def test_pool_subsets_and_empty_family_preserve_design():
    cc, norm = cases(), (1., 2.)
    matrix = j.joint_matrix(cc, NAMES, ("intercept",), ("rate",), norm)
    assert matrix.names == ("intercept", "B::rate")
    full = j.joint_statistics(cc, NAMES, norm)
    subset = j.subset_stats(full, j.label_names(NAMES), matrix.names)
    for key, value in s.statistics(matrix.x, matrix.y).items():
        np.testing.assert_allclose(subset[key], value, rtol=2e-13, atol=2e-13)
    ma, md = j.fit_joint(full, j.label_names(NAMES), ("intercept",), 0., 0., ["1"], ["1"])
    assert ma["feature_names"] == ["intercept"]
    assert md["feature_names"] == [] and md["coefficients"] == []
    empty = j.joint_matrix(cc, NAMES, (), (), norm)
    assert empty.x.shape == (len(empty.y), 0)


def test_only_supplied_cases_and_anchored_absolute_targets_are_used():
    cc = cases()
    before = j.joint_statistics(cc[:6], NAMES, (1., 1.))
    for c in cc[6:]:
        c["train_dy"] += 1e8
        c["y"] += 1e9
    changed = deepcopy(cc[:6])
    for c in changed:
        if not c["anchored"]:
            c["y"] += 1e10
    after = j.joint_statistics(changed, NAMES, (1., 1.))
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_invalid_input_fails_explicitly():
    cc = cases()
    with pytest.raises(ValueError, match="anchored"):
        j.joint_statistics([c for c in cc if not c["anchored"]], NAMES, (1., 1.))
    with pytest.raises(ValueError, match="positive"):
        j.joint_statistics(cc, NAMES, (0., 1.))
    with pytest.raises(ValueError, match="unique"):
        j.label_names(("intercept", "flow", "flow"))
    stat = j.joint_statistics(cc, NAMES, (1., 1.))
    with pytest.raises(ValueError, match="nonnegative"):
        j.fit_joint(stat, j.label_names(NAMES), ("intercept",), -.1, 0., ["1"], ["1"])
