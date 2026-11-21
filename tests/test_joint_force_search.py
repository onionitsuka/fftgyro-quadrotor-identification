"""Validate deployed joint predictions, nested selection and QR/SVD handoff."""
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest

import analysis_search as s
from scripts import joint_force_fit as j
from scripts import joint_force_search as search


NAMES = ["intercept", "flow", "rate", "coupling"]
POOLS = dict(core=NAMES[:3], coupled=NAMES)
STAGES = dict(core=("core",), interactions=("core", "coupled"))


def cases():
    rng = np.random.default_rng(3351)
    result = []
    for condition in range(9):
        for repeat in range(1 + (condition % 3 == 0)):
            n = 40 + 9*repeat
            xr = np.r_[1., rng.normal(size=3)]
            dx = np.c_[np.zeros(n), rng.normal(size=(n, 3))]
            dy = dx @ np.array([0., .8, -.3, .6]) + rng.normal(size=n)*.01
            # A and B intentionally have different coefficients.
            y = xr @ np.array([.2, -.7, .5, -.1]) + dy
            result.append(dict(run=f"run-{condition}-{repeat}", condition=str(condition),
                fold=condition % 3, anchored=True, validation=True, x=dx+xr,
                xr=xr, dx=dx, dy=dy, y=y, train_dx=dx.copy(), train_dy=dy.copy(),
                weights=np.ones(n)/n))
    return result


def model(names, coef):
    return dict(feature_names=list(names), coefficients=list(coef),
                coefficient_by_name=dict(zip(names, coef)))


def test_model_scores_equal_explicit_condition_mean_composite_errors():
    cc = cases()
    cc[0]["anchored"] = False
    cc[2]["validation"] = False
    models = {
        "different_order": (model(["rate", "intercept", "flow"], [.4, .2, -.7]),
                            model(["coupling", "flow"], [.1, .8])),
        "different_terms": (model(["coupling"], [-.9]), model(["rate"], [.3])),
        "empty_delta": (model(NAMES, [.2, -.1, .4, .8]), model([], []))}
    norm = (.7, .4)
    scores = search.score_models(cc, models, NAMES, norm)
    for key, (ma, md) in models.items():
        absolute, delta = defaultdict(list), defaultdict(list)
        for c in cc:
            if not c["validation"]:
                continue
            increment = s.prediction(c, md, NAMES, "dx")
            delta[c["condition"]].append(np.mean((increment-c["dy"])**2))
            if c["anchored"]:
                prediction = s.prediction(c, ma, NAMES, "xr") + increment
                absolute[c["condition"]].append(np.mean((prediction-c["y"])**2))
        aa = np.mean([np.mean(v) for v in absolute.values()])
        dd = np.mean([np.mean(v) for v in delta.values()])
        expected = dict(absolute_mse=aa, increment_mse=dd, joint=.5*(aa/norm[0]+dd/norm[1]))
        assert scores[key] == pytest.approx(expected, rel=2e-13, abs=2e-14)


def test_validation_scores_ignore_augmented_training_rows_and_weights():
    cc = cases()
    models = {"pair": (model(NAMES, [.2, -.7, .5, -.1]),
                        model(NAMES[1:], [.8, -.3, .6]))}
    norm = (.7, .4)
    before = search.score_models(cc, models, NAMES, norm)
    changed = deepcopy(cc)
    for c in changed:
        c["train_dx"] = np.vstack([c["dx"]*100, c["dx"]*-300])
        c["train_dy"] = np.r_[c["dy"]+1e8, c["dy"]-1e9]
        c["weights"] = np.linspace(2., 200., len(c["train_dy"]))
    after = search.score_models(changed, models, NAMES, norm)
    assert before == after
    # The scorer replaces fields in shallow case copies, not the supplied inputs.
    assert len(changed[0]["train_dy"]) == 2*len(changed[0]["dy"])
    assert changed[0]["weights"][-1] == 200.


def test_unused_dictionary_column_does_not_change_scores_or_verified_fit():
    cc = cases()
    names = NAMES + ["unused"]
    expanded = deepcopy(cc)
    for c in expanded:
        c["xr"] = np.r_[c["xr"], .7]
        for key in ("x", "dx", "train_dx"):
            c[key] = np.column_stack([c[key], np.arange(len(c[key]))*.01])
    models = {"pair": (model(NAMES, [.2, -.7, .5, -.1]), model(NAMES[1:], [.8, -.3, .6]))}
    before = search.score_models(cc, models, NAMES, (.7, .4))
    after = search.score_models(expanded, models, names, (.7, .4))
    for key in before:
        assert after[key] == pytest.approx(before[key], rel=1e-13, abs=1e-14)
    norm = s.scales_from_training(expanded, 1e-12)
    selected = list(j.label_names(NAMES))
    ids = [c["run"] for c in expanded]
    pair = j.fit_joint(j.joint_statistics(expanded, names, norm), j.label_names(names),
                       selected, .01, .1, ids, ids)
    original = deepcopy(pair)
    result = search.verify_pair(pair, expanded, names)
    for old, new in zip(original, result):
        assert "unused" not in new["feature_names"]
        np.testing.assert_allclose(new["coefficients"], old["coefficients"], rtol=2e-11, atol=2e-12)


def test_outer_holdout_targets_do_not_change_nested_selection_or_coefficients():
    cc = cases()
    kwargs = dict(ridges=(0., .01), stages=STAGES, folds=(0,))
    first = search.search_channel(cc, NAMES, POOLS, "Fx", "measured", **kwargs)
    changed = deepcopy(cc)
    for c in changed:
        if c["fold"] == 0:
            c["y"] += 1e9
            c["dy"] -= 1e8
            c["train_dy"] *= -1e7
    second = search.search_channel(changed, NAMES, POOLS, "Fx", "measured", **kwargs)
    assert first == second
    heldout = {c["run"] for c in cc if c["fold"] == 0}
    for stage in first[0].values():
        for pair in stage.values():
            for m in pair:
                assert not heldout & set(m["training_runs"])
                assert m["zero_updated"] is False
    scores = {audit["stage"]: audit["chosen_score"]["joint"] for audit in first[1]}
    assert scores["interactions"] <= scores["core"]


def test_unanchored_absolute_zero_cannot_affect_model_search():
    cc = cases()
    for c in cc:
        c["anchored"] = int(c["condition"]) % 2 == 0
    before = search.models_for(cc, NAMES, POOLS, (0., .01))
    changed = deepcopy(cc)
    for c in changed:
        if not c["anchored"]:
            c["y"] += 1e10
    after = search.models_for(changed, NAMES, POOLS, (0., .01))
    assert before == after


@pytest.mark.parametrize("ridge_a,ridge_d", [(0., 0.), (.01, 1.), (1., .001)])
def test_verify_pair_matches_augmented_svd_and_preserves_family_order(ridge_a, ridge_d):
    cc = cases()
    norm = s.scales_from_training(cc, 1e-12)
    # Deliberately interleaved selection order, unlike the tall matrix A/B order.
    selected = ["B::coupling", "A::rate", "intercept", "A::flow", "B::flow",
                "B::rate", "A::coupling"]
    ids = [c["run"] for c in cc]
    pair = j.fit_joint(j.joint_statistics(cc, NAMES, norm), j.label_names(NAMES),
                       selected, ridge_a, ridge_d, ids, ids)
    original = deepcopy(pair)
    block = j.joint_matrix(cc, NAMES, pair[0]["feature_names"], pair[1]["feature_names"], norm)
    scale = np.linalg.norm(block.x, axis=0)
    penalty = [0. if k == "intercept" else ridge_d if k.startswith("B::") else ridge_a
               for k in block.names]
    xx = np.vstack([block.x/scale, np.diag(np.sqrt(penalty))])
    yy = np.r_[block.y, np.zeros(len(scale))]
    expected = np.linalg.lstsq(xx, yy, rcond=None)[0]/scale
    verified = search.verify_pair(pair, cc, NAMES)
    coef = np.r_[verified[0]["coefficients"], verified[1]["coefficients"]]
    np.testing.assert_allclose(coef, expected, rtol=2e-11, atol=2e-12)
    for old, new in zip(original, verified):
        assert new["feature_names"] == old["feature_names"]
        assert new["training_runs"] == old["training_runs"]
        assert new["solver"]["method"] == "verified_tall_joint_design_QR_SVD"
        assert new["zero_updated"] is False
        np.testing.assert_allclose(list(new["coefficient_by_name"].values()), new["coefficients"])


def test_condition_may_not_span_outer_folds():
    cc = cases()
    duplicate = deepcopy(cc[0])
    duplicate["fold"] = 2
    with pytest.raises(ValueError, match="condition spans"):
        search.search_channel(cc+[duplicate], NAMES, POOLS, "Fx", "measured",
                              ridges=(0.,), stages=STAGES, folds=(0,))
