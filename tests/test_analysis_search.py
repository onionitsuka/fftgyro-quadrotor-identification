"""Uniform candidate coverage, training isolation and sufficient-statistic regression."""
import sys
from pathlib import Path
import numpy as np
import pytest
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a
import analysis_search as s


def test_uniform_eligibility_and_nested_degrees():
    sets = {d: set(s.uniform_names(d)) for d in (2, 3, 4)}
    assert sets[2] < sets[3] < sets[4]
    for f in s.FLOW:
        for m in s.MOTION:
            assert f"cross__{f}*{m}" in sets[2]
            assert f"cross__{f}^4*{m}" in sets[4]
    for m in s.MOTION:
        assert f"cross__{m}^2" in sets[2]
    assert len(s.local_names()) == 156
    assert len(set(s.local_names())) == 156


@pytest.mark.parametrize("seed", range(8))
def test_gram_selector_matches_qr_sun_and_coefficients(seed):
    rng = np.random.default_rng(seed)
    x = np.c_[np.ones(300), rng.normal(size=(300, 16))]
    y = .6 + x[:, 3]*2 - x[:, 8]*.5 + rng.normal(size=300)*.2
    names = ["intercept"]+[f"c{i}" for i in range(16)]
    matrix = a.RegressionBlocks(x, y, tuple(names), {})
    selected_qr, _ = a.select_extra_terms(matrix)
    ss = s.statistics(x, y)
    selected_gram, _ = s.gram_select(ss, names)
    assert set(selected_gram) == set(selected_qr)
    for ridge in (0, .01, 1):
        actual = s.fit_stats(ss, names, selected_gram, ridge, ["training"])
        expected = a.fit_regularized(a.subset_blocks(matrix, selected_gram), ridge=ridge)
        np.testing.assert_allclose(actual["coefficients"], expected["coefficients"], atol=1e-12)


def test_rank_duplicates_are_not_forced():
    x = np.linspace(-1, 1, 100)
    matrix = np.c_[np.ones(100), x, 2*x, np.zeros(100)]
    selected, _ = s.gram_select(s.statistics(matrix, 1+3*x), ["intercept", "x", "twox", "zero"])
    assert "intercept" in selected and "zero" not in selected
    assert len(selected) == 2


def test_statistics_addition_preserves_loss():
    rng = np.random.default_rng(77)
    x, y = rng.normal(size=(90, 9)), rng.normal(size=90)
    chunks = s.add_statistics([s.statistics(x[:35], y[:35]), s.statistics(x[35:], y[35:])])
    for k, v in s.statistics(x, y).items():
        np.testing.assert_allclose(chunks[k], v, atol=1e-12)


def test_existing_propulsion_gain_bound_is_preserved():
    rng = np.random.default_rng(17)
    x = np.c_[np.ones(100), rng.normal(size=100)]
    names = ["intercept", "prop_gain"]
    y = 2 + 3*x[:, 1]
    model = s.fit_stats(s.statistics(x, y), names, names, 0., ["training"],
                        bounds={"prop_gain":(-.5, .5)})
    assert model["coefficient_by_name"]["prop_gain"] == pytest.approx(.5)
    expected = a.fit_regularized(a.RegressionBlocks(x, y, tuple(names), {}),
                                bounds={"prop_gain":(-.5, .5)})
    np.testing.assert_allclose(model["coefficients"], expected["coefficients"], atol=1e-9)


def test_no_reference_load_enters_predictions():
    c = dict(dx=np.array([[0., 1.], [0., 2.]]), xr=np.array([1., 3.]),
             y=np.array([20., 30.]), dy=np.array([10., 20.]))
    model = dict(feature_names=["intercept", "x"], coefficients=[2., 4.])
    p = s.prediction(c, model, ["intercept", "x"], "xr")
    c["y"] *= 1e6
    assert s.prediction(c, model, ["intercept", "x"], "xr") == p == 14.


def test_joint_scoring_penalizes_bad_absolute_even_with_good_delta():
    c = dict(condition="A", anchored=True, dx=np.array([[0., 1.], [0., 2.]]),
             xr=np.array([1., 0.]), y=np.array([3., 4.]), dy=np.array([1., 2.]))
    good = dict(feature_names=["intercept", "x"], coefficients=[2., 1.])
    bad = dict(feature_names=["intercept", "x"], coefficients=[20., 1.])
    result = s.pair_scores([c], {0:good, 1:bad}, {0:good}, ["intercept", "x"], [1., 1.])
    assert result[0, 0]["joint"] == 0
    assert result[1, 0]["joint"] > 100


def test_restores_all_output_dictionaries_and_formula():
    cfg = a.Settings(fz_model="sun_graybox", yaw_motion_model="sun_blended")
    raw = dict(air=np.tile([5., -2., 1.], (100, 1)), omega=np.tile([.1, -.2, .3], (100, 1)),
               valid_motion=np.ones(100, bool))
    d = a.route_state(raw, np.full((100, 4), 4000.), cfg)
    d["rms_speed_ratio"] = np.ones(100)
    for output in a.OUTPUTS:
        bank, restored = s.expanded_bank(d, cfg, output)
        if output in a.EXTRA_CHANNELS:
            expected = a.extra_feature_names(output, "expanded")
        else:
            expected = a.feature_dictionary(output, "expanded")
        assert set(expected) <= restored
        if output == "My":
            assert "mu_x^5*abs_mu_y^2*qbar" in restored
        if output == "Mz":
            assert "mu_y^3*rbar*sun_beta_60_90" in restored
        scale = d["force_scale"] if output.startswith("F") else d["moment_scale"]
        np.testing.assert_allclose(bank["cross__mu_z^4*ur"],
                                  scale*d["state"]["mu_z"]**4*d["state"]["ur"], equal_nan=True)


def test_inner_fit_never_sees_outer_targets(monkeypatch):
    monkeypatch.setattr(s, "SPECS", ("restored",))
    monkeypatch.setattr(s, "RIDGES", (0., .1))
    cases = []
    for i in range(18):
        x = np.c_[np.ones(20), np.linspace(-1, 1, 20)+i/10]
        xr = x[:3].mean(axis=0)
        y = 2+.3*x[:, 1]
        cases.append(dict(run=str(i), condition=f"c{i}", fold=i%3, anchored=True,
            x=x, xr=xr, y=y, dx=x-xr, dy=y-y[:3].mean(), train_dx=x-xr,
            train_dy=y-y[:3].mean(), weights=np.full(20, .05)))
    names = ["intercept", "x"]
    first, _ = s.fit_channel(cases, names, dict(restored=names), {}, .001, "My", "measured")
    json.dumps(first, allow_nan=False)
    for c in cases:
        if c["fold"] == 0:
            c["y"] += 1000
            c["dy"] += 1000
            c["train_dy"] += 1000
    second, _ = s.fit_channel(cases, names, dict(restored=names), {}, .001, "My", "measured")
    assert first["0"] == second["0"]
