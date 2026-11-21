"""Theory bases: mechanics, angular continuity, selection and route/fold isolation."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a


def state(velocity, rate=None, speed=None, cfg=None):
    cfg = cfg or a.Settings()
    velocity = np.asarray(velocity, float)
    rate = np.zeros_like(velocity) if rate is None else np.asarray(rate, float)
    speed = np.full((len(velocity), 4), 400.) if speed is None else np.asarray(speed, float)
    raw = dict(air=velocity, omega=rate)
    return dict(air_frd_m_s=velocity, omega_body_rad_s=rate,
                rotor_flow=a.rotor_local_flow(raw, speed, cfg))


@pytest.mark.parametrize("output", a.OUTPUTS)
def test_names_match_bank_and_are_finite_at_rest_and_stopped_rotors(output):
    data = state([[0, 0, 0], [4, -3, 1]], speed=np.zeros((2, 4)))
    bank = a.theory_feature_bank(data, a.Settings(), output)
    assert set(bank) == set(a.theory_feature_names(output))
    assert all(v.shape == (2,) and np.isfinite(v).all() for v in bank.values())
    assert all(v[0] == 0 for v in bank.values())


def test_drag_is_exact_vector_quadratic_opposes_motion_and_reverses():
    cfg = a.Settings()
    v = np.array([[6., -2., 1.], [-6., 2., -1.], [0., 0., 0.]])
    data = state(v)
    f = np.column_stack([a.theory_feature_bank(data, cfg, k)["theory__body_drag__1"]
                         for k in ("Fx", "Fy", "Fz")])
    expected = -0.5 * cfg.rho * (4 * np.pi * cfg.rotor_radius_m**2) * np.linalg.norm(v, axis=1)[:, None] * v
    np.testing.assert_allclose(f, expected)
    np.testing.assert_allclose(f[1], -f[0])
    assert np.all(np.sum(f*v, axis=1) <= 0)


def test_angle_factors_represent_incidence_not_world_euler_angles():
    cfg = a.Settings()
    alpha, beta = np.deg2rad([25., -35.])
    n = np.array([np.cos(alpha)*np.cos(beta), np.sin(beta), np.sin(alpha)*np.cos(beta)])
    data = state([7*n])
    b = a.theory_feature_bank(data, cfg, "Fx")
    base = b["theory__body_drag__1"]
    for j, key in enumerate(("nx", "ny", "nz")):
        np.testing.assert_allclose(b["theory__body_drag__"+key], base*n[j])
    np.testing.assert_allclose(b["theory__body_drag__nx_nz"], base*n[0]*n[2])
    data["euler"] = np.array([[1000., -2000., 3000.]])
    for key, value in a.theory_feature_bank(data, cfg, "Fx").items():
        np.testing.assert_array_equal(value, b[key])


@pytest.mark.parametrize("output", a.OUTPUTS)
def test_full_turn_branch_and_zero_speed_are_continuous(output):
    angle = np.array([np.pi-1e-8, -np.pi+1e-8])
    v = np.c_[5*np.cos(angle), 5*np.sin(angle), np.zeros(2)]
    b = a.theory_feature_bank(state(v), a.Settings(), output)
    for value in b.values():
        # Coefficients are unit values here (rotor drag can be hundreds of N).
        assert abs(value[0]-value[1]) < 2e-5
    quiet = state(np.array([[1., 2., 3.]])*1e-12, speed=np.zeros((1, 4)))
    assert max(abs(v[0]) for v in a.theory_feature_bank(quiet, a.Settings(), output).values()) < 1e-20


def test_normal_forces_perpendicular_and_arm_moments_are_cross_products():
    cfg = a.Settings()
    data = state([[6., -2., 1.], [1., 4., -3.]])
    banks = {k: a.theory_feature_bank(data, cfg, k) for k in a.OUTPUTS}
    for label in ("body_normal_y", "body_normal_z"):
        f = np.column_stack([banks[k]["theory__"+label] for k in ("Fx", "Fy", "Fz")])
        np.testing.assert_allclose((f*data["air_frd_m_s"]).sum(axis=1), 0, atol=1e-14)
    drag = np.column_stack([banks[k]["theory__body_drag__1"] for k in ("Fx", "Fy", "Fz")])
    for j, name in enumerate("xyz"):
        expected = np.cross(cfg.arm_xy_m*np.eye(3)[j], drag)
        for axis, output in enumerate(("Mx", "My", "Mz")):
            if axis != j:
                np.testing.assert_allclose(banks[output][f"theory__body_drag_arm_{name}__1"], expected[:, axis])


def test_rotor_local_forces_and_moments_use_same_positions_and_speeds():
    cfg = a.Settings(cg_flu_m=(0.01, -0.02, 0.095))
    data = state([[6., -2., 1.]], [[0.4, -0.3, 0.6]], [[200., 400., 600., 800.]], cfg)
    flow = data["rotor_flow"]
    plane = flow["velocity_frd_m_s"].copy()
    plane[:, :, 2] = 0
    forces = -cfg.rho*np.pi*cfg.rotor_radius_m**3*flow["omega_abs_rad_s"][:, :, None]*plane
    moments = np.cross(flow["positions_frd_m"], forces).sum(axis=1)
    for j, output in enumerate(a.OUTPUTS):
        expected = forces.sum(axis=1)[:, j] if j < 3 else moments[:, j-3]
        np.testing.assert_allclose(a.theory_feature_bank(data, cfg, output)["theory__rotor_plane_linear"], expected)


def test_unit_positive_damping_opposes_rotation():
    cfg = a.Settings()
    rate = np.array([[0.3, -0.4, 0.8], [-0.3, 0.4, -0.8]])
    data = state([[5., 1., 0.]]*2, rate)
    for j, output in enumerate(("Mx", "My", "Mz")):
        bank = a.theory_feature_bank(data, cfg, output)
        for name in ("body", "rotor", "quadratic"):
            assert np.all(bank["theory__damping_"+name]*rate[:, j] < 0)


def test_missing_rpm_is_not_replaced_and_body_drag_is_rpm_independent():
    cfg = a.Settings()
    data = state([[5., 1., 0.]]*2, speed=[[400.]*4, [np.nan]*4])
    b = a.theory_feature_bank(data, cfg, "Fx")
    assert np.isnan(b["theory__rotor_plane_linear"][1])
    assert b["theory__body_drag__1"][0] == b["theory__body_drag__1"][1]


def test_predicted_route_theory_never_reads_measured_rpm():
    cfg = a.Settings(smooth_seconds=0.1, fz_model="sun_graybox")
    raw = dict(air=np.tile([5., -2., 1.], (100, 1)), omega=np.tile([0.1, -0.2, 0.3], (100, 1)),
               valid_motion=np.ones(100, bool), rpm=np.full((100, 4), 3000.))
    command_rpm = np.full((100, 4), 4000.)
    first = a.route_state(raw, command_rpm, cfg)
    raw["rpm"] *= 2
    second = a.route_state(raw, command_rpm, cfg)
    for output in a.OUTPUTS:
        b1, b2 = (a.theory_feature_bank(d, cfg, output) for d in (first, second))
        for key in b1:
            np.testing.assert_allclose(b1[key], b2[key], equal_nan=True)


def test_theoretical_drag_and_angle_term_can_actually_be_selected():
    cfg = a.Settings()
    rng = np.random.default_rng(42)
    bank = a.theory_feature_bank(state(rng.normal(size=(600, 3))*5), cfg, "Fx")
    names = ("theory__body_drag__1", "theory__body_drag__nz", "theory__body_normal_y")
    x = np.column_stack([bank[k] for k in names])
    y = x[:, 0]*0.8 + x[:, 1]*0.5
    model = a.select_theory_model(a.RegressionBlocks(x, y, names, {}), dict(ridge=0.))
    assert set(model["feature_names"]) == set(names[:2])
    np.testing.assert_allclose(a.predict_correction(model, bank), y, atol=1e-11)
    assert model["theory_candidates"] == a.THEORY_VERSION
    assert not model["frozen_structure"]


def prepared_case():
    """Small all-channel signal-derived fixture, with deliberately different routes."""
    rng = np.random.default_rng(19)
    cfg = a.Settings()
    runs, yaw = [], []
    for i in range(18):
        v = rng.normal(size=(60, 3))*3 + [4, 1, 0.5]
        rates = rng.normal(size=(60, 3))*0.4
        run = dict(run=str(i), role="T", condition_id=f"c{i:03d}", fold=i % 3, anchored=i % 2 == 0, channels={})
        yr = dict(run=str(i), condition_id=f"c{i:03d}", fold=i % 3, original_count=60, route={})
        for output in a.OUTPUTS:
            ch = dict(reference_available=True, take=np.ones(60, bool))
            for route in a.ROUTES:
                speed = rng.uniform(300, 700, (60, 4)) if route == "measured" else rng.uniform(350, 650, (60, 4))
                bank = a.theory_feature_bank(state(v, rates, speed), cfg, output)
                bank["legacy"] = v[:, 0]
                features = {"expanded": bank}
                if output in a.EXTRA_CHANNELS:
                    for spec in ("theory_compact", "theory_expanded"):
                        features[spec] = {k: bank.get(k, v[:, 0]) for k in a.extra_feature_names(output, spec)}
                refs = {s: {k: float(x[:5].mean()) for k, x in b.items()} for s, b in features.items()}
                d = dict(physical=np.zeros(60), reference_physical=0., features=features, reference_features=refs)
                ch[route] = d
                if output == "Mz":
                    yr["route"][route] = dict(augmented={k: x-x[:5].mean() for k, x in bank.items()},
                                              physical=np.zeros(60), rpm=np.full(60, 4000.))
            key = "theory__body_drag__1" if output.startswith("F") else "theory__body_pressure__nx_ny"
            truth = ch["measured"]["features"]["expanded"][key]*0.7 + 0.02
            if not run["anchored"]:
                truth = truth + 100 + i
            ch.update(truth=truth, reference=float(truth[:5].mean()), physical=np.zeros(60), physical_reference=0.)
            run["channels"][output] = ch
        yr["y"] = run["channels"]["Mz"]["truth"] - run["channels"]["Mz"]["reference"]
        runs.append(run)
        yaw.append(yr)
    models = dict(foundation={}, increment={}, yaw={})
    for fold in ("final", 0):
        train = [r for r in runs if fold == "final" or r["fold"] != fold]
        for route in a.ROUTES:
            for family in ("foundation", "increment"):
                for output in a.CHANNELS:
                    if family == "increment" and output == "Mz":
                        continue
                    rr = [r for r in train if r["anchored"]] if family == "foundation" and output != "Fz" else train
                    models[family][f"{fold}/{route}/{output}"] = dict(
                        feature_names=(["intercept"] if family == "foundation" else [])+["legacy"],
                        training_runs=[r["run"] for r in rr], spec="expanded", ridge=0.01,
                        predictive_candidate=dict(loss="gls", balanced=False))
            models["yaw"][f"{fold}/{route}"] = dict(feature_names=["legacy"], training_runs=[r["run"] for r in train],
                bank="augmented", ridge=0.01, extra_ratio=0.25, local_penalty=3.)
    cal = dict(ids=[r["run"] for r in runs], covariance={k: np.eye(len(runs))*0.01 for k in a.CHANNELS},
               noise={k: 0.1 for k in a.CHANNELS})
    return dict(models=models, yaw_noise=0.1), runs, yaw, cal, cfg


def test_main_refit_selects_theory_for_all_main_axes_without_heldout_load_leakage():
    profile, runs, yaw, cal, cfg = prepared_case()
    before = a.refit_models(profile, runs, yaw, cal, cfg, reselect_physics=True)
    assert all(m["local_penalty"] == 1.0 for m in before["yaw"].values())
    assert all(m["candidate_penalty_policy"] == "uniform_normalized_columns" for m in before["yaw"].values())
    for family in before:
        for key, model in before[family].items():
            assert model["theory_candidates"] == a.THEORY_VERSION
            assert any(k.startswith("theory__") for k in model["selection"]["candidate_names"])
            if key.startswith("0/"):
                assert all(r["fold"] != 0 for r in runs if r["run"] in model["training_runs"])
    for r in runs:
        if r["fold"] == 0:
            for ch in r["channels"].values():
                ch["truth"] *= -1000
                ch["reference"] *= -1000
    for r in yaw:
        if r["fold"] == 0:
            r["y"] *= -1000
    after = a.refit_models(profile, runs, yaw, cal, cfg, reselect_physics=True)
    for family in before:
        for key in before[family]:
            if key.startswith("0/"):
                assert before[family][key]["coefficient_by_name"] == after[family][key]["coefficient_by_name"]


def test_extra_axes_reselection_uses_theory_and_unanchored_only_as_differences(monkeypatch):
    _, runs, _, _, _ = prepared_case()
    monkeypatch.setattr(a, "EXTRA_CHANNELS", ("Fx",))
    before = a.extra_matrix(runs, "measured", "Fx", "theory_compact", "increment")
    shifted = copy.deepcopy(runs)
    for r in shifted:
        r["channels"]["Fx"]["truth"] += 999
        r["channels"]["Fx"]["reference"] += 999
    after = a.extra_matrix(shifted, "measured", "Fx", "theory_compact", "increment")
    np.testing.assert_allclose(before.y, after.y, atol=1e-12)
    models = a.fit_extra_channels(runs, reselect_physics=True)
    for key, model in models.items():
        assert model["spec"].startswith("theory_")
        assert "theory__body_drag__1" in model["selection"]["candidate_names"]
        if key.endswith("/foundation"):
            assert all(r["anchored"] for r in runs if r["run"] in model["training_runs"])


def test_selected_banks_routes_new_terms_without_changing_old_terms():
    cfg = a.Settings(smooth_seconds=0.1, fz_model="sun_graybox")
    raw = dict(air=np.tile([5., -2., 1.], (100, 1)), omega=np.zeros((100, 3)), valid_motion=np.ones(100, bool))
    data = a.route_state(raw, np.full((100, 4), 4000.), cfg)
    data["rms_speed_ratio"] = np.ones(100)
    for output in a.CHANNELS:
        old_names = ["mu_x"] if output != "Fz" else ["mu_h2"]
        before = a.selected_banks(data, cfg, output, {"expanded": old_names})
        after = a.selected_banks(data, cfg, output, {"expanded": old_names+list(a.theory_feature_names(output))})
        for key in old_names:
            np.testing.assert_array_equal(before["expanded"][key], after["expanded"][key])
        assert set(a.theory_feature_names(output)).issubset(after["expanded"])


def test_existing_bounds_survive_selection_and_empty_correction_is_allowed():
    x = np.linspace(-1, 1, 120)[:, None]
    matrix = a.RegressionBlocks(x, 3*x[:, 0], ("prop_gain",), {})
    model = a.select_theory_model(matrix, dict(ridge=0., bounds={"prop_gain": [-0.5, 0.5]}))
    assert model["coefficient_by_name"]["prop_gain"] == pytest.approx(0.5)
    empty = a.select_theory_model(a.RegressionBlocks(x, np.zeros(120), ("x",), {}), dict(ridge=0.))
    assert empty["feature_names"] == []


def test_theory_label_does_not_change_selection_or_normalized_fit():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(500, 4))
    y = 2*x[:, 1] - 0.4*x[:, 3] + rng.normal(size=500)*0.1
    plain = tuple(f"candidate_{i}" for i in range(4))
    tagged = tuple("theory__"+k for k in plain)
    m1 = a.select_theory_model(a.RegressionBlocks(x, y, plain, {}), dict(ridge=0.01))
    m2 = a.select_theory_model(a.RegressionBlocks(x, y, tagged, {}), dict(ridge=0.01))
    assert m1["feature_names"] == [k.removeprefix("theory__") for k in m2["feature_names"]]
    np.testing.assert_array_equal(m1["coefficients"], m2["coefficients"])


def test_uniform_yaw_penalty_does_not_depend_on_local_or_theory_name():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(300, 2))
    y = 0.7*x[:, 0] + 0.2*x[:, 1]
    local = a.fit_yaw_matrix(x, y, ["local1_mu_x", "theory__body_drag_arm_x__1"], 0.1,
                            "augmented", local_penalty=1.)
    generic = a.fit_yaw_matrix(x, y, ["a", "b"], 0.1, "augmented", local_penalty=1.)
    np.testing.assert_array_equal(local["coefficients"], generic["coefficients"])
