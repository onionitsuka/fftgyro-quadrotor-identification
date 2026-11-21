"""Physics identities and leakage checks for the composed force-model search."""
from copy import deepcopy
import pickle

import numpy as np
import pytest

import analysis as a
import analysis_search as s
from scripts import refit_theory_forces as refit


def flow_data(seed=14, n=18):
    """Independent state/rotor samples, including zero flow and stopped rotors."""
    cfg = a.Settings()
    rng = np.random.default_rng(seed)
    velocity = rng.normal(size=(n, 3)) * 4
    velocity[0] = 0
    rate = rng.normal(size=(n, 3))
    positions = np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]) * cfg.arm_xy_m
    local = velocity[:, None, :] + np.cross(rate[:, None, :], positions)
    omega = rng.uniform(100, 800, (n, 4))
    omega[1] = 0
    mean_speed = np.sqrt(np.mean(omega**2, axis=1)) * cfg.rotor_radius_m
    mu = np.divide(velocity, mean_speed[:, None], out=np.zeros_like(velocity),
                   where=mean_speed[:, None] != 0)
    state = {"mu_x": mu[:, 0], "mu_y": mu[:, 1], "mu_z": mu[:, 2],
             "abs_mu_x": abs(mu[:, 0]), "abs_mu_y": abs(mu[:, 1])}
    state.update({k: rng.normal(size=n) for k in refit.MOTIONS})
    data = dict(state=state, air_frd_m_s=velocity, omega_body_rad_s=rate,
                force_scale=cfg.rho * 4*np.pi*cfg.rotor_radius_m**2 * mean_speed**2,
                rotor_flow=dict(velocity_frd_m_s=local, omega_abs_rad_s=omega,
                                positions_frd_m=positions))
    return cfg, data


@pytest.mark.parametrize("output,j", [("Fx", 0), ("Fy", 1)])
def test_physical_force_bases_are_dimensional_vector_equations(output, j):
    cfg, data = flow_data()
    bank, pools = refit.feature_bank(data, cfg, output)
    velocity = data["air_frd_m_s"]
    local = data["rotor_flow"]["velocity_frd_m_s"]
    omega = data["rotor_flow"]["omega_abs_rad_s"]
    area = np.pi*cfg.rotor_radius_m**2
    drag = -.5*cfg.rho*4*area*np.linalg.norm(velocity, axis=1)*velocity[:, j]
    np.testing.assert_allclose(bank["theory__body_drag__1"], drag)
    np.testing.assert_allclose(bank["theory__rotor_plane_linear"],
        -cfg.rho*area*cfg.rotor_radius_m*np.sum(omega*local[:, :, j], axis=1))
    np.testing.assert_allclose(bank["theory__rotor_plane_quadratic"],
        -cfg.rho*area*np.sum(np.linalg.norm(local[:, :, :2], axis=2)*local[:, :, j], axis=1))
    np.testing.assert_allclose(bank["local_axial_inplane"],
        -cfg.rho*area*np.sum(local[:, :, j]*local[:, :, 2], axis=1))
    assert bank["theory__rotor_plane_linear"][1] == 0  # no hidden RPM floor
    assert np.isfinite(bank["local_axial_inplane"]).all()
    assert np.linalg.norm(bank["local_axial_inplane"]) > 0
    for base in refit.BASES:
        for motion in refit.MOTIONS:
            np.testing.assert_allclose(bank[f"coupled__{base}__{motion}"],
                                       bank[base]*data["state"][motion])

    changed = deepcopy(data)
    changed["rotor_flow"]["omega_abs_rad_s"] *= 3
    new, _ = refit.feature_bank(changed, cfg, output)
    np.testing.assert_array_equal(new["theory__body_drag__1"], drag)
    np.testing.assert_array_equal(new["local_axial_inplane"], bank["local_axial_inplane"])
    assert set(pools["physical_core"]) <= set(pools["physical_direction"])
    assert set(pools["physical_direction"]) <= set(pools["physical_coupled"])


@pytest.mark.parametrize("output", ["Fx", "Fy"])
def test_physical_pools_do_not_mix_polynomial_aliases_or_zero_axial_columns(output):
    cfg, data = flow_data()
    bank, pools = refit.feature_bank(data, cfg, output)
    assert len(pools["sun"]) == 21
    common = {"intercept", "1"}
    for label in ("physical_core", "physical_direction", "physical_coupled"):
        assert set(pools[label]) & set(pools["sun"]) == common
        assert len(pools[label]) == len(set(pools[label]))
        assert not any("body_normal" in k or k in {
            "theory__rotor_axial", "theory__rotor_horizontal2", "theory__rotor_axial2",
            "rotor_inplane_linear", "rotor_inplane_quadratic"} for k in pools[label])
        assert all(np.linalg.norm(bank[k]) > 0 for k in pools[label] if k != "intercept")
    assert not any(k.endswith("__nx2") for k in pools["physical_direction"])


@pytest.mark.parametrize("output", ["Fx", "Fy"])
def test_individual_rotor_terms_sum_to_aggregate_without_duplicate_pool_columns(output):
    cfg, data = flow_data()
    bank, pools = refit.feature_bank(data, cfg, output)
    aggregates = dict(linear="theory__rotor_plane_linear",
                      quadratic="theory__rotor_plane_quadratic", axial="local_axial_inplane")
    individual = {f"individual__{i}__{kind}" for i in range(1, 5) for kind in aggregates}
    for kind, aggregate in aggregates.items():
        np.testing.assert_allclose(sum(bank[f"individual__{i}__{kind}"] for i in range(1, 5)),
                                   bank[aggregate], rtol=2e-14, atol=2e-14)
    expected = {"intercept", "1"} | individual | {
        k for k in bank if k.startswith("theory__body_drag__")}
    assert set(pools["per_rotor"]) == expected
    assert not set(aggregates.values()) & set(pools["per_rotor"])
    assert set(pools["per_rotor"]) & set(pools["sun"]) == {"intercept", "1"}
    # Inputs contain only own-route kinematics/RPM: no phase, period or other route.
    assert all(np.isfinite(bank[k]).all() for k in individual)
    assert all(bank[f"individual__{i}__linear"][1] == 0 for i in range(1, 5))


def model(names, coefficients):
    return dict(feature_names=names, coefficients=list(coefficients),
                coefficient_by_name=dict(zip(names, coefficients)))


def synthetic_cases(seed=912, conditions=9):
    rng = np.random.default_rng(seed)
    names = ["intercept", "x", "z", "xz"]
    cases = []
    for i in range(conditions):
        for repeat in range(1 + (i % 3 == 0)):
            n = 65 + 7*repeat
            x, z = rng.normal(size=(2, n))
            xx = np.c_[np.ones(n), x, z, x*z]
            xr = xx[:5].mean(axis=0)
            y = .2 + .8*x - .3*z + .6*x*z + rng.normal(size=n)*.015
            yr = float(y[:5].mean())
            dx, dy = xx-xr, y-yr
            cases.append(dict(run=f"run-{i}-{repeat}", condition=f"condition-{i}",
                fold=i % 3, anchored=True, validation=True, x=xx, xr=xr, y=y,
                dx=dx, dy=dy, train_dx=dx.copy(), train_dy=dy.copy(), weights=np.ones(n)/n))
    return names, cases


def test_fast_pair_scores_match_reference_with_repeated_conditions():
    names, cases = synthetic_cases()
    cases[0]["anchored"] = False
    cases[2]["validation"] = False
    foundations = {("a", 0.): model(names, [.1, .2, -.3, .5]),
                   ("a", .01): model(names, [-.2, .7, .1, -.2])}
    increments = {("b", 0.): model(names[1:], [.7, -.2, .4]),
                  ("b", .01): model([], [])}
    norm = (1.3, .7)
    fast = refit.score_pairs(cases, foundations, increments, names, norm)
    reference = s.pair_scores(cases, foundations, increments, names, norm)
    assert fast.keys() == reference.keys()
    for key in fast:
        for metric in fast[key]:
            assert fast[key][metric] == pytest.approx(reference[key][metric], rel=2e-13, abs=2e-14)


def test_outer_holdout_targets_cannot_change_search_or_coefficients():
    names, cases = synthetic_cases()
    pools = dict(core=names[:3], coupled=names)
    stages = dict(core=("core",), interactions=("core", "coupled"))
    kwargs = dict(ridges=(0., .01), stages=stages, folds=(0,))
    first, audit = refit.search_channel(cases, names, pools, "Fx", "measured", **kwargs)
    changed = deepcopy(cases)
    for c in changed:
        if c["fold"] == 0:
            c["y"] += 1e8
            c["dy"] -= 1e7
            c["train_dy"] *= -1e9
    second, second_audit = refit.search_channel(changed, names, pools, "Fx", "measured", **kwargs)
    assert first == second
    assert audit == second_audit
    heldout_ids = {c["run"] for c in cases if c["fold"] == 0}
    for stage in first.values():
        for ma, md in stage.values():
            assert not heldout_ids & set(ma["training_runs"])
            assert not heldout_ids & set(md["training_runs"])


def test_more_candidate_pools_cannot_worsen_best_inner_objective():
    names, cases = synthetic_cases()
    pools = dict(core=names[:3], coupled=names)
    stages = dict(core=("core",), interactions=("core", "coupled"))
    _, audit = refit.search_channel(cases, names, pools, "Fx", "measured",
        ridges=(0., .01), stages=stages, folds=("final",))
    scores = {v["stage"]: v["chosen_score"]["joint"] for v in audit}
    assert scores["interactions"] <= scores["core"]
    assert scores["interactions"] < .1*scores["core"]


def test_attaching_one_rpm_route_cannot_read_or_mutate_the_other():
    cfg, data = flow_data()
    data["indices"] = dict(selected=np.arange(4, 18), reference=np.arange(4))
    channel = dict(features={}, reference_features={})
    run = dict(search_data=dict(measured=data, command_route=deepcopy(data)),
        channels={"Fx": dict(measured=deepcopy(channel), command_route=deepcopy(channel))})
    untouched = pickle.dumps((run["search_data"]["command_route"], run["channels"]["Fx"]["command_route"]))
    names, pools = refit.attach_banks([run], cfg, "Fx", "measured")
    assert pickle.dumps((run["search_data"]["command_route"], run["channels"]["Fx"]["command_route"])) == untouched
    before = deepcopy(run["channels"]["Fx"]["measured"])
    # Deliberately incompatible values in the inactive route must be irrelevant.
    run["search_data"]["command_route"] = {"invalid_other_route": True}
    other = dict(features={"unrelated": np.array([1e30])}, reference_features={})
    run["channels"]["Fx"]["command_route"] = other
    again_names, again_pools = refit.attach_banks([run], cfg, "Fx", "measured")
    assert again_names == names and again_pools == pools
    for group in before:
        for name, value in before[group]["systematic"].items():
            np.testing.assert_array_equal(run["channels"]["Fx"]["measured"][group]["systematic"][name], value)
    assert run["channels"]["Fx"]["command_route"] is other
