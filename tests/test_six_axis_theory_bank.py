"""Six-axis dimensional mechanisms, stable dictionaries and route isolation."""
from copy import deepcopy

import numpy as np
import pytest

import analysis as a
from scripts import refit_theory_forces as old
from scripts import six_axis_theory_bank as six


def data(cfg=None, n=500):
    cfg = cfg or a.Settings(fz_model="sun_graybox", yaw_motion_model="sun_blended")
    rng = np.random.default_rng(765432)
    velocity, rate = rng.normal(size=(n, 3))*6., rng.normal(size=(n, 3))
    omega = rng.uniform(200., 700., (n, 4))
    bar = np.sqrt(np.mean(omega**2, axis=1))
    mu, pqr = velocity/(bar[:, None]*cfg.rotor_radius_m), rate*cfg.arm_xy_m/(bar[:, None]*cfg.rotor_radius_m)
    state = {k: v for k, v in zip(("mu_x", "mu_y", "mu_z"), mu.T)}
    state.update({k: v for k, v in zip(("pbar", "qbar", "rbar"), pqr.T)})
    state.update(abs_mu_x=abs(mu[:, 0]), abs_mu_y=abs(mu[:, 1]))
    for k in ("pbar", "qbar", "rbar"):
        state["abs_"+k] = abs(state[k])
    for name, signs in (("up", [-1, 1, 1, -1]), ("uq", [1, -1, 1, -1]), ("ur", a.SIGNS)):
        state[name] = omega**2 @ np.asarray(signs)/bar**2
        state["abs_"+name] = abs(state[name])
    raw = dict(air=velocity, omega=rate)
    flow = a.rotor_local_flow(raw, omega, cfg)
    state.update(a.rotor_local_yaw(raw, omega, cfg, flow=flow))
    state.update(a.sun_yaw_regions(mu, pqr[:, 2], state["ur"], rate[:, 2], cfg))
    state.update(mu_h2=mu[:, 0]**2+mu[:, 1]**2, inflow=.1-mu[:, 2])
    force_scale = cfg.rho*4*np.pi*cfg.rotor_radius_m**4*bar**2
    physical = np.zeros((n, 6))
    physical[:, 2] = -cfg.kf*np.sum(omega**2, axis=1)
    return dict(air_frd_m_s=velocity, omega_body_rad_s=rate, rotor_flow=flow,
                state=state, force_scale=force_scale, moment_scale=force_scale*cfg.arm_xy_m,
                physical=physical, rms_speed_ratio=bar/(4000*2*np.pi/60)), cfg


@pytest.mark.parametrize("output", a.OUTPUTS)
def test_all_axes_have_same_pool_contract_and_finite_dimensional_values(output):
    d, cfg = data()
    bank, pools = six.feature_bank(d, cfg, output)
    assert set(pools) == {"reference", "physical_core", "physical_direction", "physical_coupled", "per_rotor"}
    for names in pools.values():
        assert len(names) == len(set(names)) and "intercept" in names
        assert set(names)-{"intercept"} <= set(bank)
    assert all(v.shape == (500,) and np.isfinite(v).all() for v in bank.values())
    if output not in ("Fx", "Fy"):
        assert not any("rotor_sum__" in k for k in pools["per_rotor"])
        assert not any("individual__" in k for k in pools["physical_direction"])
        assert not any("body_drag_arm" in k or "body_normal" in k for k in pools["physical_direction"])


@pytest.mark.parametrize("output", ("Fx", "Fy"))
def test_previous_force_values_and_pools_are_exactly_preserved(output):
    d, cfg = data()
    old_bank, old_pools = old.feature_bank(d, cfg, output)
    bank, pools = six.feature_bank(d, cfg, output)
    assert set(bank) == set(old_bank)
    assert pools == {"reference" if k == "sun" else k: v for k, v in old_pools.items()}
    for name in bank:
        np.testing.assert_array_equal(bank[name], old_bank[name])


@pytest.mark.parametrize("output", ("Fz", "Mx", "My", "Mz"))
def test_physical_pools_have_no_global_zero_or_exact_dependent_columns(output):
    d, cfg = data(n=500)
    bank, pools = six.feature_bank(d, cfg, output)
    for pool, names in pools.items():
        if pool == "reference":
            continue
        x = np.column_stack([np.ones(500) if k == "intercept" else bank[k] for k in names])
        norms = np.linalg.norm(x, axis=0)
        assert (norms > 1e-20).all(), (output, pool)
        singular = np.linalg.svd(x/norms, compute_uv=False)
        assert singular[-1] > singular[0]*1e-10, (output, pool, singular[-1])


def test_rotor_sums_use_actual_lever_arms_and_thrust_scaling():
    cfg = a.Settings(cg_flu_m=(.012, -.024, .105))
    d, _ = data(cfg)
    flow, radius = d["rotor_flow"], cfg.rotor_radius_m
    thrust = np.zeros_like(flow["velocity_frd_m_s"])
    thrust[:, :, 2] = -cfg.rho*np.pi*radius**4*flow["omega_abs_rad_s"]**2
    moments = np.cross(flow["positions_frd_m"], thrust)
    for j, output in enumerate(("Mx", "My")):
        bank, _ = six.feature_bank(d, cfg, output)
        np.testing.assert_allclose(bank["rotor_sum__thrust"], moments[:, :, j].sum(axis=1))
        for i in range(4):
            np.testing.assert_allclose(bank[f"individual__{i+1}__thrust"], moments[:, i, j])
    bank, pools = six.feature_bank(d, cfg, "Fz")
    np.testing.assert_allclose(bank["rotor_sum__thrust"], thrust[:, :, 2].sum(axis=1))
    assert "prop_gain" in pools["reference"] and "prop_gain" not in pools["physical_core"]
    assert "1" not in pools["per_rotor"]


def test_thrust_bounds_preserve_existing_half_gain_limit():
    d, cfg = data()
    bank, _ = six.feature_bank(d, cfg, "Fz")
    bounds = six.coefficient_bounds(cfg, "Fz", bank)
    assert bounds["prop_gain"] == (-.5, .5)
    bound = bounds["rotor_sum__thrust"][1]
    np.testing.assert_allclose(bound*bank["rotor_sum__thrust"], .5*d["physical"][:, 2])
    for i in range(1, 5):
        assert bounds[f"individual__{i}__thrust"] == (-bound, bound)
    assert six.coefficient_bounds(cfg, "My", bank) == {}


def test_pools_do_not_depend_on_run_excitation_and_no_other_route_or_target_is_read():
    d, cfg = data()
    first = {output: six.feature_bank(d, cfg, output) for output in a.OUTPUTS}
    changed = deepcopy(d)
    changed.update(measured_rpm=np.full((500, 4), 1e8), truth=np.full((500, 6), 1e9),
                   bias_sensor=np.full(6, -1e12), run="heldout")
    for output in a.OUTPUTS:
        bank, pools = six.feature_bank(changed, cfg, output)
        assert pools == first[output][1]
        for key in bank:
            np.testing.assert_array_equal(bank[key], first[output][0][key])
    # A zero angular-rate realization must not silently delete candidate columns.
    quiet = deepcopy(d)
    for k in six.MOTIONS:
        quiet["state"][k] = np.zeros(500)
    for output in a.OUTPUTS:
        assert six.feature_bank(quiet, cfg, output)[1] == first[output][1]


def test_symmetric_signed_horizontal_torque_identity_is_removed():
    d, cfg = data()
    _, pools = six.feature_bank(d, cfg, "Mz")
    assert "rotor_sum__torque_horizontal2" not in pools["physical_core"]
    removed = six.candidate_audit(cfg, "Mz")["removed"]
    assert "rotor_sum__torque_horizontal2" in removed["physical_core"]
    assert any("individual__" in k and "torque_horizontal2" in k for k in removed["per_rotor"])
