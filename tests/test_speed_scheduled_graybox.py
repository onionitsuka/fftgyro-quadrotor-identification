import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from analysis import Settings  # noqa: E402
from simulate_graybox_forward_flight import OUTPUTS, Aircraft  # noqa: E402
from speed_scheduled_graybox import (  # noqa: E402
    ScheduledAircraft,
    fit_schedule,
    interpolation_weights,
)


def fixture(common=0):
    models = {o: dict(coefficients=[], selected_terms=[]) for o in OUTPUTS}
    models["Fz"] = dict(
        coefficients=[common, common], selected_terms=[[["mu_h2", 1]], [["mu_z", 2]]]
    )
    artifact = dict(
        settings=asdict(Settings(fz_model="sun_graybox")),
        motor_models=[],
        models={"measured": models},
    )
    base = Aircraft(artifact, "physical")
    nodes = [
        dict(
            reference_air_m_s=[v, 0, 0],
            reference_rotor_squared_rad_s2=base.hover_w**2,
            baseline_coefficients=[[v, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        )
        for v in (0, 5, 10)
    ]
    schedule = dict(speeds_m_s=[0, 5, 10], routes={"measured": nodes})
    return artifact, schedule


@pytest.mark.parametrize(
    "v,expected",
    [(0, [1, 0, 0]), (2.5, [0.5, 0.5, 0]), (5, [0, 1, 0]), (7.5, [0, 0.5, 0.5]), (10, [0, 0, 1])],
)
def test_weights(v, expected):
    np.testing.assert_allclose(interpolation_weights(v, np.array([0.0, 5, 10])), expected)


@pytest.mark.parametrize("v", [-0.1, 10.1, np.nan])
def test_no_silent_extrapolation(v):
    with pytest.raises(ValueError):
        interpolation_weights(v, np.array([0.0, 5, 10]))


def test_global_velocity_squared_gauge_cancels_locally():
    artifact, schedule = fixture(common=100)
    a = ScheduledAircraft(artifact, "measured", schedule, anchored=False)
    w = np.full(4, a.hover_w)
    direction = np.array([np.cos(0.2), 0, np.sin(0.2)])
    # Huge unobservable isotropic V² contribution cancels at each fixed-speed node.
    load = a.loads(7.5 * direction, np.zeros(3), w, np.zeros(4))
    physics = a.physics.loads(7.5 * direction, np.zeros(3), w, np.zeros(4))
    np.testing.assert_allclose(load - physics, [7.5, 0, 0, 0, 0, 0], atol=1e-8)


def test_initial_hover_is_anchored_once():
    artifact, schedule = fixture()
    a = ScheduledAircraft(artifact, "measured", schedule)
    np.testing.assert_allclose(
        a.loads(np.zeros(3), np.zeros(3), np.full(4, a.hover_w), np.zeros(4)),
        [0, 0, -a.mass * 9.80665, 0, 0, 0],
        atol=1e-12,
    )


def test_load_is_continuous_at_node():
    artifact, schedule = fixture()
    a = ScheduledAircraft(artifact, "measured", schedule, anchored=False)
    w = np.full(4, a.hover_w)
    left = a.loads(np.array([5 - 1e-7, 0, 0]), np.zeros(3), w, np.zeros(4))
    right = a.loads(np.array([5 + 1e-7, 0, 0]), np.zeros(3), w, np.zeros(4))
    np.testing.assert_allclose(left, right, atol=3e-7)


def test_baseline_does_not_fit_extra_rpm_slope_or_use_holdout():
    artifact, _ = fixture()
    artifact["models"]["command_route"] = artifact["models"]["measured"]
    physics = Aircraft(artifact, "physical", anchored=False)
    rows = []
    for speed in (0, 10):
        for run in ("one", "two"):
            for rpm in (4200, 4300):
                w = np.full(4, rpm * 2 * np.pi / 60)
                truth = physics.loads(np.array([speed, 0, 0]), np.zeros(3), w, np.zeros(4))
                row = dict(
                    wind_m_s=speed, heading_deg=0, pitch_deg=0, yaw_deg=0,
                    role="T", run=run, rate_max_deg_s=0, accel_max_deg_s2=0,
                    wind_basis="measured",
                )
                row.update({f"rpm_M{j}": rpm for j in range(1, 5)})
                row.update({f"target_{out}": truth[i] + 0.1 for i, out in enumerate(OUTPUTS)})
                rows.append(row)
    rows.append({**rows[0], "role": "H", "target_Fx": 10000})
    schedule = fit_schedule(artifact, rows)
    assert schedule["speeds_m_s"] == [0, 10]
    assert all(node["adopted"] for node in schedule["speed_node_coverage"])
    for node in schedule["routes"]["measured"]:
        assert node["rank"] == 1
        np.testing.assert_allclose(node["baseline_coefficients"][0], 0.1, atol=1e-12)
        np.testing.assert_allclose(node["baseline_coefficients"][1], 0)


def test_all_eligible_speed_nodes_used_without_holdout_leakage():
    artifact, _ = fixture()
    artifact["models"]["command_route"] = artifact["models"]["measured"]
    physics = Aircraft(artifact, "physical", anchored=False)
    rows = []
    for speed in (0, 3.75, 5, 8.75, 10):
        for run in ("one", "two"):
            rpm = 4300
            w = np.full(4, rpm * 2 * np.pi / 60)
            row = dict(wind_m_s=speed, heading_deg=0, pitch_deg=0, yaw_deg=0,
                       role="H" if speed == 8.75 else "T", run=f"{speed}-{run}",
                       rate_max_deg_s=0, accel_max_deg_s2=0, wind_basis="measured")
            row.update({f"rpm_M{j}": rpm for j in range(1, 5)})
            truth = physics.loads(np.array([speed, 0, 0]), np.zeros(3), w, np.zeros(4))
            row.update({f"target_{out}": truth[i] for i, out in enumerate(OUTPUTS)})
            rows.append(row)
    schedule = fit_schedule(artifact, rows)
    assert schedule["speeds_m_s"] == [0, 3.75, 5, 10]
    excluded = next(n for n in schedule["speed_node_coverage"] if n["speed_m_s"] == 8.75)
    assert not excluded["adopted"]
    assert excluded["reason"] == "NO_T_WINDOWS"
