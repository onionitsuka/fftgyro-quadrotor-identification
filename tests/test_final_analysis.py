"""Frozen final analysis: scientific invariants independent of private logs."""

import ast
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a


def test_single_file_has_no_legacy_analysis_imports():
    tree = ast.parse(Path(a.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("scripts", "graybox", "yaw_", "refit_"))


def test_no_stop_anchor_does_not_manufacture_zero():
    assert a.anchor_mean([], np.array([1.0])) is None
    anchors = [
        dict(center_ns=60e9, bias_sensor_FLU=[1.0] * 6),
        dict(center_ns=180e9, bias_sensor_FLU=[3.0] * 6),
    ]
    np.testing.assert_allclose(
        a.anchor_mean(anchors, np.array([0.0, 2.0, 4.0])), [[1.0] * 6, [2.0] * 6, [3.0] * 6]
    )


def test_unanchored_run_offset_is_removed_from_regression_not_predictions():
    x = np.linspace(-1, 1, 60)
    runs = [
        a.RunBlock("a", "T", {"x": x}, 3 * x + 1000, False, 1.0),
        a.RunBlock("b", "T", {"x": x}, 3 * x - 400, False, 1.0),
    ]
    model = a.fit_regularized(a.assemble_blocks(runs, ["x"]))
    np.testing.assert_allclose(model["coefficients"], [3.0], atol=1e-11)
    assert not model["absolute_prediction_supported"]
    np.testing.assert_allclose(a.predict_correction(model, {"x": x}), 3 * x, atol=1e-11)


def test_sensor_separation_preserves_mass_first_moment_and_inertia():
    cfg = a.Settings()
    m = 0.072
    r = np.array([0.001, -0.002, 0.01])
    sensor = dict(mass=m, h=m * r, inertia=np.diag([2e-5, 3e-5, 4e-5]) + a.parallel_axis(m, r))
    body, inertia = a.split_calibration(cfg, sensor)
    old = np.array(cfg.cg_flu_m) * a.FLU_TO_FRD
    new = np.array(body.cg_flu_m) * a.FLU_TO_FRD
    np.testing.assert_allclose(body.mass_kg * new + sensor["h"], cfg.mass_kg * old)
    np.testing.assert_allclose(
        inertia + a.parallel_axis(body.mass_kg, new) + sensor["inertia"],
        np.diag(cfg.inertia_kg_m2) + a.parallel_axis(cfg.mass_kg, old),
    )


def test_self_load_includes_tangential_and_centripetal_acceleration():
    m = 0.072
    r = np.array([0.001, 0.002, 0.01])
    w = np.array([[1.0, 2.0, 3.0]])
    alpha = np.array([[0.2, 0.4, 0.3]])
    gravity = np.array([[0.0, 0.0, 9.8]])
    a0 = np.array([[1.0, 0.1, 0.3]])
    sensor = dict(mass=m, h=m * r, inertia=a.parallel_axis(m, r))
    force = m * (gravity - a0 - np.cross(alpha, r) - np.cross(w, np.cross(w, r)))
    np.testing.assert_allclose(
        a.self_wrench(gravity, a0, w, alpha, sensor), np.c_[force, np.cross(r, force)]
    )


def test_predicted_rpm_uses_only_command_voltage_and_own_initial_map():
    cfg = a.Settings()
    u = np.full((300, 4), 0.3)
    motors = [
        dict(
            map_coefficients=[10000.0, 0.0, 0.0],
            voltage_exponent=1.0,
            delay_s=0.05,
            tau_up_s=0.1,
            tau_down_s=0.2,
        )
        for _ in range(4)
    ]
    rpm, _ = a.predict_rpm(u, np.full(300, 16.0), motors, cfg)
    low, _ = a.predict_rpm(u, np.full(300, 12.0), motors, cfg)
    np.testing.assert_allclose(rpm, 3000.0)
    np.testing.assert_allclose(low, 2250.0)


def test_interpolation_never_crosses_power_clock_or_extrapolates():
    target = dict(run="b", key=["power1", "clock1"], condition_id="b", minute=2.0)
    left = dict(run="a", key=target["key"], condition_id="a", minute=1.0)
    right = dict(run="c", key=target["key"], condition_id="c", minute=3.0)
    assert a.bracket(target, [left])["status"] == "one_sided"
    assert a.bracket(target, [left, dict(right, key=["power2", "clock1"])])["status"] == "one_sided"
    plan = a.bracket(target, [left, right])
    assert plan["status"] == "interpolated"
    np.testing.assert_allclose(plan["weights"], [0.5, 0.5])
    with pytest.raises(ValueError):
        a.blend([1.0, 2.0], [-0.1, 1.1])


def test_same_condition_is_never_both_training_and_validation():
    runs = [dict(run="a", condition_id="one"), dict(run="b", condition_id="one")]
    with pytest.raises(ValueError):
        a.check_fold({"training_runs": ["a"]}, runs[1], "grouped_OOF", runs)


def test_reference_average_after_nonlinear_features():
    x = np.array([1.0, 3.0])
    values = a.evaluate_names({"mu_x": x}, np.ones(2), ["mu_x^2"])
    assert values["mu_x^2"].mean() == 5.0
    assert values["mu_x^2"].mean() != x.mean() ** 2


def test_error_does_not_remove_nonfinite_or_subtract_mean():
    m = a.error_metrics(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert m["rmse"] == 1.0 and m["bias"] == 1.0 and m["centered_rmse"] == 0.0
    with pytest.raises(ValueError):
        a.error_metrics(np.array([1.0, 2.0]), np.array([1.0, np.nan]))


def test_yaw_spline_is_continuous_and_partition_of_unity():
    x = np.linspace(-np.pi / 2, np.pi / 2, 301)
    w = a.spline_weights(x)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(w, w[::-1], atol=1e-14)


def test_sensor_action_reaction_shift_has_correct_sign():
    wrench = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    r = np.array([0.01, 0.02, 0.03])
    shifted = a.shift_wrench(wrench, r)
    np.testing.assert_allclose(shifted[:, :3], wrench[:, :3])
    np.testing.assert_allclose(shifted[:, 3:], wrench[:, 3:] - np.cross(r, wrench[:, :3]))


def test_saved_foundation_loss_preserves_condition_equal_weights():
    runs = []
    for name, condition in (("a", "same"), ("b", "same"), ("c", "different")):
        d = dict(features={"compact": {"x": np.array([1.0, 2.0])}}, physical=np.zeros(2))
        ch = dict(measured=d, take=np.ones(2, bool), truth=np.array([2.0, 4.0]))
        runs.append(dict(run=name, condition_id=condition, anchored=True, channels={"Mz": ch}))
    template = dict(
        training_runs=[r["run"] for r in runs],
        feature_names=["x"],
        spec="compact",
        loss="condition_equal",
    )
    x, y, _, audit = a.assemble_foundation(runs, {}, "measured", "Mz", template)
    assert audit[0]["weight"] == pytest.approx(0.5)
    assert audit[2]["weight"] == pytest.approx(1 / np.sqrt(2))
    old = dict(template, foundation_loss=template["loss"])
    x2, y2, _, _ = a.assemble_foundation(runs, {}, "measured", "Mz", old)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)


def test_shared_measured_anchor_window_is_not_overwritten_by_command_route():
    measured = dict(run="a", route="measured", anchor_state={"start": 1.0, "end": 2.0}, donors=[])
    command = dict(
        run="a", route="command_route", anchor_state={"start": 3.0, "end": 4.0}, donors=[]
    )
    for entries in ([measured, command], [command, measured]):
        _, windows = a.transfer_windows(entries)
        assert windows["a"] == measured["anchor_state"]
