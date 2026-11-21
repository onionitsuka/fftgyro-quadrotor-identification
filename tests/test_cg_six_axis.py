import numpy as np
import pytest

from fftgyro_quad_id.analysis.cg_six_axis import fit_six_axis, predict_six_axis


def test_shared_bias_two_configurations():
    a = np.deg2rad([0, -15, -30, -45] * 2)
    gravity = 15.6 * np.column_stack((np.sin(a), np.zeros(8), -np.cos(a)))
    nominal = np.array([-0.001, 0, 0.081])
    groups = ["a"] * 4 + ["b"] * 4
    shifts = {"a": np.array([0.002, -0.003, 0]), "b": np.array([-0.001, 0.004, 0])}
    bias = np.array([1.2, -0.8, 4.0, 0.01, -0.04, 0.08])
    wrench = np.array(
        [
            np.r_[f, np.cross(nominal + shifts[g], f)] + bias
            for f, g in zip(gravity, groups, strict=True)
        ]
    )
    fit = fit_six_axis(gravity, wrench, groups, ["shared"] * 8, nominal)
    assert fit["bias_6"]["shared"] == pytest.approx(bias)
    assert fit["delta_xy_mm"]["a"] == pytest.approx([2, -3])
    assert fit["delta_xy_mm"]["b"] == pytest.approx([-1, 4])
    assert predict_six_axis(gravity[0], "a", "shared", fit) == pytest.approx(wrench[0])


def test_force_offset_does_not_silently_change_cg():
    a = np.deg2rad([0, -15, -30, -45])
    f = 15.6 * np.column_stack((np.sin(a), np.zeros(4), -np.cos(a)))
    r = np.array([0.002, 0.001, 0.081])
    w = np.column_stack((f, np.cross(r, f)))
    first = fit_six_axis(f, w, ["g"] * 4, ["e"] * 4, [0, 0, 0.081])
    w[:, :3] += [1, 2, 3]
    second = fit_six_axis(f, w, ["g"] * 4, ["e"] * 4, [0, 0, 0.081])
    assert second["delta_xy_mm"]["g"] == pytest.approx(first["delta_xy_mm"]["g"])
    assert second["bias_6"]["e"][:3] == pytest.approx([1, 2, 3])
