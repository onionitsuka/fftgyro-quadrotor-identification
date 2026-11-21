import copy
import json
from pathlib import Path

import pytest

from fftgyro_quad_id.experiment_runner.axis_session import build_session, expand_condition, requested_rate_cap_mismatch

ROOT = Path(__file__).resolve().parents[2]
# Synthetic waveform cases; no real run numbers or campaign catalog.
DYNAMIC = [{'role': 'T',
  'group': 'N',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 90.0,
  'frequency_hz': 0.1,
  'amplitude_deg': 2,
  'id': 'synthetic-pitch-T-N',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'SIDE',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 36.666667,
  'frequency_hz': 0.3,
  'amplitude_deg': 2,
  'id': 'synthetic-pitch-T-SIDE',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'RPM',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 36.666667,
  'frequency_hz': 0.3,
  'amplitude_deg': 2,
  'rpm_ratio': 0.8,
  'id': 'synthetic-pitch-T-RPM',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'E',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 135,
  'id': 'synthetic-pitch-T-E',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'F1',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 49.428571,
  'id': 'synthetic-pitch-T-F1',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'F2',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 47,
  'amplitudes_deg': [7.5, 10],
  'id': 'synthetic-pitch-T-F2',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'F3',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 108.125,
  'rates_dps': [15, 30, 60],
  'id': 'synthetic-pitch-T-F3',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'N',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 90.0,
  'frequency_hz': 0.1,
  'amplitude_deg': 2,
  'id': 'synthetic-yaw-T-N',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'SIDE',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 36.666667,
  'frequency_hz': 0.3,
  'amplitude_deg': 2,
  'id': 'synthetic-yaw-T-SIDE',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'RPM',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 36.666667,
  'frequency_hz': 0.3,
  'amplitude_deg': 2,
  'rpm_ratio': 0.8,
  'id': 'synthetic-yaw-T-RPM',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'E',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 135,
  'id': 'synthetic-yaw-T-E',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'F1',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 49.428571,
  'id': 'synthetic-yaw-T-F1',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'F2',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 68,
  'amplitudes_deg': [10, 20, 30],
  'id': 'synthetic-yaw-T-F2',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'F3',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 168.125,
  'rates_dps': [15, 30, 60, 90],
  'id': 'synthetic-yaw-T-F3',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'J',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 130,
  'id': 'synthetic-pitch-T-J',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'S',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 209,
  'id': 'synthetic-yaw-T-S',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'T',
  'group': 'W',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 461,
  'id': 'synthetic-yaw-T-W',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'D',
  'group': 'N',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 150,
  'id': 'synthetic-pitch-D-N',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'D',
  'group': 'F',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 150,
  'id': 'synthetic-pitch-D-F',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'D',
  'group': 'N',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 150,
  'id': 'synthetic-yaw-D-N',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'D',
  'group': 'F',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 150,
  'id': 'synthetic-yaw-D-F',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'D',
  'group': 'J',
  'axis': 'pitch',
  'repeats': 2,
  'nominal_seconds': 140,
  'id': 'synthetic-pitch-D-J',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'D',
  'group': 'S',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 131,
  'id': 'synthetic-yaw-D-S',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'D',
  'group': 'W',
  'axis': 'yaw',
  'repeats': 2,
  'nominal_seconds': 495,
  'id': 'synthetic-yaw-D-W',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'H',
  'group': 'N',
  'axis': 'pitch',
  'repeats': 3,
  'nominal_seconds': 150,
  'id': 'synthetic-pitch-H-N',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'H',
  'group': 'F',
  'axis': 'pitch',
  'repeats': 3,
  'nominal_seconds': 150,
  'id': 'synthetic-pitch-H-F',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'H',
  'group': 'N',
  'axis': 'yaw',
  'repeats': 3,
  'nominal_seconds': 150,
  'id': 'synthetic-yaw-H-N',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'H',
  'group': 'F',
  'axis': 'yaw',
  'repeats': 3,
  'nominal_seconds': 150,
  'id': 'synthetic-yaw-H-F',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'H',
  'group': 'J',
  'axis': 'pitch',
  'repeats': 3,
  'nominal_seconds': 150,
  'id': 'synthetic-pitch-H-J',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'H',
  'group': 'S',
  'axis': 'yaw',
  'repeats': 3,
  'nominal_seconds': 117,
  'id': 'synthetic-yaw-H-S',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0},
 {'role': 'H',
  'group': 'W',
  'axis': 'yaw',
  'repeats': 3,
  'nominal_seconds': 478,
  'id': 'synthetic-yaw-H-W',
  'wind_mps': 0,
  'pitch_deg': 0,
  'heading_deg': 0}]


@pytest.mark.parametrize("row", DYNAMIC, ids=lambda r: r["id"])
def test_all_dynamic_conditions_keep_nominal_time_and_identity(row):
    for rep in range(1, row["repeats"] + 1):
        parts = expand_condition(row, 0.2, rep)
        assert sum(p["duration_s"] for p in parts) == pytest.approx(
            row["nominal_seconds"], abs=1e-5
        )
        assert len({p["segment_id"] for p in parts}) == len(parts)
        assert all(p["condition_id"] == row["id"] and p["duration_s"] > 0 for p in parts)
        for p in parts:
            if p["kind"] == "SINE":
                assert p["fade_s"] < p["analysis_start_s"] < p["analysis_end_s"] < p["duration_s"]


def profile(axis):
    result = json.loads((ROOT / "onboard_executor/config/axis-session.example.json").read_text())
    result["pid"]["axis"] = axis.upper()
    result["collective_by_rpm_ratio"] = {"1": 0.2, ".8": 0.15, "0.8": 0.15, "1.2": 0.25}
    result["maximum_duration_s"] = 3600
    return result


def test_role_mechanical_changes_and_repetitions_cannot_be_concatenated():
    a = next(r for r in DYNAMIC if r["axis"] == "yaw" and r["role"] == "T")
    for field, value in [
        ("wind_mps", 9),
        ("heading_deg", 15),
        ("pitch_deg", -30),
        ("role", "H"),
        ("axis", "pitch"),
    ]:
        b = {**a, "id": "different", field: value}
        with pytest.raises(ValueError, match="separate session"):
            build_session([a, b], profile("yaw"), "test", "parent", 1)
    with pytest.raises(ValueError, match="Repetition"):
        build_session([a], profile("yaw"), "test", "parent", 3)


def test_draft_does_not_inherit_authorization_or_guess_rpm_mapping():
    row = next(r for r in DYNAMIC if r["group"] == "RPM" and r["axis"] == "pitch")
    p = profile("pitch")
    p["runtime_qualified"] = True
    out = build_session([row], p, "test", "parent", 1)
    assert not out["runtime_qualified"]
    assert out["segments"][0]["require_settled"]
    assert out["segments"][-1]["require_settled"]
    assert out["source_conditions"] == [row]
    p["collective_by_rpm_ratio"] = {"1": 0.2}
    with pytest.raises(ValueError, match="Missing"):
        build_session([row], p, "test", "parent", 1)


def test_dynamic_session_requires_explicit_evidence_for_promotion():
    row = next(r for r in DYNAMIC if r["axis"] == "pitch" and r["role"] == "T")
    evidence = {"schema": "fftgyro-dynamic-qualification-v1", "status": "QUALIFIED"}
    out = build_session([row], profile("pitch"), "test", "parent", 1,
                        qualification_evidence=evidence)
    assert out["runtime_qualified"] is True
    assert out["qualification_evidence"] == evidence
    assert out["profile_kind"] == "qualified_dynamic"


def test_high_rate_mismatch_is_advisory_and_does_not_change_qualification():
    row = next(r for r in DYNAMIC if r["group"] == "F3" and r["axis"] == "pitch")
    p = profile("pitch")
    p["pid"]["cascade"] = {"target_rate_limit_deg_s": 30}
    out = build_session([row], p, "test-rate", "parent-rate", 1,
                        qualification_evidence={"status": "QUALIFIED"})
    assert out["runtime_qualified"] is True
    assert out["profile_kind"] == "qualified_dynamic"
    assert requested_rate_cap_mismatch(out) == pytest.approx((60, 30))
    assert out["rate_cap_mismatch"]["controller_target_cap_deg_s"] == 30


@pytest.mark.parametrize("role", ["D", "H"])
@pytest.mark.parametrize("axis", ["pitch", "yaw"])
def test_diagonal_dynamic_validation_waveforms_are_draft_and_reversed(role, axis):
    row = {
        "id": f"{role}-SIDE-SUPPLEMENT-{axis}", "role": role, "group": "SIDE",
        "axis": axis, "wind_mps": 5, "pitch_deg": -15 if role == "D" else -30,
        "heading_deg": 15, "repeats": 2, "nominal_seconds": 150,
    }
    first, second = (expand_condition(row, .3, rep) for rep in (1, 2))
    assert sum(p["duration_s"] for p in first) == pytest.approx(150)
    assert sum(p["duration_s"] for p in second) == pytest.approx(150)
    assert [p["frequency_hz"] for p in first if p["kind"] == "SINE"] != [
        p["frequency_hz"] for p in second if p["kind"] == "SINE"
    ]
    for parts in (first, second):
        assert any(p.get("external_amplitude", 0) > 0 for p in parts)
        assert all(p["condition_id"] == row["id"] for p in parts)
    session = build_session([row], profile(axis), "supplement", row["id"], 1)
    assert not session["runtime_qualified"]
    assert session["heading_deg"] == 15
    assert session["source_conditions"] == [row]


def test_fixed_input_cannot_be_misinterpreted_as_angle_pid():
    with pytest.raises(ValueError, match="Fixed input"):
        expand_condition({"axis": "fixed"}, 0.2, 1)


def test_pitch_centers_can_change_but_duration_must_fit():
    row = next(r for r in DYNAMIC if r["axis"] == "pitch")
    other = {**row, "id": "other-center", "pitch_deg": -15}
    p = profile("pitch")
    out = build_session([row, other], p, "test", "parent", 1)
    assert out["nominal_condition_seconds"] == row["nominal_seconds"] * 2
    p["maximum_duration_s"] = 10
    with pytest.raises(ValueError, match="split"):
        build_session([row, other], p, "test", "parent", 1)


def test_second_scan_reverses_direction_without_changing_speed_or_holds():
    for row in [r for r in DYNAMIC if r["group"] in {"S", "W"}]:
        first, second = (expand_condition(row, 0.2, rep) for rep in [1, 2])
        for a, b in zip(first, second, strict=True):
            assert b["duration_s"] == a["duration_s"]
            assert b["start_deg"] == -a["start_deg"]


def test_profile_is_not_mutated():
    p = profile("pitch")
    original = copy.deepcopy(p)
    row = next(r for r in DYNAMIC if r["axis"] == "pitch")
    build_session([row], p, "test", "parent", 1)
    assert p == original


def test_short_preparation_preserves_all_scientific_segments():
    row = next(r for r in DYNAMIC if r["axis"] == "pitch")
    other = {**row, "id": "other-center", "pitch_deg": -15}
    p = profile("pitch")
    original = build_session([row, other], p, "test", "parent", 1)
    p["preparation_hold_s"] = 1
    shorter = build_session([row, other], p, "test", "parent", 1)
    duration_key = "expanded_seconds_excluding_transitions_and_ramps"
    assert original[duration_key] - shorter[duration_key] == 12
    assert shorter["nominal_condition_seconds"] == original["nominal_condition_seconds"]
    for before, after in zip(original["segments"], shorter["segments"], strict=True):
        if after.get("preparation_only"):
            assert before["duration_s"] == 5 and after["duration_s"] == 1
            assert after["require_settled"] and after.get("analysis_end_s", 0) == 0
        else:
            assert before == after


@pytest.mark.parametrize("value", [0, .5, 5.1, float("nan"), float("inf"), True, "1", None])
def test_bad_preparation_time_rejected(value):
    row = next(r for r in DYNAMIC if r["axis"] == "pitch")
    p = profile("pitch")
    p["preparation_hold_s"] = value
    with pytest.raises(ValueError, match="preparation_hold_s"):
        build_session([row], p, "test", "parent", 1)
