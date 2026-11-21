import math

import pytest

from fftgyro_quad_id.experiment_runner.fixed_session import build_fixed_session, expand_fixed_condition


@pytest.fixture
def profile():
    return {
        "collective_by_rpm_ratio": {"0.8": .24, "0.9": .27, "0.95": .285, "1": .30,
                                    "1.05": .315, "1.1": .33, "1.2": .36},
        "fixed_amplitudes": {"small": .005, "medium": .01, "large": .02,
                             "combined": .01, "diagnostic": .0125,
                             "diagnostic_combined": .00625, "c_small": .005},
    }


def row(group, seconds, **extra):
    return {"id": f"T-{group}-001", "role": "T", "group": group, "axis": "fixed",
            "wind_mps": 0, "pitch_deg": 0, "heading_deg": 0, "repeats": 2,
            "nominal_seconds": seconds, **extra}


@pytest.mark.parametrize("group,seconds,holds", [("B", 101, 17), ("G", 155, 26),
                                                   ("A", 149, 25), ("M", 149, 25)])
def test_fixed_hold_groups_match_prescribed_wall_time(profile, group, seconds, holds):
    parts = expand_fixed_condition(row(group, seconds), profile, 1)
    transitions = sum(a["fixed_input"] != b["fixed_input"] for a, b in zip(parts, parts[1:]))
    assert len(parts) == holds
    assert sum(p["duration_s"] for p in parts) + transitions == seconds
    assert parts[0]["fixed_input"][1:] == [0, 0, 0]
    assert parts[-1]["fixed_input"][1:] == [0, 0, 0]


@pytest.mark.parametrize("group,seconds", [("X", 129), ("FX", sum(6/f for f in [.7,.8,1,1.5,2])+30)])
def test_fixed_frequency_groups_match_plan(profile, group, seconds):
    parts = expand_fixed_condition(row(group, seconds, mode="P"), profile, 1)
    transitions = sum(a["fixed_input"] != b["fixed_input"] for a, b in zip(parts, parts[1:]))
    assert math.isclose(sum(p["duration_s"] for p in parts) + transitions, seconds)
    assert all(p["fixed_amplitude"][1] == .005 for p in parts if p["kind"] == "SINE")


def test_formal_d_fixed_has_25_holds(profile):
    r = row("FIX", 149)
    r.update(id="D-FIX-001", role="D")
    parts = expand_fixed_condition(r, profile, 1)
    assert len(parts) == 25
    assert sum(p["duration_s"] for p in parts) + 24 == 149


def test_repeated_fixed_sessions_remain_unqualified(profile):
    for group, seconds in [("B", 101), ("G", 155), ("A", 149), ("M", 149)]:
        for repetition in (1, 2):
            session = build_fixed_session(row(group, seconds), profile, repetition)
            assert session["runtime_qualified"] is False



def test_fixed_session_requires_explicit_evidence_for_promotion(profile):
    condition = row("B", 101)
    evidence = {"schema": "fftgyro-fixed-input-qualification-v1", "status": "QUALIFIED"}
    session = build_fixed_session(condition, profile, 1, qualification_evidence=evidence)
    assert session["runtime_qualified"] is True
    assert session["hardware"]["fixed_input_mapping_verified"] is True
    assert session["qualification_evidence"] == evidence
    assert session["profile_kind"] == "qualified_fixed_input"


def test_fixed_session_captures_locked_pose_as_its_attitude_reference(profile):
    profile["startup"] = {"capture_reference": False}
    session = build_fixed_session(row("B", 101), profile, 1)
    assert session["startup"]["capture_reference"] is True
