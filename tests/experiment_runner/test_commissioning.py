import copy
import json
from pathlib import Path

import pytest

from fftgyro_quad_id.experiment_runner.commissioning import build_commissioning

PROFILE = json.loads((Path(__file__).resolve().parents[2] /
                      "onboard_executor/config/props-off-hold.example.json").read_text())


@pytest.mark.parametrize("axis,kind", [(a, k) for a in ("PITCH", "YAW")
                                      for k in ("fixed", "hold", "sine", "move", "scan")
                                      if a == "YAW" or k != "scan"])
def test_profiles(axis, kind):
    source = copy.deepcopy(PROFILE)
    source["pid"]["axis"] = axis
    source["runtime_qualified"] = True
    source["hardware"]["stop_verified"] = True
    original = copy.deepcopy(source)
    c = build_commissioning(source, kind)
    assert source == original
    assert c["runtime_qualified"] is False and c["role"] == "PILOT"
    assert not any(v for k, v in c["hardware"].items() if k.endswith("_verified"))
    assert c["hardware"]["axis_mix"] == source["hardware"]["axis_mix"]
    assert c["pid"]["axis"] == axis
    assert c["segments"][0]["require_settled"] and c["segments"][-1]["require_settled"]
    assert len({s["segment_id"] for s in c["segments"]}) == len(c["segments"])
    assert all(s["collective"] <= 0.1 for s in c["segments"])
    if kind == "fixed":
        assert c["control_mode"] == "FIXED_HOLD"
        assert all(c["pid"][key] == 0 for key in ("kp", "ki", "kd"))
    else:
        assert c["control_mode"] == "AXIS_PID"
        assert all(c["pid"][key] == source["pid"][key] for key in ("kp", "ki", "kd"))


def test_no_axis_guess_or_out_of_bounds_motion():
    with pytest.raises(ValueError):
        build_commissioning(PROFILE, "scan")
    narrow = copy.deepcopy(PROFILE)
    narrow["pid"]["angle_max_deg"] = 0.1
    with pytest.raises(ValueError, match="envelope"):
        build_commissioning(narrow, "sine")
