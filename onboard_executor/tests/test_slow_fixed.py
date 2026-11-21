"""Synthetic PTY only: ten-second startup, hold and stop, bounded to 18 s."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from test_session_transport import check

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fftgyro_quad_id.experiment_runner.commissioning import build_commissioning

exe = Path(sys.argv[1]).resolve(strict=True)
config = json.loads(Path(sys.argv[2]).read_text())
if config.get("control_mode") != "FIXED_HOLD":
    config = build_commissioning(config, "fixed")
    config["hardware"]["startup_ramp_rate"] = .01
    config["maximum_duration_s"] = 18
assert config["control_mode"] == "FIXED_HOLD"
assert config["hardware"]["startup_ramp_rate"] == .01
assert config["maximum_duration_s"] == 18
config["pid"]["rate_limit_deg_s"] = 60
config["pid"]["angle_min_deg"] = -20
config["pid"]["angle_max_deg"] = 20
config["pid"]["tracking_limit_deg"] = 20
config["pid"]["swing_limit_deg"] = 20
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / "session.json"
    for rate, duration, valid in (
        (.01, 18, True),
        (.01, 17, False),
        (.02, 13, True),
        (.1, 13, False),
        (.01, 19, False),
    ):
        c = copy.deepcopy(config)
        c["hardware"]["startup_ramp_rate"] = rate
        c["maximum_duration_s"] = duration
        path.write_text(json.dumps(c))
        result = subprocess.run([str(exe), "session-fixed-commissioning-validate", str(path)],
                                capture_output=True, text=True, timeout=5)
        assert (result.returncode == 0) == valid, result.stderr
    for field, value in (("angle_min_deg", -20.01), ("angle_max_deg", 20.01),
                         ("tracking_limit_deg", 20.01), ("swing_limit_deg", 20.01)):
        c = copy.deepcopy(config)
        c["pid"][field] = value
        path.write_text(json.dumps(c))
        result = subprocess.run([str(exe), "session-fixed-commissioning-validate", str(path)],
                                capture_output=True, text=True, timeout=5)
        assert result.returncode != 0, (field, result.stdout)
for scenario in ("normal", "fixed_offset", "swing_limit", "operator_stop", "angle_limit", "rate_limit", "ack_loss", "replayed_ack", "telemetry_loss", "rejected", "release_loss"):
    check(exe, config, scenario, preserve_segments=True, fixed_commissioning=True)
