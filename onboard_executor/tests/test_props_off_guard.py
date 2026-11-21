"""Read-only validation of props-off gates. Never runs a valid motor command."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    exe = str(Path(sys.argv[1]).resolve(strict=True))
    template = json.loads(Path(sys.argv[2]).read_text())
    fixed = "--fixed-commissioning" in sys.argv[3:]
    pitch = "--pitch-commissioning" in sys.argv[3:]
    command = "session-pitch-hold-commissioning" if pitch else "session-fixed-commissioning" if fixed else "session-props-off"
    flags = ["--props-installed" if fixed or pitch else "--props-removed",
             "--pitch-only-free" if pitch else "--fixed", "--operator-stop-ready", "--charged-4s"]
    changes = [
        ("hardware", "axis_mix", [1, -1, -1, 1]),  # Previous erroneous left/right mix.
        ("hardware", "motor_max", 0.800001 if pitch else 0.13), ("hardware", "motor_min", 0.01),
        ("hardware", "startup_ramp_rate", 0.2), ("hardware", "abort_ramp_rate", 0.3),
        ("hardware", "axis_mix", [1, 1, -1, -1]), ("hardware", "uart_device", "/dev/ttyACM0"),
        ("pid", "additional_limits_enabled", False), ("pid", "kp", 0.007001 if pitch else 0.003),
        ("pid", "ki", 0.001001 if pitch else 0.001), ("pid", "kd", 0.001701 if pitch else 0.002),
        ("pid", "differential_limit", 0.3001 if pitch else 0.02), ("pid", "integral_limit", 0.100001 if pitch else 0.005),
        ("pid", "motor_slew_time_s", 0.1 if pitch else 0),
        ("pid", "motor_slew_time_s", -0.1), ("pid", "rate_limit_deg_s", 61 if fixed or pitch else 21),
        ("pid", "angle_min_deg", -20.01 if fixed else -90.01 if pitch else -10), ("pid", "feedback_timeout_s", 0.15),
        ("pid", "saturation_timeout_s", 3), ("settling", "maximum_hold_s", 15),
        ("settling", "dwell_s", 0.1), ("shutdown", "minimum_duration_s", 1),
        (None, "maximum_duration_s", 20), (None, "wind_m_s", 1),
        ("segment", "collective", 0.11), ("segment", "start_deg", 1),
        ("segment", "duration_s", 6), ("segment", "kind", "MOVE"),
    ]
    if pitch:
        changes += [("pid", "ki", .0010001), ("pid", "ki", -.0001),
                    ("pid", "kd", 0), ("pid", "differential_limit", .300001),
                    ("pid", "angle_max_deg", 90.01), ("pid", "rate_limit_deg_s", 60.001),
                    ("pid", "tracking_limit_deg", 90.01), ("pid", "swing_limit_deg", 90.01),
                    ("settling", "error_deg", 5.01), ("settling", "rate_deg_s", 5.01),
                    ("hardware", "motor_mapping_verified", False), ("hardware", "feedback_frame_verified", False),
                    ("hardware", "mechanically_locked_verified", True), ("startup", "control_ready_verified", True),
                    ("hardware", "non_target_pid_enabled", True), (None, "control_mode", "FIXED_HOLD")]
        changes += [("hardware", "target_system", 2), ("hardware", "target_component", 2),
                    ("segment", "collective", 0)]
    with tempfile.TemporaryDirectory(prefix="props-off-gates-") as temp:
        root = Path(temp)
        config = copy.deepcopy(template)
        config["hardware"]["uart_device"] = "/dev/pts/999999"
        config["hardware"]["log_directory"] = str(root / "logs")
        path = root / "session.json"

        def call(data, command, extra=()):
            path.write_text(json.dumps(data))
            return subprocess.run([exe, command, str(path), *extra], capture_output=True, text=True, timeout=5)

        assert call(config, command + "-validate").returncode == 0
        if pitch:
            for delta_limit in (0.005, 0.3):
                allowed = copy.deepcopy(config)
                allowed["pid"]["differential_limit"] = delta_limit
                allowed["pid"]["integral_limit"] = min(allowed["pid"]["integral_limit"], delta_limit)
                assert call(allowed, command + "-validate").returncode == 0
            for ki in (0, 0.0001, 0.001):
                allowed = copy.deepcopy(config)
                allowed["pid"]["ki"] = ki
                assert call(allowed, command + "-validate").returncode == 0
            for slew in (0, 0.2):
                allowed = copy.deepcopy(config)
                allowed["pid"]["motor_slew_time_s"] = slew
                assert call(allowed, command + "-validate").returncode == 0
        boundary = copy.deepcopy(config)
        boundary["pid"]["rate_limit_deg_s"] = 60 if fixed or pitch else 20
        assert call(boundary, command + "-validate").returncode == 0
        assert call(config, "session-run").returncode != 0
        assert not (root / "logs").exists()
        for missing in range(4):
            assert call(config, command, flags[:missing] + flags[missing+1:]).returncode != 0
        for group, key, value in changes:
            bad = copy.deepcopy(config)
            destination = bad if group is None else bad["segments"][0] if group == "segment" else bad[group]
            destination[key] = value
            result = call(bad, command, flags)
            assert result.returncode != 0, (group, key, result.stdout)
            assert not (root / "logs").exists(), "invalid config reached device initialization"
        print(f"PASS {command} guards: valid preview, normal-run refusal, 4 missing flags, {len(changes)} rejected profiles")


if __name__ == "__main__":
    main()
