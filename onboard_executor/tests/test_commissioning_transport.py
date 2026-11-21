"""Synthetic PTY tests only. No hardware ports or real qualification writes."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fftgyro_quad_id.experiment_runner.commissioning import build_commissioning
from test_session_transport import check


def main():
    executable = Path(sys.argv[1]).resolve(strict=True)
    template = json.loads(Path(sys.argv[2]).read_text())
    # Yaw mapping here is synthetic only, not an assertion about the actual motors.
    profiles = []
    for axis, kind in (("PITCH", "fixed"), ("PITCH", "sine"), ("PITCH", "move"),
                       ("YAW", "sine"), ("YAW", "scan")):
        source = copy.deepcopy(template)
        source["pid"]["axis"] = axis
        if axis == "YAW":
            source["hardware"]["axis_mix"] = [1, -1, 1, -1]  # Synthetic only; retain existing Yaw fixture.
        profile = build_commissioning(source, kind)
        profiles.append(profile)
        print(f"SYNTHETIC {axis} {kind}", flush=True)
        check(executable, profile, "normal", preserve_segments=True)
    for profile in (profiles[0], profiles[1], profiles[-1]):
        for mode in ("ack_loss", "replayed_ack", "telemetry_loss", "rejected", "release_loss"):
            print(f"SYNTHETIC {profile['session_id']} {mode}", flush=True)
            check(executable, profile, mode, preserve_segments=True)
    with tempfile.TemporaryDirectory(prefix="commissioning-guards-") as directory:
        path = Path(directory) / "invalid.json"
        for field, value in (("kp", 0.001), ("ki", 0.001), ("kd", 0.001)):
            bad = copy.deepcopy(profiles[0])
            bad["pid"][field] = value
            path.write_text(json.dumps(bad))
            result = subprocess.run([str(executable), "session-validate", str(path)],
                                    capture_output=True, text=True, timeout=5)
            assert result.returncode != 0 and "zero PID gains" in result.stderr
        for profile in profiles:
            # Unqualified normal entry must fail before creating any logs/serial access.
            profile["hardware"]["uart_device"] = "/dev/pts/999999"
            profile["hardware"]["log_directory"] = str(Path(directory) / "no-logs")
            path.write_text(json.dumps(profile))
            result = subprocess.run([str(executable), "session-run", str(path)],
                                    capture_output=True, text=True, timeout=5)
            assert result.returncode != 0 and "runtime_qualified=false" in result.stderr
            assert not (Path(directory) / "no-logs").exists()
    print("PASS commissioning transport and draft guards", flush=True)


if __name__ == "__main__":
    main()
