"""Synthetic PTY only. No hardware ports or SSH; no actual qualification."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from test_session_transport import check


def main():
    executable = Path(sys.argv[1]).resolve(strict=True)
    template = json.loads(Path(sys.argv[2]).read_text())
    # These two assertions describe the synthetic apparatus ONLY.
    template["hardware"]["motor_mapping_verified"] = True
    template["hardware"]["feedback_frame_verified"] = True
    with tempfile.TemporaryDirectory(prefix="pitch-hold-gates-") as directory:
        config = Path(directory) / "synthetic.json"
        config.write_text(json.dumps(template))
        subprocess.run([sys.executable, str(Path(__file__).with_name("test_props_off_guard.py")),
                        str(executable), str(config), "--pitch-commissioning"], check=True, timeout=45)
    for mode in ("normal", "pd_response", "relaxed_settling", "nonidentity_reference", "operator_stop", "bad_reference",
                 "micro_vibration_preflight",
                 "moving_preflight", "unsettled", "angle_limit", "swing_limit", "rate_limit", "px4_reboot",
                 "ack_loss", "replayed_ack", "telemetry_loss", "rejected", "release_loss"):
        check(executable, template, mode, pitch_commissioning=True)


if __name__ == "__main__":
    main()
