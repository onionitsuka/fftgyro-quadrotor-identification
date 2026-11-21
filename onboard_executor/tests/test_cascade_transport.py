"""Cascade stage timing and failure tests: synthetic PTY only, never hardware."""
import json
from pathlib import Path
import sys

from test_session_transport import check


def main():
    executable = Path(sys.argv[1]).resolve(strict=True)
    template = json.loads(Path(sys.argv[2]).read_text())
    for mode in ("normal", "pd_response", "relaxed_settling", "nonidentity_reference", "bad_reference",
                 "micro_vibration_preflight",
                 "moving_preflight", "unsettled", "angle_limit", "swing_limit", "rate_limit",
                 "operator_stop", "px4_reboot", "ack_loss", "replayed_ack", "telemetry_loss",
                 "rejected", "release_loss"):
        check(executable, template, mode, preserve_segments=True, pitch_commissioning=True)
    # Qualification for normal session-run is confined to the temporary PTY fixture in check().
    template["maximum_duration_s"] = 30
    prepare = template["segments"][0]
    template["segments"] += [
        {**prepare, "segment_id": "scientific-hold", "hold_id": "scientific-hold",
         "duration_s": 5, "preparation_only": False, "analysis_start_s": .25, "analysis_end_s": 4.75},
        {**prepare, "segment_id": "prepare-02", "hold_id": "prepare-02"},
    ]
    check(executable, template, "pd_response", preserve_segments=True)


if __name__ == "__main__":
    main()
