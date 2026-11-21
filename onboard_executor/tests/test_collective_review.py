"""Compensation JSON rejection and real executable/CSV checks. Synthetic PTY only."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from test_session_transport import check


def main():
    executable = Path(sys.argv[1]).resolve(strict=True)
    source = Path(sys.argv[2]).resolve(strict=True)
    template = json.loads(source.read_text())
    with tempfile.TemporaryDirectory(prefix="fftgyro-compensation-config-") as temp:
        config_file = Path(temp) / "config.json"

        def validates(config):
            config_file.write_text(json.dumps(config))
            result = subprocess.run([str(executable), "session-validate", str(config_file)],
                                    capture_output=True, text=True, timeout=5)
            return result.returncode == 0

        assert validates(template), "review draft does not validate"
        for missing in ("reference_collective", "thrust_curve_linear", "thrust_curve_quadratic",
                        "command_min", "command_max"):
            bad = copy.deepcopy(template)
            del bad["pid"]["cascade"]["collective_gain_compensation"][missing]
            assert not validates(bad), f"enabled model silently defaulted {missing}"
        for key, value in (("command_min", 0), ("command_max", .5), ("reference_collective", .01),
                           ("thrust_curve_quadratic", 1e308)):
            bad = copy.deepcopy(template)
            bad["pid"]["cascade"]["collective_gain_compensation"][key] = value
            assert not validates(bad), f"invalid {key} accepted"
        for gain in (-1, 1.01):
            bad = copy.deepcopy(template)
            bad["pid"]["cascade"]["trajectory_rate_feedforward"] = gain
            assert not validates(bad), "invalid FF gain accepted"
        # Inspect analytic rates written by the production preview path.
        config_file.write_text(json.dumps(template))
        preview = Path(temp) / "preview.csv"
        subprocess.run([str(executable), "session-preview", str(config_file), str(preview)],
                       check=True, capture_output=True, timeout=5)
        import csv
        with preview.open() as stream:
            next(stream)
            rows = list(csv.DictReader(stream))
        move = [r for r in rows if r["segment_id"] == "move-pos30"]
        assert math.isclose(max(float(r["trajectory_rate_deg_s"]) for r in move), 30 / 12 * 1.875)

    # Match the existing PTY fixture's 5-second startup at c=.1, then exercise c=.3,
    # transition FF, MOVE, faded sine excitation and shutdown in session-run.
    template["maximum_duration_s"] = 40
    template["startup"]["minimum_duration_s"] = 5
    template["hardware"]["startup_ramp_rate"] = .04
    template["pid"]["cascade"]["trajectory_rate_feedforward"] = 1
    # This transport fixture reports a fixed angle, not a responding plant. Integral
    # behavior is covered by the C++ closed-loop tests; avoid synthetic windup here.
    template["pid"]["cascade"]["ki"] = 0
    template["segments"][0]["collective"] = .1
    common = {"condition_id": "synthetic-compensation", "collective": .3}
    template["segments"] = [template["segments"][0],
        {**common, "segment_id": "move", "kind": "MOVE", "duration_s": 2, "start_deg": .5, "end_deg": 1},
        {**common, "segment_id": "sine", "kind": "SINE", "duration_s": 4, "start_deg": 1,
         "amplitude_deg": .5, "frequency_hz": .3, "fade_s": 1,
         "external_amplitude": .005, "external_frequency_hz": .3},
        {**common, "segment_id": "final", "hold_id": "final", "kind": "HOLD", "duration_s": 5,
         "start_deg": 0, "require_settled": True}]
    check(executable, template, "normal", preserve_segments=True)
    print("PASS compensation config, preview, transport and CSV invariants")


if __name__ == "__main__":
    main()
