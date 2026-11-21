"""Offline force-delay scenarios: pairing sensitivity, NOT latency calibration.

Positive delay moves force samples earlier on the aircraft Pi clock. All
scenarios use the same force rows within shared PID and four-RPM coverage.
RPM is an interval average between adjacent edges, assuming two pulses/rev.
No latency is selected and no physical force model is fitted.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path


DELAYS_MS = (0, 1, 3, 5, 10)


def preceding(times: list[int], stamp: int) -> int:
    if not times or not times[0] <= stamp < times[-1]:
        raise ValueError("outside common coverage")
    return bisect.bisect_right(times, stamp) - 1


def rpm_at(edges: list[int], stamp: int, pulses_per_rev: int = 2) -> tuple[int, float]:
    if pulses_per_rev <= 0:
        raise ValueError("positive pulses/revolution required")
    i = preceding(edges, stamp)
    dt = edges[i + 1] - edges[i]
    if dt <= 0:
        raise ValueError("nonpositive edge interval")
    return i, 60e9 / (pulses_per_rev * dt)


def delta_stats(values: list[float]) -> dict:
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("empty/nonfinite comparison")
    return {"mean_absolute": sum(abs(v) for v in values) / len(values),
            "rms": math.sqrt(sum(v * v for v in values) / len(values)),
            "maximum_absolute": max(abs(v) for v in values)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def ordered(times: list[int]) -> None:
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("need strictly increasing timestamps")


def analyze(grid_dir: Path, aligned_dir: Path, output: Path) -> dict:
    grid_path = grid_dir / "force_grid.csv"
    grid_meta_path = grid_dir / "summary.json"
    aligned_meta_path = aligned_dir / "summary.json"
    grid_meta = json.loads(grid_meta_path.read_text())
    aligned_meta = json.loads(aligned_meta_path.read_text())
    if grid_meta["state"] not in ("NOMINAL_GRID_ESTIMATED_NOT_CALIBRATED",
                                  "NOMINAL_GRID_ESTIMATED_WITH_VENDOR_DELAY"):
        raise ValueError("incomplete nominal grid")
    if aligned_meta["state"] != "ALIGNED_CONDITIONAL_NOT_QUALIFIED":
        raise ValueError("incomplete alignment")
    if hashlib.sha256(aligned_meta_path.read_bytes()).hexdigest() != grid_meta["source_sha256"]["summary.json"]:
        raise ValueError("grid/align mismatch")
    run = aligned_dir.parent
    manifest_path = run / "run.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != grid_meta["source_manifest_sha256"]:
        raise ValueError("run manifest changed")
    manifest = json.loads(manifest_path.read_text())
    for key in ("aircraft_boot_id", "leptrino_boot_id"):
        if not grid_meta[key] == aligned_meta[key] == manifest[key] == manifest[key + "_after"]:
            raise ValueError("boot mismatch")
    controls_path = run / "pid/control.csv"
    edges_path = aligned_dir / "rpm_edges_aligned.csv"
    force = read_csv(grid_path)
    controls = read_csv(controls_path)
    edges: dict[int, list[int]] = {ch: [] for ch in range(1, 5)}
    for row in read_csv(edges_path):
        if row["within_qt_anchors"] != "True":
            raise ValueError("RPM outside clock anchors")
        stamp = int(row["aircraft_estimate_ns"])
        if not int(row["aircraft_lower_ns"]) <= stamp <= int(row["aircraft_upper_ns"]):
            raise ValueError("invalid RPM clock interval")
        edges[int(row["channel"])].append(stamp)
    times = [int(row["sent_ns"]) for row in controls]
    ordered(times)
    for channel_edges in edges.values():
        ordered(channel_edges)
    force_time_field = ("estimated_measurement_aircraft_ns"
                        if force and "estimated_measurement_aircraft_ns" in force[0]
                        else "nominal_sample_aircraft_estimate_ns")
    ftimes = [int(row[force_time_field]) for row in force]
    ordered(ftimes)
    if len(force) != grid_meta["samples"] or any(int(r["index"]) != i for i, r in enumerate(force)):
        raise ValueError("force count/order mismatch")
    start = max(times[0], *(e[0] for e in edges.values()))
    end = min(times[-1], *(e[-1] for e in edges.values()))
    # No scenario-specific trimming: compare exactly the same force indices.
    common = [r for r, t in zip(force, ftimes)
              if start + max(DELAYS_MS) * 1_000_000 <= t < end]
    if len(common) < 2:
        raise ValueError("insufficient common coverage")
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text('{"state":"INCOMPLETE"}\n')
    scenario_reports = []
    names = [f"m{i}" for i in range(1, 5)]
    force_names = ["fx_N", "fy_N", "fz_N", "mx_Nm", "my_Nm", "mz_Nm"]
    with (output / "pairings.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["assumed_delay_ms", "force_index", "shifted_aircraft_ns",
                         "pid_row", *names, *[f"rpm_s{ch}" for ch in edges], *force_names])
        for delay in DELAYS_MS:
            command_deltas = {name: [] for name in names}
            rpm_deltas = {str(ch): [] for ch in edges}
            changed_pid = 0
            changed_rpm = {str(ch): 0 for ch in edges}
            for row in common:
                baseline = int(row[force_time_field])
                stamp = baseline - delay * 1_000_000
                base_pid, pid = preceding(times, baseline), preceding(times, stamp)
                changed_pid += int(base_pid != pid)
                commands = [float(controls[pid][name]) for name in names]
                for name, value in zip(names, commands):
                    command_deltas[name].append(value - float(controls[base_pid][name]))
                rpms = []
                for ch, channel_edges in edges.items():
                    base_edge, base_rpm = rpm_at(channel_edges, baseline)
                    edge, rpm = rpm_at(channel_edges, stamp)
                    rpms.append(rpm)
                    changed_rpm[str(ch)] += int(edge != base_edge)
                    rpm_deltas[str(ch)].append(rpm - base_rpm)
                writer.writerow([delay, row["index"], stamp, pid, *commands, *rpms,
                                 *(row[name] for name in force_names)])
            scenario_reports.append({
                "assumed_delay_ms": delay, "common_force_samples": len(common),
                "pid_row_changed_percent": 100 * changed_pid / len(common),
                "command_difference_from_zero_delay": {k: delta_stats(v) for k, v in command_deltas.items()},
                "rpm_interval_changed_percent": {k: 100 * v / len(common) for k, v in changed_rpm.items()},
                "rpm_difference_from_zero_delay": {k: delta_stats(v) for k, v in rpm_deltas.items()},
                "illustrative_phase_deg": {str(f): 360 * f * delay / 1000 for f in (.5, 1, 2)},
            })
    report = {
        "state": "PAIRING_SENSITIVITY_ONLY_NOT_IDENTIFICATION_OR_CALIBRATION",
        "delays_ms": DELAYS_MS, "common_force_samples": len(common),
        "total_force_samples": len(force),
        "first_force_index": int(common[0]["index"]), "last_force_index": int(common[-1]["index"]),
        "force_time_field": force_time_field,
        "vendor_sensor_delay_ms": grid_meta.get("vendor_sensor_delay_ns", 0) / 1e6,
        "common_unshifted_span_s": (int(common[-1][force_time_field])
                                    - int(common[0][force_time_field])) / 1e9,
        "pulses_per_revolution_assumed": 2,
        "largest_rpm_interval_ms": {str(ch): max(b-a for a, b in zip(e, e[1:])) / 1e6
                                    for ch, e in edges.items()},
        "scenarios": scenario_reports,
        "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (grid_path, grid_meta_path, aligned_meta_path, manifest_path,
                                    controls_path, edges_path, Path(__file__))},
        "limitations": [
            "Delay scenarios are hypothetical, not a measured interval or upper bound; none selected as best.",
            "This props-off commissioning run cannot establish robustness of aerodynamic identification.",
            "Changes quantify different pairings with identical force rows, not force prediction RMSE.",
            "PID association uses preceding command send, not physical actuator application.",
            "RPM uses bracketing edge interval average, available offline only, without interpolation or extrapolation.",
            "RPM two-pulses/rev assumption and optical edge quality are not independently calibrated.",
            "Clock interval uncertainty and sample-rate error are not propagated; point-estimate sensitivity only.",
            "No source values changed, no correlation maximization, no motor operation.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path)
    parser.add_argument("aligned", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.grid, args.aligned, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2))
