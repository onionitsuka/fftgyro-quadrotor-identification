"""Offline nominal-rate force time grid with documented sensor-delay correction.

The latest causal grid uses min(receive_i - nominal_elapsed_i) as its origin.
Its causal consistency on fitted data is a construction, not accuracy validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .pi_affine import fit, fit_piecewise_midpoint, fit_smooth_interval, read_exchanges

LEPTRINO_DELAY_NS_1200_HZ = 4_130_000


def infer_grid(received: list[int], rate_hz: int = 1200) -> tuple[list[int], dict]:
    if not isinstance(rate_hz, int) or isinstance(rate_hz, bool) or not 1 <= rate_hz <= 1_000_000:
        raise ValueError("rate must be a positive integer <= 1000000 Hz")
    if len(received) < 2 or any(t < 0 for t in received):
        raise ValueError("need at least two nonnegative timestamps")
    if any(b < a for a, b in zip(received, received[1:], strict=False)):
        raise ValueError("receive clock reversed")
    # Integer arithmetic avoids losing sub-microsecond precision at large uptime.
    elapsed = [(i * 1_000_000_000 + rate_hz // 2) // rate_hz for i in range(len(received))]
    residual = [t - dt for t, dt in zip(received, elapsed, strict=False)]
    origin = min(residual)
    if origin < 0:
        raise ValueError("nominal grid would precede boot")
    grid = [origin + dt for dt in elapsed]
    advances = sorted(t - g for t, g in zip(received, grid, strict=False))
    blocks: dict[int, int] = {}
    for i, r in enumerate(residual):
        block = i // (rate_hz * 10)
        blocks[block] = min(blocks.get(block, r), r)
    # Hold out whole alternating nominal one-second blocks, never individual
    # rows from the same USB batch as the sole validation strategy.
    # This checks offset stability only, not physical latency or sample loss.
    train = [r for i, r in enumerate(residual) if (i // rate_hz) % 2 == 0]
    held = [r for i, r in enumerate(residual) if (i // rate_hz) % 2 == 1]
    trained_origin = min(train)
    violations = [trained_origin - r for r in held if r < trained_origin]
    details = {
        "rate_hz_assumed": rate_hz,
        "samples": len(received),
        "origin_leptrino_ns": origin,
        "anchor_index": residual.index(origin),
        "nominal_span_s": elapsed[-1] / 1e9,
        "advance_from_receive_ms": {
            "min": advances[0] / 1e6,
            "median": advances[len(advances) // 2] / 1e6,
            "p95": advances[(95 * (len(advances) - 1)) // 100] / 1e6,
            "max": advances[-1] / 1e6,
        },
        "ten_second_origin_above_global_ms": {
            str(k): (v - origin) / 1e6 for k, v in blocks.items()
        },
        "alternating_second_holdout": {
            "samples": len(held),
            "causal_violations": len(violations),
            "maximum_excess_ms": max(violations, default=0) / 1e6,
            "training_origin_above_global_ms": (trained_origin - origin) / 1e6,
        },
    }
    return grid, details


def reconstruct(aligned: Path, output: Path, *, run_root: Path | None = None) -> dict:
    """Use the previously checked alignment and the same run's Pi exchanges."""
    source = aligned / "force_aligned.csv"
    meta_path = aligned / "summary.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["state"] != "ALIGNED_CONDITIONAL_NOT_QUALIFIED":
        raise ValueError("requires a validated conditional alignment")
    run = aligned.parent if run_root is None else run_root
    manifest_path = run / "run.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != meta["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("leptrino_boot_id", "aircraft_boot_id"):
        after = manifest.get(key + "_after")
        accepted_missing_after = meta.get("adoption_override", False) and after is None
        if not meta[key] == manifest[key] or (
            after != manifest[key] and not accepted_missing_after
        ):
            raise ValueError("boot mismatch")
    clock_path = run / "clock/exchanges.csv"
    expected_model = meta["pi_model"]
    exchanges = read_exchanges(clock_path)
    method = expected_model.get("method", "affine")
    pi = (
        fit(exchanges)
        if method == "affine"
        else fit_smooth_interval(exchanges)
        if method == "smooth_interval"
        else fit_piecewise_midpoint(exchanges)
        if method == "piecewise_midpoint"
        else None
    )
    if pi is None:
        raise ValueError(f"unsupported Pi clock model: {method}")
    if expected_model.get("method", "affine") == "affine":
        if (
            pi.client_origin_ns != expected_model["client_origin_ns"]
            or pi.server_origin_ns != expected_model["server_origin_ns"]
            or [list(v) for v in pi.vertices] != expected_model["vertices"]
        ):
            raise ValueError("clock data no longer matches the alignment")
    else:
        anchors = pi.stamps if method == "smooth_interval" else pi.client_ns
        if len(anchors) != expected_model["anchors"]:
            raise ValueError("non-affine clock data no longer matches the alignment")
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if len(rows) != meta["force_samples"]:
        raise ValueError("sample count mismatch")
    previous_counter = None
    for i, row in enumerate(rows):
        if int(row["index"]) != i:
            raise ValueError("missing/reordered sample index")
        payload = bytes.fromhex(row["payload_hex"])
        if len(payload) != 20 or payload[0] != 20 or payload[2:4] != b"\x32\x00":
            raise ValueError("invalid force payload")
        if previous_counter is not None and (payload[19] - previous_counter) % 256 != 1:
            raise ValueError("candidate sequence gap; cannot assume contiguous samples")
        previous_counter = payload[19]
    received = [int(row["receive_monotonic_ns"]) for row in rows]
    grid, details = infer_grid(received)
    mapped = [pi.map(t) for t in grid]  # refuses extrapolation
    measurement = [t - LEPTRINO_DELAY_NS_1200_HZ for t in grid]
    if measurement[0] < 0:
        raise ValueError("delay-corrected grid would precede boot")
    measurement_mapped = [pi.map(t) for t in measurement]
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary_path.write_text('{"state":"INCOMPLETE"}\n', encoding="utf-8")
    added = [
        "nominal_sample_leptrino_ns",
        "nominal_sample_aircraft_estimate_ns",
        "nominal_sample_aircraft_clock_lower_ns",
        "nominal_sample_aircraft_clock_upper_ns",
        "advance_from_receive_ns",
        "estimated_measurement_leptrino_ns",
        "estimated_measurement_aircraft_ns",
        "estimated_measurement_aircraft_clock_lower_ns",
        "estimated_measurement_aircraft_clock_upper_ns",
    ]
    if any(name in fields for name in added):
        raise ValueError("already reconstructed")
    with (output / "force_grid.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*fields, *added])
        for row, stamp, mapping, receive, measured, measured_mapping in zip(
            rows, grid, mapped, received, measurement, measurement_mapped, strict=False
        ):
            writer.writerow(
                [
                    *(row[k] for k in fields),
                    stamp,
                    *mapping,
                    receive - stamp,
                    measured,
                    *measured_mapping,
                ]
            )
    details.update(
        state="NOMINAL_GRID_ESTIMATED_WITH_VENDOR_DELAY",
        vendor_sensor_delay_ns=LEPTRINO_DELAY_NS_1200_HZ,
        vendor_sensor_delay_basis=(
            "Leptrino technical support reply 2026-09-12: "
            "3.3 ms AD settling + one 1200 Hz sample (0.83 ms)"
        ),
        source_sha256={
            str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source, meta_path, clock_path)
        },
        source_manifest_sha256=meta["source_manifest_sha256"],
        source_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        leptrino_boot_id=meta["leptrino_boot_id"],
        aircraft_boot_id=meta["aircraft_boot_id"],
        limitations=[
            "Assumes exactly 1200 Hz relative to Leptrino Pi monotonic time; "
            "oscillator drift uncalibrated.",
            "The 4.13 ms correction ends at sensor output; USB/driver/host-read "
            "latency remains uncalibrated.",
            "The output grid is the latest causal nominal grid, not a directly "
            "observed sensor timestamp.",
            "Full-data causality is guaranteed by construction, not independent validation.",
            "Candidate byte19 continuity is checked, not a verified hardware counter; "
            "256-sample losses may evade it.",
            "Clock lower/upper columns exclude sample-grid error and unknown latency.",
            "Original PID association columns still refer to USB receive time, NOT the new grid.",
            "No force values, original times, RPM data, or original failure states changed.",
        ],
    )
    summary_path.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    return details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aligned", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.aligned, args.output), indent=2))
