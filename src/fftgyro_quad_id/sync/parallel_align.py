"""Join force USB receive times and Nano edge times on the aircraft Pi clock.

Preserves clock intervals and original data; does not equate USB arrival with
force sampling or command send with actuator application. Rejects wrong boots.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import itertools
import json
from pathlib import Path

from ..io.leptrino_raw import decode
from .pi_affine import (
    fit,
    read_exchanges,
)
from .pi_affine import (
    fit_piecewise_midpoint as fit_piecewise_pi,
)
from .pi_affine import (
    fit_smooth_interval as fit_smooth_pi,
)
from .rpm_clock_compare import (
    compare,
    fit_local,
    fit_piecewise_midpoint,
    fit_smooth_interval,
)
from .rpm_pid import fit_clock, read_capture


def match_pid(
    times: list[int], estimate: int, lower: int, upper: int
) -> tuple[int | str, bool | str]:
    """Report interval ambiguity separately from the estimate's preceding row."""
    if not lower <= estimate <= upper:
        raise ValueError("clock estimate outside bounds")
    if not times or not times[0] <= estimate <= times[-1]:
        return "", ""
    index = bisect.bisect_right(times, estimate) - 1
    uncertain = (
        lower < times[0]
        or upper > times[-1]
        or bisect.bisect_right(times, lower) != bisect.bisect_right(times, upper)
    )
    return index, uncertain


def align_run(
    root: Path,
    output: Path,
    *,
    rpm_model: str = "local",
    recover_postprocessing: bool = False,
    accept_incomplete: bool = False,
    pi_model: str = "affine",
) -> dict:
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    clock = json.loads((root / "clock/summary.json").read_text(encoding="utf-8"))
    force = json.loads((root / "leptrino/summary.json").read_text(encoding="utf-8"))
    recovered = False
    if recover_postprocessing:
        # One narrow recovery path: 20260909 coordinator stopped post-decode and
        # the still-running clock probe. Never change original failure records.
        if (
            manifest["state"] != "INCOMPLETE"
            or manifest.get("process_returncodes") != [0, 1, 0]
            or force.get("error") != "KeyboardInterrupt: signal 15"
            or not force.get("stop_acknowledged")
            or "raw_writer_peak_queue" not in force
            or "storage_error" in force
            or clock["state"] != "INCOMPLETE"
        ):
            raise ValueError("not the recognized post-processing interruption")
        if not (root / "clock/exchanges.csv").read_bytes().endswith(b"\n"):
            raise ValueError("truncated clock row")
        recovered = True
    accepted_incomplete = accept_incomplete and manifest["state"] == "INCOMPLETE"
    if manifest["state"] != "CAPTURED" and not recovered and not accepted_incomplete:
        raise ValueError("run not captured successfully")
    if not recovered and (
        clock["state"] != "COMPLETE" or force["state"] not in ("COMPLETE", "RAW_COMPLETE")
    ):
        raise ValueError("incomplete clock or force acquisition")
    if (
        clock["client_boot_id"] != force["boot_id"]
        or clock["client_boot_id"] != manifest["leptrino_boot_id"]
        or (not recovered and clock["server_boot_id"] != manifest["aircraft_boot_id"])
    ):
        raise ValueError("boot identity mismatch")
    for key in ("aircraft_boot_id", "leptrino_boot_id"):
        after = manifest.get(key + "_after")
        if after != manifest[key] and not (accept_incomplete and after is None):
            raise ValueError("Pi rebooted during acquisition")
    if (root / "rpm/result.txt").read_text().splitlines()[0] != "COMPLETE":
        raise ValueError("incomplete RPM acquisition")
    unanswered = []
    nano_events, qt = read_capture(root / "rpm/rpm.csv", unanswered=unanswered)
    if rpm_model not in (
        "local",
        "global",
        "piecewise_midpoint",
        "smooth_interval",
        "local_or_piecewise",
        "local_or_smooth",
    ):
        raise ValueError("invalid RPM clock model")
    validation, _ = compare(qt, [q.nano_us for q in qt])
    resolved_rpm_model = rpm_model
    if rpm_model == "local_or_piecewise":
        selected = validation["models"]["local"]
        if selected["fit"] == "OK" and selected["held_out"].get("violations") == 0:
            resolved_rpm_model = "local"
        else:
            resolved_rpm_model = "piecewise_midpoint"
            validation["selected_fallback"] = {
                "model": "piecewise_midpoint",
                "status": "AUTOMATIC_FALLBACK_REQUIRES_REVIEW",
                "reason": selected,
            }
    if rpm_model == "local_or_smooth":
        selected = validation["models"]["local"]
        if selected["fit"] == "OK" and selected["held_out"].get("violations") == 0:
            resolved_rpm_model = "local"
        else:
            resolved_rpm_model = "smooth_interval"
            validation["selected_fallback"] = {
                "model": "smooth_interval",
                "status": "AUTOMATIC_FALLBACK_REQUIRES_REVIEW",
                "reason": selected,
            }
    if resolved_rpm_model == "piecewise_midpoint":
        nano = fit_piecewise_midpoint(qt)
        fallback = validation.setdefault("selected_fallback", {})
        fallback.update(
            model="piecewise_midpoint",
            status=fallback.get("status", "OPERATOR_APPROVED_CLOCK_REALIGNMENT"),
            limitations=[
                "Uses all Q/T midpoint anchors and has no independent holdout.",
                "Bounds are interpolated causal exchange intervals, not accuracy claims.",
            ],
        )
    elif resolved_rpm_model == "smooth_interval":
        nano = fit_smooth_interval(qt)
        fallback = validation.setdefault("selected_fallback", {})
        fallback.update(
            model="smooth_interval",
            status=fallback.get("status", "OPERATOR_APPROVED_CLOCK_REALIGNMENT"),
            limitations=[
                "Anchor estimates are constrained to causal Q/T intervals.",
                "Smoothness and robust midpoint weighting do not identify path asymmetry.",
                "No extrapolation or absolute timing-calibration claim.",
            ],
            optimizer_iterations=nano.iterations,
            maximum_anchor_residual_ms=nano.maximum_anchor_residual_ns / 1e6,
        )
    else:
        nano = fit_local(qt) if resolved_rpm_model == "local" else fit_clock(qt)
        selected = validation["models"][resolved_rpm_model]
        if selected["fit"] != "OK" or selected["held_out"].get("violations") != 0:
            raise ValueError("RPM model/held-out validation failed")
    if pi_model not in (
        "affine",
        "piecewise_midpoint",
        "smooth_interval",
        "affine_or_piecewise",
        "affine_or_smooth",
    ):
        raise ValueError("invalid Pi clock model")
    pi_rows = read_exchanges(root / "clock/exchanges.csv")
    resolved_pi_model = pi_model
    if pi_model == "affine_or_piecewise":
        try:
            pi = fit(pi_rows)
            resolved_pi_model = "affine"
        except ValueError:
            pi = fit_piecewise_pi(pi_rows)
            resolved_pi_model = "piecewise_midpoint"
    elif pi_model == "affine_or_smooth":
        try:
            pi = fit(pi_rows)
            resolved_pi_model = "affine"
        except ValueError:
            pi = fit_smooth_pi(pi_rows)
            resolved_pi_model = "smooth_interval"
    else:
        pi = (
            fit(pi_rows)
            if pi_model == "affine"
            else fit_smooth_pi(pi_rows)
            if pi_model == "smooth_interval"
            else fit_piecewise_pi(pi_rows)
        )
    encoder_required = bool(manifest.get("encoder_required", False))
    encoder_metadata = None
    encoder_clock = None
    encoder_pi = None
    if encoder_required:
        encoder_metadata = json.loads((root / "encoder/metadata.json").read_text(encoding="utf-8"))
        encoder_clock = json.loads(
            (root / "encoder-clock/summary.json").read_text(encoding="utf-8")
        )
        if encoder_metadata.get("state") != "COMPLETE" or encoder_clock.get("state") != "COMPLETE":
            raise ValueError("incomplete FFT GYRO encoder or Windows clock acquisition")
        if encoder_clock.get("server_boot_id") != manifest["aircraft_boot_id"]:
            raise ValueError("encoder clock aircraft boot identity mismatch")
        encoder_rows = read_exchanges(root / "encoder-clock/exchanges.csv")
        resolved_encoder_pi_model = pi_model
        if pi_model == "affine_or_piecewise":
            try:
                encoder_pi = fit(encoder_rows)
                resolved_encoder_pi_model = "affine"
            except ValueError:
                encoder_pi = fit_piecewise_pi(encoder_rows)
                resolved_encoder_pi_model = "piecewise_midpoint"
        elif pi_model == "affine_or_smooth":
            try:
                encoder_pi = fit(encoder_rows)
                resolved_encoder_pi_model = "affine"
            except ValueError:
                encoder_pi = fit_smooth_pi(encoder_rows)
                resolved_encoder_pi_model = "smooth_interval"
        else:
            encoder_pi = (
                fit(encoder_rows)
                if pi_model == "affine"
                else fit_smooth_pi(encoder_rows)
                if pi_model == "smooth_interval"
                else fit_piecewise_pi(encoder_rows)
            )
    else:
        resolved_encoder_pi_model = None
    source_force = root / "leptrino/samples.csv"
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text('{"state":"INCOMPLETE"}\n', encoding="utf-8")
    if force["state"] == "RAW_COMPLETE" or recovered:
        source_force = output / "force_decoded.csv"
        decoded = decode(
            root / "leptrino/chunks.bin",
            source_force,
            tuple(force["rated_limits_N_Nm"]),
            force["start_command_before_ns"],
        )
        if decoded["candidate_counter_jumps"]:
            raise ValueError("force candidate sequence gap")
        force.update(decoded)
    controls = []
    if (root / "pid/control.csv").exists():
        pid_result = (root / "pid/result.txt").read_text()
        if "status=COMPLETE" not in pid_result and not (
            accept_incomplete and "status=ABORTED" in pid_result
        ):
            raise ValueError("PID run did not complete")
        with (root / "pid/control.csv").open(newline="") as stream:
            controls = list(csv.DictReader(stream))
        times = [int(row["sent_ns"]) for row in controls]
        if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("PID timestamps invalid")
    else:
        times = []
    force_ambiguous = 0
    force_overlap = 0
    force_count = 0
    force_width_max = 0
    with (
        source_force.open(newline="", encoding="utf-8") as source,
        (output / "force_aligned.csv").open("x", newline="", encoding="utf-8") as target,
    ):
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        first = next(reader, None)
        if first is None:
            raise ValueError("empty force CSV")
        origin = times[0] if times else pi.map(int(first["receive_monotonic_ns"]))[0]
        previous_stamp = -1
        writer = csv.writer(target)
        writer.writerow(
            [
                *fields,
                "aircraft_receive_estimate_ns",
                "aircraft_receive_lower_ns",
                "aircraft_receive_upper_ns",
                "relative_s",
                "pid_row_index",
                "pid_receive_cycle_ambiguous",
            ]
        )
        for row in itertools.chain((first,), reader):
            stamp = int(row["receive_monotonic_ns"])
            if stamp < previous_stamp:
                raise ValueError("force timestamps reversed")
            previous_stamp = stamp
            estimate, low, high = pi.map(stamp)  # refuses extrapolation
            force_width_max = max(force_width_max, (high - low) / 1e6)
            force_count += 1
            index, ambiguous = match_pid(times, estimate, low, high)
            force_overlap += int(index != "")
            force_ambiguous += int(ambiguous is True)
            writer.writerow(
                [
                    *(row[k] for k in fields),
                    estimate,
                    low,
                    high,
                    (estimate - origin) / 1e9,
                    index,
                    ambiguous,
                ]
            )
    if force_count != force["samples"]:
        raise ValueError("force CSV count mismatch")
    encoder_count = encoder_overlap = encoder_ambiguous = encoder_outside = 0
    encoder_width_max = 0.0
    if encoder_required:
        with (
            (root / "encoder/samples.csv").open(newline="", encoding="utf-8") as source,
            (output / "encoder_aligned.csv").open("x", newline="", encoding="utf-8") as target,
        ):
            reader = csv.DictReader(source)
            fields = reader.fieldnames
            if not fields or "receive_monotonic_ns" not in fields:
                raise ValueError("invalid encoder CSV")
            writer = csv.writer(target)
            writer.writerow(
                [
                    *fields,
                    "aircraft_receive_estimate_ns",
                    "aircraft_receive_lower_ns",
                    "aircraft_receive_upper_ns",
                    "relative_s",
                    "within_clock_anchors",
                    "pid_row_index",
                    "pid_receive_cycle_ambiguous",
                ]
            )
            previous_stamp = -1
            for row in reader:
                stamp = int(row["receive_monotonic_ns"])
                if stamp < previous_stamp:
                    raise ValueError("encoder timestamps reversed")
                previous_stamp = stamp
                encoder_count += 1
                bracketed = encoder_pi.first_ns <= stamp <= encoder_pi.last_ns
                if not bracketed:
                    encoder_outside += 1
                    writer.writerow([*(row[k] for k in fields), "", "", "", "", False, "", ""])
                    continue
                estimate, low, high = encoder_pi.map(stamp)
                encoder_width_max = max(encoder_width_max, (high - low) / 1e6)
                index, ambiguous = match_pid(times, estimate, low, high)
                encoder_overlap += int(index != "")
                encoder_ambiguous += int(ambiguous is True)
                writer.writerow(
                    [
                        *(row[k] for k in fields),
                        estimate,
                        low,
                        high,
                        (estimate - origin) / 1e9,
                        True,
                        index,
                        ambiguous,
                    ]
                )
        if encoder_count != encoder_metadata.get("encoder_frames"):
            raise ValueError("encoder CSV count mismatch")
    edge_counts = {str(ch): 0 for ch in range(1, 5)}
    rpm_widths = []
    outside = 0
    rpm_overlap = 0
    rpm_ambiguous = 0
    with (output / "rpm_edges_aligned.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "channel",
                "sequence",
                "nano_unwrapped_us",
                "host_received_ns",
                "aircraft_estimate_ns",
                "aircraft_lower_ns",
                "aircraft_upper_ns",
                "relative_s",
                "within_qt_anchors",
                "pid_row_index",
                "pid_cycle_ambiguous",
            ]
        )
        for event in nano_events:
            if event.kind != "E":
                continue
            stamp = event.unwrapped_us
            bracketed = qt[0].nano_us <= stamp <= qt[-1].nano_us
            outside += int(not bracketed)
            edge_counts[str(event.value)] += 1
            if not bracketed:
                writer.writerow(
                    [
                        event.value,
                        event.sequence,
                        stamp,
                        event.host_ns,
                        "",
                        "",
                        "",
                        "",
                        False,
                        "",
                        "",
                    ]
                )
                continue
            estimate, low, high = nano.map(stamp)
            rpm_widths.append((high - low) / 1e6)
            index, ambiguous = match_pid(times, estimate, low, high)
            rpm_overlap += int(index != "")
            rpm_ambiguous += int(ambiguous is True)
            writer.writerow(
                [
                    event.value,
                    event.sequence,
                    stamp,
                    event.host_ns,
                    estimate,
                    low,
                    high,
                    (estimate - origin) / 1e9,
                    bracketed,
                    index,
                    ambiguous,
                ]
            )
    if times and (not force_overlap or not rpm_overlap):
        raise ValueError("no force/RPM overlap with PID")
    if encoder_required and times and not encoder_overlap:
        raise ValueError("no FFT GYRO encoder overlap with PID")
    summary = dict(
        state="ALIGNED_CONDITIONAL_NOT_QUALIFIED",
        aircraft_boot_id=manifest["aircraft_boot_id"],
        leptrino_boot_id=manifest["leptrino_boot_id"],
        common_clock="aircraft CLOCK_MONOTONIC ns",
        force_samples=force_count,
        rpm_edges_by_channel=edge_counts,
        pid_rows=len(controls),
        force_documented_status_fault_samples=force.get("documented_status_fault_samples", 0),
        force_documented_status_fault_counts=force.get("documented_status_fault_counts", {}),
        force_receive_pid_overlap=force_overlap,
        force_receive_pid_ambiguous=force_ambiguous,
        rpm_pid_overlap=rpm_overlap,
        rpm_pid_ambiguous=rpm_ambiguous,
        force_clock_interval_max_ms=force_width_max,
        rpm_clock_interval_max_ms=max(rpm_widths, default=None),
        encoder_required=encoder_required,
        encoder_samples=encoder_count if encoder_required else None,
        encoder_pid_overlap=encoder_overlap if encoder_required else None,
        encoder_pid_ambiguous=encoder_ambiguous if encoder_required else None,
        encoder_clock_interval_max_ms=encoder_width_max if encoder_required else None,
        encoder_samples_outside_clock_anchors=encoder_outside if encoder_required else None,
        rpm_edges_outside_anchors=outside,
        nano_exchanges=len(qt),
        rpm_model=resolved_rpm_model,
        rpm_model_validation=validation,
        unanswered_nano_q=unanswered,
        max_nano_anchor_gap_ms=max(
            (b.nano_us - a.nano_us) / 1000 for a, b in zip(qt, qt[1:], strict=False)
        ),
        recovered_postprocessing=recovered,
        accepted_incomplete=accepted_incomplete,
        adoption_override=accept_incomplete,
        recovery_limits=(
            "Clock CSV prefix validated against per-run before/after boot IDs; "
            "original clock process did not finish; no claim of full requested duration."
        )
        if recovered
        else None,
        source_manifest_sha256=hashlib.sha256((root / "run.json").read_bytes()).hexdigest(),
        pi_model=(
            dict(
                method="affine",
                client_origin_ns=pi.client_origin_ns,
                server_origin_ns=pi.server_origin_ns,
                vertices=pi.vertices,
            )
            if resolved_pi_model == "affine"
            else dict(
                method=resolved_pi_model,
                anchors=len(pi.stamps if resolved_pi_model == "smooth_interval" else pi.client_ns),
                first_ns=pi.first_ns,
                last_ns=pi.last_ns,
                limitations=[
                    "Uses interval-constrained interpolation without an affine-clock claim."
                ],
                **(
                    {
                        "optimizer_iterations": pi.iterations,
                        "maximum_anchor_residual_ms": pi.maximum_anchor_residual_ns / 1e6,
                    }
                    if resolved_pi_model == "smooth_interval"
                    else {}
                ),
            )
        ),
        encoder_pi_model=(
            (
                dict(
                    method="affine",
                    client_origin_ns=encoder_pi.client_origin_ns,
                    server_origin_ns=encoder_pi.server_origin_ns,
                    vertices=encoder_pi.vertices,
                )
                if resolved_encoder_pi_model == "affine"
                else dict(
                    method=resolved_encoder_pi_model,
                    anchors=len(
                        encoder_pi.stamps
                        if resolved_encoder_pi_model == "smooth_interval"
                        else encoder_pi.client_ns
                    ),
                    first_ns=encoder_pi.first_ns,
                    last_ns=encoder_pi.last_ns,
                    **(
                        {
                            "optimizer_iterations": encoder_pi.iterations,
                            "maximum_anchor_residual_ms": (
                                encoder_pi.maximum_anchor_residual_ns / 1e6
                            ),
                        }
                        if resolved_encoder_pi_model == "smooth_interval"
                        else {}
                    ),
                )
            )
            if encoder_required
            else None
        ),
        limits=[
            "Force timestamps are USB read completion, NOT sensor sample time.",
            "FFT GYRO timestamps are Windows serial-read completion, NOT encoder sample time.",
            "Clock bounds assume affine clocks and causal timestamps, not absolute calibration.",
            "Nano timestamps include assumed 10 us Q/T slack.",
            "PID rows are nearest preceding command send, NOT physical application time.",
            "No resampling, pulses/revolution assumption, or missing-data interpolation.",
        ],
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recover-postprocessing", action="store_true")
    parser.add_argument("--accept-incomplete", action="store_true")
    parser.add_argument(
        "--rpm-model",
        choices=(
            "local",
            "global",
            "piecewise_midpoint",
            "smooth_interval",
            "local_or_piecewise",
            "local_or_smooth",
        ),
        default="local",
    )
    parser.add_argument(
        "--pi-model",
        choices=(
            "affine",
            "piecewise_midpoint",
            "smooth_interval",
            "affine_or_piecewise",
            "affine_or_smooth",
        ),
        default="affine",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            align_run(
                args.run,
                args.output,
                recover_postprocessing=args.recover_postprocessing,
                accept_incomplete=args.accept_incomplete,
                rpm_model=args.rpm_model,
                pi_model=args.pi_model,
            ),
            indent=2,
        )
    )
