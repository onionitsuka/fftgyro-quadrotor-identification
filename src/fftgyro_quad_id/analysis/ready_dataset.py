"""Build signal-specific, masked 100 Hz datasets without modifying raw data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fftgyro_quad_id.sync.force_grid import reconstruct
from fftgyro_quad_id.sync.parallel_align import align_run

RPM_CLOCK_REALIGN = {34, 639}
MAX_SOURCE_GAP_NS = 30_000_000
RPM_MAX_GAP_NS = 50_000_000
ULOG_DROPOUT_PAD_NS = 50_000_000
PREFIX_SEGMENT_MAX = {765: 24}


@dataclass(frozen=True)
class BuildResult:
    number: int
    run: str
    rows: int
    state: str
    output: str
    valid_joint_rows: int
    sync_residual_p95_ms: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row.get(name, "nan") or "nan") for row in rows])


def _int_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([int(row[name]) for row in rows], dtype=np.int64)


def _zoh(
    source_ns: np.ndarray, values: np.ndarray, grid_ns: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.searchsorted(source_ns, grid_ns, side="right") - 1
    valid = indices >= 0
    clipped = np.clip(indices, 0, max(len(source_ns) - 1, 0))
    age = np.full(grid_ns.shape, np.iinfo(np.int64).max, dtype=np.int64)
    result = np.full(grid_ns.shape, np.nan, dtype=float)
    if len(source_ns):
        age[valid] = grid_ns[valid] - source_ns[clipped[valid]]
        result[valid] = values[clipped[valid]]
    return result, valid & (age <= MAX_SOURCE_GAP_NS), clipped


def _linear_with_gap(
    source_ns: np.ndarray,
    values: np.ndarray,
    grid_ns: np.ndarray,
    *,
    max_gap_ns: int = MAX_SOURCE_GAP_NS,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    source_ns, values = source_ns[finite], values[finite]
    result = np.full(grid_ns.shape, np.nan, dtype=float)
    valid = np.zeros(grid_ns.shape, dtype=bool)
    if len(source_ns) < 2:
        return result, valid
    right = np.searchsorted(source_ns, grid_ns, side="left")
    inside = (right > 0) & (right < len(source_ns))
    selected = np.flatnonzero(inside)
    left_index = right[selected] - 1
    right_index = right[selected]
    gap = source_ns[right_index] - source_ns[left_index]
    good = gap <= max_gap_ns
    selected = selected[good]
    left_index = left_index[good]
    right_index = right_index[good]
    weight = (grid_ns[selected] - source_ns[left_index]) / (
        source_ns[right_index] - source_ns[left_index]
    )
    result[selected] = values[left_index] + weight * (values[right_index] - values[left_index])
    valid[selected] = True
    return result, valid


def _unwrap_degrees(values: np.ndarray) -> np.ndarray:
    """Unwrap periodic encoder degrees before interpolation across +/-180."""
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(values, dtype=float))))


def fit_px4_clock(rows: list[dict[str, str]]) -> tuple[float, float, dict[str, float]]:
    x = _float_column(rows, "px4_boot_ms") * 1e6
    y = _float_column(rows, "feedback_received_ns")
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[valid], y[valid]
    if len(x) < 20:
        raise ValueError("insufficient PX4/Pi clock pairs")
    origin = float(x[0])
    slope, intercept_centered = np.polyfit(x - origin, y, 1)
    predicted = slope * (x - origin) + intercept_centered
    residual = y - predicted
    # Earliest observed receipt is the least-delay causal anchor.
    shift = float(np.quantile(residual, 0.01))
    intercept = intercept_centered - slope * origin + shift
    adjusted = residual - shift
    details = {
        "slope": float(slope),
        "drift_ppm": float((slope - 1.0) * 1e6),
        "residual_p05_ms": float(np.quantile(adjusted, 0.05) / 1e6),
        "residual_p50_ms": float(np.quantile(adjusted, 0.50) / 1e6),
        "residual_p95_ms": float(np.quantile(adjusted, 0.95) / 1e6),
        "pairs": int(len(x)),
        "method": "least_squares_drift_plus_earliest_1pct_receive_anchor",
    }
    return float(slope), float(intercept), details


def _analysis_rows(rows: list[dict[str, str]], number: int) -> np.ndarray:
    explicit = np.asarray([_truth(row.get("analysis_window", "")) for row in rows])
    selected = (
        explicit if explicit.any() else np.asarray([row.get("stage") == "SEGMENT" for row in rows])
    )
    if number in PREFIX_SEGMENT_MAX:
        limit = PREFIX_SEGMENT_MAX[number]
        suffix = np.asarray(
            [
                int(match.group(1))
                if (match := re.search(r"-(\d+)$", row.get("segment_id", "")))
                else 10**9
                for row in rows
            ]
        )
        selected &= suffix <= limit
    return selected


def _find_ulog(run_root: Path, relative: str) -> Path:
    path = run_root.parent / relative
    if path.exists():
        return path
    candidates = sorted((run_root / "flightlog").glob("*.ulg"))
    if len(candidates) != 1:
        raise ValueError(f"expected one ULog, found {len(candidates)}")
    return candidates[0]


def _ensure_alignment(
    run_root: Path,
    cache_root: Path,
    *,
    number: int,
    accept_incomplete: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    existing_grid = sorted(run_root.glob("*/force_grid.csv"))
    existing_aligned = sorted(run_root.glob("*/rpm_edges_aligned.csv"))
    if existing_grid and existing_aligned:
        return existing_aligned[0].parent, existing_grid[0].parent, {"source": "existing"}
    aligned = cache_root / run_root.name / "aligned"
    grid = cache_root / run_root.name / "force-grid"
    if not (aligned / "summary.json").exists():
        summary = align_run(
            run_root,
            aligned,
            rpm_model=(
                "smooth_interval" if number in RPM_CLOCK_REALIGN else "local_or_smooth"
            ),
            accept_incomplete=accept_incomplete,
            pi_model=(
                "smooth_interval" if number in RPM_CLOCK_REALIGN else "affine_or_smooth"
            ),
        )
    else:
        summary = json.loads((aligned / "summary.json").read_text())
    if not (grid / "summary.json").exists():
        reconstruct(aligned, grid, run_root=run_root)
    return aligned, grid, summary


def _aggregate_force(
    path: Path, grid_ns: np.ndarray, step_ns: int
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, int]]]:
    rows = _read_csv(path)
    time_name = (
        "estimated_measurement_aircraft_ns"
        if rows and "estimated_measurement_aircraft_ns" in rows[0]
        else "aircraft_receive_estimate_ns"
    )
    times = _int_column(rows, time_name)
    bins = ((times - grid_ns[0]) // step_ns).astype(np.int64)
    in_range = (bins >= 0) & (bins < len(grid_ns))
    bins = bins[in_range]
    counts = np.bincount(bins, minlength=len(grid_ns))
    result: dict[str, np.ndarray] = {}
    for name in ("fx_N", "fy_N", "fz_N", "mx_Nm", "my_Nm", "mz_Nm"):
        values = _float_column(rows, name)[in_range]
        sums = np.bincount(bins, weights=values, minlength=len(grid_ns))
        result[name] = np.divide(sums, counts, out=np.full(len(grid_ns), np.nan), where=counts > 0)
    status = np.zeros(len(grid_ns), dtype=np.uint32)
    raw_status = np.asarray([int(row.get("status", "0") or 0) for row in rows], dtype=np.uint32)
    np.bitwise_or.at(status, bins, raw_status[in_range])
    valid = counts >= 6
    receive = _int_column(rows, "receive_monotonic_ns")
    receive_gap_masks: list[dict[str, int]] = []
    for index in np.flatnonzero(np.diff(receive) > MAX_SOURCE_GAP_NS):
        gap_ns = int(receive[index + 1] - receive[index])
        end_ns = int(times[index + 1])
        start_ns = end_ns - gap_ns
        valid &= ~((grid_ns >= start_ns) & (grid_ns <= end_ns))
        receive_gap_masks.append({"start_ns": start_ns, "end_ns": end_ns, "receive_gap_ns": gap_ns})
    return result, valid, status, receive_gap_masks


def _rpm_channels(path: Path, grid_ns: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows = _read_csv(path)
    result: dict[str, np.ndarray] = {}
    all_valid = np.ones(len(grid_ns), dtype=bool)
    for channel in range(1, 5):
        selected = [
            row for row in rows if int(row["channel"]) == channel and row["aircraft_estimate_ns"]
        ]
        times = _int_column(selected, "aircraft_estimate_ns")
        if len(times) < 2:
            result[f"rpm_{channel}"] = np.full(len(grid_ns), np.nan)
            all_valid &= False
            continue
        delta = np.diff(times)
        rpm = 30e9 / delta  # two edges per revolution
        values, valid = _linear_with_gap(times[1:], rpm, grid_ns, max_gap_ns=RPM_MAX_GAP_NS)
        result[f"rpm_{channel}"] = values
        all_valid &= valid & (values >= 100) & (values <= 20_000)
    return result, all_valid


def _encoder_motion(path: Path, grid_ns: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    empty = np.full(len(grid_ns), np.nan)
    if not path.exists():
        return {
            "encoder_pitch_deg": empty.copy(),
            "encoder_yaw_deg": empty.copy(),
        }, np.zeros(len(grid_ns), dtype=bool)
    rows = [
        row
        for row in _read_csv(path)
        if row.get("aircraft_receive_estimate_ns")
        and _truth(row.get("within_clock_anchors", "true"))
    ]
    if len(rows) < 2:
        return {
            "encoder_pitch_deg": empty.copy(),
            "encoder_yaw_deg": empty.copy(),
        }, np.zeros(len(grid_ns), dtype=bool)
    times = _int_column(rows, "aircraft_receive_estimate_ns")
    result: dict[str, np.ndarray] = {}
    valid_all = np.ones(len(grid_ns), dtype=bool)
    for source, target in (
        ("experimental_pitch_deg", "encoder_pitch_deg"),
        ("experimental_yaw_deg", "encoder_yaw_deg"),
    ):
        values, valid = _linear_with_gap(
            times, _unwrap_degrees(_float_column(rows, source)), grid_ns
        )
        result[target] = values
        valid_all &= valid
    return result, valid_all


def _ulog_motion(
    path: Path,
    grid_ns: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, int]]]:
    from pyulog import ULog

    ulog = ULog(str(path), message_name_filter_list=["vehicle_angular_velocity"])
    topics = [item for item in ulog.data_list if item.name == "vehicle_angular_velocity"]
    if not topics:
        raise ValueError("vehicle_angular_velocity missing from ULog")
    data = topics[0].data
    px4_us = np.asarray(data["timestamp_sample"], dtype=np.float64)
    times = np.rint(slope * px4_us * 1000.0 + intercept).astype(np.int64)
    result: dict[str, np.ndarray] = {}
    valid_all = np.ones(len(grid_ns), dtype=bool)
    for axis, name in enumerate(("p_deg_s", "q_deg_s", "r_deg_s")):
        source = np.rad2deg(np.asarray(data[f"xyz[{axis}]"], dtype=float))
        values, valid = _linear_with_gap(times, source, grid_ns)
        result[name] = values
        valid_all &= valid
    intervals: list[dict[str, int]] = []
    for dropout in ulog.dropouts:
        end_us = int(dropout.timestamp)
        start_us = end_us - int(dropout.duration) * 1000
        start_ns = round(slope * start_us * 1000 + intercept) - ULOG_DROPOUT_PAD_NS
        end_ns = round(slope * end_us * 1000 + intercept) + ULOG_DROPOUT_PAD_NS
        valid_all &= ~((grid_ns >= start_ns) & (grid_ns <= end_ns))
        intervals.append({"start_ns": start_ns, "end_ns": end_ns})
    return result, valid_all, intervals


def _write_parquet(path: Path, columns: dict[str, np.ndarray]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table(columns), path, compression="zstd")


def build_run(
    adoption: dict[str, str],
    *,
    data_root: Path,
    output_root: Path,
    rate_hz: int = 100,
) -> BuildResult:
    number = int(adoption["number"])
    run = adoption["run"]
    run_root = data_root / run
    target = output_root / run
    metadata_path = target / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        return BuildResult(
            number,
            run,
            metadata["rows"],
            "SKIPPED_COMPLETE",
            str(target),
            metadata["valid_joint_rows"],
            metadata["px4_clock"]["residual_p95_ms"],
        )
    partial = output_root / f".{run}.partial-{os.getpid()}"
    partial.mkdir(parents=True, exist_ok=False)
    try:
        aligned, force_grid, alignment = _ensure_alignment(
            run_root,
            output_root / "_alignment_cache_v3",
            number=number,
            accept_incomplete=True,
        )
        controls = _read_csv(run_root / "pid/control.csv")
        selected = _analysis_rows(controls, number)
        control_ns = _int_column(controls, "sent_ns")
        if not selected.any():
            raise ValueError("no adopted analysis rows")
        step_ns = round(1e9 / rate_hz)
        start_ns = math.ceil(int(control_ns[selected][0]) / step_ns) * step_ns
        end_ns = math.floor(int(control_ns[selected][-1]) / step_ns) * step_ns
        grid_ns = np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)
        t_s = (grid_ns - grid_ns[0]) / 1e9

        signals: dict[str, np.ndarray] = {"t_s": t_s, "aircraft_time_ns": grid_ns}
        masks: dict[str, np.ndarray] = {"t_s": t_s}
        input_valid = np.ones(len(grid_ns), dtype=bool)
        source_analysis = selected.astype(float)
        _, analysis_valid, source_index = _zoh(control_ns, source_analysis, grid_ns)
        input_valid &= analysis_valid & selected[source_index]
        masks["analysis_window"] = selected[source_index]
        for name in ("stage", "segment_id", "hold_id"):
            source = np.asarray([row.get(name, "") for row in controls])
            masks[name] = source[source_index]
        for name in (
            "target_deg",
            "external_delta",
            "collective",
            "fixed_c",
            "fixed_p",
            "fixed_r",
            "fixed_y",
            "sent1",
            "sent2",
            "sent3",
            "sent4",
        ):
            values, valid, _ = _zoh(control_ns, _float_column(controls, name), grid_ns)
            signals[name] = values
            input_valid &= valid

        slope, intercept, px4_clock = fit_px4_clock(controls)
        motion, motion_valid, dropout_intervals = _ulog_motion(
            _find_ulog(run_root, adoption["ulog_path"]), grid_ns, slope, intercept
        )
        signals.update(motion)
        force, load_valid, force_status, force_gap_intervals = _aggregate_force(
            force_grid / "force_grid.csv", grid_ns, step_ns
        )
        signals.update(force)
        signals["force_status"] = force_status
        rpm, rpm_valid = _rpm_channels(aligned / "rpm_edges_aligned.csv", grid_ns)
        signals.update(rpm)
        encoder, encoder_valid = _encoder_motion(aligned / "encoder_aligned.csv", grid_ns)
        signals.update(encoder)

        masks.update(
            valid_input=input_valid,
            valid_motion=motion_valid,
            valid_rpm=rpm_valid,
            valid_load=load_valid,
            valid_encoder=encoder_valid,
            valid_sync=np.ones(len(grid_ns), dtype=bool),
        )
        masks["valid_joint"] = input_valid & motion_valid & rpm_valid & load_valid
        _write_parquet(partial / "signals.parquet", signals)
        _write_parquet(partial / "masks.parquet", masks)
        valid_joint_rows = int(masks["valid_joint"].sum())
        metadata = {
            "schema": "fftgyro-analysis-ready-v1",
            "state": "COMPLETE_CONDITIONAL_ALIGNMENT",
            "number": number,
            "run": run,
            "condition_id": adoption["condition_id"],
            "rows": len(grid_ns),
            "rate_hz": rate_hz,
            "common_clock": "aircraft CLOCK_MONOTONIC ns",
            "analysis_start_ns": int(grid_ns[0]),
            "analysis_end_ns": int(grid_ns[-1]),
            "valid_joint_rows": valid_joint_rows,
            "adoption": {
                key: adoption[key]
                for key in (
                    "motion_adoption",
                    "rpm_adoption",
                    "load_adoption",
                    "px4_adoption",
                    "sync_adoption",
                )
            },
            "px4_clock": px4_clock,
            "ulog_dropout_masks": dropout_intervals,
            "force_receive_gap_masks": force_gap_intervals,
            "rpm_clock_model": alignment.get("rpm_model", "existing"),
            "alignment_state": alignment.get("state", alignment.get("source")),
            "force_resampling": "10 ms bin mean; valid with at least 6 source samples",
            "rpm_assumption": "2 edges per revolution; gap-limited interpolation",
            "raw_data_modified": False,
            "sources": {
                "run_json_sha256": _sha256(run_root / "run.json"),
                "control_csv_sha256": _sha256(run_root / "pid/control.csv"),
                "ulog_sha256": _sha256(_find_ulog(run_root, adoption["ulog_path"])),
            },
            "limitations": [
                "PX4-to-aircraft clock uses a causal earliest-receipt estimate "
                "and remains reviewable.",
                "Leptrino nominal sample grid includes the documented 4.13 ms vendor delay only.",
                "Rig response is not interpreted as free-flight response.",
            ],
        }
        (partial / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        partial.rename(target)
        return BuildResult(
            number,
            run,
            len(grid_ns),
            "COMPLETE",
            str(target),
            valid_joint_rows,
            px4_clock["residual_p95_ms"],
        )
    except Exception:
        (partial / "FAILED").write_text("dataset construction failed; inspect runner log\n")
        raise


def load_adoption(path: Path) -> list[dict[str, str]]:
    return _read_csv(path)
