from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .flight_log import FlightExtractionReport, extract_ardupilot_flight_csv
from .lgr101 import read_lgr101_csv
from .rpm_capture import MICROS_MODULUS, parse_uno_line
from .session import ExperimentSession


@dataclass(frozen=True, slots=True)
class NumericSeries:
    name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[float, tuple[float, ...]], ...]
    interpolation: str = "linear"


@dataclass(frozen=True, slots=True)
class FinalizeReport:
    run_id: str
    status: str
    flight: dict[str, Any]
    source_files: dict[str, dict[str, Any]]
    aligned_csv: str
    aligned_rows: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_bundle_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    required_tables = ("session", "flight", "alignment", "arduino", "hioki")
    missing = [name for name in required_tables if not isinstance(document.get(name), dict)]
    if missing:
        raise ValueError(f"configuration_tables_missing:{','.join(missing)}")
    return document


def create_run_from_config(config_path: Path, run_id: str) -> ExperimentSession:
    document = load_bundle_config(config_path)
    session = document["session"]
    output_root = Path(str(session["output_root"])).expanduser()
    created = ExperimentSession.create(
        output_root,
        run_id=run_id,
        protocol_id=str(session["protocol_id"]),
        protocol_version=str(session["protocol_version"]),
        profile=str(session["profile"]),
        operator=str(session["operator"]),
        hardware_config_id=str(session["hardware_config_id"]),
        esc_inventory_id=str(session["esc_inventory_id"]),
        fixture_id=_optional_text(session.get("fixture_id")),
        notes=str(session.get("notes", "")),
    )
    (created.root / "RUN_IN_PROGRESS").touch(exist_ok=False)
    created.append_event(
        "acquisition_contract_created",
        {
            "required_sources": ["flight", "leptrino", "hioki", "arduino"],
            "config": str(config_path.resolve()),
        },
    )
    return created


def finalize_run(
    config_path: Path,
    run_root: Path,
    *,
    flight_path: Path,
    leptrino_path: Path,
    hioki_path: Path,
    arduino_path: Path,
    arm_index: int = 0,
    allow_multiple_arm_windows: bool = False,
) -> FinalizeReport:
    config = load_bundle_config(config_path)
    session = ExperimentSession.open_existing(run_root)
    inputs = {
        "flight": flight_path,
        "leptrino": leptrino_path,
        "hioki": hioki_path,
        "arduino": arduino_path,
    }
    for source, path in inputs.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required_source_missing_or_empty:{source}:{path}")

    destinations = {
        "flight": session.raw_source_dir("pixhawk") / f"flight{flight_path.suffix}",
        "leptrino": session.raw_source_dir("loadcell_lgr101u") / f"leptrino{leptrino_path.suffix}",
        "hioki": session.raw_source_dir("aux_lr8450") / f"hioki{hioki_path.suffix}",
        "arduino": session.raw_source_dir("sync_uno") / f"arduino{arduino_path.suffix}",
    }
    source_manifest: dict[str, dict[str, Any]] = {}
    for source, input_path in inputs.items():
        destination = destinations[source]
        _copy_new(input_path, destination)
        source_manifest[source] = {
            "original_path": str(input_path.resolve()),
            "stored_path": str(destination.relative_to(run_root)),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
        session.append_event("source_stored", {"source": source, **source_manifest[source]})

    processed = run_root / "processed"
    if processed.exists():
        raise FileExistsError(f"processed_destination_already_exists:{processed}")
    motor_channels = tuple(int(value) for value in config["flight"]["motor_channels"])
    with tempfile.TemporaryDirectory(prefix=".processed-", dir=run_root) as temporary:
        staging = Path(temporary)
        flight_csv = staging / "flight_motor_outputs.csv"
        flight_report = extract_ardupilot_flight_csv(
            destinations["flight"],
            flight_csv,
            arm_index=arm_index,
            motor_channels=motor_channels,
        )
        if flight_report.arm_window_count != 1 and not allow_multiple_arm_windows:
            raise ValueError(
                "flight_log_must_contain_exactly_one_complete_arm_window:"
                f"count={flight_report.arm_window_count}"
            )

        flight_series = _read_normalized_csv(
            flight_csv,
            "flight",
            interpolation="previous",
            ignore_columns={"time_us", "source_message"},
        )
        leptrino_series = normalize_leptrino(
            destinations["leptrino"], staging / "leptrino.csv"
        )
        arduino_series, arduino_duration = normalize_arduino(
            destinations["arduino"],
            staging / "arduino_rpm.csv",
            marks_per_revolution=int(config["arduino"].get("marks_per_revolution", 2)),
            minimum_edge_interval_us=int(
                config["arduino"].get("minimum_edge_interval_us", 5_000)
            ),
            require_two_sync_markers=bool(
                config["arduino"].get("require_arm_and_disarm_markers", True)
            ),
        )
        hioki_series, hioki_duration = normalize_hioki(
            destinations["hioki"],
            staging / "hioki.csv",
            config["hioki"],
        )

        duration = flight_report.arm_window.duration_s
        tolerance = float(config["alignment"].get("duration_tolerance_s", 0.1))
        warnings = list(flight_report.warnings)
        _check_duration("leptrino", _series_end(leptrino_series), duration, tolerance)
        if arduino_duration is not None:
            _check_sync_duration("arduino", arduino_duration, duration, tolerance)
        elif _series_end(arduino_series) < duration - tolerance:
            warnings.append("arduino_rpm_does_not_cover_full_arm_window")
        if hioki_duration is not None:
            _check_sync_duration("hioki", hioki_duration, duration, tolerance)
        _check_duration("hioki", _series_end(hioki_series), duration, tolerance)

        aligned_rows = write_aligned_csv(
            staging / "aligned.csv",
            (flight_series, leptrino_series, hioki_series, arduino_series),
            duration_s=duration,
            sample_rate_hz=float(config["alignment"].get("sample_rate_hz", 100.0)),
        )
        staging.rename(processed)

    manifest_path = run_root / "source_manifest.json"
    _write_new_json(manifest_path, source_manifest)
    report = FinalizeReport(
        run_id=session.metadata.run_id,
        status="completed",
        flight=flight_report.to_dict(),
        source_files=source_manifest,
        aligned_csv=str(Path("processed") / "aligned.csv"),
        aligned_rows=aligned_rows,
        warnings=tuple(warnings),
    )
    _write_new_json(run_root / "report.json", report.to_dict())
    session.finalize(status="completed", summary=report.to_dict())
    marker = run_root / "RUN_IN_PROGRESS"
    if marker.exists():
        marker.unlink()
    return report


def normalize_leptrino(input_path: Path, output_path: Path) -> NumericSeries:
    log = read_lgr101_csv(input_path)
    origin = log.samples[0].elapsed_s
    columns = ("fx_n", "fy_n", "fz_n", "mx_nm", "my_nm", "mz_nm")
    rows = tuple(
        (sample.elapsed_s - origin, sample.values)
        for sample in log.samples
    )
    series = NumericSeries("leptrino", columns, rows)
    _write_series(output_path, series)
    return series


def normalize_arduino(
    input_path: Path,
    output_path: Path,
    *,
    marks_per_revolution: int,
    require_two_sync_markers: bool,
    minimum_edge_interval_us: int = 5_000,
) -> tuple[NumericSeries, float | None]:
    if marks_per_revolution <= 0:
        raise ValueError("marks_per_revolution_must_be_positive")
    if minimum_edge_interval_us <= 0:
        raise ValueError("minimum_edge_interval_us_must_be_positive")
    events: list[tuple[str, str | None, int]] = []
    wraps = 0
    previous_raw: int | None = None
    with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line in stream:
            event = parse_uno_line(line)
            if event is None:
                continue
            if event.kind == "O":
                raise ValueError("arduino_ring_buffer_overflow_run_rejected")
            raw = event.micros
            if previous_raw is not None and raw < previous_raw:
                wraps += 1
            previous_raw = raw
            events.append((event.kind, event.label, raw + wraps * MICROS_MODULUS))
    command_on = [time_us for kind, label, time_us in events if kind == "M" and label == "COMMAND_ON"]
    command_off = [time_us for kind, label, time_us in events if kind == "M" and label == "COMMAND_OFF"]
    sync_times = [time_us for kind, _, time_us in events if kind == "S"]
    if command_on:
        origin = command_on[0]
        end_markers = command_off
    else:
        if not sync_times:
            raise ValueError("arduino_command_on_or_legacy_sync_marker_missing")
        origin = sync_times[0]
        end_markers = sync_times[1:]
    if require_two_sync_markers and not end_markers:
        raise ValueError("arduino_command_off_or_legacy_end_marker_missing")
    sync_duration = (end_markers[-1] - origin) / 1_000_000.0 if end_markers else None

    rows: list[tuple[float, tuple[float, ...]]] = []
    previous_edge: int | None = None
    for kind, _, time_us in events:
        if kind != "R" or time_us < origin:
            continue
        if previous_edge is not None and time_us > previous_edge:
            delta_us = time_us - previous_edge
            if delta_us < minimum_edge_interval_us:
                continue
            rpm = 60_000_000.0 / (marks_per_revolution * delta_us)
            rows.append(((time_us - origin) / 1_000_000.0, (rpm,)))
        previous_edge = time_us
    series = NumericSeries("arduino", ("rpm",), tuple(rows), interpolation="previous")
    _write_series(output_path, series)
    return series, sync_duration


def normalize_hioki(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> tuple[NumericSeries, float | None]:
    encoding = str(config.get("encoding", "utf-8-sig"))
    delimiter = str(config.get("delimiter", ","))
    time_column = str(config["time_column"])
    selected = config.get("columns", {})
    if not isinstance(selected, dict) or not selected:
        raise ValueError("hioki_columns_mapping_missing")
    header_row = int(config.get("header_row", 0))

    with input_path.open("r", encoding=encoding, newline="") as stream:
        rows = list(csv.reader(stream, delimiter=delimiter))
    if header_row >= len(rows):
        raise ValueError("hioki_header_row_out_of_range")
    header = [cell.strip() for cell in rows[header_row]]
    try:
        time_index = header.index(time_column)
        value_indices = {name: header.index(str(column)) for name, column in selected.items()}
    except ValueError as exc:
        raise ValueError(f"hioki_required_column_missing:{exc}") from exc

    sync_column = _optional_text(config.get("sync_column"))
    sync_index = header.index(sync_column) if sync_column else None
    threshold = float(config.get("sync_threshold", 0.5))
    parsed: list[tuple[float, tuple[float, ...], float | None]] = []
    for line_number, row in enumerate(rows[header_row + 1 :], start=header_row + 2):
        if not row or all(not cell.strip() for cell in row):
            continue
        try:
            time_s = float(row[time_index])
            values = tuple(float(row[index]) for index in value_indices.values())
            sync = float(row[sync_index]) if sync_index is not None else None
        except (IndexError, ValueError) as exc:
            raise ValueError(f"hioki_line_{line_number}_not_numeric") from exc
        parsed.append((time_s, values, sync))
    if len(parsed) < 2:
        raise ValueError("hioki_fewer_than_two_rows")

    sync_edges: list[float] = []
    previous_sync: float | None = None
    for time_s, _, sync in parsed:
        if sync is not None and previous_sync is not None:
            if previous_sync < threshold <= sync:
                sync_edges.append(time_s)
        previous_sync = sync
    require_markers = bool(config.get("require_arm_and_disarm_markers", True))
    if sync_index is not None:
        if not sync_edges:
            raise ValueError("hioki_arm_sync_marker_missing")
        if require_markers and len(sync_edges) < 2:
            raise ValueError("hioki_disarm_sync_marker_missing")
        origin = sync_edges[0]
        sync_duration = sync_edges[-1] - origin if len(sync_edges) >= 2 else None
    else:
        if str(config.get("sync_mode", "")).lower() != "recording_start":
            raise ValueError("hioki_sync_column_or_recording_start_mode_required")
        origin = parsed[0][0]
        sync_duration = None

    normalized_rows = tuple(
        (time_s - origin, values)
        for time_s, values, _ in parsed
        if time_s >= origin
    )
    series = NumericSeries("hioki", tuple(value_indices), normalized_rows)
    _write_series(output_path, series)
    return series, sync_duration


def write_aligned_csv(
    output_path: Path,
    series: Iterable[NumericSeries],
    *,
    duration_s: float,
    sample_rate_hz: float,
) -> int:
    if duration_s <= 0 or sample_rate_hz <= 0:
        raise ValueError("alignment_duration_and_rate_must_be_positive")
    materialized = tuple(series)
    column_names = tuple(
        f"{item.name}_{column}" for item in materialized for column in item.columns
    )
    row_count = math.floor(duration_s * sample_rate_hz) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", *column_names))
        for index in range(row_count):
            time_s = index / sample_rate_hz
            values: list[str | float] = []
            for item in materialized:
                sampled = _sample(item, time_s)
                values.extend("" if value is None else value for value in sampled)
            writer.writerow((f"{time_s:.9f}", *values))
    return row_count


def _sample(series: NumericSeries, time_s: float) -> tuple[float | None, ...]:
    if not series.rows:
        return tuple(None for _ in series.columns)
    if time_s < series.rows[0][0] or time_s > series.rows[-1][0]:
        return tuple(None for _ in series.columns)
    lower_index = 0
    upper_index = len(series.rows) - 1
    while lower_index <= upper_index:
        middle = (lower_index + upper_index) // 2
        if series.rows[middle][0] <= time_s:
            lower_index = middle + 1
        else:
            upper_index = middle - 1
    previous = series.rows[max(0, upper_index)]
    if series.interpolation == "previous" or upper_index + 1 >= len(series.rows):
        return previous[1]
    following = series.rows[upper_index + 1]
    span = following[0] - previous[0]
    if span <= 0:
        return previous[1]
    fraction = (time_s - previous[0]) / span
    return tuple(
        left + fraction * (right - left)
        for left, right in zip(previous[1], following[1])
    )


def _read_normalized_csv(
    path: Path,
    name: str,
    *,
    interpolation: str,
    ignore_columns: set[str],
) -> NumericSeries:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "time_s" not in reader.fieldnames:
            raise ValueError(f"normalized_csv_time_missing:{path}")
        columns = tuple(
            column for column in reader.fieldnames
            if column != "time_s" and column not in ignore_columns
        )
        rows = tuple(
            (float(row["time_s"]), tuple(float(row[column]) for column in columns))
            for row in reader
        )
    return NumericSeries(name, columns, rows, interpolation)


def _write_series(path: Path, series: NumericSeries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", *series.columns))
        for time_s, values in series.rows:
            writer.writerow((f"{time_s:.9f}", *values))


def _series_end(series: NumericSeries) -> float:
    return series.rows[-1][0] if series.rows else 0.0


def _check_duration(source: str, observed: float, required: float, tolerance: float) -> None:
    if observed < required - tolerance:
        raise ValueError(
            f"source_does_not_cover_arm_window:{source}:observed={observed:.6f}:"
            f"required={required:.6f}"
        )


def _check_sync_duration(source: str, observed: float, required: float, tolerance: float) -> None:
    residual = abs(observed - required)
    if residual > tolerance:
        raise ValueError(
            f"sync_duration_mismatch:{source}:residual={residual:.6f}:"
            f"tolerance={tolerance:.6f}"
        )


def _copy_new(source: Path, destination: Path) -> None:
    if destination.exists():
        if _sha256(source) == _sha256(destination):
            return
        raise FileExistsError(f"raw_destination_exists_with_different_content:{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, document: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
