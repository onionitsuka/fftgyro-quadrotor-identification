from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROCESSING_VERSION = 1
VALID_FINAL_STATUSES = {"completed", "completed_with_approximate_offline_sync"}


class PreparationError(RuntimeError):
    """Base error for analysis preparation."""


class DataIntegrityError(PreparationError):
    """A referenced input is missing or has changed."""


class ConfigurationError(PreparationError):
    """The requested roots or run selector are invalid."""


@dataclass(frozen=True, slots=True)
class CsvQc:
    path: str
    rows: int
    columns: int
    timestamp_column: str | None
    timestamp_first: float | None
    timestamp_last: float | None
    timestamp_reversals: int
    timestamp_duplicates: int
    timestamp_rate_hz: float | None
    nan_cells: int
    nonfinite_cells: int
    saturated_cells: int
    ack_error_rows: int
    parse_errors: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _write_exact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise DataIntegrityError(f"refusing to overwrite different output: {path}")
        return
    path.write_bytes(content)


def _write_readme(path: Path, content: bytes, *, verify_only: bool) -> None:
    if not verify_only or not path.exists():
        _write_exact(path, content)
        return

    existing = path.read_bytes()
    if existing == content:
        return
    existing_lines = existing.splitlines(keepends=True)
    expected_lines = content.splitlines(keepends=True)
    differences = [
        (left, right)
        for left, right in zip(existing_lines, expected_lines)
        if left != right
    ]
    verify_mode_lines = {
        b"- Verify only: False\n",
        b"- Verify only: True\n",
    }
    if (
        len(existing_lines) == len(expected_lines)
        and len(differences) == 1
        and set(differences[0]) == verify_mode_lines
    ):
        return
    raise DataIntegrityError(f"refusing to overwrite different output: {path}")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _status(run_root: Path) -> tuple[str, dict[str, Any] | None, list[str]]:
    final_path = run_root / "final.json"
    if not final_path.is_file():
        return "no-final", None, ["final_json_missing"]
    try:
        final = _read_json(final_path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return "invalid-final", None, [f"final_json_invalid:{exc}"]
    return str(final.get("status", "unknown")), final, []


def _manifest_entries(run_root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    path = run_root / "source_manifest.json"
    if not path.is_file():
        return {}, []
    try:
        raw = _read_json(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"source_manifest_invalid:{exc}"]
    entries: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, item in raw.items():
        if not isinstance(item, dict) or "sha256" not in item:
            errors.append(f"source_manifest_entry_invalid:{name}")
            continue
        relative = item.get("path") or item.get("stored_path") or name
        relative_path = Path(str(relative))
        if (
            relative_path.parts
            and relative_path.parts[0] not in {"raw", "processed"}
            and not relative_path.is_absolute()
        ):
            relative_path = Path("raw") / relative_path
        normalized = {
            **item,
            "path": str(relative_path),
            "_source": item.get("source")
            or (relative_path.parts[1] if len(relative_path.parts) > 1 else str(name)),
        }
        entries[str(name)] = normalized
        source = run_root / relative_path
        if not source.is_file():
            errors.append(f"source_missing:{name}")
            continue
        if source.stat().st_size != int(item.get("bytes", source.stat().st_size)):
            errors.append(f"source_size_mismatch:{name}")
        if _sha256(source) != str(item["sha256"]):
            errors.append(f"source_sha256_mismatch:{name}")
    return entries, errors


def _entry_by_source(
    entries: dict[str, dict[str, Any]], source: str
) -> dict[str, Any] | None:
    if source in entries:
        return entries[source]
    return next((item for item in entries.values() if item.get("_source") == source), None)


def _ulog_entry(run_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    association = run_root / "flight-log-association.json"
    if not association.is_file():
        return None, []
    try:
        value = _read_json(association)
        session = str(value["session"])
        path = run_root / "raw" / "pixhawk" / f"{session}_log100.ulg"
        if not path.is_file():
            return value, ["ulog_materialized_file_missing"]
        if path.stat().st_size != int(value["bytes"]):
            return value, ["ulog_size_mismatch"]
        if _sha256(path) != str(value["archive_sha256"]):
            return value, ["ulog_sha256_mismatch"]
        return value, []
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"ulog_association_invalid:{exc}"]


def _float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result


def inspect_csv(path: Path, timestamp_candidates: Iterable[str]) -> CsvQc:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        first = stream.readline()
        while first.startswith("#"):
            first = stream.readline()
        if not first:
            return CsvQc(str(path), 0, 0, None, None, None, 0, 0, None, 0, 0, 0, 0, 0)
        reader = csv.DictReader([first, *stream])
        fields = reader.fieldnames or []
        timestamp_column = next((name for name in timestamp_candidates if name in fields), None)
        rows = reversals = duplicates = nan_cells = nonfinite = saturated = ack_errors = 0
        parse_errors = 0
        first_timestamp: float | None = None
        last_timestamp: float | None = None
        previous: float | None = None
        for row in reader:
            rows += 1
            for value in row.values():
                if value is None:
                    parse_errors += 1
                    continue
                number = _float(value)
                if value.strip().lower() == "nan":
                    nan_cells += 1
                elif number is not None and not math.isfinite(number):
                    nonfinite += 1
                # Non-finite values are reported by the dedicated counter;
                # do not double-count +/-inf as a finite sensor saturation.
                if number is not None and math.isfinite(number) and abs(number) >= 1e30:
                    saturated += 1
            if timestamp_column is None:
                continue
            if "ack_status" in row:
                ack = str(row.get("ack_status", "")).strip().upper()
                # Protocol enum: 0=ACCEPTED and 1=RELEASED are both successful.
                if ack and ack not in {"0", "1", "OK", "ACCEPTED", "RELEASED", "ACK_OK"}:
                    ack_errors += 1
            timestamp = _float(row.get(timestamp_column, ""))
            if timestamp is None:
                parse_errors += 1
                continue
            if first_timestamp is None:
                first_timestamp = timestamp
            if previous is not None:
                if timestamp < previous:
                    reversals += 1
                elif timestamp == previous:
                    duplicates += 1
            previous = timestamp
            last_timestamp = timestamp
    rate = None
    if first_timestamp is not None and last_timestamp is not None and last_timestamp > first_timestamp:
        scale = 1e9 if timestamp_column and timestamp_column.endswith("_ns") else 1e6
        rate = (rows - 1) * scale / (last_timestamp - first_timestamp)
    return CsvQc(
        str(path), rows, len(fields), timestamp_column, first_timestamp, last_timestamp,
        reversals, duplicates, rate, nan_cells, nonfinite, saturated, ack_errors, parse_errors,
    )


def _clock_brackets(final: dict[str, Any] | None) -> list[dict[str, Any]]:
    if final is None:
        return []
    summary = final.get("summary") or {}
    alignment = summary.get("clock_alignment") or {}
    brackets = alignment.get("brackets") or []
    return [item for item in brackets if isinstance(item, dict)]


def _pi_to_pc_epoch_ns(monotonic_ns: int, brackets: list[dict[str, Any]]) -> int:
    points = []
    for item in brackets:
        pi_mono = int(item["pi_monotonic_ns"])
        correction = int(item["pi_realtime_minus_monotonic_ns"]) + int(
            item["pc_minus_pi_realtime_ns"]
        )
        points.append((pi_mono, correction))
    if not points:
        raise ValueError("clock_alignment_brackets_missing")
    points.sort()
    if len(points) == 1 or points[-1][0] == points[0][0]:
        return monotonic_ns + points[0][1]
    left, right = points[0], points[-1]
    fraction = (monotonic_ns - left[0]) / (right[0] - left[0])
    correction = round(left[1] + fraction * (right[1] - left[1]))
    return monotonic_ns + correction


def _write_aligned_acquisition(source: Path, destination: Path) -> dict[str, Any]:
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as src:
        reader = csv.DictReader(src)
        fields = reader.fieldnames or []
        if "Trigger PC us" not in fields:
            raise ValueError("acquisition_pc_epoch_column_missing")
        rows = []
        for row in reader:
            rows.append({"pc_epoch_us": row["Trigger PC us"], **row})
    text = _csv_text(["pc_epoch_us", *fields], rows)
    _write_exact(destination, text.encode("utf-8"))
    return {"rows": len(rows), "time_basis": "windows_monotonic_anchored_epoch_us"}


def _write_aligned_pi(
    source: Path, destination: Path, brackets: list[dict[str, Any]]
) -> dict[str, Any]:
    comments: list[str] = []
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as src:
        first = src.readline()
        while first.startswith("#"):
            comments.append(first.rstrip("\r\n"))
            first = src.readline()
        reader = csv.DictReader([first, *src])
        fields = reader.fieldnames or []
        clock_field = "sent_ns" if "sent_ns" in fields else "scheduled_ns"
        if clock_field not in fields:
            raise ValueError("pi_monotonic_timestamp_missing")
        rows = []
        for row in reader:
            epoch_ns = _pi_to_pc_epoch_ns(int(row[clock_field]), brackets)
            rows.append({"pc_epoch_us": str(epoch_ns // 1000), **row})
    text = _csv_text(["pc_epoch_us", *fields], rows, comments)
    _write_exact(destination, text.encode("utf-8"))
    return {
        "rows": len(rows),
        "time_basis": "pi_monotonic_ns_to_pc_epoch_us_linear_bracket",
        "source_clock_column": clock_field,
        "hardware_synchronized": False,
    }


def _csv_text(fields: list[str], rows: list[dict[str, Any]], comments: list[str] | None = None) -> str:
    import io

    output = io.StringIO(newline="")
    for comment in comments or []:
        output.write(comment + "\n")
    writer = csv.DictWriter(
        output, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _required_sources(run_root: Path, final: dict[str, Any] | None) -> set[str]:
    if not run_root.name.startswith(("E", "O")):
        return set()
    required = {"fftgyro_acquisition"}
    if final is None:
        return required
    experiment = ((final.get("summary") or {}).get("experiment") or {})
    if str(experiment.get("mode", "")) == "executor":
        required.add("pi_executor")
    return required


def _conditions(final: dict[str, Any] | None) -> dict[str, Any]:
    if final is None:
        return {}
    experiment = ((final.get("summary") or {}).get("experiment") or {})
    keys = (
        "number", "condition_id", "attempt", "name", "mode", "day", "block",
        "priority", "dataset_role", "apparatus", "pitch_deg", "beta_deg",
        "wind_speed_mps", "repeat", "executor_kind",
    )
    return {key: experiment.get(key) for key in keys}


def prepare_run(run_root: Path, *, verify_only: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    status, final, issues = _status(run_root)
    entries, manifest_issues = _manifest_entries(run_root)
    ulog, ulog_issues = _ulog_entry(run_root)
    integrity_issues = [*manifest_issues, *ulog_issues]
    issues.extend(integrity_issues)
    required = _required_sources(run_root, final)
    missing_required = sorted(
        source for source in required if _entry_by_source(entries, source) is None
    )
    issues.extend(f"required_source_missing:{source}" for source in missing_required)

    qc_files: list[dict[str, Any]] = []
    acquisition = _entry_by_source(entries, "fftgyro_acquisition")
    if acquisition:
        qc_files.append(asdict(inspect_csv(run_root / acquisition["path"], ("Trigger PC us",))))
    pi = _entry_by_source(entries, "pi_executor")
    if pi:
        qc_files.append(asdict(inspect_csv(run_root / pi["path"], ("sent_ns", "scheduled_ns"))))
    for item in qc_files:
        if item["rows"] == 0:
            issues.append(f"csv_empty:{Path(item['path']).name}")
        if item["timestamp_reversals"]:
            issues.append(f"timestamp_reversals:{Path(item['path']).name}")
        if item["parse_errors"]:
            issues.append(f"csv_parse_errors:{Path(item['path']).name}")
        if item["nan_cells"] or item["nonfinite_cells"]:
            issues.append(f"nonfinite_values:{Path(item['path']).name}")
        if item["saturated_cells"]:
            issues.append(f"numeric_saturation:{Path(item['path']).name}")
        if item["ack_error_rows"]:
            issues.append(f"ack_errors:{Path(item['path']).name}")
        minimum_rate = 50.0 if Path(item["path"]).name == "executor.csv" else 500.0
        if item["timestamp_rate_hz"] is not None and item["timestamp_rate_hz"] < minimum_rate:
            issues.append(f"sample_rate_low:{Path(item['path']).name}")

    brackets = _clock_brackets(final)
    sync_status = "pc_epoch_native" if acquisition else "not_available"
    if pi:
        sync_status = "approximate_software_clock" if brackets else "unresolved"
        if not brackets:
            issues.append("pi_clock_alignment_missing")
    if ulog:
        issues.append("ulog_common_epoch_not_materialized")

    if integrity_issues or not entries:
        decision = "rejected"
    elif status not in VALID_FINAL_STATUSES:
        decision = "reference_only"
    elif missing_required:
        decision = "rejected"
    elif (
        sync_status == "unresolved"
        or ulog is not None
        or any("calibration" in issue for issue in issues)
    ):
        decision = "needs_review"
    else:
        decision = "accepted"

    processed_outputs: dict[str, Any] = {}
    if not verify_only and decision != "rejected":
        processed = run_root / "processed"
        if acquisition:
            try:
                processed_outputs["fftgyro_acquisition"] = _write_aligned_acquisition(
                    run_root / acquisition["path"], processed / "acquisition.pc-epoch.csv"
                )
            except (OSError, ValueError) as exc:
                issues.append(f"acquisition_alignment_failed:{exc}")
                decision = "needs_review"
        if pi and brackets:
            try:
                processed_outputs["pi_executor"] = _write_aligned_pi(
                    run_root / pi["path"], processed / "executor.pc-epoch.csv", brackets
                )
            except (OSError, ValueError) as exc:
                issues.append(f"pi_alignment_failed:{exc}")
                decision = "needs_review"
        processing_manifest = {
            "schema_version": PROCESSING_VERSION,
            "run_id": run_root.name,
            "decision": decision,
            "raw_immutable": True,
            "coordinate_system": "not_transformed; source-native coordinates retained",
            "calibration": "source acquisition calibration retained; no inferred calibration applied",
            "sync_status": sync_status,
            "outputs": processed_outputs,
            "issues": sorted(set(issues)),
        }
        _write_exact(processed / "processing-manifest.json", _json_bytes(processing_manifest))

    conditions = _conditions(final)
    catalog = {
        "run_id": run_root.name,
        "status": status,
        "decision": decision,
        **conditions,
        "fftgyro_acquisition": acquisition is not None,
        "pi_executor": pi is not None,
        "px4_ulog": ulog is not None and not ulog_issues,
        "source_count": len(entries) + (1 if ulog else 0),
        "source_bytes": sum(int(item.get("bytes", 0)) for item in entries.values())
        + (int(ulog.get("bytes", 0)) if ulog else 0),
        "sync_status": sync_status,
        "issue_count": len(set(issues)),
        "issues": sorted(set(issues)),
    }
    qc = {
        "run_id": run_root.name,
        "status": status,
        "decision": decision,
        "required_sources": sorted(required),
        "missing_required_sources": missing_required,
        "integrity_ok": not integrity_issues,
        "sync_status": sync_status,
        "files": qc_files,
        "issues": sorted(set(issues)),
    }
    return catalog, qc


def prepare_analysis(
    runs_root: Path,
    output_root: Path,
    *,
    run_id: str | None = None,
    verify_only: bool = False,
) -> dict[str, Any]:
    if not runs_root.is_dir():
        raise ConfigurationError(f"runs root is not a directory: {runs_root}")
    if run_id:
        selected = [runs_root / run_id]
        if not selected[0].is_dir():
            raise ConfigurationError(f"run does not exist: {run_id}")
    else:
        selected = sorted(path for path in runs_root.iterdir() if path.is_dir())
    catalogs: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for run_root in selected:
        catalog, qc = prepare_run(run_root, verify_only=verify_only)
        catalogs.append(catalog)
        quality.append(qc)

    _write_exact(output_root / "run-catalog.json", _json_bytes(catalogs))
    _write_exact(output_root / "qc-summary.json", _json_bytes(quality))
    _write_catalog_csv(output_root / "run-catalog.csv", catalogs)
    _write_qc_csv(output_root / "qc-summary.csv", quality)
    groups = {
        "accepted": [item["run_id"] for item in catalogs if item["decision"] == "accepted"],
        "reference": [
            item["run_id"] for item in catalogs if item["decision"] == "reference_only"
        ],
        "rejected": [item["run_id"] for item in catalogs if item["decision"] == "rejected"],
        "needs-review": [
            item["run_id"] for item in catalogs if item["decision"] == "needs_review"
        ],
    }
    for name, values in groups.items():
        _write_exact(output_root / f"{name}-runs.txt", ("\n".join(values) + ("\n" if values else "")).encode())
    counts = Counter(item["decision"] for item in catalogs)
    summary = {
        "schema_version": PROCESSING_VERSION,
        "runs_root": str(runs_root),
        "run_count": len(catalogs),
        "decisions": dict(sorted(counts.items())),
        "verify_only": verify_only,
        "integrity_failures": sum(not item["integrity_ok"] for item in quality),
    }
    _write_readme(
        output_root / "README.md",
        _readme(summary, quality).encode("utf-8"),
        verify_only=verify_only,
    )
    return summary


def _write_catalog_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_id", "status", "decision", "condition_id", "number", "attempt", "name",
        "mode", "day", "block", "priority", "dataset_role", "apparatus", "pitch_deg",
        "beta_deg", "wind_speed_mps", "repeat", "executor_kind", "fftgyro_acquisition",
        "pi_executor", "px4_ulog", "source_count", "source_bytes", "sync_status",
        "issue_count", "issues",
    ]
    normalized = [{**row, "issues": ";".join(row["issues"])} for row in rows]
    _write_exact(path, _csv_text(fields, normalized).encode("utf-8"))


def _write_qc_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_id", "status", "decision", "integrity_ok", "sync_status",
        "required_sources", "missing_required_sources", "issues",
    ]
    normalized = [
        {
            **row,
            "required_sources": ";".join(row["required_sources"]),
            "missing_required_sources": ";".join(row["missing_required_sources"]),
            "issues": ";".join(row["issues"]),
        }
        for row in rows
    ]
    _write_exact(path, _csv_text(fields, normalized).encode("utf-8"))


def _readme(summary: dict[str, Any], quality: list[dict[str, Any]]) -> str:
    issue_counts = Counter(issue for row in quality for issue in row["issues"])
    lines = [
        "# FFTGYRO analysis preflight preparation",
        "",
        "Generated deterministically from immutable run raw data.",
        "",
        f"- Runs: {summary['run_count']}",
        f"- Decisions: {json.dumps(summary['decisions'], ensure_ascii=False, sort_keys=True)}",
        f"- Integrity failures: {summary['integrity_failures']}",
        f"- Verify only: {summary['verify_only']}",
        "",
        "## Exclusion and review reasons",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in issue_counts.most_common())
    lines.extend(
        [
            "",
            "`accepted` is the primary-analysis candidate set. `reference_only` retains failed or",
            "aborted attempts with usable data. `rejected` lacks trustworthy required raw data.",
            "`needs_review` requires a manual synchronization or calibration decision.",
            "",
        ]
    )
    return "\n".join(lines)
