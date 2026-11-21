from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from fftgyro_quad_id.experiment_runner.analysis_preparation import (
    DataIntegrityError,
    inspect_csv,
    prepare_analysis,
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _source(run: Path, relative: str, content: str) -> dict[str, object]:
    path = run / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    raw = path.read_bytes()
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _final(status: str, *, mode: str = "capture_only", brackets: bool = False) -> dict[str, object]:
    alignment = {}
    if brackets:
        alignment = {
            "brackets": [
                {
                    "pi_monotonic_ns": 1_000,
                    "pi_realtime_minus_monotonic_ns": 1_000_000,
                    "pc_minus_pi_realtime_ns": 100,
                },
                {
                    "pi_monotonic_ns": 3_000,
                    "pi_realtime_minus_monotonic_ns": 1_000_000,
                    "pc_minus_pi_realtime_ns": 300,
                },
            ]
        }
    return {
        "status": status,
        "summary": {
            "experiment": {"condition_id": "E001", "mode": mode, "name": "test"},
            "clock_alignment": alignment,
        },
    }


def test_prepare_completed_capture_run_and_is_idempotent(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "E001_A01_test"
    run.mkdir(parents=True)
    acquisition = _source(
        run,
        "raw/fftgyro_acquisition/acquisition.csv",
        "Trigger PC us,value\n1000000,1\n1010000,2\n",
    )
    _json(run / "source_manifest.json", {"fftgyro_acquisition": acquisition})
    _json(run / "final.json", _final("completed"))
    output = tmp_path / "analysis"

    first = prepare_analysis(runs, output)
    second = prepare_analysis(runs, output)

    assert first == second
    assert first["decisions"] == {"accepted": 1}
    with (run / "processed/acquisition.pc-epoch.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["pc_epoch_us"] for row in rows] == ["1000000", "1010000"]


def test_verify_only_accepts_normal_output_without_changing_readme(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "E001_A01_test"
    run.mkdir(parents=True)
    acquisition = _source(
        run,
        "raw/fftgyro_acquisition/acquisition.csv",
        "Trigger PC us,value\n1000000,1\n1010000,2\n",
    )
    _json(run / "source_manifest.json", {"fftgyro_acquisition": acquisition})
    _json(run / "final.json", _final("completed"))
    output = tmp_path / "analysis"

    prepare_analysis(runs, output)
    readme = output / "README.md"
    original = readme.read_bytes()
    result = prepare_analysis(runs, output, verify_only=True)

    assert result["verify_only"] is True
    assert readme.read_bytes() == original
    assert b"- Verify only: False\n" in original


def test_verify_only_rejects_genuinely_different_readme(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "E001_A01_test"
    run.mkdir(parents=True)
    acquisition = _source(
        run,
        "raw/fftgyro_acquisition/acquisition.csv",
        "Trigger PC us,value\n1000000,1\n1010000,2\n",
    )
    _json(run / "source_manifest.json", {"fftgyro_acquisition": acquisition})
    _json(run / "final.json", _final("completed"))
    output = tmp_path / "analysis"

    prepare_analysis(runs, output)
    readme = output / "README.md"
    tampered = readme.read_bytes().replace(
        b"Generated deterministically", b"Changed unexpectedly"
    )
    readme.write_bytes(tampered)

    with pytest.raises(DataIntegrityError):
        prepare_analysis(runs, output, verify_only=True)
    assert readme.read_bytes() == tampered


def test_pi_clock_is_converted_and_failed_run_is_reference(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "E001_A01_test"
    run.mkdir(parents=True)
    acquisition = _source(
        run, "raw/fftgyro_acquisition/acquisition.csv", "Trigger PC us,value\n1,1\n2,2\n"
    )
    pi = _source(
        run,
        "raw/pi_executor/executor.csv",
        "# run_id=1\nsequence,sent_ns,value\n0,1000,1\n1,3000,2\n",
    )
    _json(
        run / "source_manifest.json",
        {"fftgyro_acquisition": acquisition, "pi_executor": pi},
    )
    _json(run / "final.json", _final("failed", mode="executor", brackets=True))

    result = prepare_analysis(runs, tmp_path / "analysis")

    assert result["decisions"] == {"reference_only": 1}
    with (run / "processed/executor.pc-epoch.csv").open(newline="") as stream:
        rows = list(csv.DictReader(row for row in stream if not row.startswith("#")))
    # 1,001,100 ns and 1,003,300 ns are deliberately floored to integer us.
    assert [row["pc_epoch_us"] for row in rows] == ["1001", "1003"]


def test_empty_and_no_final_runs_are_rejected_or_reference(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    empty = runs / "empty"
    empty.mkdir(parents=True)
    failed = runs / "failed"
    failed.mkdir()
    acquisition = _source(
        failed, "raw/fftgyro_acquisition/acquisition.csv", "Trigger PC us,value\n1,1\n2,2\n"
    )
    _json(failed / "source_manifest.json", {"fftgyro_acquisition": acquisition})
    _json(failed / "final.json", _final("aborted"))

    result = prepare_analysis(runs, tmp_path / "analysis", verify_only=True)

    assert result["decisions"] == {"reference_only": 1, "rejected": 1}
    assert not (failed / "processed").exists()


def test_csv_qc_detects_timestamp_and_numeric_problems(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "sent_ns,value\n1000,1\n1000,nan\n900,inf\n800,1e31\n",
        encoding="utf-8",
    )

    result = inspect_csv(path, ("sent_ns",))

    assert result.timestamp_duplicates == 1
    assert result.timestamp_reversals == 2
    assert result.nan_cells == 1
    assert result.nonfinite_cells == 1
    assert result.saturated_cells == 1


def test_csv_qc_accepts_protocol_success_ack_values(tmp_path: Path) -> None:
    path = tmp_path / "ack.csv"
    path.write_text(
        "sent_ns,ack_status\n1000,0\n2000,1\n3000,-1\n4000,2\n",
        encoding="utf-8",
    )

    result = inspect_csv(path, ("sent_ns",))

    assert result.ack_error_rows == 2


def test_different_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "E001"
    run.mkdir(parents=True)
    acquisition = _source(
        run, "raw/fftgyro_acquisition/acquisition.csv", "Trigger PC us,value\n1,1\n2,2\n"
    )
    _json(run / "source_manifest.json", {"fftgyro_acquisition": acquisition})
    _json(run / "final.json", _final("completed"))
    output = tmp_path / "analysis"
    output.mkdir()
    (output / "run-catalog.json").write_text("different", encoding="utf-8")

    with pytest.raises(DataIntegrityError):
        prepare_analysis(runs, output)


def test_verified_but_unaligned_ulog_requires_review(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "E001"
    run.mkdir(parents=True)
    acquisition = _source(
        run, "raw/fftgyro_acquisition/acquisition.csv", "Trigger PC us,value\n1,1\n2,2\n"
    )
    _json(run / "source_manifest.json", {"fftgyro_acquisition": acquisition})
    _json(run / "final.json", _final("completed"))
    ulog = run / "raw/pixhawk/sess1_log100.ulg"
    ulog.parent.mkdir(parents=True)
    ulog.write_bytes(b"ulog")
    digest = hashlib.sha256(b"ulog").hexdigest()
    _json(
        run / "flight-log-association.json",
        {"session": "sess1", "bytes": 4, "archive_sha256": digest},
    )

    result = prepare_analysis(runs, tmp_path / "analysis", verify_only=True)

    assert result["decisions"] == {"needs_review": 1}
