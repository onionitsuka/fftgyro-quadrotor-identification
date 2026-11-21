from __future__ import annotations

import csv
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CHANNELS = ("fx", "fy", "fz", "mx", "my", "mz")
EXPECTED_HEADER = (
    "経過時間[sec]",
    "Fx[N]",
    "Fy[N]",
    "Fz[N]",
    "Mx[Nm]",
    "My[Nm]",
    "Mz[Nm]",
)


class LgrFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LgrMetadata:
    sample_rate_hz: float
    filter_setting: str
    ratings: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class LgrSample:
    elapsed_s: float
    fx_n: float
    fy_n: float
    fz_n: float
    mx_nm: float
    my_nm: float
    mz_nm: float

    @property
    def values(self) -> tuple[float, float, float, float, float, float]:
        return (self.fx_n, self.fy_n, self.fz_n, self.mx_nm, self.my_nm, self.mz_nm)


@dataclass(frozen=True, slots=True)
class LgrLog:
    metadata: LgrMetadata
    samples: tuple[LgrSample, ...]


@dataclass(frozen=True, slots=True)
class LgrQualityReport:
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    row_count: int
    duration_s: float
    nominal_sample_rate_hz: float
    observed_sample_rate_hz: float | None
    estimated_missing_samples: int
    filter_setting: str
    ratings: dict[str, float]
    peak_abs: dict[str, float]
    peak_utilization: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise LgrFormatError(f"{context}:not_a_number:{value!r}") from exc
    if not math.isfinite(parsed):
        raise LgrFormatError(f"{context}:not_finite")
    return parsed


def read_lgr101_csv(path: Path) -> LgrLog:
    try:
        text = path.read_bytes().decode("cp932")
    except UnicodeDecodeError as exc:
        raise LgrFormatError("file_is_not_valid_cp932") from exc

    rows = list(csv.reader(text.splitlines()))
    if len(rows) < 5:
        raise LgrFormatError("file_too_short")
    if rows[0] != ["CFSLGR_DataFile"]:
        raise LgrFormatError("magic_header_mismatch")
    if len(rows[1]) < 5 or rows[1][0] != "データ保存[個/秒]":
        raise LgrFormatError("sample_rate_metadata_missing")
    sample_rate_hz = _float(rows[1][1], "sample_rate_hz")
    if sample_rate_hz <= 0:
        raise LgrFormatError("sample_rate_must_be_positive")
    if rows[1][3] != "Filter":
        raise LgrFormatError("filter_metadata_missing")
    filter_setting = rows[1][4].strip()

    if len(rows[2]) != 7 or rows[2][0] != "定格":
        raise LgrFormatError("rating_metadata_missing")
    ratings = tuple(_float(value, "rating") for value in rows[2][1:])
    if len(ratings) != 6 or any(value <= 0 for value in ratings):
        raise LgrFormatError("ratings_must_be_six_positive_values")
    if tuple(rows[3]) != EXPECTED_HEADER:
        raise LgrFormatError("data_header_mismatch")

    samples: list[LgrSample] = []
    previous_time: float | None = None
    for line_number, row in enumerate(rows[4:], start=5):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != 7:
            raise LgrFormatError(f"line_{line_number}:expected_7_columns")
        values = tuple(_float(value, f"line_{line_number}") for value in row)
        elapsed_s = values[0]
        if elapsed_s < 0:
            raise LgrFormatError(f"line_{line_number}:negative_elapsed_time")
        if previous_time is not None and elapsed_s <= previous_time:
            raise LgrFormatError(f"line_{line_number}:time_not_strictly_increasing")
        previous_time = elapsed_s
        samples.append(LgrSample(*values))
    if len(samples) < 2:
        raise LgrFormatError("fewer_than_two_data_rows")

    return LgrLog(
        metadata=LgrMetadata(
            sample_rate_hz=sample_rate_hz,
            filter_setting=filter_setting,
            ratings=ratings,  # type: ignore[arg-type]
        ),
        samples=tuple(samples),
    )


def inspect_lgr_log(
    log: LgrLog,
    *,
    required_sample_rate_hz: float,
    utilization_limit: float = 0.80,
) -> LgrQualityReport:
    if required_sample_rate_hz <= 0:
        raise ValueError("required_sample_rate_hz must be positive")
    if not 0 < utilization_limit <= 1:
        raise ValueError("utilization_limit must be in (0, 1]")

    blockers: list[str] = []
    warnings: list[str] = []
    metadata = log.metadata
    samples = log.samples
    deltas = tuple(
        current.elapsed_s - previous.elapsed_s
        for previous, current in zip(samples, samples[1:])
    )
    median_delta = statistics.median(deltas)
    observed_rate = 1.0 / median_delta if median_delta > 0 else None
    expected_delta = 1.0 / metadata.sample_rate_hz
    missing = sum(max(0, round(delta / expected_delta) - 1) for delta in deltas)

    if metadata.sample_rate_hz < required_sample_rate_hz:
        blockers.append("nominal_sample_rate_below_protocol_requirement")
    if observed_rate is None or observed_rate < required_sample_rate_hz * 0.99:
        blockers.append("observed_sample_rate_below_protocol_requirement")
    if observed_rate is not None:
        mismatch = abs(observed_rate - metadata.sample_rate_hz) / metadata.sample_rate_hz
        if mismatch > 0.01:
            blockers.append("observed_rate_differs_from_file_metadata_above_1_percent")
    if missing:
        blockers.append("sample_gaps_detected")

    peaks = tuple(max(abs(sample.values[index]) for sample in samples) for index in range(6))
    utilizations = tuple(peak / rating for peak, rating in zip(peaks, metadata.ratings))
    peak_utilization = max(utilizations)
    if peak_utilization >= utilization_limit:
        blockers.append("loadcell_at_or_above_80_percent")
    if metadata.filter_setting.upper() != "OFF":
        warnings.append("logger_low_pass_filter_enabled_record_setting_in_manifest")

    return LgrQualityReport(
        ready=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        row_count=len(samples),
        duration_s=samples[-1].elapsed_s - samples[0].elapsed_s,
        nominal_sample_rate_hz=metadata.sample_rate_hz,
        observed_sample_rate_hz=observed_rate,
        estimated_missing_samples=missing,
        filter_setting=metadata.filter_setting,
        ratings=dict(zip(CHANNELS, metadata.ratings)),
        peak_abs=dict(zip(CHANNELS, peaks)),
        peak_utilization=peak_utilization,
    )


def inspect_lgr_file(
    path: Path,
    *,
    required_sample_rate_hz: float,
    utilization_limit: float = 0.80,
) -> LgrQualityReport:
    return inspect_lgr_log(
        read_lgr101_csv(path),
        required_sample_rate_hz=required_sample_rate_hz,
        utilization_limit=utilization_limit,
    )
