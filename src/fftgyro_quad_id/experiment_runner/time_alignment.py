from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, tzinfo
from pathlib import Path


HIOKI_TIME = re.compile(r"^'?(?P<stamp>\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)s?$")


@dataclass(frozen=True, slots=True)
class CurrentAlignment:
    reference: str
    status: str
    current_column: str
    threshold_a: float
    sample_period_s: float
    hioki_current_on_utc: str
    hioki_current_off_utc: str
    host_command_on_utc: str
    host_command_off_utc: str
    host_to_hioki_offset_s: float
    observed_current_duration_s: float
    ideal_step_duration_s: float
    duration_residual_s: float
    estimated_sync_uncertainty_s: float
    phase_analysis_allowed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def align_hioki_current(
    hioki_csv: Path,
    timeline_json: Path,
    *,
    current_column: str | None = None,
    minimum_rise_a: float = 0.1,
) -> CurrentAlignment:
    events = json.loads(timeline_json.read_text(encoding="utf-8"))["events"]
    command_on = _one_event(events, "COMMAND_ON")
    command_off = _one_event(events, "COMMAND_OFF")
    host_on = datetime.fromisoformat(str(command_on["host_utc"]))
    host_off = datetime.fromisoformat(str(command_off["host_utc"]))
    times, currents, selected_column = read_hioki_absolute_current(
        hioki_csv, current_column=current_column, timezone=host_on.tzinfo
    )
    intervals = [(right - left).total_seconds() for left, right in zip(times, times[1:])]
    sample_period = statistics.median(intervals)
    baseline = statistics.median(currents)
    mad = statistics.median(abs(value - baseline) for value in currents)
    threshold = baseline + max(minimum_rise_a, 8.0 * 1.4826 * mad)
    start_index, end_index = _strongest_current_region(
        currents,
        threshold=threshold,
        maximum_gap_samples=max(1, math.ceil(0.03 / sample_period)),
        minimum_samples=max(2, math.ceil(0.05 / sample_period)),
    )
    current_on = times[start_index]
    current_off = times[end_index]
    observed_duration = (current_off - current_on).total_seconds()
    ideal_duration = (host_off - host_on).total_seconds()
    residual = observed_duration - ideal_duration
    # This is an offline threshold crossing, not shared hardware sync. The
    # uncertainty includes one logger sample plus half the ON/OFF duration mismatch.
    uncertainty = sample_period + abs(residual) / 2.0
    return CurrentAlignment(
        reference="hioki_current_onset",
        status="approximate_offline_alignment",
        current_column=selected_column,
        threshold_a=threshold,
        sample_period_s=sample_period,
        hioki_current_on_utc=current_on.isoformat(),
        hioki_current_off_utc=current_off.isoformat(),
        host_command_on_utc=host_on.isoformat(),
        host_command_off_utc=host_off.isoformat(),
        host_to_hioki_offset_s=(current_on - host_on).total_seconds(),
        observed_current_duration_s=observed_duration,
        ideal_step_duration_s=ideal_duration,
        duration_residual_s=residual,
        estimated_sync_uncertainty_s=uncertainty,
        phase_analysis_allowed=False,
    )


def read_hioki_absolute_current(
    path: Path,
    *,
    current_column: str | None,
    timezone: tzinfo | None,
) -> tuple[list[datetime], list[float], str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Logger Utility's Japanese Windows build exports CP932 by default.
        text = raw.decode("cp932")
    rows = list(csv.reader(text.splitlines()))
    header_index = next(
        (index for index, row in enumerate(rows) if row and row[0].strip() == "Time"),
        None,
    )
    if header_index is None:
        raise ValueError("hioki_time_header_missing")
    header = [cell.strip() for cell in rows[header_index]]
    if current_column is None:
        candidates = [name for name in header if "[A]" in name or name.lower() == "current_a"]
        if len(candidates) != 1:
            raise ValueError(f"hioki_current_column_ambiguous:{candidates}")
        current_column = candidates[0]
    try:
        current_index = header.index(current_column)
    except ValueError as exc:
        raise ValueError(f"hioki_current_column_missing:{current_column}") from exc
    parsed_times: list[datetime] = []
    currents: list[float] = []
    for line_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if not row or not row[0].strip():
            continue
        match = HIOKI_TIME.match(row[0].strip())
        if match is None:
            raise ValueError(f"hioki_absolute_time_invalid:line={line_number}:{row[0]}")
        stamp = match.group("stamp")
        time_format = "%y-%m-%d %H:%M:%S.%f" if "." in stamp else "%y-%m-%d %H:%M:%S"
        parsed_times.append(datetime.strptime(stamp, time_format).replace(tzinfo=timezone))
        currents.append(float(row[current_index]))
    if len(parsed_times) < 3:
        raise ValueError("hioki_fewer_than_three_current_samples")
    return parsed_times, currents, current_column


def _strongest_current_region(
    values: list[float],
    *,
    threshold: float,
    maximum_gap_samples: int,
    minimum_samples: int,
) -> tuple[int, int]:
    active = [index for index, value in enumerate(values) if value >= threshold]
    if not active:
        raise ValueError("hioki_current_onset_not_found")
    regions: list[tuple[int, int]] = []
    start = previous = active[0]
    for index in active[1:]:
        if index - previous > maximum_gap_samples + 1:
            regions.append((start, previous))
            start = index
        previous = index
    regions.append((start, previous))
    candidates = [region for region in regions if region[1] - region[0] + 1 >= minimum_samples]
    if not candidates:
        raise ValueError("hioki_current_region_too_short")
    return max(
        candidates,
        key=lambda region: sum(max(0.0, value - threshold) for value in values[region[0] : region[1] + 1]),
    )


def _one_event(events: list[dict[str, object]], name: str) -> dict[str, object]:
    matching = [event for event in events if event.get("name") == name]
    if len(matching) != 1:
        raise ValueError(f"timeline_requires_exactly_one_{name}:count={len(matching)}")
    return matching[0]
