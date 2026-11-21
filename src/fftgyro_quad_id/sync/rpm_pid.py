"""Offline Nano -> Pi steady-clock alignment; no hardware access.

Q precedes request transmission; T carries Nano time at response creation.
Thus its mapped time lies within [Q.host_ns, T.host_ns], allowing explicit
timestamp slack. Bounds below are conditional on an affine clock model,
correct firmware timestamps and SAME Pi boot, not an accuracy certificate.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

MODULUS = 1 << 32


@dataclass(frozen=True)
class Event:
    host_ns: int
    kind: str
    capture_id: int
    sequence: int
    nano_us: int
    value: int
    unwrapped_us: int | None


@dataclass(frozen=True)
class Exchange:
    token: int
    nano_us: int
    sent_ns: int
    received_ns: int


def read_capture(path: Path, *, unanswered: list[tuple[int, int]] | None = None) -> tuple[list[Event], list[Exchange]]:
    """Validate one complete capture; never turn queued old edges into wraps."""
    events: list[Event] = []
    exchanges: list[Exchange] = []
    pending: dict[int, int] = {}
    tokens: set[int] = set()
    capture = None
    expected = 0
    high_water: int | None = None
    previous_host = -1
    ended = False
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            host, raw, value, seq, identity = (
                int(row[key])
                for key in ("host_ns", "nano_us", "value", "sequence", "capture_id")
            )
            kind = row["kind"]
            if ended or kind not in "BQTHEZ" or len(kind) != 1:
                raise ValueError("invalid kind or data after END")
            if host < 0 or host < previous_host or not 0 <= raw < MODULUS:
                raise ValueError("invalid/nonmonotonic timestamp")
            if previous_host >= 0 and host - previous_host >= (MODULUS // 2) * 1000:
                raise ValueError("gap too long for unambiguous 32-bit unwrapping")
            previous_host = host
            if capture is None:
                if kind != "B" or not 0 < identity < MODULUS:
                    raise ValueError("missing BEGIN/capture id")
                capture = identity
            if identity != capture:
                raise ValueError("capture changed (possible Nano reset)")
            unwrapped = None
            if kind == "Q":
                if value in tokens or seq != 0 or raw != 0:
                    raise ValueError("duplicate/malformed Q token")
                tokens.add(value)
                pending[value] = host
            else:
                if seq != expected:
                    raise ValueError("sequence gap/reset")
                expected = (expected + 1) % MODULUS
                if kind == "B" and events:
                    raise ValueError("duplicate BEGIN")
                if high_water is None:
                    unwrapped = raw
                else:
                    delta = (raw - high_water + MODULUS // 2) % MODULUS - MODULUS // 2
                    unwrapped = high_water + delta
                    if delta < 0 and kind != "E":
                        raise ValueError("non-edge Nano clock moved backwards")
                    if unwrapped < int(events[0].unwrapped_us or 0):
                        raise ValueError("edge predates capture")
                high_water = max(high_water or unwrapped, unwrapped)
                if kind == "E" and value not in (1, 2, 3, 4):
                    raise ValueError("invalid RPM channel")
                if kind in ("H", "Z") and value != 0:
                    raise ValueError("Nano reported dropped edges")
                if kind == "T":
                    if value not in pending:
                        raise ValueError("unmatched/duplicate T token")
                    sent = pending.pop(value)
                    if host <= sent:
                        raise ValueError("nonpositive round-trip time")
                    exchanges.append(Exchange(value, unwrapped, sent, host))
                ended = kind == "Z"
            events.append(Event(host, kind, identity, seq, raw, value, unwrapped))
    if not ended or (pending and unanswered is None):
        raise ValueError("incomplete capture or unanswered Q")
    if unanswered is not None:
        unanswered.extend(sorted(pending.items(), key=lambda pair: pair[1]))
    if len(exchanges) < 3:
        raise ValueError("at least three Q/T pairs required")
    return events, exchanges


@dataclass(frozen=True)
class ClockModel:
    nano_origin_us: int
    pi_origin_ns: int
    slope: float
    offset_s: float
    vertices: tuple[tuple[float, float], ...]
    first_anchor_us: int
    last_anchor_us: int

    def map(self, nano_us: int) -> tuple[int, int, int]:
        x = (nano_us - self.nano_origin_us) * 1e-6
        values = [a * x + b for a, b in self.vertices]
        estimate = self.pi_origin_ns + round((self.slope * x + self.offset_s) * 1e9)
        lower = self.pi_origin_ns + math.floor(min(values) * 1e9)
        upper = self.pi_origin_ns + math.ceil(max(values) * 1e9)
        return estimate, lower, upper


def fit_clock(exchanges: list[Exchange], slack_ns: int = 10_000, *,
              weight_by_rtt: bool = False) -> ClockModel:
    """Fit midpoint regression constrained by ALL Q/T causality intervals.

    Enumerate feasible affine-model vertices so each mapped edge has an
    interval including unknown request/response asymmetry. No outlier is
    silently discarded to obtain a fit. Slack is an assumption, not measured.
    """
    if slack_ns < 0 or len(exchanges) < 3:
        raise ValueError("invalid slack or insufficient exchanges")
    origin = exchanges[0].nano_us
    pi_origin = exchanges[0].sent_ns
    x = [(e.nano_us - origin) * 1e-6 for e in exchanges]
    low = [(e.sent_ns - pi_origin - slack_ns) * 1e-9 for e in exchanges]
    high = [(e.received_ns - pi_origin + slack_ns) * 1e-9 for e in exchanges]
    if any(x[i] <= x[i - 1] for i in range(1, len(x))):
        raise ValueError("Q/T anchors must advance in Nano time")
    slope_low = max(0.0, (low[-1] - high[0]) / x[-1])
    slope_high = (high[-1] - low[0]) / x[-1]
    if slope_low <= 0 or slope_high < slope_low:
        raise ValueError("no positive affine clock satisfies Q/T intervals")

    def intercept_bounds(a: float) -> tuple[float, float]:
        return max(y - a * t for t, y in zip(x, low, strict=True)), min(
            y - a * t for t, y in zip(x, high, strict=True)
        )

    # Clip a convex (slope, intercept) polygon by each Q/T interval. Avoid
    # enumerating all pairs of boundaries for long recordings.
    vertices = [(slope_low, low[0]), (slope_high, low[0]),
                (slope_high, high[0]), (slope_low, high[0])]
    for t, lo, hi in zip(x, low, high, strict=True):
        for sign, bound in ((1, lo), (-1, hi)):
            clipped = []
            if not vertices:
                break
            previous = vertices[-1]
            previous_d = sign * (previous[0] * t + previous[1] - bound)
            for current in vertices:
                current_d = sign * (current[0] * t + current[1] - bound)
                if (current_d >= 0) != (previous_d >= 0):
                    fraction = previous_d / (previous_d - current_d)
                    clipped.append((previous[0] + fraction * (current[0] - previous[0]),
                                    previous[1] + fraction * (current[1] - previous[1])))
                if current_d >= 0:
                    clipped.append(current)
                previous, previous_d = current, current_d
            vertices = clipped
    if not vertices:
        raise ValueError("no affine clock satisfies all Q/T intervals")
    slope_low = min(a for a, _ in vertices)
    slope_high = max(a for a, _ in vertices)
    mid = [(lo + hi) / 2 for lo, hi in zip(low, high, strict=True)]
    rtts = [e.received_ns - e.sent_ns for e in exchanges]
    if any(rtt <= 0 for rtt in rtts):
        raise ValueError("nonpositive round-trip time")
    minimum_rtt = min(rtts)
    weights = [(minimum_rtt / rtt) ** 2 if weight_by_rtt else 1.0 for rtt in rtts]
    total = sum(weights)
    x_mean = sum(w * t for w, t in zip(weights, x, strict=True)) / total
    y_mean = sum(w * y for w, y in zip(weights, mid, strict=True)) / total
    slope = sum(w * (t - x_mean) * (y - y_mean)
                for w, t, y in zip(weights, x, mid, strict=True)) / sum(
        w * (t - x_mean) ** 2 for w, t in zip(weights, x, strict=True)
    )
    slope = min(slope_high, max(slope_low, slope))
    lo, hi = intercept_bounds(slope)
    return ClockModel(origin, pi_origin, slope, (lo + hi) / 2, tuple(vertices), origin,
                      exchanges[-1].nano_us)


def align(rpm_csv: Path, pid_csv: Path, output: Path, *, same_pi_boot: bool,
          slack_ns: int = 10_000) -> dict[str, object]:
    if not same_pi_boot:
        raise ValueError("must confirm both logs use steady_clock on the same Pi boot")
    capture_result = rpm_csv.parent / "result.txt"
    if capture_result.exists():
        result_lines = capture_result.read_text(encoding="utf-8").splitlines()
        if not result_lines or result_lines[0] != "COMPLETE":
            raise ValueError("RPM logger did not report COMPLETE")
    events, exchanges = read_capture(rpm_csv)
    model = fit_clock(exchanges, slack_ns)
    with pid_csv.open(newline="", encoding="utf-8-sig") as stream:
        controls = list(csv.DictReader(stream))
    times = [int(row["sent_ns"]) for row in controls]
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("PID sent_ns must strictly increase, with at least two records")
    mapped = [model.map(int(e.unwrapped_us)) for e in events if e.kind == "E"]
    if not mapped or not any(times[0] <= t <= times[-1] for t, _, _ in mapped):
        raise ValueError("no RPM edge overlap with PID; check log pairing/clock domain")
    output.mkdir(parents=False, exist_ok=False)
    widths = []
    delays = []
    ambiguous = 0
    outside_anchors = 0
    with (output / "rpm_edges_aligned.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("capture_id", "sequence", "channel", "nano_us", "nano_unwrapped_us",
                         "host_received_ns", "pi_estimated_ns", "pi_lower_ns", "pi_upper_ns",
                         "relative_to_pid_start_s", "within_qt_anchors", "pid_row_index",
                         "pid_sent_ns", "pid_cycle_ambiguous"))
        for event in events:
            if event.kind != "E":
                continue
            nano = int(event.unwrapped_us)
            estimate, lower, upper = model.map(nano)
            bracketed = model.first_anchor_us <= nano <= model.last_anchor_us
            index = bisect.bisect_right(times, estimate) - 1
            in_pid = times[0] <= estimate <= times[-1]
            uncertain = (bisect.bisect_right(times, lower) != bisect.bisect_right(times, upper))
            ambiguous += int(in_pid and uncertain)
            outside_anchors += int(not bracketed)
            widths.append((upper - lower) / 1e6)
            delays.append((event.host_ns - estimate) / 1e6)
            writer.writerow((event.capture_id, event.sequence, event.value, event.nano_us, nano,
                             event.host_ns, estimate, lower, upper, (estimate - times[0]) / 1e9,
                             bracketed, index if in_pid else "", times[index] if in_pid else "",
                             uncertain if in_pid else ""))
    with (output / "clock_exchanges.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("token", "nano_unwrapped_us", "q_sent_ns", "t_received_ns",
                         "mapped_ns", "round_trip_ms", "midpoint_residual_ms"))
        for exchange in exchanges:
            estimate, _, _ = model.map(exchange.nano_us)
            writer.writerow((exchange.token, exchange.nano_us, exchange.sent_ns, exchange.received_ns,
                             estimate, (exchange.received_ns - exchange.sent_ns) / 1e6,
                             (estimate - (exchange.sent_ns + exchange.received_ns) / 2) / 1e6))
    summary: dict[str, object] = {
        "rpm_source": str(rpm_csv.resolve()), "pid_source": str(pid_csv.resolve()),
        "same_pi_boot_confirmed": True, "status": "ALIGNED_CONDITIONAL_NOT_QUALIFIED",
        "clock": "Pi std::chrono::steady_clock nanoseconds, NOT wall time/PX4 boot time",
        "timestamp_slack_ns_assumed": slack_ns, "capture_id": events[0].capture_id,
        "rpm_result_file_checked": capture_result.exists(),
        "exchange_count": len(exchanges), "edge_count": len(mapped),
        "nano_origin_us": model.nano_origin_us, "pi_origin_ns": model.pi_origin_ns,
        "scale_pi_seconds_per_nano_second": model.slope, "offset_at_origin_s": model.offset_s,
        "scale_correction_ppm": (model.slope - 1) * 1e6,
        "scale_correction_ppm_range": [(min(a for a, _ in model.vertices) - 1) * 1e6,
                                       (max(a for a, _ in model.vertices) - 1) * 1e6],
        "round_trip_ms_median": statistics.median(
            (e.received_ns - e.sent_ns) / 1e6 for e in exchanges),
        "edge_time_interval_width_ms_max": max(widths),
        "host_receive_minus_edge_estimate_ms_median": statistics.median(delays),
        "pid_cycle_ambiguous_edges": ambiguous, "edges_outside_qt_anchors": outside_anchors,
        "limits": ["Bounds assume affine clocks and valid Q/T causality; not a calibration.",
                   "Sensor latency, missed optical pulses and RPM accuracy are not included.",
                   "PID matching uses command-send time, not physical motor application time.",
                   "Raw edge times retained; no RPM resampling or pulses/revolution assumption.",
                   "Clock-domain identity is operator-confirmed, not derivable from these CSVs."],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rpm_csv", type=Path)
    parser.add_argument("pid_csv", type=Path)
    parser.add_argument("output", type=Path, help="new directory, parent must exist")
    parser.add_argument("--same-pi-boot", action="store_true", required=True)
    parser.add_argument("--timestamp-slack-ns", type=int, default=10_000)
    args = parser.parse_args()
    summary = align(args.rpm_csv, args.pid_csv, args.output,
                    same_pi_boot=args.same_pi_boot, slack_ns=args.timestamp_slack_ns)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
