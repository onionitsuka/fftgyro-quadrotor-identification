"""Small, hardware-independent pieces used by the free-decay recorder."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import median


FRAME_START = 0x7A
FRAME_END = 0x7B
FRAME_LENGTHS = {0x7B: 32, 1: 41, 2: 49}


def wrap_degrees(value: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (value + 180.0) % 360.0 - 180.0


def parse_encoder_frame(frame: bytes) -> tuple[float, float, float] | None:
    """Decode one FFT GYRO encoder frame, preserving the product axis names."""
    if len(frame) not in (32, 41, 49) or frame[0] != FRAME_START or frame[-1] != FRAME_END:
        return None
    if frame[1] != 2:
        return None
    try:
        values = tuple(
            float(frame[start : start + 8].decode("ascii").strip(" \x00"))
            for start in (4, 13, 22)
        )
    except (UnicodeDecodeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


class FrameReader:
    """Incrementally recover 32/41/49-byte FFT GYRO frames."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.discarded_bytes = 0

    def push(self, chunk: bytes) -> list[bytes]:
        self.buffer.extend(chunk)
        frames: list[bytes] = []
        while len(self.buffer) >= 32:
            try:
                start = self.buffer.index(FRAME_START)
            except ValueError:
                self.discarded_bytes += len(self.buffer)
                self.buffer.clear()
                break
            if start:
                self.discarded_bytes += start
                del self.buffer[:start]
            if len(self.buffer) < 32:
                break
            length = FRAME_LENGTHS.get(self.buffer[31])
            if length is None:
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < length:
                break
            if self.buffer[length - 1] != FRAME_END:
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            frames.append(bytes(self.buffer[:length]))
            del self.buffer[:length]
        return frames


class AngleUnwrapper:
    def __init__(self) -> None:
        self.previous: float | None = None
        self.value: float | None = None

    def update(self, wrapped: float) -> float:
        if self.previous is None:
            self.previous = wrapped
            self.value = wrapped
        else:
            assert self.value is not None
            self.value += wrap_degrees(wrapped - self.previous)
            self.previous = wrapped
        return self.value


class WindowedRateEstimator:
    """Estimate rate across a window so encoder quantization is not differentiated at 100 Hz."""

    def __init__(self, window_s: float = 0.25) -> None:
        self.window_s = window_s
        self.samples: deque[tuple[float, float]] = deque()

    def update(self, time_s: float, angle_deg: float) -> float:
        self.samples.append((time_s, angle_deg))
        while len(self.samples) > 2 and time_s - self.samples[0][0] > self.window_s:
            self.samples.popleft()
        if len(self.samples) < 2:
            return 0.0
        old_time, old_angle = self.samples[0]
        elapsed = time_s - old_time
        return (angle_deg - old_angle) / elapsed if elapsed > 0 else 0.0


@dataclass(frozen=True, slots=True)
class StopDecision:
    stopped: bool
    window_span_deg: float
    maximum_rate_deg_s: float


class StopDetector:
    """Detect sustained rest from a time window, not from a single sample."""

    def __init__(
        self,
        *,
        window_s: float = 2.0,
        angle_span_deg: float = 0.15,
        rate_deg_s: float = 0.5,
        minimum_after_release_s: float = 1.0,
    ) -> None:
        self.window_s = window_s
        self.angle_span_deg = angle_span_deg
        self.rate_deg_s = rate_deg_s
        self.minimum_after_release_s = minimum_after_release_s
        self.samples: deque[tuple[float, float, float]] = deque()

    def update(self, elapsed_s: float, angle_deg: float, rate_deg_s: float) -> StopDecision:
        self.samples.append((elapsed_s, angle_deg, rate_deg_s))
        while self.samples and elapsed_s - self.samples[0][0] > self.window_s:
            self.samples.popleft()
        angles = [sample[1] for sample in self.samples]
        rates = [abs(sample[2]) for sample in self.samples]
        span = max(angles) - min(angles)
        maximum_rate = max(rates)
        covers_window = bool(self.samples) and elapsed_s - self.samples[0][0] >= self.window_s * 0.95
        stopped = (
            elapsed_s >= self.minimum_after_release_s
            and covers_window
            and span <= self.angle_span_deg
            and maximum_rate <= self.rate_deg_s
        )
        return StopDecision(stopped, span, maximum_rate)


def robust_reference(values: list[float]) -> float:
    if not values:
        raise ValueError("reference requires at least one encoder sample")
    return float(median(values))
