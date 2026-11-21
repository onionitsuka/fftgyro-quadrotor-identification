from fftgyro_quad_id.rig.free_decay import (
    AngleUnwrapper,
    FrameReader,
    StopDetector,
    WindowedRateEstimator,
    parse_encoder_frame,
    wrap_degrees,
)


def encoder_frame(roll: str, pitch: str, yaw: str, length: int = 32) -> bytes:
    frame = bytearray(b" " * length)
    frame[0] = 0x7A
    frame[1] = 2
    frame[2] = 0
    frame[3] = ord(",")
    frame[4:12] = roll.rjust(8).encode()
    frame[12] = ord(",")
    frame[13:21] = pitch.rjust(8).encode()
    frame[21] = ord(",")
    frame[22:30] = yaw.rjust(8).encode()
    frame[30] = ord(",")
    if length == 32:
        frame[31] = 0x7B
    else:
        frame[31] = 1 if length == 41 else 2
        frame[-1] = 0x7B
    return bytes(frame)


def test_parse_and_incremental_framing() -> None:
    expected = encoder_frame("10.0000", "-2.5000", "30.0000")
    reader = FrameReader()
    assert reader.push(b"noise" + expected[:12]) == []
    assert reader.push(expected[12:]) == [expected]
    assert parse_encoder_frame(expected) == (10.0, -2.5, 30.0)
    assert reader.discarded_bytes == 5


def test_parse_real_encoder_padding_nul() -> None:
    frame = bytearray(encoder_frame("237.9199", "184.4824", "83.8477"))
    frame[22:30] = b"83.8477\x00"
    assert parse_encoder_frame(bytes(frame)) == (237.9199, 184.4824, 83.8477)


def test_extended_frames_are_preserved() -> None:
    frames = [encoder_frame("1", "2", "3", length) for length in (41, 49)]
    reader = FrameReader()
    assert reader.push(b"".join(frames)) == frames


def test_wrap_and_unwrap_crossing_180_degrees() -> None:
    unwrapper = AngleUnwrapper()
    assert wrap_degrees(181) == -179
    values = [unwrapper.update(value) for value in (179.0, -179.0, -178.0)]
    assert values == [179.0, 181.0, 182.0]


def test_stop_requires_a_full_quiet_window() -> None:
    detector = StopDetector(window_s=2.0, angle_span_deg=0.15, rate_deg_s=0.5)
    decision = None
    for index in range(22):
        elapsed = index * 0.1
        decision = detector.update(elapsed, 0.02 * (index % 2), 0.2)
    assert decision is not None and decision.stopped


def test_motion_in_window_prevents_stop() -> None:
    detector = StopDetector(window_s=2.0, angle_span_deg=0.15, rate_deg_s=0.5)
    decision = None
    for index in range(22):
        elapsed = index * 0.1
        decision = detector.update(elapsed, elapsed, 1.0)
    assert decision is not None and not decision.stopped


def test_windowed_rate_rejects_one_count_encoder_jitter() -> None:
    estimator = WindowedRateEstimator(window_s=0.25)
    rates = [
        estimator.update(index * 0.01, 0.088 if index % 2 else 0.0)
        for index in range(100)
    ]
    assert max(abs(value) for value in rates[30:]) < 0.5


def test_windowed_rate_preserves_sustained_motion() -> None:
    estimator = WindowedRateEstimator(window_s=0.25)
    rates = [estimator.update(index * 0.01, index * 0.1) for index in range(100)]
    assert 9.5 < rates[-1] < 10.5
