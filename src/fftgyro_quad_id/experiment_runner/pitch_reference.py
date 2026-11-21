"""Anchor PX4 joint feedback to the existing FFT GYRO horizontal encoder zero."""
from __future__ import annotations

import copy
import math
import statistics


YAW_NATIVE_REFERENCE_ANGLE_TOLERANCE_DEG = 3.0
YAW_NATIVE_REFERENCE_SWING_TOLERANCE_DEG = 5.0
YAW_NATIVE_REFERENCE_RATE_TOLERANCE_DEG_S = 10.0
PITCH_PRELIMINARY_ENCODER_SPAN_TOLERANCE_DEG = 3.0


def needs_pitch_anchor(session: dict) -> bool:
    return (session.get('control_mode') == 'AXIS_PID'
            and session.get('pid', {}).get('axis') == 'PITCH')


def needs_yaw_anchor(session: dict) -> bool:
    return (session.get('control_mode') == 'AXIS_PID'
            and session.get('pid', {}).get('axis') == 'YAW')


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _validate_px4_reference(reference: dict) -> tuple[list[float], float, float]:
    summary = reference['summary']
    q = summary['quaternion_wxyz']
    if (len(q) != 4 or any(not math.isfinite(v) for v in q)
            or not .999 <= math.sqrt(sum(v*v for v in q)) <= 1.001
            or summary.get('armed') is not False or summary.get('samples', 0) < 20):
        raise ValueError('invalid PX4 absolute reference')
    deviation = float(summary['max_reference_deviation_deg'])
    rates = [float(v) for v in summary['max_body_rate_deg_s']]
    if len(rates) != 3 or any(not math.isfinite(v) for v in rates):
        raise ValueError('invalid PX4 reference rates')
    return list(q), deviation, rates[1]


def resolve_pitch_anchor(session: dict, reference: dict, samples: list[dict],
                         metadata: dict, begin_ns: int, end_ns: int) -> tuple[dict, dict]:
    """Pair stationary measurements within the enclosing local capture interval.

    This is a bounded static-pose association, not sample-level clock alignment.
    Native recapture subsequently preserves this absolute angle association.
    """
    if not needs_pitch_anchor(session):
        raise ValueError('absolute encoder anchor is for Pitch AXIS_PID only')
    if not session.get('startup', {}).get('capture_reference'):
        raise ValueError('Pitch horizontal mode requires native reference capture')
    if metadata.get('pitch_sign') != -1 or metadata.get('offset_roll_deg') != -147.1289:
        raise ValueError('encoder horizontal calibration differs from approved existing zero')
    if not 0 < (end_ns - begin_ns) / 1e9 <= 15:
        raise ValueError('absolute reference capture interval invalid or too long')
    selected = [r for r in samples if begin_ns <= int(r['receive_monotonic_ns']) <= end_ns]
    if len(selected) < 20:
        raise ValueError('insufficient simultaneous encoder reference samples')
    times = [int(r['receive_monotonic_ns']) for r in selected]
    if (times[0] - begin_ns > 300_000_000 or end_ns - times[-1] > 300_000_000
            or any(b < a or b-a > 200_000_000 for a, b in zip(times, times[1:]))):
        raise ValueError('encoder reference is stale or discontinuous')
    angles = [float(r['experimental_pitch_deg']) for r in selected]
    if (any(not math.isfinite(a) for a in angles)
            or max(angles)-min(angles) > PITCH_PRELIMINARY_ENCODER_SPAN_TOLERANCE_DEG):
        raise ValueError('encoder moved during absolute reference capture')
    q, deviation, rate = _validate_px4_reference(reference)
    # The encoder directly proves that the released Pitch joint is stationary.
    # Permit wind/rig vibration in PX4's full-attitude metric here; the native
    # executor still requires the controlled joint rate <= 5 deg/s for its
    # configured dwell immediately before any motor command.
    if not math.isfinite(deviation) or not math.isfinite(rate) or deviation > 5 or rate > 15:
        raise ValueError('PX4 moved during absolute reference capture')
    angle = statistics.median(angles)
    if not session['pid']['angle_min_deg'] <= angle <= session['pid']['angle_max_deg']:
        raise ValueError('initial absolute Pitch outside configured angle bounds')
    resolved = copy.deepcopy(session)
    resolved['pid']['zero_quaternion_wxyz'] = list(q)
    resolved['pid']['zero_angle_deg'] = angle
    resolved['startup']['encoder_absolute_anchor'] = True
    # Keep the deployed Pitch production executor (v32) compatible while the
    # shared Pitch/Yaw anchor key rolls out.  New executors accept either key.
    resolved['startup']['encoder_pitch_anchor'] = True
    evidence = dict(basis='existing_encoder_zero_is_horizontal', initial_pitch_deg=angle,
                    encoder_span_deg=max(angles)-min(angles), encoder_samples=len(angles),
                    capture_begin_monotonic_ns=begin_ns, capture_end_monotonic_ns=end_ns,
                    offset_roll_deg=metadata['offset_roll_deg'], pitch_sign=metadata['pitch_sign'],
                    preliminary_encoder_span_limit_deg=PITCH_PRELIMINARY_ENCODER_SPAN_TOLERANCE_DEG,
                    preliminary_px4_attitude_deviation_limit_deg=5,
                    preliminary_px4_pitch_rate_limit_deg_s=15,
                    native_pitch_rate_limit_deg_s=5,
                    px4_reference=reference,
                    association='stationary enclosing interval; not hardware-synchronous',
                    first_target_deg=resolved['segments'][0]['start_deg'],
                    initial_move_deg=resolved['segments'][0]['start_deg']-angle)
    resolved.setdefault('operator_context', {})['pitch_angle_basis'] = 'encoder_horizontal_zero'
    resolved['operator_context']['reference_record'] = 'pitch-horizontal-reference.json'
    return resolved, evidence


def resolve_yaw_anchor(session: dict, reference: dict, samples: list[dict],
                       metadata: dict, calibration: dict,
                       begin_ns: int, end_ns: int) -> tuple[dict, dict]:
    """Pair the instantaneous pose with the registered fixed-test Yaw front."""
    if not needs_yaw_anchor(session):
        raise ValueError('Yaw front anchor is for Yaw AXIS_PID only')
    if not session.get('startup', {}).get('capture_reference'):
        raise ValueError('Yaw front mode requires native reference capture')
    if (metadata.get('yaw_sign') != calibration.get('yaw_sign')
            or metadata.get('offset_yaw_deg') != calibration.get('existing_encoder_offset_yaw_deg')):
        raise ValueError('encoder Yaw conversion differs from fixed-test front calibration')
    if not 0 < (end_ns - begin_ns) / 1e9 <= 15:
        raise ValueError('Yaw front capture interval invalid or too long')
    selected = [r for r in samples if begin_ns <= int(r['receive_monotonic_ns']) <= end_ns]
    if len(selected) < 20:
        raise ValueError('insufficient simultaneous Yaw encoder reference samples')
    times = [int(r['receive_monotonic_ns']) for r in selected]
    if (times[0] - begin_ns > 300_000_000 or end_ns - times[-1] > 300_000_000
            or any(b < a or b-a > 200_000_000 for a, b in zip(times, times[1:]))):
        raise ValueError('Yaw encoder reference is stale or discontinuous')
    encoder = [float(r['experimental_yaw_deg']) for r in selected]
    if any(not math.isfinite(a) for a in encoder):
        raise ValueError('invalid Yaw encoder value during front reference capture')
    encoder_span = max(encoder)-min(encoder)
    pitch_key = f"{float(session.get('fixed_pitch_deg', 0)):g}"
    registered = calibration.get('pitch_calibrations', {}).get(pitch_key)
    if registered is None:
        raise ValueError(f'no fixed-test Yaw front for fixed Pitch {pitch_key} deg')
    front = float(registered['front_experimental_yaw_deg'])
    # Yaw cannot be held perfectly still in wind before the motors start.  Use
    # the newest valid encoder/PX4 values as one instantaneous pose association
    # instead of rejecting the run based on motion across the capture window.
    encoder_instant = encoder[-1]
    # Read the moving fixture instantaneously, but retain the mechanically
    # registered front as zero.  No stationary average is required.
    angle = _wrap_degrees(encoder_instant - front)
    summary = reference.get('summary', {})
    if (summary.get('armed') is not False or summary.get('samples', 0) < 5):
        raise ValueError('invalid instantaneous PX4 Yaw reference')
    deviation = float(summary.get('max_reference_deviation_deg', math.nan))
    if not math.isfinite(deviation):
        raise ValueError('invalid instantaneous PX4 Yaw deviation')
    raw_px4 = reference.get('raw_samples', [])
    if not raw_px4:
        raise ValueError('missing instantaneous PX4 Yaw reference sample')
    latest_px4 = raw_px4[-1]
    q = [float(latest_px4[name]) for name in ('q1', 'q2', 'q3', 'q4')]
    if (any(not math.isfinite(v) for v in q)
            or not .999 <= math.sqrt(sum(v*v for v in q)) <= 1.001):
        raise ValueError('invalid instantaneous PX4 Yaw reference quaternion')
    rates = summary.get('max_body_rate_deg_s', [])
    if len(rates) != 3:
        raise ValueError('invalid instantaneous PX4 Yaw rates')
    yaw_rate = float(rates[2])
    if not math.isfinite(yaw_rate):
        raise ValueError('invalid PX4 Yaw rate during front reference capture')
    resolved = copy.deepcopy(session)
    # Yaw is circular.  The registered front can therefore be more than 90 deg
    # from the freely resting start pose in wind.  Permit the full shortest-path
    # range while retaining the independent 60 deg/s body-rate safety limit.
    resolved['pid']['angle_min_deg'] = -180.0
    resolved['pid']['angle_max_deg'] = 180.0
    # Yaw itself is circular, but swing is motion away from the released Yaw
    # axis.  Keep the deployed v33 non-target-attitude safety envelope.
    resolved['pid']['swing_limit_deg'] = 90.0
    resolved['pid']['tracking_limit_deg'] = 180.0
    if not resolved['pid']['angle_min_deg'] <= angle <= resolved['pid']['angle_max_deg']:
        raise ValueError('initial front-relative Yaw outside configured angle bounds')
    resolved['pid']['zero_quaternion_wxyz'] = q
    resolved['pid']['zero_angle_deg'] = angle
    resolved['startup']['encoder_absolute_anchor'] = True
    # Wind-tunnel fixture vibration must not reject the front association before
    # motor start.  These remain bounded by the executor's validated startup
    # ranges and do not alter any in-run attitude/rate safety limit.
    resolved['startup']['reference_angle_tolerance_deg'] = YAW_NATIVE_REFERENCE_ANGLE_TOLERANCE_DEG
    resolved['startup']['reference_swing_tolerance_deg'] = YAW_NATIVE_REFERENCE_SWING_TOLERANCE_DEG
    resolved['startup']['reference_rate_tolerance_deg_s'] = YAW_NATIVE_REFERENCE_RATE_TOLERANCE_DEG_S
    resolved.setdefault('operator_context', {})['yaw_angle_basis'] = 'fixed_test_encoder_front_instantaneous'
    resolved['operator_context']['reference_record'] = 'yaw-front-reference.json'
    evidence = dict(basis='fixed_test_encoder_front_instantaneous', fixed_pitch_deg=float(pitch_key),
                    registered_front_experimental_yaw_deg=front,
                    current_experimental_yaw_deg=encoder_instant,
                    initial_front_relative_yaw_deg=angle,
                    encoder_span_deg=encoder_span, encoder_samples=len(encoder),
                    capture_begin_monotonic_ns=begin_ns, capture_end_monotonic_ns=end_ns,
                    offset_yaw_deg=metadata['offset_yaw_deg'], yaw_sign=metadata['yaw_sign'],
                    calibration=calibration, px4_reference=reference,
                    association='latest valid samples at capture end; not hardware-synchronous',
                    first_target_deg=resolved['segments'][0]['start_deg'],
                    initial_move_deg=_wrap_degrees(resolved['segments'][0]['start_deg']-angle),
                    preliminary_encoder_motion_rejection=False,
                    preliminary_px4_motion_rejection=False,
                    observed_px4_attitude_deviation_deg=deviation,
                    observed_px4_yaw_rate_deg_s=yaw_rate,
                    native_yaw_angle_tolerance_deg=YAW_NATIVE_REFERENCE_ANGLE_TOLERANCE_DEG,
                    native_yaw_swing_tolerance_deg=YAW_NATIVE_REFERENCE_SWING_TOLERANCE_DEG,
                    native_yaw_rate_limit_deg_s=YAW_NATIVE_REFERENCE_RATE_TOLERANCE_DEG_S)
    return resolved, evidence
