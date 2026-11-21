"""One shared rig reference for R_base Ry(pitch) Rz(yaw)."""
from __future__ import annotations

import copy
import math
import statistics
from .pitch_reference import resolve_pitch_anchor


def multiply(a: list[float], b: list[float]) -> list[float]:
    w,x,y,z=a; v,i,j,k=b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]


def resolve_coupled_anchor(session: dict, reference: dict, samples: list[dict],
                           metadata: dict, calibration: dict, begin: int, end: int) -> tuple[dict, dict]:
    probe=copy.deepcopy(session)
    probe['startup']['capture_reference']=True
    resolved,evidence=resolve_pitch_anchor(probe,reference,samples,metadata,begin,end)
    if (metadata.get('yaw_sign') != calibration.get('yaw_sign') or
        metadata.get('offset_yaw_deg') != calibration.get('existing_encoder_offset_yaw_deg')):
        raise ValueError('coupled Yaw calibration mismatch')
    # All registered fixed-Pitch front calibrations must agree for this shared base.
    fronts=[float(v['front_experimental_yaw_deg']) for v in calibration['pitch_calibrations'].values()]
    if not fronts or any(not math.isfinite(v) or abs(v-fronts[0])>1e-6 for v in fronts):
        raise ValueError('coupled reference requires one consistent Yaw front')
    angles=[float(s['experimental_yaw_deg']) for s in samples if begin<=int(s['receive_monotonic_ns'])<=end]
    if not angles or any(not math.isfinite(v) for v in angles):
        raise ValueError('missing coupled Yaw observations')
    angles=[angles[0]+math.remainder(v-angles[0],360) for v in angles]
    if max(angles)-min(angles)>3:
        raise ValueError('coupled Yaw reference span exceeds 3 degrees')
    yaw=math.remainder(statistics.median(angles)-fronts[0],360)
    pitch=evidence['initial_pitch_deg']
    if not session['coupled_yaw']['pid']['angle_min_deg']<=yaw<=session['coupled_yaw']['pid']['angle_max_deg']:
        raise ValueError('initial coupled Yaw outside angle bounds')
    p,y=math.radians(pitch)/2,math.radians(yaw)/2
    q=reference['summary']['quaternion_wxyz']
    # q = base * Ry(pitch) * Rz(yaw)
    base=multiply(multiply(q,[math.cos(y),0,0,-math.sin(y)]),[math.cos(p),0,-math.sin(p),0])
    resolved['rig_base_quaternion_wxyz']=base
    resolved['coupled_anchor_resolved']=True
    for s in (resolved,resolved['coupled_yaw']):
        s['pid']['zero_angle_deg']=0
        s['pid']['zero_quaternion_wxyz']=[1,0,0,0]
        s['startup'].update(capture_reference=False,encoder_absolute_anchor=False,encoder_pitch_anchor=False)
    evidence.update(initial_front_relative_yaw_deg=yaw,rig_base_quaternion_wxyz=base,
                    yaw_span_deg=max(angles)-min(angles),basis='shared_encoder_pitch_yaw_rig_base')
    return resolved,evidence
