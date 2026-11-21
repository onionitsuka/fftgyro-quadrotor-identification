import copy
import json
import math
from pathlib import Path

import pytest
from fftgyro_quad_id.experiment_runner.coupled_reference import multiply, resolve_coupled_anchor

ROOT=Path(__file__).resolve().parents[1]


def fixture_data():
    s=json.loads((ROOT/'onboard_executor/config/coupled-session.example.json').read_text())
    pitch,yaw=map(math.radians,(-30,20))
    q=multiply([math.cos(pitch/2),0,math.sin(pitch/2),0],[math.cos(yaw/2),0,0,math.sin(yaw/2)])
    r={'summary':dict(quaternion_wxyz=q,armed=False,samples=30,max_reference_deviation_deg=.1,max_body_rate_deg_s=[0,0,0])}
    samples=[dict(receive_monotonic_ns=1_000_000_000+i*10_000_000,experimental_pitch_deg=-30,experimental_yaw_deg=20) for i in range(30)]
    meta=dict(pitch_sign=-1,offset_roll_deg=-147.1289,yaw_sign=1,offset_yaw_deg=0)
    cal=dict(yaw_sign=1,existing_encoder_offset_yaw_deg=0,pitch_calibrations={'-30':dict(front_experimental_yaw_deg=0)})
    return s,r,samples,meta,cal,1_000_000_000,1_300_000_000


def test_shared_anchor_recovers_rig_base():
    args=fixture_data()
    resolved,evidence=resolve_coupled_anchor(*args)
    assert resolved['rig_base_quaternion_wxyz']==pytest.approx([1,0,0,0])
    assert resolved['coupled_anchor_resolved']
    assert not resolved['startup']['capture_reference']
    assert resolved['pid']['zero_angle_deg']==0
    assert evidence['initial_pitch_deg']==-30
    assert evidence['initial_front_relative_yaw_deg']==20
    assert not args[0]['coupled_anchor_resolved']


def test_reference_rejects_motion_and_changed_front():
    args=list(fixture_data());args[2][-1]['experimental_yaw_deg']=25
    with pytest.raises(ValueError,match='span'): resolve_coupled_anchor(*args)
    args=list(fixture_data());args[4]['pitch_calibrations']['0']={'front_experimental_yaw_deg':1}
    with pytest.raises(ValueError,match='consistent'): resolve_coupled_anchor(*args)


