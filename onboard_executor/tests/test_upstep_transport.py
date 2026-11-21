"""Synthetic UART only; assert upward jump and downward slew on actual sent commands."""
import json
import sys
from pathlib import Path
from test_session_transport import check
c=json.loads(Path(sys.argv[2]).read_text())
c['maximum_duration_s']=20
c['startup']['minimum_duration_s']=2
c['shutdown']['minimum_duration_s']=2
c['hardware']['startup_ramp_rate']=.1
c['hardware']['normal_ramp_rate']=.1
c['pid']['zero_quaternion_wxyz']=[1,0,0,0]
c['control_mode'] = 'FIXED_HOLD'
c['hardware']['mechanically_locked_verified'] = False  # qualified only inside the synthetic PTY harness
c['hardware']['motor_min'] = 0
c['hardware']['motor_max'] = .3
c['segments'] = [dict(condition_id='synthetic-upstep', segment_id=f'hold-{i}',
    hold_id=f'hold-{i}', kind='HOLD', start_deg=0, end_deg=0) for i in range(3)]
for i,(s,d) in enumerate(zip(c['segments'],[.1,.2,.1])):
    s.update(collective=d,duration_s=1,require_settled=i!=1,preparation_only=i!=1,collective_step_up=i==1)
check(Path(sys.argv[1]).resolve(),c,'normal',preserve_segments=True)
print('up-step/down-ramp sent-command test passed')
