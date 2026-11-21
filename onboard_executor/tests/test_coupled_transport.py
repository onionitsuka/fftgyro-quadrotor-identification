"""Synthetic PX4 only: both joint waveforms, control allocation and release."""
import csv
import json
import math
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import sys
import tempfile
import time
from test_session_transport import packet, frames


def run(executable, source, loss=False):
    with tempfile.TemporaryDirectory(prefix='coupled-pty-') as tmp:
        root=Path(tmp);master,slave=pty.openpty();proc=None
        try:
            c=json.loads(Path(source).read_text())
            c['coupled_anchor_resolved']=True
            for s in (c,c['coupled_yaw']):
                s.update(session_id='coupled-synthetic',maximum_duration_s=60)
                s['hardware'].update(uart_device=os.ttyname(slave),log_directory=str(root/'logs'))
                s['startup'].update(minimum_duration_s=1)
                s['hardware']['startup_ramp_rate']=.1
                s['shutdown']['minimum_duration_s']=2
                for seg in s['segments']:
                    seg['collective']=.1
                    if seg['kind']=='SINE':
                        seg.update(duration_s=6,fade_s=1,analysis_start_s=1,analysis_end_s=5,amplitude_deg=1)
            path=root/'session.json';path.write_text(json.dumps(c))
            # Also validate every production coupled condition before synthetic execution.
            for source_path in Path(source).parent.glob('98[0-9][0-9].json'):
                subprocess.run([executable,'session-validate',str(source_path)],check=True,capture_output=True)
            with (root/'stdout').open('w+') as output:
                proc=subprocess.Popen([executable,'session-run',str(path)],stdout=output,stderr=output)
                started=time.monotonic();next_feedback=0;buffer=bytearray();count=0;release=False
                while proc.poll() is None and time.monotonic()-started<65:
                    now=time.monotonic();elapsed=now-started
                    if now>=next_feedback:
                        if not loss or count<100:
                            # Independent physical joint oscillations within the settling envelope.
                            p=math.radians(.3*math.sin(elapsed));y=math.radians(.4*math.cos(elapsed))
                            q=[math.cos(y/2)*math.cos(p/2),math.sin(y/2)*math.sin(p/2),
                               math.cos(y/2)*math.sin(p/2),math.sin(y/2)*math.cos(p/2)]
                            payload=struct.pack('<I7f',int(1000+elapsed*1000),*q,0,0,0)
                            os.write(master,packet(31,payload,246))
                        next_feedback=now+.01
                    if select.select([master],[],[],.002)[0]:
                        buffer.extend(os.read(master,8192))
                        for system,component,payload in frames(buffer):
                            run_id,seq,*v=struct.unpack('<II4fHBBB',payload)
                            assert not release
                            assert all(math.isfinite(u) and 0<=u<=.8 for u in v[:4])
                            release=v[-1]==1
                            count+=not release
                            os.write(master,packet(60001,struct.pack('<QIIBBB',int(elapsed*1e6),run_id,seq,system,component,int(release)),221))
                proc.wait(timeout=2);output.seek(0);diagnostic=output.read()
            assert release and count>100,diagnostic
            result=(root/'logs/coupled-synthetic/result.txt').read_text()
            assert ('status=ABORTED' if loss else 'status=COMPLETE') in result,result
            rows=list(csv.DictReader((root/'logs/coupled-synthetic/control.csv').open()))
            if not loss:
                active=[r for r in rows if r['stage']=='SEGMENT' and r['segment_id'].endswith('-1')]
                assert max(float(r['target_deg']) for r in active)-min(float(r['target_deg']) for r in active)>1
                assert max(float(r['yaw_target_deg']) for r in active)-min(float(r['yaw_target_deg']) for r in active)>1
                assert any(abs(float(r['yaw_delta']))>.0001 for r in active)
                assert any(abs(float(r['delta']))>.0001 for r in active)
                shutdown=[r for r in rows if r['stage']=='RAMP_DOWN']
                assert shutdown
                for before,after in zip(shutdown,shutdown[1:]):
                    assert all(float(after[f'sent{i}'])<=float(before[f'sent{i}'])+1e-6 for i in range(1,5))
                assert all(float(shutdown[-1][f'sent{i}'])==0 for i in range(1,5))
            print('PASS coupled transport', 'feedback loss' if loss else 'complete')
        finally:
            if proc and proc.poll() is None: proc.kill();proc.wait()
            os.close(master);os.close(slave)


if __name__=='__main__':
    run(sys.argv[1],sys.argv[2]);run(sys.argv[1],sys.argv[2],True)
