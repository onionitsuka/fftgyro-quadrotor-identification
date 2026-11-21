#!/usr/bin/env python3
"""Create run folders, timestamp events, and audit this kit's mocap CSV."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


def audit(path: Path) -> dict:
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        expected = {'frame', 'rx_monotonic_ns', 'tracking_valid', 'x_m', 'y_m', 'z_m', 'qw'}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError("This command accepts mocap_px4.py CSV; export Motive CSV separately.")
        rows = list(reader)
    times = [int(r['rx_monotonic_ns']) / 1e9 for r in rows]
    frames = [int(r['frame']) for r in rows]
    gaps = [b-a for a, b in zip(times, times[1:])]
    jumps = [b-a for a, b in zip(frames, frames[1:])]
    valid = sum(r['tracking_valid'] == 'True' for r in rows)
    finite = 0
    for r in rows:
        try:
            finite += all(math.isfinite(float(r[k])) for k in ('x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw'))
        except (KeyError, ValueError):
            pass
    span = times[-1] - times[0] if len(times) > 1 else 0
    return {'rows': len(rows), 'duration_s': span, 'rate_hz': (len(rows)-1)/span if span > 0 else None,
            'valid_fraction': valid/len(rows) if rows else None, 'finite_pose_rows': finite,
            'max_receive_gap_s': max(gaps) if gaps else None,
            'missing_frame_numbers': sum(max(0, j-1) for j in jumps),
            'duplicate_or_reset_frames': sum(j <= 0 for j in jumps),
            'nonmonotonic_receive_times': sum(g <= 0 for g in gaps),
            'vision_sent': sum(r.get('sent_vision') == 'True' for r in rows),
            'notice': 'Receive gaps are not end-to-end latency. File integrity does not prove flight readiness.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    init = sub.add_parser('init')
    init.add_argument('directory', type=Path)
    init.add_argument('--phase', choices=['bench', 'manual', 'position', 'offboard'], required=True)
    event = sub.add_parser('event')
    event.add_argument('directory', type=Path)
    event.add_argument('label')
    qc = sub.add_parser('audit')
    qc.add_argument('csv', type=Path)
    seal = sub.add_parser('seal')
    seal.add_argument('directory', type=Path)
    a = p.parse_args()
    if a.cmd == 'init':
        a.directory.mkdir(parents=True, exist_ok=False)
        for name in ('motive', 'flight', 'network', 'video', 'settings'):
            (a.directory/name).mkdir()
        metadata = {'phase': a.phase, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'vehicle': 'Holybro X500 / Pixhawk 6X / PX4', 'firmware_git': None,
                    'actual_mode': None, 'pilot': None, 'observer': None, 'battery_id': None,
                    'motive_version': None, 'natnet_version': None, 'rigid_body_id': None,
                    'motive_take': None, 'ulog_file': None, 'sync_method': None,
                    'clock_mapping': None, 'sync_residual_ms': None,
                    'coordinate_axes': None, 'rigid_body_offset_m': None,
                    'calibration_file': None, 'mass_kg': None, 'cg_reference': None,
                    'payload_and_markers': None, 'notes': ''}
        (a.directory/'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        print(a.directory)
    elif a.cmd == 'event':
        if not (a.directory/'run.json').is_file():
            p.error('run.json not found; initialize the session first')
        with (a.directory/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'unix_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
                                'label': a.label}, ensure_ascii=False)+'\n')
    elif a.cmd == 'audit':
        print(json.dumps(audit(a.csv), ensure_ascii=False, indent=2))
    else:
        results = {}
        for path in sorted(a.directory.rglob('*')):
            if path.is_file() and path.name != 'sha256.json':
                h = hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda: f.read(1024*1024), b''):
                        h.update(block)
                results[str(path.relative_to(a.directory))] = {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}
        required = {name: any(p.is_file() for p in (a.directory/name).glob('**/*')) for name in ('motive', 'flight')}
        report = {'files': results, 'required_groups_have_files': required,
                  'notice': 'Hashes record integrity; inspect take/ULog contents and synchronization separately.'}
        with (a.directory/'sha256.json').open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(required))


if __name__ == '__main__':
    main()
