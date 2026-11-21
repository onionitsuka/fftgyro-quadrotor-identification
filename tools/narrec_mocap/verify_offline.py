#!/usr/bin/env python3
"""No network/hardware: decode synthetic NatNet frames and verify MAVLink wire data."""
import copy
import csv
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from scipy.spatial.transform import Rotation
from MAVProxy.modules.mavproxy_optitrack.NatNetClient import NatNetClient
from mocap_px4 import eligible, frame_row, load_config, transform_pose
import mocap_px4
from session import audit
from pymavlink.dialects.v20 import common as mav

ROOT = Path(__file__).resolve().parent


def packet_payload(version=(3, 1), valid=True, missing=False):
    extended = version >= (4, 1)
    def block(n=0, data=b''):
        return struct.pack('<i', n) + (struct.pack('<i', len(data)) if extended else b'') + data
    body = struct.pack('<i3f4ffh', 1, .25, 1., -.5, 0., 0., 0., 1., .001, int(valid))
    data = struct.pack('<i', 42) + block() + block() + block(0 if missing else 1, b'' if missing else body)
    data += block()  # skeletons
    if extended:
        data += block()  # assets
    data += block() + block() + block()  # labelled markers, force plates, devices
    data += struct.pack('<IIdQQQ', 0, 0, 1.25, 100, 200, 300)
    if version[0] >= 4:
        data += struct.pack('<ii', 1, 0)
    return data + struct.pack('<hi', 1, 0)


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT/'config.example.json')

    def decode(self, version=(3, 1), **kw):
        client = NatNetClient()
        payload = packet_payload(version, **kw)
        _, data = client._NatNetClient__unpack_mocap_data(payload, len(payload), *version)
        return frame_row(data, 1, 2_000_000_000, 1_000_000_000)

    def test_natnet3_and4_complete_frame(self):
        for version in ((3, 1), (4, 0), (4, 1), (4, 5)):
            with self.subTest(version=version):
                r = self.decode(version)
                self.assertEqual(r['frame'], 42)
                self.assertTrue(r['tracking_valid'])
                self.assertEqual(r['server_timestamp_s'], 1.25)
                self.assertAlmostEqual(r['y_m'], 1)
                self.assertTrue(eligible(self.c, r, 1_010_000_000)[0])

    def test_invalid_and_missing_not_sent(self):
        for kw in ({'valid': False}, {'missing': True}):
            r = self.decode(**kw)
            self.assertFalse(eligible(self.c, r, 1_010_000_000)[0])

    def test_stale_nonfinite_out_of_bounds(self):
        r = self.decode()
        self.assertFalse(eligible(self.c, r, 1_200_000_000)[0])
        for value in (float('nan'), float('inf'), 100):
            r['x_m'] = value
            self.assertFalse(eligible(self.c, r, 1_010_000_000)[0])

    def test_world_translation_axes(self):
        c = copy.deepcopy(self.c)
        c['origin_motive_m'] = [1, 2, 3]
        p, _ = transform_pose(c, [2, 4, 6], [0, 0, 0, 1])
        np.testing.assert_allclose(p, [1, 3, -2])

    def test_body_alignment_and_yaw(self):
        c = copy.deepcopy(self.c)
        # Motive rigid orientation is identity, body calibrated to FC FRD.
        c['rigid_from_body'] = np.asarray(c['world_from_motive']).T.tolist()
        p, e = transform_pose(c, [0, 1, 0], [0, 0, 0, 1])
        np.testing.assert_allclose(p, [0, 0, -1])
        np.testing.assert_allclose(e, [0, 0, 0], atol=1e-10)
        # A +90 degree rotation around FC down gives a +90 degree yaw.
        q = Rotation.from_euler('y', -90, degrees=True).as_quat()
        _, e = transform_pose(c, [0, 1, 0], q)
        self.assertAlmostEqual(e[2], np.pi/2)

    def test_rotated_offset(self):
        c = copy.deepcopy(self.c)
        c['world_from_motive'] = np.eye(3).tolist()
        c['body_to_rigid_m'] = [.1, 0, 0]
        p, _ = transform_pose(c, [0, 0, -1], Rotation.from_euler('z', 90, degrees=True).as_quat())
        np.testing.assert_allclose(p, [0, -.1, -1], atol=1e-10)

    def test_bad_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'config.json'
            c = copy.deepcopy(self.c)
            c['world_from_motive'][0][0] = -1  # reflection
            path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_mavlink2_wire_pose_and_covariance(self):
        stream = io.BytesIO()
        tx = mav.MAVLink(stream, srcSystem=245, srcComponent=197)
        cov = [0.0]*21
        for i in (0, 6, 11):
            cov[i] = .03**2
        for i in (15, 18, 20):
            cov[i] = .1**2
        tx.vision_position_estimate_send(2_000_000, .25, -.5, -1, 0, 0, .2, cov, 0)
        wire = stream.getvalue()
        self.assertEqual(wire[0], 0xFD)
        rx = mav.MAVLink(None).parse_buffer(wire)[0]
        self.assertEqual(rx.get_type(), 'VISION_POSITION_ESTIMATE')
        self.assertAlmostEqual(rx.z, -1)
        self.assertAlmostEqual(rx.covariance[0], .03**2)
        self.assertEqual(rx.get_srcSystem(), 245)

    def test_audit_preserves_missing_and_reports_reset(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'test.csv'
            rows = [self.decode(), self.decode(missing=True), self.decode()]
            for i, row in enumerate(rows):
                row['rx_monotonic_ns'] += i*10_000_000
                row['frame'] = [42, 44, 43][i]
            with path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            result = audit(path)
            self.assertEqual(result['rows'], 3)
            self.assertAlmostEqual(result['valid_fraction'], 2/3)
            self.assertEqual(result['missing_frame_numbers'], 1)
            self.assertEqual(result['duplicate_or_reset_frames'], 1)

    def test_cli_recording_and_optional_vision(self):
        # Exercise actual main loop, full-frame adapter, CSV output, and optional
        # sender. The NatNet transport and FC connection are mocks; no sockets.
        from pymavlink import mavutil
        for send in (False, True):
            with self.subTest(send=send), tempfile.TemporaryDirectory() as d:
                out = Path(d)/'capture'
                config = Path(d)/'config.json'
                c = copy.deepcopy(self.c)
                c['bench_transform_verified'] = True
                config.write_text(json.dumps(c))
                argv = ['mocap_px4.py', '--server', '127.0.0.1', '--client', '127.0.0.1',
                        '--config', str(config), '--output', str(out), '--duration', '.05']
                if send:
                    argv += ['--send-vision', '--mavlink', 'mock']
                def fake_run(client):
                    for i, valid in enumerate((True, False, True)):
                        payload = bytearray(packet_payload(valid=valid))
                        struct.pack_into('<i', payload, 0, 42+i)
                        # Timestamp offset: payload suffix has two uint32 before it.
                        struct.pack_into('<d', payload, len(payload)-38, 1.25+i*.01)
                        client._NatNetClient__unpack_mocap_data(bytes(payload), len(payload), 3, 1)
                    return True
                conn = MagicMock()
                conn.wait_heartbeat.return_value.autopilot = mavutil.mavlink.MAV_AUTOPILOT_PX4
                conn.recv_match.return_value = None
                with patch('sys.argv', argv), patch.object(NatNetClient, 'run', fake_run), \
                        patch.object(mavutil, 'mavlink_connection', return_value=conn):
                    mocap_px4.main()
                with (out/'mocap.csv').open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[1]['tracking_valid'], 'False')
                self.assertEqual(rows[1]['sent_vision'], 'False')
                self.assertEqual(conn.mav.vision_position_estimate_send.call_count, int(send))
                self.assertEqual(json.loads((out/'capture.json').read_text())['frames'], 3)
                self.assertFalse(conn.mav.command_long_send.called)
                self.assertFalse(conn.mav.set_mode_send.called)


if __name__ == '__main__':
    unittest.main(verbosity=2)
