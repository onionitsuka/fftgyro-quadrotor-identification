import copy
import unittest
from fftgyro_quad_id.experiment_runner.pitch_reference import resolve_pitch_anchor, resolve_yaw_anchor


class PitchReferenceTests(unittest.TestCase):
    def setUp(self):
        self.session = {'control_mode': 'AXIS_PID', 'pid': {'axis': 'PITCH',
            'angle_min_deg': -90, 'angle_max_deg': 90, 'cascade': {'kp': .14}},
            'startup': {'capture_reference': True}, 'segments': [{'start_deg': -30}]}
        self.ref = {'summary': {'quaternion_wxyz': [1, 0, 0, 0], 'armed': False,
            'samples': 100, 'max_reference_deviation_deg': .1, 'max_body_rate_deg_s': [0, .1, 0]}}
        self.meta = {'pitch_sign': -1, 'offset_roll_deg': -147.1289}
        self.samples = [{'receive_monotonic_ns': 1_000_000_000 + i*10_000_000,
                         'experimental_pitch_deg': -26.9} for i in range(101)]

    def run_anchor(self):
        return resolve_pitch_anchor(self.session, self.ref, self.samples, self.meta,
                                    1_000_000_000, 2_000_000_000)

    def test_absolute_target_not_added_to_initial_pose(self):
        old = copy.deepcopy(self.session)
        resolved, evidence = self.run_anchor()
        self.assertAlmostEqual(evidence['initial_move_deg'], -3.1)
        self.assertEqual(resolved['pid']['zero_angle_deg'], -26.9)
        self.assertEqual(resolved['segments'], old['segments'])
        self.assertEqual(self.session, old)
        self.assertTrue(resolved['startup']['encoder_absolute_anchor'])
        self.assertTrue(resolved['startup']['encoder_pitch_anchor'])

    def test_positive_and_zero(self):
        for angle in (0, 20):
            for sample in self.samples: sample['experimental_pitch_deg'] = angle
            _, evidence = self.run_anchor()
            self.assertEqual(evidence['initial_move_deg'], -30-angle)

    def test_preliminary_encoder_span_matches_native_three_degree_envelope(self):
        self.samples[0]['experimental_pitch_deg'] = -28.3
        resolved, evidence = self.run_anchor()
        self.assertAlmostEqual(evidence['encoder_span_deg'], 1.4)
        self.assertEqual(evidence['preliminary_encoder_span_limit_deg'], 3)
        self.samples[0]['experimental_pitch_deg'] = -30.1
        with self.assertRaisesRegex(ValueError, 'encoder moved'):
            self.run_anchor()

    def test_reject_missing_moving_stale_wrong_sign_and_nan(self):
        original = copy.deepcopy(self.samples)
        for samples in ([], original[:30], original[31:],
                        [dict(r, experimental_pitch_deg=float('nan')) for r in original],
                        [dict(r, experimental_pitch_deg=i) for i,r in enumerate(original)]):
            self.samples = samples
            with self.assertRaises(ValueError): self.run_anchor()
        self.samples = original
        self.meta['pitch_sign'] = 1
        with self.assertRaises(ValueError): self.run_anchor()

    def test_no_yaw_or_moving_px4(self):
        self.ref['summary']['max_reference_deviation_deg'] = 5.1
        with self.assertRaises(ValueError): self.run_anchor()
        self.ref['summary']['max_reference_deviation_deg'] = 2
        self.ref['summary']['max_body_rate_deg_s'][1] = 15.1
        with self.assertRaises(ValueError): self.run_anchor()
        self.ref['summary']['max_body_rate_deg_s'][1] = 10
        self.run_anchor()  # preliminary check tolerates vibration; native dwell remains stricter
        self.session['pid']['axis'] = 'YAW'
        with self.assertRaises(ValueError): self.run_anchor()


class YawReferenceTests(unittest.TestCase):
    def setUp(self):
        self.session = {'control_mode': 'AXIS_PID', 'fixed_pitch_deg': -15,
            'pid': {'axis': 'YAW', 'angle_min_deg': -200, 'angle_max_deg': 200},
            'startup': {'capture_reference': True}, 'segments': [{'start_deg': 0}]}
        self.ref = {'summary': {'quaternion_wxyz': [1, 0, 0, 0], 'armed': False,
            'samples': 100, 'max_reference_deviation_deg': .2,
            'max_body_rate_deg_s': [0, 0, .2]},
            'raw_samples': [{'q1': 1, 'q2': 0, 'q3': 0, 'q4': 0}]}
        self.meta = {'yaw_sign': 1.0, 'offset_yaw_deg': -83.1445}
        self.cal = {'yaw_sign': 1.0, 'existing_encoder_offset_yaw_deg': -83.1445,
                    'pitch_calibrations': {'-15': {'front_experimental_yaw_deg': -1.1425}}}
        self.samples = [{'receive_monotonic_ns': 1_000_000_000+i*10_000_000,
                         'experimental_yaw_deg': -0.6425} for i in range(101)]

    def resolve(self):
        return resolve_yaw_anchor(self.session, self.ref, self.samples, self.meta, self.cal,
                                  1_000_000_000, 2_000_000_000)

    def test_instantaneous_value_remains_relative_to_fixed_front(self):
        resolved, evidence = self.resolve()
        self.assertAlmostEqual(resolved['pid']['zero_angle_deg'], .5)
        self.assertAlmostEqual(evidence['initial_move_deg'], -.5)
        self.assertEqual(evidence['basis'], 'fixed_test_encoder_front_instantaneous')
        self.assertTrue(resolved['startup']['encoder_absolute_anchor'])
        self.assertEqual(resolved['startup']['reference_angle_tolerance_deg'], 3)
        self.assertEqual(resolved['startup']['reference_swing_tolerance_deg'], 5)
        self.assertEqual(resolved['startup']['reference_rate_tolerance_deg_s'], 10)
        self.assertEqual(resolved['pid']['angle_min_deg'], -180)
        self.assertEqual(resolved['pid']['angle_max_deg'], 180)
        self.assertEqual(resolved['pid']['swing_limit_deg'], 90)
        self.assertEqual(resolved['pid']['tracking_limit_deg'], 180)

    def test_latest_sample_is_used_even_while_fixture_moves(self):
        self.samples[0]['experimental_yaw_deg'] = -2.5
        self.samples[-1]['experimental_yaw_deg'] = 6
        resolved, evidence = self.resolve()
        self.assertAlmostEqual(resolved['pid']['zero_angle_deg'], 7.1425)
        self.assertEqual(evidence['current_experimental_yaw_deg'], 6)
        self.assertEqual(evidence['association'],
                         'latest valid samples at capture end; not hardware-synchronous')

    def test_start_pose_beyond_90_degrees_uses_full_yaw_envelope(self):
        for row in self.samples:
            row['experimental_yaw_deg'] = 133.506
        resolved, evidence = self.resolve()
        self.assertAlmostEqual(evidence['initial_front_relative_yaw_deg'], 134.6485)
        self.assertAlmostEqual(resolved['pid']['zero_angle_deg'], 134.6485)
        self.assertEqual((resolved['pid']['angle_min_deg'], resolved['pid']['angle_max_deg']),
                         (-180, 180))

    def test_wrap_and_missing_pitch_calibration(self):
        for row in self.samples: row['experimental_yaw_deg'] = 179.5
        self.cal['pitch_calibrations']['-15']['front_experimental_yaw_deg'] = -179.5
        resolved, _ = self.resolve()
        self.assertAlmostEqual(resolved['pid']['zero_angle_deg'], -1)
        self.session['fixed_pitch_deg'] = -7.5
        with self.assertRaisesRegex(ValueError, 'no fixed-test'):
            self.resolve()

    def test_motion_is_allowed_but_wrong_conversion_is_rejected(self):
        self.samples[-1]['experimental_yaw_deg'] = 6
        self.resolve()
        self.samples[-1]['experimental_yaw_deg'] = -.6425
        self.meta['yaw_sign'] = -1
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.resolve()

    def test_instantaneous_capture_requires_only_five_px4_samples(self):
        self.ref['summary']['samples'] = 5
        self.resolve()
        self.ref['summary']['samples'] = 4
        with self.assertRaisesRegex(ValueError, 'instantaneous PX4 Yaw reference'):
            self.resolve()
