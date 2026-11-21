import unittest

from fftgyro_quad_id.experiment_runner.sync_protocol import (
    arm_command,
    disarm_command,
    parse_pulse_ack,
    ping_command,
    pulse_command,
)


class SyncProtocolTests(unittest.TestCase):
    def test_commands(self) -> None:
        self.assertEqual(ping_command(), b"PING\n")
        self.assertEqual(arm_command("run_001"), b"ARM run_001\n")
        self.assertEqual(pulse_command(42), b"PULSE 42 1000\n")
        self.assertEqual(disarm_command(), b"DISARM\n")

    def test_invalid_run_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            arm_command("run id with spaces")

    def test_pulse_bounds_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            pulse_command(1, 99)
        with self.assertRaises(ValueError):
            pulse_command(1, 10_001)

    def test_acknowledgement_is_parsed(self) -> None:
        ack = parse_pulse_ack("PULSED run_001 42 12345678 1000\n")
        self.assertEqual(ack.run_id, "run_001")
        self.assertEqual(ack.sequence, 42)
        self.assertEqual(ack.release_time_us, 12345678)
        self.assertEqual(ack.width_us, 1000)


if __name__ == "__main__":
    unittest.main()
