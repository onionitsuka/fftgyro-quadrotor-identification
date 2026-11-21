"""Common free-flight plant and ideal sensors for localhost PX4 SITL.

All routes share mass/inertia, rotor dynamics, clock and virtual sensors.
Only Aircraft.loads changes. This is not native Gazebo X500 aerodynamics.
"""
from __future__ import annotations

import math
import numpy as np

from simulate_graybox_forward_flight import Aircraft, G, motor_map, rotation, rhs, euler_deg
from speed_scheduled_graybox import ScheduledAircraft

GAZEBO_AERO = dict(rotor_drag_coefficient=8.06428e-5,
                   rolling_moment_coefficient=1e-6,
                   source="gz-sim8 MulticopterMotorModel; PX4 X500 model.sdf",
                   propeller="Holybro X500 V2 standard 1045; aero coefficients NOT propeller-calibrated")


class GazeboAeroAircraft(Aircraft):
    """Gazebo rotor-drag equations with shared measured propulsion/rigid body.

    No identified residuals: adding these terms to the identified model would
    double count aerodynamic loads. This is not the native Gazebo solver.
    Rotor-link velocity includes body rotation at the rotor position.
    """
    def __init__(self, artifact: dict):
        a = artifact["settings"]["arm_xy_m"]
        self.positions = np.array([[a,a,0], [-a,-a,0], [a,-a,0], [-a,a,0]])
        super().__init__(artifact, "physical")

    def loads(self, air, omega, rotor, rotor_dot):
        result = super().loads(air, omega, rotor, rotor_dot)
        local_velocity = air + np.cross(omega, self.positions)
        perpendicular = local_velocity.copy()
        perpendicular[:, 2] = 0
        drag = -GAZEBO_AERO["rotor_drag_coefficient"] * np.abs(rotor[:, None]) * perpendicular
        rolling = -GAZEBO_AERO["rolling_moment_coefficient"] * np.abs(rotor[:, None]) * perpendicular
        result[:3] += drag.sum(axis=0)
        result[3:] += (np.cross(self.positions, drag) + rolling).sum(axis=0)
        return result


class Plant:
    def __init__(self, artifact: dict, schedule: dict, route: str, dt: float = .004):
        self.vehicle = (GazeboAeroAircraft(artifact) if route == "gazebo_aero" else
                        Aircraft(artifact, route) if route == "physical" else
                        ScheduledAircraft(artifact, route, schedule))
        self.dt = dt
        self.voltage = 16.0
        self.state = np.zeros(13)
        self.state[6] = 1
        self.rotor = np.full(4, .01)
        self.history: list[np.ndarray] = []
        self.specific_force = np.array([0., 0., -G])
        self.load = np.zeros(6)
        self.rng = np.random.default_rng(20261005)

    def step(self, command: np.ndarray, armed: bool) -> None:
        v = self.vehicle
        command = np.clip(command, 0, 1) if armed else np.zeros(4)
        self.history.append(command.copy())
        delayed = np.array([self.history[max(0, len(self.history) - 1 - round(m["delay_s"] / self.dt))][j]
                            for j, m in enumerate(v.motors)])
        goal = motor_map(delayed, self.voltage, v.motors)
        tau = np.array([m["tau_up_s"] if goal[j] >= self.rotor[j] else m["tau_down_s"]
                        for j, m in enumerate(v.motors)])
        new_rotor = np.maximum(.01, goal + (self.rotor - goal) * np.exp(-self.dt / tau))
        mean, rate = (self.rotor + new_rotor) / 2, (new_rotor - self.rotor) / self.dt
        old_velocity = self.state[3:6].copy()
        k1, self.load = rhs(self.state, v, mean, rate)
        # Unilateral ground support: do not evaluate a fictitious falling RK
        # midpoint while the aircraft is still resting on the floor. Inflow
        # models are evaluated at the supported state, not at a descent state
        # with stopped rotors. Release only when net acceleration is upward.
        if self.state[2] == 0 and k1[5] >= 0:
            self.state[:] = 0
            self.state[6] = 1
            self.specific_force = np.array([0., 0., -G])
            self.rotor = new_rotor
            return
        k2, self.load = rhs(self.state + self.dt * k1 / 2, v, mean, rate)
        self.state += self.dt * k2
        self.state[6:10] /= np.linalg.norm(self.state[6:10])
        # Common ground contact only during takeoff/landing; no rig support in flight.
        if self.state[2] >= 0:
            self.state[:] = 0
            self.state[6] = 1
        self.specific_force = rotation(self.state[6:10]).T @ (
            (self.state[3:6] - old_velocity) / self.dt - np.array([0, 0, G]))
        self.rotor = new_rotor
        if not np.isfinite(self.state).all() or np.linalg.norm(self.state[3:6]) > 40:
            raise ValueError("Plant left finite/speed support")
        if max(abs(euler_deg(self.state[6:10])[:2])) > 70:
            raise ValueError("Plant roll/pitch exceeds 70 degrees")

    def sensor_values(self) -> dict:
        x = self.state
        latitude = 47.397742 + math.degrees(x[0] / 6378137)
        longitude = 8.545594 + math.degrees(x[1] / (6378137 * math.cos(math.radians(47.397742))))
        altitude = 488.0 - x[2]
        return dict(accel=self.specific_force, gyro=x[10:13],
                    # PX4 suppresses identical mag samples; shared seeded 0.5 mG noise.
                    mag=rotation(x[6:10]).T @ np.array([.22, .006, .43]) + self.rng.normal(0, .0005, 3),
                    pressure_hpa=1013.25 * (1 - altitude / 44330) ** 5.255 + self.rng.normal(0, .001),
                    lat=int(round(latitude * 1e7)), lon=int(round(longitude * 1e7)),
                    alt_mm=int(round(altitude * 1000)))


def common_parameters(artifact: dict) -> dict[str, float]:
    """PX4 allocation matches the physical model's motor numbering and geometry."""
    c = artifact["settings"]
    a = c["arm_xy_m"]
    # Shared conservative PX4 gains for the measured 50-ms motor-command delay.
    # These are frozen for all routes, not stock X500 gains or a per-model retune.
    result = dict(MPC_THR_HOVER=.427, MAV_0_BROADCAST=0, MAV_1_BROADCAST=0,
                  MC_ROLL_P=3., MC_PITCH_P=3., MC_YAW_P=2.,
                  MC_ROLLRATE_P=.06, MC_PITCHRATE_P=.06, MC_YAWRATE_P=.1,
                  MC_ROLLRATE_I=.05, MC_PITCHRATE_I=.05, MC_YAWRATE_I=.03,
                  MC_ROLLRATE_D=.001, MC_PITCHRATE_D=.001)
    # PX4 ControlAllocator zeros an axis if every effectiveness entry <= .05.
    # CT=1 with measured KM~=0.019 silently removed yaw authority. Retain the
    # stock X500 CT=6.5 allocation scale (not the physical thrust coefficient).
    # Common scaling cancels in normalized allocation; KM=KQ/KF stays measured.
    allocation_ct = 6.5
    if allocation_ct * abs(c["kq"] / c["kf"]) <= .05:
        raise ValueError("PX4 allocation yaw authority below its .05 threshold")
    for j, (x, y, spin) in enumerate(zip((a, -a, a, -a), (a, -a, -a, a), (1, 1, -1, -1), strict=True)):
        result.update({f"CA_ROTOR{j}_PX": x, f"CA_ROTOR{j}_PY": y,
                       f"CA_ROTOR{j}_KM": spin * c["kq"] / c["kf"], f"CA_ROTOR{j}_CT": allocation_ct})
    return result
