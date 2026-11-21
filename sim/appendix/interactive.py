#!/usr/bin/env python3
"""Local browser simulator. No aircraft connection; reuses the thesis load evaluator.

Run: python3 interactive.py  (http://127.0.0.1:8765)
The browser sends velocity/height/heading requests; Python integrates the vehicle.
Changing the load model or coefficient route restarts from its own hover trim.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import webbrowser

import numpy as np
from scipy.spatial.transform import Rotation

import simulate as sim

HERE = Path(__file__).resolve().parent
DT = 0.005
# Interactive-only control limits; model coefficients / thesis controller unchanged.
ATTITUDE_SLEW_RAD_S = np.deg2rad(40)
TORQUE_LIMIT_NM = np.array([1.5, 1.5, .1])


class Flight:
    def __init__(self, model="identified", route="command_route", dynamics=None):
        if model not in ("existing", "identified") or route not in ("measured", "command_route"):
            raise ValueError("Unknown model or coefficient route")
        self.config = json.loads((HERE / "model.json").read_text())
        self.model, self.route = model, route
        self.dynamics = dynamics or sim.derivative
        self.vehicle = sim.Loads(self.config, model == "identified", route)
        self.voltage = 16.0
        self.x, self.rotor, self.trim_control, self.trim_error = sim.hover_trim(self.vehicle, self.voltage)
        self.x[2] = -2.0
        self.trim_rotation = sim.rotation(self.x[6:10])
        self.previous_desired = self.trim_rotation.copy()
        self.integral = np.zeros(6)
        self.wind = np.zeros(3)
        self.height, self.heading = 2.0, 0.0
        self.t, self.steps = 0.0, 0
        self.command = sim.motor_command(self.rotor, self.voltage, self.config)
        self.wrench = self.vehicle(np.zeros(3), np.zeros(3), self.rotor, np.zeros(4))
        self.saturated = False
        self.torque_saturated = False
        self.motor_saturated = False
        self.speed_command = 2.0
        self.status = "ready"
        self.mode = "velocity"
        self.inputs = np.zeros(4)
        self.rows = []
        c = self.vehicle.cfg
        self.allocation = np.vstack([np.full(4, c.kf), c.kf*c.arm_xy_m*sim.ROLL,
                                     c.kf*c.arm_xy_m*sim.PITCH, c.kq*sim.SPIN])
        self.low = sim.motor_speed(np.full(4, .1), self.voltage, self.config)
        self.high = sim.motor_speed(np.ones(4), self.voltage, self.config)
        self.tau = np.array([m["tau_up_s"] for m in self.config["motor_models"]])
        # This UI supports the adopted, common-tau / zero-delay propulsion model.
        assert all(m["delay_s"] == 0 and m["tau_up_s"] == m["tau_down_s"] for m in self.config["motor_models"])
        self.log()

    def controller(self, forward, right, climb, yaw_rate, speed, mode="velocity"):
        """Common PI velocity/altitude and geometric attitude controller, not PX4.

        Direction commands are relative to the requested heading. Releasing keys
        requests zero horizontal velocity and holds altitude/heading (no XY hold).
        Gains, mixer and trim convention follow the appendix controller.
        """
        v, g = self.vehicle, self.config["controller"]
        self.heading += DT*yaw_rate
        cy, sy = np.cos(self.heading), np.sin(self.heading)
        if mode == "attitude":
            # Mode 2: tilt commands plus direct collective around hover thrust.
            # Neutral is hover collective, not motor-off; no XY/altitude feedback.
            yaw_rotation = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            desired = yaw_rotation@sim.rotation(sim.attitude(right*np.deg2rad(25), -forward*np.deg2rad(25)))@self.trim_rotation
            thrust = self.trim_control[0]*(1+.65*climb)
            self.integral[:3] = 0
            return self.attitude_control(desired, thrust)
        self.height = float(np.clip(self.height + DT*climb, .4, 12.0))
        horizontal = np.array([forward, right], float)
        horizontal /= max(1.0, np.linalg.norm(horizontal))
        heading_xy = np.array([[cy, -sy], [sy, cy]])
        target = np.r_[heading_xy@horizontal*speed, -g["height_kp"]*(self.x[2]+self.height)]
        error = target-self.x[3:6]
        limits = np.array([g["horizontal_acceleration_limit_m_s2"]]*2+[g["vertical_acceleration_limit_m_s2"]])
        candidate = np.clip(self.integral[:3]+DT*error, -20, 20)
        raw = g["velocity_kp"]*error+g["velocity_ki"]*candidate
        blocked = (np.abs(raw)>limits)&(raw*error>0)
        self.integral[:3] = np.where(blocked, self.integral[:3], candidate)
        acceleration = np.clip(g["velocity_kp"]*error+g["velocity_ki"]*self.integral[:3], -limits, limits)
        force = v.mass*(acceleration-np.array([0., 0., sim.G]))
        bz = -force/np.linalg.norm(force)
        by = np.cross(bz, [cy, sy, 0.]); by /= np.linalg.norm(by)
        desired = np.column_stack([np.cross(by, bz), by, bz])@self.trim_rotation
        actual = sim.rotation(self.x[6:10])
        thrust = self.trim_control[0]*np.linalg.norm(force)/(v.mass*sim.G)*max(0., desired[:, 2]@actual[:, 2])
        return self.attitude_control(desired, thrust)

    def attitude_control(self, desired, thrust):
        v, g = self.vehicle, self.config["controller"]
        # A released velocity stick must not create an instantaneous tilt step.
        delta = Rotation.from_matrix(self.previous_desired.T@desired).as_rotvec()
        fraction = min(1., ATTITUDE_SLEW_RAD_S*DT/max(np.linalg.norm(delta), 1e-12))
        desired = self.previous_desired@Rotation.from_rotvec(fraction*delta).as_matrix()
        self.previous_desired = desired
        actual = sim.rotation(self.x[6:10])
        skew = .5*(desired.T@actual-actual.T@desired)
        ea = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        old_integral = self.integral[3:].copy()
        self.integral[3:] = np.clip(old_integral+DT*ea, -2, 2)
        frequency = np.array(g["attitude_frequency_rad_s"])
        torque = self.trim_control[1:]-v.inertia*(frequency**2*(ea+g["attitude_ki"]*self.integral[3:])
                                               +2*g["attitude_damping"]*frequency*self.x[10:13])
        torque += np.cross(self.x[10:13], v.inertia*self.x[10:13])
        self.torque_saturated = bool(np.any(np.abs(torque)>TORQUE_LIMIT_NM))
        # Do not integrate an error that pushes farther into torque saturation.
        blocked = (np.abs(torque)>TORQUE_LIMIT_NM)&(torque*(-ea)>0)
        self.integral[3:] = np.where(blocked, old_integral, self.integral[3:])
        torque = np.clip(torque, -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)
        squared = np.linalg.solve(self.allocation, np.r_[thrust, torque])
        self.motor_saturated = bool(np.any((squared<self.low**2)|(squared>self.high**2)))
        if self.motor_saturated:
            self.integral[3:] = old_integral
        self.saturated = self.torque_saturated or self.motor_saturated
        return sim.motor_command(np.sqrt(np.clip(squared, self.low**2, self.high**2)), self.voltage, self.config)

    def advance(self, request):
        if self.status not in ("ready", "running"):
            return self.snapshot()
        # Fixed 50 ms of simulation per request; no wall-clock time enters dynamics.
        u = {}
        for name, lo, hi, default in [("forward", -1, 1, 0), ("right", -1, 1, 0),
                                    ("climb", -1, 1, 0), ("yaw", -1, 1, 0),
                                    ("speed", .5, 5, 2), ("wind", 0, 8, 0),
                                    ("wind_heading", -180, 180, 0)]:
            value = float(request.get(name, default))
            if not np.isfinite(value):
                raise ValueError("Controls must be finite")
            u[name] = float(np.clip(value, lo, hi))
        wind_angle = np.deg2rad(u["wind_heading"])
        target_wind = u["wind"]*np.array([np.cos(wind_angle), np.sin(wind_angle), 0.])
        mode = request.get("mode", "velocity")
        if mode not in ("velocity", "attitude"):
            raise ValueError("Unknown control mode")
        self.mode = mode
        self.speed_command = u["speed"]
        self.inputs = np.array([u["forward"], u["right"], u["climb"], u["yaw"]])
        self.status = "running"
        for _ in range(10):
            try:
                self.wind += (1-np.exp(-DT/2))*(target_wind-self.wind)
                command = self.controller(u["forward"], u["right"], u["climb"], u["yaw"]*.6, u["speed"], mode)
                goal = sim.motor_speed(command, self.voltage, self.config)
                half = goal+(self.rotor-goal)*np.exp(-DT/(2*self.tau))
                dx, _ = self.dynamics(self.x, self.vehicle, self.rotor, (goal-self.rotor)/self.tau, self.wind)
                dx_mid, wrench = self.dynamics(self.x+DT*dx/2, self.vehicle, half, (goal-half)/self.tau, self.wind)
                candidate = self.x+DT*dx_mid
                candidate[6:10] /= np.linalg.norm(candidate[6:10])
                if not np.isfinite(candidate).all():
                    raise ValueError("数値が非有限値になりました")
                if np.linalg.norm(candidate[3:6]) > 20:
                    raise ValueError("速度が計算範囲を超えました")
                if sim.rotation(candidate[6:10])[2, 2] < np.cos(np.deg2rad(70)):
                    raise ValueError("傾きが70度を超えました")
                if np.linalg.norm(candidate[3:6]-self.wind) > self.config["airspeed_limit_m_s"]:
                    raise ValueError("対気速度がモデルの範囲11.5 m/sを超えました")
                self.x = candidate
                self.rotor = goal+(self.rotor-goal)*np.exp(-DT/self.tau)
                self.command, self.wrench = command, wrench
                self.steps += 1
                self.t = self.steps*DT
                if self.steps % 10 == 0:
                    self.log()
                if self.x[2] >= -.12:
                    self.status = "地面に到達しました。リセットで再開できます。"
                    self.log()
                    break
                if self.t >= 600:
                    self.status = "600秒の記録が終了しました。CSVを保存してリセットできます。"
                    break
            except (ValueError, FloatingPointError, OverflowError) as e:
                self.status = "計算停止: "+str(e)
                break
        return self.snapshot()

    def snapshot(self):
        r = sim.rotation(self.x[6:10])
        angles = np.rad2deg([np.arctan2(r[2, 1], r[2, 2]), np.arcsin(np.clip(-r[2, 0], -1, 1)), np.arctan2(r[1, 0], r[0, 0])])
        return dict(time=self.t, position=self.x[:3].tolist(), velocity=self.x[3:6].tolist(),
                    rotation=r.tolist(), angles=angles.tolist(), rate=self.x[10:13].tolist(),
                    rpm=(self.rotor*60/(2*np.pi)).tolist(), command=self.command.tolist(),
                    wrench=self.wrench.tolist(), wind=self.wind.tolist(), height_target=self.height,
                    heading_target=float(np.rad2deg(self.heading)), saturated=self.saturated,
                    torque_saturated=self.torque_saturated, motor_saturated=self.motor_saturated,
                    status=self.status, model=self.model, route=self.route, control_mode=self.mode,
                    input_axes=self.inputs.tolist(), trim_error=self.trim_error,
                    model_sha256=sim.sha(HERE/"model.json"))

    def log(self):
        self.rows.append(np.r_[self.t, self.x, self.rotor*60/(2*np.pi), self.command, self.wrench,
                               self.wind, self.height, self.heading, int(self.saturated),
                               self.inputs, int(self.mode == "attitude"), self.speed_command,
                               int(self.torque_saturated), int(self.motor_saturated)].tolist()
                         + [self.model, self.route])

    def csv(self):
        stream = io.StringIO(); writer = csv.writer(stream)
        writer.writerow(["time_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
                         "qw", "qx", "qy", "qz", "p_rad_s", "q_rad_s", "r_rad_s",
                         *[f"rpm_M{i}" for i in range(1, 5)], *[f"command_M{i}" for i in range(1, 5)],
                         *sim.AXES, "wind_x_m_s", "wind_y_m_s", "wind_z_m_s", "target_height_m", "target_heading_rad", "saturated",
                         "stick_forward", "stick_right", "stick_collective_or_climb", "stick_yaw", "attitude_mode",
                         "speed_command_m_s", "torque_saturated", "motor_saturated", "load_model", "coefficient_route"])
        writer.writerows(self.rows)
        return stream.getvalue()


def serve(port=8765, open_browser=True, flight_factory=Flight, html_path=None, metadata=None):
    flight, lock = flight_factory(), threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, content, content_type="application/json; charset=utf-8", status=200):
            data = content.encode() if isinstance(content, str) else content
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            with lock:
                if self.path == "/":
                    self.send((html_path or HERE/"interactive.html").read_bytes(), "text/html; charset=utf-8")
                elif self.path == "/state":
                    self.send(json.dumps(flight.snapshot(), allow_nan=False))
                elif self.path == "/trace.csv":
                    self.send(flight.csv(), "text/csv; charset=utf-8")
                elif self.path == "/session.json":
                    self.send(json.dumps(dict(config=flight.config, state=flight.snapshot(),
                        controller="interactive PI velocity/height + geometric attitude; not PX4", dt_s=DT,
                        interactive_control=dict(attitude_slew_rad_s=ATTITUDE_SLEW_RAD_S,
                                                 torque_limit_Nm=TORQUE_LIMIT_NM.tolist()),
                        backend=metadata or {"name": "Python rigid body"},
                        rpm_source="generated from command and voltage", validation=False,
                        code_sha256={n:sim.sha(HERE/n) for n in ("interactive.py", "simulate.py", "aerodynamics.py")}), indent=2))
                else:
                    self.send("Not found", "text/plain", 404)

        def do_POST(self):
            nonlocal flight
            # Only same-origin local requests; server never opens serial/UART/MAVLink.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{port}":
                self.send("Origin rejected", "text/plain", 403); return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if size > 4096:
                    raise ValueError("Request too large")
                data = json.loads(self.rfile.read(size))
                with lock:
                    if self.path == "/reset":
                        flight = flight_factory(data.get("model", "identified"), data.get("route", "command_route"))
                        result = flight.snapshot()
                    elif self.path == "/step":
                        result = flight.advance(data)
                    else:
                        self.send("Not found", "text/plain", 404); return
                self.send(json.dumps(result, allow_nan=False))
            except (ValueError, TypeError, KeyError) as e:
                self.send(json.dumps({"error": str(e)}), status=400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Simulator: http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(.4, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(args.port, not args.no_browser)
