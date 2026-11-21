"""Gazebo Python system: replace native rotor aero with frozen identified aero.

Native thrust, reaction torque, joints, sensors and integration are untouched.
Only add (identified residual - native drag/rolling moment). Adding identified
loads on top of native drag without subtraction would double count aero.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from gz.math7 import Vector3d
from gz.sim8 import Link, Model
from speed_scheduled_graybox import ScheduledAircraft
from matched_px4_plant import GAZEBO_AERO

FLIP = np.array([1.,-1.,-1.])


def array(vector):
    return np.array([vector.x(),vector.y(),vector.z()])


def vector(values):
    return Vector3d(*(float(x) for x in values))


class IdentifiedSystem:
    def configure(self, entity, sdf, ecm, event_mgr):
        self.model = Model(entity)
        self.body = Link(self.model.link_by_name(ecm,"base_link"))
        self.rotors = [Link(self.model.link_by_name(ecm,f"rotor_{j}")) for j in range(4)]
        for link in [self.body,*self.rotors]:
            link.enable_velocity_checks(ecm,True)
        self.route = sdf.get_string("coefficient_route")
        self.output = Path(sdf.get_string("output"))
        self.aircraft = None
        if self.route != "baseline":
            artifact = json.loads(Path(sdf.get_string("models")).read_text())
            schedule = json.loads(Path(sdf.get_string("schedule")).read_text())
            self.aircraft = ScheduledAircraft(artifact,self.route,schedule)
        self.failed = False
        self.last_time = None
        self.stream = (self.output / "applied_aero.csv").open("w",newline="")
        self.writer = csv.writer(self.stream)
        self.writer.writerow(["sim_time_s","active",*[f"rotor{j}_rad_s" for j in range(4)],
            "delta_Fx","delta_Fy","delta_Fz","delta_Mx","delta_My","delta_Mz"])
        (self.output / "plugin_ready.json").write_text(json.dumps(dict(route=self.route,
            plugin_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            rpm_source="Native rotor-link minus base-link angular velocity, times slowdown 10",
            runup_gate="Identified aero excluded below 0.5 nominal hover RMS rotor speed",
            loads="Native propulsion + identified residual; native aero subtracted when active",
            anchor="Frozen assumed initial-hover anchor; not absolute aero validation")))

    def pre_update(self, info, ecm):
        # PX4 lockstep can trigger zero-dt callbacks at a repeated clock. Never
        # accumulate the same external wrench more than once per physics step.
        time = info.sim_time.total_seconds()
        if info.paused or self.failed or info.dt.total_seconds() <= 0 or time == self.last_time:
            return
        self.last_time = time
        try:
            self.apply_step(info,ecm)
        except Exception as error:
            self.failed = True
            self.stream.flush()
            (self.output / "plugin_failure.json").write_text(json.dumps(dict(
                error=repr(error), sim_time_s=info.sim_time.total_seconds())))
            # Loader can swallow exceptions; the launcher watches the marker and
            # stops the run instead of reporting a silently disabled model.

    # Do not name this helper `update`: PythonSystemLoader registers that name
    # as a second simulator callback, causing double wrench application.
    def apply_step(self, info, ecm):
        pose = self.body.world_pose(ecm)
        velocity = self.body.world_linear_velocity(ecm)
        rate = self.body.world_angular_velocity(ecm)
        if pose is None or velocity is None or rate is None:
            return
        attitude = pose.rot()
        body_rate = array(attitude.rotate_vector_reverse(rate))
        body_air = array(attitude.rotate_vector_reverse(velocity))
        rotor, native_force, native_moment = [], np.zeros(3), np.zeros(3)
        for link in self.rotors:
            angular = link.world_angular_velocity(ecm)
            linear = link.world_linear_velocity(ecm)
            rotor_pose = link.world_pose(ecm)
            if angular is None or linear is None or rotor_pose is None:
                return
            axis = array(rotor_pose.rot().rotate_vector(Vector3d(0,0,1)))
            real_speed = abs((array(angular)-array(rate)) @ axis) * 10
            rotor.append(real_speed)
            relative = array(linear)  # zero ambient wind in this experiment
            perpendicular = relative - (relative @ axis)*axis
            drag = -GAZEBO_AERO["rotor_drag_coefficient"] * real_speed * perpendicular
            rolling = -GAZEBO_AERO["rolling_moment_coefficient"] * real_speed * perpendicular
            offset = array(rotor_pose.pos()) - array(pose.pos())
            native_force += drag
            native_moment += np.cross(offset,drag)+rolling
        rotor = np.array(rotor)
        delta = np.zeros(6)
        active = self.aircraft is not None and np.sqrt(np.mean(rotor**2)) >= .5*self.aircraft.hover_w
        if active:
            air, omega = body_air*FLIP, body_rate*FLIP
            residual = (self.aircraft.loads(air,omega,rotor,np.zeros(4)) -
                        self.aircraft.physics.loads(air,omega,rotor,np.zeros(4)))
            native = np.r_[array(attitude.rotate_vector_reverse(vector(native_force)))*FLIP,
                           array(attitude.rotate_vector_reverse(vector(native_moment)))*FLIP]
            delta = residual-native
            if not np.isfinite(delta).all() or np.linalg.norm(air)>11.5:
                raise ValueError("Identified load outside finite/speed support")
            if max(abs(pose.rot().euler().x()),abs(pose.rot().euler().y())) > np.deg2rad(70):
                raise ValueError("Native identified run roll/pitch exceeds 70 degrees")
            self.body.add_world_wrench(ecm,
                attitude.rotate_vector(vector(delta[:3]*FLIP)),
                attitude.rotate_vector(vector(delta[3:]*FLIP)))
        if info.iterations % 5 == 0:
            self.writer.writerow([info.sim_time.total_seconds(),int(active),*rotor,*delta])
            self.stream.flush()


def get_system():
    return IdentifiedSystem()
