"""Current A(reference)+B(state)-B(reference) loads for offline free-flight.

No reconstruction, sensor zero, future RPM, filtering or coefficient fitting.
The fixed reference is initial hover, not a continuously reset local trim.
"""
from dataclasses import asdict
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a
from scripts import simulate_graybox_forward_flight as flight


def instantaneous_data(air, rate, rotor, rotor_dot, cfg, physical):
    """Same nondimensional variables as analysis.route_state, without log filters."""
    rotor = np.asarray(rotor, float)
    if not np.isfinite(rotor).all() or np.any(rotor <= 0):
        raise ValueError("Positive finite simulation RPM is required; no RPM floor")
    air, rate = np.asarray(air, float), np.asarray(rate, float)
    if not np.isfinite(np.r_[air, rate, rotor_dot]).all():
        raise ValueError("Nonfinite simulated state")
    bar = np.sqrt(np.mean(rotor**2))
    denominator = bar*cfg.rotor_radius_m
    mu = air[None, :]/denominator
    pqr = rate[None, :]*cfg.arm_xy_m/denominator
    state = dict(zip(("mu_x", "mu_y", "mu_z"), mu.T))
    state.update(zip(("pbar", "qbar", "rbar"), pqr.T))
    for k, signs in (("up", flight.ROLL), ("uq", flight.PITCH), ("ur", a.SIGNS)):
        state[k] = np.array([rotor**2 @ signs/bar**2])
    for k in ("mu_x", "mu_y", "pbar", "qbar", "rbar", "up", "uq", "ur"):
        state["abs_"+k] = np.abs(state[k])
    raw = dict(air=air[None, :], omega=rate[None, :])
    flow = a.rotor_local_flow(raw, rotor[None, :], cfg)
    state.update(a.rotor_local_yaw(raw, rotor[None, :], cfg, flow=flow))
    state.update(a.sun_yaw_regions(mu, pqr[:, 2], state["ur"], raw["omega"][:, 2], cfg))
    induced = flight.inflow(mu[0], cfg.kf/(cfg.rho*np.pi*cfg.rotor_radius_m**4))
    state.update(inflow=np.array([induced]), mu_h2=mu[:, 0]**2+mu[:, 1]**2)
    scale = np.array([cfg.rho*4*np.pi*cfg.rotor_radius_m**2*denominator**2])
    return dict(state=state, physical=np.asarray(physical)[None, :],
        force_scale=scale, moment_scale=scale*cfg.arm_xy_m,
        rotor_flow=flow, air_frd_m_s=raw["air"], omega_body_rad_s=raw["omega"],
        rpm=rotor[None, :]*60/(2*np.pi),
        rms_speed_ratio=np.array([bar/(4000*2*np.pi/60)]))


def selected_features(data, cfg, output, model):
    """Evaluate only saved terms; reuse original generators for special families."""
    names = model["feature_names"]
    state = dict(data["state"])
    scale = data["force_scale"] if output.startswith("F") else data["moment_scale"]
    state["inflow_delta"] = state["inflow"]-np.sqrt(
        cfg.kf/(cfg.rho*np.pi*cfg.rotor_radius_m**4)/2)
    state["mu_total2"] = sum(state[k]**2 for k in ("mu_x", "mu_y", "mu_z"))
    bank = dict(intercept=np.ones(1), prop_gain=data["physical"][:, a.OUTPUTS.index(output)])
    if output in a.EXTRA_CHANNELS:
        bank.update(a.extra_feature_bank(data, output, model["spec"], cfg))
    if output in ("My", "Mz"):
        bank.update(a.speed_features(data, output))
    if output == "Mz":
        bank.update(a.continuous_bank(data))
        bank.update(a.physics_bank(data, cfg, a.SIGNS))
    if any(k.startswith("theory__") for k in names):
        bank.update(a.theory_feature_bank(data, cfg, output))
    if any(k.startswith(("individual__", "rotor_sum__", "coupled__")) for k in names):
        from scripts.six_axis_theory_bank import feature_bank
        bank.update(feature_bank(data, cfg, output)[0])
    for key in names:
        if key in bank:
            continue
        if key.startswith("localflow__"):
            _, i, j, motion = key.split("__")
            flow = data["rotor_flow"]
            value = scale*flow["omega_squared_weight"][:, int(i)]/4*flow["mu"][:, int(i), int(j)]
            bank[key] = value if motion == "1" else value*state[motion]
        else:
            plain = key.removeprefix("cross__")
            bank[key] = a.evaluate_names(state, scale, [plain])[plain]
    return {k: bank[k] for k in names}


def correction(data, cfg, output, model):
    bank = selected_features(data, cfg, output, model)
    return float(sum(value*bank[key][0] for key, value in model["coefficient_by_name"].items()))


class CurrentAircraft:
    """Faithful fixed-reference composition, with optional EXPLICIT hover anchor."""
    def __init__(self, cfg, motors, models, extra_models, route, *, conditioned=False, startup_blend=False):
        self.settings, self.cfg, self.route = cfg, asdict(cfg), route
        self.motors, self.mass = motors, cfg.mass_kg
        self.inertia = np.asarray(cfg.inertia_kg_m2)
        self.physics = flight.Aircraft(dict(settings=self.cfg, motor_models=motors, models={}),
                                       "physical", anchored=False)
        self.hover_w = self.physics.hover_w
        self.startup_blend = startup_blend
        self.anchor = np.zeros(6)
        self.pairs = []
        for out in a.OUTPUTS:
            if out in a.EXTRA_CHANNELS:
                prefix = f"final/{route}/{out}/"
                pair = (extra_models["extra"][prefix+"foundation"],
                        extra_models["extra"][prefix+"increment"])
            else:
                key = f"final/{route}/{out}"
                pair = (models["foundation"][key],
                        models["yaw"][f"final/{route}"] if out == "Mz" else
                        models["increment"][key] if out == "My" else models["foundation"][key])
            self.pairs.append(pair)
        w = np.full(4, self.hover_w)
        p0 = self.physics.loads(np.zeros(3), np.zeros(3), w, np.zeros(4))
        ref = instantaneous_data(np.zeros(3), np.zeros(3), w, np.zeros(4), cfg, p0)
        self.offset = np.array([correction(ref, cfg, out, pair[0])-correction(ref, cfg, out, pair[1])
                                for out, pair in zip(a.OUTPUTS, self.pairs)])
        self.unanchored_hover = self.loads(np.zeros(3), np.zeros(3), w, np.zeros(4))
        if conditioned:
            self.anchor = np.r_[0., 0., -self.mass*flight.G, 0., 0., 0.]-self.unanchored_hover

    def loads(self, air, omega, rotor, rotor_dot):
        if np.linalg.norm(air) > 11.5:
            raise ValueError("STOPPED: airspeed exceeds tested 11.5 m/s; no extrapolation")
        physical = self.physics.loads(air, omega, rotor, rotor_dot)
        blend = float(np.clip((np.min(rotor)/self.hover_w-.3)/.2,0.,1.)) if self.startup_blend else 1.
        if blend == 0:
            return physical
        data = instantaneous_data(air, omega, rotor, rotor_dot, self.settings, physical)
        delta = np.array([correction(data, self.settings, out, pair[1])
                          for out, pair in zip(a.OUTPUTS, self.pairs)])
        result = physical+blend*(self.offset+delta+self.anchor)
        if not np.isfinite(result).all():
            raise ValueError("STOPPED: nonfinite predicted wrench")
        return result
