"""Real PX4 standard cascaded controller + current frozen Python aircraft.

Localhost virtual sensors only. No physical Pixhawk or motor operation.
No model-dependent feedback/feedforward, coefficient refit or load anchoring.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analysis as a
from current_graybox_aircraft import CurrentAircraft
from simulate_graybox_forward_flight import Aircraft, inverse_motor_map
from matched_px4_plant import Plant, common_parameters
import run_matched_px4_sitl as launcher


def px4_parameters(artifact, gains="standard"):
    params=common_parameters(artifact)
    if gains=="standard":
        params={k:v for k,v in params.items() if not k.startswith("MC_")}
    elif gains=="tuned":
        # One fixed shared change, no aerodynamic feedforward or per-model tuning.
        params.update(MC_ROLL_P=4., MC_PITCH_P=4., MPC_XY_VEL_I_ACC=.8)
    cfg=artifact["settings"]
    hover=np.full(4,np.sqrt(cfg["mass_kg"]*9.80665/(4*cfg["kf"])))
    params["MPC_THR_HOVER"]=float(np.mean(inverse_motor_map(hover,16.,artifact["motor_models"])))
    return params


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("inputs","models","extra","px4-root","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--variant",choices=("physics","identified"),default="identified")
    p.add_argument("--route",choices=a.ROUTES,default="measured")
    p.add_argument("--gains",choices=("standard","conservative","tuned"),default="standard")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    inputs=a.read_json(args.inputs)
    cfg=a.Settings(**inputs["settings"])
    motors=inputs["motor_models"]
    models,extra=a.read_json(args.models),a.read_json(args.extra)
    artifact=dict(settings=asdict(cfg),motor_models=motors,models={})
    artifact_path=args.output/"aircraft.json"
    a.write_json(artifact_path,artifact)
    schedule_path=args.output/"adapter.json"
    a.write_json(schedule_path,dict(model_sha256=launcher.sha(artifact_path),
        note="No speed schedule. Current fixed-reference A+B composition."))

    def plant_factory(artifact,schedule,route):
        plant=Plant(artifact,{},"physical")
        plant.vehicle=(Aircraft(artifact,"physical",anchored=False) if args.variant=="physics" else
            CurrentAircraft(cfg,motors,models,extra,args.route,startup_blend=True))
        return plant

    launcher.main(["--px4-root",str(args.px4_root),"--models",str(artifact_path),
        "--schedule",str(schedule_path),"--route","physical" if args.variant=="physics" else args.route,
        "--output",str(args.output/"flight"),"--target-v","5","--acceleration","2.5",
        "--hold-s","1.5","--altitude","1.5"],
        plant_factory=plant_factory,
        parameters_factory=lambda artifact:px4_parameters(artifact,args.gains),
        metadata=dict(controller_policy="PX4 stock generic-quad gains" if args.gains=="standard" else
                      "Shared PX4 gains for 50ms motor delay; no per-model retune",
            controller_gains=args.gains,coefficient_source_sha256=launcher.sha(args.models),
            extra_source_sha256=launcher.sha(args.extra),initial_hover_anchor="NONE",
            plant="Current absolute A(initial hover)+B(state)-B(initial hover)",
            rpm_source="SIMULATION_GENERATED from shared command/voltage motor maps",
            limitations=["Real PX4 SITL controller/EKF, custom Python rigid-body plant; not native Gazebo.",
                "Generic-quad airframe defaults, not verified real-X500 tuning.",
                "Only geometry allocation and nominal hover thrust normalized to the current aircraft.",
                "Low-RPM startup 0.3..0.5 nominal hover speed transition is an explicit assumption.",
                "No free-flight accuracy or hardware safety claim."]))


if __name__=="__main__":
    main()
