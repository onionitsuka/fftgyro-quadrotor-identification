from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import SimulationAdapter
from .analysis_preparation import (
    ConfigurationError,
    DataIntegrityError,
    prepare_analysis,
)
from .allocation import build_plan as build_allocation_plan
from .allocation import summarize as summarize_allocation
from .angle_grid import PROFILES as ANGLE_GRID_PROFILES
from .angle_grid import summarize as summarize_angle_grid
from .calibration import build_factory_calibrated_plan
from .calibration import build_full_validation_plan
from .calibration import summarize as summarize_calibration
from .collective_mapping import build_collective_mapping
from .esc_readiness import ReadinessStage, evaluate_inventory
from .esc_readout import evaluate_readout
from .flight_stack_readiness import FlightStackStage, evaluate_flight_stack
from .flight_log import extract_ardupilot_flight_csv
from .hardware_status import evaluate_hardware_status
from .export import export_plan
from .integrated_rig import (
    IntegratedRigError,
    IntegratedRigRunner,
    load_integrated_rig_config,
)
from .motor_single import PROFILES, build_plan, summarize
from .lgr101 import LgrFormatError, inspect_lgr_file
from .lr8450_readiness import Lr8450Stage, evaluate_lr8450
from .power_supply_readiness import SupplyStage, evaluate_supply
from .rpm_calibration import calibrate_rpm_command_map
from .rpm_capture import capture_uno_serial
from .rpm_estimation import load_rpm_estimator
from .run_bundle import create_run_from_config, finalize_run
from .phase4 import build_plan as build_phase4_plan
from .phase4 import summarize as summarize_phase4
from .phase5 import build_phase5b_plan, build_phase5c_plan
from .phase5 import summarize as summarize_phase5
from .phase5 import summarize_phase5c
from .phase6 import build_phase6a_plan, build_phase6b_plan
from .phase6 import summarize_phase6a, summarize_phase6b
from .runner import ExperimentRunner
from .wind_quality import WindObservation, evaluate_wind_observation
from .wind_sensor_commissioning import (
    SensorMode,
    build_wind_sensor_commissioning_plan,
    summarize_wind_sensor_commissioning,
)
from .wind_sensor_readiness import WindSensorStage, evaluate_wind_sensors


def _comma_separated_floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated degree values") from exc


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated integer values") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fftgyro-exp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("plan", "export", "simulate"):
        sub = subparsers.add_parser(command)
        sub.add_argument(
            "protocol",
            choices=(
                "motor-single",
                "calibration",
                "allocation",
                "phase4",
                "phase5b",
                "phase5c",
                "phase6a",
                "phase6b",
            ),
        )
        sub.add_argument("--profile", choices=tuple(PROFILES), default="recommended")
        sub.add_argument(
            "--calibration-mode",
            choices=("factory", "full"),
            default="factory",
        )
        sub.add_argument(
            "--angle-profile",
            choices=tuple(ANGLE_GRID_PROFILES),
            default="sun-beta45-coarse",
        )
        sub.add_argument("--wind-speeds-mps", type=_comma_separated_floats)
        if command == "export":
            sub.add_argument("--output", type=Path, required=True)

    readiness = subparsers.add_parser("esc-readiness")
    readiness.add_argument("inventory", type=Path)
    readiness.add_argument(
        "--stage",
        choices=tuple(stage.value for stage in ReadinessStage),
        default=ReadinessStage.INSPECT.value,
    )
    readout = subparsers.add_parser("esc-readout")
    readout.add_argument("receipt", type=Path)
    collective = subparsers.add_parser("collective-map")
    collective.add_argument("configuration", type=Path)
    rpm_readiness = subparsers.add_parser("rpm-readiness")
    rpm_readiness.add_argument("configuration", type=Path)
    rpm_calibrate = subparsers.add_parser("rpm-calibrate")
    rpm_calibrate.add_argument("csv", type=Path)
    rpm_capture = subparsers.add_parser("rpm-capture")
    rpm_capture.add_argument("port")
    rpm_capture.add_argument("--output", type=Path, required=True)
    rpm_capture.add_argument("--duration-s", type=float)
    rpm_estimate = subparsers.add_parser("rpm-estimate")
    rpm_estimate.add_argument("configuration", type=Path)
    rpm_estimate.add_argument("--motor", required=True)
    rpm_estimate.add_argument("--command", dest="rpm_command", type=float, required=True)
    rpm_estimate.add_argument("--voltage-v", type=float, required=True)
    rpm_estimate.add_argument("--initial-rpm", type=float)
    rpm_estimate.add_argument("--dt-s", type=float)
    wind_check = subparsers.add_parser("wind-check")
    wind_check.add_argument("configuration", type=Path)
    wind_check.add_argument("--target", type=float, required=True)
    wind_check.add_argument("--q-derived", type=float, required=True)
    wind_check.add_argument("--sensor1", type=float, required=True)
    wind_check.add_argument("--sensor2", type=float)
    wind_sensor_plan = subparsers.add_parser("wind-sensor-plan")
    wind_sensor_plan.add_argument(
        "--mode", choices=tuple(item.value for item in SensorMode), required=True
    )
    wind_sensor_plan.add_argument(
        "--wind-speeds-mps", type=_comma_separated_floats, default=(0.0, 5.0, 10.0)
    )
    wind_sensor_plan.add_argument("--output", type=Path)
    wind_sensor_readiness = subparsers.add_parser("wind-sensor-readiness")
    wind_sensor_readiness.add_argument("configuration", type=Path)
    wind_sensor_readiness.add_argument(
        "--stage",
        choices=tuple(item.value for item in WindSensorStage),
        default=WindSensorStage.IDENTIFY.value,
    )
    hardware_status = subparsers.add_parser("hardware-status")
    hardware_status.add_argument(
        "--config-dir", type=Path, default=Path("configs/experiment_runner")
    )
    hardware_status.add_argument(
        "--defer",
        action="append",
        choices=(
            "power-supply",
            "flight-stack",
            "lr8450",
            "wind-sensors",
            "collective-mapping",
        ),
        default=[],
    )
    flight_stack = subparsers.add_parser("flight-stack-readiness")
    flight_stack.add_argument("configuration", type=Path)
    flight_stack.add_argument(
        "--stage",
        choices=tuple(item.value for item in FlightStackStage),
        default=FlightStackStage.IDENTIFY.value,
    )
    supply = subparsers.add_parser("supply-readiness")
    supply.add_argument("configuration", type=Path)
    supply.add_argument(
        "--stage",
        choices=tuple(stage.value for stage in SupplyStage),
        default=SupplyStage.IDENTIFY.value,
    )
    lgr = subparsers.add_parser("lgr-inspect")
    lgr.add_argument("csv", type=Path)
    lgr.add_argument("--required-rate", type=float, required=True)
    lr8450 = subparsers.add_parser("lr8450-readiness")
    lr8450.add_argument("configuration", type=Path)
    lr8450.add_argument(
        "--stage",
        choices=tuple(stage.value for stage in Lr8450Stage),
        default=Lr8450Stage.IDENTIFY.value,
    )
    angle_grid = subparsers.add_parser("angle-grid")
    angle_grid.add_argument("--profile", choices=tuple(ANGLE_GRID_PROFILES), required=True)
    angle_grid.add_argument("--verified-alpha-clearance", type=float)
    angle_grid.add_argument("--verified-beta-clearance", type=float)
    angle_grid.add_argument("--alpha-fixtures-verified", action="store_true")
    angle_grid.add_argument("--alpha-levels-deg", type=_comma_separated_floats)
    experiment_init = subparsers.add_parser("experiment-init")
    experiment_init.add_argument("configuration", type=Path)
    experiment_init.add_argument("--run-id", required=True)
    experiment_finalize = subparsers.add_parser("experiment-finalize")
    experiment_finalize.add_argument("configuration", type=Path)
    experiment_finalize.add_argument("run_root", type=Path)
    experiment_finalize.add_argument("--flight", type=Path, required=True)
    experiment_finalize.add_argument("--leptrino", type=Path, required=True)
    experiment_finalize.add_argument("--hioki", type=Path, required=True)
    experiment_finalize.add_argument("--arduino", type=Path, required=True)
    experiment_finalize.add_argument("--arm-index", type=int, default=0)
    experiment_finalize.add_argument("--allow-multiple-arm-windows", action="store_true")
    prepare = subparsers.add_parser(
        "prepare-analysis",
        help="inventory, verify, classify, and time-align captured run data",
    )
    prepare.add_argument("--runs-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--run-id")
    prepare.add_argument("--verify-only", action="store_true")
    flight_extract = subparsers.add_parser("flight-extract")
    flight_extract.add_argument("flight_log", type=Path)
    flight_extract.add_argument("--output", type=Path, required=True)
    flight_extract.add_argument("--arm-index", type=int, default=0)
    flight_extract.add_argument(
        "--motor-channels", type=_comma_separated_ints, default=(9, 10, 11, 12)
    )
    integrated_preflight = subparsers.add_parser(
        "preflight",
        help="check the Windows/Pi/Pixhawk/FFTGYRO/Leptrino integrated rig",
    )
    integrated_preflight.add_argument("configuration", type=Path)
    integrated_preflight.add_argument(
        "experiment_number",
        type=int,
        nargs="?",
        help="check one numbered experiment; omit to check the full catalog",
    )
    integrated_list = subparsers.add_parser(
        "list",
        help="show the ordered experiment catalog without touching hardware",
    )
    integrated_list.add_argument("configuration", type=Path)
    integrated_run = subparsers.add_parser(
        "run",
        help="execute one numbered PX4/Pi/FFTGYRO/Leptrino experiment",
    )
    integrated_run.add_argument("configuration", type=Path)
    integrated_run.add_argument(
        "experiment_number",
        type=int,
        help="number from the ordered [[experiments]] catalog",
    )
    integrated_run.add_argument(
        "--dry-run",
        action="store_true",
        help="perform the complete preflight without starting the executor",
    )
    integrated_status = subparsers.add_parser(
        "status",
        help="show Pi reachability and executor state without starting it",
    )
    integrated_status.add_argument("configuration", type=Path)
    integrated_stop = subparsers.add_parser(
        "stop",
        help="request the Pi executor abort ramp and release",
    )
    integrated_stop.add_argument("configuration", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()

    if args.command == "prepare-analysis":
        try:
            result = prepare_analysis(
                args.runs_root,
                args.output_root,
                run_id=args.run_id,
                verify_only=args.verify_only,
            )
        except DataIntegrityError as exc:
            print(json.dumps({"complete": False, "error": str(exc)}, ensure_ascii=False, indent=2))
            raise SystemExit(1) from exc
        except (ConfigurationError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            print(json.dumps({"complete": False, "error": str(exc)}, ensure_ascii=False, indent=2))
            raise SystemExit(2) from exc
        print(json.dumps({"complete": True, **result}, ensure_ascii=False, indent=2))
        raise SystemExit(1 if result["integrity_failures"] else 0)

    if args.command in {"preflight", "list", "run", "status", "stop"}:
        try:
            configuration = load_integrated_rig_config(args.configuration)
            runner = IntegratedRigRunner(configuration)
            if args.command == "preflight":
                report = runner.preflight(args.experiment_number)
                print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
                raise SystemExit(0 if report.ready else 2)
            if args.command == "list":
                print(
                    json.dumps(
                        {"experiments": runner.experiment_catalog()},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return
            if args.command == "status":
                print(json.dumps(runner.status(), ensure_ascii=False, indent=2))
                return
            if args.command == "stop":
                print(json.dumps(runner.stop(), ensure_ascii=False, indent=2))
                return
            roots = runner.run(
                args.experiment_number,
                dry_run=args.dry_run,
            )
            print(
                json.dumps(
                    {
                        "complete": True,
                        "dry_run": args.dry_run,
                        "experiment_number": args.experiment_number,
                        "run_roots": [str(path) for path in roots],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        except IntegratedRigError as exc:
            print(
                json.dumps(
                    {"complete": False, "error": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from exc
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(
                json.dumps(
                    {"complete": False, "error": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from exc

    if args.command == "experiment-init":
        try:
            session = create_run_from_config(args.configuration, args.run_id)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(json.dumps({"ready": False, "error": str(exc)}, ensure_ascii=False, indent=2))
            raise SystemExit(2) from exc
        print(
            json.dumps(
                {"ready": True, "run_id": session.metadata.run_id, "run_root": str(session.root)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "experiment-finalize":
        try:
            result = finalize_run(
                args.configuration,
                args.run_root,
                flight_path=args.flight,
                leptrino_path=args.leptrino,
                hioki_path=args.hioki,
                arduino_path=args.arduino,
                arm_index=args.arm_index,
                allow_multiple_arm_windows=args.allow_multiple_arm_windows,
            )
        except (OSError, KeyError, RuntimeError, TypeError, ValueError, UnicodeError) as exc:
            print(json.dumps({"ready": False, "error": str(exc)}, ensure_ascii=False, indent=2))
            raise SystemExit(2) from exc
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "flight-extract":
        try:
            result = extract_ardupilot_flight_csv(
                args.flight_log,
                args.output,
                arm_index=args.arm_index,
                motor_channels=args.motor_channels,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(json.dumps({"ready": False, "error": str(exc)}, ensure_ascii=False, indent=2))
            raise SystemExit(2) from exc
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "esc-readiness":
        result = evaluate_inventory(
            args.inventory,
            ReadinessStage(args.stage),
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ready else 2)

    if args.command == "esc-readout":
        result = evaluate_readout(args.receipt)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ready else 2)

    if args.command == "collective-map":
        result = build_collective_mapping(args.configuration)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ready else 2)

    if args.command == "rpm-calibrate":
        try:
            result = calibrate_rpm_command_map(args.csv)
        except (OSError, ValueError) as exc:
            print(
                json.dumps(
                    {"ready": False, "blockers": [f"rpm_calibration_error:{exc}"]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from exc
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "rpm-capture":
        try:
            event_count = capture_uno_serial(
                args.port,
                args.output,
                duration_s=args.duration_s,
            )
        except (OSError, RuntimeError, ValueError, UnicodeError) as exc:
            print(
                json.dumps(
                    {"ready": False, "blockers": [f"rpm_capture_error:{exc}"]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from exc
        print(
            json.dumps(
                {"ready": True, "events": event_count, "output": str(args.output)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command in {"rpm-readiness", "rpm-estimate"}:
        try:
            estimator, readiness = load_rpm_estimator(args.configuration)
        except (OSError, ValueError) as exc:
            print(
                json.dumps(
                    {"ready": False, "blockers": [f"rpm_estimator_error:{exc}"]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from exc
        if args.command == "rpm-readiness":
            print(json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2))
            raise SystemExit(0 if readiness.ready else 2)
        if not readiness.ready:
            print(json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2))
            raise SystemExit(2)
        try:
            result = estimator.estimate(
                args.motor,
                args.rpm_command,
                args.voltage_v,
                initial_rpm=args.initial_rpm,
                dt_s=args.dt_s,
            )
        except ValueError as exc:
            print(
                json.dumps(
                    {"ready": False, "blockers": [str(exc)]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from exc
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "wind-check":
        result = evaluate_wind_observation(
            args.configuration,
            WindObservation(
                target_mps=args.target,
                q_derived_mps=args.q_derived,
                sensor1_mps=args.sensor1,
                sensor2_mps=args.sensor2,
            ),
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ready_for_identification else 2)

    if args.command == "wind-sensor-plan":
        segments = build_wind_sensor_commissioning_plan(
            args.mode, wind_speeds_mps=args.wind_speeds_mps
        )
        summary = summarize_wind_sensor_commissioning(args.mode, segments)
        if args.output is not None:
            manifest_path, summary_path = export_plan(args.output, segments, summary)
            print(manifest_path)
            print(summary_path)
        else:
            print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "wind-sensor-readiness":
        result = evaluate_wind_sensors(
            args.configuration, WindSensorStage(args.stage)
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ready else 2)

    if args.command == "hardware-status":
        result = evaluate_hardware_status(
            args.config_dir, deferred=tuple(args.defer)
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ready else 2)

    if args.command == "flight-stack-readiness":
        result = evaluate_flight_stack(
            args.configuration, FlightStackStage(args.stage)
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ready else 2)

    if args.command == "supply-readiness":
        result = evaluate_supply(
            args.configuration,
            SupplyStage(args.stage),
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ready else 2)

    if args.command == "lgr-inspect":
        try:
            result = inspect_lgr_file(
                args.csv,
                required_sample_rate_hz=args.required_rate,
            )
        except (OSError, LgrFormatError) as exc:
            print(
                json.dumps(
                    {"ready": False, "blockers": [f"lgr_file_error:{exc}"]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from exc
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ready else 2)

    if args.command == "lr8450-readiness":
        result = evaluate_lr8450(
            args.configuration,
            Lr8450Stage(args.stage),
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ready else 2)

    if args.command == "angle-grid":
        result = summarize_angle_grid(
            args.profile,
            verified_abs_alpha_clearance_deg=args.verified_alpha_clearance,
            verified_abs_beta_clearance_deg=args.verified_beta_clearance,
            alpha_fixture_levels_verified=args.alpha_fixtures_verified,
            alpha_fixture_levels_deg=args.alpha_levels_deg,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.protocol == "motor-single":
        segments = build_plan(args.profile)
        summary = summarize(args.profile, segments)
    elif args.protocol == "calibration":
        if args.calibration_mode == "factory":
            segments = build_factory_calibrated_plan()
        else:
            segments = build_full_validation_plan()
        summary = summarize_calibration(segments, mode=args.calibration_mode)
    elif args.protocol == "phase4":
        segments = build_phase4_plan(args.profile)
        summary = summarize_phase4(args.profile, segments)
    elif args.protocol == "phase5b":
        if args.profile != "minimum":
            raise SystemExit(
                "Phase 5B recommended differential-input selection is not fully "
                "defined by the PDF; use --profile minimum for the neutral grid"
            )
        phase5_profile = "minimum"
        segments = build_phase5b_plan(phase5_profile, args.angle_profile)
        summary = summarize_phase5(phase5_profile, args.angle_profile, segments)
    elif args.protocol == "phase5c":
        segments = build_phase5c_plan(args.profile)
        summary = summarize_phase5c(args.profile, segments)
    elif args.protocol == "phase6a":
        segments = build_phase6a_plan(
            args.profile,
            args.angle_profile,
            wind_speeds_mps=args.wind_speeds_mps,
        )
        summary = summarize_phase6a(args.profile, args.angle_profile, segments)
    elif args.protocol == "phase6b":
        segments = build_phase6b_plan(args.profile, args.wind_speeds_mps)
        summary = summarize_phase6b(args.profile, segments)
    else:
        segments = build_allocation_plan()
        summary = summarize_allocation(segments)

    if args.command == "plan":
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "simulate":
        if args.protocol != "motor-single":
            raise SystemExit(
                f"{args.protocol} remains symbolic until its operator values are set; "
                "physical/simulated command execution is disabled"
            )
        result = ExperimentRunner(SimulationAdapter()).run_simulation(segments)
        print(
            json.dumps(
                {
                    "status": result.status.value,
                    "completed_segments": result.completed_segments,
                    "physical_output": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    manifest_path, summary_path = export_plan(args.output, segments, summary)
    print(manifest_path)
    print(summary_path)


if __name__ == "__main__":
    main()
