from dataclasses import asdict
import numpy as np

import analysis as a
from scripts.run_current_px4_sitl import px4_parameters


def fixture():
    return dict(settings=asdict(a.Settings()),motor_models=[
        dict(map_coefficients=[3000.,5000.,2000.]) for _ in range(4)])


def test_standard_keeps_px4_controller_gains_and_corrects_geometry_only():
    params=px4_parameters(fixture())
    assert not any(k.startswith("MC_") for k in params)
    assert 0<params["MPC_THR_HOVER"]<1
    assert params["CA_ROTOR0_CT"]==6.5
    assert abs(params["CA_ROTOR0_KM"])*6.5>.05


def test_conservative_is_one_shared_parameter_set_not_a_load_model_fit():
    params=px4_parameters(fixture(),"conservative")
    assert params["MC_PITCHRATE_P"]==.06
    assert params["MC_PITCHRATE_I"]==.05


def test_tuned_adjusts_outer_response_without_changing_inner_rate_gains():
    low=px4_parameters(fixture(),"conservative")
    tuned=px4_parameters(fixture(),"tuned")
    assert tuned["MC_PITCH_P"]==4. and tuned["MPC_XY_VEL_I_ACC"]==.8
    assert tuned["MC_PITCHRATE_P"]==low["MC_PITCHRATE_P"]
    assert tuned["MC_PITCHRATE_I"]==low["MC_PITCHRATE_I"]
