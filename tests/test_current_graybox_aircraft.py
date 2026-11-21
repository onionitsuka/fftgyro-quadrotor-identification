from dataclasses import asdict, replace

import numpy as np
import pytest

import analysis as a
import analysis_search as search
from scripts.current_graybox_aircraft import instantaneous_data, selected_features, CurrentAircraft
from scripts import simulate_graybox_forward_flight as flight
from scripts.simulate_current_graybox import takeoff_height


def data_fixture():
    cfg = replace(a.Settings(), fz_model="sun_graybox", yaw_motion_model="sun_blended")
    air, rate, rotor = np.array([5., -2., -.3]), np.array([.1, -.2, .3]), np.array([380.,420.,400.,410.])
    physics = flight.Aircraft(dict(settings=asdict(cfg),models={},motor_models=[]), "physical", anchored=False)
    physical = physics.loads(air, rate, rotor, np.zeros(4))
    return cfg, air, rate, rotor, instantaneous_data(air,rate,rotor,np.zeros(4),cfg,physical)


def test_instantaneous_state_matches_constant_log_pipeline_without_filter_delay():
    cfg,air,rate,rotor,data = data_fixture()
    raw = dict(air=np.tile(air,(160,1)),omega=np.tile(rate,(160,1)),valid_motion=np.ones(160,bool))
    expected = a.route_state(raw,np.tile(rotor*60/(2*np.pi),(160,1)),cfg)
    for key,value in expected["state"].items():
        np.testing.assert_allclose(data["state"][key][0],value[80],atol=1e-11,rtol=1e-10,err_msg=key)
    np.testing.assert_allclose(data["physical"][0],expected["physical"][80],atol=1e-10)


@pytest.mark.parametrize("output,names",[
    ("My",["intercept","mu_x^2*mu_z^2","qbar","cross__mu_x*abs_ur"]),
    ("Mz",["rotor3_torque","flow_000_u","body_h4_sin","speed_mu_y","local1_ur",
           "theory__rotor_plane_linear","localflow__2__0__up"]),
    ("Fz",["prop_gain","inflow_delta","mu_total2","cross__mu_z^4","theory__body_drag__nz"]),
    ("Fx",["intercept","rotor_inplane_linear","mu_z^3","1"]),
])
def test_saved_terms_equal_analysis_dictionary(output,names):
    cfg,_,_,_,data=data_fixture()
    expected,_=search.expanded_bank(data,cfg,output)
    actual=selected_features(data,cfg,output,dict(feature_names=names,spec="expanded"))
    for key in names:
        np.testing.assert_allclose(actual[key],np.ones(1) if key=="intercept" else expected[key],atol=1e-12)


def test_fixed_reference_composition_and_explicit_anchor():
    cfg=a.Settings()
    model=lambda x:dict(feature_names=["intercept"],coefficient_by_name={"intercept":x},spec="compact")
    main=dict(foundation={},increment={},yaw={"final/measured":model(.5)})
    extra=dict(extra={})
    for out in a.CHANNELS:
        key="final/measured/"+out
        main["foundation"][key]=model(.1)
        main["increment"][key]=model(.5 if out!="Fz" else .1)
    for out in a.EXTRA_CHANNELS:
        for family in ("foundation","increment"):
            extra["extra"][f"final/measured/{out}/{family}"]=model(.1 if family=="foundation" else .5)
    absolute=CurrentAircraft(cfg,[],main,extra,"measured")
    conditioned=CurrentAircraft(cfg,[],main,extra,"measured",conditioned=True)
    w=np.full(4,absolute.hover_w)
    p=absolute.physics.loads(np.zeros(3),np.zeros(3),w,np.zeros(4))
    np.testing.assert_allclose(absolute.unanchored_hover,p+.1,atol=1e-12)
    np.testing.assert_allclose(conditioned.loads(np.zeros(3),np.zeros(3),w,np.zeros(4)),p,atol=1e-12)
    with pytest.raises(ValueError,match="11.5"):
        absolute.loads(np.array([12.,0.,0.]),np.zeros(3),w,np.zeros(4))


def test_takeoff_reference_and_ground_reaction():
    assert takeoff_height(0)==(0.,0.)
    assert takeoff_height(4)==(1.5,0.)
    assert takeoff_height(10)==(1.5,0.)
    cfg=a.Settings()
    vehicle=flight.Aircraft(dict(settings=asdict(cfg),motor_models=[],models={}),"physical",anchored=False)
    state=np.zeros(13); state[6]=1.
    dx,_=flight.rhs(state,vehicle,np.zeros(4),np.zeros(4),ground_contact=True)
    np.testing.assert_array_equal(dx,np.zeros(13))
    dx,_=flight.rhs(state,vehicle,np.full(4,vehicle.hover_w*1.1),np.zeros(4),ground_contact=True)
    assert dx[5]<0  # Release ground constraint when thrust exceeds weight.
