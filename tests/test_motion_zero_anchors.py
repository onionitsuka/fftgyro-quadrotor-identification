from dataclasses import replace
import copy
import numpy as np

import analysis as a
from scripts.recover_motion_zero_anchors import Limits, epoch_validation, motion_bias, predict_sensor_load


def mechanics_case():
    cfg=a.Settings(axis_roll_deg=-.725, pivot_flu_m=(0,0,.1046))
    t=np.arange(0,8,.01)
    theta=.15*np.sin(t)
    psi=.1*np.cos(.7*t)
    td=.15*np.cos(t); pd=-.07*np.sin(.7*t)
    tdd=-.15*np.sin(t); pdd=-.049*np.cos(.7*t)
    w=np.c_[np.sin(psi)*td,np.cos(psi)*td,pd]
    alpha=np.c_[np.sin(psi)*tdd+np.cos(psi)*pd*td,
                np.cos(psi)*tdd-np.sin(psi)*pd*td,pdd]
    sensor=dict(mass=.072,h=np.array([1e-5,2e-5,-.0006]),inertia=np.diag([8e-5,3e-5,3e-5]))
    raw=dict(t=t,omega=w,alpha=alpha,rotation=a.rig_rotation(theta,psi,0.,cfg),
             filtered_load=np.zeros((len(t),6)),wind=0.)
    return cfg,sensor,raw


def test_moving_sensor_inverse_matches_newton_euler_and_known_zero():
    cfg,sensor,raw=mechanics_case()
    cg=np.array(cfg.cg_flu_m)*a.FLU_TO_FRD
    radius=cg-np.array(cfg.pivot_flu_m)*a.FLU_TO_FRD
    w,alpha=raw['omega'],raw['alpha']
    ag=np.cross(alpha,radius)+np.cross(w,np.cross(w,radius))
    force=cfg.mass_kg*(ag-9.80665*raw['rotation'][:,2,:])
    moment=alpha*np.array(cfg.inertia_kg_m2)+np.cross(w,w*np.array(cfg.inertia_kg_m2))+np.cross(cg,force)
    expected=a.mechanics_filter(-np.c_[force,moment]*np.tile(a.FLU_TO_FRD,2),cfg)
    np.testing.assert_allclose(predict_sensor_load(raw,cfg,sensor),expected,atol=2e-11)
    known=np.array([.3,-.7,4.,.03,-.03,.067])
    raw['filtered_load']=expected+known
    np.testing.assert_allclose(motion_bias(raw,cfg,sensor),np.broadcast_to(known,expected.shape),atol=2e-11)


def test_new_epoch_needs_other_run_and_does_not_hide_its_other_endpoint():
    cfg,sensor,_=mechanics_case()
    def window(run,mz):
        return dict(run=run,bias_sensor_FLU=[0,0,0,0,0,mz])
    assert not epoch_validation([window('a',0),window('a',0)],cfg,sensor,Limits())['accepted']
    out=epoch_validation([window('a',0),window('a',0),window('b',.02)],cfg,sensor,Limits())
    assert not out['accepted']
    np.testing.assert_allclose(out['leave_run_out_rmse'][5],.02)
    assert epoch_validation([window('a',.001),window('b',.002)],cfg,sensor,Limits())['accepted']


def test_recovered_reference_is_labelled_and_does_not_replace_static_zero():
    profile=dict(anchors=[dict(epoch='old',aircraft_boot_id='clock')],runs=[
        dict(run='old',key=['old','clock'],anchored=True),
        dict(run='new',key=['new','clock'],anchored=False)])
    original=copy.deepcopy(profile['anchors'])
    recovery=dict(schema='fftgyro-motion-zero-audit-v1',profile_sha256='hash',
        epochs=[dict(key=['new','clock'],accepted=True,old_anchored=False)],
        accepted_anchors=[dict(epoch='new',aircraft_boot_id='clock',wind=0,window_consistent=True)],
        newly_supported_runs=['new'])
    out=a.apply_zero_recovery(profile,recovery,'hash')
    assert profile['anchors'][:1]==original
    assert profile['runs'][1]['absolute_reference_kind']=='motion_corrected_absolute'
    assert out['new_runs']==['new']


def test_interpolation_covariance_correlates_runs_and_grows_outside_anchors():
    observations = np.repeat((np.eye(6)*.01)[None], 2, axis=0)
    cov = a.interpolated_zero_covariance([1., 2.], [1., 1.5, 2., 5., 10.],
                                        observations, np.full(6, .02))
    np.testing.assert_allclose(cov[0, :, 0, :], observations[0], atol=1e-12)
    np.testing.assert_allclose(cov[2, :, 2, :], observations[1], atol=1e-12)
    assert cov[1, 0, 2, 0] > 0
    assert cov[4, 0, 4, 0] > cov[3, 0, 3, 0] > cov[2, 0, 2, 0]
    flat = cov.reshape(30, 30)
    np.testing.assert_allclose(flat, flat.T, atol=1e-12)
    assert np.linalg.eigvalsh(flat).min() >= -1e-12
    # Translating the arbitrary clock origin changes nothing.
    other = a.interpolated_zero_covariance([101., 102.], [101., 101.5, 102., 105., 110.],
                                          observations, np.full(6, .02))
    np.testing.assert_allclose(other, cov, atol=1e-12)


def test_new_foundation_targets_keep_validation_conditions_out():
    runs = [dict(run=name, fold=fold, condition_id=cond, channels={'My':{'reference_available':True}},
                 absolute_reference_kind='motion_corrected_absolute')
            for name, fold, cond in [('a', 0, 'same'), ('b', 0, 'same'), ('c', 1, 'other')]]
    profile = dict(runs=runs, models={'foundation':{
        '0/measured/My':{'training_runs':['old']},
        '1/measured/My':{'training_runs':['old']},
        'final/measured/My':{'training_runs':['old']}}})
    a.extend_foundation_training(profile)
    models = profile['models']['foundation']
    assert models['0/measured/My']['training_runs'] == ['c', 'old']
    assert models['1/measured/My']['training_runs'] == ['a', 'b', 'old']
    assert models['final/measured/My']['training_runs'] == ['a', 'b', 'c', 'old']
    a.check_fold(models['0/measured/My'], runs[0], 'grouped_OOF', runs)


def test_calibration_extension_preserves_old_block_and_zero_values():
    from dataclasses import asdict
    cfg = a.Settings()
    c = dict(mass_kg=.02, first_moment_x_kgm=0., first_moment_y_kgm=0., first_moment_z_kgm=0.,
             Ixx_kgm2=.00001, Iyy_kgm2=.00001, Izz_kgm2=.00001,
             Ixy_kgm2=0., Ixz_kgm2=0., Iyz_kgm2=0.)
    observations = [dict(epoch='new', aircraft_boot_id='clock', center_ns=t*60e9,
                         block_bias_sensor_FLU=[[0.]*6, [.001]*6, [-.001]*6]) for t in (1., 2.)]
    rec = dict(accepted_anchors=observations, newly_supported_runs=['b','c'],
               epochs=[dict(key=['new','clock'],accepted=True,old_anchored=False,
                            leave_run_out_rmse=[.1,.1,.1,.01,.01,.001])])
    profile = dict(settings=asdict(cfg), sensor_summary={'models':{'rigid':{'coefficients':c}}},
        calibration=dict(ids=['old'], covariance={k:np.array([[.123]]) for k in ('My','Mz','Fz')}),
        runs=[dict(run=name,key=['new','clock'],center=t,absolute_reference_kind='motion_corrected_absolute')
              for name,t in [('b',1.5),('c',3.)]])
    original = copy.deepcopy(observations)
    a.extend_recovered_calibration(profile,rec,{'slope_prior_sigma':[.001]*6})
    assert profile['calibration']['ids']==['old','b','c']
    assert observations==original
    for cov in profile['calibration']['covariance'].values():
        assert cov[0,0]==.123 and np.all(cov[0,1:]==0) and cov[1,2]>0
        assert np.linalg.eigvalsh(cov).min()>0
