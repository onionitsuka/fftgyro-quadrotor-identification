"""Initial-load foundation tuning: algebra, split isolation, no target leakage."""
import copy

import numpy as np

import analysis as a


def case():
    runs = []
    for i in range(14):
        x = np.linspace(-1, 1, 9)+i/10
        xref = -.8+i/10
        # Deliberately different state sampling and initial target.
        d = dict(features={'test':{'intercept':np.ones(9),'x':x}},
                 reference_features={'test':{'intercept':1.,'x':xref}},
                 physical=np.zeros(9), reference_physical=0.)
        ch = dict(take=np.ones(9,dtype=bool),truth=1+2*x,
                  reference_available=True,reference=1+2*xref, measured=d)
        runs.append(dict(run=f'r{i}',condition_id=f'c{i//2}', anchored=True,
                         absolute_reference_kind=('independent_absolute' if i%2 else 'motion_corrected_absolute'),
                         channels={'My':ch}))
    template = dict(training_runs=[r['run'] for r in runs[:12]],
                    feature_names=['intercept','x'],spec='test',loss='equal_run_absolute_point',ridge=.1)
    return runs, template


def test_reference_loss_preserves_total_weight_per_run():
    runs, template = case()
    template = dict(template,training_runs=['r0'],reference_fraction=1.)
    x,y,names,audit = a.assemble_foundation(runs,{},'measured','My',template)
    np.testing.assert_allclose(x,[[1.,-.8]])
    np.testing.assert_allclose(y,[-.6])
    assert names==('intercept','x')


def test_mixed_design_matches_declared_weighted_loss():
    runs, template = case()
    template = dict(template,training_runs=['r0'],reference_fraction=.3)
    x,y,_,_ = a.assemble_foundation(runs,{},'measured','My',template)
    beta = np.array([.5,1.3])
    state = runs[0]['channels']['My']['measured']['features']['test']['x']
    point = np.mean((beta[0]+beta[1]*state-(1+2*state))**2)
    reference = (beta@[1.,-.8]-(-.6))**2
    np.testing.assert_allclose(np.sum((x@beta-y)**2),.7*point+.3*reference)


def test_nested_selection_never_uses_outer_held_out_targets():
    runs, template = case()
    baseline = a.fit_foundation(runs,{},'measured','My',template)
    first = a.tune_foundation(runs,{},'measured','My',template,baseline)
    altered = copy.deepcopy(runs)
    for r in altered[12:]:
        r['channels']['My']['reference'] += 1e6
        r['channels']['My']['truth'] += 1e6
    second = a.tune_foundation(altered,{},'measured','My',template,baseline)
    assert first==second
    audit = first['foundation_polish']
    assert set(audit['selection_training_runs'])==set(template['training_runs'])
    index = {r['run']:r for r in runs}
    for fold in audit['inner_folds']:
        train = {index[k]['condition_id'] for k in fold['training_runs']}
        valid = {index[k]['condition_id'] for k in fold['validation_runs']}
        assert not train&valid
        assert 'c6' not in train|valid
    # The predicted initial load uses features, never the measured label.
    pred = a.reference_prediction(runs[0]['channels']['My']['measured'],first)
    runs[0]['channels']['My']['reference'] += 1e6
    assert a.reference_prediction(runs[0]['channels']['My']['measured'],first)==pred


def test_reference_only_prediction_stays_available_without_reference_label():
    runs,template = case()
    fitted = a.fit_foundation(runs,{},'measured','My',dict(template,reference_fraction=1.,ridge=0.))
    np.testing.assert_allclose(fitted['coefficients'],[1,2],atol=1e-12)
    data = dict(reference_physical=0.,reference_features={'test':{'x':3.,'intercept':1.}})
    np.testing.assert_allclose(a.reference_prediction(data,fitted),7.,atol=1e-12)


def test_too_few_training_conditions_preserves_baseline():
    runs,template = case()
    template = dict(template,training_runs=['r0','r1','r2','r3'])
    baseline = a.fit_foundation(runs,{},'measured','My',template)
    polished = a.tune_foundation(runs,{},'measured','My',template,baseline)
    assert polished['coefficients']==baseline['coefficients']
    assert polished['foundation_polish']==dict(selected=False,reason='fewer_than_six_training_conditions')
