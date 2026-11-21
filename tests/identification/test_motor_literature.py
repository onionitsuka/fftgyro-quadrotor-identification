import numpy as np
from fftgyro_quad_id.identification.motor_literature import fit_static, steady_rpm, predict


def test_published_maps_recover_synthetic_static_curves():
    rng = np.random.default_rng(42)
    u=rng.uniform(.12,.6,500);v=rng.uniform(13.5,16.5,500)
    for model in [dict(kind='kaufmann2023',coefficients=[100,-500,6000,2000,1500]),
                  dict(kind='wildmann2022',coefficients=[700]),
                  dict(kind='shastry2023',coefficients=[3,4,.2])]:
        y=steady_rpm(u,v,model)
        fitted=fit_static(u,v,y,np.ones(len(u)),model['kind'])
        np.testing.assert_allclose(steady_rpm(u,v,fitted),y,rtol=2e-7,atol=1e-4)


def test_replay_is_causal_and_restarts_without_measured_rpm():
    models=[dict(kind='wildmann2022',coefficients=[1000])]*4
    u=np.full((1000,4),.2);u[500:]=.4;v=np.full(1000,15.)
    a,w=predict(u,v,models,tau=.05,delay=.004)
    changed=u.copy();changed[600:]=.8
    b,_=predict(changed,v,models,tau=.05,delay=.004)
    np.testing.assert_allclose(a[:600],b[:600])
    np.testing.assert_allclose(a[:500],3000)
    assert not w[:200].any() and w[200:].all()
    u[400:410]=np.nan
    c,w=predict(u,v,models,tau=.05)
    assert np.isnan(c[400:410]).all() and not w[410:610].any()
    np.testing.assert_allclose(c[410],3000)
