"""Total inflow, baseline subtraction, and online/offline evaluation agree."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a
from scripts.current_graybox_aircraft import instantaneous_data, selected_features


def test_squared_total_inflow_is_not_squared_change_and_reference_scales_first():
    cfg = a.Settings()
    lam0 = np.sqrt(cfg.kf / (cfg.rho * np.pi * cfg.rotor_radius_m**4) / 2)
    lam = np.array([lam0, lam0 + .02, lam0 + .05])
    scale = np.array([12., 17., 23.])
    st = dict(inflow=lam, mu_x=np.zeros(3), mu_y=np.zeros(3),
              mu_z=np.zeros(3), mu_h2=np.zeros(3))
    b = a.evaluate_features(st, -np.ones(3), scale, 'Fz', 'compact', cfg)
    np.testing.assert_allclose(b['inflow^2'], scale * lam**2)
    assert not np.allclose(b['inflow^2'], scale * (lam-lam0)**2)
    contrast = b['inflow^2'] - b['inflow^2'][:2].mean()
    np.testing.assert_allclose(contrast, scale*lam**2 - np.mean(scale[:2]*lam[:2]**2))
    assert 'inflow_delta' in a.feature_dictionary('Fz', 'expanded')


def test_online_and_training_feature_evaluation_match():
    cfg = a.Settings()
    data = instantaneous_data([5., 1., -.5], [.1, -.2, .3],
                              [370., 420., 390., 440.], np.zeros(4), cfg, np.zeros(6))
    train = a.evaluate_features(data['state'], data['physical'][:, 2],
                                data['force_scale'], 'Fz', 'expanded', cfg)
    online = selected_features(data, cfg, 'Fz', dict(feature_names=['inflow^2']))
    np.testing.assert_allclose(train['inflow^2'], online['inflow^2'], atol=1e-12)
