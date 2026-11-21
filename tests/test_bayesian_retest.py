import pytest

from fftgyro_quad_id.identification.bayesian_retest import CONDITIONS, propose


def history():
    return {"evaluation_role": "D", "evaluation_id": "frozen-D5",
            "baseline_rmse": [0.02, 0.01], "observations": []}


def populated():
    h = history()
    before = h["baseline_rmse"]
    for index, (axis, wind) in enumerate(CONDITIONS):
        after = [x * 0.97 for x in before]
        h["observations"].append({"axis": axis, "wind_m_s": wind, "role": "T",
            "qc_pass": True, "evaluation_id": h["evaluation_id"], "powered_runs": 4,
            "training_runs_before": 4*index, "rmse_before": before,
            "rmse_after": after, "gain_se": [0.0002, 0.0001]})
        before = after
    return h


def test_empty_does_not_claim_bo_fitted():
    p = propose(history())
    assert p["decision"] == "BOOTSTRAP_COVERAGE"
    assert not p["candidate_ranking"]
    assert p["counts"]["base_full_scope"] == 95
    assert p["next_block"]["powered_runs"] == 4
    assert not p["runtime_qualified"]


def test_fitted_ranking_and_budget():
    h = populated()
    p = propose(h)
    assert p["decision"] == "BO_SELECT_ADDITIONAL_T"
    assert len(p["candidate_ranking"]) == 6
    assert p["counts"]["full_scope_with_mandatory_coverage"] == 95
    assert propose(h, max_extra_blocks=0)["decision"] == "BUDGET_LIMIT_NOT_PRECISION_SUCCESS"


@pytest.mark.parametrize("field,value", [("evaluation_role", "H"), ("baseline_rmse", [0, 1])])
def test_reject_invalid_history(field, value):
    h = history()
    h[field] = value
    with pytest.raises(ValueError):
        propose(h)


def test_reject_partial_or_failed_qc():
    h = populated()
    h["observations"][0]["qc_pass"] = False
    with pytest.raises(ValueError):
        propose(h)


def test_reject_changed_evaluation():
    h = populated()
    h["observations"][-1]["evaluation_id"] = "new-D"
    with pytest.raises(ValueError):
        propose(h)


def test_unmeasured_baseline_is_allowed_only_for_bootstrap():
    h = history()
    h["baseline_rmse"] = None
    assert propose(h)["decision"] == "BOOTSTRAP_COVERAGE"
