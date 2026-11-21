"""Evidence validation using synthetic profiles rather than actual run records."""
import hashlib
import json

import pytest

from fftgyro_quad_id.experiment_runner import supplement_qualification as qualification


@pytest.fixture
def evidence_files(tmp_path):
    profile = tmp_path / 'profile.json'
    profile.write_text('{"synthetic": true}')
    record = dict(schema='fftgyro-dynamic-qualification-v1', status='QUALIFIED',
                  axis='PITCH', profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                  runs=[dict(state='CAPTURED', run_json_sha256='0' * 64)])
    evidence = tmp_path / 'evidence.json'
    evidence.write_text(json.dumps(record))
    return profile, evidence, record


@pytest.mark.parametrize('axis', ['pitch', 'yaw'])
def test_exact_profile_and_matching_axis_are_required(evidence_files, axis):
    profile, evidence, record = evidence_files
    record['axis'] = axis.upper()
    evidence.write_text(json.dumps(record))
    assert qualification.dynamic_evidence(evidence, axis, profile)['status'] == 'QUALIFIED'


def test_changed_profile_cannot_be_promoted(evidence_files):
    profile, evidence, _ = evidence_files
    profile.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='profile hash differs'):
        qualification.dynamic_evidence(evidence, 'pitch', profile)


def test_unqualified_evidence_cannot_be_promoted(evidence_files):
    profile, evidence, record = evidence_files
    record['status'] = 'DRAFT'
    evidence.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='evidence is invalid'):
        qualification.dynamic_evidence(evidence, 'pitch', profile)


def test_collective_mapping_requires_complete_evidence(tmp_path):
    mapping = {str(r): .3 * r for r in (.8, .9, .95, 1, 1.05, 1.1, 1.2)}
    record = dict(schema='fftgyro-fixed-input-qualification-v1', status='QUALIFIED',
                  collective_by_rpm_ratio=mapping,
                  runs=[dict(state='CAPTURED', run_json_sha256='0' * 64)] * 5)
    path = tmp_path / 'mapping.json'
    path.write_text(json.dumps(record))
    assert qualification.measured_collective(path) == mapping
    record['runs'] = []
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='mapping'):
        qualification.measured_collective(path)
