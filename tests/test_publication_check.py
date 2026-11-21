"""Guard against accidental disclosure without echoing credential values."""
from pathlib import Path

from scripts import check_publication as check


def test_generated_data_is_rejected_even_when_utf8(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'measurements.csv'
    path.write_text('time,value\n0,1\n')
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(check, 'source_paths', lambda: [path])
    assert check.main() == 1
    assert 'must remain outside Git' in capsys.readouterr().out


def test_secret_is_detected_without_printing_the_value(tmp_path, monkeypatch, capsys):
    value = 'ghp_' + 'A' * 30
    path = tmp_path / 'config.py'
    path.write_text(f'credential = {value!r}\n')
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(check, 'source_paths', lambda: [path])
    assert check.main() == 1
    output = capsys.readouterr().out
    assert 'GitHub credential' in output
    assert value not in output


def test_example_home_is_allowed_and_personal_home_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'config.py'
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(check, 'source_paths', lambda: [path])
    path.write_text('/' + 'Users/researcher/project/')
    assert check.main() == 0
    path.write_text('/' + 'Users/alice/project/')
    assert check.main() == 1


def test_internal_handoff_document_is_rejected(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'HANDOFF.md'
    path.write_text('Internal operations notes')
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(check, 'source_paths', lambda: [path])
    assert check.main() == 1
    assert 'must remain private' in capsys.readouterr().out
