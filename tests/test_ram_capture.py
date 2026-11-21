import json
from types import SimpleNamespace

import pytest

from fftgyro_quad_id.io import ram_capture


def make_spool(tmp_path):
    destination = tmp_path / 'sd'
    root = tmp_path / 'ram'
    destination.mkdir()
    root.mkdir()
    spool = ram_capture.RamCapture(destination, 30, root)
    for name in ('clock', 'leptrino'):
        (spool.runtime / name).mkdir()
        (spool.runtime / name / 'sample.bin').write_bytes(b'preserve raw bytes')
    return spool


def test_persist_verified_and_cleanup(tmp_path):
    spool = make_spool(tmp_path)
    spool.persist()
    assert not spool.runtime.exists()
    assert (spool.destination / 'leptrino/sample.bin').read_bytes() == b'preserve raw bytes'
    assert json.loads((spool.destination / 'ram-storage.json').read_text())['state'] == 'PERSISTED_VERIFIED'


@pytest.mark.parametrize('failure', ['copy', 'checksum', 'existing'])
def test_failure_preserves_ram(tmp_path, monkeypatch, failure):
    spool = make_spool(tmp_path)
    if failure == 'copy':
        def broken(*args, **kwargs):
            raise OSError('SD full')
        monkeypatch.setattr(ram_capture.shutil, 'copytree', broken)
    elif failure == 'checksum':
        monkeypatch.setattr(spool, 'digest', lambda path: str(path))
    else:
        (spool.destination / 'clock').mkdir()
    with pytest.raises(OSError):
        spool.persist()
    assert (spool.runtime / 'leptrino/sample.bin').read_bytes() == b'preserve raw bytes'
    assert json.loads((spool.destination / 'ram-storage.json').read_text())['state'] == 'PERSIST_FAILED_RAM_RETAINED'


def test_capacity_preflight_and_runtime_guard(tmp_path, monkeypatch):
    spool = make_spool(tmp_path)
    monkeypatch.setattr(ram_capture.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match='low'):
        spool.check()
    with pytest.raises(RuntimeError, match='capacity'):
        ram_capture.RamCapture(spool.destination, 30, spool.root)
