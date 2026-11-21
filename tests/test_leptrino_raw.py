import struct
import argparse
import json
import signal
import sys
import time
import types
from pathlib import Path

import pytest

from fftgyro_quad_id.io.leptrino import frame, Decoder
from fftgyro_quad_id.io.leptrino_raw import HEADER, MAGIC, RawWriter, chunks, decode, capture


def stream(counters):
    data = frame(bytes([4,255,50,0]))
    for counter in counters:
        data += frame(bytes([20,255,50,0])+struct.pack("<6h",1,2,3,4,5,6)+bytes([0,0,64,counter]))
    return data+frame(bytes([4,255,51,0]))


def test_roundtrip_decode_retains_stop_tail(tmp_path):
    writer = RawWriter(tmp_path/"raw.bin")
    data = stream([254,255,0,1])
    writer.push(10, data[:13])
    writer.push(20, data[13:])
    writer.finish()
    assert b"".join(data for _, data in chunks(tmp_path/"raw.bin")) == data
    result = decode(tmp_path/"raw.bin", tmp_path/"samples.csv", (100,100,100,2,2,2), 0)
    assert result["samples"] == 4
    assert result["candidate_counter_jumps"] == []


def test_counter_gap_recorded(tmp_path):
    writer = RawWriter(tmp_path/"raw.bin")
    writer.push(10, stream([1,2,63]))
    writer.finish()
    result = decode(tmp_path/"raw.bin", tmp_path/"samples.csv", (100,100,100,2,2,2), 0)
    assert result["candidate_counter_jumps"] == [{"index":2,"delta_mod256":61}]


def test_qc_can_bound_reserved_byte_diagnostics_without_dropping_count(tmp_path):
    writer = RawWriter(tmp_path / "raw.bin")
    writer.push(10, stream([1, 9, 9, 9]))
    writer.finish()
    result = decode(tmp_path / "raw.bin", tmp_path / "samples.csv",
                    (100, 100, 100, 2, 2, 2), 0, counter_jump_limit=0)
    assert result["candidate_counter_jumps"] == []
    assert result["candidate_counter_jump_count"] == 3


def test_documented_status_fault_counted(tmp_path):
    writer = RawWriter(tmp_path/"raw.bin")
    data = frame(bytes([4,255,50,0]))
    data += frame(bytes([20,255,50,0])+struct.pack("<6h",1,2,3,4,5,6)+bytes([0,0,0x44,1]))
    data += frame(bytes([4,255,51,0]))
    writer.push(10, data)
    writer.finish()
    result = decode(tmp_path/"raw.bin", tmp_path/"samples.csv", (100,100,100,2,2,2), 0)
    assert result["documented_status_fault_samples"] == 1
    assert result["documented_status_fault_counts"]["rated_overrange"] == 1


@pytest.mark.parametrize("data", [b"", MAGIC+b"x", MAGIC+HEADER.pack(1,4)+b"ab",
                                 MAGIC+HEADER.pack(1,4097), MAGIC+HEADER.pack(1,0)])
def test_truncated_or_invalid_raw_rejected(tmp_path, data):
    path = tmp_path/"bad.bin"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        list(chunks(path))


def test_missing_stop_rejected(tmp_path):
    writer = RawWriter(tmp_path/"raw.bin")
    writer.push(10, stream([1])[:-9])
    writer.finish()
    with pytest.raises(ValueError):
        decode(tmp_path/"raw.bin", tmp_path/"samples.csv", (100,100,100,2,2,2), 0)


def test_writer_error_is_not_success(tmp_path):
    writer = RawWriter(tmp_path/"raw.bin")
    writer.finish()
    writer.error = "simulated disk failure"
    with pytest.raises(OSError):
        writer.push(10,b"data")
    with pytest.raises(OSError):
        writer.finish()


@pytest.mark.parametrize("operator_stop", [False,True])
def test_capture_raw_only_and_graceful_signal(tmp_path,monkeypatch,operator_stop):
    commands=[]
    class Port:
        def __init__(self,**kwargs):
            self.is_open=False
            self.data=bytearray()
            self.streaming=False
            self.signalled=False
        def open(self): self.is_open=True
        def close(self): self.is_open=False
        @property
        def in_waiting(self): return len(self.data)
        def write(self,data):
            command=Decoder().feed(data)[0][2]
            commands.append(command)
            if command==0x2a:
                reply=bytes([38,255,command,0])+b'ASM0551012R0A00'.ljust(16,b'\0')+b'2506002\0'+b'4009'+b'1200\0\0'
            elif command==0x2b:
                reply=bytes([28,255,command,0])+struct.pack('<6f',100,100,100,2,2,2)
            else:
                reply=bytes([4,255,command,0])
            self.data.extend(frame(reply))
            if command==0x32:
                self.streaming=True
                self.data.extend(frame(bytes([20,255,50,0])+struct.pack('<6h',1,2,3,4,5,6)+bytes([0,0,64,0])))
            return len(data)
        def read(self,size):
            if operator_stop and self.streaming and not self.signalled:
                self.signalled=True
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
            if not self.data: time.sleep(.001)
            data=bytes(self.data[:size]); del self.data[:size]
            return data
    monkeypatch.setitem(sys.modules,'serial',types.SimpleNamespace(Serial=Port))
    original=Path.read_text
    def read_text(path,*args,**kwargs):
        if str(path).replace('\\','/')=='/proc/sys/kernel/random/boot_id': return 'test-boot'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_text)
    out=tmp_path/'run'
    args=argparse.Namespace(output=str(out),port='fake',seconds=.01,
                            expected_serial='2506002',expected_product='ASM0551012R0A00',decode_on_pi=False)
    assert capture(args)==0
    summary=json.loads((out/'summary.json').read_text())
    assert summary['state']=='RAW_COMPLETE'
    assert summary['raw_storage_complete'] and summary['stop_acknowledged']
    assert summary['stopped_early']==operator_stop
    assert commands==[0x33,0x2a,0x2b,0x32,0x33]
    result=decode(out/'chunks.bin',out/'decoded.csv',(100,100,100,2,2,2),summary['start_command_before_ns'])
    assert result['samples']==1
