"""Per-run RAM spool; persist closed collectors before reporting success."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


class RamCapture:
    def __init__(self, destination: Path, seconds: float, root: Path = Path('/dev/shm')):
        self.destination = destination
        self.root = root.resolve()
        # Raw USB plus record headers/clock JSON; conservative 160 kB/s.
        self.required = int((seconds + 40) * 160_000) + 32 * 1024**2
        if shutil.disk_usage(root).free < self.required:
            raise RuntimeError('Insufficient RAM spool capacity; capture not started')
        self.check_memory(self.required)
        if shutil.disk_usage(destination).free < self.required:
            raise RuntimeError('Insufficient SD capacity; capture not started')
        self.runtime = Path(tempfile.mkdtemp(prefix='fftgyro-leptrino-', dir=root))
        self.record = dict(state='RAM_RECORDING', runtime=str(self.runtime),
                           required_bytes=self.required, power_loss_risk=True)
        self.save_record()

    @staticmethod
    def check_memory(required: int):
        info = Path('/proc/meminfo')
        if info.exists():
            values = dict(line.split(':', 1) for line in info.read_text().splitlines())
            if int(values['MemAvailable'].split()[0]) * 1024 < required:
                raise RuntimeError('Insufficient available memory; capture not started')

    def check(self):
        if shutil.disk_usage(self.root).free < 16 * 1024**2:
            raise RuntimeError('RAM spool low; stopping capture')
        self.check_memory(32 * 1024**2)

    def save_record(self):
        with (self.destination / 'ram-storage.json').open('w') as stream:
            json.dump(self.record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def digest(path: Path):
        result = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                result.update(chunk)
        return result.hexdigest()

    def persist(self):
        """Call only after both children have exited, including on failure."""
        try:
            for name in ('clock', 'leptrino'):
                source = self.runtime / name
                if not source.exists():
                    continue
                target = self.destination / name
                staging = self.destination / (name + '.partial')
                if target.exists() or staging.exists():
                    raise FileExistsError('Refusing to overwrite existing capture')
                shutil.copytree(source, staging)
                for original in source.rglob('*'):
                    if original.is_file():
                        copied = staging / original.relative_to(source)
                        with copied.open('r+b') as stream:
                            os.fsync(stream.fileno())
                        if self.digest(original) != self.digest(copied):
                            raise OSError('RAM/SD checksum mismatch')
                staging.rename(target)
            # Flush this run's directory metadata. A global os.sync() also waits
            # for unrelated SD activity and made every post-run pause unbounded.
            if os.name == 'posix':
                directory_fd = os.open(self.destination, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            self.record['state'] = 'PERSISTED_VERIFIED'
            self.save_record()
            if self.runtime.resolve().parent != self.root:
                raise RuntimeError('Unexpected RAM cleanup path')
            shutil.rmtree(self.runtime)
        except Exception as error:
            self.record.update(state='PERSIST_FAILED_RAM_RETAINED', error=str(error))
            self.save_record()
            raise
