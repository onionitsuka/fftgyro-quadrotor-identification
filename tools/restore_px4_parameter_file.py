"""Restore a QGroundControl parameter file through the verified PX4 setter."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from set_px4_parameters import main as set_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('device', help='COM port, for example COM7')
    parser.add_argument('parameters', type=Path)
    args = parser.parse_args()
    settings = {}
    for line in args.parameters.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        _, _, name, value, parameter_type = line.split()
        if name == '_HASH_CHECK':
            continue
        if name in settings:
            raise ValueError('Duplicate parameter: ' + name)
        settings[name] = float(value) if int(parameter_type) == 9 else int(value)
    with tempfile.TemporaryDirectory(prefix='px4-restore-') as folder:
        path = Path(folder) / 'settings.json'
        path.write_text(json.dumps(settings), encoding='utf-8')
        sys.argv = [sys.argv[0], args.device, str(path)]
        return set_parameters()


if __name__ == '__main__':
    raise SystemExit(main())
