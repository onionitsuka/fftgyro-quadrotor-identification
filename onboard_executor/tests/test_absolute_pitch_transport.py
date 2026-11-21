"""Absolute encoder anchor survives native quaternion recapture; PTY only."""
import json
import sys
from pathlib import Path
from test_session_transport import check

if __name__ == '__main__':
    exe = Path(sys.argv[1]).resolve()
    template = json.loads(Path(sys.argv[2]).read_text())
    for mode in ('absolute_anchor_negative', 'absolute_anchor_positive'):
        check(exe, template, mode, preserve_segments=True)
