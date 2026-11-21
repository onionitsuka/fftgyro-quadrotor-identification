#!/usr/bin/env python3
"""Check tracked public sources for accidental data, credentials and local identity.

Without a Git checkout, inspect source files while ignoring build/cache folders.
This heuristic complements human review; it does not certify confidentiality.
"""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.ulg', '.csv', '.npz', '.npy', '.mat', '.h5', '.hdf5', '.bag',
                      '.mcap', '.pdf', '.pptx', '.mp4', '.mov', '.zip', '.parquet',
                      '.pkl', '.pickle', '.pem'}
RULES = {
    'private key': r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
    'GitHub credential': r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}',
    'AWS access key': r'AKIA[0-9A-Z]{16}',
    'personal Mac home': r'/[U]sers/(?!researcher(?:/|$)|user(?:/|$)|test(?:/|$))[^/\s"\x27]+/',
    'personal Windows home': r'[A-Z]:[/\\]Users[/\\](?!researcher|user|test)[^/\\\s"\x27]+',
    'private IPv4 address': r'(?<![\d.])(?:192\.168\.[0-9]+\.[0-9]+|100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+)(?![\d.])',
}
INTERNAL_DOCUMENT = re.compile(
    r'handoff|引き継|引継|試験計画|実験計画|実施履歴|進捗|(?:^|/)[^/]*(?:plan|history|progress)[^/]*\.(?:md|json)$', re.IGNORECASE)
INTERNAL_FILENAMES = {'PUBLICATION.md', 'CLEANUP.json', 'PUBLIC_SOURCE_CHANGES.json',
                      'VALIDATION.json', 'production.json'}

IGNORED_PARTS = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache',
                 'build', 'dist', 'output', 'outputs', 'results'}


def source_paths():
    if (ROOT / '.git').exists():
        result = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'],
                                check=True, capture_output=True)
        return [ROOT / p for p in result.stdout.decode().split('\0') if p]
    return [p for p in ROOT.rglob('*') if p.is_file()
            and not any(x in IGNORED_PARTS or x.startswith('.venv') or x.endswith('.egg-info')
                        for x in p.relative_to(ROOT).parts)]


def main():
    failures = []
    paths = source_paths()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if path.name in INTERNAL_FILENAMES or (
                path.suffix.lower() in {'.md', '.json'} and INTERNAL_DOCUMENT.search(relative)):
            failures.append(f'{relative}: internal planning or handoff artifact must remain private')
        if path.is_symlink():
            failures.append(f'{relative}: symbolic link requires manual review')
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name == '.env':
            failures.append(f'{relative}: data or credential file must remain outside Git')
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeError:
            failures.append(f'{relative}: binary file requires explicit review')
            continue
        for label, pattern in RULES.items():
            for number, line in enumerate(content.splitlines(), 1):
                if re.search(pattern, line):
                    failures.append(f'{relative}:{number}: {label}')
    if failures:
        print('\n'.join(failures))
        return 1
    print(f'Publication source check passed ({len(paths)} files; heuristic patterns only).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
