"""Append-only audit trail for every experiment run.

Summary CSVs hold only the LATEST value per (dataset, model) — they are
overwritten as configs change. This module keeps the history those files throw
away: one tab-separated line per model fit, with the wall time, the environment
it ran in, and the git commit it ran at, so any number in the paper can be
traced back to when and where it was produced.

Never rewritten, only appended. `journal/results/run_log.tsv` is the file to
grep when asking "what did this used to be, and how long did it take?".
"""
import getpass
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / 'results'
LOG = RESULTS / 'run_log.tsv'

FIELDS = (
    'ts_utc', 'script', 'protocol', 'dataset', 'model',
    'median', 'p95', 'max', 'mean', 'n', 'seconds',
    'git_commit', 'git_dirty', 'host', 'os_arch', 'python',
    'sklearn', 'numpy', 'scipy', 'notes',
)


def _versions() -> dict:
    out = {'python': platform.python_version()}
    for name in ('sklearn', 'numpy', 'scipy'):
        try:
            out[name] = __import__(name).__version__
        except Exception:                      # pragma: no cover - env probe
            out[name] = 'unknown'
    return out


def _git() -> tuple:
    """(commit, dirty) for the repo this file lives in; ('unknown','?') if git fails."""
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        status = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'],
                                capture_output=True, text=True, timeout=10).stdout
        return (commit or 'unknown', 'dirty' if status.strip() else 'clean')
    except Exception:                          # pragma: no cover - git probe
        return ('unknown', '?')


def _context() -> dict:
    commit, dirty = _git()
    ctx = {
        'git_commit': commit,
        'git_dirty': dirty,
        'host': socket.gethostname(),
        'os_arch': f'{platform.system()}-{platform.machine()}',
    }
    ctx.update(_versions())
    return ctx


def _clean(value) -> str:
    """TSV safety: no tabs or newlines inside a field."""
    return str(value).replace('\t', ' ').replace('\n', ' ')


def append(script: str, protocol: str, dataset: str, model: str,
           stats: dict, seconds: float, notes: str = '', path: Path = None) -> None:
    """Append one fit's outcome. Writes the header on first use.

    stats: the dict from evaluate.summarize (median/p95/max/mean/n).
    protocol: how it was evaluated, e.g. '5fold-kfold', '5fold-grouped',
              'ifra-within-run', 'ifra-cross-run', 'ifra-loo'.
    """
    path = Path(path) if path is not None else LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        'ts_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'script': script, 'protocol': protocol, 'dataset': dataset, 'model': model,
        'median': round(float(stats['median']), 4),
        'p95': round(float(stats['p95']), 4),
        'max': round(float(stats['max']), 4),
        'mean': round(float(stats['mean']), 4),
        'n': int(stats['n']),
        'seconds': round(float(seconds), 1),
        'notes': notes,
    }
    row.update(_context())
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, 'a', encoding='utf-8') as fh:
        if write_header:
            fh.write('\t'.join(FIELDS) + '\n')
        fh.write('\t'.join(_clean(row.get(f, '')) for f in FIELDS) + '\n')
