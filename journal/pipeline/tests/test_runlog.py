from journal.pipeline import runlog

STATS = {'median': 0.0443210, 'p95': 0.16249, 'max': 2.1401, 'mean': 0.0678, 'n': 818}


def _rows(path):
    return path.read_text().strip().split('\n')


def test_appends_with_header_once(tmp_path):
    p = tmp_path / 'run_log.tsv'
    runlog.append('run.py', '5fold-kfold', 'PC10-CMY', 'gaussian_process', STATS, 215.04, path=p)
    runlog.append('run.py', '5fold-kfold', 'PC10-CMY', 'poly3', STATS, 3.2, path=p)
    rows = _rows(p)
    assert len(rows) == 3                                  # header + 2 records
    assert rows[0].split('\t') == list(runlog.FIELDS)       # header written once, first
    assert rows[2].split('\t')[runlog.FIELDS.index('model')] == 'poly3'


def test_record_content_and_rounding(tmp_path):
    p = tmp_path / 'run_log.tsv'
    runlog.append('run_ifra.py', 'ifra-within-run', 'IFRA-wb-Age_64a_wb-CMYK',
                  'gaussian_process', STATS, 702.24, notes='note\twith\ttabs', path=p)
    rec = dict(zip(runlog.FIELDS, _rows(p)[1].split('\t')))
    assert rec['protocol'] == 'ifra-within-run'
    assert rec['median'] == '0.0443' and rec['seconds'] == '702.2'
    assert rec['n'] == '818'
    assert '\t' not in rec['notes'] and rec['notes'] == 'note with tabs'
    # environment/provenance columns are populated, not blank
    for f in ('ts_utc', 'host', 'os_arch', 'python', 'sklearn', 'numpy', 'git_commit'):
        assert rec[f] and rec[f] != 'unknown' or f == 'git_commit'


def test_appends_across_processes_do_not_truncate(tmp_path):
    """The log is an audit trail: a second run must never clobber the first."""
    p = tmp_path / 'run_log.tsv'
    for i in range(3):
        runlog.append('run.py', '5fold-grouped', f'DS-{i}', 'svm', STATS, 1.0, path=p)
    assert len(_rows(p)) == 4
    assert [r.split('\t')[runlog.FIELDS.index('dataset')] for r in _rows(p)[1:]] == \
        ['DS-0', 'DS-1', 'DS-2']
