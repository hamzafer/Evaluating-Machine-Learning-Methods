from journal.llm.protocol import build_split, build_prompt, parse_xyz


def test_split_disjoint_and_sized():
    tr, te, spec = build_split('PC10-CMY', n_train=200, n_test=50)
    assert len(tr) == 200 and len(te) == 50
    overlap = tr.merge(te, on=list(spec.input_cols))   # same recipe both sides = leakage
    assert len(overlap) == 0


def test_prompt_contains_examples_and_query():
    tr, te, spec = build_split('PC10-CMY', 5, 1)
    p = build_prompt(tr, te.iloc[0], spec.input_cols)
    assert p.count('->') >= 5 and 'JSON' in p


def test_parse_xyz_tolerates_prose():
    assert parse_xyz('Sure! {"X": 32.71, "Y": 16.81, "Z": 11.32}') == (32.71, 16.81, 11.32)
    assert parse_xyz('no numbers here') is None
