import numpy as np
import pytest

from journal.llm.equation import (
    EquationParseError,
    max_exponent,
    parse_equation,
    validate_expression_text,
)

BLOCK = """
Here is my fit.

```equations
X = 2 + 3*c + 0.5*c**2*m
Y = 1 - 0.25*m + 0.001*y**3
Z = 10
```

Hope that helps.
"""


def test_parse_known_cubic_reproduces_values():
    eq = parse_equation(BLOCK, ('c', 'm', 'y'))
    X = np.array([[10.0, 20.0, 40.0], [0.0, 0.0, 0.0]])
    got = eq(X)
    c, m, y = X[:, 0], X[:, 1], X[:, 2]
    want = np.column_stack([
        2 + 3 * c + 0.5 * c ** 2 * m,
        1 - 0.25 * m + 0.001 * y ** 3,
        np.full_like(c, 10.0),
    ])
    assert np.allclose(got, want)


def test_max_exponent_is_total_degree():
    eq = parse_equation(BLOCK, ('c', 'm', 'y'))
    assert max_exponent(eq.exprs) == 3
    assert eq.max_var_exponent == 3          # y**3
    assert eq.n_terms == 3 + 3 + 1


def test_caret_and_uppercase_ink_names_are_recovered():
    text = "X = 1 + C^2\nY = M\nZ = Y"       # uppercase inks, ^ for power
    eq = parse_equation(text, ('c', 'm', 'y'))
    assert 'uppercase' in ' '.join(eq.notes).lower()
    got = eq(np.array([[3.0, 5.0, 7.0]]))
    assert np.allclose(got, [[10.0, 5.0, 7.0]])


def test_exponent_four_is_flagged_not_silently_accepted():
    text = "X = c**4\nY = 0\nZ = 0"
    eq = parse_equation(text, ('c', 'm', 'y'))
    assert max_exponent(eq.exprs) == 4
    assert eq.violates_exponent_cap is True


def test_unknown_symbols_and_calls_are_rejected():
    for bad in ["__import__('os').system('ls')",
                "open('/etc/passwd').read()",
                "1 + q",
                "gamma(c)",
                "c; import os"]:
        with pytest.raises(EquationParseError):
            validate_expression_text(bad, ('c', 'm', 'y'))


def test_whitelisted_functions_allowed_but_flagged_nonpolynomial():
    eq = parse_equation("X = sqrt(c)\nY = 0\nZ = 0", ('c', 'm', 'y'))
    assert eq.nonpolynomial is True
    assert np.allclose(eq(np.array([[9.0, 0.0, 0.0]])), [[3.0, 0.0, 0.0]])


def test_missing_block_raises():
    with pytest.raises(EquationParseError):
        parse_equation("I cannot help with that request.", ('c', 'm', 'y'))
