"""Safe parsing/evaluation of an LLM-emitted ink->XYZ equation (Plan 09, Task 1).

The model is asked (see journal/llm/run_equation.py) to return exactly one fenced
block of three lines::

    X = <expression in c, m, y>
    Y = <expression in c, m, y>
    Z = <expression in c, m, y>

SAFETY: model text is NEVER passed to `eval`/`exec`. Two gates, in order:

1. `validate_expression_text` tokenizes the right-hand side with Python's
   `tokenize` module and rejects the string unless every token is either a
   number, one of a small operator set, or a NAME in a whitelist (the ink
   variables plus a handful of scalar maths functions). Attribute access,
   calls to anything unlisted, statements, strings, commas and dunders are all
   rejected before sympy ever sees the text.
2. Only then is the (already whitelisted) string handed to
   `sympy.parse_expr` with an explicit `local_dict` and no auto-eval of
   unknown callables, and lambdified over numpy for vectorised evaluation.

A refusal, a missing block, or an out-of-whitelist expression raises
`EquationParseError` -- callers record that as the model's outcome rather than
hiding it.
"""
from __future__ import annotations

import io
import re
import token as token_mod
import tokenize
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

# Scalar maths functions a model may plausibly reach for. Allowed so that a
# non-polynomial answer is still *scored* (flagged `nonpolynomial`) instead of
# being thrown away; nothing here can touch the filesystem or interpreter.
ALLOWED_FUNCS = {
    'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
    'abs': sp.Abs, 'Abs': sp.Abs, 'Min': sp.Min, 'Max': sp.Max,
    'min': sp.Min, 'max': sp.Max,
}

_ALLOWED_OPS = {'+', '-', '*', '/', '**', '(', ')', '^'}

OUTPUT_NAMES = ('X', 'Y', 'Z')


class EquationParseError(Exception):
    """The model's answer could not be turned into a safe, evaluable equation."""


def validate_expression_text(text: str, input_names: tuple) -> str:
    """Gate 1: tokenizer whitelist. Returns the text unchanged or raises."""
    allowed_names = set(input_names) | {n.upper() for n in input_names} | set(ALLOWED_FUNCS)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text.strip()).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        raise EquationParseError(f"not tokenizable: {e}") from e

    for t in toks:
        if t.type in (token_mod.ENDMARKER, token_mod.NEWLINE, token_mod.NL,
                      token_mod.INDENT, token_mod.DEDENT, token_mod.COMMENT):
            continue
        if t.type == token_mod.NUMBER:
            continue
        if t.type == token_mod.NAME:
            if t.string not in allowed_names:
                raise EquationParseError(f"disallowed name {t.string!r}")
            continue
        if t.type == token_mod.OP:
            if t.string not in _ALLOWED_OPS:
                raise EquationParseError(f"disallowed operator {t.string!r}")
            continue
        raise EquationParseError(f"disallowed token {tokenize.tok_name[t.type]} {t.string!r}")
    return text


@dataclass
class EquationSet:
    """Three sympy expressions in the ink variables, plus provenance flags."""
    exprs: tuple                  # (X_expr, Y_expr, Z_expr)
    symbols: tuple                # sympy symbols in dataset column order
    notes: list = field(default_factory=list)
    nonpolynomial: bool = False

    def __post_init__(self):
        self._f = sp.lambdify(self.symbols, list(self.exprs), modules='numpy')

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """X: (n, n_inks) ink percentages -> (n, 3) predicted XYZ."""
        X = np.asarray(X, dtype=float)
        cols = [X[:, i] for i in range(X.shape[1])]
        out = self._f(*cols)
        # lambdify returns python scalars for constant expressions
        out = [np.broadcast_to(np.asarray(o, dtype=float), (X.shape[0],)) for o in out]
        return np.column_stack(out)

    @property
    def max_total_degree(self) -> int:
        return max_exponent(self.exprs)

    @property
    def max_var_exponent(self) -> int:
        """Largest exponent on any single variable (Phil's literal wording)."""
        best = 0
        for e in self.exprs:
            for p in sp.preorder_traversal(sp.expand(e)):
                if p.is_Pow and p.exp.is_number:
                    try:
                        best = max(best, int(abs(p.exp)))
                    except TypeError:
                        best = max(best, 99)
        return best

    @property
    def violates_exponent_cap(self) -> bool:
        if self.nonpolynomial:
            return True
        return self.max_total_degree > 3 or self.max_var_exponent > 3

    @property
    def n_terms(self) -> int:
        """Additive terms across the three expanded expressions."""
        n = 0
        for e in self.exprs:
            ex = sp.expand(e)
            n += len(ex.args) if ex.is_Add else 1
        return n


def max_exponent(exprs) -> int:
    """Max total polynomial degree over the given expressions (0 if constant).

    Returns -1 for a non-polynomial expression (sympy cannot give a degree).
    """
    best = 0
    for e in exprs:
        syms = sorted(e.free_symbols, key=str)
        if not syms:
            continue
        try:
            best = max(best, int(sp.Poly(sp.expand(e), *syms).total_degree()))
        except (sp.PolynomialError, TypeError, ValueError):
            return -1
    return best


def _extract_lines(text: str) -> dict:
    """Pull `X = ...`, `Y = ...`, `Z = ...` out of the model's answer.

    Prefers a fenced code block (any fence tag); falls back to scanning the
    whole answer. The LAST assignment to each output wins, so a model that
    restates its final equation after working is handled.
    """
    blocks = re.findall(r'```[^\n]*\n(.*?)```', text, re.S)
    haystacks = blocks + [text] if blocks else [text]
    for hay in haystacks:
        found = {}
        for line in hay.splitlines():
            line = line.strip().rstrip(';,')
            line = re.sub(r'^[-*\d.)\s]*', '', line)           # bullet/number prefixes
            m = re.match(r'^([XYZ])\s*(?:\(\s*c\s*,\s*m\s*,\s*y\s*\))?\s*=\s*(.+)$', line)
            if m:
                found[m.group(1)] = m.group(2).strip()
        if all(k in found for k in OUTPUT_NAMES):
            return found
    raise EquationParseError("no X=/Y=/Z= assignment triple found in the answer")


def parse_equation(text: str, input_names: tuple) -> EquationSet:
    """Parse an LLM answer into an EquationSet. Raises EquationParseError."""
    raw = _extract_lines(text)
    notes = []
    symbols = sp.symbols(' '.join(input_names))
    if not isinstance(symbols, tuple):
        symbols = (symbols,)
    local = dict(zip(input_names, symbols))
    local.update(ALLOWED_FUNCS)

    exprs = []
    nonpoly = False
    for name in OUTPUT_NAMES:
        s = raw[name]
        s = s.replace('·', '*').replace('×', '*').replace('−', '-')
        s = re.sub(r'(?<=[\d)])\s*\^\s*', '**', s)              # a^2 -> a**2
        s = s.replace('^', '**')
        validate_expression_text(s, tuple(input_names))
        # Uppercase ink names (C, M, Y) are unambiguous on a right-hand side --
        # the outputs only ever appear on the left -- so recover them.
        upper = {n.upper() for n in input_names}
        if any(re.search(rf'\b{u}\b', s) for u in upper):
            for n in input_names:
                s = re.sub(rf'\b{n.upper()}\b', n, s)
            notes.append('uppercase ink names on RHS lowercased by parser')
        try:
            expr = sp.parse_expr(s, local_dict=local, evaluate=True)
        except (sp.SympifyError, SyntaxError, TypeError) as e:
            raise EquationParseError(f"sympy could not parse {name}: {e}") from e
        stray = expr.free_symbols - set(symbols)
        if stray:
            raise EquationParseError(f"{name} references unknown symbols {stray}")
        if any(f.func in (sp.exp, sp.log, sp.Abs, sp.Min, sp.Max) or
               (f.is_Pow and not f.exp.is_Integer)
               for f in sp.preorder_traversal(expr)):
            nonpoly = True
        exprs.append(expr)

    if nonpoly:
        notes.append('non-polynomial term present (sqrt/exp/log/abs/min/max or '
                     'non-integer power) -- polynomial degree not defined')
    seen = set()
    notes = [n for n in notes if not (n in seen or seen.add(n))]
    return EquationSet(exprs=tuple(exprs), symbols=symbols, notes=notes,
                       nonpolynomial=nonpoly)
