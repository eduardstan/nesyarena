"""ProbLog anytime k-best adapter (deployed approximate inference).

ProbLog's `kbest` evaluatable (Kimmig et al.) is an *anytime bounds*
algorithm: it repeatedly adds the current best explanation to a lower border
(and refutations to an upper border) until the gap closes below a convergence
threshold eps. The deployed semantic object is therefore an **interval**
[lb, ub] claimed to contain the distribution-semantics value; the lower
border after j updates is the probability of the union of the j best
explanations — i.e. exactly the arena's top-k-proofs semantics with k = j.

Registered predictions (checked by experiments/conformance_problog_kbest.py):
  P1 (soundness):     P*(q) in [lb, ub] on every frozen instance, every eps.
  P2 (tight eps):     at eps = 1e-9 both borders equal P*(q) up to ~1e-9.
  P3 (lower border):  at loose eps, lb equals the reference TopK(k) value for
                      some k (tie-free instances), since both compute the WMC
                      of a best-explanation prefix.

`infer()` returns the lower bound (ProbLog's own `lower_only` answer — the
sound approximation a deployed pipeline would use as "the" probability);
`infer_bounds()` exposes the full interval, which is the object conformance
is judged on. No gradient support.
"""

from __future__ import annotations

from ..ir import Atom, GroundProgram


def _term(x) -> str:
    s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(x))
    return s if s and (s[0].islower() or s[0].isdigit()) else "c_" + s


def render_atom(atom: Atom) -> str:
    if not atom.args:
        return _term(atom.pred)
    return f"{_term(atom.pred)}({', '.join(_term(a) for a in atom.args)})"


def render_program(program: GroundProgram, base: dict[Atom, float], query: Atom) -> str:
    lines = [f"{float(p)}::{render_atom(a)}." for a, p in sorted(base.items(), key=repr)]
    lines += [f"{render_atom(r.head)} :- {', '.join(render_atom(b) for b in r.body)}."
              for r in program.rules]
    lines.append(f"query({render_atom(query)}).")
    return "\n".join(lines)


class ProbLogKBestAdapter:
    supports_grad = False

    def __init__(self, convergence: float = 1e-9):
        from problog import get_evaluatable  # noqa: F401  deferred probe

        self.convergence = float(convergence)
        self.name = f"problog:kbest(eps={convergence})"
        self.claimed_semantics = "distribution semantics (anytime bounds)"

    def infer_bounds(self, program: GroundProgram, base: dict[Atom, float],
                     queries: list[Atom]) -> dict[Atom, tuple[float, float]]:
        from problog import get_evaluatable
        from problog.program import PrologString

        out: dict[Atom, tuple[float, float]] = {}
        for q in queries:  # one query per solve keeps the result mapping trivial
            src = render_program(program, base, q)
            res = get_evaluatable("kbest").create_from(PrologString(src)).evaluate(
                convergence=self.convergence)
            (val,) = res.values()
            lb, ub = (float(val[0]), float(val[1])) if isinstance(val, tuple) \
                else (float(val), float(val))
            out[q] = (lb, ub)
        return out

    def infer(self, program, base, queries) -> dict[Atom, float]:
        return {q: b[0] for q, b in self.infer_bounds(program, base, queries).items()}
