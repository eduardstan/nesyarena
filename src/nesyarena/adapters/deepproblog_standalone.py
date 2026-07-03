"""DeepProbLog (standalone) adapter — the original system, exact engine.

DeepProbLog (Manhaeve et al., NeurIPS 2018; `pip install deepproblog`, here
2.0.6) extends ProbLog with neural predicates. On programs whose facts carry
constant probabilities — the arena's frozen batteries — its ExactEngine
grounds via ProbLog and compiles to SDDs, so the claimed semantics is
distribution semantics computed exactly.

Registered prediction: value conformance to the oracle at compilation
precision on every frozen battery, including the recursive ones (chains,
cyclic, probes), since the ProbLog engine handles cycles by tabling.

Scope: the ApproximateEngine (DPLA*, k-best-style search) requires
SWI-Prolog/PySwip and is not measured here; it is the natural follow-up once
a SWI-Prolog environment is available. Gradients: constant-probability
programs expose no differentiable path in this system (learning flows through
neural predicates only), so supports_grad = False.
"""

from __future__ import annotations

import os
import tempfile

from ..ir import Atom, GroundProgram
from .problog_kbest import render_program


class DeepProbLogStandaloneAdapter:
    supports_grad = False

    def __init__(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # PySwip-absent notice is expected
            from deepproblog.engines import ExactEngine  # noqa: F401
        self.name = "deepproblog:exact-engine"
        self.claimed_semantics = "distribution semantics"

    @staticmethod
    def _term(atom: Atom):
        from problog.logic import Term

        if not atom.args:
            return Term(atom.pred)
        return Term(atom.pred, *[Term(str(a)) for a in atom.args])

    def infer(self, program: GroundProgram, base: dict[Atom, float],
              queries: list[Atom]) -> dict[Atom, float]:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from deepproblog.engines import ExactEngine
            from deepproblog.model import Model
            from deepproblog.query import Query

        src = render_program(program, base)  # no query directive: queries go via the API
        out: dict[Atom, float] = {}
        for q in queries:  # one query per solve keeps the result mapping trivial
            with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as fh:
                fh.write(src)
                path = fh.name
            try:
                model = Model(path, [])
                model.set_engine(ExactEngine(model))
                res = model.solve([Query(self._term(q))])
                vals = list(res[0].result.values())
                out[q] = float(vals[0]) if vals else 0.0
            finally:
                os.unlink(path)
        return out
