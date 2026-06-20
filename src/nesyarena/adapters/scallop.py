"""Scallop adapter over the IR.

Compiles a GroundProgram to a scallopy context: every EDB atom becomes a
tuple of the single relation `fact(String)` (keyed by the atom's repr, which
sidesteps scallopy's lack of 0-ary EDB relations), IDB atoms keep their
predicate and constant arguments, and ground rules are emitted verbatim —
including recursive ones, which Scallop evaluates natively to fixpoint.

The compile helpers are pure (testable without scallopy); only ScallopAdapter
itself needs the scallopy runtime (cp310 wheel — see INSTALL_SCALLOP.md).

Gradients: Scallop's value function is differentiated by central finite
differences through infer(). This measures the gradient of *what Scallop
computes*; comparing it to the reference SUTs' system-faithful gradients is
the gradient-conformance gate. (The diff* provenances via torch tags are a
later, optional cross-check.)
"""

from __future__ import annotations

from ..ir import Atom, GroundProgram


def fact_key(atom: Atom) -> str:
    return repr(atom)


def render_atom(program: GroundProgram, atom: Atom) -> str:
    if program.is_edb(atom):
        return f'fact("{fact_key(atom)}")'
    if not atom.args:
        return f"{atom.pred}()"
    return f"{atom.pred}(" + ", ".join(f'"{a}"' for a in atom.args) + ")"


def compile_rules(program: GroundProgram) -> list[str]:
    return [f"{render_atom(program, r.head)} = "
            + " and ".join(render_atom(program, b) for b in r.body)
            for r in program.rules]


class ScallopAdapter:
    def __init__(self, provenance: str, k: int | None = None):
        import scallopy  # deferred: the rest of the package never needs it

        self.scallopy = scallopy
        self.provenance = provenance
        self.k = k
        self.name = f"scallop:{provenance}" + (f"(k={k})" if k else "")
        self.claimed_semantics = "distribution semantics"
        self.supports_grad = True  # via finite differences of infer()

    def _run(self, program: GroundProgram, base: dict[Atom, float]):
        kw = dict(provenance=self.provenance)
        if self.k is not None:
            kw["k"] = self.k
        ctx = self.scallopy.ScallopContext(**kw)
        ctx.add_relation("fact", (str,))
        ctx.add_facts("fact", [(float(p), (fact_key(a),)) for a, p in base.items()])
        for rule in compile_rules(program):
            ctx.add_rule(rule)
        ctx.run()
        return ctx

    def infer(self, program, base, queries):
        ctx = self._run(program, base)
        out: dict[Atom, float] = {}
        by_pred: dict[str, list] = {}
        for q in queries:
            if q.pred not in by_pred:
                by_pred[q.pred] = list(ctx.relation(q.pred))
            args = tuple(str(a) for a in q.args)
            val = 0.0
            for row in by_pred[q.pred]:
                tag, tup = row if (len(row) == 2 and isinstance(row[0], float)) else (None, row)
                if tuple(tup) == args:
                    if tag is None:
                        raise RuntimeError(
                            f"untagged output {row!r} under provenance {self.provenance}; "
                            "inspect and extend the adapter, do not guess")
                    val = float(tag)
                    break
            out[q] = val
        return out

    def grad(self, program, base, query, wrt, h: float = 1e-5):
        g = {}
        for a in wrt:
            up, dn = dict(base), dict(base)
            up[a] = min(1.0, base[a] + h)
            dn[a] = max(0.0, base[a] - h)
            vu = self.infer(program, up, [query])[query]
            vd = self.infer(program, dn, [query])[query]
            g[a] = (vu - vd) / (up[a] - dn[a])
        return g
