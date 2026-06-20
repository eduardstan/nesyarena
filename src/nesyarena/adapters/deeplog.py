"""DeepLog adapter (pydeeplog, ML-KULeuven/deeplog).

DeepLog is a substrate: it compiles formula specifications to algebraic
circuits and evaluates them exactly under a chosen structure. The adapter
expresses a query's proof DNF as a weighted-model-count specification —

    sum over fact assignments of
        [probability transform of  OR_j AND_{f in proof_j} (f = true)]
        times  PROD_f p(f)            (assignment-aware fact probabilities)

— mirroring the construction in DeepLog's own weighted-model-count test.
Claimed semantics: distribution semantics (oracle: WMC); the registered
prediction is fidelity 1.0 by compilation, at the float32 precision of the
circuit internals (~1e-7).

Two weighting modes:
  - value(): probabilities baked in as numeric labels (constants).
  - value_and_grad(): probabilities passed as *symbolic* labels, which
    DeepLog exposes as module inputs (non-numeric labels survive
    `_LabelProbabilityPredicate._resolve_argument`); torch autograd then
    yields DeepLog's own gradients for comparison with the analytic oracle.

Proof DNFs (not raw programs) are the interface because DeepLog's input
language is formulas; recursion is unrolled by the caller via the arena's
bounded enumeration.
"""

from __future__ import annotations

TRUE = ("true",)


def _var(fact) -> tuple:
    return (repr(fact),)


class DeepLogAdapter:
    name = "deeplog:probability(exact-circuit)"
    claimed_semantics = "distribution semantics"
    supports_grad = True

    def __init__(self):
        import deeplog  # noqa: F401  deferred

    # ------------------------------------------------------------ builders --

    def _build(self, proofs, facts, symbolic_weights: bool, probs=None):
        from deeplog import DeepLogModuleFactory

        factory = DeepLogModuleFactory()

        def conj(nodes, op):
            acc = nodes[0]
            for nd in nodes[1:]:
                acc = factory.create_binary_node(op, acc, nd)
            return acc

        proof_nodes = [
            conj([factory.create_atom(("_", ("=", _var(f), TRUE), ("boolean",)))
                  for f in sorted(pr, key=repr)], "and")
            for pr in proofs
        ]
        transformed = factory.create_transformation("probability", conj(proof_nodes, "or"))

        def label(f):
            if symbolic_weights:
                return ("_", ("w", (repr(f),)), ("probability",))
            return ("_", (str(float(probs[f])),), ("probability",))

        weight = conj(
            [factory.create_atom(("_", ("p", _var(f), label(f)), ("probability",)))
             for f in facts], "times")
        weighted = factory.create_binary_node("times", transformed, weight)
        summed = factory.create_aggregation("sum", [_var(f) for f in facts], [], weighted)
        return summed.to_module()

    # ------------------------------------------------------------ value -----

    def value(self, proofs, probs) -> float:
        facts = sorted(set().union(*proofs), key=repr)
        out = self._build(proofs, facts, symbolic_weights=False, probs=probs)()
        return float(out.reshape(-1)[0])

    def infer(self, program, base, queries, max_depth: int = 8):
        out = {}
        for q in queries:
            proofs = program.proof_supports(q, max_depth)
            out[q] = self.value(proofs, base) if proofs else 0.0
        return out

    # ------------------------------------------------------------ gradient --

    def value_and_grad(self, proofs, probs) -> tuple[float, dict]:
        """DeepLog's value and its autograd gradient w.r.t. each fact
        probability (the deployed differentiation behavior)."""
        import torch
        from deeplog.shape import get_all_symbols

        facts = sorted(set().union(*proofs), key=repr)
        module = self._build(proofs, facts, symbolic_weights=True)
        syms = list(get_all_symbols(module.get_input_shape()))

        def sym_fact_name(s):
            inner = s[1] if s[0] == "_" else s
            return inner[1][0]

        by_name = {repr(f): f for f in facts}
        ins = [torch.tensor([[probs[by_name[sym_fact_name(s)]]]], dtype=torch.float64,
                            requires_grad=True) for s in syms]
        out = module(*ins).reshape(-1)[0]
        out.backward()
        grad = {by_name[sym_fact_name(s)]: float(t.grad.reshape(-1)[0])
                for t, s in zip(ins, syms, strict=True)}
        return float(out.detach()), grad

    def grad(self, program, base, query, wrt, max_depth: int = 8):
        proofs = program.proof_supports(query, max_depth)
        if not proofs:
            return {a: 0.0 for a in wrt}
        _, g = self.value_and_grad(proofs, base)
        return {a: g.get(a, 0.0) for a in wrt}
