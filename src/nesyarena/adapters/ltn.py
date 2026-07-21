"""LTN adapter (LTNtorch) — the fuzzy axis.

Logic Tensor Networks evaluate formulas under real logic: a t-norm for
conjunction and a t-conorm for disjunction, differentiable end to end. These
two Provenances aggregate a query's proof DNF with the *deployed* connective
implementations from ``ltn.fuzzy_ops`` (never reimplemented here) and return
LTN's own autograd gradients — the same rules every other backend follows
(docs/ADAPTERS.md: report what the system computes, differentiate its own
value function).

Claimed semantics, per configuration:

- ``ltn:product`` (AndProd, OrProbSum: a AND b = ab, a OR b = a + b - ab)
  claims **distribution semantics**: on fact-disjoint proofs the events are
  independent, and the probabilistic sum is then exactly P(A or B), so the
  claim is correct there and degrades precisely where proofs share facts —
  the connectives assume independence that the overlap structure violates.
  Its semantic error is therefore a direct, measurable image of the
  independence assumption, and its sign flips with structure: positive
  (over-count) on overlapping proofs, negative (under-count) on mutually
  exclusive ones, where independence is again the wrong assumption in the
  opposite direction.

- ``ltn:godel`` (AndMin, OrMax) claims **Gödel real logic**, whose exact
  evaluation on a monotone DNF *is* the min-max algebra (t-norm = min,
  t-conorm = max — a one-line property, not an empirical fact). Its oracle is
  therefore that evaluation itself, so conformance error is ~0 by
  construction at tensor precision, and the interesting quantity is the
  *cross-semantics distance* to distribution semantics — identical to the
  min-max reference SUT's error, which doubles as an independent
  implementation check of ``fuzzy_ops``.

Gradient semantics note: torch's min/max split gradients across tied inputs,
so on homogeneous (tie-heavy) batteries ``ltn:godel`` spreads gradient mass
where the reference MinMax routes a one-hot subgradient; on tie-free inputs
the two coincide. Gradient metrics for LTN are therefore quoted from the
tie-free battery only.

Scope: proof-DNF level (like the reference SUTs); recursive programs reach
these Provenances through the arena's bounded enumeration.
"""

from __future__ import annotations

import torch

from ..suts import MinMax, Provenance


class _LTNProvenance(Provenance):
    """Shared aggregation: deployed fuzzy_ops AND within each proof, deployed
    fuzzy_ops OR across proofs; gradients via torch autograd through the
    deployed operators."""

    def __init__(self):
        import ltn.fuzzy_ops  # deferred: the arena runs without LTNtorch

        self._ops = ltn.fuzzy_ops
        self.and_op, self.or_op = self._make_ops()

    def _make_ops(self):
        raise NotImplementedError

    def _formula(self, proofs, tensors: dict) -> torch.Tensor:
        # Pairwise folds with no unit seed: nested binary connectives are the
        # deployed usage, and under stable=True even And(1, x) != x, so
        # seeding at 1 would add a spurious stabilization pass per fact.
        vals = []
        for pr in proofs:
            facts = sorted(pr, key=repr)
            v = tensors[facts[0]]
            for f in facts[1:]:
                v = self.and_op(v, tensors[f])
            vals.append(v)
        acc = vals[0]
        for v in vals[1:]:
            acc = self.or_op(acc, v)
        return acc

    def value(self, proofs, probs) -> float:
        if not proofs:
            return 0.0
        tensors = {f: torch.tensor(float(p)) for f, p in probs.items()}
        return float(self._formula(proofs, tensors))

    def grad(self, proofs, probs) -> dict:
        if not proofs:
            return {}
        facts = sorted(set().union(*proofs), key=repr)
        tensors = {f: torch.tensor(float(probs[f]), requires_grad=True)
                   for f in facts}
        self._formula(proofs, tensors).backward()
        return {f: float(tensors[f].grad) for f in facts}


class LTNProduct(_LTNProvenance):
    """Product real logic (AndProd / OrProbSum). Claim: distribution
    semantics — exact on fact-disjoint proofs, independence-biased
    elsewhere (see module docstring for the sign-flip law)."""

    name = "ltn:product"
    claimed = "distribution semantics"

    def _make_ops(self):
        return self._ops.AndProd(), self._ops.OrProbSum()


class LTNGodel(_LTNProvenance):
    """Gödel real logic (AndMin / OrMax). Claim: Gödel evaluation, whose
    exact value on a monotone DNF is the min-max algebra; the oracle is that
    evaluation, so this row measures implementation conformance. The
    cross-semantics distance to distribution semantics equals the min-max
    reference SUT's error and is reported separately."""

    name = "ltn:godel"
    claimed = "Gödel real logic (min-max evaluation)"

    def _make_ops(self):
        return self._ops.AndMin(), self._ops.OrMax()

    def oracle(self, proofs, probs) -> float:
        return MinMax().value(proofs, probs)

    def cross_semantics_error(self, proofs, probs) -> float:
        """Signed distance to distribution semantics (WMC) — the fuzzy-vs-
        probabilistic gap, distinct from conformance to the Gödel claim."""
        from ..oracle import wmc

        return self.value(proofs, probs) - wmc(proofs, probs)
