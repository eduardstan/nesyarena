"""Reference systems-under-test (SUTs) for NeSyArena.

Each provenance mirrors a deployed NeSy aggregation strategy and provides
value() and grad() with the same differentiation behavior real systems use
(e.g. top-k differentiates through the selected proofs with selection frozen,
min-max routes a one-hot subgradient, clamped add-mult has a flat region).

`claimed` declares the semantics against which semantic error is measured
(D1 of the protocol). For probabilistic provenances the claim is distribution
semantics; for LSE the claim is the max-join (its oracle is max over proofs).
"""

from __future__ import annotations
import numpy as np
from .oracle import wmc, wmc_with_grad


def proof_score(pr, probs) -> float:
    return float(np.prod([probs[f] for f in pr]))


class Provenance:
    name = "abstract"
    claimed = "distribution semantics"

    def value(self, proofs, probs) -> float:
        raise NotImplementedError

    def grad(self, proofs, probs) -> dict:
        raise NotImplementedError

    def oracle(self, proofs, probs) -> float:
        return wmc(proofs, probs)

    def error(self, proofs, probs) -> float:
        return self.value(proofs, probs) - self.oracle(proofs, probs)


class ExactWMC(Provenance):
    name = "exact-wmc"

    def value(self, proofs, probs):
        return wmc(proofs, probs)

    def grad(self, proofs, probs):
        return wmc_with_grad(proofs, probs)[1]


class AddMult(Provenance):
    """Proof-sum: Sigma_j Prod_{f in j} p_f, optionally clamped to [0,1].
    Mirrors add-mult provenances / naive proof enumeration (KR Rung P violation)."""

    def __init__(self, clamp=True):
        self.clamp = clamp
        self.name = "add-mult" + ("(clamped)" if clamp else "(raw)")

    def value(self, proofs, probs):
        s = sum(proof_score(pr, probs) for pr in proofs)
        return min(1.0, s) if self.clamp else s

    def grad(self, proofs, probs):
        raw = sum(proof_score(pr, probs) for pr in proofs)
        facts = sorted(set().union(*proofs))
        if self.clamp and raw >= 1.0:
            return {f: 0.0 for f in facts}  # flat region: gradient blackout
        g = {f: 0.0 for f in facts}
        for pr in proofs:
            sc = proof_score(pr, probs)
            for f in pr:
                g[f] += sc / max(probs[f], 1e-12)
        return g


class TopK(Provenance):
    """Exact WMC restricted to the k highest-scoring proofs
    (difftopkproofs-style: differentiates the retained set, selection frozen)."""

    def __init__(self, k):
        self.k = k
        self.name = f"top-{k}-proofs"

    def _top(self, proofs, probs):
        return sorted(proofs, key=lambda pr: proof_score(pr, probs), reverse=True)[: self.k]

    def value(self, proofs, probs):
        return wmc(self._top(proofs, probs), probs)

    def grad(self, proofs, probs):
        v, g = wmc_with_grad(self._top(proofs, probs), probs)
        for f in set().union(*proofs):
            g.setdefault(f, 0.0)
        return g


class MinMax(Provenance):
    """max over proofs of min over facts (min-max-prob style).
    Subgradient: one-hot at the bottleneck fact of the best proof."""

    name = "min-max-prob"

    def value(self, proofs, probs):
        return max(min(probs[f] for f in pr) for pr in proofs)

    def grad(self, proofs, probs):
        best = max(proofs, key=lambda pr: min(probs[f] for f in pr))
        fstar = min(best, key=lambda f: probs[f])
        g = {f: 0.0 for f in set().union(*proofs)}
        g[fstar] = 1.0
        return g


class LSE(Provenance):
    """Smooth surrogate for the max-join over proof scores (NTP-style).
    Claimed semantics: max over proofs (its oracle is the join, not WMC)."""

    claimed = "max-join over proof scores"

    def __init__(self, tau):
        self.tau = tau
        self.name = f"lse(tau={tau})"

    def value(self, proofs, probs):
        s = np.array([proof_score(pr, probs) for pr in proofs])
        m = s.max()
        return float(m + self.tau * np.log(np.exp((s - m) / self.tau).sum()))

    def oracle(self, proofs, probs):
        return max(proof_score(pr, probs) for pr in proofs)

    def grad(self, proofs, probs):
        s = np.array([proof_score(pr, probs) for pr in proofs])
        w = np.exp((s - s.max()) / self.tau)
        w = w / w.sum()
        g = {f: 0.0 for f in set().union(*proofs)}
        for wt, pr in zip(w, proofs):
            sc = proof_score(pr, probs)
            for f in pr:
                g[f] += float(wt) * sc / max(probs[f], 1e-12)
        return g


def registry():
    return [ExactWMC(), AddMult(clamp=True), TopK(1), TopK(3), MinMax()]
