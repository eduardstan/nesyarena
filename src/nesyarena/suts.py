"""Reference systems-under-test (SUTs).

Each provenance mirrors a deployed NeSy aggregation strategy and provides
value() and grad() with the same differentiation behavior real systems use:
top-k differentiates through the selected proofs with the selection frozen,
min-max routes a one-hot subgradient through the bottleneck fact, clamped
add-mult has a flat region (gradient blackout) once the raw sum exceeds 1.

`claimed` declares the semantics against which semantic error is measured
(protocol D1). For probabilistic provenances the claim is distribution
semantics (oracle: WMC); for LSE the claim is the max-join over proof scores
(its oracle is max, not WMC).

Inputs mirror the oracle layer: `proofs` is a list of EDB support sets over
hashable fact keys, `probs` maps fact -> probability in [0, 1].
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
        """D1, signed: value under the SUT minus value under the claimed semantics."""
        return self.value(proofs, probs) - self.oracle(proofs, probs)


class ExactWMC(Provenance):
    name = "exact-wmc"

    def value(self, proofs, probs):
        return wmc(proofs, probs)

    def grad(self, proofs, probs):
        return wmc_with_grad(proofs, probs)[1]


class AddMult(Provenance):
    """Proof-sum Sigma_j Prod_{f in j} p_f, optionally clamped to [0, 1].
    Mirrors add-mult provenances / naive proof enumeration (rung-P violation:
    over-counts overlapping proofs, Prop. 1)."""

    def __init__(self, clamp: bool = True):
        self.clamp = clamp
        self.name = "add-mult" + ("(clamped)" if clamp else "(raw)")

    def value(self, proofs, probs):
        s = sum(proof_score(pr, probs) for pr in proofs)
        return min(1.0, s) if self.clamp else s

    def grad(self, proofs, probs):
        raw = sum(proof_score(pr, probs) for pr in proofs)
        facts = sorted(set().union(*proofs), key=repr)
        if self.clamp and raw >= 1.0:
            return {f: 0.0 for f in facts}  # flat region: gradient blackout
        g = {f: 0.0 for f in facts}
        for pr in proofs:
            sc = proof_score(pr, probs)
            for f in pr:
                g[f] += sc / max(probs[f], 1e-12)
        return g


class AddMultStraightThrough(Provenance):
    """Deployed-faithful clamp per finding F-3 (Scallop diffaddmultprob):
    the VALUE is the clamped proof-sum min(1, sum_j s_j), but the GRADIENT is
    that of the UNCLAMPED sum — validated against deployed torch-tag autograd
    to 4.8e-08 on the frozen saturation battery. The (value, gradient) pair is
    mutually inconsistent in the clamp region by construction: that is the
    deployed behavior being modeled, not a bug here."""

    name = "add-mult(straight-through)"

    def value(self, proofs, probs):
        return min(1.0, sum(proof_score(pr, probs) for pr in proofs))

    def grad(self, proofs, probs):
        return AddMult(clamp=False).grad(proofs, probs)


class TopK(Provenance):
    """Exact WMC restricted to the k highest-scoring proofs
    (difftopkproofs-style: differentiates the retained set, selection frozen;
    under-counts by the dropped mass, Prop. 2)."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"top-{k}-proofs"

    def _top(self, proofs, probs):
        return sorted(proofs, key=lambda pr: proof_score(pr, probs), reverse=True)[: self.k]

    def value(self, proofs, probs):
        return wmc(self._top(proofs, probs), probs)

    def grad(self, proofs, probs):
        _, g = wmc_with_grad(self._top(proofs, probs), probs)
        for f in set().union(*proofs):
            g.setdefault(f, 0.0)
        return g


class MinMax(Provenance):
    """max over proofs of min over facts (min-max-prob style).
    Subgradient: one-hot at the bottleneck fact of the best proof.

    Deviation from the toy (deliberate): ties between proofs/facts are broken
    lexicographically. The toy broke ties by frozenset iteration order, which
    depends on PYTHONHASHSEED — the one-hot landed on a different fact across
    processes. Any tied fact is a valid subgradient; we pick a deterministic
    one."""

    name = "min-max-prob"

    def value(self, proofs, probs):
        return max(min(probs[f] for f in pr) for pr in proofs)

    def grad(self, proofs, probs):
        bestval = self.value(proofs, probs)
        best = min((pr for pr in proofs if min(probs[f] for f in pr) == bestval),
                   key=lambda pr: sorted(repr(f) for f in pr))
        fstar = min((f for f in best if probs[f] == bestval), key=repr)
        g = {f: 0.0 for f in set().union(*proofs)}
        g[fstar] = 1.0
        return g


class LSE(Provenance):
    """Smooth surrogate for the max-join over proof scores (NTP-style).
    Bias law: max < LSE_tau <= max + tau ln P, equality iff equal scores
    (Prop. 3)."""

    claimed = "max-join over proof scores"

    def __init__(self, tau: float):
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
        for wt, pr in zip(w, proofs, strict=True):
            sc = proof_score(pr, probs)
            for f in pr:
                g[f] += float(wt) * sc / max(probs[f], 1e-12)
        return g


def registry() -> list[Provenance]:
    """The standard SUT lineup for sweeps and scorecards."""

    # return [ExactWMC(), AddMult(clamp=True), TopK(1), TopK(3), MinMax()]
    from .ltn_provenance import LTNProdProvenance, LTNGodelProvenance
    
    return [ExactWMC(), AddMult(clamp=True), TopK(1), TopK(3), MinMax(),
            LTNProdProvenance(), LTNGodelProvenance()]

