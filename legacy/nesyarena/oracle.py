"""Exact oracles for NeSyArena.

Distribution-semantics oracle: brute-force weighted model counting (WMC) over
independent Boolean facts, with analytic gradients in a single vectorized pass:

    dP/dp_f = sum_{worlds w |= q} pi(w) * (b_f(w) - p_f) / (p_f (1 - p_f))

which equals P(q | f) - P(q | not f). Exact, differentiable, and fast for
m <= ~22 facts. Beyond that, use the ProbLog adapter (adapters.py).
"""

from __future__ import annotations
import numpy as np

MAX_FACTS = 22


def _setup(proofs, probs):
    facts = sorted(set().union(*proofs)) if proofs else []
    m = len(facts)
    if m > MAX_FACTS:
        raise ValueError(f"{m} facts exceeds brute-force limit {MAX_FACTS}; use ProbLog oracle")
    idx = {f: i for i, f in enumerate(facts)}
    masks = np.array([sum(1 << idx[f] for f in pr) for pr in proofs], dtype=np.int64)
    worlds = np.arange(1 << m, dtype=np.int64)
    bits = ((worlds[:, None] >> np.arange(m)) & 1).astype(bool) if m else np.zeros((1, 0), bool)
    p = np.array([probs[f] for f in facts], dtype=float)
    wprob = np.prod(np.where(bits, p, 1.0 - p), axis=1) if m else np.array([1.0])
    sat = np.zeros(len(worlds), dtype=bool)
    for mk in masks:
        sat |= (worlds & mk) == mk
    return facts, p, bits, wprob, sat


def wmc(proofs, probs) -> float:
    """Exact P( OR_j AND_{f in proof_j} f ) under independent facts."""
    if not proofs:
        return 0.0
    _, _, _, wprob, sat = _setup(proofs, probs)
    return float(wprob[sat].sum())


def wmc_with_grad(proofs, probs):
    """Returns (value, {fact: dP/dp_fact}) in one vectorized pass."""
    if not proofs:
        return 0.0, {}
    facts, p, bits, wprob, sat = _setup(proofs, probs)
    swp = wprob * sat
    val = float(swp.sum())
    denom = np.clip(p * (1.0 - p), 1e-12, None)
    gvec = (swp[:, None] * (bits.astype(float) - p[None, :])).sum(axis=0) / denom
    return val, {f: float(g) for f, g in zip(facts, gvec)}


def graph_value(algebra: str, edges, probs, src, dst) -> float:
    """Direct graph oracles for non-probabilistic algebras on transitive closure.

    algebra in {'boolean', 'tropical', 'maxprod'}; probs maps edge->weight.
    Implemented as fixed-point iteration to convergence (exact for these
    idempotent algebras on finite graphs)."""
    nodes = sorted({u for u, _ in edges} | {v for _, v in edges})
    if algebra == "boolean":
        plus, times, zero = max, min, 0.0
    elif algebra == "tropical":
        plus, times, zero = min, (lambda a, b: a + b), float("inf")
    elif algebra == "maxprod":
        plus, times, zero = max, (lambda a, b: a * b), 0.0
    else:
        raise ValueError(algebra)
    val = {(u, v): probs[(u, v)] for (u, v) in edges}
    for _ in range(2 * len(nodes) + 4):
        new = dict(val)
        for (u, v) in edges:
            for w in nodes:
                if (v, w) in val:
                    cand = times(probs[(u, v)], val[(v, w)])
                    cur = new.get((u, w), zero)
                    new[(u, w)] = plus(cur, cand)
        val = new
    return val.get((src, dst), zero)
