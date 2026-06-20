"""Exact oracles.

Distribution-semantics oracle: brute-force weighted model counting (WMC) over
independent Boolean facts, with analytic gradients in a single vectorized pass:

    dP/dp_f = sum_{worlds w |= q} pi(w) * (b_f(w) - p_f) / (p_f (1 - p_f))
            = P(q | f) - P(q | not f).

Exact and differentiable for m <= MAX_FACTS facts; beyond that use the ProbLog
oracle (exact PLP inference via knowledge compilation, validated against the
brute force to ~1e-16 by the test battery — the battery is the contract that
the oracle never silently rots).

Facts are any hashable keys (Atom from ir.py, or plain strings); a proof is an
iterable of facts (its EDB support set); `proofs` is the query's DNF.
"""

from __future__ import annotations

import numpy as np

MAX_FACTS = 22


def _setup(proofs, probs):
    facts = sorted(set().union(*proofs), key=repr) if proofs else []
    m = len(facts)
    if m > MAX_FACTS:
        raise ValueError(f"{m} facts exceeds brute-force limit {MAX_FACTS}; use problog_value")
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
    proofs = list(proofs)
    if not proofs:
        return 0.0
    _, _, _, wprob, sat = _setup(proofs, probs)
    return float(wprob[sat].sum())


def wmc_with_grad(proofs, probs):
    """Returns (value, {fact: dP/dp_fact}) in one vectorized pass."""
    proofs = list(proofs)
    if not proofs:
        return 0.0, {}
    facts, p, bits, wprob, sat = _setup(proofs, probs)
    swp = wprob * sat
    val = float(swp.sum())
    denom = np.clip(p * (1.0 - p), 1e-12, None)
    gvec = (swp[:, None] * (bits.astype(float) - p[None, :])).sum(axis=0) / denom
    return val, {f: float(g) for f, g in zip(facts, gvec, strict=True)}


# ----------------------------------------------------------- ProbLog oracle ----

def problog_program(proofs, probs, fact_name=None) -> str:
    """Render a DNF query as a ProbLog program. `fact_name` maps a fact key to
    a valid ProbLog atom name (default: lowercase repr, alphanumerics + _)."""
    name = fact_name or _default_fact_name
    facts = sorted(set().union(*proofs), key=repr)
    lines = [f"{probs[f]}::{name(f)}." for f in facts]
    for pr in proofs:
        lines.append("q :- " + ", ".join(sorted(name(f) for f in pr)) + ".")
    lines.append("query(q).")
    return "\n".join(lines)


def _default_fact_name(f) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in repr(f)).lower().strip("_")
    return "f_" + s


def problog_value(proofs, probs, fact_name=None) -> float:
    """Exact inference via ProbLog (the > MAX_FACTS oracle)."""
    from problog import get_evaluatable
    from problog.program import PrologString

    src = problog_program(proofs, probs, fact_name)
    res = get_evaluatable().create_from(PrologString(src)).evaluate()
    return float(list(res.values())[0])


# ------------------------------------------------------------ graph oracles ----

def graph_value(algebra: str, edges, weights, src, dst) -> float:
    """Direct oracles for idempotent algebras on transitive closure:
    'boolean' (reachability), 'tropical' (shortest path), 'maxprod'
    (max-reliability path). Fixed-point iteration to convergence — exact for
    idempotent algebras on finite graphs."""
    nodes = sorted({u for u, _ in edges} | {v for _, v in edges})
    if algebra == "boolean":
        plus, times, zero = max, min, 0.0
    elif algebra == "tropical":
        plus, times, zero = min, (lambda a, b: a + b), float("inf")
    elif algebra == "maxprod":
        plus, times, zero = max, (lambda a, b: a * b), 0.0
    else:
        raise ValueError(algebra)
    val = {(u, v): weights[(u, v)] for (u, v) in edges}
    for _ in range(2 * len(nodes) + 4):
        new = dict(val)
        for (u, v) in edges:
            for w in nodes:
                if (v, w) in val:
                    cand = times(weights[(u, v)], val[(v, w)])
                    cur = new.get((u, w), zero)
                    new[(u, w)] = plus(cur, cand)
        val = new
    return val.get((src, dst), zero)
