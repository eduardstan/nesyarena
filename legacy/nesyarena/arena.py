"""Operator-semantics engine (KR Defs. 11-13, Remark 6 truncation),
program generators (protocol G1-G3), scoring, witness synthesis, adapters."""

from __future__ import annotations
import numpy as np
from .oracle import wmc

# ---------------------------------------------------------------- engine ----

def iterate(ground_rules, f0, oplus, otimes, zero, one, n_steps):
    """I^(0)=f0 ; I^(t+1) = f0 (+) T_P(I^(t)). Returns the full history."""
    I = dict(f0)
    history = [dict(I)]
    for _ in range(n_steps):
        new = dict(f0)
        for head, body in ground_rules:
            v = one
            for b in body:
                v = otimes(v, I.get(b, zero))
            new[head] = oplus(new.get(head, zero), v)
        I = new
        history.append(dict(I))
    return history


def ground_tc(edges, nodes):
    """path(x,y) <- edge(x,y) ; path(x,z) <- edge(x,y) /\\ path(y,z)."""
    rules = []
    for (u, v) in edges:
        rules.append((("path", u, v), [("edge", u, v)]))
        for w in nodes:
            rules.append((("path", u, w), [("edge", u, v), ("path", v, w)]))
    return rules

# ------------------------------------------------------------ generators ----

def overlap_family(P, L, c, p, rng=None, het=False):
    """G1: P proofs of length L sharing c trunk facts. het=True draws
    heterogeneous probabilities in [0.35, 0.9] from rng."""
    draw = (lambda: float(rng.uniform(0.35, 0.9))) if het else (lambda: p)
    probs, proofs = {}, []
    shared = [f"s{i}" for i in range(c)]
    for f in shared:
        probs[f] = draw()
    for j in range(P):
        priv = [f"x{j}_{i}" for i in range(L - c)]
        for f in priv:
            probs[f] = draw()
        proofs.append(frozenset(shared + priv))
    return proofs, probs


def chain(L, p=0.9):
    nodes = [f"v{i}" for i in range(L + 1)]
    edges = [(f"v{i}", f"v{i+1}") for i in range(L)]
    probs = {e: p for e in edges}
    return nodes, edges, probs


def subset_union(proofs, idxs):
    """A query defined as the union of a subset of proofs (for E6 batteries)."""
    return [proofs[i] for i in idxs]

# --------------------------------------------------------------- scoring ----

def sweep_overlap(provs, Ps, cs, ps, L=3):
    recs = []
    for p in ps:
        for c in cs:
            for P in Ps:
                if c >= L:
                    continue
                m = c + P * (L - c)
                if m > 22:
                    continue
                proofs, probs = overlap_family(P, L, c, p)
                ex = wmc(proofs, probs)
                for pv in provs:
                    recs.append(dict(p=p, c=c, P=P, L=L, m=m, sut=pv.name,
                                     exact=ex, err=pv.value(proofs, probs) - ex))
    return recs


def find_witness(prov, delta=0.05):
    """Smallest G1 configuration (by total facts, then P, L, c) with |err|>delta,
    then greedily shrunk along each dimension (QuickCheck-style)."""
    cands = sorted(
        [(c + P * (L - c), P, L, c, p)
         for L in (2, 3, 4) for c in range(L) for P in range(1, 7)
         for p in (0.3, 0.6, 0.9) if c + P * (L - c) <= 22])

    def err_at(P, L, c, p):
        proofs, probs = overlap_family(P, L, c, p)
        return prov.value(proofs, probs) - wmc(proofs, probs)

    hit = None
    for (_, P, L, c, p) in cands:
        if abs(err_at(P, L, c, p)) > delta:
            hit = [P, L, c, p]
            break
    if hit is None:
        return None
    P, L, c, p = hit
    improved = True
    while improved:
        improved = False
        for cand in ((P - 1, L, c, p), (P, L - 1, min(c, L - 2), p), (P, L, c - 1, p)):
            P2, L2, c2, p2 = cand
            if P2 >= 1 and L2 >= 1 and 0 <= c2 < L2 and abs(err_at(P2, L2, c2, p2)) > delta:
                P, L, c, p = P2, L2, c2, p2
                improved = True
                break
    return dict(P=P, L=L, c=c, p=p, m=c + P * (L - c), err=err_at(P, L, c, p))

# -------------------------------------------------------------- adapters ----

def problog_program(proofs, probs):
    lines = [f"{probs[f]}::{f}." for f in sorted(set().union(*proofs))]
    for pr in proofs:
        lines.append("q :- " + ", ".join(sorted(pr)) + ".")
    lines.append("query(q).")
    return "\n".join(lines)


def problog_value(proofs, probs) -> float:
    """Exact inference via ProbLog (validated vs brute force to ~1e-17)."""
    from problog.program import PrologString
    from problog import get_evaluatable
    res = get_evaluatable().create_from(PrologString(problog_program(proofs, probs))).evaluate()
    return float(list(res.values())[0])


SCALLOP_ADAPTER_NOTE = """
Scallop adapter (for coder B; requires Rust >= 1.85 via rustup, or the
official Docker image -- pip wheel does not exist and apt Rust 1.75 fails on
an edition2024 transitive dependency, verified 2026-06-10):

    import scallopy
    ctx = scallopy.ScallopContext(provenance="topkproofs", k=3)
    ctx.add_relation("fact", (int,))
    ctx.add_facts("fact", [(probs[f], (i,)) for i, f in enumerate(facts)])
    ctx.add_rule("q() = fact(0), fact(1), fact(2)")   # one rule per proof
    ctx.run()
    value = list(ctx.relation("q"))                    # tagged result

Gate G2 (protocol day 7): compare {addmultprob, minmaxprob, topkproofs(k)}
against the reference SUTs in provenances.py on 50 G1 instances; any
discrepancy is a finding about Scallop -- log it, do not 'fix' it away.
"""
