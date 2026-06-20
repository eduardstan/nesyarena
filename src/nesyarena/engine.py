"""Operator-semantics engine: bounded iteration of the fact-augmented
immediate-consequence operator (KR Defs. 11-13, Remark 6 truncation), and the
proof-side aggregation it must equal (Theorem 1) — the correspondence is a
standing numeric test, not an assumption.
"""

from __future__ import annotations

from .algebra import Semiring
from .ir import Atom, GroundProgram


def iterate(program: GroundProgram, f0: dict[Atom, float], semiring: Semiring,
            n_steps: int) -> list[dict[Atom, float]]:
    """I^(0) = f0 ; I^(t+1) = f0 (+) T_P(I^(t)). Returns the full history
    [I^(0), ..., I^(n)]. f0 supplies EDB atoms; absent atoms read as zero."""
    sr = semiring
    interp = dict(f0)
    history = [dict(interp)]
    for _ in range(n_steps):
        new = dict(f0)
        for rule in program.rules:
            v = sr.one
            for b in rule.body:
                v = sr.otimes(v, interp.get(b, sr.zero))
            new[rule.head] = sr.oplus(new.get(rule.head, sr.zero), v)
        interp = new
        history.append(dict(interp))
    return history


def infer_bounded(program: GroundProgram, f0: dict[Atom, float], semiring: Semiring,
                  query: Atom, n: int) -> float:
    """I^(n)(query) — what an n-step unroller computes (truncation semantics)."""
    return iterate(program, f0, semiring, n)[n].get(query, semiring.zero)


def proof_aggregate(program: GroundProgram, f0: dict[Atom, float], semiring: Semiring,
                    query: Atom, n: int, max_proofs: int = 100_000) -> float:
    """Infer^(n)(query): oplus over proof trees of depth <= n of the otimes of
    their leaf values with multiplicity (KR Defs. 15-16). Theorem 1 says this
    equals infer_bounded for any commutative semiring; the test battery checks
    it numerically on cyclic programs with non-idempotent oplus.

    Caveat: trees are identified by their leaf multiset (ir.py); two distinct
    trees with identical leaf multisets would collapse, which matters only for
    non-idempotent oplus. On the arena's program families (one rule per proof;
    transitive closure, where derivations are walks and a walk's edge multiset
    is unique) the identification is exact.
    """
    trees = program.proof_leaf_multisets(query, n, max_proofs)
    base = f0.get(query, semiring.zero) if program.is_edb(query) else semiring.zero
    return semiring.oplus_all(
        [base] if program.is_edb(query) else
        [semiring.otimes_all(f0.get(leaf, semiring.zero) for leaf in leaves)
         for leaves in sorted(trees)])


def converge(program: GroundProgram, f0: dict[Atom, float], semiring: Semiring,
             query: Atom, tol: float = 1e-12, max_steps: int = 10_000) -> float:
    """Iterate until the query value moves less than tol (run-to-convergence
    mode). For non-convergent cases (e.g. sum-product on cyclic programs,
    Remark 12) this stops at max_steps; the caller sees the divergence in the
    returned value's magnitude."""
    sr = semiring
    interp = dict(f0)
    for _ in range(max_steps):
        new = dict(f0)
        for rule in program.rules:
            v = sr.one
            for b in rule.body:
                v = sr.otimes(v, interp.get(b, sr.zero))
            new[rule.head] = sr.oplus(new.get(rule.head, sr.zero), v)
        # convergence of the whole interpretation, not just the query: the
        # query can sit at zero for its first minimal-proof-depth iterations
        delta = max((0.0 if new.get(a, sr.zero) == interp.get(a, sr.zero)
                     else abs(new.get(a, sr.zero) - interp.get(a, sr.zero))
                     for a in set(new) | set(interp)), default=0.0)
        interp = new
        if delta <= tol:
            break
    return interp.get(query, sr.zero)
