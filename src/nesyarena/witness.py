"""Witness synthesis (protocol D6): the smallest generator configuration on
which a SUT's semantic error exceeds delta, found by grid search over the G1
family and then greedily shrunk along each parameter (QuickCheck-style).
The machine-found analogue of hand-crafted failure micro-examples.
"""

from __future__ import annotations

from .generators import overlap_family
from .suts import Provenance


def _error_at(sut: Provenance, P: int, L: int, c: int, p: float) -> float:
    inst = overlap_family(P, L, c, p)
    return sut.error(inst.proofs, inst.probs)


def find_witness(sut: Provenance, delta: float = 0.05) -> dict | None:
    """Returns the shrunk minimal failing configuration (with its signed
    error), or None if the SUT stays within delta on the whole grid."""
    cands = sorted(
        (c + P * (L - c), P, L, c, p)
        for L in (2, 3, 4) for c in range(L) for P in range(1, 7)
        for p in (0.3, 0.6, 0.9) if c + P * (L - c) <= 22)

    hit = None
    for (_, P, L, c, p) in cands:
        if abs(_error_at(sut, P, L, c, p)) > delta:
            hit = [P, L, c, p]
            break
    if hit is None:
        return None
    P, L, c, p = hit
    improved = True
    while improved:
        improved = False
        for (P2, L2, c2, p2) in ((P - 1, L, c, p),
                                 (P, L - 1, min(c, L - 2), p),
                                 (P, L, c - 1, p)):
            if P2 >= 1 and L2 >= 1 and 0 <= c2 < L2 and abs(_error_at(sut, P2, L2, c2, p2)) > delta:
                P, L, c, p = P2, L2, c2, p2
                improved = True
                break
    return dict(P=P, L=L, c=c, p=p, m=c + P * (L - c), err=_error_at(sut, P, L, c, p))
