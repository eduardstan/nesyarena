"""Scoring metrics (protocol D1-D5).

D1 (semantic error) lives on the SUT itself (`Provenance.error`): signed,
oracle-grounded, computed from the identical base interpretation. This module
adds the derived metrics: fidelity profiles over sweeps, depth horizons, and
gradient liveness. Learning-level metrics (true-marginal calibration, transfer
efficiency) arrive with the experiment harness.
"""

from __future__ import annotations

from typing import Callable, Iterable

from .suts import Provenance


def fidelity(sut: Provenance, instances: Iterable) -> float:
    """phi over a sweep family (D2): 1 - mean |semantic error|, clamped to
    [0, 1] for scoring (raw errors are what sweeps report)."""
    errs = [abs(sut.error(inst.proofs, inst.probs)) for inst in instances]
    if not errs:
        raise ValueError("empty sweep")
    return max(0.0, min(1.0, 1.0 - sum(errs) / len(errs)))


def depth_horizon(error_at_depth: Callable[[int], float], delta: float = 1e-6,
                  max_depth: int = 32) -> int | None:
    """D3: smallest required proof depth L at which |error| exceeds delta
    (None if the system stays faithful through max_depth). For an n-step
    unroller the theory predicts L = n + 1 (Thm. 1 truncation)."""
    for depth in range(1, max_depth + 1):
        if abs(error_at_depth(depth)) > delta:
            return depth
    return None


def gradient_liveness(sut_grad: dict, oracle_grad: dict, tol: float = 1e-9) -> float:
    """D4 starvation indicator, aggregated: the fraction of facts with
    non-zero oracle gradient that also receive non-zero gradient from the
    SUT. 1.0 = no starvation; min-max's one-hot subgradient gives 1/m."""
    live = [f for f, g in oracle_grad.items() if abs(g) > tol]
    if not live:
        return 1.0
    return sum(abs(sut_grad.get(f, 0.0)) > tol for f in live) / len(live)
