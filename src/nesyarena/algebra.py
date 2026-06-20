"""Truth-value algebras (commutative semirings) used by the engine and oracles.

The semiring reads otimes as composition along a rule body and oplus as
aggregation over alternative derivations (KR Def. 4). Idempotence of oplus is
recorded because it separates the join-based (rungs 2-5) from the
probabilistic regime (rung P): non-idempotent aggregation over proofs is
where double-counting lives.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Semiring:
    name: str
    oplus: Callable[[float, float], float]
    otimes: Callable[[float, float], float]
    zero: float
    one: float
    idempotent_oplus: bool

    def oplus_all(self, values) -> float:
        acc = self.zero
        for v in values:
            acc = self.oplus(acc, v)
        return acc

    def otimes_all(self, values) -> float:
        acc = self.one
        for v in values:
            acc = self.otimes(acc, v)
        return acc


BOOLEAN = Semiring("boolean", max, min, 0.0, 1.0, True)
MAXPROD = Semiring("maxprod", max, operator.mul, 0.0, 1.0, True)
SUMPROD = Semiring("sumprod", operator.add, operator.mul, 0.0, 1.0, False)
TROPICAL = Semiring("tropical", min, operator.add, float("inf"), 0.0, True)

ALGEBRAS = {a.name: a for a in (BOOLEAN, MAXPROD, SUMPROD, TROPICAL)}
