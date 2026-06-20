"""The adapter interface (protocol C1 — the community-facing standard).

An adapter wraps one backend configuration (e.g. "scallop:topkproofs(k=3)")
and answers queries over a ground program and a base interpretation. The
contract:

  - `claimed_semantics` names the semantics the system claims to compute;
    the arena scores the adapter against the *oracle for that claim*.
  - infer() consumes the identical (program, base) the oracle consumes.
  - grad(), where supported, returns d value / d base[fact] with the
    backend's own differentiation behavior (not the oracle's).

Keep adapters thin: no normalization, no correction. A backend that deviates
from its claim must deviate through the adapter too — discrepancies are
findings, not bugs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..ir import Atom, GroundProgram
from ..suts import Provenance


@runtime_checkable
class Adapter(Protocol):
    name: str
    claimed_semantics: str
    supports_grad: bool

    def infer(self, program: GroundProgram, base: dict[Atom, float],
              queries: list[Atom]) -> dict[Atom, float]: ...

    def grad(self, program: GroundProgram, base: dict[Atom, float],
             query: Atom, wrt: list[Atom]) -> dict[Atom, float]: ...


class ReferenceAdapter:
    """Wraps a reference SUT (suts.py) behind the adapter interface: proofs
    are enumerated from the program to `max_depth`, then aggregated by the
    SUT. The 'idealized' system the deployed backends are compared against."""

    def __init__(self, sut: Provenance, max_depth: int = 8):
        self.sut = sut
        self.max_depth = max_depth
        self.name = f"reference:{sut.name}"
        self.claimed_semantics = sut.claimed
        self.supports_grad = True

    def _proofs(self, program: GroundProgram, query: Atom):
        return program.proof_supports(query, self.max_depth)

    def infer(self, program, base, queries):
        out = {}
        for q in queries:
            proofs = self._proofs(program, q)
            out[q] = self.sut.value(proofs, base) if proofs else 0.0
        return out

    def grad(self, program, base, query, wrt):
        proofs = self._proofs(program, query)
        g = self.sut.grad(proofs, base) if proofs else {}
        return {a: g.get(a, 0.0) for a in wrt}
