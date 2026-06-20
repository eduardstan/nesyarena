"""Structured program families (protocol G1-G3).

Generators emit *programs* (ir.GroundProgram) plus a base interpretation;
proof sets are derived by enumeration, and external adapters compile the same
program. Fact naming and rng draw order replicate the toy generator exactly,
so the golden fixtures gate this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import Atom, GroundProgram, Rule, transitive_closure

QUERY = Atom("q")


@dataclass(frozen=True)
class Instance:
    """A generated program instance: what every SUT and oracle consumes."""

    program: GroundProgram
    query: Atom
    probs: dict = field(hash=False)
    params: dict = field(hash=False)

    @property
    def proofs(self) -> list[frozenset[Atom]]:
        # depth 1 suffices for G1 (one rule per proof); TC instances override
        # via the depth recorded in params
        return self.program.proof_supports(self.query, self.params.get("depth", 1))


def overlap_family(P: int, L: int, c: int, p: float, rng=None, het: bool = False) -> Instance:
    """G1: P proofs of length L sharing c trunk facts; homogeneous prob p, or
    heterogeneous draws in [0.35, 0.9] (draw order identical to the toy:
    shared facts first, then private facts per proof j)."""
    draw = (lambda: float(rng.uniform(0.35, 0.9))) if het else (lambda: p)
    probs: dict[Atom, float] = {}
    shared = [Atom(f"s{i}") for i in range(c)]
    for f in shared:
        probs[f] = draw()
    rules = []
    for j in range(P):
        priv = [Atom(f"x{j}_{i}") for i in range(L - c)]
        for f in priv:
            probs[f] = draw()
        rules.append(Rule(QUERY, tuple(shared + priv)))
    program = GroundProgram(tuple(rules))
    return Instance(program, QUERY, probs,
                    dict(family="G1", P=P, L=L, c=c, p=p, het=het, depth=1))


def chain_family(L: int, p: float = 0.9) -> Instance:
    """G2: chain v_0 -> ... -> v_L with edge prob p, transitive closure,
    query path(v_0, v_L); minimal proof depth is L."""
    nodes = [f"v{i}" for i in range(L + 1)]
    edges = [(f"v{i}", f"v{i+1}") for i in range(L)]
    program = transitive_closure(edges, nodes)
    probs = {Atom("edge", e): p for e in edges}
    return Instance(program, Atom("path", ("v0", f"v{L}")), probs,
                    dict(family="G2-chain", L=L, p=p, depth=L))


def cyclic_family(p_ab: float = 0.9, p_ba: float = 0.9, p_bc: float = 0.8) -> Instance:
    """G2: the KR Section-5 cyclic instance a <-> b -> c, query path(a, c) —
    the canonical recursion stress test (infinitely many proofs)."""
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "a"), ("b", "c")]
    program = transitive_closure(edges, nodes)
    probs = {Atom("edge", ("a", "b")): p_ab,
             Atom("edge", ("b", "a")): p_ba,
             Atom("edge", ("b", "c")): p_bc}
    return Instance(program, Atom("path", ("a", "c")), probs,
                    dict(family="G2-cyclic", depth=2))


def surrogate_scores(P: int, s: float = 0.6, delta: float = 0.0):
    """G3: proof-score multisets for the surrogate axis, encoded as P
    single-fact proofs (fact j has probability = score of proof j). Equal
    scores when delta = 0; one boosted proof otherwise. Closed-form
    predictions: LSE bias = tau*ln(P) at delta = 0 (Prop. 3)."""
    proofs = [frozenset({Atom(f"a{j}")}) for j in range(P)]
    probs = {Atom(f"a{j}"): (s + delta if j == 0 else s) for j in range(P)}
    return proofs, probs
