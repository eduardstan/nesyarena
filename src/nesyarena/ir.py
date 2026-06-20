"""Ground-program intermediate representation.

The toy implementation passed proof sets around directly; the rebuild makes
the *program* the primary object (KR Defs. 1-3, 11-16), so that

  - generators emit programs,
  - proof enumeration is derived (this module),
  - the operator engine iterates T_P on the same object (engine.py), and
  - external adapters (ProbLog, Scallop, DeepLog) compile the same object
    to backend syntax instead of re-encoding by hand.

Conventions (KR Def. 3 / Def. 9): a predicate is IDB if it heads at least one
rule, else EDB; the base interpretation f_theta supplies values for EDB atoms
and assigns the additive identity to IDB atoms, so a proof tree is only
non-trivial when fully expanded to EDB leaves within the depth budget
(Remark 6 truncation semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product


@dataclass(frozen=True, order=True)
class Atom:
    """Ground atom: predicate + constant arguments. 0-ary facts are Atom("f")."""

    pred: str
    args: tuple = ()

    def __repr__(self) -> str:
        return self.pred if not self.args else f"{self.pred}({','.join(map(str, self.args))})"


@dataclass(frozen=True)
class Rule:
    head: Atom
    body: tuple[Atom, ...]

    def __post_init__(self):
        if not isinstance(self.body, tuple):
            object.__setattr__(self, "body", tuple(self.body))

    def __repr__(self) -> str:
        return f"{self.head!r} <- {' & '.join(map(repr, self.body))}"


class ProofExplosion(RuntimeError):
    """Raised when bounded proof enumeration exceeds the configured cap."""


@dataclass(frozen=True)
class GroundProgram:
    """A finite set of ground rules. Facts live in f_theta, not in the rules."""

    rules: tuple[Rule, ...]
    _by_head: dict = field(init=False, repr=False, compare=False, hash=False, default=None)

    def __post_init__(self):
        if not isinstance(self.rules, tuple):
            object.__setattr__(self, "rules", tuple(self.rules))
        by_head: dict[Atom, list[Rule]] = {}
        for r in self.rules:
            by_head.setdefault(r.head, []).append(r)
        object.__setattr__(self, "_by_head", by_head)

    @property
    def idb_preds(self) -> frozenset[str]:
        return frozenset(r.head.pred for r in self.rules)

    @property
    def edb_atoms(self) -> frozenset[Atom]:
        idb = self.idb_preds
        return frozenset(a for r in self.rules for a in r.body if a.pred not in idb)

    def is_edb(self, atom: Atom) -> bool:
        return atom.pred not in self.idb_preds

    def rules_for(self, head: Atom) -> list[Rule]:
        return self._by_head.get(head, [])

    # ------------------------------------------------------------ proofs ----

    def proof_leaf_multisets(self, query: Atom, max_depth: int,
                             max_proofs: int = 100_000) -> set[tuple[Atom, ...]]:
        """All proof trees for `query` of depth <= max_depth (KR Defs. 14-16),
        each represented by its sorted tuple of EDB leaves *with multiplicity*
        (a fact used by two subtrees appears twice — the non-idempotent
        semiring view of Def. 15). IDB atoms that hit the depth budget are
        dropped (f_theta gives them the additive identity, Remark 6).
        """

        @lru_cache(maxsize=None)
        def rec(atom: Atom, budget: int) -> frozenset[tuple[Atom, ...]]:
            if self.is_edb(atom):
                return frozenset({(atom,)})
            if budget <= 0:
                return frozenset()
            out: set[tuple[Atom, ...]] = set()
            for rule in self.rules_for(atom):
                child_sets = [rec(b, budget - 1) for b in rule.body]
                if any(not cs for cs in child_sets):
                    continue
                for combo in product(*child_sets):
                    leaves: list[Atom] = []
                    for leaf_tuple in combo:
                        leaves.extend(leaf_tuple)
                    out.add(tuple(sorted(leaves)))
                    if len(out) > max_proofs:
                        raise ProofExplosion(
                            f"{atom!r} at depth {max_depth}: > {max_proofs} proofs")
            return frozenset(out)

        return set(rec(query, max_depth))

    def proof_supports(self, query: Atom, max_depth: int,
                       max_proofs: int = 100_000) -> list[frozenset[Atom]]:
        """Distinct EDB support *sets* of bounded proofs, deterministically
        ordered (by size, then lexicographically). This is the DNF the
        probabilistic SUTs and the WMC oracle consume."""
        supports = {frozenset(leaves)
                    for leaves in self.proof_leaf_multisets(query, max_depth, max_proofs)}
        return sorted(supports, key=lambda s: (len(s), sorted(repr(a) for a in s)))


def transitive_closure(edges, nodes) -> GroundProgram:
    """path(x,y) <- edge(x,y) ; path(x,z) <- edge(x,y) & path(y,z),
    ground over the given nodes (the KR paper's running program)."""
    rules = []
    for (u, v) in edges:
        rules.append(Rule(Atom("path", (u, v)), (Atom("edge", (u, v)),)))
        for w in nodes:
            rules.append(Rule(Atom("path", (u, w)),
                              (Atom("edge", (u, v)), Atom("path", (v, w)))))
    return GroundProgram(tuple(rules))
