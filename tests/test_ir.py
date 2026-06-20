import pytest

from nesyarena.ir import Atom, GroundProgram, ProofExplosion, Rule, transitive_closure

Q = Atom("q")


def g1_program(P, L, c):
    """One rule per proof: q <- s_0..s_{c-1} & x_j_0..x_j_{L-c-1} (toy G1 shape)."""
    shared = [Atom(f"s{i}") for i in range(c)]
    rules = []
    for j in range(P):
        priv = [Atom(f"x{j}_{i}") for i in range(L - c)]
        rules.append(Rule(Q, tuple(shared + priv)))
    return GroundProgram(tuple(rules))


def test_edb_idb_partition():
    prog = g1_program(2, 3, 1)
    assert prog.idb_preds == {"q"}
    assert Atom("s0") in prog.edb_atoms and not prog.is_edb(Q)


def test_g1_proofs_are_one_per_rule_at_depth_1():
    prog = g1_program(3, 3, 1)
    assert prog.proof_supports(Q, 0) == []
    supports = prog.proof_supports(Q, 1)
    assert len(supports) == 3
    assert frozenset({Atom("s0"), Atom("x1_0"), Atom("x1_1")}) in supports
    # depth saturates: deeper budgets add nothing for a depth-1 program
    assert supports == prog.proof_supports(Q, 5)


def test_chain_proof_depth_equals_length():
    L = 8
    nodes = [f"v{i}" for i in range(L + 1)]
    edges = [(f"v{i}", f"v{i+1}") for i in range(L)]
    prog = transitive_closure(edges, nodes)
    query = Atom("path", ("v0", f"v{L}"))
    assert prog.proof_supports(query, L - 1) == []  # below horizon: no proofs
    sup = prog.proof_supports(query, L)
    assert sup == [frozenset(Atom("edge", e) for e in edges)]


def test_cyclic_enumeration_terminates_and_loops_add_support():
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "a"), ("b", "c")]
    prog = transitive_closure(edges, nodes)
    query = Atom("path", ("a", "c"))
    simple = frozenset({Atom("edge", ("a", "b")), Atom("edge", ("b", "c"))})
    loop = simple | {Atom("edge", ("b", "a"))}
    sup2 = prog.proof_supports(query, 2)
    assert sup2 == [simple]
    sup6 = prog.proof_supports(query, 6)
    assert simple in sup6 and loop in sup6 and len(sup6) == 2


def test_multiplicity_view_keeps_repeated_facts():
    # via the a<->b loop, edge(a,b) is used twice in the once-around proof
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "a"), ("b", "c")]
    prog = transitive_closure(edges, nodes)
    multis = prog.proof_leaf_multisets(Atom("path", ("a", "c")), 4)
    twice = [m for m in multis if m.count(Atom("edge", ("a", "b"))) == 2]
    assert twice, "loop proof should use edge(a,b) with multiplicity 2"


def test_explosion_guard():
    prog = g1_program(40, 2, 0)
    with pytest.raises(ProofExplosion):
        prog.proof_leaf_multisets(Q, 1, max_proofs=10)


def test_deterministic_support_order():
    prog = g1_program(4, 3, 2)
    assert prog.proof_supports(Q, 1) == prog.proof_supports(Q, 1)
    sizes = [len(s) for s in prog.proof_supports(Q, 1)]
    assert sizes == sorted(sizes)
