"""Adapter protocol + pure Scallop compile helpers (no scallopy needed here;
the live gates run in the scallop env via scripts/gate_scallop_ir.py)."""

import pytest

from nesyarena.adapters import Adapter, ReferenceAdapter
from nesyarena.adapters.scallop import compile_rules, fact_key, render_atom
from nesyarena.generators import chain_family, overlap_family
from nesyarena.ir import Atom
from nesyarena.oracle import wmc
from nesyarena.suts import ExactWMC, TopK


def test_reference_adapter_satisfies_protocol():
    ad = ReferenceAdapter(ExactWMC())
    assert isinstance(ad, Adapter)
    assert ad.claimed_semantics == "distribution semantics"


def test_reference_adapter_infer_matches_direct_path():
    inst = overlap_family(3, 3, 1, 0.6)
    ad = ReferenceAdapter(ExactWMC(), max_depth=1)
    out = ad.infer(inst.program, inst.probs, [inst.query])
    assert out[inst.query] == pytest.approx(wmc(inst.proofs, inst.probs))


def test_reference_adapter_on_recursive_program():
    inst = chain_family(4, 0.9)
    ad = ReferenceAdapter(TopK(1), max_depth=4)
    out = ad.infer(inst.program, inst.probs, [inst.query])
    assert out[inst.query] == pytest.approx(0.9 ** 4)
    g = ad.grad(inst.program, inst.probs, inst.query, list(inst.probs))
    assert set(g) == set(inst.probs)
    # unreachable query: zero value, zero gradient, no crash
    far = Atom("path", ("v4", "v0"))
    assert ad.infer(inst.program, inst.probs, [far])[far] == 0.0


def test_scallop_compile_g1_shape():
    inst = overlap_family(2, 2, 1, 0.6)
    rules = compile_rules(inst.program)
    assert rules == ['q() = fact("s0") and fact("x0_0")',
                     'q() = fact("s0") and fact("x1_0")']


def test_scallop_compile_recursive_tc_keeps_constants_quoted():
    inst = chain_family(2, 0.9)
    rules = compile_rules(inst.program)
    assert 'path("v0", "v1") = fact("edge(v0,v1)")' in rules
    assert ('path("v0", "v2") = fact("edge(v0,v1)") and path("v1", "v2")') in rules
    # IDB atoms keep their predicate; EDB atoms are routed through fact/1
    assert all('edge(' not in r.split(" = ")[0] for r in rules)


def test_fact_key_is_atom_repr():
    assert fact_key(Atom("edge", ("a", "b"))) == "edge(a,b)"
    assert fact_key(Atom("s0")) == "s0"


def test_render_idb_zero_arity():
    inst = overlap_family(1, 2, 0, 0.5)
    assert render_atom(inst.program, inst.query) == "q()"
