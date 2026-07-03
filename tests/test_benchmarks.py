"""The frozen instance set is self-consistent and reproducible:
stored proofs re-derive from the stored programs, stored oracle values and
gradients recompute from the stored probabilities, and the anchor instances
carry the published numbers."""

import os

import pytest

from nesyarena.benchmarks import DEFAULT_PATH, load_instances, parse_atom
from nesyarena.ir import Atom
from nesyarena.oracle import wmc, wmc_with_grad

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_PATH),
                                reason="benchmarks/instances_v1.json not generated")


@pytest.fixture(scope="module")
def instances():
    return load_instances()


def test_expected_batteries_and_counts(instances):
    per = {}
    for i in instances:
        per[i.battery] = per.get(i.battery, 0) + 1
    assert per == {"values": 50, "gradients": 10, "chains": 4,
                   "cyclic": 1, "probes": 2, "witnesses": 4}


def test_proofs_rederive_from_programs(instances):
    for inst in instances:
        got = set(inst.program.proof_supports(inst.query, inst.depth))
        assert got == set(inst.proofs), inst.id


def test_oracle_values_recompute(instances):
    for inst in instances:
        assert wmc(inst.proofs, inst.probs) == pytest.approx(
            inst.oracle_value, abs=1e-12), inst.id


def test_oracle_gradients_recompute(instances):
    checked = 0
    for inst in instances:
        if not inst.oracle_grad:
            continue
        _, g = wmc_with_grad(inst.proofs, inst.probs)
        for a, val in inst.oracle_grad.items():
            assert g[a] == pytest.approx(val, abs=1e-12), (inst.id, a)
        checked += 1
    assert checked == 10  # the whole gradients battery carries grads


def test_anchor_values(instances):
    by_id = {i.id: i for i in instances}
    # chain L=8, p=0.9: single proof, value 0.9^8
    assert by_id["chain-L8"].oracle_value == pytest.approx(0.9 ** 8)
    # cyclic Section-5 instance: exact value 0.72 (loop supports subsumed)
    assert by_id["cyclic-s5"].oracle_value == pytest.approx(0.72)
    # diamond probes: exact 0.5904 with or without the back-edge
    assert by_id["probe-diamond-dag"].oracle_value == pytest.approx(0.5904)
    assert by_id["probe-diamond-cycle"].oracle_value == pytest.approx(0.5904)
    # witness configs: two disjoint single-fact proofs at 0.6 -> 0.84
    assert by_id["witness-addmult"].oracle_value == pytest.approx(0.84)


def test_atom_roundtrip():
    for s in ("s0", "x1_0", "edge(a,b)", "path(v0,v10)"):
        assert repr(parse_atom(s)) == s
    assert parse_atom("edge(a,b)") == Atom("edge", ("a", "b"))
