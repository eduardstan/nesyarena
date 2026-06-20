"""Generator parity with the toy: identical fact names, probabilities (incl.
heterogeneous rng draw order), and proof sets — but now derived from programs."""

import math

import numpy as np
import pytest

from nesyarena.generators import chain_family, cyclic_family, overlap_family, surrogate_scores
from nesyarena.ir import Atom
from nesyarena.suts import LSE


def _names(proofs):
    return {frozenset(a.pred for a in pr) for pr in proofs}


def test_g1_parity_with_toy_fixtures(gold):
    for fx in gold["g1_instances"]:
        rng = np.random.default_rng(fx["seed"]) if fx["het"] else None
        inst = overlap_family(fx["P"], fx["L"], fx["c"], fx["p"], rng=rng, het=fx["het"])
        assert {a.pred: v for a, v in inst.probs.items()} == pytest.approx(fx["probs"])
        assert _names(inst.proofs) == {frozenset(pr) for pr in fx["proofs"]}
        assert len(inst.proofs) == fx["P"]


def test_g1_proofs_derived_from_program_one_per_rule():
    inst = overlap_family(4, 3, 2, 0.6)
    assert len(inst.program.rules) == 4
    assert all(len(pr) == 3 for pr in inst.proofs)
    trunk = {Atom("s0"), Atom("s1")}
    assert all(trunk <= pr for pr in inst.proofs)


def test_chain_family_structure():
    inst = chain_family(4, 0.5)
    assert inst.query == Atom("path", ("v0", "v4"))
    sup = inst.program.proof_supports(inst.query, inst.params["depth"])
    assert sup == [frozenset(inst.probs)]
    assert inst.program.proof_supports(inst.query, inst.params["depth"] - 1) == []


def test_cyclic_family_is_the_section5_instance():
    inst = cyclic_family()
    assert inst.probs[Atom("edge", ("a", "b"))] == 0.9
    assert inst.probs[Atom("edge", ("b", "c"))] == 0.8
    assert len(inst.program.proof_supports(inst.query, 6)) == 2  # simple + loop


def test_surrogate_scores_drive_the_bias_law():
    for P in (2, 5):
        proofs, probs = surrogate_scores(P, s=0.6)
        bias = LSE(0.1).error(proofs, probs)
        assert bias == pytest.approx(0.1 * math.log(P), rel=1e-12)
    proofs, probs = surrogate_scores(3, s=0.5, delta=0.2)
    assert LSE(0.05).error(proofs, probs) < 0.05 * math.log(3)
