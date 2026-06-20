"""SUT parity with the toy fixtures + the registered error-law battery
(Propositions 1-3; Prop. 4 truncation lives with the engine tests)."""

import math

import numpy as np
import pytest
from conftest import rand_overlap_instance

from nesyarena.oracle import wmc
from nesyarena.suts import LSE, AddMult, ExactWMC, MinMax, TopK, proof_score

SUTS = {s.name: s for s in
        [ExactWMC(), AddMult(clamp=True), AddMult(clamp=False), TopK(1), TopK(3), MinMax()]}


def _proofs_probs(inst):
    return [frozenset(pr) for pr in inst["proofs"]], inst["probs"]


def test_sut_value_and_grad_parity_with_toy(gold):
    for inst in gold["g1_instances"]:
        proofs, probs = _proofs_probs(inst)
        for name, rec in inst["suts"].items():
            sut = SUTS[name]
            assert sut.value(proofs, probs) == pytest.approx(rec["value"], abs=1e-14), name
            g = sut.grad(proofs, probs)
            assert set(g) == set(rec["grad"]), name
            if name == "min-max-prob" and not inst["het"]:
                # homogeneous probs: the toy's one-hot tie-break depended on
                # PYTHONHASHSEED; check the semantic property, not the tie
                hot = [f for f, gv in g.items() if gv != 0.0]
                assert len(hot) == 1
                assert probs[hot[0]] == pytest.approx(sut.value(proofs, probs), abs=1e-14)
                continue
            for f, gv in rec["grad"].items():
                assert g[f] == pytest.approx(gv, abs=1e-12), (name, f)


def test_lse_parity_with_toy(gold):
    for case in gold["lse_cases"]:
        proofs, probs = rand_overlap_instance(  # deterministic: homogeneous p
            np.random.default_rng(0), P=case["P"], L=case["L"], c=case["c"], p=case["p"])
        sut = LSE(case["tau"])
        assert sut.value(proofs, probs) == pytest.approx(case["value"], abs=1e-14)
        assert sut.oracle(proofs, probs) == pytest.approx(case["oracle"], abs=1e-14)
        g = sut.grad(proofs, probs)
        for f, gv in case["grad"].items():
            assert g[f] == pytest.approx(gv, abs=1e-12)


# ------------------------------------------------------- error-law battery ----

def test_prop1_addmult_overcounts_within_bonferroni():
    """0 <= proof-sum - P* <= sum_{i<j} P(pi_i & pi_j); zero iff disjoint."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        proofs, probs = rand_overlap_instance(rng)
        raw = AddMult(clamp=False)
        err = raw.error(proofs, probs)
        assert err >= -1e-12
        pairwise = sum(proof_score(proofs[i] | proofs[j], probs)
                       for i in range(len(proofs)) for j in range(i + 1, len(proofs)))
        assert err <= pairwise + 1e-12
    # fact-disjoint supports are NOT event-disjoint: for P=2 the over-count
    # is exactly P(pi_1 & pi_2) = prod over the union (inclusion-exclusion)
    proofs, probs = rand_overlap_instance(rng, P=2, L=2, c=0, p=0.6)
    err = AddMult(clamp=False).error(proofs, probs)
    assert err == pytest.approx(proof_score(proofs[0] | proofs[1], probs), abs=1e-14)
    # a single proof is the only structurally guaranteed zero-error case
    proofs1, probs1 = rand_overlap_instance(rng, P=1, L=3, c=0, p=0.6)
    assert AddMult(clamp=False).error(proofs1, probs1) == pytest.approx(0.0, abs=1e-14)


def test_prop1_clamp_blackout():
    proofs, probs = rand_overlap_instance(np.random.default_rng(0), P=5, L=2, c=0, p=0.9)
    clamped = AddMult(clamp=True)
    assert clamped.value(proofs, probs) == 1.0
    assert all(g == 0.0 for g in clamped.grad(proofs, probs).values())


def test_prop2_topk_undercounts_by_at_most_dropped_mass():
    rng = np.random.default_rng(11)
    for _ in range(20):
        # cap instance size: P*L <= 18 facts keeps us under the WMC limit
        proofs, probs = rand_overlap_instance(rng, P=int(rng.integers(2, 7)),
                                              L=int(rng.integers(2, 4)))
        for k in (1, 3):
            sut = TopK(k)
            err = sut.error(proofs, probs)
            assert err <= 1e-12
            kept = sorted(proofs, key=lambda pr: proof_score(pr, probs), reverse=True)[:k]
            dropped = [pr for pr in proofs if pr not in kept]
            assert -err <= sum(proof_score(pr, probs) for pr in dropped) + 1e-12
            if len(proofs) <= k:
                assert err == pytest.approx(0.0, abs=1e-14)


def test_prop3_lse_bias_law_exact_for_equal_scores():
    for P in (2, 3, 5, 8):
        for tau in (0.005, 0.05, 0.4):
            proofs, probs = rand_overlap_instance(np.random.default_rng(0), P=P, L=2, c=0, p=0.6)
            bias = LSE(tau).error(proofs, probs)  # equal scores by construction
            assert bias == pytest.approx(tau * math.log(P), rel=1e-12)


def test_prop3_lse_bias_bounded_for_unequal_scores():
    rng = np.random.default_rng(3)
    proofs = [frozenset({f"a{j}"}) for j in range(4)]
    probs = {f"a{j}": float(rng.uniform(0.2, 0.9)) for j in range(4)}
    bias = LSE(0.1).error(proofs, probs)
    assert 0.0 < bias < 0.1 * math.log(4)


def test_minmax_subgradient_is_one_hot():
    proofs, probs = rand_overlap_instance(np.random.default_rng(5), P=3, L=3, c=1)
    g = MinMax().grad(proofs, probs)
    assert sorted(g.values(), reverse=True)[:1] == [1.0]
    assert sum(v != 0.0 for v in g.values()) == 1


def test_exact_wmc_is_zero_error_by_definition():
    rng = np.random.default_rng(9)
    proofs, probs = rand_overlap_instance(rng)
    assert ExactWMC().error(proofs, probs) == 0.0
    assert ExactWMC().value(proofs, probs) == wmc(proofs, probs)
