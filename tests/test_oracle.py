"""Oracle parity (vs toy golden fixtures) and cross-validation (vs ProbLog)."""

import json
import pathlib

import numpy as np
import pytest

from nesyarena.oracle import graph_value, problog_value, wmc, wmc_with_grad

GOLD = json.loads((pathlib.Path(__file__).parent / "fixtures" / "toy_golden.json").read_text())


def _proofs_probs(inst):
    return [frozenset(pr) for pr in inst["proofs"]], inst["probs"]


@pytest.mark.parametrize("i", range(len(GOLD["g1_instances"])))
def test_wmc_value_parity_with_toy(i):
    inst = GOLD["g1_instances"][i]
    proofs, probs = _proofs_probs(inst)
    assert wmc(proofs, probs) == pytest.approx(inst["oracle"]["value"], abs=1e-14)


@pytest.mark.parametrize("i", range(len(GOLD["g1_instances"])))
def test_wmc_grad_parity_with_toy(i):
    inst = GOLD["g1_instances"][i]
    proofs, probs = _proofs_probs(inst)
    val, grad = wmc_with_grad(proofs, probs)
    assert val == pytest.approx(inst["oracle"]["value"], abs=1e-14)
    assert set(grad) == set(inst["oracle"]["grad"])
    for f, g in inst["oracle"]["grad"].items():
        assert grad[f] == pytest.approx(g, abs=1e-12)


def test_analytic_grad_equals_finite_difference():
    inst = GOLD["g1_instances"][5]
    proofs, probs = _proofs_probs(inst)
    _, grad = wmc_with_grad(proofs, probs)
    h = 1e-7
    for f in probs:
        up, dn = dict(probs), dict(probs)
        up[f] += h
        dn[f] -= h
        fd = (wmc(proofs, up) - wmc(proofs, dn)) / (2 * h)
        assert grad[f] == pytest.approx(fd, abs=1e-6)


def test_wmc_against_problog_battery():
    """The protocol's standing gate: reference WMC == ProbLog exact inference."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(10):
        P, L, c = int(rng.integers(2, 5)), int(rng.integers(2, 4)), 0
        c = int(rng.integers(0, L))
        facts_shared = [f"s{i}" for i in range(c)]
        proofs, probs = [], {}
        for f in facts_shared:
            probs[f] = float(rng.choice([0.3, 0.6, 0.9]))
        for j in range(P):
            priv = [f"x{j}_{i}" for i in range(L - c)]
            for f in priv:
                probs[f] = float(rng.choice([0.3, 0.6, 0.9]))
            proofs.append(frozenset(facts_shared + priv))
        worst = max(worst, abs(wmc(proofs, probs) - problog_value(proofs, probs)))
    assert worst < 1e-10


def test_empty_and_certain_proofs():
    assert wmc([], {}) == 0.0
    assert wmc([frozenset({"a"})], {"a": 1.0}) == pytest.approx(1.0)
    val, grad = wmc_with_grad([frozenset({"a"})], {"a": 0.3})
    assert val == pytest.approx(0.3) and grad["a"] == pytest.approx(1.0)


def test_graph_oracle_parity_with_toy():
    edges = [("a", "b"), ("b", "a"), ("b", "c")]
    ew = {("a", "b"): 0.9, ("b", "a"): 0.9, ("b", "c"): 0.8}
    g = GOLD["graph_oracles"]
    assert graph_value("boolean", edges, {e: 1.0 for e in edges}, "a", "c") == g["boolean"]
    assert graph_value("tropical", edges, {e: 1.0 for e in edges}, "a", "c") == g["tropical"]
    assert graph_value("maxprod", edges, ew, "a", "c") == pytest.approx(g["maxprod"], abs=1e-15)
