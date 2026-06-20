"""Witness-synthesis parity with the toy + unit semantics of D2-D4 metrics."""

import pytest

from nesyarena.algebra import MAXPROD
from nesyarena.engine import converge, infer_bounded
from nesyarena.generators import chain_family, overlap_family
from nesyarena.metrics import depth_horizon, fidelity, gradient_liveness
from nesyarena.oracle import wmc_with_grad
from nesyarena.suts import AddMult, ExactWMC, MinMax, TopK
from nesyarena.witness import find_witness

SUTS = {s.name: s for s in [AddMult(clamp=True), TopK(1), TopK(3), MinMax()]}


def test_witness_parity_with_toy(gold):
    for name, fx in gold["witnesses"].items():
        w = find_witness(SUTS[name])
        assert w is not None
        assert (w["P"], w["L"], w["c"], w["p"]) == (fx["P"], fx["L"], fx["c"], fx["p"]), name
        assert w["err"] == pytest.approx(fx["err"], abs=1e-12), name


def test_exact_wmc_has_no_witness():
    assert find_witness(ExactWMC()) is None


def test_fidelity_profile_ordering():
    sweep = [overlap_family(P, 3, 1, 0.6) for P in (2, 3, 4)]
    f_exact = fidelity(ExactWMC(), sweep)
    f_add = fidelity(AddMult(clamp=True), sweep)
    assert f_exact == 1.0
    assert 0.0 <= f_add < 1.0


def test_depth_horizon_is_n_plus_one():
    n = 4

    def err(depth_required: int) -> float:
        inst = chain_family(depth_required, 0.9)
        truncated = infer_bounded(inst.program, dict(inst.probs), MAXPROD, inst.query, n)
        exact = converge(inst.program, dict(inst.probs), MAXPROD, inst.query)
        return truncated - exact

    assert depth_horizon(err, delta=1e-6, max_depth=8) == n + 1


def test_gradient_liveness_semantics():
    inst = overlap_family(3, 2, 1, 0.6)
    _, og = wmc_with_grad(inst.proofs, inst.probs)
    assert gradient_liveness(ExactWMC().grad(inst.proofs, inst.probs), og) == 1.0
    # min-max keeps exactly one fact alive out of m oracle-live facts
    m = len(inst.probs)
    assert gradient_liveness(MinMax().grad(inst.proofs, inst.probs), og) == pytest.approx(1 / m)
    # clamp blackout: saturated add-mult starves every fact
    sat = overlap_family(5, 2, 0, 0.9)
    _, og_sat = wmc_with_grad(sat.proofs, sat.probs)
    assert gradient_liveness(AddMult(clamp=True).grad(sat.proofs, sat.probs), og_sat) == 0.0
