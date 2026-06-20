"""Engine parity with toy histories + the Theorem-1 numeric correspondence
(operator iterates == depth-bounded proof aggregation) now derived end-to-end
from the IR, and Prop. 4 truncation semantics."""

import pytest

from nesyarena.algebra import BOOLEAN, MAXPROD, SUMPROD
from nesyarena.engine import converge, infer_bounded, iterate, proof_aggregate
from nesyarena.generators import chain_family, cyclic_family


def _cyclic_f0(boolean=False):
    inst = cyclic_family()
    return inst, ({a: 1.0 for a in inst.probs} if boolean else dict(inst.probs))


def test_cyclic_history_parity_with_toy(gold):
    inst, f0p = _cyclic_f0()
    q = inst.query
    for name, sr, f0 in [("boolean", BOOLEAN, {a: 1.0 for a in inst.probs}),
                         ("maxprod", MAXPROD, f0p),
                         ("sumprod", SUMPROD, f0p)]:
        hist = iterate(inst.program, f0, sr, 20)
        got = [h.get(q, sr.zero) for h in hist]
        assert got == pytest.approx(gold["cyclic_path_ac"][name], abs=1e-14), name


def test_chain8_truncation_parity_with_toy(gold):
    inst = chain_family(8, 0.9)
    hist = iterate(inst.program, dict(inst.probs), MAXPROD, 10)
    got = [h.get(inst.query, 0.0) for h in hist]
    assert got == pytest.approx(gold["chain8_maxprod_path_v0_v8"], abs=1e-14)


@pytest.mark.parametrize("n", range(0, 11))
def test_theorem1_iterates_equal_proof_aggregation_nonidempotent(n):
    """The KR Theorem-1 check, strengthened: proofs are *enumerated from the
    program* (not hand-built) and aggregated under sum-product on the cyclic
    program, then compared to operator iterates."""
    inst, f0 = _cyclic_f0()
    lhs = infer_bounded(inst.program, f0, SUMPROD, inst.query, n)
    rhs = proof_aggregate(inst.program, f0, SUMPROD, inst.query, n)
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_theorem1_on_chain_maxprod():
    inst = chain_family(5, 0.9)
    for n in range(0, 8):
        lhs = infer_bounded(inst.program, dict(inst.probs), MAXPROD, inst.query, n)
        rhs = proof_aggregate(inst.program, dict(inst.probs), MAXPROD, inst.query, n)
        assert lhs == pytest.approx(rhs, abs=1e-14)


def test_prop4_truncation_zero_below_horizon_exact_at_horizon():
    L = 8
    inst = chain_family(L, 0.9)
    f0 = dict(inst.probs)
    for n in range(L):
        assert infer_bounded(inst.program, f0, MAXPROD, inst.query, n) == 0.0
    assert infer_bounded(inst.program, f0, MAXPROD, inst.query, L) == pytest.approx(0.9 ** L)
    # zero below the horizon is *flat* zero: perturbing any edge keeps it zero,
    # so the (finite-difference) gradient is identically zero (Prop. 4)
    for a in inst.probs:
        bumped = dict(f0)
        bumped[a] = min(1.0, bumped[a] + 1e-3)
        assert infer_bounded(inst.program, bumped, MAXPROD, inst.query, L - 1) == 0.0


def test_converge_on_idempotent_algebras():
    inst = cyclic_family()
    f0 = dict(inst.probs)
    assert converge(inst.program, f0, MAXPROD, inst.query) == pytest.approx(0.72)
    assert converge(inst.program, {a: 1.0 for a in f0}, BOOLEAN, inst.query) == 1.0


def test_sumprod_on_cyclic_diverges_past_one():
    """Remark 12 / H7: 'probability' under proof-sum on a cyclic program
    exceeds 1 (limit 0.72/0.19 = 3.789...) — the engine exposes it."""
    inst = cyclic_family()
    val = converge(inst.program, dict(inst.probs), SUMPROD, inst.query,
                   tol=1e-9, max_steps=400)
    assert val > 1.0
    assert val == pytest.approx(0.72 / 0.19, abs=1e-3)
