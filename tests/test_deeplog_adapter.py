"""DeepLog conformance: its exact-circuit probability semantics vs our WMC
oracle (registered prediction: fidelity 1.0 — exactness by compilation)."""

import numpy as np
import pytest

pytest.importorskip("deeplog")

from nesyarena.adapters.deeplog import DeepLogAdapter  # noqa: E402
from nesyarena.generators import chain_family, overlap_family  # noqa: E402
from nesyarena.oracle import wmc  # noqa: E402


def test_deeplog_burglary_sanity():
    ad = DeepLogAdapter()
    proofs = [frozenset({"b"}), frozenset({"e"})]
    probs = {"b": 0.8, "e": 0.3}
    assert ad.value(proofs, probs) == pytest.approx(1 - 0.2 * 0.7, abs=1e-7)


def test_deeplog_matches_wmc_on_g1_battery():
    rng = np.random.default_rng(2)
    worst = 0.0
    for i in range(8):
        P, L = int(rng.integers(2, 5)), int(rng.integers(2, 4))
        c = int(rng.integers(0, L))
        inst = overlap_family(P, L, c, p=float(rng.choice([0.3, 0.6, 0.9])),
                              rng=rng, het=(i % 2 == 0))
        v = DeepLogAdapter().value(inst.proofs, inst.probs)
        worst = max(worst, abs(v - wmc(inst.proofs, inst.probs)))
    assert worst < 1e-6, f"DeepLog deviates from WMC by {worst} — a finding, log it"


def test_deeplog_infer_on_recursive_program_via_unrolling():
    inst = chain_family(4, 0.9)
    ad = DeepLogAdapter()
    out = ad.infer(inst.program, inst.probs, [inst.query], max_depth=4)
    assert out[inst.query] == pytest.approx(0.9 ** 4, abs=1e-7)


def test_deeplog_autograd_matches_analytic_oracle_gradients():
    """Gradient conformance: symbolic-label inputs expose DeepLog's autograd
    gradients; prediction (exact circuits) is agreement at float32 precision."""
    from nesyarena.oracle import wmc_with_grad

    rng = np.random.default_rng(11)
    ad = DeepLogAdapter()
    worst_v = worst_g = 0.0
    for i in range(4):
        P, L = int(rng.integers(2, 5)), int(rng.integers(2, 4))
        c = int(rng.integers(0, L))
        inst = overlap_family(P, L, c, p=float(rng.choice([0.3, 0.6, 0.9])),
                              rng=rng, het=(i % 2 == 0))
        v, g = ad.value_and_grad(inst.proofs, inst.probs)
        ov, og = wmc_with_grad(inst.proofs, inst.probs)
        worst_v = max(worst_v, abs(v - ov))
        worst_g = max(worst_g, max(abs(g[f] - og[f]) for f in og))
    assert worst_v < 1e-6 and worst_g < 1e-6, (worst_v, worst_g)
