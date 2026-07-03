"""Gate: autograd through the torch provenance layer must reproduce the
reference SUTs' values AND system-faithful gradients (heterogeneous probs:
tie-free, so subgradient choices are unambiguous)."""

import numpy as np
import pytest
import torch

from nesyarena.generators import overlap_family
from nesyarena.learning import BatchStructure, prov_value
from nesyarena.suts import AddMult, ExactWMC, MinMax, TopK

PAIRS = [("exact", None, ExactWMC()), ("addmult", None, AddMult(clamp=True)),
         ("top1", 1, TopK(1)), ("top3", 3, TopK(3)), ("minmax", None, MinMax())]


def _torch_value_grad(name, S, probs_vec, k):
    p = torch.tensor(probs_vec[None, :], dtype=torch.float64, requires_grad=True)
    v = prov_value(name, S, p, k)
    v.sum().backward()
    return float(v.detach()[0]), p.grad[0].numpy()


def test_value_and_grad_parity_with_reference_suts():
    rng = np.random.default_rng(21)
    for _ in range(8):
        P, L = int(rng.integers(2, 5)), int(rng.integers(2, 4))
        c = int(rng.integers(0, L))
        inst = overlap_family(P, L, c, 0.6, rng=rng, het=True)  # tie-free
        facts = sorted(inst.probs, key=repr)
        S = BatchStructure(inst.proofs, facts)
        pv = np.array([inst.probs[f] for f in facts])
        for name, k, ref in PAIRS:
            v, g = _torch_value_grad(name, S, pv, k)
            assert v == pytest.approx(ref.value(inst.proofs, inst.probs), abs=1e-12), name
            rg = ref.grad(inst.proofs, inst.probs)
            for j, f in enumerate(facts):
                assert g[j] == pytest.approx(rg.get(f, 0.0), abs=1e-10), (name, f)


def test_clamp_blackout_in_autograd():
    inst = overlap_family(5, 2, 0, 0.9)
    facts = sorted(inst.probs, key=repr)
    S = BatchStructure(inst.proofs, facts)
    p = torch.full((1, len(facts)), 0.9, dtype=torch.float64, requires_grad=True)
    v = prov_value("addmult", S, p)
    v.sum().backward()
    assert float(v[0]) == 1.0
    assert torch.all(p.grad == 0.0)


def test_batched_equals_per_sample():
    inst = overlap_family(3, 3, 1, 0.6)
    facts = sorted(inst.probs, key=repr)
    S = BatchStructure(inst.proofs, facts)
    rng = np.random.default_rng(5)
    Pb = torch.tensor(rng.uniform(0.1, 0.9, size=(7, len(facts))))
    for name, k, _ in PAIRS:
        vb = prov_value(name, S, Pb, k)
        for b in range(7):
            v1 = prov_value(name, S, Pb[b:b + 1], k)
            assert float(vb[b]) == pytest.approx(float(v1[0]), abs=1e-12)


def test_truth_matches_wmc_at_extremes():
    inst = overlap_family(4, 3, 2, 0.6)
    facts = sorted(inst.probs, key=repr)
    S = BatchStructure(inst.proofs, facts)
    rng = np.random.default_rng(3)
    Z = torch.tensor(rng.random((40, len(facts))) < 0.5)
    v = S.wmc(Z.to(torch.float64))
    assert torch.allclose(v, S.truth(Z).to(torch.float64))


def test_straight_through_clamp_matches_f3_semantics():
    """F-3 model: clamped value, unclamped gradient — including at saturation,
    where the min-clamp op has zero gradient."""
    from nesyarena.suts import AddMultStraightThrough

    inst = overlap_family(5, 2, 0, 0.9)  # raw sum >> 1 (clamp region)
    facts = sorted(inst.probs, key=repr)
    S = BatchStructure(inst.proofs, facts)
    p = torch.full((1, len(facts)), 0.9, dtype=torch.float64, requires_grad=True)
    v = prov_value("addmult_st", S, p)
    v.sum().backward()
    assert float(v.detach()[0]) == 1.0                      # clamped value
    raw_ref = AddMult(clamp=False)
    rg = raw_ref.grad(inst.proofs, inst.probs)
    for j, f in enumerate(facts):
        assert float(p.grad[0, j]) == pytest.approx(rg[f], abs=1e-10)
    st = AddMultStraightThrough()
    assert st.value(inst.proofs, inst.probs) == 1.0
    assert st.grad(inst.proofs, inst.probs) == rg
