"""E1 registered-numbers regression: the crossover endpoints quoted in the
paper draft (P=4, L=4, p=0.6) must reproduce through the rebuilt core, and
the |error| ranking must flip between c=1 and c=2 (finding H3)."""

import pytest

from nesyarena.generators import overlap_family
from nesyarena.oracle import wmc
from nesyarena.suts import AddMult, TopK


def _err(sut, c):
    inst = overlap_family(4, 4, c, 0.6)
    return sut.value(inst.proofs, inst.probs) - wmc(inst.proofs, inst.probs)


def test_registered_crossover_endpoints():
    assert _err(AddMult(clamp=True), 0) == pytest.approx(0.092, abs=5e-4)
    assert _err(AddMult(clamp=True), 3) == pytest.approx(0.308, abs=5e-4)
    assert _err(TopK(1), 0) == pytest.approx(-0.296, abs=5e-4)
    assert _err(TopK(1), 3) == pytest.approx(-0.081, abs=5e-4)


def test_h3_ranking_flips_between_c1_and_c2():
    am, t1 = AddMult(clamp=True), TopK(1)
    assert abs(_err(am, 1)) < abs(_err(t1, 1))   # low overlap: add-mult better
    assert abs(_err(am, 2)) > abs(_err(t1, 2))   # high overlap: top-1 better


def test_opposite_monotone_sensitivities():
    am_errs = [_err(AddMult(clamp=True), c) for c in range(4)]
    t1_errs = [_err(TopK(1), c) for c in range(4)]
    assert all(b > a for a, b in zip(am_errs[:-1], am_errs[1:], strict=True))  # worsens
    assert all(b > a for a, b in zip(t1_errs[:-1], t1_errs[1:], strict=True))  # heals
    assert all(e > 0 for e in am_errs) and all(e < 0 for e in t1_errs)
