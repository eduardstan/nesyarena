"""E2 registered-numbers regression (RESULTS.md): horizons are exactly n+1,
the L=8/p=0.9 chain values at and above the horizon, and H7 divergence."""

import pytest

from nesyarena.algebra import MAXPROD, SUMPROD
from nesyarena.engine import converge, infer_bounded
from nesyarena.generators import chain_family, cyclic_family
from nesyarena.metrics import depth_horizon


def test_registered_horizons_n_plus_one():
    for n in (2, 4, 6, 8):
        def err(L, n=n):
            inst = chain_family(L, 0.9)
            return (infer_bounded(inst.program, dict(inst.probs), MAXPROD, inst.query, n)
                    - converge(inst.program, dict(inst.probs), MAXPROD, inst.query))
        assert depth_horizon(err, delta=1e-6, max_depth=12) == n + 1


def test_registered_chain8_values():
    """0.430467 = 0.9^8 at n=8 and 0.478297 = 0.9^7 for the L=7 chain."""
    i8 = chain_family(8, 0.9)
    assert infer_bounded(i8.program, dict(i8.probs), MAXPROD, i8.query, 8) == \
        pytest.approx(0.430467, abs=5e-7)
    i7 = chain_family(7, 0.9)
    assert infer_bounded(i7.program, dict(i7.probs), MAXPROD, i7.query, 8) == \
        pytest.approx(0.478297, abs=5e-7)


def test_registered_h7_divergence_value():
    inst = cyclic_family()
    sp = converge(inst.program, dict(inst.probs), SUMPROD, inst.query,
                  tol=1e-9, max_steps=400)
    assert sp == pytest.approx(0.72 / 0.19, abs=1e-3)  # 3.7895
