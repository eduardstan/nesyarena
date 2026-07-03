"""ProbLog anytime k-best conformance (fast CI subset; the full 71x4 run is
experiments/conformance_problog_kbest.py). Soundness + tight-eps exactness +
the coarse-eps lower border realizing the best-explanation value."""

import pytest

pytest.importorskip("problog")

import os  # noqa: E402

from nesyarena.adapters.problog_kbest import ProbLogKBestAdapter  # noqa: E402
from nesyarena.benchmarks import DEFAULT_PATH, load_instances  # noqa: E402
from nesyarena.suts import TopK  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_PATH),
                                reason="frozen instances not generated")


@pytest.fixture(scope="module")
def witnesses():
    return load_instances(battery="witnesses")


def test_soundness_and_tight_eps_exactness(witnesses):
    tight = ProbLogKBestAdapter(convergence=1e-9)
    for inst in witnesses:
        lb, ub = tight.infer_bounds(inst.program, inst.probs, [inst.query])[inst.query]
        assert lb - 1e-7 <= inst.oracle_value <= ub + 1e-7, inst.id
        assert lb == pytest.approx(inst.oracle_value, abs=1e-7), inst.id
        assert ub == pytest.approx(inst.oracle_value, abs=1e-7), inst.id


def test_coarse_eps_lower_border_is_best_explanation(witnesses):
    coarse = ProbLogKBestAdapter(convergence=0.5)
    for inst in witnesses:
        lb, ub = coarse.infer_bounds(inst.program, inst.probs, [inst.query])[inst.query]
        assert lb - 1e-7 <= inst.oracle_value <= ub + 1e-7, inst.id
        # after one border update the lower bound is the best single explanation
        if len(inst.proofs) > 1:
            assert lb == pytest.approx(TopK(1).value(inst.proofs, inst.probs),
                                       abs=1e-7), inst.id
