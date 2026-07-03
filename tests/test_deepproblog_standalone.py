"""DeepProbLog standalone (exact engine) conformance — fast CI subset
(witnesses + cyclic; the full 71-instance run is
experiments/conformance_deepproblog.py)."""

import os

import pytest

pytest.importorskip("deepproblog")

from nesyarena.adapters.deepproblog_standalone import (  # noqa: E402
    DeepProbLogStandaloneAdapter,
)
from nesyarena.benchmarks import DEFAULT_PATH, load_instances  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_PATH),
                                reason="frozen instances not generated")


def test_exact_engine_matches_oracle_on_witnesses_and_cyclic():
    ad = DeepProbLogStandaloneAdapter()
    insts = load_instances(battery="witnesses") + load_instances(battery="cyclic")
    for inst in insts:
        v = ad.infer(inst.program, inst.probs, [inst.query])[inst.query]
        assert v == pytest.approx(inst.oracle_value, abs=1e-7), inst.id
