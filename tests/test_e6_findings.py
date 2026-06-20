"""E6 fact-table findings replication at reduced budget (fast CI variant of
the full experiment, which reproduced the registered table exactly —
out/E6_facttable.json). Asserts the *findings*, not the full-budget numbers:

  (i)   exact transfers near-zero on the overlap treatment;
  (ii)  add-mult is harmed on treatment (an order of magnitude over exact);
  (iii) on the disjoint control, exact and add-mult are bit-identical
        (shared code path — H9 made structural);
  (iv)  top-1 is harmed on the control (truncation is what sum-structured
        tasks CAN see).
"""

import copy
import os

import pytest
import yaml

from experiments.e6_facttable import run_control, run_treatment

CFG_PATH = os.path.join(os.path.dirname(__file__), "..",
                        "experiments", "configs", "E6_facttable.yaml")


@pytest.fixture(scope="module")
def results():
    cfg = yaml.safe_load(open(CFG_PATH, "rb"))
    cfg = copy.deepcopy(cfg)
    cfg["seeds"] = 2
    cfg["optimizer"]["steps"] = 600
    return run_treatment(cfg), run_control(cfg)


def test_exact_transfers_near_zero_on_treatment(results):
    treat, _ = results
    assert max(treat["exact"]["trans"]) < 0.02


def test_addmult_harmed_on_treatment(results):
    treat, _ = results
    import numpy as np
    assert np.mean(treat["addmult"]["trans"]) > 5 * max(np.mean(treat["exact"]["trans"]), 0.01)


def test_exact_equals_addmult_exactly_on_control(results):
    _, ctrl = results
    assert ctrl["exact"]["trans"] == ctrl["addmult"]["trans"]


def test_top1_harmed_on_control(results):
    _, ctrl = results
    import numpy as np
    assert np.mean(ctrl["top1"]["trans"]) > 3 * np.mean(ctrl["exact"]["trans"])
