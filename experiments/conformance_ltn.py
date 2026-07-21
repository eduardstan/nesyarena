"""Conformance of LTN (LTNtorch) on the frozen instance set — the fuzzy axis.

Run:  .venv/bin/python -m experiments.conformance_ltn

Two deployed configurations (adapters/ltn.py, LTNtorch 1.0.2, deployed
defaults incl. stable=True):

  ltn:product  claims distribution semantics via the independence assumption.
  ltn:godel    claims Gödel real logic; its exact evaluation on a monotone
               DNF is the min-max algebra (a provable property), so its
               conformance error is ~0 by construction and the informative
               number is the cross-semantics distance to WMC.

REGISTERED PREDICTIONS (before this run):
  P1: ltn:product is exact up to the deployed stabilization (~1e-3) on
      fact-disjoint instances (c = 0: independence holds), and over-counts
      (signed error > 0) on every shared-trunk instance (c >= 1) — the
      independence assumption's overlap face.
  P2: ltn:godel conformance-to-claim ~0 on every instance; its
      cross-semantics distance equals the min-max reference error.
  P3: gradients on the tie-free battery: ltn:godel coincides with the
      min-max one-hot subgradient (ties are the only divergence mechanism);
      liveness for LTN is quoted from this battery only.

Writes out/conformance_ltn.{json,md}.
"""

from __future__ import annotations

import json
import os

import numpy as np

import nesyarena
from nesyarena.adapters.ltn import LTNGodel, LTNProduct
from nesyarena.benchmarks import load_instances
from nesyarena.metrics import gradient_liveness
from nesyarena.suts import MinMax

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))
STAB_TOL = 2e-3  # deployed stable=True shifts values by ~1e-4 per connective


def main():
    prod, godel, mm = LTNProduct(), LTNGodel(), MinMax()
    vals = load_instances(battery="values")
    grads = load_instances(battery="gradients")

    rows = []
    p1_disjoint_ok = p1_overlap_ok = p2_ok = 0
    n_disjoint = n_overlap = 0
    for inst in vals:
        e_prod = prod.value(inst.proofs, inst.probs) - inst.oracle_value
        e_godel_claim = godel.error(inst.proofs, inst.probs)
        e_godel_cross = godel.cross_semantics_error(inst.proofs, inst.probs)
        c = inst.params["c"]
        if c == 0:
            n_disjoint += 1
            p1_disjoint_ok += abs(e_prod) < STAB_TOL
        else:
            n_overlap += 1
            p1_overlap_ok += e_prod > 0
        p2_ok += (abs(e_godel_claim) < 1e-5
                  and abs(e_godel_cross - mm.error(inst.proofs, inst.probs)) < 1e-5)
        rows.append(dict(id=inst.id, c=c, prod_err=e_prod,
                         godel_conformance=e_godel_claim,
                         godel_cross_semantics=e_godel_cross))

    prod_errs = [r["prod_err"] for r in rows]
    phi_prod = 1.0 - float(np.mean(np.abs(prod_errs)))

    live_prod, live_godel, godel_onehot_ok = [], [], 0
    for inst in grads:  # tie-free by construction
        gp = prod.grad(inst.proofs, inst.probs)
        gg = godel.grad(inst.proofs, inst.probs)
        live_prod.append(gradient_liveness(gp, inst.oracle_grad, tol=1e-9))
        live_godel.append(gradient_liveness(gg, inst.oracle_grad, tol=1e-9))
        rg = mm.grad(inst.proofs, inst.probs)
        godel_onehot_ok += all(abs(gg[f] - rg.get(f, 0.0)) < 1e-5 for f in gg)

    verdict = dict(
        P1_disjoint_exact=f"{p1_disjoint_ok}/{n_disjoint}",
        P1_overlap_positive=f"{p1_overlap_ok}/{n_overlap}",
        P2_godel_conformance=f"{p2_ok}/{len(vals)}",
        P3_godel_onehot_tiefree=f"{godel_onehot_ok}/{len(grads)}",
        phi_product_vs_wmc=phi_prod,
        worst_prod_err=float(max(prod_errs, key=abs)),
        liveness_tiefree=dict(product=float(np.mean(live_prod)),
                              godel=float(np.mean(live_godel))))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "conformance_ltn.json"), "w") as fh:
        json.dump(dict(experiment="conformance-ltn",
                       package_version=nesyarena.__version__,
                       ltntorch_version="1.0.2", verdict=verdict, rows=rows),
                  fh, indent=1, sort_keys=True)

    md = [
        "# Conformance — LTN (LTNtorch 1.0.2, deployed defaults incl. stable=True)",
        "",
        "Frozen instance set v1; registered predictions and verdicts:",
        "",
        f"- **P1a** ltn:product exact (up to ~1e-3 stabilization) on fact-disjoint"
        f" instances (c=0): **{p1_disjoint_ok}/{n_disjoint}**",
        f"- **P1b** ltn:product over-counts (signed err > 0) on every shared-trunk"
        f" instance (c>=1): **{p1_overlap_ok}/{n_overlap}**",
        f"- **P2** ltn:godel conformance-to-claim ~0 AND cross-semantics distance"
        f" == min-max reference: **{p2_ok}/{len(vals)}**",
        f"- **P3** ltn:godel gradient == min-max one-hot on the tie-free battery:"
        f" **{godel_onehot_ok}/{len(grads)}**",
        "",
        f"phi(ltn:product vs distribution semantics) = **{phi_prod:.4f}**;"
        f" worst signed error {max(prod_errs, key=abs):+.3f}.",
        f"Tie-free gradient liveness: product {np.mean(live_prod):.3f},"
        f" godel {np.mean(live_godel):.3f} (the homogeneous-battery 1.0 reported"
        " in earlier runs is a tie-splitting artifact of torch min/max and is"
        " not quoted).",
        "",
        "Reading: ltn:product's error is the independence assumption made",
        "measurable — near-zero where facts are disjoint, positive growth with",
        "shared structure (and negative on mutually exclusive proofs, per the",
        "E6 disjoint control). ltn:godel is conformant to its own (Gödel)",
        "claim by the min-max property; its distance to distribution semantics",
        "is the min-max row of the arena.",
    ]
    with open(os.path.join(OUT, "conformance_ltn.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[2:14]))


if __name__ == "__main__":
    main()
