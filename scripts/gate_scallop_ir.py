"""Extended Scallop gates through the rebuilt IR (run in the scallop env):

    ~/miniconda3/envs/scallop-py310/bin/python scripts/gate_scallop_ir.py

A. Value conformance on G1 (50 instances): scallop vs reference SUTs vs oracle
   — the legacy gate, re-validated through program compilation.
B. Recursion: chains L=2..8 and the cyclic Section-5 instance, where Scallop
   runs its own fixpoint — including what addmultprob does on a cyclic
   program (saturation vs divergence is a finding either way).
C. Gradient conformance on heterogeneous G1 (unambiguous tie-free instances):
   central finite differences through Scallop vs the reference SUTs'
   system-faithful gradients.

Writes out/G2b_scallop_ir.json and prints a summary. Exits non-zero if value
conformance (A) breaks 1e-9. Discrepancies are findings: logged, not fixed.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from nesyarena.adapters.scallop import ScallopAdapter
from nesyarena.algebra import BOOLEAN, MAXPROD
from nesyarena.engine import converge
from nesyarena.generators import chain_family, cyclic_family, overlap_family
from nesyarena.oracle import wmc
from nesyarena.suts import AddMult, MinMax, TopK

OUT = os.environ.get("NESYARENA_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
R: dict = {"scallopy_version": "0.2.4-release-wheel"}

PAIRS = [(("addmultprob", None), AddMult(clamp=True)),
         (("minmaxprob", None), MinMax()),
         (("topkproofs", 1), TopK(1)),
         (("topkproofs", 3), TopK(3))]


def adapters():
    return [(ScallopAdapter(p, k), ref) for (p, k), ref in PAIRS]


# ---------------------------------------------------------------- battery A --

def battery_a(n_inst=50, seed=1):
    rng = np.random.default_rng(seed)
    rows = {f"scallop:{p}" + (f"(k={k})" if k else ""): dict(vs_ref=0.0, vs_exact=0.0)
            for (p, k), _ in PAIRS}
    for i in range(n_inst):
        P, L = int(rng.integers(2, 5)), int(rng.integers(2, 4))
        c = int(rng.integers(0, L))
        inst = overlap_family(P, L, c, p=float(rng.choice([0.3, 0.6, 0.9])),
                              rng=rng, het=(i % 3 == 0))
        ex = wmc(inst.proofs, inst.probs)
        for ad, ref in adapters():
            v_s = ad.infer(inst.program, inst.probs, [inst.query])[inst.query]
            v_r = ref.value(inst.proofs, inst.probs)
            rows[ad.name]["vs_ref"] = max(rows[ad.name]["vs_ref"], abs(v_s - v_r))
            rows[ad.name]["vs_exact"] = max(rows[ad.name]["vs_exact"], abs(v_s - ex))
    R["A_g1_values"] = rows
    print("\nA. G1 value conformance (50 instances)")
    print(f"{'adapter':30} {'max|scallop-ref|':>17} {'max|scallop-exact|':>19}")
    for n, r in rows.items():
        print(f"{n:30} {r['vs_ref']:>17.2e} {r['vs_exact']:>19.2e}")
    return max(r["vs_ref"] for r in rows.values())


# ---------------------------------------------------------------- battery B --

def battery_b():
    rec = {}
    # chains: single proof of depth L — every provenance should agree with
    # the convergent fixpoint
    for prov, k, sr in [("topkproofs", 1, MAXPROD), ("minmaxprob", None, BOOLEAN),
                        ("addmultprob", None, MAXPROD)]:
        ad = ScallopAdapter(prov, k)
        worst = 0.0
        for L in (2, 4, 6, 8):
            inst = chain_family(L, 0.9)
            v_s = ad.infer(inst.program, inst.probs, [inst.query])[inst.query]
            v_e = converge(inst.program, dict(inst.probs), sr, inst.query)
            worst = max(worst, abs(v_s - v_e))
        rec[f"chain:{ad.name}"] = worst
    # cyclic Section-5 instance: Scallop's own fixpoint vs references on
    # depth-8 enumerated proofs
    cyc = cyclic_family()
    proofs8 = cyc.program.proof_supports(cyc.query, 8)
    for ad, ref in adapters():
        v_s = ad.infer(cyc.program, cyc.probs, [cyc.query])[cyc.query]
        v_r = ref.value(proofs8, cyc.probs)
        rec[f"cyclic:{ad.name}"] = dict(scallop=v_s, reference_depth8=v_r,
                                        exact_wmc_depth8=wmc(proofs8, cyc.probs))
    R["B_recursion"] = rec
    print("\nB. Recursion")
    for name, v in rec.items():
        print(f"  {name}: {v}")


# ---------------------------------------------------------------- battery C --

def battery_c(n_inst=10, seed=3, h=1e-5, tol=1e-4):
    rng = np.random.default_rng(seed)
    rows = {}
    skipped = 0
    for _ in range(n_inst):
        P, L = int(rng.integers(2, 5)), int(rng.integers(2, 4))
        c = int(rng.integers(0, L))
        inst = overlap_family(P, L, c, 0.6, rng=rng, het=True)  # tie-free a.s.
        raw_sum = sum(float(np.prod([inst.probs[f] for f in pr])) for pr in inst.proofs)
        for ad, ref in adapters():
            if "addmult" in ad.name and abs(raw_sum - 1.0) < 50 * h:
                skipped += 1  # FD would straddle the clamp kink
                continue
            wrt = list(inst.probs)
            g_s = ad.grad(inst.program, inst.probs, inst.query, wrt, h=h)
            g_r = ref.grad(inst.proofs, inst.probs)
            d = max(abs(g_s[a] - g_r.get(a, 0.0)) for a in wrt)
            rows[ad.name] = max(rows.get(ad.name, 0.0), d)
    R["C_grad_fd"] = dict(max_dev=rows, skipped_near_clamp=skipped, h=h, tol=tol)
    print("\nC. Gradient conformance (FD vs reference, heterogeneous G1)")
    for n, d in rows.items():
        flag = "OK" if d < tol else "DISCREPANCY (finding)"
        print(f"  {n:30} max dev {d:.2e}  {flag}")


if __name__ == "__main__":
    worst_a = battery_a()
    battery_b()
    battery_c()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "G2b_scallop_ir.json"), "w") as fh:
        json.dump(R, fh, indent=1, sort_keys=True, default=str)
    print(f"\nwrote {os.path.join(OUT, 'G2b_scallop_ir.json')}")
    sys.exit(0 if worst_a < 1e-9 else 1)
