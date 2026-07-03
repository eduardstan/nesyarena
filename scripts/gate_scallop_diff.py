"""Gate G2d — Scallop differentiable provenances (torch tags) on the frozen
instance set. Run in the scallop env:

    ~/miniconda3/envs/scallop-py310/bin/python scripts/gate_scallop_diff.py

Measures the *deployed gradient semantics* of diffaddmultprob, diffminmaxprob
and difftopkproofs(k=1,3): fact probabilities are supplied as torch tensors
with requires_grad, the output tag is a torch tensor, and backward() yields
Scallop's own gradients (no finite differences).

Batteries (benchmarks/instances_v1.json):
  gradients   10 tie-free instances, analytic oracle gradients embedded —
              compare deployed grads vs the reference SUTs' system-faithful
              gradients and vs the oracle.
  saturation  witness-addmult (raw proof-sum 1.2 > 1) + every frozen values
              instance whose raw proof-sum exceeds 1 — the clamp region the
              earlier finite-difference gate deliberately skipped. Reference
              clamp model predicts ZERO gradients there; measure what the
              deployed diff provenance actually returns.

Writes out/G2d_scallop_diff.json and prints a summary. Deviations are
findings: logged with the witnessing instance, never normalized away.
"""

from __future__ import annotations

import json
import os

import torch

from nesyarena.adapters.scallop import compile_rules, fact_key
from nesyarena.benchmarks import load_instances
from nesyarena.suts import AddMult, MinMax, TopK

OUT = os.environ.get("NESYARENA_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))

PAIRS = [("diffaddmultprob", None, AddMult(clamp=True)),
         ("diffminmaxprob", None, MinMax()),
         ("difftopkproofs", 1, TopK(1)),
         ("difftopkproofs", 3, TopK(3))]


def run_diff(provenance, k, inst):
    """Returns (value, {Atom: grad}) from Scallop's torch-tagged execution."""
    import scallopy

    kw = dict(provenance=provenance)
    if k is not None:
        kw["k"] = k
    ctx = scallopy.ScallopContext(**kw)
    ctx.add_relation("fact", (str,))
    facts = sorted(inst.probs, key=repr)
    tags = {f: torch.tensor(float(inst.probs[f]), dtype=torch.float64,
                            requires_grad=True) for f in facts}
    ctx.add_facts("fact", [(tags[f], (fact_key(f),)) for f in facts])
    for rule in compile_rules(inst.program):
        ctx.add_rule(rule)
    ctx.run()
    rows = [r for r in ctx.relation(inst.query.pred)
            if tuple(r[1]) == tuple(str(a) for a in inst.query.args)]
    if not rows:
        return 0.0, {f: 0.0 for f in facts}
    tag = rows[0][0]
    if tag.requires_grad:
        tag.backward()
    return float(tag.detach()), {f: (float(tags[f].grad) if tags[f].grad is not None
                                     else 0.0) for f in facts}


def main():
    grad_insts = load_instances(battery="gradients")
    sat_insts = [i for i in load_instances()
                 if i.battery in ("values", "witnesses")
                 and sum(AddMult(clamp=False).value([pr], i.probs)
                         for pr in i.proofs) >= 1.0]
    R = dict(scallopy_version="0.2.4-release-wheel", torch="cpu",
             gradients={}, saturation=[])

    print("A. gradient battery (10 tie-free instances, deployed autograd)")
    print(f"{'provenance':26} {'max|val dev|':>13} {'max|grad-ref|':>14} {'max|grad-oracle|':>17}")
    for prov, k, ref in PAIRS:
        name = prov + (f"(k={k})" if k else "")
        dv = dg = do = 0.0
        for inst in grad_insts:
            v, g = run_diff(prov, k, inst)
            rv = ref.value(inst.proofs, inst.probs)
            rg = ref.grad(inst.proofs, inst.probs)
            dv = max(dv, abs(v - rv))
            dg = max(dg, max(abs(g[f] - rg.get(f, 0.0)) for f in g))
            do = max(do, max(abs(g[f] - inst.oracle_grad.get(f, 0.0)) for f in g))
        R["gradients"][name] = dict(val_vs_ref=dv, grad_vs_ref=dg, grad_vs_oracle=do)
        print(f"{name:26} {dv:>13.2e} {dg:>14.2e} {do:>17.2e}")

    print(f"\nB. saturation probe — diffaddmultprob on {len(sat_insts)} instances "
          "with raw proof-sum >= 1 (reference clamp model: value 1.0, ALL grads 0)")
    ref = AddMult(clamp=True)
    n_live = 0
    for inst in sat_insts:
        v, g = run_diff("diffaddmultprob", None, inst)
        live = sum(1 for x in g.values() if abs(x) > 1e-12)
        n_live += bool(live)
        R["saturation"].append(dict(id=inst.id, value=v,
                                    ref_value=ref.value(inst.proofs, inst.probs),
                                    live_grads=live, n_facts=len(g),
                                    max_abs_grad=max(abs(x) for x in g.values())))
    worst = max(R["saturation"], key=lambda r: r["max_abs_grad"])
    print(f"   instances with ANY live gradient in the clamp region: "
          f"{n_live}/{len(sat_insts)}")
    print(f"   worst case: {worst['id']} value={worst['value']:.4f} "
          f"(ref {worst['ref_value']:.4f}) live={worst['live_grads']}/{worst['n_facts']} "
          f"max|grad|={worst['max_abs_grad']:.4f}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "G2d_scallop_diff.json"), "w") as fh:
        json.dump(R, fh, indent=1, sort_keys=True)
    print(f"\nwrote {os.path.join(OUT, 'G2d_scallop_diff.json')}")


if __name__ == "__main__":
    main()
