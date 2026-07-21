"""Conformance of Scallop's deployed provenances, on the frozen instance set.

Scallop's scallopy wheel is Python-3.10-only, so this runner executes in the
dedicated env (see INSTALL_SCALLOP.md), unlike the other experiments:

    ~/miniconda3/envs/scallop-py310/bin/python -m experiments.conformance_scallop
    ... --battery values|recursion|gradients|all      (default: all)

Batteries (benchmarks/instances_v1.json):
  values      addmultprob / minmaxprob / topkproofs(k=1,3) vs the reference
              SUTs and the exact oracle, via full program compilation.
  recursion   chains (native Scallop fixpoint vs convergent engine), the
              cyclic instance, and the diamond probes — the battery that
              produced finding F-1 (tuple-set-fixpoint truncation).
  gradients   diff* provenances with torch-tagged facts: Scallop's own
              autograd vs the reference SUTs' system-faithful gradients —
              the battery that produced finding F-2 (straight-through clamp),
              including the saturation sweep. Needs torch in the scallop env.

Findings log (the document of record): out/conformance_scallop.md.
Raw numbers: out/conformance_scallop.json. Deviations are findings — logged
with the witnessing instance, never normalized away.
"""

from __future__ import annotations

import argparse
import json
import os

from nesyarena.adapters.scallop import ScallopAdapter, compile_rules, fact_key
from nesyarena.algebra import BOOLEAN, MAXPROD
from nesyarena.benchmarks import load_instances
from nesyarena.engine import converge
from nesyarena.suts import AddMult, MinMax, TopK

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))

PAIRS = [("addmultprob", None, AddMult(clamp=True)),
         ("minmaxprob", None, MinMax()),
         ("topkproofs", 1, TopK(1)),
         ("topkproofs", 3, TopK(3))]


def battery_values():
    insts = load_instances(battery="values")
    rows = {}
    for prov, k, ref in PAIRS:
        ad = ScallopAdapter(prov, k)
        dv_ref = dv_or = 0.0
        for inst in insts:
            v = ad.infer(inst.program, inst.probs, [inst.query])[inst.query]
            dv_ref = max(dv_ref, abs(v - ref.value(inst.proofs, inst.probs)))
            dv_or = max(dv_or, abs(v - inst.oracle_value))
        rows[ad.name] = dict(vs_reference=dv_ref, vs_oracle=dv_or)
    print(f"A. values ({len(insts)} frozen instances)")
    print(f"{'provenance':28} {'max|scallop-ref|':>17} {'max|scallop-oracle|':>20}")
    for n, r in rows.items():
        print(f"{n:28} {r['vs_reference']:>17.2e} {r['vs_oracle']:>20.2e}")
    return rows


def battery_recursion():
    rec = {}
    for prov, k, sr in [("topkproofs", 1, MAXPROD), ("minmaxprob", None, BOOLEAN),
                        ("addmultprob", None, MAXPROD)]:
        ad = ScallopAdapter(prov, k)
        worst = 0.0
        for inst in load_instances(battery="chains"):
            v = ad.infer(inst.program, inst.probs, [inst.query])[inst.query]
            e = converge(inst.program, dict(inst.probs), sr, inst.query)
            worst = max(worst, abs(v - e))
        rec[f"chains:{ad.name}"] = worst
    for inst in load_instances(battery="cyclic") + load_instances(battery="probes"):
        for ad, ref in [(ScallopAdapter(p, k), r) for p, k, r in PAIRS]:
            v = ad.infer(inst.program, inst.probs, [inst.query])[inst.query]
            rec[f"{inst.id}:{ad.name}"] = dict(
                scallop=v, reference=ref.value(inst.proofs, inst.probs),
                oracle=inst.oracle_value)
    print("\nB. recursion (chains + cyclic + probes)")
    for name, v in rec.items():
        print(f"  {name}: {v}")
    return rec


def run_diff(provenance, k, inst):
    import scallopy
    import torch

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


def battery_gradients():
    grad_insts = load_instances(battery="gradients")
    sat_insts = [i for i in load_instances()
                 if i.battery in ("values", "witnesses")
                 and AddMult(clamp=False).value(i.proofs, i.probs) >= 1.0]
    R = dict(tie_free={}, saturation=[])
    print(f"\nC. gradients ({len(grad_insts)} tie-free instances, deployed autograd)")
    print(f"{'provenance':26} {'max|val dev|':>13} {'max|grad-ref|':>14} {'max|grad-oracle|':>17}")
    for prov, k, ref in PAIRS:
        name = "diff" + prov + (f"(k={k})" if k else "")
        dv = dg = do = 0.0
        for inst in grad_insts:
            v, g = run_diff("diff" + prov, k, inst)
            dv = max(dv, abs(v - ref.value(inst.proofs, inst.probs)))
            rg = ref.grad(inst.proofs, inst.probs)
            dg = max(dg, max(abs(g[f] - rg.get(f, 0.0)) for f in g))
            do = max(do, max(abs(g[f] - inst.oracle_grad.get(f, 0.0)) for f in g))
        R["tie_free"][name] = dict(val_vs_ref=dv, grad_vs_ref=dg, grad_vs_oracle=do)
        print(f"{name:26} {dv:>13.2e} {dg:>14.2e} {do:>17.2e}")
    print(f"\n   saturation sweep — diffaddmultprob on {len(sat_insts)} instances with "
          "raw proof-sum >= 1 (min-clamp model: all grads 0)")
    raw = AddMult(clamp=False)
    n_live = 0
    for inst in sat_insts:
        v, g = run_diff("diffaddmultprob", None, inst)
        rg = raw.grad(inst.proofs, inst.probs)  # gradient of the UNCLAMPED sum
        straight_through_dev = max(abs(g[f] - rg.get(f, 0.0)) for f in g)
        live = sum(1 for x in g.values() if abs(x) > 1e-12)
        n_live += bool(live)
        R["saturation"].append(dict(id=inst.id, value=v, live_grads=live,
                                    n_facts=len(g),
                                    vs_unclamped_grad=straight_through_dev))
    st_worst = max(r["vs_unclamped_grad"] for r in R["saturation"])
    print(f"   live-gradient instances: {n_live}/{len(sat_insts)}; "
          f"max |deployed grad − unclamped-sum grad| = {st_worst:.2e} "
          "(straight-through clamp, finding F-2)")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", default="all",
                    choices=["values", "recursion", "gradients", "all"])
    which = ap.parse_args().battery
    R = dict(scallopy_version="0.2.4-release-wheel", instance_set="v1")
    if which in ("values", "all"):
        R["values"] = battery_values()
    if which in ("recursion", "all"):
        R["recursion"] = battery_recursion()
    if which in ("gradients", "all"):
        R["gradients"] = battery_gradients()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "conformance_scallop.json")
    with open(path, "w") as fh:
        json.dump(R, fh, indent=1, sort_keys=True, default=str)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
