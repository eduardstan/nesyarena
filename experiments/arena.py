"""THE ARENA — one leaderboard, every system, same frozen programs, same oracle.

Run:  .venv/bin/python -m experiments.arena

This is the consolidated view the project is named after. Each row is one
deployed reasoner configuration; every number is computed on the identical
frozen instances (benchmarks/instances_v1.json) against the identical exact
oracle. Two kinds of numbers, kept in separate columns because they answer
different questions:

  SEMANTIC FIDELITY (the science)  — how far is what the system computes from
      what its claimed semantics defines?  phi = 1 - mean|error| on the
      values battery; plus the worst signed error, gradient quality, and the
      cyclic-recursion value (oracle: 0.72).
  IMPLEMENTATION CONFORMANCE (the audit) — how far is the deployed system
      from the best-understood model of its approximation?  Machine precision
      here means "we know exactly what it computes"; it does NOT mean the
      system is semantically faithful. This column is what authorizes
      computing a deployed row via its validated model.

Deployed rows are populated from live measurements where cheap (ProbLog
k-best runs in-process) and from the *validated models* elsewhere (Scallop
rows use the reference SUTs, validated to ~2e-16 on this same battery;
Scallop's diffaddmultprob gradient uses the straight-through model validated
to 4.8e-08 — finding F-2). The provenance of every number is in the
`measured via` column. Writes out/ARENA.md and out/ARENA.json.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

import nesyarena
from nesyarena.benchmarks import load_instances
from nesyarena.metrics import gradient_liveness
from nesyarena.suts import AddMult, ExactWMC, MinMax, TopK

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))


def cosine(g1: dict, g2: dict) -> float:
    keys = sorted(set(g1) | set(g2), key=repr)
    a = np.array([g1.get(k, 0.0) for k in keys])
    b = np.array([g2.get(k, 0.0) for k in keys])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def value_metrics(value_fn, insts):
    errs = [value_fn(i) - i.oracle_value for i in insts]
    worst = max(errs, key=abs)
    return dict(phi=1.0 - float(np.mean(np.abs(errs))),
                mean_abs_err=float(np.mean(np.abs(errs))),
                worst_signed_err=float(worst))


def grad_metrics(grad_fn, insts):
    cos, live = [], []
    for i in insts:
        g = grad_fn(i)
        cos.append(cosine(g, i.oracle_grad))
        live.append(gradient_liveness(g, i.oracle_grad, tol=1e-12))
    return dict(grad_cos=float(np.mean(cos)), grad_liveness=float(np.mean(live)))


def main():
    vals = load_instances(battery="values")
    grads = load_instances(battery="gradients")
    cyc = load_instances(battery="cyclic")[0]

    sc = json.load(open(os.path.join(OUT, "conformance_scallop.json")))
    pk = json.load(open(os.path.join(OUT, "conformance_problog_kbest.json")))

    def cyc_scallop(prov):
        return sc["recursion"][f"cyclic-s5:scallop:{prov}"]["scallop"]

    rows = []

    # ---- anchor ------------------------------------------------------------
    ex = ExactWMC()
    rows.append(dict(
        system="exact WMC (oracle semantics)", version="—",
        claimed="distribution semantics",
        **value_metrics(lambda i: ex.value(i.proofs, i.probs), vals),
        **grad_metrics(lambda i: ex.grad(i.proofs, i.probs), grads),
        cyclic=cyc.oracle_value, conformance=0.0,
        via="definition", findings="—"))

    # ---- Scallop (scallopy 0.2.4) — via validated reference models ----------
    scallop_conf = {"addmultprob": sc["values"]["scallop:addmultprob"]["vs_reference"],
                    "minmaxprob": sc["values"]["scallop:minmaxprob"]["vs_reference"],
                    "topkproofs(k=1)": sc["values"]["scallop:topkproofs(k=1)"]["vs_reference"],
                    "topkproofs(k=3)": sc["values"]["scallop:topkproofs(k=3)"]["vs_reference"]}
    raw = AddMult(clamp=False)  # F-2: deployed diff gradient = unclamped sum
    tk1, tk3, mm = TopK(1), TopK(3), MinMax()
    st_dev = max(r["vs_unclamped_grad"] for r in sc["gradients"]["saturation"])
    for label, sut, gfn, conf, findings, via in [
        ("scallop addmultprob", AddMult(clamp=True),
         lambda i: raw.grad(i.proofs, i.probs),
         max(scallop_conf["addmultprob"], st_dev),
         "F-1 (recursion), F-2 (straight-through clamp)",
         "reference model (values 2.2e-16); F-2 grad model (4.8e-08)"),
        ("scallop topkproofs k=1", tk1,
         lambda i, s=tk1: s.grad(i.proofs, i.probs),
         max(scallop_conf["topkproofs(k=1)"],
             sc["gradients"]["tie_free"]["difftopkproofs(k=1)"]["grad_vs_ref"]),
         "—", "reference model (values 5.6e-17, grads 2.9e-08)"),
        ("scallop topkproofs k=3", tk3,
         lambda i, s=tk3: s.grad(i.proofs, i.probs),
         max(scallop_conf["topkproofs(k=3)"],
             sc["gradients"]["tie_free"]["difftopkproofs(k=3)"]["grad_vs_ref"]),
         "—", "reference model (values 2.2e-16, grads 2.6e-08)"),
        ("scallop minmaxprob", mm,
         lambda i, s=mm: s.grad(i.proofs, i.probs),
         scallop_conf["minmaxprob"],
         "—", "reference model (values 0.0, grads 0.0)"),
    ]:
        rows.append(dict(
            system=label, version="scallopy 0.2.4", claimed="distribution semantics",
            **value_metrics(lambda i, s=sut: s.value(i.proofs, i.probs), vals),
            **grad_metrics(gfn, grads),
            cyclic=cyc_scallop({"scallop addmultprob": "addmultprob",
                                "scallop topkproofs k=1": "topkproofs(k=1)",
                                "scallop topkproofs k=3": "topkproofs(k=3)",
                                "scallop minmaxprob": "minmaxprob"}[label]),
            conformance=conf, via=via, findings=findings))

    # ---- ProbLog k-best (problog 2.2.10) — measured live per instance ------
    by_eps = {}
    for r in pk["rows"]:
        if r["battery"] == "values":
            by_eps.setdefault(r["eps"], {})[r["id"]] = r
    for eps in (0.2, 1e-9):
        recs = by_eps[eps]
        errs = [recs[i.id]["lb"] - i.oracle_value for i in vals]
        worst = max(errs, key=abs)
        rows.append(dict(
            system=f"problog kbest lower bound (eps={eps:g})", version="problog 2.2.10",
            claimed="distribution semantics (anytime bounds)",
            phi=1.0 - float(np.mean(np.abs(errs))),
            mean_abs_err=float(np.mean(np.abs(errs))),
            worst_signed_err=float(worst),
            grad_cos=math.nan, grad_liveness=math.nan,
            cyclic=[r for r in pk["rows"]
                    if r["id"] == "cyclic-s5" and r["eps"] == eps][0]["lb"],
            conformance=0.0, via="measured live (this battery)",
            findings="sound bounds 284/284; lower border implicant-based"))

    # ---- DeepProbLog standalone (deepproblog 2.0.6) — measured live ---------
    dp = json.load(open(os.path.join(OUT, "conformance_deepproblog.json")))
    dp_vals = {r["id"]: r for r in dp["rows"] if r["battery"] == "values"}
    dp_errs = [dp_vals[i.id]["value"] - i.oracle_value for i in vals]
    rows.append(dict(
        system="deepproblog exact engine", version="deepproblog 2.0.6",
        claimed="distribution semantics",
        phi=1.0 - float(np.mean(np.abs(dp_errs))),
        mean_abs_err=float(np.mean(np.abs(dp_errs))),
        worst_signed_err=float(max(dp_errs, key=abs)),
        grad_cos=math.nan, grad_liveness=math.nan,
        cyclic=[r for r in dp["rows"] if r["id"] == "cyclic-s5"][0]["value"],
        conformance=0.0, via="measured live (this battery); see conformance_deepproblog.md",
        findings="—"))

    # ---- LTN (LTNtorch 1.0.2) — the fuzzy axis, measured live ---------------
    lt = json.load(open(os.path.join(OUT, "conformance_ltn.json")))
    lt_errs = [r["prod_err"] for r in lt["rows"]]
    lt_live = lt["verdict"]["liveness_tiefree"]
    rows.append(dict(
        system="ltn product real logic", version="LTNtorch 1.0.2",
        claimed="distribution semantics (independence)",
        phi=1.0 - float(np.mean(np.abs(lt_errs))),
        mean_abs_err=float(np.mean(np.abs(lt_errs))),
        worst_signed_err=float(max(lt_errs, key=abs)),
        grad_cos=math.nan, grad_liveness=lt_live["product"],
        cyclic=math.nan, conformance=0.0,
        via="measured live (this battery); see conformance_ltn.md",
        findings="error = the independence assumption, sign flips with structure"))
    rows.append(dict(
        system="ltn Godel real logic", version="LTNtorch 1.0.2",
        claimed="Godel real logic (min-max evaluation)",
        phi=1.0, mean_abs_err=0.0, worst_signed_err=0.0,
        grad_cos=math.nan, grad_liveness=lt_live["godel"],
        cyclic=math.nan, conformance=max(abs(r["godel_conformance"])
                                         for r in lt["rows"]),
        via="measured live; conformant to its own claim (Godel property)",
        findings="cross-semantics distance to WMC == the min-max row"))

    # ---- DeepLog (pydeeplog 3.0.3) — exact circuits -------------------------
    rows.append(dict(
        system="deeplog exact circuits", version="pydeeplog 3.0.3",
        claimed="distribution semantics",
        phi=1.0, mean_abs_err=0.0, worst_signed_err=0.0,
        grad_cos=1.0, grad_liveness=1.0, cyclic=cyc.oracle_value,
        conformance=1.3e-7,
        via="measured (values <1e-6 f32, grads 1.3e-07); see conformance_deeplog.md",
        findings="—"))

    # ------------------------------------------------------------- output ---
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ARENA.json"), "w") as fh:
        json.dump(dict(package_version=nesyarena.__version__,
                       instance_set="v1", rows=rows), fh, indent=1, sort_keys=True)

    def num(x, fmt="{:.3f}"):
        return "n/a" if isinstance(x, float) and math.isnan(x) else fmt.format(x)

    L = [
        "# THE ARENA — semantic fidelity of deployed NeSy reasoners",
        "",
        "Every number on the identical frozen programs (`benchmarks/instances_v1.json`,",
        "50-instance values battery, 10-instance gradients battery, the cyclic",
        "instance) against the identical exact oracle. Higher phi = closer to the",
        "claimed semantics; **phi = 1 only for exact inference**.",
        "",
        "| system | version | phi | worst signed err | grad cos | grad live"
        " | cyclic (oracle 0.72) | impl. conformance | findings |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["phi"]):
        L.append(f"| {r['system']} | {r['version']} | **{r['phi']:.3f}** "
                 f"| {r['worst_signed_err']:+.3f} | {num(r['grad_cos'])} "
                 f"| {num(r['grad_liveness'])} | {num(r['cyclic'])} "
                 f"| {r['conformance']:.1e} | {r['findings']} |")
    L += [
        "",
        "## How to read this table",
        "",
        "- **phi** (semantic fidelity, the headline) = 1 − mean |computed −",
        "  claimed-semantics value| on the values battery. It scores the",
        "  *approximation*, not the code quality: a perfectly implemented",
        "  approximation still loses phi.",
        "- **worst signed err**: sign matters — positive = over-counts",
        "  (add-mult-style), negative = under-counts (top-k-style). Opposite",
        "  signs are why no method dominates (the crossover, E1).",
        "- **grad cos / grad live**: direction agreement with the analytic oracle",
        "  gradient, and the share of oracle-live facts that receive any gradient",
        "  (tie-free gradients battery). Low liveness = starved learning signal.",
        "- **cyclic**: the value returned on the canonical recursive instance",
        "  (exact answer 0.720). Deviations here are recursion-policy effects",
        "  (finding F-1).",
        "- **impl. conformance** (the audit, NOT the score): distance between the",
        "  deployed system and its best validated model on this same battery.",
        "  ~1e-16 means we know *exactly* what the system computes — which is",
        "  what authorizes the `measured via` shortcuts. A system can be at",
        "  machine-precision conformance and still have low phi: it faithfully",
        "  computes an unfaithful approximation.",
        "- **measured via** (provenance of each row):",
        "",
    ]
    for r in rows:
        L.append(f"  - {r['system']}: {r['via']}")
    L += [
        "",
        "Reference (idealized) SUT rows are omitted: after validation they are",
        "numerically identical to the deployed rows they model. Per-framework",
        "detail and findings: `conformance_scallop.md` (F-1, F-2),",
        "`conformance_deeplog.md`, `conformance_problog_kbest.md`.",
        "Learning-consequence results (calibration, transfer): `RESULTS.md`.",
    ]
    with open(os.path.join(OUT, "ARENA.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:20]))
    print(f"\nwrote {os.path.join(OUT, 'ARENA.md')} and ARENA.json")


if __name__ == "__main__":
    main()
