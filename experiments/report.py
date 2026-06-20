"""Collate all experiment outputs into out/RESULTS.md (auto-generated).

Run:  .venv/bin/python -m experiments.report
Reads whichever of E1/E2/E3/E4/E6/E7/E5-mnist/scorecard JSONs exist in out/.
"""

from __future__ import annotations

import json
import os

import nesyarena

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(ms):
    return f"{ms[0]:.3f} ± {ms[1]:.3f}"


def main():
    L = [f"# NeSyArena — measured results (auto-generated, package {nesyarena.__version__})",
         "",
         "All numbers produced by the rebuilt core (`src/nesyarena/`) via the",
         "config-driven runners in `experiments/`; configs and their sha256 are",
         "embedded in each JSON. Conformance/finding logs: `G2_scallop.md`,",
         "`G2b_scallop_ir.md` (finding F-1), `conformance_deeplog.md`.", ""]

    if (e1 := load("E1_results.json")):
        L += ["## E1 — overlap sweep (fidelity over the G1 grid)", "",
              "| SUT | phi(overlap) | mean abs err |", "|---|---|---|"]
        for n, d in sorted(e1["scorecard"].items()):
            L.append(f"| {n} | {d['phi_overlap']:.4f} | {d['mean_abs_err']:.4f} |")
        L.append("")

    if (e2 := load("E2_results.json")):
        L += ["## E2 — depth horizons (theory: n + 1)", "",
              f"measured: {e2['horizons']}",
              f"H7 cyclic sum-product: {e2['cyclic_h7']['sumprod']:.4f} (> 1: "
              f"{e2['cyclic_h7']['leaves_unit_interval']}); "
              f"max-product: {e2['cyclic_h7']['maxprod']:.4f}", ""]

    if (e3 := load("E3_results.json")):
        L += ["## E3 — surrogate bias law",
              "", f"max |bias − tau·ln P| = {e3['bias_law_max_abs_dev']:.2e}", ""]

    if (e4 := load("E4_witness_table.json")):
        L += ["## E4 — minimal witnesses (machine-found, shrunk)", "",
              "| SUT | P | L | c | p | m | signed error |", "|---|---|---|---|---|---|---|"]
        for n, w in sorted(e4["witnesses"].items()):
            L.append(f"| {n} | — | — | — | — | — | none |" if w is None else
                     f"| {n} | {w['P']} | {w['L']} | {w['c']} | {w['p']} | {w['m']} "
                     f"| {w['err']:+.3f} |")
        L.append("")

    if (sc := load("scorecard.json")):
        L += ["## Gradient liveness (share of oracle-live facts kept alive)", ""]
        L += [f"- {n}: **{v:.3f}**" for n, v in sorted(sc["gradient_liveness"].items())]
        L.append("")

    if (e6f := load("E6_facttable.json")):
        L += ["## E6 fact-table (population-loss limit, 5 seeds): held-out error", "",
              "| SUT | control (disjoint) | treatment (overlap) | facts-quality | fact MAE |",
              "|---|---|---|---|---|"]
        for n in e6f["treatment"]:
            t, c = e6f["treatment"][n], e6f["control"][n]
            L.append(f"| {n} | {fmt(c['trans'])} | {fmt(t['trans'])} "
                     f"| {t['facts_q'][0]:.3f} | {t['fact_mae'][0]:.3f} |")
        L.append("")

    if (e6p := load("E6_pixels.json")):
        L += ["## E6 pixels (H8 headline, 3 seeds): accuracy ties, fidelity diverges", "",
              "| SUT | accuracy | calibration vs Bayes | transfer | facts-quality |",
              "|---|---|---|---|---|"]
        for n in e6p["treatment"]:
            d = e6p["treatment"][n]
            L.append(f"| {n} | {fmt(d['acc'])} | {fmt(d['cal'])} | {fmt(d['trans'])} "
                     f"| {fmt(d['fq'])} |")
        L += ["", "Control (sum structure; exact == addmult by construction):", "",
              "| SUT | accuracy | calibration | transfer |", "|---|---|---|---|"]
        for n in e6p["control"]:
            d = e6p["control"][n]
            L.append(f"| {n} | {fmt(d['acc'])} | {fmt(d['cal'])} | {fmt(d['trans'])} |")
        L.append("")

    if (e7 := load("E7_results.json")):
        L += ["## E7 — depth learning (chain depth 6)", ""]
        L += [f"- {m}: AUC **{fmt(v)}**" for m, v in e7["final_auc"].items()]
        L.append("")

    if (e5 := load("E5_mnist.json")):
        L += ["## E5/E6 at MNIST scale (real digits, 3 seeds)", "",
              "MNIST-path treatment (overlap):", "",
              "| SUT | accuracy | perception MAE | transfer | facts-quality |",
              "|---|---|---|---|---|"]
        for n in e5["treatment"]:
            d = e5["treatment"][n]
            L.append(f"| {n} | {fmt(d['acc'])} | {fmt(d['perc'])} | {fmt(d['trans'])} "
                     f"| {fmt(d['fq'])} |")
        L += ["", "MNIST-sum control:", "", "| SUT | accuracy | transfer |", "|---|---|---|"]
        for n in e5["control"]:
            d = e5["control"][n]
            L.append(f"| {n} | {fmt(d['acc'])} | {fmt(d['trans'])} |")
        L.append("")

    if (e5b := load("E5b_noise.json")):
        v = e5b["verdict"]
        p1 = ("confirmed" if v["P1_monotone_decrease"]
              else "REFUTED — collapse at the first increment")
        p2 = "confirmed" if v["P2_reversal_at_05"] else "refuted"
        curve = {k: round(val, 3) for k, val in v["advantage_curve"].items()}
        L += ["## E5b — label-noise ablation (registered predictions, measured verdict)", "",
              f"advantage(exact − top1) by eta: {curve}",
              f"- P1 (monotone decrease): **{p1}**",
              f"- P2 (reversal by eta=0.5): **{p2}**",
              "", "top-1's advantage on deterministic-perception controls is a",
              "knife-edge artifact: the smallest tested latent noise destroys it.", ""]

    if (e8 := load("E8_results.json")):
        L += ["## E8 — CLUTRR-style generalization (lineal fragment, train k<=3)", "",
              f"learned composition table accuracy: "
              f"{e8['table_accuracy'][0]:.3f} ± {e8['table_accuracy'][1]:.3f}", "",
              "| budget | " + " | ".join(f"k={k}" for k in
                                         sorted(map(int, next(iter(
                                             e8["accuracy"].values())).keys()))) + " |",
              "|---|" + "---|" * len(next(iter(e8["accuracy"].values())))]
        for b, d in e8["accuracy"].items():
            ks = sorted(map(int, d.keys()))
            L.append(f"| {b} | " + " | ".join(f"{d[str(k)][0]:.2f}" for k in ks) + " |")
        L.append("")

    path = os.path.join(OUT, "RESULTS.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
