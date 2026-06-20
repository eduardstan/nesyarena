"""E3 — surrogate axis: the LSE temperature dilemma (Prop. 3 / H6).

Run:  .venv/bin/python -m experiments.e3_surrogate [--config experiments/configs/E3.yaml]

Equal scores verify the closed-form bias = tau*ln(P) to machine precision;
the perturbed family (one boosted proof) exposes the dilemma: as tau -> 0 the
bias vanishes but gradient mass concentrates on the maximal proof (starving
the rest); as tau grows, non-maximal proofs receive gradient but the
optimized objective departs from the claimed max-join. No temperature gives
both. Writes out/E3_results.json and out/F4_surrogate.png.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import nesyarena  # noqa: E402
from nesyarena.generators import surrogate_scores  # noqa: E402
from nesyarena.suts import LSE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))


def load_config(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def nonmax_grad_share(sut: LSE, proofs, probs) -> float:
    """Share of total gradient mass reaching facts outside the maximal proof
    (single-fact proofs: proof j's gradient lands on its one fact)."""
    g = sut.grad(proofs, probs)
    scores = {f: probs[f] for pr in proofs for f in pr}
    fmax = max(scores, key=lambda f: scores[f])
    tot = sum(g.values())
    return (tot - g[fmax]) / tot if tot > 0 else 0.0


def sweep(cfg):
    taus = np.geomspace(cfg["tau_geomspace"]["start"], cfg["tau_geomspace"]["stop"],
                        cfg["tau_geomspace"]["num"])
    rows = []
    for P in cfg["P"]:
        eq = surrogate_scores(P, s=cfg["equal_scores"]["s"])
        pt = surrogate_scores(P, s=cfg["perturbed"]["s"], delta=cfg["perturbed"]["delta"])
        for tau in taus:
            sut = LSE(float(tau))
            rows.append(dict(P=P, tau=float(tau), family="equal",
                             bias=sut.error(*eq),
                             law=float(tau) * math.log(P),
                             share=nonmax_grad_share(sut, *eq)))
            rows.append(dict(P=P, tau=float(tau), family="perturbed",
                             bias=sut.error(*pt), law=None,
                             share=nonmax_grad_share(sut, *pt)))
    return rows


def fig_f4(rows, cfg):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2), constrained_layout=True)
    for P in cfg["figures"]["F4"]["P_show"]:
        eq = sorted([(r["tau"], r["bias"]) for r in rows
                     if r["P"] == P and r["family"] == "equal"])
        ax1.plot([t for t, _ in eq], [b for _, b in eq], "o-", ms=3, label=f"P={P}")
        ax1.plot([t for t, _ in eq], [t * math.log(P) for t, _ in eq], "k--", lw=0.7)
        pt = sorted([(r["tau"], r["bias"], r["share"]) for r in rows
                     if r["P"] == P and r["family"] == "perturbed"])
        ax2.plot([b for _, b, _ in pt], [s for _, _, s in pt], "o-", ms=3, label=f"P={P}")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"temperature $\tau$")
    ax1.set_ylabel("bias  LSE$_\\tau$ $-$ max")
    ax1.set_title(r"Bias law: $\tau\ln P$ (dashed) — exact for equal scores", fontsize=9)
    ax1.legend(fontsize=8)
    ax2.set_xlabel("bias (semantic error vs claimed max-join)")
    ax2.set_ylabel("gradient share to non-maximal proofs")
    ax2.set_title("The dilemma: no $\\tau$ reaches top-left", fontsize=9)
    ax2.legend(fontsize=8)
    path = os.path.join(OUT, "F4_surrogate.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(config_path):
    cfg, cfg_hash = load_config(config_path)
    rows = sweep(cfg)
    worst_law = max(abs(r["bias"] - r["law"]) for r in rows if r["family"] == "equal")
    os.makedirs(OUT, exist_ok=True)
    payload = dict(experiment="E3", package_version=nesyarena.__version__,
                   config=cfg, config_sha256=cfg_hash,
                   bias_law_max_abs_dev=worst_law, rows=rows)
    res = os.path.join(OUT, "E3_results.json")
    with open(res, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    f4 = fig_f4(rows, cfg)
    print(f"E3: bias-law max |dev| = {worst_law:.2e} (prediction: machine precision)")
    print(f"wrote {res}\n      {f4}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "E3.yaml"))
    main(ap.parse_args().config)
