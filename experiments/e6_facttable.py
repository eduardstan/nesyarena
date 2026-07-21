"""E6 (fact-table setting) — training a free fact table through each
reasoner, in the population-loss limit (H8, fact-table arm).

Run:  .venv/bin/python -m experiments.e6_facttable

Treatment: G1 overlap structure (P=4, L=3, c=2), heterogeneous ground-truth
fact probabilities; supervision is a 7-query battery (pairs + union) with
exact-WMC targets; evaluation is held-out queries (singletons + a triple)
against exact ground truth, *aggregated by the same reasoner at deployment*.
Control: the categorical-sum structure (MNIST-sum_2 shape, |dom| = 3) whose
proofs are mutually exclusive — rung-P violations are invisible there by
construction (H9), and exact/add-mult share one code path deliberately, so
their identity is structural, not numeric luck.

Registered findings to replicate (RESULTS.md, toy run): exact transfers at
~0 on treatment; add-mult/top-1/min-max are harmed on treatment; on control,
exact == add-mult exactly while top-1 is harmed (truncation is what
sum-structured tasks CAN see).

Writes out/E6_facttable.json and out/F6_learning_transfer.png.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import nesyarena  # noqa: E402
from experiments.palette import sut_color  # noqa: E402
from nesyarena.generators import overlap_family  # noqa: E402
from nesyarena.learning import BatchStructure, prov_value  # noqa: E402
from nesyarena.oracle import wmc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))
SUT_K = {"exact": None, "addmult": None, "addmult_st": None,
         "top1": 1, "top3": 3, "minmax": None,
         "ltn_product": None, "ltn_godel": None}

EPS = 1e-4


def load_config(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def bce(v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    v = v.clamp(EPS, 1 - EPS)
    return -(y * v.log() + (1 - y) * (1 - v).log()).mean()


# ------------------------------------------------------------- treatment ----

def run_treatment(cfg):
    tc = cfg["treatment"]
    P, L, c = tc["structure"]["P"], tc["structure"]["L"], tc["structure"]["c"]
    res = {n: dict(trans=[], facts_q=[], fact_mae=[]) for n in cfg["suts"]}
    for seed in range(cfg["seeds"]):
        rng = np.random.default_rng(tc["truth_seeds_base"] + seed)
        inst = overlap_family(P, L, c, p=0.6, rng=rng, het=True)
        proofs, q_true = inst.proofs, inst.probs
        facts = sorted(q_true, key=repr)
        train_S = [BatchStructure([proofs[i] for i in idxs], facts)
                   for idxs in tc["train_queries"]]
        targets = [wmc([proofs[i] for i in idxs], q_true) for idxs in tc["train_queries"]]
        held = [(idxs, BatchStructure([proofs[i] for i in idxs], facts))
                for idxs in tc["held_queries"]]
        for name in cfg["suts"]:
            init = -1.0 + 0.1 * np.random.default_rng(seed).standard_normal(len(facts))
            th = torch.tensor(init, dtype=torch.float64, requires_grad=True)
            opt = torch.optim.Adam([th], lr=cfg["optimizer"]["lr"])
            yv = torch.tensor(targets, dtype=torch.float64)
            for _ in range(cfg["optimizer"]["steps"]):
                opt.zero_grad()
                p = torch.sigmoid(th).clamp(EPS, 1 - EPS)[None, :]
                vals = torch.cat([prov_value(name, S, p, SUT_K[name]) for S in train_S])
                bce(vals, yv).backward()
                opt.step()
            learned_vec = torch.sigmoid(th.detach())
            learned = {f: float(v) for f, v in zip(facts, learned_vec, strict=True)}
            te, fq = [], []
            for idxs, S in held:
                truth = wmc([proofs[i] for i in idxs], q_true)
                v_dep = float(prov_value(name, S, learned_vec[None, :], SUT_K[name])[0])
                te.append(abs(v_dep - truth))
                fq.append(abs(wmc([proofs[i] for i in idxs], learned) - truth))
            res[name]["trans"].append(float(np.mean(te)))
            res[name]["facts_q"].append(float(np.mean(fq)))
            res[name]["fact_mae"].append(
                float(np.mean([abs(learned[f] - q_true[f]) for f in q_true])))
            print(f"  [treatment seed {seed}] {name} done")
    return res


# --------------------------------------------------------------- control ----

def cat_value(name, terms, pA, pB, k):
    """Disjoint categorical sum. exact and addmult share one path: on
    mutually exclusive proofs add-mult IS distribution semantics (H9).
    ltn_product does NOT join that identity: its Or-prod (a+b-ab) assumes
    INDEPENDENCE, not mutual exclusivity, so on this disjoint control it
    systematically UNDER-estimates (opposite bias from the G1 overlap
    treatment, where it over-estimates by assuming independence where
    facts are actually shared/correlated). Genuinely different failure
    mode, not a bug -- report it, don't "fix" it.
    ltn_godel is byte-identical to minmax by definition (And=min, Or=max)."""

    vals = torch.stack([pA[i] * pB[j] for (i, j) in terms])
    if name in ("exact", "addmult", "addmult_st"):  # disjoint: no clamp active
        return vals.sum()
    if name.startswith("top"):
        order = torch.argsort(vals.detach(), descending=True)[:k]  # frozen
        return vals[order].sum()
    if name in ("minmax", "ltn_godel"):
         mins = torch.stack([torch.minimum(pA[i], pB[j]) for (i, j) in terms])
         return mins.amax()
    if name == "ltn_product":
        acc = vals[0]
        for v in vals[1:]:
            acc = acc + v - acc * v
        return acc

    raise ValueError(name)


def run_control(cfg):
    dom = cfg["control"]["domain"]
    sums = {s: [(i, s - i) for i in range(dom) if 0 <= s - i < dom]
            for s in range(2 * dom - 1)}
    diag = [(i, i) for i in range(dom)]
    res = {n: dict(trans=[]) for n in cfg["suts"]}
    for seed in range(cfg["seeds"]):
        rng = np.random.default_rng(cfg["control"]["truth_seeds_base"] + seed)
        tA, tB = rng.dirichlet(np.ones(dom)), rng.dirichlet(np.ones(dom))
        targets = {s: float(sum(tA[i] * tB[j] for (i, j) in sums[s])) for s in sums}
        truth_diag = float(sum(tA[i] * tB[i] for i in range(dom)))
        for name in cfg["suts"]:
            k = SUT_K[name] or 1
            init = 0.1 * np.random.default_rng(seed).standard_normal(2 * dom)
            th = torch.tensor(init, dtype=torch.float64, requires_grad=True)
            opt = torch.optim.Adam([th], lr=cfg["optimizer"]["lr"])
            for _ in range(cfg["optimizer"]["steps"]):
                opt.zero_grad()
                pA, pB = torch.softmax(th[:dom], 0), torch.softmax(th[dom:], 0)
                vals = torch.stack([cat_value(name, sums[s], pA, pB, k).clamp(EPS, 1 - EPS)
                                    for s in sums])
                yv = torch.tensor([targets[s] for s in sums], dtype=torch.float64)
                bce(vals, yv).backward()
                opt.step()
            with torch.no_grad():
                pA, pB = torch.softmax(th[:dom], 0), torch.softmax(th[dom:], 0)
                v = float(cat_value(name, diag, pA, pB, k))
            res[name]["trans"].append(abs(v - truth_diag))
            print(f"  [treatment seed {seed}] {name} done")
    return res


def fig_f6(tstats, cstats, names):
    """tstats/cstats[n][metric] = [mean, std] over seeds — the JSON's
    `treatment`/`control` fields, so the figure regenerates from the JSON
    alone."""
    fig, ax = plt.subplots(figsize=(8.4, 4.0), constrained_layout=True)
    x = np.arange(len(names))
    w = 0.38
    cm = [cstats[n]["trans"][0] for n in names]
    cs = [cstats[n]["trans"][1] for n in names]
    tm = [tstats[n]["trans"][0] for n in names]
    ts = [tstats[n]["trans"][1] for n in names]
    cols = [sut_color(n) for n in names]
    ax.bar(x - w / 2, cm, w, yerr=cs, color=cols, alpha=0.45,
           label="control: disjoint task (MNIST-sum structure)")
    ax.bar(x + w / 2, tm, w, yerr=ts, color=cols,
           label="treatment: overlap task (G1)")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("held-out query error after training (5 seeds)")
    ax.set_title("Training through a misreasoner corrupts transferred knowledge —\n"
                 "except on the disjoint task, where the default benchmark lives", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    path = os.path.join(OUT, "F6_learning_transfer.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(config_path):
    cfg, cfg_hash = load_config(config_path)
    torch.manual_seed(0)
    treat = run_treatment(cfg)
    ctrl = run_control(cfg)
    os.makedirs(OUT, exist_ok=True)
    payload = dict(experiment="E6-facttable", package_version=nesyarena.__version__,
                   config=cfg, config_sha256=cfg_hash,
                   treatment={n: {m: [float(np.mean(v)), float(np.std(v))]
                                  for m, v in d.items()} for n, d in treat.items()},
                   control={n: {m: [float(np.mean(v)), float(np.std(v))]
                                for m, v in d.items()} for n, d in ctrl.items()})
    res = os.path.join(OUT, "E6_facttable.json")
    with open(res, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    f6 = fig_f6(payload["treatment"], payload["control"], cfg["suts"])
    print(f"{'SUT':10} {'control':>10} {'treatment':>10}   (mean held-out error, 5 seeds)")
    for n in cfg["suts"]:
        print(f"{n:10} {np.mean(ctrl[n]['trans']):>10.3f} {np.mean(treat[n]['trans']):>10.3f}")
    print(f"wrote {res}\n      {f6}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "E6_facttable.yaml"))
    main(ap.parse_args().config)
