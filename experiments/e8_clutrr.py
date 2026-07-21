"""E8 — CLUTRR-style systematic generalization (train short, test long).

Run:  .venv/bin/python -m experiments.e8_clutrr

Scope (stated honestly, as in the paper): the *lineal* fragment of CLUTRR's
kinship domain — relations are (gender, generation-offset) pairs with offset
in [-2, 2], ten relations total, six base relations (|offset| <= 1):
father, mother, son, daughter, brother, sister. Composition is deterministic
and closed: (g1, d1) o (g2, d2) = (g2, d1 + d2), and the generator only emits
chains whose prefix offsets stay in range, so every query is well-defined.
This is CLUTRR's core mechanic with our own generator (no text layer; the
perception analogue is the *learned composition table*, NeuralLP-style).

Setup. A chain of k base relations between k+1 entities; the query is the
endpoint relation (10-way classification). Learnable: an input embedding
U (6 -> 10 distributions) and a composition table T (10 x 6 -> 10
distributions), composed by sum-product along the chain. Training: cross-
entropy on chains of length k in {2, 3} (within every budget, so all modes
share the SAME learned tables). Deployment: compose at most n steps
(truncation semantics); a chain of length k needs k - 1 steps.

REGISTERED PREDICTIONS:
  P1: convergent test accuracy stays high out to k = 10 (the learned table
      approaches the true deterministic composition from k <= 3 supervision).
  P2: a budget-n reasoner is exact for k <= n + 1 and collapses for
      k > n + 1 — the depth horizon as a generalization cliff at n + 1.

Writes out/E8_results.json and out/F11_clutrr.png.
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
from experiments.palette import truncation_color  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))

# relations: (gender, offset), offset in [-2, 2] -> 10 relations
GENDERS = ("M", "F")
OFFSETS = (-2, -1, 0, 1, 2)
RELS = [(g, d) for d in OFFSETS for g in GENDERS]
RIDX = {r: i for i, r in enumerate(RELS)}
BASE = [(g, d) for d in (-1, 0, 1) for g in GENDERS]   # 6 base relations
BIDX = {r: i for i, r in enumerate(BASE)}
NAMES = {("M", 1): "father", ("F", 1): "mother", ("M", -1): "son",
         ("F", -1): "daughter", ("M", 0): "brother", ("F", 0): "sister",
         ("M", 2): "grandfather", ("F", 2): "grandmother",
         ("M", -2): "grandson", ("F", -2): "granddaughter"}


def load_config(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def sample_chain(k, rng):
    """k base relations whose prefix offsets stay in [-2, 2]; returns
    (base indices, endpoint relation index)."""
    while True:
        chain, off = [], 0
        ok = True
        for _ in range(k):
            g, d = BASE[rng.integers(0, len(BASE))]
            if not (-2 <= off + d <= 2):
                ok = False
                break
            off += d
            chain.append((g, d))
        if ok:
            endpoint = (chain[-1][0], off)   # gender of last target, total offset
            return [BIDX[c] for c in chain], RIDX[endpoint]


def gen_set(ks, n, rng):
    X, y, lens = [], [], []
    for _ in range(n):
        k = int(rng.choice(ks))
        c, e = sample_chain(k, rng)
        X.append(c)
        y.append(e)
        lens.append(k)
    return X, y, lens


def compose_batch(chains, U, T, budget):
    """Sum-product composition with at most `budget` steps (truncation).
    Returns (B, 10) state distributions describing rel(entity0, entity_{m+1})
    where m = min(len-1, budget) — for truncated chains this is the relation
    to an INTERMEDIATE entity, i.e. the wrong query, exactly as a bounded
    unroller deploys."""
    out = []
    Usm = torch.softmax(U, dim=1)
    Tsm = torch.softmax(T, dim=2)
    for c in chains:
        s = Usm[c[0]]
        for b in c[1:][:budget]:
            s = torch.einsum("r,rs->s", s, Tsm[:, b, :])
        out.append(s)
    return torch.stack(out)


def fig_f11(summary, cfg):
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    for b in cfg["budgets"]:
        ks = cfg["test_k"]
        d = summary[str(b)]  # inner keys are ints in-process, strings from JSON
        mu = [d.get(k, d.get(str(k)))[0] for k in ks]
        sd = [d.get(k, d.get(str(k)))[1] for k in ks]
        label = "convergent" if b == "convergent" else f"budget n={b}"
        ax.errorbar(ks, mu, yerr=sd, fmt="o-", ms=4,
                    color=truncation_color(b), label=label)
        if b != "convergent":
            ax.axvline(int(b) + 1, ls=":", lw=0.8, color="gray")
    ax.axhline(1 / len(RELS), color="k", lw=0.7, ls=":", label="chance (1/10)")
    ax.set_xlabel("test chain length k (trained on k ∈ {2, 3})")
    ax.set_ylabel("endpoint-relation accuracy")
    ax.set_title("CLUTRR-style generalization: cliffs at the depth horizon k = n+1",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "F11_clutrr.png"), dpi=160)
    plt.close(fig)


def main(config_path):
    cfg, cfg_hash = load_config(config_path)
    acc = {str(b): {k: [] for k in cfg["test_k"]} for b in cfg["budgets"]}
    table_acc = []
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(800 + seed)
        Xtr, ytr, _ = gen_set(cfg["train_k"], cfg["n_train"], rng)
        tests = {k: gen_set([k], cfg["n_test_per_k"], rng)[:2] for k in cfg["test_k"]}
        torch.manual_seed(4000 + seed)
        U = torch.zeros(len(BASE), len(RELS), dtype=torch.float64, requires_grad=True)
        T = torch.zeros(len(RELS), len(BASE), len(RELS), dtype=torch.float64,
                        requires_grad=True)
        with torch.no_grad():
            U += 0.01 * torch.randn_like(U)
            T += 0.01 * torch.randn_like(T)
        opt = torch.optim.Adam([U, T], lr=cfg["lr"])
        yv = torch.tensor(ytr)
        for t in range(cfg["steps"]):
            ix = np.random.default_rng(t * 23 + seed).integers(0, len(Xtr), 64)
            s = compose_batch([Xtr[i] for i in ix], U, T, budget=10_000)
            loss = -torch.log(s[torch.arange(len(ix)), yv[ix]].clamp_min(1e-9)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            # learned-table quality: argmax of T rows vs true composition
            hits = tot = 0
            for r, (_g1, d1) in enumerate(RELS):
                for b, (g2, d2) in enumerate(BASE):
                    if -2 <= d1 + d2 <= 2:
                        tot += 1
                        hits += int(torch.argmax(T[r, b]) == RIDX[(g2, d1 + d2)])
            table_acc.append(hits / tot)
            for k in cfg["test_k"]:
                Xte, yte = tests[k]
                yt = torch.tensor(yte)
                for b in cfg["budgets"]:
                    budget = 10_000 if b == "convergent" else int(b)
                    s = compose_batch(Xte, U, T, budget)
                    acc[str(b)][k].append(float((s.argmax(1) == yt).double().mean()))
        print(f"  [E8 seed {seed}] table accuracy {table_acc[-1]:.3f}")

    summary = {b: {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()}
               for b, d in acc.items()}
    cliffs = {}
    for b in cfg["budgets"]:
        if b == "convergent":
            continue
        h = next((k for k in cfg["test_k"]
                  if summary[str(b)][k][0] < 0.5), None)
        cliffs[str(b)] = h
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "E8_results.json"), "w") as fh:
        json.dump(dict(experiment="E8", package_version=nesyarena.__version__,
                       config=cfg, config_sha256=cfg_hash,
                       relations={str(r): NAMES[r] for r in RELS},
                       table_accuracy=[float(np.mean(table_acc)), float(np.std(table_acc))],
                       accuracy=summary, first_k_below_chance_plus=cliffs),
                  fh, indent=1, sort_keys=True)

    fig_f11(summary, cfg)

    print(f"learned-table accuracy: {np.mean(table_acc):.3f} ± {np.std(table_acc):.3f}")
    for b in cfg["budgets"]:
        row = " ".join(f"k{k}:{summary[str(b)][k][0]:.2f}" for k in cfg["test_k"])
        print(f"  {str(b):>10}: {row}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "E8.yaml"))
    main(ap.parse_args().config)
