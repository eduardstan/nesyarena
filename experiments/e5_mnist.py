"""E5/E6 at MNIST scale — real-digit replication of the learning experiments.

Run:  .venv/bin/python -m experiments.e5_mnist

Treatment (MNIST-path, overlap structure): 3x3 grid of MNIST digits; cell
open iff digit even; query = monotone reachability corner-to-corner (6 paths
of 5 cells, genuinely overlapping). Control (MNIST-sum structure): digit
pairs from {0,1,2}; train on the sum query (mutually exclusive proofs;
exact and add-mult share one code path), transfer to the equality query.

Real images have no closed-form Bayes posterior, so (per the draft's scope
note) this experiment measures the accuracy-ties-vs-transfer-divergence
finding, not exact calibration; perception quality is reported against the
true cell states. Ground truth per sample is deterministic (digit labels are
known), so held-out transfer is measured against exact truth.

Writes out/E5_mnist.json and out/F9_mnist.png.
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
from nesyarena.learning import BatchStructure, prov_value  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))
DATA = os.path.join(HERE, "..", ".data")
SUT_K = {"exact": None, "addmult": None, "top1": 1, "top3": 3, "minmax": None,
        "ltn_product": None, "ltn_godel": None}

EPS = 1e-4


def load_config(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def mnist_arrays():
    import torchvision
    tr = torchvision.datasets.MNIST(root=DATA, train=True, download=True)
    te = torchvision.datasets.MNIST(root=DATA, train=False, download=True)
    def prep(ds):
        X = ds.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
        X = (X - 0.1307) / 0.3081
        return X, ds.targets.numpy()
    return prep(tr), prep(te)


def monotone_paths(g: int):
    """All monotone (right/down) paths (0,0) -> (g-1,g-1) as cell-name sets."""
    paths = []

    def rec(r, c, acc):
        if (r, c) == (g - 1, g - 1):
            paths.append(frozenset(acc))
            return
        if r + 1 < g:
            rec(r + 1, c, acc + [f"c{r+1}{c}"])
        if c + 1 < g:
            rec(r, c + 1, acc + [f"c{r}{c+1}"])

    rec(0, 0, ["c00"])
    return paths


def make_net(hidden, dout, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(784, hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, dout)).double()


def bce(v, y):
    v = v.clamp(EPS, 1 - EPS)
    return -(y * v.log() + (1 - y) * (1 - v).log()).mean()


# ------------------------------------------------------------- treatment ----

def run_treatment(cfg, mnist):
    (Xtr_all, ytr_all), (Xte_all, yte_all) = mnist
    g = cfg["treatment"]["grid"]
    ncell = g * g
    proofs = monotone_paths(g)
    facts = sorted(set().union(*proofs))
    S_full = BatchStructure(proofs, facts)
    held = [BatchStructure([proofs[i] for i in idxs], facts)
            for idxs in cfg["treatment"]["held_queries"]]
    res = {n: dict(acc=[], perc=[], trans=[], fq=[]) for n in cfg["treatment"]["suts"]}
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(600 + seed)
        itr = rng.integers(0, len(Xtr_all), (cfg["n_train"], ncell))
        ite = rng.integers(0, len(Xte_all), (cfg["n_test"], ncell))
        Ztr = (ytr_all[itr] % 2 == 0)
        Zte = (yte_all[ite] % 2 == 0)
        ytr = S_full.truth(torch.tensor(Ztr)).double().numpy()
        yte = S_full.truth(torch.tensor(Zte)).double().numpy()
        Xte_t = torch.tensor(Xte_all[ite.reshape(-1)])
        Zte_t = torch.tensor(Zte.astype(np.float64))
        for name in cfg["treatment"]["suts"]:
            net = make_net(cfg["hidden"], 1, 1000 + seed)
            opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
            steps = cfg["epochs"] * cfg["n_train"] // cfg["batch"]
            for t in range(steps):
                ix = np.random.default_rng(t * 13 + seed).integers(0, cfg["n_train"],
                                                                   cfg["batch"])
                xb = torch.tensor(Xtr_all[itr[ix].reshape(-1)])
                p = torch.sigmoid(net(xb)).reshape(cfg["batch"], ncell).clamp(EPS, 1 - EPS)
                v = prov_value(name, S_full, p, SUT_K[name])
                loss = bce(v, torch.tensor(ytr[ix]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                pte = torch.sigmoid(net(Xte_t)).reshape(-1, ncell).clamp(EPS, 1 - EPS)
                v_full = prov_value(name, S_full, pte, SUT_K[name]).numpy()
                res[name]["acc"].append(float(np.mean((v_full >= 0.5) == (yte >= 0.5))))
                res[name]["perc"].append(float((pte - Zte_t).abs().mean()))
                te, fq = [], []
                for H in held:
                    truth = H.truth(torch.tensor(Zte)).double().numpy()
                    vh = prov_value(name, H, pte, SUT_K[name]).numpy()
                    wq = H.wmc(pte).numpy()
                    te.append(float(np.mean(np.abs(vh - truth))))
                    fq.append(float(np.mean(np.abs(wq - truth))))
                res[name]["trans"].append(float(np.mean(te)))
                res[name]["fq"].append(float(np.mean(fq)))
        print(f"  [mnist-path seed {seed}] done")
    return res


# --------------------------------------------------------------- control ----

def cat_value(name, terms, PA, PB, k=None):
    vals = torch.stack([PA[:, i] * PB[:, j] for (i, j) in terms], dim=1)
    if name in ("exact", "addmult"):
        return vals.sum(dim=1)
    if name.startswith("top"):
        k = k if k is not None else int(name[3:])
        if k >= len(terms):
            # k >= P: retains every (mutually exclusive) proof — exact by
            # construction, but computed through the top-k path on purpose
            k = len(terms)
        order = torch.argsort(vals.detach(), dim=1, descending=True)[:, :k]  # frozen
        return vals.gather(1, order).sum(dim=1)
    if name in ("minmax", "ltn_godel"):
        mins = torch.stack([torch.minimum(PA[:, i], PB[:, j]) for (i, j) in terms], dim=1)
        return mins.amax(dim=1)
    if name == "ltn_product":
        acc = vals[:, 0]
        for j in range(1, vals.shape[1]):
            b = vals[:, j]
            acc = acc + b - acc * b
        return acc

    raise ValueError(name)


def run_control(cfg, mnist):
    (Xtr_all, ytr_all), (Xte_all, yte_all) = mnist
    digs = cfg["control"]["digits"]
    tr_mask = np.isin(ytr_all, digs)
    te_mask = np.isin(yte_all, digs)
    Xd, yd = Xtr_all[tr_mask], ytr_all[tr_mask]
    Xdt, ydt = Xte_all[te_mask], yte_all[te_mask]
    terms_tr = [tuple(t) for t in cfg["control"]["train_terms"]]
    terms_he = [tuple(t) for t in cfg["control"]["held_terms"]]
    res = {n: dict(acc=[], trans=[]) for n in cfg["control"]["suts"]}
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(700 + seed)
        iA = rng.integers(0, len(Xd), cfg["n_train"])
        iB = rng.integers(0, len(Xd), cfg["n_train"])
        ytr = (yd[iA] + yd[iB] == 2).astype(float)
        iAte = rng.integers(0, len(Xdt), cfg["n_test"])
        iBte = rng.integers(0, len(Xdt), cfg["n_test"])
        yte = (ydt[iAte] + ydt[iBte] == 2).astype(float)
        yhe = (ydt[iAte] == ydt[iBte]).astype(float)
        XA_t, XB_t = torch.tensor(Xdt[iAte]), torch.tensor(Xdt[iBte])
        for name in cfg["control"]["suts"]:
            net = make_net(cfg["hidden"], len(digs), 2000 + seed)
            opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
            steps = cfg["epochs"] * cfg["n_train"] // cfg["batch"]
            for t in range(steps):
                ix = np.random.default_rng(t * 17 + seed).integers(0, cfg["n_train"],
                                                                   cfg["batch"])
                PA = torch.softmax(net(torch.tensor(Xd[iA[ix]])), dim=1).clamp(EPS, 1.0)
                PB = torch.softmax(net(torch.tensor(Xd[iB[ix]])), dim=1).clamp(EPS, 1.0)
                loss = bce(cat_value(name, terms_tr, PA, PB), torch.tensor(ytr[ix]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                PA = torch.softmax(net(XA_t), dim=1).clamp(EPS, 1.0)
                PB = torch.softmax(net(XB_t), dim=1).clamp(EPS, 1.0)
                v = cat_value(name, terms_tr, PA, PB).numpy()
                vh = cat_value(name, terms_he, PA, PB).numpy()
                res[name]["acc"].append(float(np.mean((v >= 0.5) == (yte >= 0.5))))
                res[name]["trans"].append(float(np.mean(np.abs(vh - yhe))))
        print(f"  [mnist-sum seed {seed}] done")
    return res


def fig_f9(treat, ctrl, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    names = cfg["treatment"]["suts"]
    x = np.arange(len(names))
    for ax, metric, title in [(axes[0], "acc", "MNIST-path: task accuracy"),
                              (axes[1], "trans", "MNIST-path: held-out transfer error")]:
        ax.bar(x, [np.mean(treat[n][metric]) for n in names],
               yerr=[np.std(treat[n][metric]) for n in names], width=0.6)
        ax.set_xticks(x, names, fontsize=8, rotation=20)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylim(0.5, 1.0)
    cn = cfg["control"]["suts"]
    xc = np.arange(len(cn))
    axes[2].bar(xc, [np.mean(ctrl[n]["trans"]) for n in cn],
                yerr=[np.std(ctrl[n]["trans"]) for n in cn], width=0.6)
    axes[2].set_xticks(xc, cn, fontsize=8, rotation=20)
    axes[2].set_title("MNIST-sum control: transfer error", fontsize=9)
    axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle("Real-digit replication: accuracy ties, transfer diverges; "
                 "the sum-structured control cannot see double-counting", fontsize=10)
    path = os.path.join(OUT, "F9_mnist.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(config_path):
    cfg, cfg_hash = load_config(config_path)
    mnist = mnist_arrays()
    print("treatment (MNIST-path):")
    treat = run_treatment(cfg, mnist)
    print("control (MNIST-sum):")
    ctrl = run_control(cfg, mnist)
    os.makedirs(OUT, exist_ok=True)

    def summ(d):
        return {n: {m: [float(np.mean(v)), float(np.std(v))] for m, v in dd.items()}
                for n, dd in d.items()}

    payload = dict(experiment="E5-mnist", package_version=nesyarena.__version__,
                   config=cfg, config_sha256=cfg_hash,
                   treatment=summ(treat), control=summ(ctrl))
    res = os.path.join(OUT, "E5_mnist.json")
    with open(res, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    f9 = fig_f9(treat, ctrl, cfg)
    print(f"\n{'SUT':9} {'accuracy':>15} {'perception':>15} {'transfer':>15}")
    for n in cfg["treatment"]["suts"]:
        print(f"{n:9} {np.mean(treat[n]['acc']):>7.3f} ± {np.std(treat[n]['acc']):.3f}"
              f" {np.mean(treat[n]['perc']):>7.3f} ± {np.std(treat[n]['perc']):.3f}"
              f" {np.mean(treat[n]['trans']):>7.3f} ± {np.std(treat[n]['trans']):.3f}")
    print("\ncontrol:")
    for n in cfg["control"]["suts"]:
        print(f"{n:9} acc {np.mean(ctrl[n]['acc']):.3f}  trans {np.mean(ctrl[n]['trans']):.3f}")
    print(f"wrote {res}\n      {f9}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "E5_mnist.yaml"))
    main(ap.parse_args().config)
