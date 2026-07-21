"""E6 (pixel-perception setting) — the H8 headline experiment.

Run:  .venv/bin/python -m experiments.e6_pixels

A perception MLP is trained end-to-end *through* each reasoner on sampled
binary labels. The generative model is known, so Bayes-optimal posteriors are
closed-form and calibration is measured against exact ground truth, not
proxies. Treatment: the G1 overlap structure (genuine rung-P structure).
Control: a categorical-sum task (mutually exclusive proofs — the
MNIST-sum-style cell where double-counting is invisible by construction, and
exact/add-mult share one code path).

Registered findings to replicate (RESULTS.md): task accuracy statistically
indistinguishable across reasoners while calibration-vs-Bayes and held-out
transfer separate them (exact = floor; add-mult ~ +44%/+50% over floor;
min-max elevated; top-1/top-3 at floor under pixel parameterization — the
refuted-H-A result); on control, exact == add-mult identically.

Writes out/E6_pixels.json and out/F7_accuracy_vs_fidelity.png.
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
from nesyarena.generators import overlap_family  # noqa: E402
from nesyarena.learning import BatchStructure, prov_value  # noqa: E402

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


def make_mlp(din, dh, dout, np_rng) -> torch.nn.Sequential:
    """MLP with the toy's init scheme (0.15 * randn, zero biases)."""
    net = torch.nn.Sequential(torch.nn.Linear(din, dh), torch.nn.Tanh(),
                              torch.nn.Linear(dh, dout)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(np_rng.standard_normal((din, dh)).T * 0.15))
        net[0].bias.zero_()
        net[2].weight.copy_(torch.tensor(np_rng.standard_normal((dh, dout)).T * 0.15))
        net[2].bias.zero_()
    return net


def bce(v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    v = v.clamp(EPS, 1 - EPS)
    return -(y * v.log() + (1 - y) * (1 - v).log()).mean()


# ------------------------------------------------------------- treatment ----

def gen_treatment(n, rng, sig, cfg):
    Z = rng.random((n, cfg["ncell"])) < cfg["pi"]
    X = Z[:, :, None] * cfg["alpha"] * sig[None, None, :] \
        + rng.standard_normal((n, cfg["ncell"], cfg["din"]))
    logit = cfg["alpha"] * (X @ sig) - cfg["alpha"] ** 2 / 2 \
        + np.log(cfg["pi"] / (1 - cfg["pi"]))
    bayes = 1.0 / (1.0 + np.exp(-logit))
    return X, Z, bayes


def run_treatment(cfg):
    tc = cfg["treatment"]
    st = tc["structure"]
    inst = overlap_family(st["P"], st["L"], st["c"], 0.6)
    facts = sorted(inst.probs, key=repr)
    S_full = BatchStructure(inst.proofs, facts)
    held = [BatchStructure([inst.proofs[i] for i in idxs], facts)
            for idxs in tc["held_queries"]]
    res = {n: dict(acc=[], cal=[], trans=[], fq=[]) for n in tc["suts"]}
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(300 + seed)
        sig = rng.standard_normal(cfg["din"])
        sig /= np.linalg.norm(sig)
        Xtr, Ztr, _ = gen_treatment(cfg["n_train"], rng, sig, cfg)
        ytr = S_full.truth(torch.tensor(Ztr)).double().numpy()
        Xte, Zte, bay_te = gen_treatment(cfg["n_test"], rng, sig, cfg)
        yte = S_full.truth(torch.tensor(Zte)).double().numpy()
        Xte_t = torch.tensor(Xte.reshape(-1, cfg["din"]))
        for name in tc["suts"]:
            net = make_mlp(cfg["din"], cfg["hidden"], 1, np.random.default_rng(1000 + seed))
            opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
            steps = cfg["epochs"] * cfg["n_train"] // cfg["batch"]
            for t in range(steps):
                ix = np.random.default_rng(t * 7 + seed).integers(0, cfg["n_train"],
                                                                  cfg["batch"])
                xb = torch.tensor(Xtr[ix].reshape(-1, cfg["din"]))
                p = torch.sigmoid(net(xb)).reshape(cfg["batch"], cfg["ncell"])
                p = p.clamp(EPS, 1 - EPS)
                v = prov_value(name, S_full, p, SUT_K[name])
                loss = bce(v, torch.tensor(ytr[ix]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                pte = torch.sigmoid(net(Xte_t)).reshape(-1, cfg["ncell"]).clamp(EPS, 1 - EPS)
                v_full = prov_value(name, S_full, pte, SUT_K[name]).numpy()
                res[name]["acc"].append(float(np.mean((v_full >= 0.5) == (yte >= 0.5))))
                res[name]["cal"].append(float(np.mean(np.abs(pte.numpy() - bay_te))))
                bay_t = torch.tensor(np.clip(bay_te, EPS, 1 - EPS))
                te, fq = [], []
                for H in held:
                    truth = H.wmc(bay_t).numpy()
                    vh = prov_value(name, H, pte, SUT_K[name]).numpy()
                    wq = H.wmc(pte).numpy()
                    te.append(float(np.mean(np.abs(vh - truth))))
                    fq.append(float(np.mean(np.abs(wq - truth))))
                res[name]["trans"].append(float(np.mean(te)))
                res[name]["fq"].append(float(np.mean(fq)))
        print(f"  [treatment seed {seed}] done")
    return res


# --------------------------------------------------------------- control ----

def cat_value(name, terms, PA, PB):
    """Batched disjoint categorical value; exact/add-mult share one path.
    top-k with k >= len(terms) retains every proof — exact by construction,
    computed through the top-k path so the identity is measured."""
    vals = torch.stack([PA[:, i] * PB[:, j] for (i, j) in terms], dim=1)
    if name in ("exact", "addmult", "addmult_st"):
        return vals.sum(dim=1)
    if name.startswith("top"):
        k = min(int(name[3:]), len(terms))
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


def run_control(cfg):
    cc = cfg["control"]
    pA_true, pB_true = np.array(cc["pA_true"]), np.array(cc["pB_true"])
    terms_tr = [tuple(t) for t in cc["train_terms"]]
    terms_he = [tuple(t) for t in cc["held_terms"]]
    res = {n: dict(acc=[], cal=[], trans=[]) for n in cc["suts"]}
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(400 + seed)
        sigs = np.linalg.qr(rng.standard_normal((cfg["din"], 3)))[0].T

        def gen(m, rng=rng, sigs=sigs):
            zA = rng.choice(3, m, p=pA_true)
            zB = rng.choice(3, m, p=pB_true)
            XA = cfg["alpha"] * sigs[zA] + rng.standard_normal((m, cfg["din"]))
            XB = cfg["alpha"] * sigs[zB] + rng.standard_normal((m, cfg["din"]))

            def bayes(X, marg, sigs=sigs):
                lo = cfg["alpha"] * (X @ sigs.T) - cfg["alpha"] ** 2 / 2 + np.log(marg)
                e = np.exp(lo - lo.max(1, keepdims=True))
                return e / e.sum(1, keepdims=True)

            return XA, XB, zA, zB, bayes(XA, pA_true), bayes(XB, pB_true)

        XA, XB, zA, zB, _, _ = gen(cfg["n_train"])
        ytr = (zA + zB == 2).astype(float)
        XAte, XBte, zAte, zBte, bayA, bayB = gen(cfg["n_test"])
        yte = (zAte + zBte == 2).astype(float)
        for name in cc["suts"]:
            net = make_mlp(cfg["din"], cfg["hidden"], 3, np.random.default_rng(2000 + seed))
            opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
            steps = cfg["epochs"] * cfg["n_train"] // cfg["batch"]
            for t in range(steps):
                ix = np.random.default_rng(t * 11 + seed).integers(0, cfg["n_train"],
                                                                   cfg["batch"])
                PA = torch.softmax(net(torch.tensor(XA[ix])), dim=1).clamp(EPS, 1.0)
                PB = torch.softmax(net(torch.tensor(XB[ix])), dim=1).clamp(EPS, 1.0)
                v = cat_value(name, terms_tr, PA, PB)
                loss = bce(v, torch.tensor(ytr[ix]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                PAh = torch.softmax(net(torch.tensor(XAte)), dim=1).clamp(EPS, 1.0)
                PBh = torch.softmax(net(torch.tensor(XBte)), dim=1).clamp(EPS, 1.0)
                v = cat_value(name, terms_tr, PAh, PBh).numpy()
                res[name]["acc"].append(float(np.mean((v >= 0.5) == (yte >= 0.5))))
                res[name]["cal"].append(
                    float(np.mean(np.abs(PAh.numpy() - bayA)) / 2
                          + np.mean(np.abs(PBh.numpy() - bayB)) / 2))
                vh = cat_value(name, terms_he, PAh, PBh).numpy()
                truth = (bayA * bayB).sum(1)
                res[name]["trans"].append(float(np.mean(np.abs(vh - truth))))
        print(f"  [control seed {seed}] done")
    return res


# ---------------------------------------------------------------- figure ----

def fig_f7(treat, cfg):
    names = cfg["treatment"]["suts"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    x = np.arange(len(names))
    for ax, metric, title in [(axes[0], "acc", "task accuracy (ties)"),
                              (axes[1], "cal", "calibration vs exact Bayes"),
                              (axes[2], "trans", "held-out transfer error")]:
        mu = [np.mean(treat[n][metric]) for n in names]
        sd = [np.std(treat[n][metric]) for n in names]
        ax.bar(x, mu, yerr=sd, width=0.6)
        ax.set_xticks(x, names, fontsize=8, rotation=20)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylim(0.5, 0.85)
    fig.suptitle("Accuracy cannot separate the reasoners; fidelity metrics can "
                 f"({len(cfg['seeds'])} seeds, overlap treatment)", fontsize=10)
    path = os.path.join(OUT, "F7_accuracy_vs_fidelity.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(config_path):
    cfg, cfg_hash = load_config(config_path)
    torch.manual_seed(0)
    print("treatment (overlap):")
    treat = run_treatment(cfg)
    print("control (sum structure):")
    ctrl = run_control(cfg)
    os.makedirs(OUT, exist_ok=True)

    def summarize(d):
        return {n: {m: [float(np.mean(v)), float(np.std(v))] for m, v in dd.items()}
                for n, dd in d.items()}

    payload = dict(experiment="E6-pixels", package_version=nesyarena.__version__,
                   config=cfg, config_sha256=cfg_hash,
                   treatment=summarize(treat), control=summarize(ctrl))
    res = os.path.join(OUT, "E6_pixels.json")
    with open(res, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    f7 = fig_f7(treat, cfg)
    print(f"\n{'SUT':9} {'accuracy':>16} {'calibration':>16} {'transfer':>16}")
    for n in cfg["treatment"]["suts"]:
        print(f"{n:9} {np.mean(treat[n]['acc']):>8.3f} ± {np.std(treat[n]['acc']):.3f}"
              f" {np.mean(treat[n]['cal']):>8.3f} ± {np.std(treat[n]['cal']):.3f}"
              f" {np.mean(treat[n]['trans']):>8.3f} ± {np.std(treat[n]['trans']):.3f}")
    print("\ncontrol (exact == addmult expected):")
    for n in cfg["control"]["suts"]:
        print(f"{n:9} acc {np.mean(ctrl[n]['acc']):.3f}  cal {np.mean(ctrl[n]['cal']):.3f}"
              f"  trans {np.mean(ctrl[n]['trans']):.3f}")
    print(f"wrote {res}\n      {f7}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "E6_pixels.yaml"))
    main(ap.parse_args().config)
