"""E5b — label-noise ablation on the MNIST-sum control (registered).

Run:  .venv/bin/python -m experiments.e5b_noise_ablation

Context. At eta = 0 (E5) we measured top-1 *beating* exact on control
transfer (0.019 vs 0.221). Working hypothesis H-N: with near-deterministic
perception posteriors, top-1's sharpening bias acts as a beneficial inductive
bias, while exact aggregation leaves single-query-supervised perception
under-determined; the synthetic control (diffuse posteriors) showed the
opposite ranking, so the regime variable is posterior determinism.

Design. Inject latent label noise eta: each cell's latent class z equals the
digit label with prob 1-eta, else uniform over the 3 classes. Because we
inject the only stochasticity, the true posterior given the image is closed
form — p_true(c) = (1-eta)*1[c=label] + eta/3 — restoring exact calibration
and transfer ground truth on real digits. Training: BCE on labels sampled
from the noisy latents, single sum-query supervision, identical nets/budgets
to E5's control.

REGISTERED PREDICTIONS (before running):
  P1: top-1's transfer advantage over exact, adv(eta) = trans_exact -
      trans_top1, is positive at eta = 0 and decreases monotonically in eta.
  P2: by eta = 0.5 the ranking reverses (trans_top1 >= trans_exact), the
      diffuse regime of the synthetic control.
  P3 (sanity anchor): eta = 0 reproduces E5's measured 0.221 / 0.019.

Writes out/E5b_noise.json and out/F10_noise_ablation.png.
"""

from __future__ import annotations

import json
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import nesyarena  # noqa: E402
from experiments.e5_mnist import EPS, bce, cat_value, make_net, mnist_arrays  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))

DIGITS = [0, 1, 2]
TERMS_TR = [(0, 2), (1, 1), (2, 0)]
TERMS_HE = [(0, 0), (1, 1), (2, 2)]
SUTS = ["exact", "top1", "minmax", "ltn_product", "ltn_godel"]
ETAS = [0.0, 0.15, 0.3, 0.5]


def noisy_latents(labels, eta, rng):
    z = labels.copy()
    flip = rng.random(len(z)) < eta
    z[flip] = rng.integers(0, 3, flip.sum())
    return z


def true_posterior(labels, eta):
    """(N, 3): (1-eta)*onehot(label) + eta/3 — exact by construction."""
    post = np.full((len(labels), 3), eta / 3.0)
    post[np.arange(len(labels)), labels] += 1.0 - eta
    return post


def run(seeds=(0, 1, 2), epochs=6, n_train=3000, n_test=1500, batch=64, lr=1e-3):
    (Xtr_all, ytr_all), (Xte_all, yte_all) = mnist_arrays()
    trm = np.isin(ytr_all, DIGITS)
    tem = np.isin(yte_all, DIGITS)
    Xd, yd = Xtr_all[trm], ytr_all[trm]
    Xdt, ydt = Xte_all[tem], yte_all[tem]
    res = {f"{n}@{eta}": dict(trans=[], cal=[]) for n in SUTS for eta in ETAS}
    for eta in ETAS:
        for seed in seeds:
            rng = np.random.default_rng(700 + seed)  # same image draws as E5
            iA = rng.integers(0, len(Xd), n_train)
            iB = rng.integers(0, len(Xd), n_train)
            iAte = rng.integers(0, len(Xdt), n_test)
            iBte = rng.integers(0, len(Xdt), n_test)
            nrng = np.random.default_rng(9000 + seed)
            zA = noisy_latents(yd[iA], eta, nrng)
            zB = noisy_latents(yd[iB], eta, nrng)
            ytr = (zA + zB == 2).astype(float)
            postA = true_posterior(ydt[iAte], eta)
            postB = true_posterior(ydt[iBte], eta)
            truth_he = (postA * postB).sum(1)  # P(zA == zB | images)
            XA_t, XB_t = torch.tensor(Xdt[iAte]), torch.tensor(Xdt[iBte])
            for name in SUTS:
                net = make_net(128, 3, 2000 + seed)
                opt = torch.optim.Adam(net.parameters(), lr=lr)
                for t in range(epochs * n_train // batch):
                    ix = np.random.default_rng(t * 17 + seed).integers(0, n_train, batch)
                    PA = torch.softmax(net(torch.tensor(Xd[iA[ix]])), 1).clamp(EPS, 1.0)
                    PB = torch.softmax(net(torch.tensor(Xd[iB[ix]])), 1).clamp(EPS, 1.0)
                    loss = bce(cat_value(name, TERMS_TR, PA, PB), torch.tensor(ytr[ix]))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    PA = torch.softmax(net(XA_t), 1).clamp(EPS, 1.0).numpy()
                    PB = torch.softmax(net(XB_t), 1).clamp(EPS, 1.0).numpy()
                    vals = np.stack([PA[:, i] * PB[:, j] for (i, j) in TERMS_HE], 1)
                    # vh = (vals.max(1) if name == "top1"
                    #       else np.stack([np.minimum(PA[:, i], PB[:, j])
                    #                      for (i, j) in TERMS_HE], 1).max(1)
                    #       if name == "minmax" else vals.sum(1))
                    if name == "top1":
                        vh = vals.max(1)
                    elif name in ("minmax", "ltn_godel"):
                        vh = np.stack([np.minimum(PA[:, i], PB[:, j])
                                       for (i, j) in TERMS_HE], 1).max(1)
                    elif name == "ltn_product":
                        vh = vals[:, 0]
                        for j in range(1, vals.shape[1]):
                            vh = vh + vals[:, j] - vh * vals[:, j]
                    else:
                        vh = vals.sum(1)

                    res[f"{name}@{eta}"]["trans"].append(
                        float(np.mean(np.abs(vh - truth_he))))
                    res[f"{name}@{eta}"]["cal"].append(
                        float((np.abs(PA - postA).mean() + np.abs(PB - postB).mean()) / 2))
        print(f"  [eta={eta}] done")
    return res


def main():
    res = run()
    summary = {k: dict(trans=[float(np.mean(v["trans"])), float(np.std(v["trans"]))],
                       cal=[float(np.mean(v["cal"])), float(np.std(v["cal"]))])
               for k, v in res.items()}
    adv = {eta: summary[f"exact@{eta}"]["trans"][0] - summary[f"top1@{eta}"]["trans"][0]
           for eta in ETAS}
    verdict = dict(
        P1_monotone_decrease=all(adv[a] >= adv[b] - 1e-9
                                 for a, b in zip(ETAS[:-1], ETAS[1:], strict=True)),
        P2_reversal_at_05=adv[0.5] <= 0.0,
        advantage_curve=adv)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "E5b_noise.json"), "w") as fh:
        json.dump(dict(experiment="E5b", package_version=nesyarena.__version__,
                       registered_predictions=["P1 monotone decrease",
                                               "P2 reversal at eta=0.5",
                                               "P3 anchor at eta=0"],
                       verdict=verdict, summary=summary), fh, indent=1, sort_keys=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.4), constrained_layout=True)
    for n in SUTS:
        ax1.errorbar(ETAS, [summary[f"{n}@{e}"]["trans"][0] for e in ETAS],
                     yerr=[summary[f"{n}@{e}"]["trans"][1] for e in ETAS],
                     fmt="o-", ms=4, label=n)
        ax2.errorbar(ETAS, [summary[f"{n}@{e}"]["cal"][0] for e in ETAS],
                     yerr=[summary[f"{n}@{e}"]["cal"][1] for e in ETAS],
                     fmt="o-", ms=4, label=n)
    ax1.set_xlabel(r"latent label noise $\eta$")
    ax1.set_ylabel("transfer error vs closed-form truth")
    ax1.set_title("Transfer vs posterior determinism", fontsize=9)
    ax2.set_xlabel(r"latent label noise $\eta$")
    ax2.set_ylabel("percept calibration vs true posterior")
    ax2.set_title("Calibration", fontsize=9)
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "F10_noise_ablation.png"), dpi=160)
    plt.close(fig)
    print("advantage(exact - top1) by eta:", {k: round(v, 3) for k, v in adv.items()})
    print("P1 monotone decrease:", verdict["P1_monotone_decrease"],
          "| P2 reversal at 0.5:", verdict["P2_reversal_at_05"])
    for eta in ETAS:
        row = " | ".join(f"{n}: {summary[f'{n}@{eta}']['trans'][0]:.3f}" for n in SUTS)
        print(f"  eta={eta}: {row}")
    return verdict


if __name__ == "__main__":
    main()
