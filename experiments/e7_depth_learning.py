"""E7 — depth learning: gradient starvation end-to-end (Prop. 4 / H5).

Run:  .venv/bin/python -m experiments.e7_depth_learning

Chain of 6 cells; the query is provable only at depth 6 (single proof = all
cells open). A perception module trained through convergent reasoning learns;
trained through depth-4 truncated reasoning the query value is identically 0
below the horizon (flat — engine tests prove the finite-difference gradient
is exactly zero), so the model receives no signal and stays at chance for the
entire run. Truncated mode therefore uses the constant clipped value with no
gradient path — that IS the bounded semantics, not a simulation shortcut.

Registered numbers (RESULTS.md): convergent test AUC 0.722 ± 0.014 vs
truncated 0.490 ± 0.026 (chance). Writes out/E7_results.json and
out/F8_depth_learning.png.
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
from experiments.e6_pixels import make_mlp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))
ALPHA, DIN, L, EPS = 1.2, 64, 6, 1e-4


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(scores)
    r = np.empty_like(o, dtype=float)
    r[o] = np.arange(len(scores))
    pos = y > 0.5
    if pos.all() or (~pos).all():
        return 0.5
    return float((r[pos].mean() - (pos.sum() - 1) / 2) / (~pos).sum())


def run(seeds=(0, 1, 2), epochs=20, n_train=3000, batch=64, lr=2e-3):
    curves = {"convergent (fixed point)": [], "truncated (n=4 < depth 6)": []}
    for seed in seeds:
        rng = np.random.default_rng(500 + seed)
        sig = rng.standard_normal(DIN)
        sig /= np.linalg.norm(sig)
        Z = rng.random((n_train, L)) < 0.75
        X = Z[:, :, None] * ALPHA * sig[None, None, :] \
            + rng.standard_normal((n_train, L, DIN))
        y = Z.all(1).astype(float)
        Zte = rng.random((1200, L)) < 0.75
        Xte = Zte[:, :, None] * ALPHA * sig[None, None, :] \
            + rng.standard_normal((1200, L, DIN))
        yte = Zte.all(1).astype(float)
        Xte_t = torch.tensor(Xte.reshape(-1, DIN))
        for mode in curves:
            net = make_mlp(DIN, 16, 1, np.random.default_rng(3000 + seed))
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            hist = []
            for ep in range(epochs):
                for t in range(n_train // batch):
                    ix = np.random.default_rng(ep * 97 + t).integers(0, n_train, batch)
                    xb = torch.tensor(X[ix].reshape(-1, DIN))
                    p = torch.sigmoid(net(xb)).reshape(batch, L).clamp(EPS, 1 - EPS)
                    if mode.startswith("convergent"):
                        v = p.prod(dim=1).clamp(EPS, 1 - EPS)
                    else:
                        # bounded semantics below the horizon: value 0 (clipped),
                        # identically zero gradient — no autograd path at all
                        v = torch.full((batch,), EPS, dtype=torch.float64)
                    yb = torch.tensor(y[ix])
                    loss = -(yb * v.log() + (1 - yb) * (1 - v).log()).mean()
                    opt.zero_grad()
                    if v.requires_grad:
                        loss.backward()
                    opt.step()
                with torch.no_grad():
                    pte = torch.sigmoid(net(Xte_t)).reshape(-1, L).numpy()
                hist.append(auc(pte.prod(1), yte))
            curves[mode].append(hist)
        print(f"  [E7 seed {seed}] done")
    return curves


def main():
    curves = run()
    final = {m: [float(np.mean([h[-1] for h in cs])), float(np.std([h[-1] for h in cs]))]
             for m, cs in curves.items()}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "E7_results.json"), "w") as fh:
        json.dump(dict(experiment="E7", package_version=nesyarena.__version__,
                       final_auc=final, curves=curves), fh, indent=1, sort_keys=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    for mode in curves:
        H = np.array(curves[mode])
        ax.plot(range(1, H.shape[1] + 1), H.mean(0), "-o", ms=3, label=mode)
        ax.fill_between(range(1, H.shape[1] + 1), H.mean(0) - H.std(0),
                        H.mean(0) + H.std(0), alpha=0.15)
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test AUC (chain query, depth 6)")
    ax.set_title("Gradient starvation under truncation: the model never learns", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "F8_depth_learning.png"), dpi=160)
    plt.close(fig)
    print("E7 final AUC:", {m: f"{v[0]:.3f} ± {v[1]:.3f}" for m, v in final.items()})
    return final


if __name__ == "__main__":
    main()
