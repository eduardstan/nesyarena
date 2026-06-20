"""Ablation: why was top-1 harmless with pixel perception but harmful in the
fact-table setting?

Hypothesis registered BEFORE running (H-A): the difference is SUPERVISION
RICHNESS, not pixels. The fact-table run supervised a 7-query battery; the
pixel run supervised a single union query, leaving perception free to absorb
top-1's bias. Prediction: with battery supervision on pixels, top-1's harm
reappears; under single-query supervision it stays near the oracle floor;
add-mult is harmed under BOTH; exact is the floor under both.
Alternative (H-B): top-1 stays at the floor under battery too -> the
mechanism is shared perception itself, and the fact-table harm is an
artifact of the table parameterization.

Design also de-confounds top-3: structure has P=5 proofs and held-out
queries with 3 and 4 proofs, so top-3 is approximate on held-out evaluation.
"""

from __future__ import annotations
import os
import json
import numpy as np

from nesyarena.arena import overlap_family
from experiments.run_scale import (BatchStructure, MLP, sigmoid,
                                   prov_value_grad, ALPHA, PI, DIN)

OUT = os.environ.get("NESYARENA_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out"))
NCELL = 7  # s0, s1 trunk + x0..x4 private  (P=5, L=3, c=2)


def gen(B, rng, sig):
    Z = (rng.random((B, NCELL)) < PI)
    X = Z[:, :, None] * ALPHA * sig[None, None, :] + rng.standard_normal((B, NCELL, DIN))
    bayes = sigmoid(ALPHA * (X @ sig) - ALPHA ** 2 / 2 + np.log(PI / (1 - PI)))
    return X, Z, bayes


def fast_pvg(name, S: BatchStructure, Pb, k=None):
    """prov_value_grad with vectorized fast paths: k >= #proofs -> exact WMC;
    k == 1 -> argmax proof (its WMC is its score). Falls back otherwise."""
    if name.startswith("top"):
        nproofs = len(S.pidx)
        if k >= nproofs:
            return S.wmc_batch(Pb)
        if k == 1:
            sc = S.scores(Pb)
            j = np.argmax(sc, axis=1)
            val = sc[np.arange(len(j)), j]
            grad = np.zeros_like(Pb)
            for t, pi in enumerate(S.pidx):
                m = j == t
                grad[np.ix_(m, pi)] = (val[m, None] /
                                       np.clip(Pb[np.ix_(m, pi)], 1e-9, None))
            return val, grad
    return prov_value_grad(name, S, Pb, k)


def train(name, k, regime_structs, regime_truths, X, seed, epochs=12, B=64):
    N = X.shape[0]
    net = MLP(DIN, 16, 1, np.random.default_rng(4000 + seed))
    for t in range(epochs * N // B):
        ix = np.random.default_rng(t * 13 + seed).integers(0, N, B)
        p = np.clip(sigmoid(net.forward(X[ix].reshape(-1, DIN))).reshape(B, NCELL),
                    1e-4, 1 - 1e-4)
        dLdp = np.zeros_like(p)
        for S, y in zip(regime_structs, regime_truths):
            v, dvdp = fast_pvg(name, S, p, k)
            v = np.clip(v, 1e-4, 1 - 1e-4)
            dLdv = (v - y[ix]) / (v * (1 - v)) / (B * len(regime_structs))
            dLdp += dLdv[:, None] * dvdp
        net.adam_step(net.backward((dLdp * p * (1 - p)).reshape(-1, 1)), lr=2e-3)
    return net


def main(seeds=(0, 1, 2)):
    proofs, _ = overlap_family(5, 3, 2, 0.6)
    facts = sorted(set().union(*proofs))
    import itertools
    regimes = {
        "single (union only)": [tuple(range(5))],
        "battery (10 pairs)": list(itertools.combinations(range(5), 2)),
    }
    held_idx = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (0, 1, 2, 3)]  # 3- and 4-proof queries
    held = [BatchStructure([proofs[i] for i in h], facts) for h in held_idx]
    suts = [("exact", None), ("addmult", None), ("top1", 1), ("top3", 3), ("minmax", None)]
    res = {r: {n: dict(cal=[], trans=[]) for n, _ in suts} for r in regimes}

    for seed in seeds:
        rng = np.random.default_rng(700 + seed)
        sig = rng.standard_normal(DIN); sig /= np.linalg.norm(sig)
        Xtr, Ztr, _ = gen(3000, rng, sig)
        Xte, _, bay = gen(1500, rng, sig)
        bay = np.clip(bay, 1e-4, 1 - 1e-4)
        for rname, qidx in regimes.items():
            structs = [BatchStructure([proofs[i] for i in q], facts) for q in qidx]
            truths = [S.truth(Ztr).astype(float) for S in structs]
            for name, k in suts:
                net = train(name, k, structs, truths, Xtr, seed)
                pte = np.clip(sigmoid(net.forward(Xte.reshape(-1, DIN)))
                              .reshape(-1, NCELL), 1e-4, 1 - 1e-4)
                res[rname][name]["cal"].append(float(np.mean(np.abs(pte - bay))))
                te = []
                for H in held:
                    truth, _ = H.wmc_batch(bay)
                    vh, _ = fast_pvg(name, H, pte, k)
                    te.append(float(np.mean(np.abs(vh - truth))))
                res[rname][name]["trans"].append(float(np.mean(te)))
        print(f"  seed {seed} done")

    out = {r: {n: dict(cal=[float(np.mean(d["cal"])), float(np.std(d["cal"]))],
                       trans=[float(np.mean(d["trans"])), float(np.std(d["trans"]))])
               for n, d in rd.items()} for r, rd in res.items()}
    json.dump(out, open(f"{OUT}/ablation.json", "w"), indent=1)
    print(f"\n{'SUT':8} | " + " | ".join(f"{r}: cal / trans" for r in regimes))
    for n, _ in suts:
        row = " | ".join(f"{out[r][n]['cal'][0]:.3f} / {out[r][n]['trans'][0]:.3f}"
                         for r in regimes)
        print(f"{n:8} | {row}")


if __name__ == "__main__":
    main()
