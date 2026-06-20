"""E2 — depth sweep: truncation semantics of bounded unrolling (Prop. 4 / H5)
and recursion stress (H7).

Run:  .venv/bin/python -m experiments.e2_depth [--config experiments/configs/E2.yaml]

Writes out/E2_results.json and out/F3_depth_horizon.png:
  - value error of n-step unrolling vs the convergent fixpoint on chains,
  - measured depth horizons h_delta(n) against the theoretical n+1,
  - gradient starvation (finite-difference liveness) below the horizon,
  - the cyclic H7 check (sum-product 'probability' leaving [0,1]).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import nesyarena  # noqa: E402
from nesyarena.algebra import MAXPROD, SUMPROD  # noqa: E402
from nesyarena.engine import converge, infer_bounded  # noqa: E402
from nesyarena.generators import chain_family, cyclic_family  # noqa: E402
from nesyarena.metrics import depth_horizon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))


def load_config(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def fd_live_fraction(inst, n, eps) -> float:
    """Fraction of edges whose finite-difference gradient through the
    n-step unroller is non-zero (D4 at the value level)."""
    base = infer_bounded(inst.program, dict(inst.probs), MAXPROD, inst.query, n)
    live = 0
    for a in inst.probs:
        bumped = dict(inst.probs)
        bumped[a] = min(1.0, bumped[a] + eps)
        if abs(infer_bounded(inst.program, bumped, MAXPROD, inst.query, n) - base) > 1e-12:
            live += 1
    return live / len(inst.probs)


def sweep(cfg):
    rows = []
    for p in cfg["chains"]["p"]:
        insts = {L: chain_family(L, p) for L in cfg["chains"]["L"]}
        conv = {L: converge(i.program, dict(i.probs), MAXPROD, i.query)
                for L, i in insts.items()}
        for n in cfg["chains"]["unroll_n"]:
            for L, inst in insts.items():
                val = infer_bounded(inst.program, dict(inst.probs), MAXPROD, inst.query, n)
                rows.append(dict(p=p, L=L, n=n, value=val, converged=conv[L],
                                 err=val - conv[L],
                                 fd_live=fd_live_fraction(inst, n, cfg["starvation_fd_eps"])
                                 if L in (n, n + 1) or n == 0 else None))
    return rows


def horizons(cfg):
    out = {}
    for n in cfg["chains"]["unroll_n"]:
        if n == 0:
            continue

        def err(L, n=n):
            inst = chain_family(L, 0.9)
            return (infer_bounded(inst.program, dict(inst.probs), MAXPROD, inst.query, n)
                    - converge(inst.program, dict(inst.probs), MAXPROD, inst.query))

        out[n] = depth_horizon(err, delta=cfg["horizon_delta"],
                               max_depth=max(cfg["chains"]["L"]))
    return out


def cyclic_h7():
    inst = cyclic_family()
    sp = converge(inst.program, dict(inst.probs), SUMPROD, inst.query,
                  tol=1e-9, max_steps=400)
    mp = converge(inst.program, dict(inst.probs), MAXPROD, inst.query)
    return dict(sumprod=sp, maxprod=mp, leaves_unit_interval=sp > 1.0)


def fig_f3(rows, hzn, cfg):
    fc = cfg["figures"]["F3"]
    sel = [r for r in rows if r["p"] == fc["p"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2), constrained_layout=True)
    Ls = sorted({r["L"] for r in sel})
    conv = [next(r["converged"] for r in sel if r["L"] == L) for L in Ls]
    ax1.plot(Ls, conv, "k--", label="convergent")
    for n in fc["show_n"]:
        vals = [next(r["value"] for r in sel if r["L"] == L and r["n"] == n) for L in Ls]
        ax1.plot(Ls, vals, "o-", ms=3, label=f"unroll n={n}")
    ax1.set_xlabel("required proof depth L")
    ax1.set_ylabel(f"path value (p={fc['p']})")
    ax1.set_title("Truncation: value falls to 0 past the horizon", fontsize=9)
    ax1.legend(fontsize=7)
    ns = sorted(hzn)
    ax2.plot(ns, [hzn[n] for n in ns], "o", label="measured $h_\\delta$")
    ax2.plot(ns, [n + 1 for n in ns], "k--", label="theory $n+1$")
    ax2.set_xlabel("unrolling depth n")
    ax2.set_ylabel("depth horizon")
    ax2.set_title("Horizon law (Thm. 1 truncation)", fontsize=9)
    ax2.legend(fontsize=8)
    path = os.path.join(OUT, "F3_depth_horizon.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(config_path):
    cfg, cfg_hash = load_config(config_path)
    rows = sweep(cfg)
    hzn = horizons(cfg)
    h7 = cyclic_h7() if cfg["cyclic"]["check_sumprod_divergence"] else None

    os.makedirs(OUT, exist_ok=True)
    payload = dict(experiment="E2", package_version=nesyarena.__version__,
                   config=cfg, config_sha256=cfg_hash,
                   horizons=hzn, cyclic_h7=h7, rows=rows)
    res = os.path.join(OUT, "E2_results.json")
    with open(res, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    f3 = fig_f3(rows, hzn, cfg)

    print("E2 horizons (n -> h_delta):", hzn, "(theory: n+1)")
    if h7:
        print(f"H7 cyclic: sumprod -> {h7['sumprod']:.4f} (>1: {h7['leaves_unit_interval']}), "
              f"maxprod -> {h7['maxprod']:.4f}")
    starved = [r for r in rows if r["fd_live"] == 0.0 and r["L"] == r["n"] + 1]
    print(f"starvation: {len(starved)} (n, L=n+1) cells with zero FD-liveness")
    print(f"wrote {res}\n      {f3}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "E2.yaml"))
    main(ap.parse_args().config)
