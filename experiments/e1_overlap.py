"""E1 — overlap sweep through the rebuilt core.

Run:  .venv/bin/python -m experiments.e1_overlap [--config experiments/configs/E1.yaml]

Reads the committed YAML config, sweeps the G1 grid over all reference SUTs,
and writes:
  out/E1_results.json   every (config cell, SUT) row with exact value and
                        signed error, plus the config echo and its sha256
  out/F1_error_surfaces.png   c x P signed-error heatmap per SUT
  out/F2_crossover.png        signed error vs overlap at the F2 cell

Skipped cells (fact count above the WMC limit) are logged in the output —
no silent truncation of coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import nesyarena  # noqa: E402
from nesyarena.generators import overlap_family  # noqa: E402
from nesyarena.metrics import fidelity  # noqa: E402
from nesyarena.oracle import wmc  # noqa: E402
from nesyarena.suts import registry  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))

def _registry_with_ltn():
    """Sweep runners include the LTN configurations when LTNtorch is
    installed; the core registry default stays backend-free."""
    try:
        return registry(include_ltn=True)
    except ImportError:
        return registry()



def load_config(path: str) -> tuple[dict, str]:
    with open(path, "rb") as fh:
        raw = fh.read()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def sweep(cfg: dict, suts) -> tuple[list[dict], list[dict]]:
    rows, skipped = [], []
    g = cfg["grid"]
    for p in g["p"]:
        for L in g["L"]:
            for c in range(L):
                for P in g["P"]:
                    m = c + P * (L - c)
                    if m > cfg["max_facts"]:
                        skipped.append(dict(P=P, L=L, c=c, p=p, m=m))
                        continue
                    inst = overlap_family(P, L, c, p)
                    ex = wmc(inst.proofs, inst.probs)
                    for s in suts:
                        rows.append(dict(p=p, L=L, c=c, P=P, m=m, het=False, seed=None,
                                         sut=s.name, exact=ex,
                                         err=s.value(inst.proofs, inst.probs) - ex))
    het = cfg.get("heterogeneous")
    if het:
        for seed in het["seeds"]:
            rng = np.random.default_rng(seed)
            inst = overlap_family(het["P"], het["L"], het["c"], 0.0, rng=rng, het=True)
            ex = wmc(inst.proofs, inst.probs)
            for s in suts:
                rows.append(dict(p=None, L=het["L"], c=het["c"], P=het["P"],
                                 m=len(inst.probs), het=True, seed=seed,
                                 sut=s.name, exact=ex,
                                 err=s.value(inst.proofs, inst.probs) - ex))
    return rows, skipped


def fig_f1(rows: list[dict], cfg: dict, suts) -> str:
    fc = cfg["figures"]["F1"]
    L, p = fc["L"], fc["p"]
    sel = [r for r in rows if not r["het"] and r["L"] == L and r["p"] == p]
    Ps = sorted({r["P"] for r in sel})
    cs = sorted({r["c"] for r in sel})
    names = [s.name for s in suts if s.name != "exact-wmc"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.0),
                             constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        M = np.full((len(Ps), len(cs)), np.nan)
        for r in sel:
            if r["sut"] == name:
                M[Ps.index(r["P"]), cs.index(r["c"])] = r["err"]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", aspect="auto")
        ax.set_xticks(range(len(cs)), cs)
        ax.set_yticks(range(len(Ps)), Ps)
        ax.set_xlabel("shared facts c")
        ax.set_title(name, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("proofs P")
    fig.colorbar(im, ax=axes, shrink=0.85, label="signed error")
    fig.suptitle(f"E1 signed semantic error (L={L}, p={p}); blank = > WMC fact limit",
                 fontsize=10)
    path = os.path.join(OUT, "F1_error_surfaces.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_f2(rows: list[dict], cfg: dict) -> str:
    fc = cfg["figures"]["F2"]
    sel = [r for r in rows if not r["het"] and r["L"] == fc["L"]
           and r["p"] == fc["p"] and r["P"] == fc["P"]]
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    for name, style in [("add-mult(clamped)", "o-"), ("top-1-proofs", "s-"),
                        ("top-3-proofs", "^-"), ("min-max-prob", "d-")]:
        pts = sorted([(r["c"], r["err"]) for r in sel if r["sut"] == name])
        ax.plot([c for c, _ in pts], [e for _, e in pts], style, label=name)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("shared trunk facts c (overlap)")
    ax.set_ylabel("signed semantic error")
    ax.set_title(f"Crossover: P={fc['P']}, L={fc['L']}, p={fc['p']}", fontsize=10)
    ax.legend(fontsize=8)
    path = os.path.join(OUT, "F2_crossover.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(config_path: str) -> dict:
    cfg, cfg_hash = load_config(config_path)
    suts = _registry_with_ltn()
    rows, skipped = sweep(cfg, suts)

    by_sut = {}
    grid_insts = {}
    for r in rows:
        if not r["het"]:
            grid_insts.setdefault((r["p"], r["L"], r["c"], r["P"]), None)
    instances = [overlap_family(P, L, c, p) for (p, L, c, P) in grid_insts]
    for s in suts:
        by_sut[s.name] = dict(phi_overlap=fidelity(s, instances),
                              mean_abs_err=float(np.mean(
                                  [abs(r["err"]) for r in rows if r["sut"] == s.name])))

    os.makedirs(OUT, exist_ok=True)
    payload = dict(experiment="E1", package_version=nesyarena.__version__,
                   config=cfg, config_sha256=cfg_hash,
                   n_rows=len(rows), skipped_cells=skipped,
                   scorecard=by_sut, rows=rows)
    res_path = os.path.join(OUT, "E1_results.json")
    with open(res_path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)

    f1 = fig_f1(rows, cfg, suts)
    f2 = fig_f2(rows, cfg)
    print(f"E1: {len(rows)} rows ({len(skipped)} cells skipped over the fact limit)")
    print("phi(overlap) per SUT:")
    for name, v in by_sut.items():
        print(f"  {name:22} phi={v['phi_overlap']:.4f}  mean|err|={v['mean_abs_err']:.4f}")
    print(f"wrote {res_path}\n      {f1}\n      {f2}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "E1.yaml"))
    main(ap.parse_args().config)
