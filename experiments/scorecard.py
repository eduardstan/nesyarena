"""Scorecard — measured fidelity profiles (the radar, protocol D2).

Run:  .venv/bin/python -m experiments.scorecard
      (requires out/E1_results.json, out/E6_facttable.json, out/E6_pixels.json
       — run e1_overlap, e6_facttable, e6_pixels first)

Axes (each in [0, 1], 1 = faithful):
  multiplicity        1 - mean|err| on E1 cells with c=0, p=0.6
  overlap             1 - mean|err| on E1 cells with P=4, p=0.6
  saturation          1 - mean|err| on E1 cells with p=0.9, P>=3
  gradient liveness   mean share of oracle-live facts kept alive (G1, L=3 grid)
  learning transfer   1 - E6 fact-table treatment transfer error
  pixel transfer eff. oracle-floor efficiency from E6 pixels (floor/err)

Writes out/scorecard.json and out/scorecard_radar.png.
"""

from __future__ import annotations

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import nesyarena  # noqa: E402
from nesyarena.generators import overlap_family  # noqa: E402
from nesyarena.metrics import gradient_liveness  # noqa: E402
from nesyarena.oracle import wmc_with_grad  # noqa: E402
from nesyarena.suts import registry  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))

# E6 result keys per SUT name used in the learning experiments
E6_KEY = {"exact-wmc": "exact", "add-mult(clamped)": "addmult",
          "top-1-proofs": "top1", "top-3-proofs": "top3", "min-max-prob": "minmax"}


def liveness_battery() -> dict[str, float]:
    suts = registry()
    cover = {s.name: [] for s in suts}
    for p in (0.3, 0.6, 0.9):
        for c in (0, 1, 2):
            for P in range(1, 7):
                if c + P * (3 - c) > 22:
                    continue
                inst = overlap_family(P, 3, c, p)
                _, og = wmc_with_grad(inst.proofs, inst.probs)
                for s in suts:
                    cover[s.name].append(
                        gradient_liveness(s.grad(inst.proofs, inst.probs), og, tol=1e-12))
    return {n: float(np.mean(v)) for n, v in cover.items()}


def e1_axis(rows, name, sel) -> float:
    errs = [abs(r["err"]) for r in rows if r["sut"] == name and sel(r)]
    return max(0.0, 1.0 - float(np.mean(errs))) if errs else 1.0


def main():
    e1 = json.load(open(os.path.join(OUT, "E1_results.json")))["rows"]
    e6f = json.load(open(os.path.join(OUT, "E6_facttable.json")))["treatment"]
    e6p = json.load(open(os.path.join(OUT, "E6_pixels.json")))["treatment"]
    live = liveness_battery()
    floor = e6p["exact"]["trans"][0]

    names = [s.name for s in registry()]
    profiles = {}
    for n in names:
        k = E6_KEY[n]
        profiles[n] = {
            "multiplicity (c=0, p=0.6)":
                e1_axis(e1, n, lambda r: not r["het"] and r["c"] == 0 and r["p"] == 0.6),
            "overlap (P=4, p=0.6)":
                e1_axis(e1, n, lambda r: not r["het"] and r["P"] == 4 and r["p"] == 0.6),
            "saturation (p=0.9, P>=3)":
                e1_axis(e1, n, lambda r: not r["het"] and r["p"] == 0.9 and r["P"] >= 3),
            "gradient liveness": live[n],
            "learning transfer (fact table)": max(0.0, 1.0 - e6f[k]["trans"][0]),
            "pixel transfer efficiency": min(1.0, floor / max(e6p[k]["trans"][0], 1e-9)),
        }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "scorecard.json"), "w") as fh:
        json.dump(dict(package_version=nesyarena.__version__,
                       gradient_liveness=live, profiles=profiles),
                  fh, indent=1, sort_keys=True)

    labels = list(next(iter(profiles.values())).keys())
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(7.0, 5.8), subplot_kw=dict(polar=True))
    for n in names:
        vals = [profiles[n][a] for a in labels]
        ax.plot(ang + ang[:1], vals + vals[:1], "o-", lw=1.6, ms=3, label=n)
        ax.fill(ang + ang[:1], vals + vals[:1], alpha=0.06)
    ax.set_xticks(ang)
    ax.set_xticklabels([a.replace(" (", "\n(") for a in labels], fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_title("NeSyArena scorecard — measured fidelity profiles", pad=18, fontsize=11)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.12), fontsize=7)
    fig.savefig(os.path.join(OUT, "scorecard_radar.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("gradient liveness:", {n: round(v, 3) for n, v in live.items()})
    print("wrote", os.path.join(OUT, "scorecard.json"), "and scorecard_radar.png")
    return profiles


if __name__ == "__main__":
    main()
