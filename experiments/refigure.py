"""Rebuild every figure from the committed out/*.json — no retraining.

Run:  .venv/bin/python -m experiments.refigure

Each experiment dumps its results JSON and draws its figure(s) from that
same data, so the figures (including the shared palette in
experiments/palette.py) regenerate from the JSONs alone, without
re-running any experiment. A figure is skipped when its JSON is absent.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))


def _load(name: str):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    made, skipped = [], []

    if (d := _load("E1_results.json")) is not None:
        from experiments import e1_overlap
        names = list(dict.fromkeys(r["sut"] for r in d["rows"]))
        suts = [SimpleNamespace(name=n) for n in names]
        e1_overlap.fig_f1(d["rows"], d["config"], suts)
        e1_overlap.fig_f2(d["rows"], d["config"])
        made += ["F1_error_surfaces.png", "F2_crossover.png"]
    else:
        skipped.append("E1")

    if (d := _load("E2_results.json")) is not None:
        from experiments import e2_depth
        hzn = {int(k): v for k, v in d["horizons"].items()}
        e2_depth.fig_f3(d["rows"], hzn, d["config"])
        made.append("F3_depth_horizon.png")
    else:
        skipped.append("E2")

    if (d := _load("E3_results.json")) is not None:
        from experiments import e3_surrogate
        e3_surrogate.fig_f4(d["rows"], d["config"])
        made.append("F4_surrogate.png")
    else:
        skipped.append("E3")

    if (d := _load("E6_facttable.json")) is not None:
        from experiments import e6_facttable
        e6_facttable.fig_f6(d["treatment"], d["control"], d["config"]["suts"])
        made.append("F6_learning_transfer.png")
    else:
        skipped.append("E6-facttable")

    if (d := _load("E6_pixels.json")) is not None:
        from experiments import e6_pixels
        e6_pixels.fig_f7(d["treatment"], d["config"])
        made.append("F7_accuracy_vs_fidelity.png")
    else:
        skipped.append("E6-pixels")

    if (d := _load("E7_results.json")) is not None:
        from experiments import e7_depth_learning
        e7_depth_learning.fig_f8(d["curves"])
        made.append("F8_depth_learning.png")
    else:
        skipped.append("E7")

    if (d := _load("E5_mnist.json")) is not None:
        from experiments import e5_mnist
        e5_mnist.fig_f9(d["treatment"], d["control"], d["config"])
        made.append("F9_mnist.png")
    else:
        skipped.append("E5-MNIST")

    if (d := _load("E5b_noise.json")) is not None:
        from experiments import e5b_noise_ablation
        e5b_noise_ablation.fig_f10(d["summary"])
        made.append("F10_noise_ablation.png")
    else:
        skipped.append("E5b")

    if (d := _load("E8_results.json")) is not None:
        from experiments import e8_clutrr
        e8_clutrr.fig_f11(d["accuracy"], d["config"])
        made.append("F11_clutrr.png")
    else:
        skipped.append("E8")

    print("refigured:", ", ".join(made))
    if skipped:
        print("skipped (no JSON):", ", ".join(skipped))


if __name__ == "__main__":
    main()
