"""Conformance of DeepProbLog (standalone, exact engine) on the frozen set.

Run:  .venv/bin/python -m experiments.conformance_deepproblog

Registered prediction: the ExactEngine (ProbLog grounding + SDD compilation)
matches the oracle at compilation precision on every battery, recursion
included. Per-instance values are stored so the arena can consume them.
Writes out/conformance_deepproblog.{json,md}.
"""

from __future__ import annotations

import json
import os

import nesyarena
from nesyarena.adapters.deepproblog_standalone import DeepProbLogStandaloneAdapter
from nesyarena.benchmarks import load_instances

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))


def main():
    ad = DeepProbLogStandaloneAdapter()
    rows, per_battery = [], {}
    for inst in load_instances():
        v = ad.infer(inst.program, inst.probs, [inst.query])[inst.query]
        dev = abs(v - inst.oracle_value)
        rows.append(dict(id=inst.id, battery=inst.battery, value=v,
                         oracle=inst.oracle_value, abs_dev=dev))
        per_battery[inst.battery] = max(per_battery.get(inst.battery, 0.0), dev)
    worst = max(per_battery.values())
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "conformance_deepproblog.json"), "w") as fh:
        json.dump(dict(experiment="conformance-deepproblog",
                       package_version=nesyarena.__version__,
                       deepproblog_version="2.0.6", engine="ExactEngine",
                       max_abs_dev_per_battery=per_battery, rows=rows),
                  fh, indent=1, sort_keys=True)
    md = [
        "# Conformance — DeepProbLog standalone (deepproblog 2.0.6, ExactEngine)",
        "",
        "The original NeurIPS-2018 system, run on the frozen instance set v1",
        "(all 71 instances, every battery incl. the recursive ones).",
        "Registered prediction: exact at compilation precision — **verdict:",
        f"{'PASS' if worst < 1e-7 else 'DEVIATION (a finding)'}**, max |value − oracle| "
        f"= {worst:.2e} overall.",
        "",
        "| battery | max abs dev |", "|---|---|",
    ]
    md += [f"| {b} | {d:.2e} |" for b, d in sorted(per_battery.items())]
    md += [
        "",
        "Scope: the ApproximateEngine (DPLA*) requires SWI-Prolog/PySwip and is",
        "not yet measured — queued as the natural follow-up. Gradients:",
        "constant-probability programs expose no differentiable path in this",
        "system (learning flows through neural predicates only).",
    ]
    with open(os.path.join(OUT, "conformance_deepproblog.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[2:12]))


if __name__ == "__main__":
    main()
