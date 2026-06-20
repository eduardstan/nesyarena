"""E4 — witness synthesis over the SUT registry (the paper's witness table).

Run:  .venv/bin/python -m experiments.e4_witnesses

For each reference SUT, the smallest G1 configuration with |semantic error|
above delta, greedily shrunk (D6). Writes out/E4_witness_table.{json,md}.
"""

from __future__ import annotations

import json
import os

import nesyarena
from nesyarena.suts import registry
from nesyarena.witness import find_witness

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("NESYARENA_OUT", os.path.join(HERE, "..", "out"))
DELTA = 0.05


def main():
    table = {}
    for sut in registry():
        w = find_witness(sut, delta=DELTA)
        table[sut.name] = w
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "E4_witness_table.json"), "w") as fh:
        json.dump(dict(experiment="E4", package_version=nesyarena.__version__,
                       delta=DELTA, witnesses=table), fh, indent=1, sort_keys=True)
    lines = ["# E4 — minimal witnesses (machine-found, shrunk; delta = %.2f)" % DELTA, "",
             "| SUT | P | L | c | p | facts m | signed error |",
             "|---|---|---|---|---|---|---|"]
    for name, w in table.items():
        if w is None:
            lines.append(f"| {name} | — | — | — | — | — | none on the grid |")
        else:
            lines.append(f"| {name} | {w['P']} | {w['L']} | {w['c']} | {w['p']} "
                         f"| {w['m']} | {w['err']:+.3f} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "E4_witness_table.md"), "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
