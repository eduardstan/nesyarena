"""Frozen benchmark instance sets (cross-framework consistency).

Every new adapter must be evaluated on the *identical* generated programs the
published conformance results used — not on freshly sampled ones. To make that
portable and inspectable, `experiments/freeze_instances.py` serializes the
canonical batteries (values, gradients, recursion, probes, witnesses) to
`benchmarks/instances_v1.json`, including the exact oracle value (and, where
relevant, the analytic oracle gradient) for each instance.

This module is the loader: it parses the JSON back into `GroundProgram`s,
`Atom`-keyed probability maps and proof supports, so a conformance run against
any backend is:

    for inst in load_instances():
        got = adapter.infer(inst.program, inst.probs, [inst.query])[inst.query]
        err = got - inst.oracle_value          # signed semantic error

The file is versioned; never mutate an existing version in place — add
instances_v2.json and say why.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .ir import Atom, GroundProgram, Rule

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "benchmarks", "instances_v1.json")


def atom_to_str(a: Atom) -> str:
    return repr(a)


def parse_atom(s: str) -> Atom:
    s = s.strip()
    if "(" not in s:
        return Atom(s)
    pred, rest = s.split("(", 1)
    assert rest.endswith(")"), f"malformed atom string: {s!r}"
    args = tuple(x.strip() for x in rest[:-1].split(","))
    return Atom(pred, args)


@dataclass
class BenchInstance:
    id: str
    battery: str
    params: dict
    program: GroundProgram
    probs: dict            # Atom -> float
    query: Atom
    depth: int             # enumeration depth used for the stored proofs
    proofs: list           # list[frozenset[Atom]]
    oracle_value: float
    oracle_grad: dict = field(default_factory=dict)   # Atom -> float (may be empty)


def load_instances(path: str | None = None,
                   battery: str | None = None) -> list[BenchInstance]:
    with open(path or DEFAULT_PATH) as fh:
        data = json.load(fh)
    out = []
    for rec in data["instances"]:
        if battery is not None and rec["battery"] != battery:
            continue
        rules = tuple(Rule(parse_atom(h), tuple(parse_atom(b) for b in body))
                      for h, body in rec["rules"])
        out.append(BenchInstance(
            id=rec["id"],
            battery=rec["battery"],
            params=rec["params"],
            program=GroundProgram(rules),
            probs={parse_atom(k): v for k, v in rec["probs"].items()},
            query=parse_atom(rec["query"]),
            depth=rec["depth"],
            proofs=[frozenset(parse_atom(a) for a in pr) for pr in rec["proof_supports"]],
            oracle_value=rec["oracle"]["value"],
            oracle_grad={parse_atom(k): v
                         for k, v in rec["oracle"].get("grad", {}).items()},
        ))
    return out
