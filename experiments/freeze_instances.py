"""Freeze the canonical benchmark instance set to benchmarks/instances_v1.json.

Run:  .venv/bin/python -m experiments.freeze_instances

Why this exists: every framework must be measured on the *identical* generated
programs, or cross-framework numbers are not comparable. The batteries below
replicate — rng draw for rng draw — the instance streams used by the published
conformance gates, and add the canonical recursion/probe/witness programs.
Each record carries the exact oracle value (and analytic gradient where the
battery is gradient-facing), so a conformance run needs no oracle re-derivation.

Batteries (v1):
  values     50 G1 overlap instances  (seed 1; the value-conformance battery)
  gradients  10 heterogeneous G1 instances (seed 3; tie-free, with oracle grads)
  chains     transitive-closure chains L in {2,4,6,8}, p=0.9 (single proof, depth L)
  cyclic     the canonical cyclic instance a<->b->c (depth-8 proof enumeration)
  probes     diamond DAG and diamond+back-edge (recursion-policy probes)
  witnesses  the four machine-found minimal failing configurations (E4)

The output is deterministic: same code -> byte-identical JSON. Versioned:
never edit v1 in place; add a v2 with a rationale.
"""

from __future__ import annotations

import json
import os

import numpy as np

from nesyarena.benchmarks import atom_to_str
from nesyarena.generators import chain_family, cyclic_family, overlap_family
from nesyarena.ir import Atom, transitive_closure
from nesyarena.oracle import wmc, wmc_with_grad

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "benchmarks")


def record(iid, battery, params, program, probs, query, depth,
           with_grad=False, max_proofs=100_000):
    proofs = program.proof_supports(query, depth, max_proofs)
    if with_grad:
        value, grad = wmc_with_grad(proofs, probs)
    else:
        value, grad = wmc(proofs, probs), None
    rec = dict(
        id=iid, battery=battery, params=params,
        rules=sorted([atom_to_str(r.head), sorted(atom_to_str(b) for b in r.body)]
                     for r in program.rules),
        probs={atom_to_str(a): float(p) for a, p in sorted(probs.items(), key=repr)},
        query=atom_to_str(query), depth=depth,
        proof_supports=sorted(sorted(atom_to_str(a) for a in pr) for pr in proofs),
        oracle=dict(claimed="distribution-semantics", value=float(value)),
    )
    if grad is not None:
        rec["oracle"]["grad"] = {atom_to_str(a): float(g)
                                 for a, g in sorted(grad.items(), key=repr)}
    return rec


def battery_values():
    """Replicates the 50-instance value-conformance battery (seed 1)."""
    rng = np.random.default_rng(1)
    out = []
    for i in range(50):
        P, L = int(rng.integers(2, 5)), int(rng.integers(2, 4))
        c = int(rng.integers(0, L))
        p = float(rng.choice([0.3, 0.6, 0.9]))
        het = i % 3 == 0
        inst = overlap_family(P, L, c, p, rng=rng, het=het)
        out.append(record(f"values-{i:02d}", "values",
                          dict(P=P, L=L, c=c, p=p, het=het, seed_stream=1),
                          inst.program, inst.probs, inst.query, depth=1))
    return out


def battery_gradients():
    """Replicates the 10-instance gradient battery (seed 3; heterogeneous)."""
    rng = np.random.default_rng(3)
    out = []
    for i in range(10):
        P, L = int(rng.integers(2, 5)), int(rng.integers(2, 4))
        c = int(rng.integers(0, L))
        inst = overlap_family(P, L, c, 0.6, rng=rng, het=True)
        out.append(record(f"gradients-{i:02d}", "gradients",
                          dict(P=P, L=L, c=c, het=True, seed_stream=3),
                          inst.program, inst.probs, inst.query, depth=1,
                          with_grad=True))
    return out


def battery_chains():
    out = []
    for L in (2, 4, 6, 8):
        inst = chain_family(L, 0.9)
        out.append(record(f"chain-L{L}", "chains", dict(L=L, p=0.9),
                          inst.program, inst.probs, inst.query, depth=L))
    return out


def battery_cyclic():
    inst = cyclic_family()
    return [record("cyclic-s5", "cyclic", dict(p_ab=0.9, p_ba=0.9, p_bc=0.8),
                   inst.program, inst.probs, inst.query, depth=8)]


def battery_probes():
    nodes = ["a", "b", "c", "d"]
    dag = [("a", "b"), ("b", "c"), ("a", "d"), ("d", "c")]
    cyc = dag + [("c", "a")]
    q = Atom("path", ("a", "c"))
    out = []
    for name, edges in (("probe-diamond-dag", dag), ("probe-diamond-cycle", cyc)):
        prog = transitive_closure(edges, nodes)
        probs = {Atom("edge", e): (0.9 if e == ("c", "a") else 0.6) for e in edges}
        out.append(record(name, "probes", dict(edges=[list(e) for e in edges]),
                          prog, probs, q, depth=8))
    return out


def battery_witnesses():
    configs = [("witness-addmult", 2, 1, 0, 0.6),
               ("witness-top1", 2, 1, 0, 0.3),
               ("witness-top3", 4, 1, 0, 0.3),
               ("witness-minmax", 1, 2, 0, 0.3)]
    out = []
    for iid, P, L, c, p in configs:
        inst = overlap_family(P, L, c, p)
        out.append(record(iid, "witnesses", dict(P=P, L=L, c=c, p=p),
                          inst.program, inst.probs, inst.query, depth=1))
    return out


def main():
    instances = (battery_values() + battery_gradients() + battery_chains()
                 + battery_cyclic() + battery_probes() + battery_witnesses())
    payload = dict(
        version="v1",
        description=("Frozen conformance batteries. Any new adapter must run on "
                     "exactly these instances; oracle values/gradients included. "
                     "Loader: nesyarena.benchmarks.load_instances()."),
        oracle=("brute-force weighted model counting over independent facts, "
                "cross-validated against ProbLog exact inference to <1e-10"),
        n_instances=len(instances),
        instances=instances,
    )
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "instances_v1.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    per = {}
    for r in instances:
        per[r["battery"]] = per.get(r["battery"], 0) + 1
    print(f"wrote {path}: {len(instances)} instances {per}")


if __name__ == "__main__":
    main()
