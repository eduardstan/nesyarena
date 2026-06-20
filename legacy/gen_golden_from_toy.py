"""Generate golden parity fixtures from the Day-0 toy implementation.

Run from the directory that contains the toy `nesyarena/` package (repo root
before the move to legacy/, `legacy/` after):

    .venv/bin/python scripts/gen_golden_from_toy.py [out.json]

The rebuild's test suite replays these fixtures against the new code; any
deviation is a parity failure. Fixtures are deliberately generated in a
separate process so the toy and the rebuilt package (same import name) never
coexist in one interpreter.
"""

from __future__ import annotations
import json
import sys

import numpy as np

from nesyarena.oracle import wmc, wmc_with_grad, graph_value
from nesyarena.provenances import ExactWMC, AddMult, TopK, MinMax, LSE
from nesyarena.arena import (iterate, ground_tc, overlap_family, chain,
                             find_witness)

OUT = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/toy_golden.json"
G: dict = {"comment": "golden values from Day-0 toy (commit af40af8 lineage)"}


def ser_proofs(proofs):
    return [sorted(pr) for pr in proofs]


# ---- G1 instances: deterministic homogeneous grid + seeded heterogeneous ----
instances = []
for (P, L, c, p) in [(1, 2, 0, 0.3), (2, 1, 0, 0.6), (2, 3, 1, 0.6),
                     (3, 2, 1, 0.9), (4, 4, 0, 0.6), (4, 4, 2, 0.6),
                     (4, 4, 3, 0.6), (5, 3, 2, 0.9), (6, 2, 1, 0.3),
                     (8, 2, 1, 0.6)]:
    proofs, probs = overlap_family(P, L, c, p)
    instances.append(dict(P=P, L=L, c=c, p=p, het=False, seed=None,
                          proofs=ser_proofs(proofs), probs=probs))
for seed in (1, 2, 3, 4, 5):
    rng = np.random.default_rng(seed)
    proofs, probs = overlap_family(4, 3, 1, 0.6, rng=rng, het=True)
    instances.append(dict(P=4, L=3, c=1, p=0.6, het=True, seed=seed,
                          proofs=ser_proofs(proofs), probs=probs))

suts = [ExactWMC(), AddMult(clamp=True), AddMult(clamp=False),
        TopK(1), TopK(3), MinMax()]
for inst in instances:
    proofs = [frozenset(pr) for pr in inst["proofs"]]
    probs = inst["probs"]
    val, grad = wmc_with_grad(proofs, probs)
    inst["oracle"] = dict(value=val, grad=grad)
    inst["suts"] = {}
    for s in suts:
        inst["suts"][s.name] = dict(value=s.value(proofs, probs),
                                    grad=s.grad(proofs, probs))
G["g1_instances"] = instances

# ---- LSE surrogate values (claimed semantics: max-join) ----------------------
lse_cases = []
for tau in (0.005, 0.02, 0.1, 0.4):
    for (P, L, c, p) in [(2, 2, 0, 0.6), (4, 3, 1, 0.6), (6, 2, 1, 0.9)]:
        proofs, probs = overlap_family(P, L, c, p)
        s = LSE(tau)
        lse_cases.append(dict(tau=tau, P=P, L=L, c=c, p=p,
                              value=s.value(proofs, probs),
                              oracle=s.oracle(proofs, probs),
                              grad=s.grad(proofs, probs)))
G["lse_cases"] = lse_cases

# ---- engine: cyclic program (a<->b->c), four algebras ------------------------
nodes = ["a", "b", "c"]
edges = [("a", "b"), ("b", "a"), ("b", "c")]
rules = ground_tc(edges, nodes)
f0_prob = {("edge", "a", "b"): 0.9, ("edge", "b", "a"): 0.9, ("edge", "b", "c"): 0.8}
algebras = dict(
    boolean=dict(f0={k: 1.0 for k in f0_prob}, oplus=max, otimes=min, zero=0.0, one=1.0),
    maxprod=dict(f0=f0_prob, oplus=max, otimes=float.__mul__, zero=0.0, one=1.0),
    sumprod=dict(f0=f0_prob, oplus=float.__add__, otimes=float.__mul__, zero=0.0, one=1.0),
)
cyc = {}
for name, a in algebras.items():
    hist = iterate(rules, a["f0"], a["oplus"], a["otimes"], a["zero"], a["one"], 20)
    cyc[name] = [h.get(("path", "a", "c"), a["zero"]) for h in hist]
G["cyclic_path_ac"] = cyc
G["graph_oracles"] = dict(
    boolean=graph_value("boolean", edges, {e: 1.0 for e in edges}, "a", "c"),
    tropical=graph_value("tropical", edges, {e: 1.0 for e in edges}, "a", "c"),
    maxprod=graph_value("maxprod", edges, f0_prob_e := {e: f0_prob[("edge",) + e] for e in edges}, "a", "c"),
)

# ---- engine: chain truncation (L=8, p=0.9), maxprod values at each n --------
nodes8, edges8, probs8 = chain(8, 0.9)
rules8 = ground_tc(edges8, nodes8)
f08 = {("edge", u, v): probs8[(u, v)] for (u, v) in edges8}
hist8 = iterate(rules8, f08, max, float.__mul__, 0.0, 1.0, 10)
G["chain8_maxprod_path_v0_v8"] = [h.get(("path", "v0", "v8"), 0.0) for h in hist8]

# ---- witnesses ---------------------------------------------------------------
G["witnesses"] = {s.name: find_witness(s) for s in
                  [AddMult(clamp=True), TopK(1), TopK(3), MinMax()]}

with open(OUT, "w") as fh:
    json.dump(G, fh, indent=1, sort_keys=True)
print(f"wrote {OUT}: {len(instances)} G1 instances, {len(lse_cases)} LSE cases, "
      f"{len(G['witnesses'])} witnesses")
