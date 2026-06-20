"""NeSyArena CI battery (protocol section 10: 'CI asserting oracle agreement').

Run:  python3 -m tests.run_ci     -> exits non-zero on any failure.
Covers: oracle vs ProbLog, analytic vs finite-difference gradients, the four
error laws (Propositions 1-4), Theorem-1 numeric correspondence on a cyclic
program with a non-idempotent semiring, and gradient-liveness sanity.
"""

from __future__ import annotations
import sys
import numpy as np

from nesyarena.oracle import wmc, wmc_with_grad
from nesyarena.provenances import AddMult, TopK, MinMax, LSE
from nesyarena.arena import overlap_family, iterate, ground_tc

FAIL = []


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAIL.append(name)


def main():
    rng = np.random.default_rng(7)

    # 1. oracle vs ProbLog
    try:
        from nesyarena.arena import problog_value
        d = 0.0
        for i in range(10):
            proofs, probs = overlap_family(int(rng.integers(2, 5)), 3,
                                           int(rng.integers(0, 3)), 0.6,
                                           rng=rng, het=(i % 2 == 0))
            d = max(d, abs(wmc(proofs, probs) - problog_value(proofs, probs)))
        check(f"oracle == ProbLog on 10 instances (max diff {d:.1e})", d < 1e-10)
    except ImportError:
        print("  [SKIP] ProbLog not installed")

    # 2. analytic WMC gradient vs finite differences
    proofs, probs = overlap_family(3, 3, 1, 0.6, rng=rng, het=True)
    _, g = wmc_with_grad(proofs, probs)
    f0 = sorted(probs)[0]
    h = 1e-6
    pp, pm = dict(probs), dict(probs)
    pp[f0] += h; pm[f0] -= h
    fd = (wmc(proofs, pp) - wmc(proofs, pm)) / (2 * h)
    check(f"analytic grad == finite diff ({abs(g[f0]-fd):.1e})", abs(g[f0] - fd) < 1e-6)

    # 3. error laws (Propositions 1-4)
    ok1 = ok2 = True
    for _ in range(20):
        proofs, probs = overlap_family(int(rng.integers(1, 6)), 3,
                                       int(rng.integers(0, 3)),
                                       float(rng.uniform(0.2, 0.9)))
        ex = wmc(proofs, probs)
        ok1 &= AddMult(clamp=False).value(proofs, probs) - ex >= -1e-12
        ok2 &= TopK(2).value(proofs, probs) - ex <= 1e-12
    check("Prop 1: add-mult error >= 0 (20 random instances)", ok1)
    check("Prop 2: top-k error <= 0 (20 random instances)", ok2)
    P, s, tau = 8, 0.5, 0.1
    prfs = [frozenset([f"f{j}"]) for j in range(P)]
    prbs = {f"f{j}": s for j in range(P)}
    dev = abs((LSE(tau).value(prfs, prbs) - s) - tau * np.log(P))
    check(f"Prop 3: LSE bias == tau*ln(P) ({dev:.1e})", dev < 1e-12)
    edges = [("a", "b"), ("b", "c")]
    rules = ground_tc(edges, ["a", "b", "c"])
    f0c = {("edge", u, v): 0.9 for (u, v) in edges}
    hist = iterate(rules, f0c, oplus=max, otimes=lambda a, b: a * b,
                   zero=0.0, one=1.0, n_steps=1)
    check("Prop 4: value 0 below depth horizon",
          hist[1].get(("path", "a", "c"), 0.0) == 0.0)

    # 4. Theorem-1 numeric: cyclic program, sum-product, iterate == proof agg
    edges = [("a", "b"), ("b", "a"), ("b", "c")]
    f0c = {("edge", "a", "b"): 0.9, ("edge", "b", "a"): 0.9, ("edge", "b", "c"): 0.8}
    hist = iterate(ground_tc(edges, ["a", "b", "c"]), f0c,
                   oplus=lambda a, b: a + b, otimes=lambda a, b: a * b,
                   zero=0.0, one=1.0, n_steps=20)
    md = 0.0
    for n in range(21):
        jmax = (n - 2) // 2
        agg = sum(0.72 * 0.81 ** j for j in range(jmax + 1)) if jmax >= 0 else 0.0
        md = max(md, abs(hist[n].get(("path", "a", "c"), 0.0) - agg))
    check(f"Theorem 1 numeric, non-idempotent semiring ({md:.1e})", md < 1e-12)

    # 5. liveness sanity: min-max keeps exactly one fact alive
    proofs, probs = overlap_family(4, 3, 2, 0.6)
    g = MinMax().grad(proofs, probs)
    check("min-max subgradient is one-hot",
          sum(1 for v in g.values() if abs(v) > 0) == 1)

    print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
