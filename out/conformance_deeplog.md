# DeepLog conformance — first measurement (2026-06-10)

**Verdict: conformance pass, as registered.** DeepLog (pydeeplog 3.0.3,
ML-KULeuven/deeplog, the May-2026 software-framework release) claims
exactness by compilation to algebraic circuits; the registered prediction was
fidelity 1.0 on the probabilistic axis. Measured:

- Burglary/earthquake micro-example: 0.94 exactly (1 − 0.2·0.7).
- G1 battery (8 instances, homogeneous + heterogeneous): max |deeplog − WMC
  oracle| < 1e-6 (torch float32 circuit evaluation; agreement is at single
  precision, vs the float64 2e-16 of the Scallop comparisons).
- Recursive chain program via the arena's bounded unrolling: 0.9^4 exact.

These checks run in the standing pytest suite (`tests/test_deeplog_adapter.py`,
skipped automatically when pydeeplog is absent).

## Why this matters for the paper

The leaderboard now has its intended contrast measured, not asserted:
a substrate that compiles to exact circuits (DeepLog) sits at fidelity 1.0,
while a substrate offering selectable approximate provenances (Scallop)
exhibits exactly the error-law deviations — up to 0.61 on nine-fact programs
(see G2_scallop.md), plus the recursion finding F-1 (G2b_scallop_ir.md).
Substrates execute semantics; the arena measures whether the executed object
matches the claimed one. DeepLog passing is as much a result as Scallop's
deviations: conformance has to be measured to be claimed.

## Adapter notes / future work

- Construction mirrors DeepLog's own weighted-model-count pattern
  (assignment-aware `p(F)` weights × boolean→probability transform of the
  proof DNF, summed over fact assignments); recursion is unrolled by the
  arena's bounded enumeration since DeepLog's input language is formulas.
- **Gradient conformance: exercised 2026-06-10 — pass.** Symbolic (non-
  numeric) probability labels survive `_resolve_argument` and are exposed as
  module inputs; torch autograd through the compiled circuit yields DeepLog's
  deployed gradients. On a heterogeneous G1 battery: max |value dev| 8.7e-08,
  max |grad dev| 1.3e-07 vs the analytic oracle (float32 circuit internals).
  DeepLog's gradients are exact at its working precision, as compilation
  predicts — now a standing test (`test_deeplog_autograd_matches_...`).
- Fuzzy/Boolean structures (Gödel etc.) selectable via DeepLog's algebra
  registry — a future cross-check against the arena's idempotent oracles.
- Version pin: pydeeplog 3.0.3, repo last pushed 2026-06-03.
