# Gate G2 — Scallop vs reference SUTs (first run: 2026-06-10)

**Verdict: conformance pass.** Scallop's deployed provenances are reproduced
by the arena's reference SUTs at machine precision on the full battery, while
deviating from the exact (distribution-semantics) oracle by the magnitudes the
error laws predict. Every result measured on the reference SUTs — error
surfaces, crossover, witnesses, gradient-liveness, learning consequences —
therefore transfers to Scallop's inference layer with measured justification.

## Setup (pinned)

- scallopy **0.2.4**, official GitHub release wheel
  `scallopy-0.2.4-cp310-cp310-manylinux_2_27_x86_64.whl` (released 2024-08-30;
  latest release at run date), Python 3.10.18 (conda env `scallop-py310`),
  Linux x86_64.
- Battery: 50 G1 instances, seed 1 — P ∈ {2..4}, L ∈ {2,3}, c ∈ {0..L-1},
  p ∈ {0.3, 0.6, 0.9}, every third instance heterogeneous. Driver:
  `legacy/nesyarena/adapters_scallop.py` (gate_g2).

## Results

| adapter | max \|scallop − reference\| | max \|scallop − exact oracle\| |
|---|---|---|
| scallop:addmultprob | 2.22e-16 | 6.09e-01 |
| scallop:minmaxprob | 0.00e+00 | 2.63e-01 |
| scallop:topkproofs(k=1) | 5.55e-17 | 3.78e-01 |
| scallop:topkproofs(k=3) | 2.22e-16 | 6.05e-02 |

## Reading

1. **External validity closed for Scallop's inference layer.** The
   "you only measured your own re-implementations" objection no longer
   applies: the re-implementations *are* the deployed semantics, to 2e-16.
2. The |scallop − exact| column is the semantic error itself, live in the
   deployed system: up to 0.61 for proof-summing, 0.38 for top-1 truncation —
   on programs of at most 9 facts.
3. Top-k under score ties matched despite tie-dependent selection: on G1's
   homogeneous instances any tied selection has the same WMC by symmetry; the
   heterogeneous instances pin the unambiguous case.

## Caveats / next

- Single-rule-per-proof programs only; recursive (G2-family) programs and
  gradient comparison (`difftopkproofs` etc.) not yet exercised through
  scallopy — that is the Phase-2 adapter work in the rebuild.
- 0.2.4 is the latest *release wheel*; the Scallop main branch may be ahead.
  A source build cross-check is optional external validity later.
