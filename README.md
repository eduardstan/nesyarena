# NeSyArena

Measures **semantic fidelity**: the signed, oracle-grounded error between what
a neuro-symbolic (NeSy) reasoning approximation computes and what its *claimed
semantics* defines, as a function of program structure — plus the learning
consequences (gradient liveness, calibration against exact Bayes posteriors,
held-out transfer). Companion docs: `NeSyArena_protocol_v1.md` (formal definitions, hypotheses, experiments),
`paper/main.tex` (draft with measured numbers).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,oracles,reporting]"
.venv/bin/python -m pytest          # parity gates + error-law battery (the CI contract)
make all                            # every experiment + figures + RESULTS.md
```

## Experiments (each: committed YAML config, JSON results, one-command figure)

| runner | what it measures | figure |
|---|---|---|
| `experiments.e1_overlap` | error surfaces + the crossover (no method dominates) | F1, F2 |
| `experiments.e2_depth` | truncation horizons (= n+1), starvation, H7 divergence | F3 |
| `experiments.e3_surrogate` | tau·ln P bias law + the temperature dilemma | F4 |
| `experiments.e4_witnesses` | machine-found minimal failing programs | table |
| `experiments.e6_facttable` | learning through misreasoners corrupts transfer (5 seeds) | F6 |
| `experiments.e6_pixels` | H8 headline: accuracy ties, calibration/transfer diverge | F7 |
| `experiments.e7_depth_learning` | gradient starvation end-to-end (AUC at chance) | F8 |
| `experiments.e5_mnist` | real-digit replication (MNIST-path / MNIST-sum) | F9 |
| `experiments.e5b_noise_ablation` | registered noise ablation of the control surprise | F10 |
| `experiments.e8_clutrr` | CLUTRR-style train-short/test-long: cliffs at the horizon | F11 |
| `experiments.scorecard` | fidelity-profile radar over six measured axes | radar |

Deployed-system conformance and findings: `out/G2_scallop.md`,
`out/G2b_scallop_ir.md` (F-1), `out/conformance_deeplog.md`; adapter guide:
`docs/ADAPTERS.md`.

## Layout

```
src/nesyarena/
  ir.py          ground programs (Atom/Rule/GroundProgram) + bounded proof enumeration
  algebra.py     semiring registry (boolean, maxprod, sumprod, tropical)
  engine.py      T_P iteration, run-to-convergence, proof-side aggregation (Thm-1 checked)
  oracle.py      exact WMC + analytic gradients; ProbLog oracle; graph oracles
  suts.py        reference SUTs with system-faithful gradients (add-mult, top-k, min-max, LSE)
  generators.py  structured families G1 (overlap), G2 (chain/cyclic TC), G3 (surrogate)
  metrics.py     fidelity profile, depth horizon, gradient liveness (D2-D4)
  witness.py     minimal-failure synthesis: grid + greedy shrink (D6)
tests/           parity gates vs legacy fixtures + registered error laws
legacy/          the verified Day-0 toy (executable specification; see legacy/README.md)
out/             measured results from the toy runs (RESULTS.md, figures)
```

The rebuild is gated by parity: `tests/fixtures/toy_golden.json` pins oracle
values/gradients, SUT behavior, engine histories, and witnesses produced by
the toy; the new code must reproduce them (deliberate deviations are
documented where they occur, e.g. deterministic min-max tie-breaking).

## Project rules

- Discrepancies between an external system (Scallop, ProbLog, DeepProbLog,
  DeepLog) and the reference SUTs are *findings about that system's deployed
  semantics* — log them with the instance; never normalize them away.
- Error-law predictions (sign, growth law) are registered before runs; the
  H-table is updated with outcomes either way. Refutations are results.
- The oracle battery (reference WMC ≡ ProbLog < 1e-10) must stay green on
  every commit.
