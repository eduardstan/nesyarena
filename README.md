# NeSyArena

A **measurement instrument** for neuro-symbolic (NeSy) reasoning. NeSy systems
combine a neural network with a symbolic reasoner, and the reasoner usually
*approximates* the quantity it claims to compute (it sums over proofs, keeps
the top-k, truncates recursion, smooths a `max`). NeSyArena measures the
**semantic error** that approximation introduces — the signed, oracle-grounded
gap between what a reasoner computes and what its *claimed semantics* defines —
as a function of program structure, plus what that error does to learning.

> **New here? Read [`docs/OVERVIEW.md`](docs/OVERVIEW.md) first.** It explains
> the one core idea, the end-to-end data flow with a runnable example, every
> component, how external systems (Scallop, DeepLog) plug in, and what is and
> isn't done yet. This README is the quick reference.

Companion docs: [`docs/OVERVIEW.md`](docs/OVERVIEW.md) (plain-language guide),
[`docs/ADAPTERS.md`](docs/ADAPTERS.md) (how to plug in a system),
`NeSyArena_protocol_v1.md` (formal research design), `paper/main.tex` (the
18-page technical report with all measured numbers).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,oracles,reporting]"
.venv/bin/python -m pytest          # the correctness contract (parity gates + error laws)
make all                            # every experiment + figures + RESULTS.md
```

A 5-line taste (no external dependencies):

```python
from nesyarena.generators import overlap_family
from nesyarena.suts import ExactWMC, AddMult, TopK
inst = overlap_family(P=2, L=1, c=0, p=0.6)   # query q with two 0.6-probability proofs
ExactWMC().value(inst.proofs, inst.probs)      # 0.84  ← the truth (distribution semantics)
AddMult().error(inst.proofs, inst.probs)       # +0.16 ← proof-sum over-counts
TopK(1).error(inst.proofs, inst.probs)         # −0.24 ← top-1 under-counts
```

## Experiments (each: committed YAML config, JSON results, one-command figure)

| runner | what it measures | figure |
|---|---|---|
| `experiments.e1_overlap` | error surfaces + the crossover (no method dominates) | F1, F2 |
| `experiments.e2_depth` | truncation horizons (= n+1), starvation, recursion divergence | F3 |
| `experiments.e3_surrogate` | the τ·ln P surrogate-bias law + the temperature dilemma | F4 |
| `experiments.e4_witnesses` | machine-found minimal failing programs | table |
| `experiments.e6_facttable` | learning through misreasoners corrupts transfer (5 seeds) | F6 |
| `experiments.e6_pixels` | headline: accuracy ties, calibration/transfer diverge | F7 |
| `experiments.e7_depth_learning` | gradient starvation end-to-end (AUC stuck at chance) | F8 |
| `experiments.e5_mnist` | real-digit replication (MNIST-path / MNIST-sum) | F9 |
| `experiments.e5b_noise_ablation` | registered noise ablation of a control surprise | F10 |
| `experiments.e8_clutrr` | CLUTRR-style train-short/test-long: cliffs at the horizon | F11 |
| `experiments.scorecard` | fidelity-profile radar over six measured axes | radar |

**The arena leaderboard — every deployed system, same frozen programs, same
oracle: [`out/ARENA.md`](out/ARENA.md)** (regenerate: `.venv/bin/python -m
experiments.arena`). Per-framework conformance (one log per framework): `out/conformance_scallop.md`
(findings F-1, F-3), `out/conformance_deeplog.md`,
`out/conformance_problog_kbest.md`. Measured results: `out/RESULTS.md`.

## Layout

The data flow is **generator → program → proofs → {oracle, system-under-test}
→ signed error → metrics** (see `docs/OVERVIEW.md` §2). Each module's role:

```
src/nesyarena/
  ir.py          representation: Atom / Rule / GroundProgram, and the proof
                 enumerator that turns a (program, query) into its set of proofs
  algebra.py     the semirings the engine evaluates programs under — boolean
                 (reachability), maxprod (reliability), sumprod, tropical (shortest path)
  engine.py      runs a program under a chosen algebra: bounded T_P iteration,
                 run-to-convergence, and the equivalent proof-side aggregation
  oracle.py      the ground truth: exact weighted model counting (+ analytic
                 gradients), ProbLog for large instances, graph algorithms
  suts.py        the approximations under test (reference implementations with
                 system-faithful gradients): add-mult, top-k, min-max, LSE
  adapters/      the same interface wrapping real external systems: base.py
                 (the protocol), scallop.py, deeplog.py
  generators.py  controlled program families that isolate one axis each:
                 overlap (G1), chain/cyclic recursion (G2), surrogate (G3), CLUTRR-style
  metrics.py     scoring: fidelity profile, depth horizon, gradient liveness
  witness.py     search for the smallest program where the error is large
  learning/      each reasoner as a torch op, so perception can be trained
                 through it and its corruption of the network measured
experiments/     one runner per experiment (E1–E8 + scorecard); `make all` runs all
tests/           the correctness contract: oracle ≡ ProbLog, gradients, error laws,
                 and parity against the frozen reference implementation in legacy/
legacy/          the verified Day-0 toy, kept as an executable specification
out/             measured results from the experiment runs (RESULTS.md, JSON, figures)
paper/           the 18-page technical report (main.tex → main.pdf)
docs/            OVERVIEW.md (start here) and ADAPTERS.md (plug in a system)
```

Correctness is gated by parity: `tests/fixtures/toy_golden.json` pins the
reference implementation's oracle values, gradients, engine trajectories and
witnesses; the current code must reproduce them (documented deviations aside,
e.g. deterministic min-max tie-breaking).

## Project rules

- A disagreement between an external system (Scallop, DeepLog, …) and its own
  *claimed semantics* is a **finding about that system** — recorded with the
  witnessing instance, never patched away.
- Error-law predictions (sign, growth direction) are **registered before runs**;
  measured outcomes update the record either way. Refuted predictions are
  results.
- The oracle battery (reference WMC ≡ ProbLog < 1e-10) must stay green on every
  commit — the ground truth never silently drifts.
