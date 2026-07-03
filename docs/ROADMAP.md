# NeSyArena — roadmap (phase 2: breadth)

The instrument and the core results exist and reproduce (`make all`). Phase 2
is about **breadth**: more deployed frameworks measured on the *same* frozen
instances, more seeds, and the axes that are specified but not yet built.
This file is the working plan; `docs/OVERVIEW.md` explains the system itself.

## 1. Results we already have

One line each; details in `out/RESULTS.md` (auto-generated) and the
conformance logs in `out/`.

| result | where |
|---|---|
| Structural crossover: over- vs under-counting flip with overlap; no method dominates | E1, `out/E1_results.json` |
| Depth horizons exactly n+1; flat-zero gradients below; recursion divergence (3.79 > 1) | E2 |
| Surrogate bias law τ·ln P to machine precision + the temperature dilemma | E3 |
| Machine-found minimal witnesses (incl. the canonical double-counting example) | E4 |
| Fact-table learning: misreasoners corrupt the learned facts; disjoint control is blind | E6-facttable |
| Headline: accuracy ties across 5 reasoners, calibration/transfer separate them | E6-pixels |
| Same on real MNIST digits (grid-reachability task) | E5 |
| Registered noise ablation: top-1's control win is a knife-edge artifact (one prediction refuted) | E5b |
| Gradient starvation end-to-end: truncated training never leaves chance | E7 |
| CLUTRR-style train-short/test-long: generalization cliffs at exactly the horizon | E8 |
| Scallop conformance at machine precision on acyclic programs + a deployed-semantics finding under recursion (F-1) | `out/G2*.md` |
| DeepLog conformance pass, values and gradients | `out/conformance_deeplog.md` |
| Fidelity scorecard over six measured axes | `out/scorecard.json`, radar |

## 2. Cross-framework consistency: the frozen instance set

**Every new framework must be measured on the identical generated programs.**
These are now frozen in **`benchmarks/instances_v1.json`** (71 instances:
50 value-conformance, 10 gradient-conformance with analytic oracle gradients,
4 recursion chains, the cyclic instance, 2 recursion-policy probes, 4 minimal
witnesses — the same instance streams behind the published conformance
numbers, oracle values included). Loader:

```python
from nesyarena.benchmarks import load_instances
for inst in load_instances(battery="values"):
    got = adapter.infer(inst.program, inst.probs, [inst.query])[inst.query]
    err = got - inst.oracle_value        # signed semantic error
```

Rules: never edit `instances_v1.json` in place (add a v2 with a rationale);
report per-battery max |deviation| against the reference SUT *and* against the
oracle, as in `out/G2_scallop.md`.

## 3. Framework integration matrix (candidates)

Each new backend = one adapter (see `docs/ADAPTERS.md`, ~30 lines) + a
conformance run on the frozen batteries + a short findings log in `out/`.

| framework | what it tests | claimed semantics | effort | notes |
|---|---|---|---|---|
| **ProbLog k-best** | second deployed approximate inference (anytime bounds) | distribution semantics (bounds) | **done** | measured: sound on all 284 cells; exact at tight eps (6e-16); lower border is implicant-based — `out/conformance_problog_kbest.md` |
| **Scallop `diff*` provenances** | deployed *gradient* semantics via torch tags | distribution semantics | **done** | top-k/min-max grads conform (3e-08); **finding F-3**: diffaddmultprob's clamp is straight-through (clamped value, unclamped gradient) — `out/G2d_scallop_diff.md` |
| **DeepProbLog (standalone)** | the original NeurIPS-2018 system, exact + approximate modes | distribution semantics | **M** | install may need SWI-Prolog; time-box |
| **LTN (Logic Tensor Networks)** | *fuzzy* axis: t-norm evaluation + smooth aggregators (pMean vs min) — widens scope beyond probabilistic | product real logic | **M** | new registered error laws possible (pMean bias, analogous to LSE) |
| Lobster (GPU Scallop) | GPU/CPU consistency of a deployed engine | distribution semantics | **L** | needs CUDA + source build; stretch |
| PITA / MCINTYRE (cplint) | sampling-based approximate inference (a *stochastic* error class) | distribution semantics | **M–L** | stretch |
| NeurASP, PSL, semantic-loss systems | — | non-monotone / logic-as-loss | — | out of scope for now (documented in OVERVIEW/report) |

Priority order: ProbLog k-best → Scallop `diff*` → DeepProbLog standalone →
LTN; Lobster and PITA as stretch. Two adapters can proceed fully in parallel —
they only share the frozen instance set and the adapter protocol.

## 4. Experiment extensions (beyond new frameworks)

1. **Seeds to protocol standard:** every learning run at ≥5 seeds (E6-pixels,
   E5, E7, E8 are at 3); ≥10 seeds on the high-variance MNIST-sum control.
2. **Port the supervision-richness ablation** (currently verified only on the
   frozen reference implementation) to the rebuilt core.
3. **Recursion-faithful add-mult reference variant** (per finding F-1) +
   recursive overlap sweeps against Scallop.
4. **Straight-through add-mult reference SUT** (per finding F-3) in the
   learning layer + re-run of the saturation-relevant E6 cells with it (the
   deployed-faithful clamp may corrupt perception differently than the
   min-clamp our published learning numbers model).
5. **Negation axis** (stratified / conformance-by-refusal), specified in the
   protocol, not yet ported.
6. **Random-DAG family** + two-level overlap surfaces (beyond the designed G1/G2).
7. **Fuzzy error laws** if/when LTN lands: registered sign/growth predictions
   for pMean-style aggregators before running.
8. Stretch: real-text CLUTRR; compilation-based oracle to pass the 22-fact cap.

## 5. Project page

A static page (GitHub Pages from this repo): the pitch, the scorecard radar,
2–3 headline figures with captions, the conformance leaderboard table, links
to OVERVIEW/ADAPTERS and the repro instructions. Explicitly **not** an
upload-and-evaluate service — the repo itself is the way to run your own
system (adapter + frozen instances). One page, generated from committed
assets, kept in sync with `out/`.

## 6. Milestones

- **M1 (~1 week):** frozen instance set adopted by all runs (done: v1);
  ProbLog k-best adapter + conformance log; page skeleton.
- **M2 (~3 weeks):** Scallop `diff*` gradient conformance; DeepProbLog
  standalone; seeds to standard; ablation port.
- **M3 (~6 weeks):** LTN + fuzzy laws; negation axis; recursive sweeps with
  the faithful reference variant; random DAGs.
- **M4 (~9 weeks):** consolidation — updated report and scorecard with the
  full framework matrix, page live, submission-ready packet.

Progress and measured outcomes land in `out/RESULTS.md` as always; deviations
of any framework from its claimed semantics are findings, logged with the
witnessing instance.
