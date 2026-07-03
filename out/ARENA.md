# THE ARENA — semantic fidelity of deployed NeSy reasoners

Every number on the identical frozen programs (`benchmarks/instances_v1.json`,
50-instance values battery, 10-instance gradients battery, the cyclic
instance) against the identical exact oracle. Higher phi = closer to the
claimed semantics; **phi = 1 only for exact inference**.

| system | version | phi | worst signed err | grad cos | grad live | cyclic (oracle 0.72) | impl. conformance | findings |
|---|---|---|---|---|---|---|---|---|
| exact WMC (oracle semantics) | — | **1.000** | +0.000 | 1.000 | 1.000 | 0.720 | 0.0e+00 | — |
| deeplog exact circuits | pydeeplog 3.0.3 | **1.000** | +0.000 | 1.000 | 1.000 | 0.720 | 1.3e-07 | — |
| problog kbest lower bound (eps=1e-09) | problog 2.2.10 | **1.000** | +0.000 | n/a | n/a | 0.720 | 0.0e+00 | sound bounds 284/284; lower border implicant-based |
| deepproblog exact engine | deepproblog 2.0.6 | **1.000** | +0.000 | n/a | n/a | 0.720 | 0.0e+00 | — |
| scallop topkproofs k=3 | scallopy 0.2.4 | **0.994** | -0.060 | 0.964 | 0.858 | 0.720 | 2.6e-08 | — |
| problog kbest lower bound (eps=0.2) | problog 2.2.10 | **0.914** | -0.193 | n/a | n/a | 0.720 | 0.0e+00 | sound bounds 284/284; lower border implicant-based |
| scallop minmaxprob | scallopy 0.2.4 | **0.897** | +0.263 | 0.594 | 0.152 | 0.800 | 0.0e+00 | — |
| scallop topkproofs k=1 | scallopy 0.2.4 | **0.847** | -0.378 | 0.767 | 0.351 | 0.720 | 2.9e-08 | — |
| scallop addmultprob | scallopy 0.2.4 | **0.839** | +0.609 | 0.980 | 1.000 | 0.720 | 4.8e-08 | F-1 (recursion), F-3 (straight-through clamp) |

## How to read this table

- **phi** (semantic fidelity, the headline) = 1 − mean |computed −
  claimed-semantics value| on the values battery. It scores the
  *approximation*, not the code quality: a perfectly implemented
  approximation still loses phi.
- **worst signed err**: sign matters — positive = over-counts
  (add-mult-style), negative = under-counts (top-k-style). Opposite
  signs are why no method dominates (the crossover, E1).
- **grad cos / grad live**: direction agreement with the analytic oracle
  gradient, and the share of oracle-live facts that receive any gradient
  (tie-free gradients battery). Low liveness = starved learning signal.
- **cyclic**: the value returned on the canonical recursive instance
  (exact answer 0.720). Deviations here are recursion-policy effects
  (finding F-1).
- **impl. conformance** (the audit, NOT the score): distance between the
  deployed system and its best validated model on this same battery.
  ~1e-16 means we know *exactly* what the system computes — which is
  what authorizes the `measured via` shortcuts. A system can be at
  machine-precision conformance and still have low phi: it faithfully
  computes an unfaithful approximation.
- **measured via** (provenance of each row):

  - exact WMC (oracle semantics): definition
  - scallop addmultprob: reference model (values 2.2e-16); F-3 grad model (4.8e-08)
  - scallop topkproofs k=1: reference model (values 5.6e-17, grads 2.9e-08)
  - scallop topkproofs k=3: reference model (values 2.2e-16, grads 2.6e-08)
  - scallop minmaxprob: reference model (values 0.0, grads 0.0)
  - problog kbest lower bound (eps=0.2): measured live (this battery)
  - problog kbest lower bound (eps=1e-09): measured live (this battery)
  - deepproblog exact engine: measured live (this battery); see conformance_deepproblog.md
  - deeplog exact circuits: measured (values <1e-6 f32, grads 1.3e-07); see conformance_deeplog.md

Reference (idealized) SUT rows are omitted: after validation they are
numerically identical to the deployed rows they model. Per-framework
detail and findings: `conformance_scallop.md` (F-1, F-3),
`conformance_deeplog.md`, `conformance_problog_kbest.md`.
Learning-consequence results (calibration, transfer): `RESULTS.md`.
