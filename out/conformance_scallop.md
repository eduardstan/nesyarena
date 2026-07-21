# Conformance — Scallop (scallopy 0.2.4, official release wheel)

The document of record for all Scallop measurements, consolidating three
campaigns (2026-06-10 values + recursion; 2026-07-03 differentiable
provenances). Runner: `experiments/conformance_scallop.py` (runs in the
py3.10 scallop env — see `INSTALL_SCALLOP.md`); frozen instances:
`benchmarks/instances_v1.json`; raw numbers: `conformance_scallop.json`.
Historical note: these campaigns were originally logged as "gates
G2/G2b/G2d" (the protocol's gate numbering); this file supersedes those logs.

## A. Values — conformance pass at machine precision

On the 50-instance frozen values battery (homogeneous + heterogeneous),
through full program compilation:

| provenance | max \|scallop − reference\| | max \|scallop − oracle\| |
|---|---|---|
| addmultprob | 2.2e-16 | 0.609 |
| minmaxprob | 0.0 | 0.263 |
| topkproofs(k=1) | 5.6e-17 | 0.378 |
| topkproofs(k=3) | 2.2e-16 | 0.061 |

The reference SUTs are validated as exact models of the deployed provenances
on acyclic structure; the vs-oracle column is the semantic error itself, live
in the deployed system, at the magnitudes the error laws predict. Chains run
by Scallop's native recursion agree with the convergent engine to 0.0.
Bounded-iteration mode: absent from scallopy 0.2.4 (`ScallopContext` exposes
no iteration limit; run-to-fixpoint only), so the depth-truncation axis
applies to Scallop only via program rewriting.

## B. Finding F-1 — addmultprob under recursion is not proof-summing

On the cyclic instance, the naive model of add-mult (sum all bounded-depth
proof scores, clamp at 1) predicts 1.0; deployed Scallop returns 0.72. The
diamond probes isolate the mechanism (two 0.6-edge paths, ± a back-edge):

| instance | scallop addmultprob | simple-path proof-sum | exact oracle |
|---|---|---|---|
| diamond DAG | 0.720 | 0.720 | 0.5904 |
| diamond + cycle | 0.720 | 0.720 | 0.5904 |

On DAGs, deployed addmultprob is exactly the reference proof-sum (the
predicted rung-P over-count); adding a cycle changes nothing — derivations
that re-derive an existing tuple never accumulate into its tag. Iteration
stops at the **tuple-set fixpoint**, freezing tags at their
acyclic-derivation values: under recursion the deployed semantic object is
iteration-policy-dependent — neither distribution semantics nor infinitary
proof-summing. The reference AddMult stays faithful on acyclic structure
(where the overlap axis lives); recursive sweeps need a
truncated-at-tuple-fixpoint reference variant (roadmap §4.3).

## C. Gradients — deployed autograd via torch tags

Fact probabilities as `requires_grad` tensors; backward through the output
tag (no finite differences). Tie-free frozen gradient battery (10 instances,
analytic oracle gradients embedded):

| provenance | max \|value − ref\| | max \|grad − ref\| | max \|grad − oracle\| |
|---|---|---|---|
| difftopkproofs(k=1) | 3.0e-08 | **2.9e-08** | 6.6e-01 |
| difftopkproofs(k=3) | 2.2e-08 | **2.6e-08** | 2.2e-01 |
| diffminmaxprob | 0.0 | **0.0** | 8.8e-01 |
| diffaddmultprob | 2.6e-08 | **1.7e+00** | 9.4e-01 |

Frozen top-k selection and the one-hot min-max subgradient are confirmed as
the deployed differentiation semantics (float32-tag precision). The
grad-vs-oracle column is the expected approximation gap, not a deviation.
An earlier finite-difference campaign agreed to 7.9e-12 on the same
instances away from the clamp region — which it deliberately skipped;
the torch-tag route measures it:

## D. Finding F-2 — diffaddmultprob's clamp is straight-through

On **all 22** frozen instances whose raw proof-sum exceeds 1:

- the returned **value** is the clamped sum `min(1, Σ s_j)` (matches the
  reference clamp model to 2.6e-08), but
- **every fact keeps a live gradient** (22/22 instances; up to |grad| = 3.24
  with the value pinned at 1.0), and
- the deployed gradient equals the gradient of the **unclamped** sum to
  **4.8e-08** (mechanism check vs `AddMult(clamp=False).grad`).

Deployed diffaddmultprob differentiates a different function than the one
whose value it returns: a straight-through clamp. The (value, gradient) pair
is mutually inconsistent throughout the saturation region. Learning
implication: no min-clamp gradient blackout — instead the optimizer receives
push-probabilities-up pressure from the unclamped surrogate while the
reported probability is pinned at 1 (over-confidence pressure with no value
feedback). Follow-up queued (roadmap §4.4): a straight-through add-mult
reference SUT in the learning layer + re-run of the saturation-relevant E6
cells.

---

As always: deviations are findings about the deployed system, logged with
the witnessing instances and pinned to the version; never normalized away.
