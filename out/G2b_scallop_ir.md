# Gate G2b — Scallop through the rebuilt IR: recursion + gradients (2026-06-10)

Setup as in `G2_scallop.md` (scallopy 0.2.4 release wheel, py310). Driver:
`scripts/gate_scallop_ir.py` (programs compiled from the IR via
`nesyarena.adapters.scallop`; raw numbers in `G2b_scallop_ir.json`).

## A. Value conformance on G1, via program compilation — pass

Same machine-precision agreement as the legacy gate (max |scallop − ref|
≤ 2.22e-16 across all four provenances), now exercising the full
program → compile → scallopy path rather than hand-encoded rules.

## B. Recursion

Chains (single proof, depth L ∈ {2,4,6,8}): all provenances agree with the
convergent fixpoint to 0.0. Cyclic Section-5 instance (a↔b→c):

| provenance | scallop | reference model (depth-8 proofs) | exact WMC |
|---|---|---|---|
| minmaxprob | 0.8 | 0.8 | 0.72 |
| topkproofs(k=1) | 0.72 | 0.72 | 0.72 |
| topkproofs(k=3) | 0.72 | 0.72 | 0.72 |
| **addmultprob** | **0.72** | **1.0 (clamped proof-sum)** | 0.72 |

### Finding F-1: addmultprob under recursion is *not* proof-summing

The naive model of add-mult — sum the scores of all (bounded-depth) proofs,
clamp at 1 — predicts 1.0 on the cyclic instance. Deployed Scallop returns
0.72. Designed probe (diamond a→b→c / a→d→c with 0.6 edges, ± back-edge c→a):

| instance | scallop addmultprob | sum of simple-path scores | exact WMC |
|---|---|---|---|
| DAG diamond | 0.720000 | 0.720000 | 0.590400 |
| diamond + cycle | 0.720000 | 0.720000 | 0.590400 |

On DAGs, deployed addmultprob is exactly the reference proof-sum (over-counts
vs exact, rung-P as predicted). Adding a cycle changes nothing: derivations
that revisit an already-derived tuple never accumulate into its tag — the
iteration stops at the **tuple-set fixpoint**, freezing tags at their
acyclic-derivation values. So under recursion the deployed semantic object is
*iteration-policy-dependent* ("proof-sum truncated at tuple-set fixpoint"),
which is neither distribution semantics nor infinitary proof-sum. Two
consequences worth reporting:

1. On cyclic *reachability* this accidentally protects addmultprob from the
   H7 saturation (it lands on 0.72 here because loop supports are subsumed by
   the simple path — a coincidence of this family, not a general guarantee).
2. The reference AddMult remains a faithful model of deployed addmultprob on
   acyclic structure (all G1/DAG cells), which is where the arena's overlap
   axis lives. For recursive sweeps, the reference model of "deployed
   add-mult" must be the truncated-at-tuple-fixpoint variant, not the
   depth-n proof-sum. (Action: add a `FirstFixpointAddMult` reference SUT
   before running E2-style sweeps against Scallop.)

## C. Gradient conformance — pass

Central finite differences through Scallop vs the references'
system-faithful gradients, heterogeneous (tie-free) G1 instances:
max deviation 7.91e-12 (topkproofs k=3), all four provenances OK at 1e-4.
Top-k's frozen-selection gradient and min-max's one-hot subgradient are
confirmed as the deployed differentiation behavior at the value level.

## Not yet exercised

- Bounded-iteration mode: **probed 2026-06-10 — absent.** scallopy 0.2.4's
  `ScallopContext.__init__` accepts (provenance, custom_provenance, k,
  wmc_with_disjunctions, train_k, test_k, fork_from, no_stdlib, monitors) and
  `run()` takes no arguments: deployed Scallop is run-to-fixpoint only. The
  depth-horizon axis (Prop. 4 truncation) therefore applies to Scallop only
  via explicit program rewriting; the truncation laws' deployed exemplars
  remain unrolling-based systems (TensorLog-style), modeled by the reference
  bounded engine.
- torch-tagged diff* provenances (optional cross-check of battery C).
