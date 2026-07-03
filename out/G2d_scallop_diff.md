# Gate G2d — Scallop differentiable provenances (torch tags) — 2026-07-03

Setup: scallopy 0.2.4 (release wheel, py3.10) + torch CPU; frozen instance set
v1 (`benchmarks/instances_v1.json`); driver `scripts/gate_scallop_diff.py`;
raw numbers in `G2d_scallop_diff.json`. Gradients here are Scallop's **own**
autograd (fact probabilities as `requires_grad` tensors, backward through the
output tag) — no finite differences.

## A. Gradient battery (10 tie-free instances, analytic oracle grads embedded)

| provenance | max \|value − ref\| | max \|grad − ref\| | max \|grad − oracle\| |
|---|---|---|---|
| difftopkproofs(k=1) | 3.0e-08 | **2.9e-08** | 6.6e-01 |
| difftopkproofs(k=3) | 2.2e-08 | **2.6e-08** | 2.2e-01 |
| diffminmaxprob | 0.0 | **0.0** | 8.8e-01 |
| diffaddmultprob | 2.6e-08 | **1.7e+00** | 9.4e-01 |

Top-k (frozen selection) and min-max (one-hot subgradient) **pass** gradient
conformance at the tags' float32 precision: the reference SUTs' gradient
semantics are the deployed ones. The grad-vs-oracle columns are the *expected*
approximation gaps (the whole point of the instrument), not deviations.

## B. Finding F-3: diffaddmultprob's clamp is straight-through

The earlier finite-difference gate (G2b) deliberately skipped instances near
the clamp; the torch-tag route measures them. On **all 22** frozen instances
whose raw proof-sum exceeds 1 (values + witnesses batteries):

- the returned **value** is the clamped sum `min(1, Σ s_j)` (matches the
  reference clamp model to 2.6e-08) — but
- **every fact keeps a live gradient** (22/22 instances; up to |grad| = 3.24
  with the value pinned at 1.0), and
- the deployed gradient equals the gradient of the **unclamped** sum
  `Σ s_j` to **4.8e-08** (mechanism check against `AddMult(clamp=False).grad`).

So deployed `diffaddmultprob` differentiates a *different function than the
one whose value it returns*: a straight-through clamp. Consequences:

1. The (value, gradient) pair is mutually inconsistent throughout the
   saturation region — the gradient is not the derivative of the returned
   value anywhere there.
2. Learning through it does **not** suffer the min-clamp gradient blackout
   (our reference clamp model's behavior, gradient liveness 0.611): instead
   the optimizer keeps receiving push-probabilities-up pressure from the
   unclamped surrogate even while the reported probability is pinned at 1 —
   a different failure mode (unbounded over-confidence pressure, no value
   feedback), not an absence of one.
3. Arena action item: add a **straight-through add-mult** reference SUT to
   the learning layer and re-run the E6 saturation-relevant cells with it —
   the deployed-faithful variant may corrupt perception differently than the
   min-clamp variant our published learning numbers model.

As always: logged with the witnessing instances, pinned by the gate, version
pinned to scallopy 0.2.4; not normalized away.
