# Conformance — LTN (LTNtorch 1.0.2, deployed defaults incl. stable=True)

Frozen instance set v1; registered predictions and verdicts:

- **P1a** ltn:product exact (up to ~1e-3 stabilization) on fact-disjoint instances (c=0): **19/19**
- **P1b** ltn:product over-counts (signed err > 0) on every shared-trunk instance (c>=1): **31/31**
- **P2** ltn:godel conformance-to-claim ~0 AND cross-semantics distance == min-max reference: **50/50**
- **P3** ltn:godel gradient == min-max one-hot on the tie-free battery: **10/10**

phi(ltn:product vs distribution semantics) = **0.9382**; worst signed error +0.322.
Tie-free gradient liveness: product 1.000, godel 0.152 (the homogeneous-battery 1.0 reported in earlier runs is a tie-splitting artifact of torch min/max and is not quoted).

Reading: ltn:product's error is the independence assumption made
measurable — near-zero where facts are disjoint, positive growth with
shared structure (and negative on mutually exclusive proofs, per the
E6 disjoint control). ltn:godel is conformant to its own (Gödel)
claim by the min-max property; its distance to distribution semantics
is the min-max row of the arena.
