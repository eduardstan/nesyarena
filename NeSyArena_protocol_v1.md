# NeSyArena — Protocol v1.0 (Day 0)

**Working title of the paper:** *Accuracy Hides Semantic Error: A Diagnostic Arena for Neuro-Symbolic Reasoning*
(alternatives: *The Semantic Fidelity Arena: Measuring What NeSy Systems Actually Compute*; *Conformance Testing for Neuro-Symbolic Semantics*)

**Status:** core measurement pipeline implemented and verified (see §0). Two coders, 45-day MVP track, 90-day award track. This document is the single source of truth: formal definitions, hypotheses, generators, experiment matrix, schedule, paper plan, and review-risk mapping.

---

## 0. Day-0 status: what is already running and verified

The reference engine (`arena_day0.py`, shipped with this document) faithfully implements the KR submission's Definitions 11–17 (operator semantics `T_P`, fact-augmented `F = f_θ ⊕ T_P(I)`, bounded iteration, truncation semantics per Remark 6) plus an exact possible-worlds oracle with analytic gradients. Verified today, all to machine precision:

| Check | Result | Anchor in KR paper |
|---|---|---|
| Theorem 1, numerically, for a **non-idempotent** semiring (sum-product on the cyclic program, n = 0..20) | max diff 4.4e-16 | Thm 1 |
| Section-5 stress test: same program, four algebras | Boolean 1.0, tropical 2.0, max-product 0.72, sum-product **3.7895** (= 0.72/0.19) | §5.2–5.4, Remark 12 |
| Clamped add-mult on the cyclic program | saturates at **1.0** (reports certainty) | Remark 12 |
| Overlap crossover (signed errors, opposite slopes) | add-mult +0.092→+0.308 as overlap 0→3; top-1 −0.296→−0.081 | §6.3 DeepProbLog audit, Rung P |
| min-max-prob sign is structure-dependent | +0.390 (homogeneous overlap) vs −0.472 (heterogeneous disjoint) | new (H4) |
| Truncation: zero value **and identically zero gradient** below horizon; exact match at horizon (0.430467 / 0.478297 = 0.9^8 / 0.9^7) | confirmed | Thm 1, §7.4 |
| Surrogate bias law: LSE_τ − max = τ·ln P for P equal-score proofs | max deviation 1.1e-16 | §6.3 NTP audit |
| Benchmark autopsy: MNIST-sum's proofs are mutually exclusive ⇒ add-mult error **exactly 0** by construction; top-k (k=3) drops 0.130 mass | confirmed | new (H9) |
| ProbLog exact inference vs our brute-force WMC oracle (overlap instance) | diff 2.8e-17 | week-1 gate — **passed day 0** |

Consequence: the planned week-1 gate is already passed; the science can proceed even if Scallop integration lags (the reference implementations of the provenance algebras are the fallback; Scallop then provides external validity rather than being a hard dependency).

---

## 1. Thesis and contributions

**Thesis.** NeSy systems are evaluated by task accuracy, but accuracy cannot distinguish a system that computes its claimed semantics from one that computes a different object that happens to rank answers similarly. We introduce *semantic fidelity* — oracle-grounded, signed error between the value a system computes and the value its claimed algebra defines — as an evaluation primitive, and an open arena that measures it as a function of program structure, together with its learning consequences.

**Contributions (phrased for the paper):**

- **C1 — The instrument.** Formal definitions of semantic error, fidelity profile, depth horizon, gradient fidelity, and true-marginal calibration; an open arena (generators + adapters + oracles + scoring + witness synthesis) implementing them. The adapter interface is a standard others can drop their systems into.
- **C2 — Error laws.** The algebraic assumption stack (compressed from the KR submission to a background section) is operationalized as a *hypothesis generator*: each violated assumption implies a signed error law with a growth direction (Table H below). Several are closed-form (e.g., LSE bias = τ ln P).
- **C3 — Findings.** (i) The structural crossover: approximations with opposite structural sensitivities, so no method dominates and *structure decides* — the fidelity profile, not a scalar score, is the right report. (ii) The benchmark autopsy: the field's default benchmark (MNIST-sum_n) sits in the unique cell of the structure grid where rung-P violations are provably invisible — explaining published null results about k and showing why the field has not noticed. (iii) The learning consequence: inference-time semantic error propagates to gradient bias and miscalibration of f_θ against the true marginal (H8 — the open headline experiment).
- **C4 — Adjudication.** The arena as empirical referee for the live independence/approximation dispute (van Krieken et al. ICML'24 vs. Faronius & Dos Martires '25): we measure *model-level computed semantics* under controlled structure, orthogonal to and informing their loss-level argument.

---

## 2. Formal definitions (new objects; this is the paper's §3)

Fix a ground program P, base interpretation f_θ : HB(Σ) → V (KR Def. 9), and a query q. Let Σ_claim be the *claimed semantics* of a system S (declared by its adapter, e.g. "distribution semantics", "max-product least fixed point", "depth-n truncation"), and let I*(q) be the oracle value of q under Σ_claim. Let Ŝ(q) be the value S actually returns.

- **D1 (Semantic error).** ε_S(q) = Ŝ(q) − I*(q). Signed; report also |ε| and worst-case over a sweep. Crucially, S and the oracle are fed the *identical* f_θ, so ε isolates the reasoning layer from perception (the controlled-experiment crux; cf. KR Remark 13: only changing f_θ preserves semantics — here we hold f_θ fixed and vary the reasoner).
- **D2 (Fidelity profile).** For a family of structured sweeps {G_a} (one per axis a), the profile of S is the vector φ(S) = (1 − mean_{G_a} |ε_S|)_a, with values clamped to the algebra's range for scoring (raw values also reported). The scorecard/radar is φ.
- **D3 (Depth horizon).** On the chain family with required proof depth L, h_δ(S) = min { L : |ε_S| > δ }. For unrolling-based systems, theory predicts h_δ = n + 1 (KR Thm 1).
- **D4 (Gradient fidelity).** With oracle gradient ∇* (analytic for WMC: ∂P/∂p_f = P(q|f) − P(q|¬f); autodiff elsewhere) and system gradient ∇̂: cosine similarity cos(∇̂, ∇*), relative magnitude, and the *starvation indicator* 1[∇̂ = 0 ∧ ∇* ≠ 0].
- **D5 (True-marginal calibration, TMC).** TMC(S) = E_x |Ŝ(q; f_θ(x)) − I*(q; f_θ(x))| over a data distribution, with the *same network outputs* fed to both. Distinct from ECE (which calibrates confidence against empirical accuracy) — this is the metric the PNNL assurance study did not compute, and the reason its k-sweep showed "no effect".
- **D6 (Witness).** A minimal generator configuration g with |ε_S(g)| > δ — the machine-found analogue of the KR paper's hand-crafted micro-examples (§6.3). Found by grid + shrink (property-based-testing style, à la QuickCheck shrinking).

---

## 3. Error laws and hypothesis table

Each row: prediction derived from the algebra of the KR submission, with growth law and current status.

| # | Violated assumption (rung) | System behavior | Predicted law | Anchor | Status |
|---|---|---|---|---|---|
| H1 | Rung P: proof-sum without disjointness | add-mult provenance | ε > 0; monotone ↑ in overlap c, proof count P, prob level p; bounded by Bonferroni pairwise term Σ_{i<j} P(π_i ∧ π_j); saturation regime when clamp triggers (Σ > 1 ⇒ ε = 1 − P*) | §6.3 DeepProbLog audit, Remark 12 | **verified day 0** (incl. unscripted saturation bend at P=5) |
| H2 | ⊕ truncated to k summands | top-k proofs | ε < 0; \|ε\| = exact mass of dropped proofs; monotone ↑ in P − k; monotone **↓ in overlap** (dropped proofs increasingly redundant); ε = 0 iff #proofs ≤ k | Rung P + Def. 16 | **verified day 0** |
| H3 | — (consequence of H1×H2) | crossover | ∃ overlap threshold where the \|ε\|-ranking of {add-mult, top-k} flips ⇒ no dominant method; structure decides | new | **verified day 0** (flip between c=1 and c=2 at P=4, L=4, p=0.6) |
| H4 | ⊕ replaced by max, ⊗ by min | min-max-prob | sign structure-dependent (not derivable from the stack alone — the arena measures what theory under-determines) | new | **verified day 0** (+0.39 vs −0.47) |
| H5 | bounded n below proof depth | any unroller | value = 0 and ∇ ≡ 0 (exactly) below horizon; step at h = minimal proof depth | Thm 1, §7.4 | **verified day 0** |
| H6 | join replaced by smooth surrogate | LSE_τ | bias = τ ln P (equal scores), > 0 always; gradient share to non-maximal proofs ↑ with τ ⇒ the dilemma: no τ achieves low bias and non-starved gradients for large P | §6.3 NTP audit | **verified day 0** (law to 1e-16) |
| H7 | Rung 3–5 absent under recursion | sum-product on cyclic programs | "probability" converges to value > 1 (0.72/0.19 = 3.79 on the §5 program); clamped variant saturates at certainty 1.0 | §5, Remark 12 | **verified day 0** |
| H8 | learning under any of H1–H6 | trained NeSy pipelines | inference-time \|ε\| predicts gradient bias and TMC of the trained f_θ; effect present on overlap-structured tasks, absent on disjoint-structured tasks | Remark 13 (ii)/(iii) | **open — the headline experiment** |
| H9 | — (meta) | benchmark design | MNIST-sum_n has mutually exclusive proofs ⇒ add-mult is *exactly* distribution semantics there; only top-k truncation can bite (= dropped convolution tail). The community's default benchmark cannot detect rung-P violations in principle | new | **verified day 0** (structure level); to confirm on trained Scallop pipeline |
| H10 | non-stratified negation | systems claiming it | outside the monotone contract; arena marks "outside claimed semantics" (Scallop's compiler rejects — a *conformance pass by refusal*) | §7.1 | to run |

H4 and H8 are the scientifically open rows — they are where the paper can *surprise*, which is what separates verification from discovery. H3 + H9 are the quotable findings already in hand.

---

## 4. Architecture

Five layers; the adapter interface is itself a contribution (C1).

```
generators/        # parameterized program+fact families (G1–G4)
adapters/          # uniform wrapper per backend; declares claimed_semantics
oracles/           # exact values per axis (WMC brute force, ProbLog, graph algorithms)
scoring/           # D1–D5 metrics, sweeps, witness synthesis (D6)
reporting/         # profiles (radar), error surfaces, leaderboard table, witness table
```

**Adapter interface (freeze this in week 1; it is the community-facing standard):**

```python
class Adapter(Protocol):
    name: str                      # e.g. "scallop:difftopkproofs(k=3)"
    claimed_semantics: str         # e.g. "distribution semantics"
    supports_grad: bool
    def infer(self, program: GroundProgram, base: dict[Atom, float],
              queries: list[Atom]) -> dict[Atom, float]: ...
    def grad(self, program, base, query, wrt: list[Atom]) -> dict[Atom, float]: ...
```

**Systems under test (SUTs).**
- *Primary:* Scallop provenances via `scallopy` — at minimum `minmaxprob`, `addmultprob`, `topkproofs`/`difftopkproofs` (k-swept), plus discrete `unit`/`proofs` for sanity. Confirm exact provenance names against the installed version; do not trust this list blindly.
- *Reference (already built):* our pure-Python implementations of the same algebras — used day 0; they remain in the arena as "idealized" SUTs and as the fallback if Scallop integration stalls.
- *Cross-check (optional, time-boxed to 3 days):* DeepProbLog on the probabilistic axis.
- *Oracles:* brute-force WMC (≤ 22 facts; analytic gradients) for all synthetic families; ProbLog exact inference for larger instances (**validated day 0 to 2.8e-17**); direct graph algorithms (BFS reachability, Dijkstra, max-reliability path / fixed-point iteration to convergence) for the non-probabilistic algebras.

**Witness synthesis (D6).** Grid over generator parameters; on first |ε| > δ, shrink (reduce P, L, c, p toward minimal instance). Output: a table of minimal failing programs per SUT — the machine-generated version of KR §6.3's hand-built certificates. Cheap (a day) and high-value for the artifact.

---

## 5. Generators (exact specifications)

**G1 — Overlap family (axis A1).** Parameters (P, L, c, p | heterogeneous flag). P proofs, each L facts; c shared "trunk" facts, L−c private per proof; all probs p, or heterogeneous {strong 0.9, weak 0.5} per proof. Fact count m = c + P(L−c) ≤ 22 (brute-force constraint; beyond that, ProbLog oracle). Exact value: brute-force WMC (cross-checked vs ProbLog). Default grid: P ∈ {1..8}, L ∈ {2,3,4}, c ∈ {0..L−1}, p ∈ {0.3, 0.6, 0.9}, 5 heterogeneous seeds. Extension (week 3): two-level overlap (shared trunk + pairwise-shared middles) to map the surface, not just the edge.

**G2 — Depth family (axis A2).** Chains v_0→…→v_L, L ∈ {1..14}, edge prob p ∈ {0.5, 0.9}; plus the §5 cyclic graph (a↔b→c) as the canonical recursive instance; plus random DAGs (n=10 nodes, edge density swept) where minimal proof depth varies per query. Program: transitive closure (the KR paper's own). Modes: bounded unrolling n ∈ {0..16} vs run-to-convergence. Oracles: graph algorithms per algebra. Metrics: D1, D3 (h_δ), D4 (starvation indicator).

**G3 — Surrogate family (axis A3).** Proof score multisets: equal scores (s, P) for the bias law; perturbed (s+Δ, s×(P−1)) for the dilemma curve; τ ∈ geomspace(0.005, 0.4). Pure function-level — no program needed. Closed-form predictions throughout.

**G4 — Negation family (axis A4).** Stratified win/lose programs (win(X) ← move(X,Y) ∧ ¬win(Y) on DAGs — stratified per stratum); non-stratified variants (p ← ¬q. q ← ¬p.). Expected: Scallop rejects non-stratified (conformance-by-refusal, H10); stratified handled per-stratum per KR §7.1. Lowest priority; cut first if time is short.

---

## 6. Experiment matrix

| ID | What | SUTs | Oracle | Metrics | Output | Predicted by |
|---|---|---|---|---|---|---|
| E1 | overlap sweep (G1 grid) | all provenances + reference | WMC + ProbLog | D1 | **F1** error-surface small-multiples (c × P heatmap per SUT); **F2** crossover curves | H1–H4 |
| E2 | depth sweep (G2) | bounded vs convergent modes | graph algs | D1, D3, D4 | **F3** value+gradient horizon (have day-0 version) | H5, H7 |
| E3 | surrogate (G3) | LSE_τ vs max | closed form | D1, D4 | **F4** bias law + dilemma curve (have day-0 version) | H6 |
| E4 | witness synthesis over E1–E3 | all | — | D6 | witness table (paper table) | all |
| E5 | **learning, disjoint control:** MNIST-sum_n (n = 2,3,5), CNN + Scallop, provenances × k | Scallop | exact convolution | task acc, TMC, D4 | **F5** provenances tie (predicted) | H9 |
| E6 | **learning, overlap treatment:** MNIST-path — grid of MNIST images classified open/blocked (digit parity), query s→t reachability; overlapping paths share cells ⇒ genuine rung-P structure | Scallop provenances × k | ProbLog/WMC on network outputs | task acc, TMC, gradient bias vs oracle ∇ | **F6** TMC & gradient bias diverge across provenances; scatter: inference-time \|ε\| vs trained TMC — **the headline figure** | H8 |
| E7 | depth learning: chain TC with learned edge classifier, train at unroll n < L | Scallop bounded | graph | learning curves, starvation | supplement to F3 | H5 |
| E8 | external validity: CLUTRR train-on-short(k=2,3)/test-on-long(k≤10) under bounded vs convergent reasoning | Scallop | — | generalization gap vs h_δ | **F7** | H5 |

**Figure list for the paper (7):** F1 error surfaces, F2 crossover, F3 horizon, F4 surrogate dilemma, F5 disjoint control (ties), F6 learning consequence (headline), F7 CLUTRR external validity. Plus: real-data scorecard radar (the profile φ), witness table.

**Statistical standards.** 5 seeds minimum on every learning run; report mean ± std; heterogeneous-probability variants with 5 seeds on E1; all generator configs and seeds in committed YAML; CI job asserts reference-vs-ProbLog agreement < 1e-10 on a fixed battery (the oracle never silently rots).

---

## 7. The 45-day MVP schedule (2 coders: A = arena core, B = learning/experiments)

Day-0 work already banked: reference engine, WMC oracle + analytic gradients, ProbLog cross-validation, E1/E2/E3 prototype runs, three figures.

| Days | Coder A | Coder B | Gate / deliverable |
|---|---|---|---|
| 1–3 | Repo scaffold from `arena_day0.py`; adapter interface frozen; config/seed system | Scallop install (pip wheels; else maturin build; else Docker). Reproduce MNIST-sum_2 from Scallop README | **G1 gate (day 3):** Scallop adapter returns raw query probabilities (not argmax) |
| 4–7 | G1/G2 generators productionized; ProbLog adapter; CI oracle battery | Scallop adapter for all provenances; k exposed | **G2 gate (day 7):** Scallop(addmult/topk/minmax) vs reference implementations on 50 G1 instances — agreement characterized (any mismatch = a *finding about Scallop*, log it) |
| 8–14 | E1 full grid + F1/F2 production figures; witness synthesis v1 | E2 full sweep on Scallop bounded/convergent; F3 production | inference-side results locked for axes 1–2 (**publishable core exists — day 14**) |
| 15–21 | Error-surface analysis; heterogeneous E1; scorecard pipeline (φ, radar, leaderboard) | E3 on any Scallop soft aggregation; E7 chain-learning prototype | mid-point review with full team: lock paper claims C3(i)(ii) |
| 22–31 | TMC + gradient-fidelity scoring vs oracle ∇ (analytic WMC grads); support B | **E5 + E6**: MNIST-sum_n control and MNIST-path treatment, provenances × k × 5 seeds (GPU-heaviest block) | **G3 gate (day 31):** F6 exists in draft form; H8 answered (either direction is a result) |
| 32–38 | E4 witness table; robustness ablations (prob regimes, topologies) | E8 CLUTRR; E7 finish | external validity in hand |
| 39–45 | Repro pack: configs, seeds, one-command reruns | Figure polish; numbers tables | **MVP complete:** all of F1–F7 in at least draft quality; clears every KR objection |

**Contingencies / cut-lines.** Scallop blocked > 3 days → proceed on reference SUTs (science unblocked), retry Scallop in days 22–31 as external validity. GPU budget tight → E6 at n=2 grid sizes only. Time short → cut G4/H10 first, then E8, never E6 (it is the headline). The day-14 core is the safety floor: inference-side fidelity surfaces + crossover + autopsy is already a complete, novel paper.

**Days 46–90 (award arc).** DeepProbLog cross-check; G4 negation; two-level overlap surfaces; the adjudication writeup (van Krieken / Faronius / Marconato positioning, with our E6 evidence); full draft; artifact release with adapter-contribution docs; (stretch) a DeepLog adapter — framing De Raedt's machine as a substrate the arena scores, with a generous citation posture.

---

## 8. Paper plan

**Abstract (draft, ~170 words).** Neuro-symbolic reasoning systems are compared by task accuracy, yet accuracy cannot tell whether a system computes the semantics it claims. We introduce *semantic fidelity* — the signed, oracle-grounded error between the value a system computes and the value its declared algebra defines — and NeSyArena, an open diagnostic benchmark that measures it under controlled program structure: proof overlap, proof multiplicity, recursion depth, aggregation surrogates, and negation. From an algebraic analysis of rule-based NeSy reasoning we derive signed error laws for common approximations, including a closed-form surrogate bias of τ ln P, and verify them empirically. The arena reveals that popular approximations have opposite structural sensitivities — no method dominates; structure decides — and that the field's default benchmark is provably blind to the central probabilistic failure mode, explaining published null results. Finally, we show that inference-time semantic error propagates to gradient bias and miscalibration of the learned perception module, connecting semantic fidelity to the ongoing debate on approximation assumptions in neurosymbolic learning. All generators, oracles, and system adapters are released; new systems plug in via a 30-line adapter.

**Section budget (~7 content pages, adjust to the CFP):** 1 Intro (1.0) — the accuracy-hides-error figure up front; 2 Background: the algebraic contract *compressed to half a page*, stack as a table, theorems cited not re-proven (this is the direct answer to KR Reviewer C) (0.75); 3 Semantic fidelity: D1–D6 + error laws H1–H7 as short propositions (1.25); 4 Arena architecture + witness synthesis (0.75); 5 Inference experiments E1–E4 (1.25); 6 Learning experiments E5–E8 (1.25); 7 Related work — semiring PLP/AMC, DeepLog, Scallop, van Krieken ×2, Faronius, Marconato/rsbench, PNNL assurance, CLUTRR, Khamis (0.5); 8 Discussion: what the arena cannot see (logic-as-loss systems, non-rule NeSy — *state the scope narrowly upfront*, the lesson from KR Reviewer C) (0.25).

**Terminology decision (binding):** "claimed semantics", "semantic error", "conformance", "witness". The words "contract", "audit", "certificate" do not appear. (KR Reviewer C's terminology objection, accepted in full.)

---

## 9. Review-risk table

| Objection (source) | Kill |
|---|---|
| "No empirical depth" (KR-A) | The paper *is* the empirical instrument; 7 figures, 2 learning suites, external validity on CLUTRR |
| "Only explains already-fixed problems" (KR-B) | Predictions registered (H-table) before runs; H4/H8 are open questions the instrument answers; witness synthesis finds *new* minimal failures mechanically |
| "Borrows from semiring ProbLog; same math re-presented" (KR-C) | Theorems demoted to cited background; the *new* formal objects are D1–D6 and the error laws; the contribution is measurement, which the math alone never produced |
| "PLP, not NeSy" (KR-C) | The learning axis (E5–E8, gradient fidelity, TMC) is constitutively NeSy; scope stated upfront as rule/proof-based NeSy |
| "Non-standard terminology" (KR-C) | resolved by fiat (§8) |
| "Isn't this DeepLog/Scallop?" (anticipated) | They are substrates/SUTs; neither measures fidelity against claimed semantics. Cite generously; offer DeepLog adapter as future work |
| "van Krieken already analyzed operators" (anticipated) | His analysis is fuzzy-logic, loss-level, theory-first; ours is proof-aggregation, model-level, oracle-measured — and E6 supplies the empirical regime map his debate with Faronius lacks |
| "Synthetic only" (anticipated) | E5/E6 on perception pipelines; E8 on CLUTRR |
| "Scallop-specific" (anticipated) | Reference SUTs + ProbLog + (optional) DeepProbLog + adapter standard |
| "Error laws are trivial math" (anticipated) | Individually yes — and we say so; the contribution is the *instrument*, the crossover/no-dominance finding, the autopsy, and H8. Trivial laws with non-trivial consequences is the point |

---

## 10. Repro standards

Fixed seeds everywhere; every figure regenerated by one command from committed configs; CI asserting oracle agreement (reference WMC ≡ ProbLog < 1e-10 on a 50-instance battery); SUT version pinning (Scallop commit hash); raw sweep results stored as parquet; the H-table in the repo updated with measured outcomes (the registered-predictions discipline is part of the artifact's credibility).
