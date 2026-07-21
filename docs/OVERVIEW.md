# NeSyArena — overview for new readers

This is the document to read first. It explains what the project does, how the
pieces fit together, how to use them, and what is and isn't done yet. No prior
context assumed. (Formal definitions and proofs are in the accompanying
paper; this file is the plain-language bridge.)

---

## 1. The one idea

Neuro-symbolic (NeSy) systems combine a neural network (perception) with a
symbolic reasoner. The reasoner is supposed to compute some well-defined
quantity — e.g. *"the probability that query `q` is true, given the
probabilities of the facts"*. But to be fast and differentiable, real systems
**approximate** that quantity: they sum over proofs (which double-counts
overlapping explanations), keep only the top-`k` proofs (which drops mass),
truncate recursion at a fixed depth, or replace `max` with a smooth softmax.

The field compares these systems by **task accuracy**. Our claim:

> Accuracy cannot tell whether a reasoner computes the semantics it *claims*.
> A system that computes a *distorted* value can still rank answers correctly
> (so it classifies correctly) while being badly miscalibrated as a
> probabilistic reasoner — and while silently corrupting the perception module
> that is trained through it.

**NeSyArena measures that distortion directly.** For a reasoner `S` we define

```
semantic error   ε_S(q) = (what S computes)  −  (what S's claimed semantics defines)
```

The second term is computed **exactly**, by an independent oracle, on the
**same** inputs `S` sees. So `ε` isolates the reasoning layer: it is not
confounded by perception quality, because perception is held fixed. We then
study `ε` as a function of **program structure** (how many proofs, how much
they overlap, how deep the recursion), and we measure what `ε` does to
learning (gradient quality, calibration, transfer).

That's the whole project. Everything below is machinery to make `ε` cheap,
exact, and reproducible, and experiments that report it.

---

## 2. How the pieces fit together (data flow)

```
   generators.py                 a PROGRAM  (rules)                ir.py
  ───────────────▶  +  a BASE INTERPRETATION f_θ  (fact probabilities)
                                      │
                                      │   bounded proof enumeration  (ir.py)
                                      ▼
                         the QUERY's PROOFS
                    (every bounded way to derive q,
                     as sets of facts that must hold)
                                      │
                 ┌────────────────────┴─────────────────────┐
                 ▼                                           ▼
        ORACLE (oracle.py)                       SYSTEM UNDER TEST
   the CORRECT value under the              what an approximation computes
   *claimed* semantics                      reference: suts.py
   (exact WMC / ProbLog / graph algs)       deployed:  adapters/ (Scallop, DeepLog)
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       ▼
              semantic error   ε = (SUT value) − (oracle value)      ← D1
                                       │
                                       ▼
        metrics.py  (fidelity profile, depth horizon, gradient liveness)
        witness.py  (smallest program where ε is large)
        learning/   (train perception through S; measure calibration & transfer)
```

**The invariant that makes it work:** the oracle and the system under test are
handed the *identical* `(program, fact-probabilities)`. They differ only in
*how they aggregate proofs*. So any disagreement is attributable to the
reasoning approximation, nothing else. This is the single most important
design decision in the repo.

---

## 3. A concrete example (run it in 5 lines)

Take a query `q` with **two independent proofs**, each a single fact with
probability `0.6` (think: "`q` holds if `a` holds, OR if `b` holds").

```python
from nesyarena.generators import overlap_family
from nesyarena.suts import ExactWMC, AddMult, TopK, MinMax

inst = overlap_family(P=2, L=1, c=0, p=0.6)   # 2 proofs, 1 fact each, no shared facts

ExactWMC().value(inst.proofs, inst.probs)   # 0.84   ← the TRUTH: P(a∨b)=0.6+0.6−0.6·0.6
AddMult().value(inst.proofs, inst.probs)     # 1.00   ← proof-sum: 0.6+0.6=1.2, clamped to 1
TopK(1).value(inst.proofs, inst.probs)       # 0.60   ← keeps only the best proof
MinMax().value(inst.proofs, inst.probs)      # 0.60

AddMult().error(inst.proofs, inst.probs)     # +0.16  ← over-counts (signed!)
TopK(1).error(inst.proofs, inst.probs)       # −0.24  ← under-counts
```

Same proofs, same probabilities, four different answers, and the truth is
`0.84`. `add-mult` over-counts (the two proofs overlap as *events* even though
they share no facts); `top-1` under-counts (it threw a proof away). The signs
are opposite — that is the "crossover" finding, and it is why **no single
approximation is best; the program structure decides** (experiment E1).

---

## 4. The components, by role

Grouped by what they do, with who-calls-what. Source is `src/nesyarena/`.

### Representation — *how a problem is written down*
- **`ir.py`** — the data structures. `Atom` (a fact like `edge(a,b)`), `Rule`
  (`head ← body`), `GroundProgram` (a set of rules). Plus the key algorithm:
  given a program and a query, **enumerate its proofs up to a depth bound**.
  Two views of a proof: its *multiset of leaf facts* (for semiring evaluation)
  and its *set of facts* (the "support", used by the probabilistic oracle and
  SUTs). Everything downstream consumes `GroundProgram`s, including the
  external adapters — that is what lets the oracle and Scallop score the *same*
  object.
- **`algebra.py`** — the **semiring registry**: `BOOLEAN` (reachability),
  `MAXPROD` (max-reliability path), `SUMPROD` (sum-product), `TROPICAL`
  (shortest path). A semiring fixes two operators: `⊗` = how to combine facts
  *along* one proof, `⊕` = how to aggregate *across* proofs. **This is not
  per-system configuration** — it is the set of exact algebras the *engine*
  can evaluate a program under (see §6.3 for how you pick one). The
  probabilistic reasoners under test live in `suts.py`, not here.

### Ground truth — *the correct answer*
- **`engine.py`** — runs a program under a chosen semiring: bounded
  iteration of the immediate-consequence operator `T_P` (`infer_bounded`),
  run-to-convergence (`converge`), and the proof-side aggregation that must
  equal it (`proof_aggregate`). The equality of these two is a classical
  theorem; we check it numerically as a test.
- **`oracle.py`** — the probabilistic ground truth: **exact weighted model
  counting** (`wmc`) — enumerate all `2^m` fact-assignments, sum the
  probability of those satisfying `q` — *with analytic gradients*
  (`wmc_with_grad`). For >22 facts, ProbLog's exact inference is the oracle;
  for the idempotent graph algebras, direct graph algorithms. These are what
  `ε` is measured against.

### Systems under test — *the approximations*
- **`suts.py`** — reference implementations of the deployed aggregation
  strategies, each with the *same gradient behaviour real systems use*:
  `ExactWMC` (no error, the floor), `AddMult` (proof-sum, clamped — over-counts),
  `TopK(k)` (keep best k — under-counts), `MinMax` (max-of-min), `LSE(τ)`
  (softmax surrogate). Each declares its **claimed semantics** (which oracle it
  is scored against) and exposes `.value(...)`, `.error(...)`, `.grad(...)`.
- **`adapters/`** — the same interface, but wrapping a *real external system*
  (see §5).

### Structured inputs — *the controlled experiments*
- **`generators.py`** — parameterized program families that isolate one axis
  each: `overlap_family` (G1: `P` proofs of length `L` sharing `c` facts at
  probability `p`), `chain_family`/`cyclic_family` (G2: recursion depth),
  `surrogate_scores` (G3), and the CLUTRR-style kinship generator (E8). Each
  returns an `Instance` with `.program`, `.query`, `.probs`, `.proofs`.

### Scoring — *turning ε into numbers and figures*
- **`metrics.py`** — `fidelity` (1 − mean |ε| over a sweep), `depth_horizon`
  (the depth at which a truncating reasoner breaks), `gradient_liveness`
  (fraction of facts that still receive a gradient).
- **`witness.py`** — given a reasoner, *search* generator space for the
  smallest program where |ε| exceeds a threshold, then shrink it — an
  automatically-found minimal failing example.

### Learning — *the consequences of ε for training*
- **`learning/`** — each reasoner as a batched **PyTorch** operation whose
  ordinary autograd reproduces that system's faithful gradients. This lets a
  perception network be trained *through* a reasoner with normal
  `loss.backward()`, so we can measure how the reasoner's distortion corrupts
  the learned network (calibration against the exact Bayes posterior; transfer
  to held-out queries).

### Experiments & reproduction
- **`experiments/`** — one runner per experiment (E1–E8 + scorecard), each
  reading a committed YAML config and writing JSON + a figure to `out/`.
  `make all` runs the whole suite. `experiments/report.py` collates `out/` into
  `out/RESULTS.md`.
- **`tests/`** — the correctness contract (102 tests): the oracle agrees with
  ProbLog to 1e-10, analytic gradients match finite differences, the error
  laws hold, and the rebuilt code reproduces the frozen
  golden fixtures in `tests/fixtures/`.

---

## 5. How an external system plugs in (e.g. Scallop)

**First, why adapters exist at all.** A natural objection: *"given a program
and the declared algebra, isn't that enough to do the reasoning? what is the
adapter for?"* Yes — the algebra plus the engine already do the reasoning
**correctly**; that is exactly what the oracle is. But the goal of this project
is not to do reasoning. It is to measure whether a *real, deployed* system
computes what it claims. Real systems do **not** run the declared algebra —
they approximate it (for speed and differentiability), and those deviations are
the object of study. You cannot obtain "what Scallop actually returns" by
running the algebra; you have to run *Scallop itself* on the same program. The
adapter is the bridge that lets an external system consume our program and hand
back its (possibly distorted) answer, untouched, so it can be compared to the
oracle. Without adapters there would be nothing to compare the oracle
against — you would be comparing the theory to itself.

> **The analogy:** it is a **conformance test for a compiler**. The algebra is
> the *language spec* (what a program *should* evaluate to). A NeSy system
> (Scallop, DeepLog) is a *compiler* — a real implementation that makes
> approximations and engineering choices, and may deviate. The adapter is the
> *test harness* that runs the actual compiler on the test program and captures
> its output. The oracle computes the spec's expected output. The result is the
> difference. You cannot test a compiler by only reading the spec; you must run
> it.

An **adapter** (`adapters/base.py` defines the interface) wraps one
configuration of a backend and answers queries over the *same* `GroundProgram`
the oracle uses:

```python
class Adapter(Protocol):
    name: str                 # e.g. "scallop:topkproofs(k=3)"
    claimed_semantics: str    # e.g. "distribution semantics"  → picks the oracle
    def infer(self, program, base, queries) -> dict[Atom, float]: ...
    def grad (self, program, base, query, wrt) -> dict[Atom, float]: ...
```

**Scallop** (`adapters/scallop.py`) is a real NeSy engine with selectable
"provenances" (its name for aggregation strategies: `addmultprob`,
`topkproofs`, `minmaxprob`, …). The adapter:
1. compiles our `GroundProgram` to Scallop's own syntax (facts become a
   relation, rules are emitted as-is, recursion runs natively);
2. runs Scallop and reads back the query probability;
3. returns it — **without any correction**.

Then we compare Scallop's number to the exact oracle on the identical program.
On non-recursive programs Scallop reproduces our reference SUTs to ~1e-16
(machine precision) — which *validates* that the reference SUTs faithfully
model the deployed system — while deviating from the exact value by exactly
the amounts the error laws predict. See `out/conformance_scallop.md`. **DeepLog** (`adapters/deeplog.py`) is handled the
same way; it compiles to exact circuits and passes conformance.

> Scallop needs the `scallopy` package, which only ships for Python 3.10 — see
> `INSTALL_SCALLOP.md` for the one-command install in a dedicated environment.
> The rest of the arena runs without it; the reference SUTs stand in.

**The principle (and a project rule):** if a deployed system disagrees with
its own claimed semantics, that is a *finding about the system*, recorded with
the witnessing instance — never patched away.

---

## 6. How you actually use it

### 6.1 Run the suite
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # pinned full environment (all libraries)
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest        # the correctness contract (~30 s)
make all                          # every experiment → out/*.json, out/*.png, out/RESULTS.md
```

### 6.2 Score one program by hand
See §3 — pick or build an `Instance`, call `.value()`/`.error()` on any SUT.

### 6.3 Choose an algebra and run the engine
`algebra.py` is *used by the engine*: you pass a semiring to tell the engine
how to evaluate a program. Example — the truncation effect on a chain whose
query needs 5 reasoning steps:

```python
from nesyarena.generators import chain_family
from nesyarena.algebra import MAXPROD
from nesyarena.engine import infer_bounded, converge

inst = chain_family(L=5, p=0.9)                      # path(v0,v5): minimal proof depth 5
infer_bounded(inst.program, dict(inst.probs), MAXPROD, inst.query, n=3)  # 0.0  (truncated: too shallow)
converge   (inst.program, dict(inst.probs), MAXPROD, inst.query)         # 0.59049 = 0.9**5
```

Swap `MAXPROD` for `BOOLEAN` to get reachability (1.0), `TROPICAL` for shortest
path, `SUMPROD` for sum-product. The algebra is a *parameter of the engine*,
not a property baked into a system.

### 6.4 Measure a deployed system
```python
from nesyarena.adapters.scallop import ScallopAdapter      # needs scallopy
ad = ScallopAdapter("topkproofs", k=3)
ad.infer(inst.program, inst.probs, [inst.query])           # what Scallop computes
# compare to: from nesyarena.oracle import wmc; wmc(inst.proofs, inst.probs)
```

### 6.5 Add your own system
Write ~30 lines implementing the `Adapter` interface; see **`docs/ADAPTERS.md`**
for a step-by-step guide and a conformance checklist.

---

## 7. What is done, and what is missing

**Done (in this repo, reproducible with `make all`):**
- The full measurement instrument: IR, exact oracles with analytic gradients,
  reference SUTs, structured generators, scoring, witness search, the torch
  learning layer — 102 passing tests, gated against a frozen reference
  implementation.
- All eight experiments (E1–E8): the structural crossover; depth horizons and
  recursion divergence; the closed-form surrogate-bias law; machine-found
  witnesses; the headline learning result (accuracy ties across reasoners
  while calibration and transfer separate them) on synthetic pixels **and** on
  real MNIST digits; end-to-end gradient starvation; CLUTRR-style train-short/
  test-long generalization cliffs at the predicted depth.
- Conformance of two deployed substrates (Scallop, DeepLog) measured against
  the oracle, including gradients.
- A written technical report and a one-command reproduction (`make all`).

**Missing / next (this is where help is welcome):**
- **Breadth of evidence for a paper submission:** lift every learning run to
  ≥5 seeds (some are at 3); more seeds on the high-variance MNIST-sum control.
- **More external systems:** Lobster (GPU Scallop); additional provenances /
  differentiable-`topk` gradient route; the negation/stratification axis is
  specified but not yet ported to the rebuilt engine.
- **Scaling:** the brute-force oracle caps at 22 facts (learning structures at
  16); a compilation-based oracle would lift this.
- **External validity:** real-text CLUTRR (we currently use a controlled
  symbolic stand-in); random-program sweeps beyond the designed families.
- **The recursive case:** the reference add-mult SUT models deployed behaviour
  on acyclic programs; a recursion-faithful variant is specified but not built.

**Bottom line:** the instrument and the core scientific results exist and
reproduce. What remains is breadth (seeds, systems, scale), a couple of
specified-but-unbuilt pieces, and the writing — not new core machinery. That's
the honest "what's left", and most of it is parallelizable.

---

## 8. Anticipated questions

**Q. Isn't the declared algebra enough to do the reasoning? What is the adapter
for?** Yes, the algebra + engine reason correctly — that is the oracle. The
adapter exists because the goal is to *audit a real system*, which requires
running that system (it approximates the algebra) and comparing it to the
oracle. See §5 for the full answer and the compiler-conformance analogy. In one
line: **the algebra tells you what the answer *should* be; the adapter tells
you what a real system *actually* returns; the science is the gap.**

**Q. Then what is the difference between `algebra.py`, `suts.py`, the oracle,
and an adapter?** Four distinct roles:
| | what it is |
|---|---|
| `algebra.py` + `engine.py` | the rules and the machinery to evaluate a program under them — the *ideal* computation |
| `oracle.py` | the *exact* value of the claimed semantics (the ground truth `ε` is measured against) |
| `suts.py` | *approximation strategies* re-implemented in-house (add-mult, top-k, …), with error we can characterise analytically |
| `adapters/` | a *real external system* (Scallop, DeepLog), wrapped to consume the same program and report what it actually computes |
`algebra.py` is **not** the configuration of a system under test; it is the set
of exact semirings the engine evaluates under. The systems under test are the
SUTs and the adapters.

**Q. Do I need Scallop or DeepLog installed to use the arena?** No. The whole
instrument and all experiments run on the reference SUTs alone. Scallop/DeepLog
are *external-validity checks*; they need extra packages (`scallopy` is
Python-3.10-only — see `INSTALL_SCALLOP.md`). Without them, the reference SUTs
stand in as the idealised models.

**Q. A deployed system disagrees with its claimed semantics — isn't that just a
bug to fix?** No — it is the *result*. The arena records the disagreement with
the witnessing instance and never patches it. Whether the deviation is an
intentional approximation or an outright defect, the point is that it is
invisible to accuracy and only the oracle-grounded measurement reveals it.

**Q. Why measure signed error against an oracle instead of just task accuracy?**
Because a reasoner that computes a *monotone distortion* of the correct value
ranks answers correctly — so it classifies correctly — while being arbitrarily
miscalibrated as a probabilistic reasoner, and while corrupting the perception
network trained through it. The experiments show exactly this: accuracy ties
across five reasoners while calibration and transfer separate them.

**Q. Why is perception synthetic in some experiments?** Because calibration is
measured against the *exact Bayes posterior*, which is only available in closed
form when the data-generating model is known. The MNIST experiments confirm the
findings survive real images; the synthetic generator is what makes the
reasoning-layer error exactly measurable.

**Q. Isn't this just probabilistic logic programming / semiring provenance
again?** The algebraic background is inherited from that literature and is cited
as such. What is new
here is the *measurement*: the fidelity definitions, the signed error laws as
registered predictions, the exact oracles with analytic gradients, the
conformance measurements of deployed systems, and the learning-consequence
results — none of which the algebra alone produces.
