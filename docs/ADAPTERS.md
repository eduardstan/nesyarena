# Writing a NeSyArena adapter

An adapter plugs a reasoning backend into the arena so its *semantic
fidelity* can be measured: the signed error between what the backend computes
and what its claimed semantics defines, on the same inputs the exact oracles
consume. ~30 lines is typical.

## The contract

```python
from nesyarena.adapters import Adapter   # typing.Protocol

class MyAdapter:
    name = "mysystem:provenance(k=3)"      # one backend configuration per adapter
    claimed_semantics = "distribution semantics"
    supports_grad = False

    def infer(self, program, base, queries) -> dict:   # {query_atom: float}
        ...
    def grad(self, program, base, query, wrt) -> dict: # {fact_atom: d value/d p_fact}
        ...
```

- `program` is a `nesyarena.ir.GroundProgram`; `base` maps EDB `Atom`s to
  probabilities/weights; `queries` are ground atoms.
- `claimed_semantics` selects the oracle you are scored against
  ("distribution semantics" → exact WMC; "max-join over proof scores" → max;
  idempotent graph algebras → direct graph oracles).
- If your backend consumes formulas rather than programs, derive the proof
  DNF with `program.proof_supports(query, max_depth)` and document the depth
  policy (see `adapters/deeplog.py`).
- If your backend is differentiable, `grad` should return *its own*
  gradients (autograd through the backend, or finite differences through
  `infer` — see `adapters/scallop.py`); never the oracle's.

## The three rules

1. **No normalization.** The adapter reports what the backend computes. If
   the backend deviates from its claim, the adapter must deviate identically;
   a deviation is a *finding about the backend's deployed semantics*, logged
   with the witnessing instance (see `out/conformance_scallop.md`, findings F-1 and F-2, and
   `out/conformance_ltn.md`).
2. **Shared inputs.** Oracle and backend consume the identical `(program,
   base)`; that isolation is what makes the error attributable to the
   reasoning layer.
3. **Pin versions.** Record the backend version/commit in your findings;
   conformance verdicts are per-version. Pin deviations as canary tests that
   flip when upstream changes (see the version-pinned conformance logs in `out/`).

## Worked examples in-tree

| adapter | backend interface | notes |
|---|---|---|
| `adapters/base.py::ReferenceAdapter` | in-process | wraps the reference SUTs; the idealized comparison point |
| `adapters/scallop.py` | scallopy context | program compilation (EDB atoms via `fact/1`), native recursion, FD gradients |
| `adapters/deeplog.py` | DeepLog circuits | DNF-level; constant labels for values, symbolic labels + autograd for gradients |
| `adapters/problog_kbest.py` | ProbLog k-best | program-level; sound anytime bounds, k sweep against exact inference |
| `adapters/deepproblog_standalone.py` | DeepProbLog (original) | program text rendering; exact SDD inference |
| `adapters/ltn.py` | LTNtorch `fuzzy_ops` | DNF-level Provenance; product and Gödel configurations, autograd gradients |

## Checklist before you claim conformance

- [ ] G1 value battery vs the oracle for your claimed semantics (50 instances,
      homogeneous + heterogeneous)
- [ ] recursive battery (chains + the cyclic Section-5 instance) if your
      backend supports recursion — say explicitly what it does on cycles
- [ ] gradient battery on tie-free (heterogeneous) instances if
      `supports_grad`
- [ ] version pinned; deviations logged + pinned as canaries, not fixed

## Two integration routes: Provenance (DNF-level) vs Adapter (program-level)

There are two ways to plug a backend in; pick by what the backend consumes.

**Route 1 — Provenance (DNF-level).** If the backend evaluates a *formula*
(a set of proofs over weighted facts) — like an aggregation strategy or a
fuzzy-logic library — subclass `suts.Provenance`:

```python
from nesyarena.suts import Provenance

class MySUT(Provenance):
    name = "mylib:config"                 # colon convention: backend:config
    claimed = "distribution semantics"    # picks the oracle you are scored on

    def __init__(self):
        import mylib                      # DEFERRED import — the arena must
        self._ops = mylib.ops             # run without your backend installed

    def value(self, proofs, probs): ...   # what the backend computes
    def grad(self, proofs, probs): ...    # the backend's own gradients
```

Steps: (1) put the class in `adapters/<backend>.py` with deferred imports;
(2) do **not** add it to the default `registry()` — extend the opt-in hook
(`registry(include_...)`) so the core keeps running without your package;
(3) add the batched torch op to `learning/structures.py` + a parity test in
`tests/test_learning_parity.py` (values *and* gradients vs your class, on a
tie-free instance, guarded by `pytest.importorskip`); (4) add the SUT key to
the experiment configs you want it in, with a registered prediction comment;
(5) conformance runner + `out/conformance_<backend>.md` + arena row, as in
Route 2. Worked example: `adapters/ltn.py`.

**Route 2 — Adapter (program-level).** If the backend consumes *programs*
(rules + facts, possibly recursive), implement the `Adapter` protocol at the
top of this document. Worked examples: `adapters/scallop.py` (native
recursion), `adapters/problog_kbest.py` (the most readable template).

If the backend's claimed semantics is not distribution semantics, override
`oracle()` with the exact evaluation of *its* claim (see `LTNGodel`, whose
oracle is the Gödel/min-max evaluation) and report the distance to other
semantics separately as cross-semantics distance — conformance and
cross-semantics gaps are different quantities.
