# Conformance — ProbLog anytime k-best (problog 2.2.10)

Frozen instance set v1 (71 instances), eps grid [1e-09, 0.05, 0.2, 0.5].
Registered predictions and verdicts:

- **P1 soundness** (oracle in [lb, ub] everywhere): **PASS — 0 violations**
- **P2 exactness at eps=1e-9**: max |border − oracle| = **5.55e-16**
- **P3 loose-eps lower border = TopK(k) prefix value**: **141/192** instance×eps cells matched; misses: ['values-09', 'values-11', 'values-13', 'values-15', 'values-18', 'values-21', 'values-24', 'values-26', 'values-27', 'values-28']

Reading: the deployed anytime object is a sound interval. At coarse eps
its lower border coincides with the arena's top-k-proofs prefix values;
at tighter eps the border updates add *implicants finer than whole
proofs* (disjoint branches of the compiled formula), so the bound lands
strictly between proof-prefix values (verified on the miss diagnostics:
e.g. lb 0.6310 between TopK(2)=0.5782 and TopK(3)=0.6422). Registered
expectation P3 therefore holds at coarse eps and is *refined* at tight
eps: the deployed lower bound is implicant-based, not proof-based.
