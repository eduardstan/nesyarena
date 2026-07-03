# Conformance — DeepProbLog standalone (deepproblog 2.0.6, ExactEngine)

The original NeurIPS-2018 system, run on the frozen instance set v1
(all 71 instances, every battery incl. the recursive ones).
Registered prediction: exact at compilation precision — **verdict:
PASS**, max |value − oracle| = 4.44e-16 overall.

| battery | max abs dev |
|---|---|
| chains | 0.00e+00 |
| cyclic | 0.00e+00 |
| gradients | 4.44e-16 |
| probes | 1.11e-16 |
| values | 4.44e-16 |
| witnesses | 1.11e-16 |

Scope: the ApproximateEngine (DPLA*) requires SWI-Prolog/PySwip and is
not yet measured — queued as the natural follow-up. Gradients:
constant-probability programs expose no differentiable path in this
system (learning flows through neural predicates only).
