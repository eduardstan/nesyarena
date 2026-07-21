# NeSyArena — measured results (auto-generated, package 0.1.0.dev0)

All numbers produced by the rebuilt core (`src/nesyarena/`) via the
config-driven runners in `experiments/`; configs and their sha256 are
embedded in each JSON. Conformance logs (one per framework):
`conformance_scallop.md` (findings F-1, F-2), `conformance_deeplog.md`,
`conformance_problog_kbest.md`, `conformance_deepproblog.md`,
`conformance_ltn.md`.

## E1 — overlap sweep (fidelity over the G1 grid)

| SUT | phi(overlap) | mean abs err |
|---|---|---|
| add-mult(clamped) | 0.8632 | 0.1404 |
| exact-wmc | 1.0000 | 0.0000 |
| ltn:godel | 1.0000 | 0.1747 |
| ltn:product | 0.9185 | 0.0819 |
| min-max-prob | 0.8228 | 0.1747 |
| top-1-proofs | 0.8569 | 0.1450 |
| top-3-proofs | 0.9701 | 0.0299 |

## E2 — depth horizons (theory: n + 1)

measured: {'2': 3, '4': 5, '6': 7, '8': 9, '10': 11, '12': None, '14': None}
H7 cyclic sum-product: 3.7895 (> 1: True); max-product: 0.7200

## E3 — surrogate bias law

max |bias − tau·ln P| = 2.22e-16

## E4 — minimal witnesses (machine-found, shrunk)

| SUT | P | L | c | p | m | signed error |
|---|---|---|---|---|---|---|
| add-mult(clamped) | 2 | 1 | 0 | 0.6 | 2 | +0.160 |
| exact-wmc | — | — | — | — | — | none |
| ltn:godel | — | — | — | — | — | none |
| ltn:product | 2 | 2 | 1 | 0.6 | 3 | +0.086 |
| min-max-prob | 1 | 2 | 0 | 0.3 | 2 | +0.210 |
| top-1-proofs | 2 | 1 | 0 | 0.3 | 2 | -0.210 |
| top-3-proofs | 4 | 1 | 0 | 0.3 | 4 | -0.103 |

## Gradient liveness (share of oracle-live facts kept alive)

- add-mult(clamped): **0.611**
- exact-wmc: **1.000**
- min-max-prob: **0.166**
- top-1-proofs: **0.498**
- top-3-proofs: **0.832**

## E6 fact-table (population-loss limit, 5 seeds): held-out error

| SUT | control (disjoint) | treatment (overlap) | facts-quality | fact MAE |
|---|---|---|---|---|
| addmult | 0.010 ± 0.020 | 0.101 ± 0.029 | 0.123 | 0.128 |
| addmult_st | 0.010 ± 0.020 | 0.101 ± 0.029 | 0.123 | 0.128 |
| exact | 0.010 ± 0.020 | 0.000 ± 0.000 | 0.000 | 0.029 |
| ltn_godel | 0.071 ± 0.059 | 0.095 ± 0.045 | 0.212 | 0.272 |
| ltn_product | 0.050 ± 0.041 | 0.071 ± 0.017 | 0.081 | 0.110 |
| minmax | 0.071 ± 0.059 | 0.095 ± 0.045 | 0.212 | 0.272 |
| top1 | 0.148 ± 0.076 | 0.109 ± 0.052 | 0.113 | 0.191 |
| top3 | 0.010 ± 0.020 | 0.004 ± 0.003 | 0.004 | 0.041 |

## E6 pixels (H8 headline, 3 seeds): accuracy ties, fidelity diverges

| SUT | accuracy | calibration vs Bayes | transfer | facts-quality |
|---|---|---|---|---|
| addmult | 0.735 ± 0.005 | 0.146 ± 0.005 | 0.094 ± 0.003 | 0.103 ± 0.004 |
| addmult_st | 0.735 ± 0.005 | 0.146 ± 0.005 | 0.094 ± 0.003 | 0.103 ± 0.004 |
| exact | 0.742 ± 0.004 | 0.103 ± 0.003 | 0.064 ± 0.003 | 0.064 ± 0.003 |
| ltn_godel | 0.744 ± 0.005 | 0.127 ± 0.003 | 0.085 ± 0.003 | 0.082 ± 0.002 |
| ltn_product | 0.730 ± 0.006 | 0.130 ± 0.004 | 0.083 ± 0.003 | 0.087 ± 0.003 |
| minmax | 0.744 ± 0.005 | 0.127 ± 0.003 | 0.085 ± 0.003 | 0.082 ± 0.002 |
| top1 | 0.743 ± 0.004 | 0.106 ± 0.003 | 0.070 ± 0.003 | 0.071 ± 0.004 |
| top3 | 0.742 ± 0.004 | 0.103 ± 0.003 | 0.064 ± 0.003 | 0.064 ± 0.003 |

Control (sum structure; exact == addmult by construction):

| SUT | accuracy | calibration | transfer |
|---|---|---|---|
| addmult | 0.669 ± 0.022 | 0.233 ± 0.143 | 0.147 ± 0.025 |
| addmult_st | 0.669 ± 0.022 | 0.233 ± 0.143 | 0.147 ± 0.025 |
| exact | 0.669 ± 0.022 | 0.233 ± 0.143 | 0.147 ± 0.025 |
| ltn_godel | 0.672 ± 0.025 | 0.211 ± 0.118 | 0.159 ± 0.027 |
| ltn_product | 0.664 ± 0.024 | 0.234 ± 0.143 | 0.146 ± 0.024 |
| minmax | 0.672 ± 0.025 | 0.211 ± 0.118 | 0.159 ± 0.027 |
| top1 | 0.644 ± 0.017 | 0.296 ± 0.084 | 0.194 ± 0.015 |
| top3 | 0.669 ± 0.022 | 0.233 ± 0.143 | 0.147 ± 0.025 |

## E7 — depth learning (chain depth 6)

- convergent (fixed point): AUC **0.725 ± 0.012**
- truncated (n=4 < depth 6): AUC **0.480 ± 0.027**

## E5/E6 at MNIST scale (real digits, 3 seeds)

MNIST-path treatment (overlap):

| SUT | accuracy | perception MAE | transfer | facts-quality |
|---|---|---|---|---|
| addmult | 0.930 ± 0.011 | 0.170 ± 0.038 | 0.040 ± 0.004 | 0.040 ± 0.003 |
| exact | 0.955 ± 0.006 | 0.097 ± 0.009 | 0.030 ± 0.001 | 0.030 ± 0.001 |
| ltn_godel | 0.943 ± 0.007 | 0.141 ± 0.006 | 0.039 ± 0.002 | 0.036 ± 0.001 |
| ltn_product | 0.950 ± 0.004 | 0.111 ± 0.012 | 0.032 ± 0.002 | 0.032 ± 0.002 |
| minmax | 0.943 ± 0.007 | 0.141 ± 0.006 | 0.039 ± 0.002 | 0.036 ± 0.001 |
| top1 | 0.953 ± 0.005 | 0.095 ± 0.008 | 0.030 ± 0.001 | 0.030 ± 0.001 |
| top3 | 0.955 ± 0.006 | 0.096 ± 0.009 | 0.030 ± 0.001 | 0.030 ± 0.001 |

MNIST-sum control:

| SUT | accuracy | transfer |
|---|---|---|
| addmult | 0.794 ± 0.097 | 0.260 ± 0.120 |
| exact | 0.794 ± 0.097 | 0.260 ± 0.120 |
| ltn_godel | 0.769 ± 0.109 | 0.275 ± 0.127 |
| ltn_product | 0.758 ± 0.116 | 0.268 ± 0.124 |
| minmax | 0.769 ± 0.109 | 0.275 ± 0.127 |
| top1 | 0.987 ± 0.005 | 0.020 ± 0.005 |
| top3 | 0.794 ± 0.097 | 0.260 ± 0.120 |

## E5b — label-noise ablation (registered predictions, measured verdict)

advantage(exact − top1) by eta: {'0.0': 0.202, '0.15': -0.075, '0.3': -0.036, '0.5': -0.04}
- P1 (monotone decrease): **REFUTED — collapse at the first increment**
- P2 (reversal by eta=0.5): **confirmed**

top-1's advantage on deterministic-perception controls is a
knife-edge artifact: the smallest tested latent noise destroys it.

## E8 — CLUTRR-style generalization (lineal fragment, train k<=3)

learned composition table accuracy: 0.915 ± 0.062

| budget | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.99 | 0.96 | 0.18 | 0.17 | 0.15 | 0.15 | 0.13 | 0.13 | 0.12 |
| 4 | 0.99 | 0.96 | 0.91 | 0.87 | 0.16 | 0.18 | 0.14 | 0.13 | 0.12 |
| 8 | 0.99 | 0.96 | 0.91 | 0.87 | 0.85 | 0.82 | 0.82 | 0.81 | 0.15 |
| convergent | 0.99 | 0.96 | 0.91 | 0.87 | 0.85 | 0.82 | 0.82 | 0.81 | 0.80 |

