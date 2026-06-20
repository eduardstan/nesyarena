# NeSyArena — measured results (auto-generated, package 0.1.0.dev0)

All numbers produced by the rebuilt core (`src/nesyarena/`) via the
config-driven runners in `experiments/`; configs and their sha256 are
embedded in each JSON. Conformance/finding logs: `G2_scallop.md`,
`G2b_scallop_ir.md` (finding F-1), `conformance_deeplog.md`.

## E1 — overlap sweep (fidelity over the G1 grid)

| SUT | phi(overlap) | mean abs err |
|---|---|---|
| add-mult(clamped) | 0.8632 | 0.1404 |
| exact-wmc | 1.0000 | 0.0000 |
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
| exact | 0.010 ± 0.020 | 0.000 ± 0.000 | 0.000 | 0.029 |
| minmax | 0.071 ± 0.059 | 0.095 ± 0.045 | 0.212 | 0.272 |
| top1 | 0.148 ± 0.076 | 0.109 ± 0.052 | 0.113 | 0.191 |
| top3 | 0.010 ± 0.020 | 0.004 ± 0.003 | 0.004 | 0.041 |

## E6 pixels (H8 headline, 3 seeds): accuracy ties, fidelity diverges

| SUT | accuracy | calibration vs Bayes | transfer | facts-quality |
|---|---|---|---|---|
| addmult | 0.734 ± 0.006 | 0.145 ± 0.002 | 0.093 ± 0.002 | 0.101 ± 0.002 |
| exact | 0.740 ± 0.004 | 0.101 ± 0.003 | 0.062 ± 0.002 | 0.062 ± 0.002 |
| minmax | 0.742 ± 0.007 | 0.128 ± 0.002 | 0.085 ± 0.004 | 0.084 ± 0.001 |
| top1 | 0.744 ± 0.005 | 0.105 ± 0.004 | 0.068 ± 0.003 | 0.069 ± 0.003 |
| top3 | 0.740 ± 0.004 | 0.101 ± 0.003 | 0.063 ± 0.003 | 0.063 ± 0.003 |

Control (sum structure; exact == addmult by construction):

| SUT | accuracy | calibration | transfer |
|---|---|---|---|
| addmult | 0.673 ± 0.005 | 0.192 ± 0.098 | 0.150 ± 0.028 |
| exact | 0.673 ± 0.005 | 0.192 ± 0.098 | 0.150 ± 0.028 |
| minmax | 0.677 ± 0.008 | 0.181 ± 0.084 | 0.160 ± 0.027 |
| top1 | 0.656 ± 0.003 | 0.250 ± 0.066 | 0.189 ± 0.016 |
| top3 | 0.673 ± 0.005 | 0.192 ± 0.098 | 0.150 ± 0.028 |

## E7 — depth learning (chain depth 6)

- convergent (fixed point): AUC **0.722 ± 0.014**
- truncated (n=4 < depth 6): AUC **0.490 ± 0.026**

## E5/E6 at MNIST scale (real digits, 3 seeds)

MNIST-path treatment (overlap):

| SUT | accuracy | perception MAE | transfer | facts-quality |
|---|---|---|---|---|
| addmult | 0.934 ± 0.006 | 0.160 ± 0.016 | 0.040 ± 0.002 | 0.040 ± 0.001 |
| exact | 0.952 ± 0.003 | 0.097 ± 0.008 | 0.031 ± 0.001 | 0.031 ± 0.001 |
| minmax | 0.940 ± 0.006 | 0.140 ± 0.006 | 0.038 ± 0.002 | 0.036 ± 0.002 |
| top1 | 0.951 ± 0.002 | 0.095 ± 0.006 | 0.031 ± 0.001 | 0.031 ± 0.001 |
| top3 | 0.952 ± 0.004 | 0.097 ± 0.009 | 0.031 ± 0.001 | 0.031 ± 0.001 |

MNIST-sum control:

| SUT | accuracy | transfer |
|---|---|---|
| addmult | 0.833 ± 0.109 | 0.221 ± 0.142 |
| exact | 0.833 ± 0.109 | 0.221 ± 0.142 |
| minmax | 0.810 ± 0.125 | 0.236 ± 0.151 |
| top1 | 0.987 ± 0.006 | 0.019 ± 0.004 |
| top3 | 0.833 ± 0.109 | 0.221 ± 0.142 |

## E5b — label-noise ablation (registered predictions, measured verdict)

advantage(exact − top1) by eta: {'0.0': 0.202, '0.15': -0.075, '0.3': -0.036, '0.5': -0.04}
- P1 (monotone decrease): **REFUTED — collapse at the first increment**
- P2 (reversal by eta=0.5): **confirmed**

top-1's advantage on deterministic-perception controls is a
knife-edge artifact: the smallest tested latent noise destroys it.

## E8 — CLUTRR-style generalization (lineal fragment, train k<=3)

learned composition table accuracy: 0.885 ± 0.054

| budget | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.98 | 0.95 | 0.18 | 0.16 | 0.16 | 0.15 | 0.14 | 0.13 | 0.13 |
| 4 | 0.98 | 0.95 | 0.88 | 0.82 | 0.16 | 0.16 | 0.14 | 0.13 | 0.12 |
| 8 | 0.98 | 0.95 | 0.88 | 0.82 | 0.80 | 0.75 | 0.76 | 0.73 | 0.14 |
| convergent | 0.98 | 0.95 | 0.88 | 0.82 | 0.80 | 0.75 | 0.76 | 0.73 | 0.73 |

