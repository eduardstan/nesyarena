# Installing Scallop — definitive guide (forensics done 2026-06-10)

## ✅ Working install on this machine (verified 2026-06-10)

The GitHub release wheel route works on this Linux box — the wheel is
cp310-only, so it lives in a dedicated conda env, not the project venv:

```bash
~/miniconda3/bin/conda create -y -n scallop-py310 python=3.10
curl -sL -O https://github.com/scallop-lang/scallop/releases/download/0.2.4/scallopy-0.2.4-cp310-cp310-manylinux_2_27_x86_64.whl
~/miniconda3/envs/scallop-py310/bin/pip install ./scallopy-0.2.4-cp310-cp310-manylinux_2_27_x86_64.whl numpy
~/miniconda3/envs/scallop-py310/bin/python -m experiments.conformance_scallop   # run from the repo root
```

Result: **conformance pass at machine precision** — see
`out/conformance_scallop.md`.

## The wheel mystery, solved

PyPI hosts **exactly one** scallopy wheel:

```
scallopy-0.1.0-cp39-cp39-macosx_11_0_arm64.whl
```

So `pip install scallopy` works **only on Apple Silicon macOS with Python 3.9**
— if you once installed Scallop "from a wheel" with plain pip, that was the
setup. Note it is version 0.1.0 (ancient; current Scallop is 0.2.x), so do
NOT use it for the arena: provenance behavior has changed since.

## Recommended installs (pick one)

1. **GitHub release wheels (easiest on any machine).**
   https://github.com/scallop-lang/scallop/releases — download the
   `scallopy-*-cp3XX-*-<your platform>.whl` matching `python3 --version`,
   then `pip install ./scallopy-...whl`.
   (These assets are served from `objects.githubusercontent.com`.)

2. **Source build (canonical, what the repo's makefile does).**
   ```bash
   curl https://sh.rustup.rs -sSf | sh    # need Rust >= 1.85 (edition2024 deps)
   pip install maturin
   git clone https://github.com/scallop-lang/scallop && cd scallop
   make install-scallopy    # maturin build --release + pip install the wheel
   ```

3. **Docker** — the scallop-lang repo ships a Dockerfile; use it if rustup is
   not an option.

## Why it cannot be installed in the analysis container (verified)

- No compatible PyPI wheel (see above; container is Linux/cp312).
- apt Rust is 1.75; the dependency `wit-bindgen-rust-macro 0.51.0` requires
  the `edition2024` Cargo feature (Cargo >= 1.85). Pinning was attempted;
  even `cargo generate-lockfile` stalls on a git-sourced dependency
  (`chronoutil` from a personal repo) under the egress proxy.
- GitHub release assets redirect to `objects.githubusercontent.com`, which is
  outside the container's network allowlist.
- `rustup`'s domains are likewise outside the allowlist.

Consequence (by design of the protocol): the reference provenances in
`nesyarena/provenances.py` are the systems-under-test in-container; Scallop is
external validity, run on an unrestricted machine via
`experiments/conformance_scallop.py`.

## First thing to run after installing

```bash
~/miniconda3/envs/scallop-py310/bin/python -m experiments.conformance_scallop
```

The conformance runner compares Scallop's provenances (values, recursion and
diff* gradients) against
the reference SUTs and the exact-WMC oracle on 50 G1 instances. **Any
discrepancy is a finding about Scallop's deployed semantics — log it with the
instance, do not normalize it away.**
