# legacy/ — the Day-0 toy implementation (executable specification)

This is the verified toy that produced every number in `../out/RESULTS.md` and
`../paper/main.tex`. It stays in the tree as the executable specification for
the rebuild (parity fixtures in `../tests/fixtures/toy_golden.json` were
generated from it via `gen_golden_from_toy.py`). Delete it only when the
rebuilt package reproduces all of its results.

Run from inside this directory (the toy package shares the `nesyarena` import
name with the rebuild, so never put both on one `sys.path`):

```bash
cd legacy
../.venv/bin/python -m tests.run_ci
../.venv/bin/python -m experiments.run_all     # writes to ../out via NESYARENA_OUT or default
../.venv/bin/python gen_golden_from_toy.py ../tests/fixtures/toy_golden.json
```
