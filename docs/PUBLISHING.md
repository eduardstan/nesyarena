# Publishing NeSyArena to PyPI

This guide outlines the exact commands to build, check, rehearse on TestPyPI, and publish `nesyarena` to PyPI.

> **Note:** Do not add `build` or `twine` to project dependencies. Install them in a temporary virtual environment for packaging.

---

## 1. Build and Validate Artifacts

First, install the packaging tools in an isolated virtual environment and build the source distribution (`.tar.gz`) and wheel (`.whl`):

```bash
python3 -m venv /tmp/build_venv
/tmp/build_venv/bin/pip install build twine

# Clean any existing build artifacts
rm -rf dist build src/nesyarena.egg-info

# Build source distribution and wheel
/tmp/build_venv/bin/python -m build

# Validate README rendering and metadata
/tmp/build_venv/bin/twine check dist/*
```

---

## 2. TestPyPI Rehearsal

Upload the built artifacts to TestPyPI and verify installation from a clean virtual environment:

### Upload to TestPyPI
```bash
/tmp/build_venv/bin/twine upload --repository testpypi dist/*
```

### Verification from TestPyPI
Create a clean virtual environment and install `nesyarena` from TestPyPI (using PyPI as fallback for dependencies like `numpy`):

```bash
python3 -m venv /tmp/testpypi_verify_venv
/tmp/testpypi_verify_venv/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nesyarena
/tmp/testpypi_verify_venv/bin/python -c "import nesyarena, nesyarena.generators, nesyarena.suts; print('TestPyPI install verified. Version:', nesyarena.__version__)"
```

---

## 3. Production PyPI Release

Once TestPyPI rehearsal succeeds, publish the release to PyPI:

### Upload to PyPI
```bash
/tmp/build_venv/bin/twine upload dist/*
```

### Production Verification
Verify the public PyPI release in a fresh virtual environment:

```bash
python3 -m venv /tmp/pypi_verify_venv
/tmp/pypi_verify_venv/bin/pip install nesyarena
/tmp/pypi_verify_venv/bin/python -c "import nesyarena, nesyarena.generators, nesyarena.suts; print('PyPI release verified. Version:', nesyarena.__version__)"
```
