import json
import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def gold():
    return json.loads((FIXTURES / "toy_golden.json").read_text())


def rand_overlap_instance(rng, P=None, L=None, c=None, p=None):
    """Ad-hoc G1-shaped instance over string facts (mirrors the toy generator;
    the generators module supersedes this for program-level work)."""
    P = P if P is not None else int(rng.integers(1, 6))
    L = L if L is not None else int(rng.integers(2, 5))
    c = c if c is not None else int(rng.integers(0, L))
    p = p if p is not None else float(rng.choice([0.3, 0.6, 0.9]))
    probs, proofs = {}, []
    shared = [f"s{i}" for i in range(c)]
    for f in shared:
        probs[f] = p
    for j in range(P):
        priv = [f"x{j}_{i}" for i in range(L - c)]
        for f in priv:
            probs[f] = p
        proofs.append(frozenset(shared + priv))
    return proofs, probs
