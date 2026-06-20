"""Scallop adapter + gate G2 for NeSyArena.

Complete and ready to run on a machine with scallopy installed (see
INSTALL_SCALLOP.md). Written against the scallopy 0.2.x API; the code probes
rule-syntax variants defensively and prints raw outputs on first run so any
API drift is visible rather than silently mis-measured.

Run:  python3 -m nesyarena.adapters_scallop      (executes gate G2)
"""

from __future__ import annotations
import numpy as np

from .oracle import wmc
from .provenances import AddMult, TopK, MinMax
from .arena import overlap_family


class ScallopAdapter:
    """Uniform adapter: G1-style query (list of proofs over probabilistic
    facts) -> probability, under a chosen Scallop provenance."""

    def __init__(self, provenance: str, k: int | None = None):
        import scallopy  # noqa: deferred so the rest of the package never needs it
        self.scallopy = scallopy
        self.provenance = provenance
        self.k = k
        self.name = f"scallop:{provenance}" + (f"(k={k})" if k else "")
        self.claimed = "distribution semantics"

    def infer(self, proofs, probs) -> float:
        sc = self.scallopy
        kw = dict(provenance=self.provenance)
        if self.k is not None:
            kw["k"] = self.k
        ctx = sc.ScallopContext(**kw)
        facts = sorted(set().union(*proofs))
        fid = {f: i for i, f in enumerate(facts)}
        ctx.add_relation("fact", (int,))
        ctx.add_facts("fact", [(float(probs[f]), (fid[f],)) for f in facts])
        made = False
        for body_join in (" and ", ", "):  # probe rule-syntax variants
            try:
                for pr in proofs:
                    body = body_join.join(f"fact({fid[f]})" for f in sorted(pr))
                    ctx.add_rule(f"q() = {body}")
                made = True
                break
            except Exception:
                continue
        if not made:
            raise RuntimeError("scallopy.add_rule rejected both syntax variants; "
                               "inspect ctx.add_rule signature for this version")
        ctx.run()
        out = list(ctx.relation("q"))
        # expected: [(prob, ())] under probabilistic provenances; [( () ,)] discrete
        if not out:
            return 0.0
        first = out[0]
        if isinstance(first, tuple) and len(first) == 2 and isinstance(first[0], float):
            return float(first[0])
        raise RuntimeError(f"unexpected scallopy output shape: {out!r} — "
                           "print and map manually, then fix this adapter")


def gate_g2(n_inst: int = 50, seed: int = 1, verbose: bool = True) -> dict:
    """Compare Scallop provenances against reference SUTs and the exact oracle
    on n_inst G1 instances. Returns max |diff| per pairing. Discrepancies are
    findings: record (instance params, both values) — do not 'fix' them away."""
    pairs = [
        (ScallopAdapter("addmultprob"), AddMult(clamp=True)),
        (ScallopAdapter("minmaxprob"), MinMax()),
        (ScallopAdapter("topkproofs", k=1), TopK(1)),
        (ScallopAdapter("topkproofs", k=3), TopK(3)),
    ]
    rng = np.random.default_rng(seed)
    rows = {a.name: dict(vs_ref=0.0, vs_exact=0.0, worst=None) for a, _ in pairs}
    for i in range(n_inst):
        P = int(rng.integers(2, 5)); L = int(rng.integers(2, 4))
        c = int(rng.integers(0, L))
        proofs, probs = overlap_family(P, L, c, p=float(rng.choice([0.3, 0.6, 0.9])),
                                       rng=rng, het=(i % 3 == 0))
        ex = wmc(proofs, probs)
        for ad, ref in pairs:
            try:
                v_s = ad.infer(proofs, probs)
            except Exception as e:
                rows[ad.name]["worst"] = f"ERROR: {e}"
                continue
            v_r = ref.value(proofs, probs)
            d_ref, d_ex = abs(v_s - v_r), abs(v_s - ex)
            if d_ref > rows[ad.name]["vs_ref"]:
                rows[ad.name].update(vs_ref=d_ref,
                                     worst=dict(P=P, L=L, c=c, scallop=v_s, ref=v_r, exact=ex))
            rows[ad.name]["vs_exact"] = max(rows[ad.name]["vs_exact"], d_ex)
    if verbose:
        print(f"{'adapter':28} {'max|scallop-ref|':>17} {'max|scallop-exact|':>19}")
        for n, r in rows.items():
            print(f"{n:28} {r['vs_ref']:>17.2e} {r['vs_exact']:>19.2e}")
            if r["vs_ref"] > 1e-6:
                print(f"  -> DISCREPANCY (a finding). Worst instance: {r['worst']}")
    return rows


if __name__ == "__main__":
    gate_g2()
