"""Batched provenance evaluation over a fixed proof structure, in torch.

A BatchStructure pins a proof DNF over m facts (m <= ~16: world enumeration
is 2^m) and evaluates a batch of per-sample fact-probability vectors under
each provenance, as a differentiable torch op. The crucial property — gated
by tests/test_learning_parity.py — is that *plain autograd* through these ops
reproduces each deployed system's own differentiation semantics:

  exact    world-sum WMC; autograd == analytic dP/dp_f = P(q|f) - P(q|not f)
  addmult  clamped proof-sum; torch.clamp gives the flat-region blackout
  topk     hard top-k selection on detached scores (selection frozen), then
           exact WMC restricted to the selected proofs
  minmax   max over proofs of min over facts; autograd routes the one-hot
           subgradient through torch's argmax/argmin choice

so training code is ordinary torch (loss.backward()), and the semantic
distortions of each reasoner reach the perception module exactly as they do
in the deployed systems.
"""

from __future__ import annotations

import torch


class BatchStructure:
    def __init__(self, proofs, facts):
        self.facts = list(facts)
        m = len(self.facts)
        if m > 16:
            raise ValueError(f"{m} facts: 2^m world enumeration too large")
        idx = {f: i for i, f in enumerate(self.facts)}
        self.pidx = [sorted(idx[f] for f in pr) for pr in proofs]
        self.m = m
        w = torch.arange(1 << m)
        self.bits = ((w[:, None] >> torch.arange(m)) & 1).to(torch.float64)  # (W, m)
        self.sat = self._sat_mask(range(len(self.pidx)))                     # (W,)
        self.proof_sat = torch.stack(                                        # (W, P)
            [(self.bits[:, pi] > 0.5).all(dim=1) for pi in self.pidx], dim=1
        ).to(torch.float64)

    def _sat_mask(self, proof_ids) -> torch.Tensor:
        sat = torch.zeros(self.bits.shape[0], dtype=torch.bool)
        for j in proof_ids:
            sat |= (self.bits[:, self.pidx[j]] > 0.5).all(dim=1)
        return sat.to(torch.float64)

    # ------------------------------------------------------------- helpers --

    def truth(self, Z: torch.Tensor) -> torch.Tensor:
        """Query truth on boolean fact states Z (B, m)."""
        out = torch.zeros(Z.shape[0], dtype=torch.bool)
        for pi in self.pidx:
            out |= Z[:, pi].all(dim=1)
        return out

    def world_probs(self, Pb: torch.Tensor) -> torch.Tensor:
        """(B, W) world probabilities from per-sample fact probs (B, m)."""
        b = self.bits[None]                                    # (1, W, m)
        return (b * Pb[:, None, :] + (1 - b) * (1 - Pb[:, None, :])).prod(dim=2)

    def scores(self, Pb: torch.Tensor) -> torch.Tensor:
        """(B, P) proof scores: product of member-fact probabilities."""
        return torch.stack([Pb[:, pi].prod(dim=1) for pi in self.pidx], dim=1)

    # --------------------------------------------------------- provenances --

    def wmc(self, Pb: torch.Tensor) -> torch.Tensor:
        return (self.world_probs(Pb) * self.sat[None]).sum(dim=1)

    def addmult(self, Pb: torch.Tensor) -> torch.Tensor:
        return self.scores(Pb).sum(dim=1).clamp(max=1.0)

    def addmult_st(self, Pb: torch.Tensor) -> torch.Tensor:
        """Straight-through clamp (finding F-3): forward value is the clamped
        sum, backward is the identity on the raw sum — the deployed
        diffaddmultprob differentiation semantics."""
        raw = self.scores(Pb).sum(dim=1)
        return raw + (raw.clamp(max=1.0) - raw).detach()

    def topk(self, Pb: torch.Tensor, k: int) -> torch.Tensor:
        sc = self.scores(Pb)
        if sc.shape[1] <= k:
            return self.wmc(Pb)
        order = torch.argsort(sc.detach(), dim=1, descending=True)[:, :k]  # frozen
        wp = self.world_probs(Pb)                                          # (B, W)
        # per-sample satisfaction mask over the selected proofs:
        # gather (W, P) columns per sample -> (B, W, k) -> any over k
        sel = self.proof_sat.T[order].amax(dim=1)                          # (B, W)
        return (wp * sel).sum(dim=1)

    def minmax(self, Pb: torch.Tensor) -> torch.Tensor:
        per_proof = torch.stack([Pb[:, pi].amin(dim=1) for pi in self.pidx], dim=1)
        return per_proof.amax(dim=1)

    def ltn_prod(self, Pb: torch.Tensor) -> torch.Tensor:
        """LTN, connettivi prodotto (fo.AndProd/fo.OrProbSum): per-proof AND
        e' gia' il prodotto (== scores()); tra i proof si fa OR-prod in
        sequenza: a `or` b = a + b - a*b. minmax() sopra e' gia', byte per
        byte, la variante Godel (And=min, Or=max) -- nessun metodo separato
        serve per quella."""
        sc = self.scores(Pb)                      # (B, P) AND-prod per proof
        acc = sc[:, 0]
        for j in range(1, sc.shape[1]):
            b = sc[:, j]
            acc = acc + b - acc * b
        return acc


def prov_value(name: str, S: BatchStructure, Pb: torch.Tensor,
               k: int | None = None) -> torch.Tensor:
    if name == "exact":
        return S.wmc(Pb)
    if name == "addmult":
        return S.addmult(Pb)
    if name == "addmult_st":
        return S.addmult_st(Pb)
    if name.startswith("top"):
        return S.topk(Pb, k if k is not None else int(name[3:]))
    if name == "minmax" or name == "ltn_godel":
         return S.minmax(Pb)
    if name == "ltn_product":
        return S.ltn_prod(Pb)
    raise ValueError(name)
