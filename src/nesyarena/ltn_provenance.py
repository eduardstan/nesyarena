"""Provenance basata su LTNtorch (product fuzzy logic / Real Logic).

Segue esattamente l'interfaccia di suts.Provenance (vedi docs/ADAPTERS.md):
- value(): il valore che LTN calcola per la query, aggregando i proof
  (ognuno = congiunzione AND-prod dei fatti che lo compongono) con
  OR-probsum sui proof alternativi.
- grad(): il gradiente autentico di LTN, ottenuto con autograd reale
  (non finite differences), esattamente come richiesto da ADAPTERS.md
  ("if your backend is differentiable, grad should return its own
  gradients").
- claimed_semantics = "distribution semantics": con i connettivi prodotto,
  LTN coincide ESATTAMENTE con la distribution semantics quando i proof
  sono disgiunti (fatti non condivisi); diverge quando i proof condividono
  fatti (asse di overlap G1), perche' And/Or-prod assumono indipendenza.
  Questo e' un finding testabile con experiments.e1_overlap, non un bug
  dell'adapter: va riportato, non corretto (regola del progetto).

Nota di scope: questa Provenance lavora su proof-DNF gia' enumerati
(program.proof_supports), come AddMult/TopK/MinMax. Per gli esperimenti
di profondita' ricorsiva (E2, E7) serve invece un Adapter vero, che valuti
LTN sulla struttura a regole del GroundProgram (stile engine.py/T_P),
perche' l'enumerazione dei proof esplode con la ricorsione. Vedi il
docstring in fondo per lo scheletro di quella variante.
"""

from __future__ import annotations

import torch
import ltn.fuzzy_ops as fo

from .suts import Provenance

_AND_PROD = fo.AndProd()
_OR_PROD = fo.OrProbSum()
_AND_GODEL = fo.AndMin()
_OR_GODEL = fo.OrMax()


class _LTNProvenanceBase(Provenance):
    """Scheletro comune: aggrega i proof con AND-dentro-il-proof e
    OR-tra-i-proof, usando SEMPRE le classi di ltn.fuzzy_ops (mai
    reimplementate a mano) e SEMPRE autograd reale per il gradiente."""

    and_op = None  # istanza di una classe *ConnectiveOperator di fuzzy_ops
    or_op = None

    def _proof_value(self, pr, prob_tensors: dict) -> torch.Tensor:
        v = torch.tensor(1.0)
        for f in pr:
            v = self.and_op(v, prob_tensors[f])
        return v

    def _formula_value(self, proofs, prob_tensors: dict) -> torch.Tensor:
        vals = [self._proof_value(pr, prob_tensors) for pr in proofs]
        acc = vals[0]
        for v in vals[1:]:
            acc = self.or_op(acc, v)
        return acc

    def value(self, proofs, probs) -> float:
        if not proofs:
            return 0.0
        prob_tensors = {f: torch.tensor(float(p)) for f, p in probs.items()}
        return self._formula_value(proofs, prob_tensors).item()

    def grad(self, proofs, probs) -> dict:
        if not proofs:
            return {}
        facts = sorted(set().union(*proofs), key=repr)
        prob_tensors = {
            f: torch.tensor(float(probs[f]), requires_grad=True) for f in facts
        }
        out = self._formula_value(proofs, prob_tensors)
        out.backward()
        # AndMin/OrMax non sono ovunque differenziabili (ties): PyTorch
        # restituisce comunque un sottogradiente valido (0 o 1 sul ramo
        # scelto), esattamente come fa MinMax in suts.py a mano.
        return {f: prob_tensors[f].grad.item() for f in facts}


class LTNProdProvenance(_LTNProvenanceBase):
    """LTN con connettivi prodotto (fo.AndProd, fo.OrProbSum):
    And = a*b, Or = a+b-a*b."""

    name = "ltn:product"
    claimed = "distribution semantics"
    and_op = _AND_PROD
    or_op = _OR_PROD


class LTNGodelProvenance(_LTNProvenanceBase):
    """LTN con connettivi Godel (fo.AndMin, fo.OrMax): And = min, Or = max.
    Corrisponde ESATTAMENTE al semiring BOOLEAN dell'arena (algebra.py) --
    utile come conformance check indipendente: se questa Provenance non
    coincide con l'engine sotto BOOLEAN, e' un bug nell'adapter, non un
    finding, perche' qui la semantica dichiarata (Godel) ha gia' un
    oracolo esatto nell'arena stessa (reachability booleana)."""

    name = "ltn:godel"
    claimed = "boolean reachability"
    and_op = _AND_GODEL
    or_op = _OR_GODEL
