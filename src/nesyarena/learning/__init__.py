"""Torch-native learning layer: batched provenance ops whose autograd
gradients are system-faithful by construction (clamp flat region, frozen
top-k selection, one-hot min-max subgradient). Gated by parity tests against
the reference SUTs in suts.py."""

from .structures import BatchStructure, prov_value

__all__ = ["BatchStructure", "prov_value"]
