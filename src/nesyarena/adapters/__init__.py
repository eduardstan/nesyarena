"""External-system adapters. Each adapter declares its claimed semantics and
exposes infer()/grad() over the same GroundProgram + base interpretation the
oracles consume — the shared-input invariant behind D1."""

from .base import Adapter, ReferenceAdapter

__all__ = ["Adapter", "ReferenceAdapter"]
