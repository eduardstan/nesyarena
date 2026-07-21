"""Shared figure palette — one fixed color per semantics/setting.

Every figure keys its colors here, so a setting is recognizable across the
whole paper: the same SUT never changes color between figures, and hue
families encode the semantics family — exact/WMC black, probabilistic
approximations warm/blue/green, LTN fuzzy configurations purple. Base
colors are Okabe–Ito (colorblind-safe).

Unknown SUT names raise KeyError on purpose: a new SUT must be assigned a
color here before it can appear in any figure.
"""

from __future__ import annotations

import re

from matplotlib import colormaps
from matplotlib.colors import to_hex

# canonical setting -> color (Okabe–Ito base + black)
SUT_COLOR = {
    "exact":       "#000000",  # exact/WMC oracle
    "addmult":     "#D55E00",  # vermilion      — add-mult (clamped)
    "addmult_raw": "#8C3A00",  # dark vermilion — add-mult (raw)
    "addmult_st":  "#E69F00",  # orange         — straight-through clamp (F-2)
    "top1":        "#0072B2",  # blue           — top-1 proofs
    "top3":        "#56B4E9",  # sky blue       — top-3 proofs
    "minmax":      "#009E73",  # green          — min-max algebra
    "ltn_product": "#CC79A7",  # reddish purple — LTN product real logic
    "ltn_godel":   "#8B6BB7",  # violet         — LTN Gödel real logic
    "lse":         "#999999",  # grey           — LSE surrogate
}

_ALIASES = {
    "exact": "exact", "exact-wmc": "exact", "wmc": "exact",
    "addmult": "addmult", "add-mult(clamped)": "addmult",
    "add-mult(raw)": "addmult_raw",
    "addmult_st": "addmult_st", "add-mult(straight-through)": "addmult_st",
    "top1": "top1", "top-1-proofs": "top1",
    "top3": "top3", "top-3-proofs": "top3",
    "minmax": "minmax", "min-max-prob": "minmax",
    "ltn_product": "ltn_product", "ltn:product": "ltn_product",
    "ltn_godel": "ltn_godel", "ltn:godel": "ltn_godel",
}


def canon(name: str) -> str:
    """Canonical setting key for any of the three naming conventions
    (learning keys, registry names, arena names)."""
    key = name.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    if key.startswith("lse("):
        return "lse"
    m = re.fullmatch(r"top-?(\d+)(-proofs)?", key)
    if m:
        return f"top{m.group(1)}"
    raise KeyError(f"no palette entry for SUT {name!r} — add it to "
                   "experiments/palette.py")


def sut_color(name: str) -> str:
    return SUT_COLOR[canon(name)]


# truncation settings (E2/E7/E8): convergent black, depths light -> dark blue
TRUNCATION_COLOR = {"convergent": "#000000", 2: "#9ECAE1", 4: "#4292C6",
                    6: "#2171B5", 8: "#084594"}


def truncation_color(n) -> str:
    """n is an unrolling/budget depth (int or numeric string) or
    'convergent'."""
    key = n if n == "convergent" else int(n)
    return TRUNCATION_COLOR[key]


def sequential(values, cmap: str = "viridis", lo: float = 0.2,
               hi: float = 0.85) -> dict:
    """Deterministic ordered mapping value -> color for a numeric family
    (e.g. E3's P families)."""
    vs = sorted(set(values))
    c = colormaps[cmap]
    if len(vs) == 1:
        return {vs[0]: to_hex(c(0.5))}
    return {v: to_hex(c(lo + (hi - lo) * i / (len(vs) - 1)))
            for i, v in enumerate(vs)}
