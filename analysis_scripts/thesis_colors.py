
"""
Shared color and label definitions for thesis figures.

Goal:
- Ensure identical colors and labels for Baseline / Bias / Wake / Bias+Wake
  across ALL figures (spatial, wake, bias, sensitivity).
"""

from __future__ import annotations

# ---------- Canonical display names ----------
THESIS_LABELS = {
    # global scenarios (Chapter 8 / high-level)
    "base": "Baseline",
    "standard": "Uniform",
    "bias": "PyVWF bias",
    "wake": "Tiered-density wake",
    "bias+wake": "Bias + wake",

    # wake model keys (Chapter 6)
    "glaum": "Tiered-capacity",
    "new_more": "Tiered-density",

    # bias flags (Chapter 7)
    "biasTrue": "PyVWF",
    "biasFalse": "Baseline",
    "biasUniform": "Uniform",

    # technologies
    "onwind": "Onshore",
    "offwind": "Offshore",
    "offwind-ac": "Offshore (AC)",
    "offwind-dc": "Offshore (DC)",
    "offwind-float": "Offshore (floating)",
    "offwind-combined": "Offshore (combined)",
}


WAKE_KEYS = ["base", "standard", "glaum", "new_more"]
WAKE_ORDER = ["base", "standard", "glaum", "new_more"]  # Baseline, Uniform, Tiered-capacity, Tiered-density
BIAS_KEYS = ["biasFalse", "biasUniform", "biasTrue"]
SCENARIO_KEYS = ["base", "standard", "bias", "wake", "bias+wake"]

# ---------- Aliases (backwards compatibility) ----------
# Map legacy keys -> canonical keys (or keep them identical).
ALIASES = {
    # examples if you had old names in saved results:
    "tiered-density": "new_more",
    "tiered-capacity": "glaum",
    "uniform": "standard",
}

def canon(key: str) -> str:
    """Return canonical key (resolves aliases)."""
    return ALIASES.get(key, key)

def labels_for(keys: list[str], *, latex: bool = False) -> dict[str, str]:
    return {k: label(k, latex=latex) for k in keys}

def label(key: str, *, default: str | None = None) -> str:
    """Get display label for a key, with alias resolution."""
    k = canon(key)
    reg = THESIS_LABELS
    if default is None:
        default = k
    return reg.get(k, default)


# Colorblind-safe, print-friendly palette (Elsevier-friendly)
# Chosen to remain distinct in grayscale where possible.

THESIS_COLORS = {
    # --- Neutral / baseline ---
    "base": "#4D4D4D",        # charcoal (anchor everywhere)

    # --- Reference / default ---
    "standard": "#D55E00",    # muted orange (Okabe–Ito)

    # --- Single corrections ---
    "bias": "#5DAE8B",        # muted green (PyVWF)
    "wake": "#2F4B7C",        # muted blue (wake physics)

    # --- Combined ---
    "bias+wake": "#8172B2",   # muted purple (blue + green mix)
}

WAKE_MODEL_COLORS = {
    "base":     THESIS_COLORS["base"],
    "standard": "#6B8EC1",  # light blue (baseline wake)
    "glaum":    "#4C72B0",  # medium blue
    "new_more": "#2F4B7C",  # dark blue (most advanced)
}


def get_color_cycle_thesis(order: list[str]) -> list[str]:
    """
    Return a list of colors corresponding to a scenario order.

    Args:
        order: list of scenario keys (e.g. SCEN_ORDER)

    Returns:
        list of hex color strings in matching order
    """
    return [THESIS_COLORS[o] for o in order]

def get_color_cycle_wake(order: list[str]) -> list[str]:
    """
    Return a list of colors corresponding to a scenario order.

    Args:
        order: list of scenario keys (e.g. SCEN_ORDER)

    Returns:
        list of hex color strings in matching order
    """
    return [WAKE_MODEL_COLORS[o] for o in order]


# ---------- Backward compatibility aliases ----------
# For scripts that import SCENARIO_COLORS and get_color_cycle
SCENARIO_COLORS = THESIS_COLORS


def get_color_cycle(order: list[str]) -> list[str]:
    """
    Backward compatibility alias for get_color_cycle_thesis.
    
    Args:
        order: list of scenario keys
        
    Returns:
        list of hex color strings in matching order
    """
    return get_color_cycle_thesis(order)