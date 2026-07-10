"""
Palette, ordering and label resolution for seaborn.

The wake manuscript and the sensitivity manuscript share scenario keys but
draw them from different palettes: the four wake formulations use
``WAKE_MODEL_COLORS``, the bias x wake sensitivity scenarios use
``THESIS_COLORS``. :func:`palette_for` picks the right one from the keys
themselves, so a figure never has to know which paper it belongs to.
"""

from __future__ import annotations

from thesis_colors import (
    BIAS_KEYS,
    SCENARIO_KEYS,
    THESIS_COLORS,
    WAKE_MODEL_COLORS,
    WAKE_ORDER,
    canon,
    label,
)

__all__ = [
    "hue_kwargs",
    "labels_for",
    "order_for",
    "palette_for",
    "relabel",
]

# Keys that only the wake palette defines. `base` and `standard` live in both
# palettes with the same colour, so they cannot discriminate on their own.
_WAKE_ONLY = {"glaum", "new_more"}


def _canonical(keys) -> list[str]:
    """Resolve aliases, preserving order and dropping duplicates."""
    seen: dict[str, None] = {}
    for key in keys:
        seen.setdefault(canon(key), None)
    return list(seen)


def palette_for(keys) -> dict[str, str]:
    """Return a ``{key: colour}`` mapping for *keys*.

    Uses the wake palette when the keys name wake formulations, otherwise the
    general thesis scenario palette. Keys are alias-resolved, so ``"uniform"``
    and ``"standard"`` get the same colour.
    """
    canonical = _canonical(keys)

    if _WAKE_ONLY.intersection(canonical):
        source = WAKE_MODEL_COLORS
    elif set(canonical).issubset(THESIS_COLORS):
        source = THESIS_COLORS
    elif set(canonical).issubset(WAKE_MODEL_COLORS):
        source = WAKE_MODEL_COLORS
    else:
        unknown = sorted(set(canonical) - set(THESIS_COLORS) - set(WAKE_MODEL_COLORS))
        raise KeyError(
            f"No colour defined for {unknown!r}. Add it to thesis_colors, or "
            "pass keys drawn from a single palette."
        )
    return {key: source[key] for key in canonical}


def order_for(keys) -> list[str]:
    """Return *keys* in the canonical plotting order.

    Wake formulations follow ``WAKE_ORDER`` (Baseline, Uniform,
    Tiered-capacity, Tiered-density); sensitivity scenarios follow
    ``SCENARIO_KEYS``; bias flags follow ``BIAS_KEYS``. Anything unrecognised
    is appended, sorted, so the function never silently drops a series.
    """
    canonical = set(_canonical(keys))

    for reference in (WAKE_ORDER, SCENARIO_KEYS, BIAS_KEYS):
        if canonical.issubset(reference):
            return [key for key in reference if key in canonical]

    known = [
        key
        for reference in (WAKE_ORDER, SCENARIO_KEYS, BIAS_KEYS)
        for key in reference
        if key in canonical
    ]
    seen = dict.fromkeys(known)
    rest = sorted(canonical - set(seen))
    return list(seen) + rest


def labels_for(keys) -> dict[str, str]:
    """Return a ``{key: display label}`` mapping."""
    return {key: label(key) for key in _canonical(keys)}


def hue_kwargs(keys, hue: str = "scenario") -> dict:
    """Return the seaborn ``hue``/``hue_order``/``palette`` triple for *keys*.

    Passing this into every seaborn call is what keeps colour and ordering
    consistent across both manuscripts::

        sns.boxplot(data=df, y="wake_loss", **hue_kwargs(df.scenario.unique()))
    """
    order = order_for(keys)
    return {"hue": hue, "hue_order": order, "palette": palette_for(order)}


def relabel(ax, keys=None) -> None:
    """Replace legend entries with their display labels."""
    legend = ax.get_legend()
    if legend is None:
        return
    mapping = labels_for(keys) if keys is not None else {}
    for text in legend.get_texts():
        raw = text.get_text()
        text.set_text(mapping.get(canon(raw), label(raw)))
