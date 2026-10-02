"""Colormaps `acquire` generates: a distinct colour for every label value.

Generated colours are a starting point for a figure, not a publication choice;
`known` carries the colours that must not move -- the published ones.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Distinguishable at a glance and colourblind-safe enough to start from: Okabe-Ito,
# extended by cycling with varied lightness. A generated colormap is a starting
# point, not a publication choice.
_PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#8C564B",
    "#7F7F7F",
)

# After the ten above: matplotlib's tab20 colours not already among them, then a
# few darker tones. Past these, _distinct_colours generates more, so a column
# with many values never gives two of them one colour (the palette used to wrap
# at ten, which on All of Us v9 put two race groups in the same colour).
_PALETTE_EXTENDED = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#17BECF",
    "#BCBD22",
    "#E377C2",
    "#AEC7E8",
    "#FFBB78",
    "#98DF8A",
    "#FF9896",
    "#C5B0D5",
    "#C49C94",
    "#F7B6D2",
    "#DBDB8D",
    "#9EDAE5",
    "#393B79",
    "#637939",
    "#843C39",
)


def _distinct_colours(n: int, exclude=()) -> list:
    """``n`` different colours, none of them in ``exclude``.

    The fixed palettes first, then hues spaced by the golden angle at a few
    lightnesses, so any ``n`` is met without a repeat.
    """
    import colorsys

    taken = {c.upper() for c in exclude}
    out = []
    for colour in _PALETTE + _PALETTE_EXTENDED:
        if len(out) == n:
            return out
        if colour.upper() not in taken:
            taken.add(colour.upper())
            out.append(colour)
    i = 0
    while len(out) < n:
        hue = (i * 0.618033988749895) % 1.0
        lightness = (0.40, 0.55, 0.70)[(i // 12) % 3]
        r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.65)
        colour = "#{:02X}{:02X}{:02X}".format(round(r * 255), round(g * 255), round(b * 255))
        if colour not in taken:
            taken.add(colour)
            out.append(colour)
        i += 1
    return out


def _write_generated_colormap(
    labels: pd.DataFrame, path: Path, known: Optional[dict] = None
) -> None:
    """A colour for every value of every label column, so no point goes grey.

    Generated, therefore provisional: it is the file you recolour for a figure,
    and it exists so that nobody hand-writes 22 hex codes to find out whether
    their pipeline runs.

    ``known`` maps a column to fixed colours for values whose colour must not
    move -- the published ones. Those come first, in ``known``'s order (the
    legend's order); every other value gets a distinct colour not used by them,
    in alphabetical order. Without ``known``, colours follow alphabetical
    position, so a new value can recolour the ones after it.
    """
    known = known or {}
    colormap = {}
    for column in labels.columns:
        if column == "sample_id":
            continue
        values = set(labels[column].dropna().astype(str).unique())
        fixed = {v: c for v, c in known.get(column, {}).items() if v in values}
        rest = sorted(values - set(fixed))
        colours = _distinct_colours(len(rest), exclude=known.get(column, {}).values())
        colormap[column] = {**fixed, **dict(zip(rest, colours))}

    if not colormap:
        raise ValueError("the label file has no columns besides sample_id to colour by")

    import json as _json

    path.write_text(_json.dumps(colormap, indent=2) + "\n")
