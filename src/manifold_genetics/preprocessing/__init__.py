"""SNP filtering and sample selection between `acquire` and `run`.

The filtering itself is `preprocess_cross_projection.sh`, shipped here rather
than reimplemented: it produced the published figures and cannot be validated
against UK Biobank or All of Us from a development machine, so a Python port
would diverge silently. Python assembles its arguments and owns everything
around it -- tool resolution, labels, the output config.
"""

from pathlib import Path

SHELL_SCRIPT = Path(__file__).resolve().parent / "preprocess_cross_projection.sh"

from .flags import PreprocessOptions  # noqa: E402
from .runner import preprocess  # noqa: E402
from .subsample import Group, parse_group, subsample  # noqa: E402

__all__ = [
    "SHELL_SCRIPT",
    "PreprocessOptions",
    "preprocess",
    "Group",
    "parse_group",
    "subsample",
]
