"""Run ``pbwt -paint`` on one phased VCF per chromosome.

The command is ``pbwt -readVcfGT <vcf> -writeSamples <root>.samples -paint
<root>``. Three properties of pbwt shape this module, all read from its source:

* ``-readVcfGT`` stops at the first change of chromosome, so a VCF holding
  several chromosomes is painted for the first one only, silently. That is why
  ``phased_vcf`` must contain ``{chr}`` and each chromosome is a separate run.
* It does not check that genotypes are phased. An unphased VCF is read as if it
  were phased and produces a painting that looks normal and is wrong.
* Missing genotypes are read as the reference allele.

pbwt writes its outputs at the very end, so a run is written under a
``.partial`` root and renamed only when it succeeds. A killed job therefore
never leaves files that a rerun would mistake for a finished chromosome.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

__all__ = [
    "CHR_PLACEHOLDER",
    "DEFAULT_CHROMOSOMES",
    "PAINT_SUFFIXES",
    "expand_phased_vcf",
    "paint_chromosome",
    "paint_outputs",
    "parse_chromosomes",
]

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

CHR_PLACEHOLDER = "{chr}"
DEFAULT_CHROMOSOMES = tuple(str(c) for c in range(1, 23))

#: Every file one ``-writeSamples ... -paint`` run leaves behind, by key.
PAINT_SUFFIXES = {
    "samples": "samples",
    "chunkcounts": "chunkcounts.out",
    "chunklengths": "chunklengths.out",
    "regionchunkcounts": "regionchunkcounts.out",
    "regionsquaredchunkcounts": "regionsquaredchunkcounts.out",
}


def parse_chromosomes(value) -> List[str]:
    """Normalise a chromosome selection to a list of names.

    Accepts ``None`` (autosomes 1-22), an int, a list of ints or strings, or a
    string of comma-separated names and ranges such as ``"1-22"`` or
    ``"1-3,7,X"``.
    """
    if value is None:
        return list(DEFAULT_CHROMOSOMES)
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        names = [str(v).strip() for v in value]
    else:
        names = []
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, _, end = part.partition("-")
                if not (start.strip().isdigit() and end.strip().isdigit()):
                    raise ValueError(f"Chromosome range {part!r} must be numeric, e.g. 1-22")
                lo, hi = int(start), int(end)
                if lo > hi:
                    raise ValueError(f"Chromosome range {part!r} is empty")
                names.extend(str(c) for c in range(lo, hi + 1))
            else:
                names.append(part)
    if not names or any(not n for n in names):
        raise ValueError(f"No chromosomes selected by {value!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"Chromosome selection {value!r} names a chromosome twice")
    return names


def expand_phased_vcf(template: PathLike, chromosomes: Sequence[str]) -> Dict[str, Path]:
    """``{chromosome: path}`` from a ``phased_vcf`` template containing ``{chr}``.

    Raises:
        ValueError: the template has no ``{chr}``. pbwt reads only the first
            chromosome of a file, so a single multi-chromosome VCF would be
            painted for one chromosome without any error.
    """
    template = str(template)
    if CHR_PLACEHOLDER not in template:
        raise ValueError(
            f"phased_vcf must contain {CHR_PLACEHOLDER} (for example data/chr{{chr}}.vcf.gz), "
            f"got {template!r}. pbwt paints only the first chromosome of a VCF, so each "
            "chromosome has to be its own file."
        )
    return {c: Path(template.replace(CHR_PLACEHOLDER, c)) for c in chromosomes}


def paint_outputs(root: PathLike) -> Dict[str, Path]:
    """The files a finished ``paint_chromosome(root=...)`` leaves, by key."""
    root = str(root)
    return {key: Path(f"{root}.{suffix}") for key, suffix in PAINT_SUFFIXES.items()}


def _complete(root: PathLike) -> bool:
    outputs = paint_outputs(root)
    return all(outputs[k].exists() for k in ("samples", "chunkcounts", "chunklengths"))


def paint_chromosome(
    pbwt: str,
    vcf: PathLike,
    root: PathLike,
    *,
    force: bool = False,
    log_file: Optional[PathLike] = None,
) -> Dict[str, Path]:
    """Paint one chromosome, all against all, and return its output files.

    Args:
        pbwt: The pbwt executable.
        vcf: A phased VCF or BCF holding exactly one chromosome.
        root: Output root; files land at ``<root>.chunkcounts.out`` and so on.
        force: Repaint even when finished outputs newer than the VCF exist.
        log_file: Where pbwt's stderr goes. Defaults to ``<root>.log``.

    Raises:
        FileNotFoundError: the VCF does not exist.
        RuntimeError: pbwt exited non-zero; the message carries the end of its log.
    """
    vcf, root = Path(vcf), Path(root)
    if not vcf.exists():
        raise FileNotFoundError(f"Phased VCF not found: {vcf}")
    # Reused only when newer than the VCF, so replacing a VCF repaints it.
    if (
        not force
        and _complete(root)
        and paint_outputs(root)["chunkcounts"].stat().st_mtime >= vcf.stat().st_mtime
    ):
        logger.info(f"Reusing painting for {vcf.name}: {root}.chunkcounts.out")
        return paint_outputs(root)

    root.parent.mkdir(parents=True, exist_ok=True)
    partial = root.with_name(root.name + ".partial")
    partial_outputs = paint_outputs(partial)
    log_path = Path(log_file) if log_file else Path(f"{root}.log")

    cmd = [
        pbwt,
        "-readVcfGT",
        str(vcf),
        "-writeSamples",
        str(partial_outputs["samples"]),
        "-paint",
        str(partial),
    ]
    logger.info(f"Painting {vcf.name}: {' '.join(cmd)}")
    with open(log_path, "w") as log:
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        tail = "".join(log_path.read_text().splitlines(keepends=True)[-20:])
        raise RuntimeError(
            f"pbwt failed on {vcf} (exit {completed.returncode}). Log: {log_path}\n{tail}"
        )

    final = paint_outputs(root)
    for key, path in partial_outputs.items():
        if path.exists():
            path.replace(final[key])
    if not _complete(root):
        raise RuntimeError(
            f"pbwt exited cleanly on {vcf} but did not write its painting files. Log: {log_path}"
        )
    return final
