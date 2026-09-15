"""The files the `harmonise` preset needs and would otherwise download mid-run."""

import gzip
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from ..utils.tools import ToolResolver, fetch_url

GIAB_URL = (
    "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/"
    "genome-stratifications/v3.6/GRCh38@all/Union/GRCh38_alldifficultregions.bed.gz"
)
WRAYNER_URL = "https://www.chg.ox.ac.uk/~wrayner/tools/HRC-1000G-check-bim-v4.3.0.zip"
TOPMED_URL = (
    "https://www.dropbox.com/scl/fi/jiy6ty8pmrrr2s5eox1nf/"
    "bravo-dbsnp-all.hrc_format.tab.gz?rlkey=aihwmah4uwvazla7yl438odut&st=8gklc135&dl=1"
)

# name -> (url, path relative to the tools dir, as the shell expects it)
HARMONISATION_REFERENCES: Dict[str, Tuple[str, str]] = {
    "giab": (GIAB_URL, "giab/GRCh38_alldifficultregions.bed"),
    "wrayner": (WRAYNER_URL, "wrayner/HRC-1000G-check-bim.pl"),
    "topmed": (TOPMED_URL, "topmed/bravo-dbsnp-all.hrc_format.tab.gz"),
}


def default_tools_dir() -> Path:
    return ToolResolver().download_dir / "preprocessing"


_VCF_LINE = 'print SH "$plink --bfile $newfile --real-ref-alleles --recode vcf'


def _install_giab(url: str, target: Path, fetch: Callable[[str, Path], None]) -> None:
    packed = target.with_suffix(".bed.gz")
    fetch(url, packed)
    with gzip.open(packed, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    packed.unlink()


def _install_wrayner(url: str, target: Path, fetch: Callable[[str, Path], None]) -> None:
    """Fetch, unzip and patch the checker in a scratch directory, then move it
    into place. The move is the last step and atomic, so an interruption
    anywhere before it leaves no ``.pl`` behind for a later run to mistake for
    an installed (and patched) one."""
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".wrayner-", dir=target.parent))
    try:
        packed = scratch / "HRC-1000G-check-bim.zip"
        fetch(url, packed)
        with zipfile.ZipFile(packed) as zf:
            zf.extractall(scratch)
        extracted = scratch / target.name
        # The shell comments out the VCF creation line; do the same so a
        # prefetched copy behaves like a shell-fetched one.
        extracted.write_text(extracted.read_text().replace(_VCF_LINE, "#" + _VCF_LINE))
        os.replace(extracted, target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def install_harmonisation_references(
    tools_dir: Optional[Path] = None, fetch: Callable[[str, Path], None] = fetch_url
) -> Dict[str, Path]:
    """Fetch each reference into ``tools_dir`` in the layout the shell reads.

    Idempotent: a file already in place is left alone. The GIAB bed is stored
    gunzipped and the WRayner checker unzipped, matching what the shell does
    when it fetches them itself. Every reference is attempted even when an
    earlier one fails, so one dead URL does not stop the others being placed.

    Raises:
        RuntimeError: one or more references could not be fetched. The message
            lists what was placed, what failed and where a hand-obtained copy
            goes; re-running is safe and fetches only what is still missing.
    """
    tools_dir = Path(tools_dir) if tools_dir else default_tools_dir()
    placed: Dict[str, Path] = {}
    failed: Dict[str, Tuple[str, Path, Exception]] = {}
    for name, (url, relative) in HARMONISATION_REFERENCES.items():
        target = tools_dir / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if name == "giab":
                    _install_giab(url, target, fetch)
                elif name == "wrayner":
                    _install_wrayner(url, target, fetch)
                else:
                    fetch(url, target)
            except Exception as exc:  # reported together below, after every attempt
                failed[name] = (url, target, exc)
                continue
        placed[name] = target
    if failed:
        lines = [
            f"{len(failed)} of {len(HARMONISATION_REFERENCES)} references could not be fetched."
        ]
        if placed:
            lines.append("Placed: " + ", ".join(f"{n} ({p})" for n, p in placed.items()))
        for name, (url, target, exc) in failed.items():
            lines.append(f"Failed: {name} from {url}: {exc}")
            lines.append(f"  A copy obtained another way goes at {target}")
        lines.append(
            "Re-running `manifold-genetics setup --preprocessing` is safe: it fetches only"
        )
        lines.append("what is still missing and leaves a hand-placed file alone.")
        raise RuntimeError("\n".join(lines))
    return placed


__all__ = ["HARMONISATION_REFERENCES", "default_tools_dir", "install_harmonisation_references"]
