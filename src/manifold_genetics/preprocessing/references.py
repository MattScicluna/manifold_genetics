"""The files the `harmonise` preset needs and would otherwise download mid-run."""

import gzip
import shutil
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


def install_harmonisation_references(
    tools_dir: Optional[Path] = None, fetch: Callable[[str, Path], None] = fetch_url
) -> Dict[str, Path]:
    """Fetch each reference into ``tools_dir`` in the layout the shell reads.

    Idempotent: a file already in place is left alone. The GIAB bed is stored
    gunzipped and the WRayner checker unzipped, matching what the shell does
    when it fetches them itself.
    """
    tools_dir = Path(tools_dir) if tools_dir else default_tools_dir()
    placed = {}
    for name, (url, relative) in HARMONISATION_REFERENCES.items():
        target = tools_dir / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == "giab":
                packed = target.with_suffix(".bed.gz")
                fetch(url, packed)
                with gzip.open(packed, "rb") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                packed.unlink()
            elif name == "wrayner":
                packed = target.parent / "HRC-1000G-check-bim.zip"
                fetch(url, packed)
                with zipfile.ZipFile(packed) as zf:
                    zf.extractall(target.parent)
                # The shell comments out the VCF creation line; do the same so a
                # prefetched copy behaves like a shell-fetched one.
                text = target.read_text()
                target.write_text(
                    text.replace(
                        'print SH "$plink --bfile $newfile --real-ref-alleles --recode vcf',
                        '#print SH "$plink --bfile $newfile --real-ref-alleles --recode vcf',
                    )
                )
            else:
                fetch(url, target)
        placed[name] = target
    return placed


__all__ = ["HARMONISATION_REFERENCES", "default_tools_dir", "install_harmonisation_references"]
