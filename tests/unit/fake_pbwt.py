"""A stand-in for the pbwt binary, for tests that must not need it.

``install_fake_pbwt(tmp_path)`` writes an executable that accepts exactly the
command ``pbwt -readVcfGT <vcf> -writeSamples <file> -paint <root>`` and writes
the files the real one would, in its layout. Copying is 10 between samples whose
names share a first letter and 1 otherwise, so a group structure survives into
the PCA. Every invocation is appended to ``calls.log`` beside the script, and
``FAKE_PBWT_FAIL=1`` makes it exit 1.
"""

import sys
from pathlib import Path

SCRIPT = """\
import os, sys
from pathlib import Path
args = sys.argv[1:]
here = Path(__file__).parent
with open(here / "calls.log", "a") as log:
    log.write(" ".join(args) + "\\n")
if os.environ.get("FAKE_PBWT_FAIL"):
    print("fake pbwt: forced failure", file=sys.stderr)
    sys.exit(1)
opts = dict(zip(args[::2], args[1::2]))
vcf, samples_out, root = opts["-readVcfGT"], opts["-writeSamples"], opts["-paint"]
names = []
for line in open(vcf):
    if line.startswith("#CHROM"):
        names = line.rstrip("\\n").split("\\t")[9:]
        break
Path(samples_out).write_text("".join(n + "\\n" for n in names))
header = "RECIPIENT" + "".join(f" IND{i + 1}" for i in range(len(names))) + "\\n"
for tag, scale in (("chunkcounts", 1.0), ("chunklengths", 10.0)):
    rows = []
    for i, a in enumerate(names):
        vals = []
        for j, b in enumerate(names):
            v = 0.0 if i == j else (10.0 if a[0] == b[0] else 1.0) * scale
            vals.append(f"{v:.4f}")
        rows.append(f"IND{i + 1} " + " ".join(vals) + "\\n")
    Path(f"{root}.{tag}.out").write_text(header + "".join(rows))
for tag in ("regionchunkcounts", "regionsquaredchunkcounts"):
    Path(f"{root}.{tag}.out").write_text("RECIPIENT nregions\\n")
print("written", len(names), "samples", file=sys.stderr)
"""


def install_fake_pbwt(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pbwt"
    path.write_text(f"#!{sys.executable}\n" + SCRIPT)
    path.chmod(0o755)
    return path


def calls(directory: Path) -> list:
    log = directory / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def write_vcf(path: Path, samples, chrom: str = "1") -> Path:
    """A minimal phased VCF; the fake pbwt reads only the sample names."""
    header = [
        "##fileformat=VCFv4.2",
        "#"
        + "\t".join(["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"])
        + "\t"
        + "\t".join(samples),
    ]
    row = "\t".join(
        [chrom, "100", "rs1", "A", "G", ".", "PASS", ".", "GT"] + ["0|1"] * len(samples)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + [row]) + "\n")
    return path


def write_fam(prefix: Path, iids, fids=None) -> Path:
    fids = fids or iids
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.fam").write_text("".join(f"{f} {i} 0 0 0 -9\n" for f, i in zip(fids, iids)))
    return prefix
