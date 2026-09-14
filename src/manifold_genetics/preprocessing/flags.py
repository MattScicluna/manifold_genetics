"""Turn a preset and overrides into the shell's argv. Pure, so it is testable
against the exact flags each old wrapper passed."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from . import SHELL_SCRIPT

# Bundles of boolean options, each reproducing one of the wrappers in examples/.
# `--reference-has-chr-prefix` is a property of the data, not of a preset, so it
# is never part of a bundle.
PRESET_FLAGS: Dict[str, Dict[str, bool]] = {
    # examples/generic/hgdp_1kgp_proj/prepare_data.sh: no external tools at all.
    "intersect-only": {
        "skip_wrayner": True,
        "skip_giab": True,
        "skip_hla": True,
        "skip_ld_prune": True,
        "skip_dedup": True,
        "skip_maf": True,
    },
    # examples/aou/hgdp_1kgp_proj/prepare_data.sh: everything on, MAF on the
    # reference only, intermediates deleted as they are consumed.
    "harmonise": {"skip_project_maf": True, "cleanup": True},
}

# option name -> shell flag. The shell keeps its reference/biobank vocabulary.
_BOOL_FLAGS = {
    "skip_wrayner": "--skip-wrayner",
    "skip_giab": "--skip-giab",
    "skip_hla": "--skip-hla",
    "skip_ld_prune": "--skip-ld-prune",
    "skip_dedup": "--skip-dedup",
    "skip_maf": "--skip-maf",
    "skip_project_maf": "--skip-biobank-maf",
    "cleanup": "--cleanup",
    "fit_has_chr_prefix": "--reference-has-chr-prefix",
}
_VALUE_FLAGS = {
    "maf": "--maf",
    "geno": "--geno",
    "ld_window": "--ld-window",
    "ld_step": "--ld-step",
    "ld_r2": "--ld-r2",
    "threads": "--threads",
    "memory": "--memory",
    "temp_dir": "--temp-dir",
    "tools_dir": "--tools-dir",
    "min_common_snps": "--min-common-snps",
}


@dataclass(frozen=True)
class PreprocessOptions:
    preset: Optional[str] = None
    maf: Optional[float] = None
    geno: Optional[float] = None
    ld_window: Optional[int] = None
    ld_step: Optional[int] = None
    ld_r2: Optional[float] = None
    skip_wrayner: bool = False
    skip_giab: bool = False
    skip_hla: bool = False
    skip_ld_prune: bool = False
    skip_dedup: bool = False
    skip_maf: bool = False
    skip_project_maf: bool = False
    fit_has_chr_prefix: bool = False
    cleanup: bool = False
    threads: Optional[int] = None
    memory: Optional[int] = None
    temp_dir: Optional[Path] = None
    tools_dir: Optional[Path] = None
    min_common_snps: Optional[int] = None

    def effective_flags(self) -> Dict[str, bool]:
        """Boolean options after the preset is applied; an explicit True always wins."""
        if self.preset is not None and self.preset not in PRESET_FLAGS:
            raise ValueError(
                f"Unknown preset {self.preset!r}. Choose from: {', '.join(sorted(PRESET_FLAGS))}"
            )
        flags = dict(PRESET_FLAGS.get(self.preset, {}))
        for name in _BOOL_FLAGS:
            if getattr(self, name):
                flags[name] = True
        return flags


def shell_argv(
    fit_plink: Path,
    project_plink: Path,
    output_dir: Path,
    options: PreprocessOptions,
    *,
    plink2: str,
    plink: str,
    python: str,
) -> List[str]:
    """The command that runs the shipped shell on one pair of PLINK prefixes."""
    argv = [
        "bash",
        str(SHELL_SCRIPT),
        "--reference-plink",
        str(fit_plink),
        "--biobank-plink",
        str(project_plink),
        "--output-dir",
        str(output_dir),
        "--plink2",
        plink2,
        "--plink",
        plink,
        "--python",
        python,
    ]
    for name, flag in _VALUE_FLAGS.items():
        value = getattr(options, name)
        if value is not None:
            argv += [flag, str(value)]
    effective = options.effective_flags()
    for name, flag in _BOOL_FLAGS.items():
        if effective.get(name):
            argv.append(flag)
    return argv


__all__ = ["PRESET_FLAGS", "PreprocessOptions", "shell_argv"]
