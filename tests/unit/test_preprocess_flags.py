"""The argv handed to the shell, preset by preset.

Each preset must reproduce exactly the flags one of the old wrappers passed;
those wrappers are the reference until each flow has been rerun for real.
"""

from pathlib import Path

import pytest

from manifold_genetics.preprocessing import SHELL_SCRIPT
from manifold_genetics.preprocessing.flags import PRESET_FLAGS, PreprocessOptions, shell_argv

TOOLS = dict(plink2="/t/plink2", plink="/t/plink", python="/t/python")


def _argv(options):
    return shell_argv(Path("/d/ref"), Path("/d/bio"), Path("/o/data"), options, **TOOLS)


def _flags(argv):
    """The boolean flags, in order, after the positional/tool part."""
    return [
        a
        for a in argv
        if a.startswith("--skip") or a in ("--cleanup", "--reference-has-chr-prefix")
    ]


def test_always_names_the_sides_the_output_and_the_tools():
    argv = _argv(PreprocessOptions())
    assert argv[:2] == ["bash", str(SHELL_SCRIPT)]
    for flag, value in [
        ("--reference-plink", "/d/ref"),
        ("--biobank-plink", "/d/bio"),
        ("--output-dir", "/o/data"),
        ("--plink2", "/t/plink2"),
        ("--plink", "/t/plink"),
        ("--python", "/t/python"),
    ]:
        assert argv[argv.index(flag) + 1] == value


def test_intersect_only_reproduces_the_generic_wrapper():
    argv = _argv(PreprocessOptions(preset="intersect-only"))
    assert _flags(argv) == [
        "--skip-wrayner",
        "--skip-giab",
        "--skip-hla",
        "--skip-ld-prune",
        "--skip-dedup",
        "--skip-maf",
    ]


def test_harmonise_reproduces_the_aou_wrapper():
    argv = _argv(PreprocessOptions(preset="harmonise", fit_has_chr_prefix=True))
    assert _flags(argv) == ["--skip-biobank-maf", "--cleanup", "--reference-has-chr-prefix"]


def test_no_preset_with_ukbb_flags_reproduces_the_ukbb_wrapper():
    argv = _argv(PreprocessOptions(skip_wrayner=True, skip_project_maf=True))
    assert _flags(argv) == ["--skip-wrayner", "--skip-biobank-maf"]


def test_skip_geno_is_off_by_default_and_never_set_by_a_preset():
    assert "--skip-geno" not in _flags(_argv(PreprocessOptions()))
    for name in PRESET_FLAGS:
        assert "skip_geno" not in PRESET_FLAGS[name], name


def test_skip_geno_emits_the_shell_flag():
    argv = _argv(PreprocessOptions(preset="intersect-only", skip_geno=True))
    assert "--skip-geno" in _flags(argv)


def test_explicit_flags_add_to_a_preset():
    argv = _argv(PreprocessOptions(preset="harmonise", skip_wrayner=True))
    assert "--skip-wrayner" in _flags(argv)


def test_numeric_options_are_passed_only_when_set():
    assert "--maf" not in _argv(PreprocessOptions())
    argv = _argv(
        PreprocessOptions(
            maf=0.05,
            geno=0.1,
            ld_window=150,
            ld_step=1,
            ld_r2=0.05,
            threads=8,
            memory=4000,
            min_common_snps=100,
        )
    )
    for flag, value in [
        ("--maf", "0.05"),
        ("--geno", "0.1"),
        ("--ld-window", "150"),
        ("--ld-step", "1"),
        ("--ld-r2", "0.05"),
        ("--threads", "8"),
        ("--memory", "4000"),
        ("--min-common-snps", "100"),
    ]:
        assert argv[argv.index(flag) + 1] == value


def test_an_unknown_preset_is_rejected_by_name():
    with pytest.raises(ValueError, match="harmonize"):
        _argv(PreprocessOptions(preset="harmonize"))


def test_the_preset_table_only_names_real_options():
    fields = set(PreprocessOptions.__dataclass_fields__)
    for name, flags in PRESET_FLAGS.items():
        assert set(flags) <= fields, name
