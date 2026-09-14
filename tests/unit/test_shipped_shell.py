"""The filtering shell ships inside the wheel and is syntactically valid.

It lived in examples/, which the wheel does not carry, so `preprocess` on an
installed copy would have had nothing to run.
"""

import shutil
import subprocess

import pytest

from manifold_genetics import preprocessing


def test_the_shell_ships_with_the_package():
    assert preprocessing.SHELL_SCRIPT.is_file()
    assert (preprocessing.SHELL_SCRIPT.parent / "common.sh").is_file()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not on PATH")
def test_the_shell_parses():
    for name in ("preprocess_cross_projection.sh", "common.sh"):
        subprocess.run(["bash", "-n", str(preprocessing.SHELL_SCRIPT.parent / name)], check=True)


def test_the_example_copies_are_links_to_the_shipped_one(tmp_path):
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    link = repo / "examples/_shared/preprocessing/preprocess_cross_projection.sh"
    if not link.exists():
        pytest.skip("not running from a checkout")
    assert link.is_symlink()
    assert link.resolve() == preprocessing.SHELL_SCRIPT.resolve()


NEW_FLAGS = ["--plink2", "--plink", "--python", "--min-common-snps", "--skip-geno"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not on PATH")
def test_help_lists_the_flags_python_passes():
    out = subprocess.run(
        ["bash", str(preprocessing.SHELL_SCRIPT), "--help"], capture_output=True, text=True
    )
    assert out.returncode == 0
    for flag in NEW_FLAGS:
        assert flag in out.stdout, f"{flag} missing from --help"


def test_the_shell_no_longer_reaches_for_src():
    text = preprocessing.SHELL_SCRIPT.read_text()
    assert "sys.path.insert" not in text
    assert "python3 " not in text and "python3\n" not in text, "python3 must go through ${PYTHON}"
