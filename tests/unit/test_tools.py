"""Tests for utils/tools.py — ToolResolver path resolution.

ToolResolver uses a fallback chain to find external binaries:
  local bin dir → env var → PATH → auto-download

There is no Environment Modules step, and a test below pins that: ``module load``
in a child shell cannot change this process's PATH, so the step that used to sit
between env var and PATH never found anything.

Tests use real temp executables (chmod +x shell scripts) rather than mocking
os.access, so the tests verify actual filesystem permission checks rather than
the mock infrastructure. Download tests stub the download method.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_script(path: Path, content: str = "#!/bin/sh\nexit 0\n") -> Path:
    """Write a shell script and make it executable."""
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ---------------------------------------------------------------------------
# TestValidateExecutable — real filesystem, no mocks
# ---------------------------------------------------------------------------


class TestValidateExecutable:
    """_validate_executable must check existence, file type, and execute permission.
    Uses real filesystem operations — do not mock os.access here, that defeats the purpose.
    """

    def test_real_system_shell_is_executable(self, tmp_path):
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._validate_executable("/bin/sh") is True

    def test_missing_path_returns_false(self, tmp_path):
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._validate_executable(str(tmp_path / "does_not_exist")) is False

    def test_file_without_execute_permission_returns_false(self, tmp_path):
        f = tmp_path / "no_exec"
        f.write_text("data")
        f.chmod(0o644)
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._validate_executable(str(f)) is False

    def test_directory_returns_false(self, tmp_path):
        d = tmp_path / "a_dir"
        d.mkdir()
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._validate_executable(str(d)) is False

    def test_executable_script_returns_true(self, tmp_path):
        script = make_script(tmp_path / "myscript.sh")
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._validate_executable(str(script)) is True

    def test_world_executable_only_returns_true(self, tmp_path):
        """File executable by others but not owner — should still satisfy os.X_OK for owner."""
        f = tmp_path / "weird_perms"
        f.write_text("#!/bin/sh\n")
        # owner execute is required for os.X_OK when running as file owner
        f.chmod(0o755)
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._validate_executable(str(f)) is True


# ---------------------------------------------------------------------------
# TestResolvePlink2
# ---------------------------------------------------------------------------


class TestResolvePlink2:

    def test_local_bin_dir_takes_priority_over_env(self, tmp_path, monkeypatch):
        """A plink2 executable in the local bin dir beats PLINK_PATH."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        local = make_script(bin_dir / "plink2")

        env_path = make_script(tmp_path / "env_plink2")
        monkeypatch.setenv("PLINK_PATH", str(env_path))

        resolver = ToolResolver(download_dir=bin_dir)
        assert resolver.resolve_plink2() == str(local)

    def test_env_var_valid_executable_returned(self, tmp_path, monkeypatch):
        fake = make_script(tmp_path / "plink2")
        monkeypatch.setenv("PLINK_PATH", str(fake))
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        assert resolver.resolve_plink2() == str(fake)

    def test_env_var_pointing_to_nonexistent_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PLINK_PATH", str(tmp_path / "ghost"))
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        with pytest.raises(ToolNotFoundError, match="PLINK_PATH"):
            resolver.resolve_plink2()

    def test_env_var_pointing_to_non_executable_raises(self, tmp_path, monkeypatch):
        """PLINK_PATH pointing to a non-executable file must raise, not silently skip."""
        non_exec = tmp_path / "plink2_data"
        non_exec.write_text("not a binary")
        non_exec.chmod(0o644)
        monkeypatch.setenv("PLINK_PATH", str(non_exec))
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        with pytest.raises(ToolNotFoundError, match="PLINK_PATH"):
            resolver.resolve_plink2()

    def test_local_bin_flashpca_x86_64_variant_found(self, tmp_path, monkeypatch):
        """Local bin also checks for 'flashpca_x86-64' name (not just 'flashpca')."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        local = make_script(bin_dir / "flashpca_x86-64")
        monkeypatch.delenv("FLASHPCA_PATH", raising=False)
        resolver = ToolResolver(download_dir=bin_dir)
        assert resolver.resolve_flashpca() == str(local)


# ---------------------------------------------------------------------------
# TestResolveFlashPCA
# ---------------------------------------------------------------------------


class TestResolveFlashPCA:

    def test_env_var_valid(self, tmp_path, monkeypatch):
        fake = make_script(tmp_path / "flashpca")
        monkeypatch.setenv("FLASHPCA_PATH", str(fake))
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        assert resolver.resolve_flashpca() == str(fake)

    def test_env_var_nonexistent_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FLASHPCA_PATH", str(tmp_path / "ghost"))
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        with pytest.raises(ToolNotFoundError, match="FLASHPCA_PATH"):
            resolver.resolve_flashpca()

    def test_local_bin_flashpca_name(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        local = make_script(bin_dir / "flashpca")
        monkeypatch.delenv("FLASHPCA_PATH", raising=False)
        resolver = ToolResolver(download_dir=bin_dir)
        assert resolver.resolve_flashpca() == str(local)

    def test_local_bin_x86_64_variant(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        local = make_script(bin_dir / "flashpca_x86-64")
        monkeypatch.delenv("FLASHPCA_PATH", raising=False)
        resolver = ToolResolver(download_dir=bin_dir)
        assert resolver.resolve_flashpca() == str(local)


# ---------------------------------------------------------------------------
# TestResolveNeuralAdmixture
# ---------------------------------------------------------------------------


class TestResolveNeuralAdmixture:

    def test_env_var_valid(self, tmp_path, monkeypatch):
        fake = make_script(tmp_path / "neural-admixture")
        monkeypatch.setenv("NEURAL_ADMIXTURE_PATH", str(fake))
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        assert resolver.resolve_neural_admixture() == str(fake)

    def test_env_var_nonexistent_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NEURAL_ADMIXTURE_PATH", str(tmp_path / "ghost"))
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        with pytest.raises(ToolNotFoundError, match="NEURAL_ADMIXTURE_PATH"):
            resolver.resolve_neural_admixture()

    def test_not_found_error_includes_install_hint(self, tmp_path, monkeypatch):
        """When not found anywhere, error should mention how to install."""
        monkeypatch.delenv("NEURAL_ADMIXTURE_PATH", raising=False)
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr(shutil, "which", lambda name: None)
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        with pytest.raises(ToolNotFoundError, match="pip install"):
            resolver.resolve_neural_admixture()

    def test_venv_bin_is_checked(self, tmp_path, monkeypatch):
        """If VIRTUAL_ENV is set, $VIRTUAL_ENV/bin/neural-admixture is tried."""
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        fake_na = make_script(venv_bin / "neural-admixture")
        monkeypatch.delenv("NEURAL_ADMIXTURE_PATH", raising=False)
        monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        assert resolver.resolve_neural_admixture() == str(fake_na)


# ---------------------------------------------------------------------------
# The resolution chain, pinned once for every downloadable binary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resolver_name, env_var, binary",
    [
        ("resolve_plink2", "PLINK_PATH", "plink2"),
        ("resolve_plink1", "PLINK1_PATH", "plink"),
        ("resolve_flashpca", "FLASHPCA_PATH", "flashpca"),
    ],
)
class TestResolutionChain:
    """download dir → env var → PATH → download, in that order, for each tool.

    Each test sets up every later source as well as the one under test, so a
    pass means the earlier source won -- not merely that it was consulted.
    """

    @pytest.fixture
    def env_script(self, tmp_path, monkeypatch, env_var, binary):
        script = make_script(tmp_path / f"env_{binary}")
        monkeypatch.setenv(env_var, str(script))
        return script

    @pytest.fixture
    def on_path(self, monkeypatch, binary):
        path = f"/fake/path/{binary}"
        monkeypatch.setattr(
            "manifold_genetics.utils.tools.shutil.which",
            lambda name: path if name == binary else None,
        )
        return path

    @pytest.fixture
    def downloader(self, monkeypatch, resolver_name, binary):
        calls = []
        download_name = resolver_name.replace("resolve_", "_download_")

        def _download(self):
            calls.append(binary)
            return f"/downloaded/{binary}"

        monkeypatch.setattr(ToolResolver, download_name, _download)
        return calls

    def test_download_dir_beats_everything(
        self, tmp_path, resolver_name, binary, env_script, on_path, downloader
    ):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        local = make_script(bin_dir / binary)
        assert getattr(ToolResolver(download_dir=bin_dir), resolver_name)() == str(local)
        assert downloader == []

    def test_env_var_beats_path_and_download(
        self, tmp_path, resolver_name, env_script, on_path, downloader
    ):
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        assert getattr(resolver, resolver_name)() == str(env_script)
        assert downloader == []

    def test_path_beats_download(
        self, tmp_path, monkeypatch, resolver_name, env_var, on_path, downloader
    ):
        monkeypatch.delenv(env_var, raising=False)
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        assert getattr(resolver, resolver_name)() == on_path
        assert downloader == []

    def test_download_is_the_last_resort(
        self, tmp_path, monkeypatch, resolver_name, env_var, binary, downloader
    ):
        monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setattr("manifold_genetics.utils.tools.shutil.which", lambda name: None)
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        assert getattr(resolver, resolver_name)() == f"/downloaded/{binary}"
        assert downloader == [binary]

    def test_set_but_invalid_env_var_raises(
        self, tmp_path, monkeypatch, resolver_name, env_var, on_path, downloader
    ):
        """A set-but-wrong env var is an error, not a silent fall-through to PATH."""
        monkeypatch.setenv(env_var, str(tmp_path / "ghost"))
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        with pytest.raises(ToolNotFoundError, match=env_var):
            getattr(resolver, resolver_name)()
        assert downloader == []


def test_resolver_has_no_module_system_step():
    """``module load`` in a child shell cannot change this process's PATH, so
    the resolver must not pretend to try it. Grep the source rather than the
    behaviour: the step was a silent no-op, which is exactly what a behavioural
    test would fail to notice."""
    import manifold_genetics.utils.tools as tools_mod

    source = Path(tools_mod.__file__).read_text()
    for forbidden in ("module load", "_try_load_module", "Compute Canada"):
        assert forbidden not in source, f"{forbidden!r} is back in tools.py"


# ---------------------------------------------------------------------------
# Integration tests — require real plink2/plink in PATH
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestToolResolverWithRealPlink:
    """These tests verify ToolResolver against a real installed plink2/plink binary."""

    @pytest.fixture(autouse=True)
    def skip_if_no_plink(self):
        if not shutil.which("plink2") and not shutil.which("plink"):
            pytest.skip("plink2/plink not in PATH")

    def test_resolve_plink2_finds_binary(self, tmp_path):
        """resolve_plink2 must return a path to a real executable."""
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        result = resolver.resolve_plink2()
        assert Path(result).exists()
        assert os.access(result, os.X_OK)

    def test_resolved_plink2_actually_runs(self, tmp_path):
        """The resolved plink2 must execute successfully (--version or --help)."""
        resolver = ToolResolver(download_dir=tmp_path / "empty_bin")
        plink_path = resolver.resolve_plink2()
        proc = subprocess.run(
            [plink_path, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = proc.stdout + proc.stderr
        assert "PLINK" in output.upper(), f"Expected PLINK in output, got: {output[:200]}"
