"""Tests for utils/tools.py — ToolResolver path resolution.

ToolResolver uses a fallback chain to find external binaries:
  local bin dir → env var → module system → PATH → auto-download

Tests use real temp executables (chmod +x shell scripts) rather than mocking
os.access, so the tests verify actual filesystem permission checks rather than
the mock infrastructure. Module system and download tests are isolated via
monkeypatching subprocess.run.
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
# TestModuleSystem — isolate subprocess.run failures gracefully
# ---------------------------------------------------------------------------

class TestModuleSystem:

    def test_module_available_file_not_found_returns_false(self, tmp_path, monkeypatch):
        """If 'module' command doesn't exist, _module_available returns False."""
        import manifold_genetics.utils.tools as tools_mod

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("module: command not found")

        monkeypatch.setattr(tools_mod.subprocess, "run", fake_run)
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._module_available("plink") is False

    def test_module_available_timeout_returns_false(self, tmp_path, monkeypatch):
        """Timeout on module avail returns False — doesn't propagate exception."""
        import manifold_genetics.utils.tools as tools_mod

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 5)

        monkeypatch.setattr(tools_mod.subprocess, "run", fake_run)
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._module_available("plink") is False

    def test_try_load_module_file_not_found_returns_false(self, tmp_path, monkeypatch):
        import manifold_genetics.utils.tools as tools_mod

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("module: command not found")

        monkeypatch.setattr(tools_mod.subprocess, "run", fake_run)
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._try_load_module("plink") is False

    def test_try_load_module_timeout_returns_false(self, tmp_path, monkeypatch):
        import manifold_genetics.utils.tools as tools_mod

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 5)

        monkeypatch.setattr(tools_mod.subprocess, "run", fake_run)
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._try_load_module("plink") is False

    def test_try_load_module_nonzero_exit_returns_false(self, tmp_path, monkeypatch):
        """Non-zero returncode from module load means the module wasn't loaded."""
        import manifold_genetics.utils.tools as tools_mod

        class FakeResult:
            returncode = 1
            stdout = ""
            stderr = "ERROR: Unable to locate a modulefile for 'nonexistent_module'"

        monkeypatch.setattr(tools_mod.subprocess, "run", lambda cmd, **kwargs: FakeResult())
        resolver = ToolResolver(download_dir=tmp_path)
        assert resolver._try_load_module("nonexistent_module") is False


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
            capture_output=True, text=True, timeout=15,
        )
        output = proc.stdout + proc.stderr
        assert "PLINK" in output.upper(), f"Expected PLINK in output, got: {output[:200]}"
