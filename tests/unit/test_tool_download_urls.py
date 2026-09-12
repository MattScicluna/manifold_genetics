"""Which binary gets downloaded, and where it is put.

Both were wrong in a way that only shows up once the package is installed rather
than run from a checkout:

* ``download_dir`` defaulted to ``package_root/bin``, computed from
  ``__file__``. In a checkout that is the repository's ``bin/``. In a
  ``pip install`` it is a directory *beside site-packages* -- the wrong place,
  often not writable, and wiped on upgrade. Since the constructor also created
  it eagerly, merely building a ``ToolResolver`` made a stray directory.
* every download URL was ``linux_x86_64``. On macOS the resolver fetched a Linux
  binary and then failed to execute it, which reads as a corrupt download rather
  than as "there is no build for your platform".
"""

import platform
from pathlib import Path

import pytest

from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver, download_url


class TestDownloadUrl:
    @pytest.mark.parametrize(
        "system,machine,expected",
        [
            ("Linux", "x86_64", "linux_x86_64"),
            ("Darwin", "arm64", "mac_arm64"),
            ("Darwin", "x86_64", "mac_avx2"),
            ("Windows", "AMD64", "win64"),
        ],
    )
    def test_plink2_matches_the_platform(self, system, machine, expected):
        assert expected in download_url("plink2", system=system, machine=machine)

    @pytest.mark.parametrize(
        "system,machine,expected",
        [
            ("Linux", "x86_64", "plink_linux_x86_64"),
            ("Darwin", "arm64", "plink_mac"),
            ("Windows", "AMD64", "plink_win64"),
        ],
    )
    def test_plink1_matches_the_platform(self, system, machine, expected):
        assert expected in download_url("plink1", system=system, machine=machine)

    def test_flashpca_is_available_for_linux_only(self):
        assert "x86-64" in download_url("flashpca", system="Linux", machine="x86_64")

    @pytest.mark.parametrize("system,machine", [("Darwin", "arm64"), ("Windows", "AMD64")])
    def test_flashpca_says_there_is_no_build_rather_than_fetching_a_linux_one(
        self, system, machine
    ):
        """Upstream ships one artefact, `flashpca_x86-64`. There is nothing to fetch.

        The in-process PCA backend is the answer on those platforms, so the error
        has to say so instead of looking like a broken download.
        """
        with pytest.raises(ToolNotFoundError) as error:
            download_url("flashpca", system=system, machine=machine)

        message = str(error.value)
        assert system in message
        assert "backend" in message  # points at the in-process one

    def test_an_unknown_platform_names_itself(self):
        with pytest.raises(ToolNotFoundError, match="Linux/riscv64"):
            download_url("plink2", system="Linux", machine="riscv64")

    def test_it_resolves_for_the_platform_it_is_running_on(self):
        # Guards against a table that is fine in the tests above and wrong for
        # the machine anybody actually uses.
        assert download_url("plink2", system=platform.system(), machine=platform.machine())

    def test_an_unknown_tool_is_an_error(self):
        with pytest.raises(ToolNotFoundError, match="nonesuch"):
            download_url("nonesuch", system="Linux", machine="x86_64")


class TestWhereToolsAreKept:
    def test_constructing_a_resolver_creates_nothing(self, tmp_path, monkeypatch):
        """It is constructed by `PCA(backend="flashpca")`, among other things.

        Creating a directory as a side effect of building an object means an
        import-adjacent code path writes to the filesystem, in a location that
        for an installed package is not even the right one.
        """
        monkeypatch.setenv("MANIFOLD_GENETICS_TOOL_DIR", str(tmp_path / "tools"))

        ToolResolver()

        assert not (tmp_path / "tools").exists()

    def test_an_explicit_directory_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MANIFOLD_GENETICS_TOOL_DIR", str(tmp_path / "env"))

        assert ToolResolver(download_dir=tmp_path / "given").download_dir == tmp_path / "given"

    def test_the_environment_variable_is_honoured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MANIFOLD_GENETICS_TOOL_DIR", str(tmp_path / "env"))

        assert ToolResolver().download_dir == tmp_path / "env"

    def test_it_is_not_inside_the_installed_package(self, monkeypatch):
        """The defect this whole module existed to hide.

        Whatever the default resolves to, it must not be under the package, or a
        pip upgrade throws the downloaded binaries away -- and a system install
        cannot write there at all.
        """
        monkeypatch.delenv("MANIFOLD_GENETICS_TOOL_DIR", raising=False)
        import manifold_genetics

        package = Path(manifold_genetics.__file__).resolve().parent
        resolved = ToolResolver().download_dir.resolve()

        assert package not in resolved.parents
        assert resolved != package

    def test_a_development_checkout_uses_the_repository_bin(self, monkeypatch):
        # Where `manifold-genetics setup` has always put them, and where the
        # example scripts look. Keep working for everyone already set up.
        monkeypatch.delenv("MANIFOLD_GENETICS_TOOL_DIR", raising=False)
        repo = Path(__file__).resolve().parents[2]
        if not (repo / "pyproject.toml").exists():  # pragma: no cover
            pytest.skip("not a checkout")

        assert ToolResolver().download_dir == repo / "bin"

    def test_it_falls_back_to_a_user_cache_directory(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MANIFOLD_GENETICS_TOOL_DIR", raising=False)
        # Pretend the package is installed rather than checked out.
        monkeypatch.setattr("manifold_genetics.utils.tools._repository_bin", lambda: None)

        resolved = ToolResolver().download_dir

        assert "manifold" in str(resolved).lower()
        assert resolved.is_absolute()
