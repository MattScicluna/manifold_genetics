"""Tests for ToolResolver's resolution fallbacks and the _download_* helpers.

`urllib.request.urlretrieve` and `subprocess.run` are stubbed; the real
gzip/zip extraction, chmod, and cleanup logic runs against fixture archives
written to tmp_path. No network, no real binaries.
"""

import gzip
import urllib.request
import zipfile
from pathlib import Path

import pytest

from manifold_genetics.utils.tools import ToolNotFoundError, ToolResolver


@pytest.fixture
def resolver(tmp_path):
    return ToolResolver(download_dir=tmp_path)


@pytest.fixture(autouse=True)
def _clear_tool_env(monkeypatch):
    for var in ("PLINK_PATH", "FLASHPCA_PATH", "NEURAL_ADMIXTURE_PATH"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def platform_is(monkeypatch):
    """Pin the platform the download helpers see.

    They read ``platform.system()`` and ``platform.machine()`` at call time, so
    without this every test here asserts against whatever host it happens to run
    on. That went unnoticed while CI was Linux-only: the first macOS run failed
    four of them, and in every case the product was right and the test was not
    -- flashpca correctly refused (there is no macOS build) and plink2 correctly
    had a single mac URL where the test assumed two.
    """

    def _pin(system: str, machine: str):
        monkeypatch.setattr("manifold_genetics.utils.tools.platform.system", lambda: system)
        monkeypatch.setattr("manifold_genetics.utils.tools.platform.machine", lambda: machine)

    return _pin


@pytest.fixture(autouse=True)
def _default_platform(platform_is):
    """Linux x86-64 unless a test says otherwise, so results do not vary by host."""
    platform_is("Linux", "x86_64")


def _gz_writer(payload=b"#!/bin/sh\necho tool\n"):
    def _fake(url, dest):
        with gzip.open(dest, "wb") as fh:
            fh.write(payload)

    return _fake


def _zip_writer(members):
    def _fake(url, dest):
        with zipfile.ZipFile(dest, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)

    return _fake


# ---------------------------------------------------------------------------
# resolve_* fallback chain (module -> PATH -> download)
# ---------------------------------------------------------------------------


def test_resolve_plink2_uses_module_then_path(resolver, monkeypatch):
    monkeypatch.setattr(resolver, "_try_load_module", lambda m: m == "plink")
    monkeypatch.setattr(
        "manifold_genetics.utils.tools.shutil.which",
        lambda n: "/opt/plink2" if n == "plink2" else None,
    )
    assert resolver.resolve_plink2() == "/opt/plink2"


def test_resolve_plink2_falls_back_to_path(resolver, monkeypatch):
    monkeypatch.setattr(resolver, "_try_load_module", lambda m: False)
    monkeypatch.setattr(
        "manifold_genetics.utils.tools.shutil.which",
        lambda n: "/usr/bin/plink2" if n == "plink2" else None,
    )
    assert resolver.resolve_plink2() == "/usr/bin/plink2"


def test_resolve_plink2_falls_back_to_download(resolver, monkeypatch):
    monkeypatch.setattr(resolver, "_try_load_module", lambda m: False)
    monkeypatch.setattr("manifold_genetics.utils.tools.shutil.which", lambda n: None)
    monkeypatch.setattr(resolver, "_download_plink2", lambda: "/downloaded/plink2")
    assert resolver.resolve_plink2() == "/downloaded/plink2"


def test_resolve_flashpca_falls_back_to_path(resolver, monkeypatch):
    monkeypatch.setattr(
        "manifold_genetics.utils.tools.shutil.which",
        lambda n: "/usr/bin/flashpca" if n == "flashpca" else None,
    )
    assert resolver.resolve_flashpca() == "/usr/bin/flashpca"


def test_resolve_flashpca_falls_back_to_download(resolver, monkeypatch):
    monkeypatch.setattr("manifold_genetics.utils.tools.shutil.which", lambda n: None)
    monkeypatch.setattr(resolver, "_download_flashpca", lambda: "/downloaded/flashpca")
    assert resolver.resolve_flashpca() == "/downloaded/flashpca"


def test_resolve_neural_admixture_uses_path(resolver, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "")
    monkeypatch.setattr(
        "manifold_genetics.utils.tools.shutil.which",
        lambda n: "/usr/bin/neural-admixture" if n == "neural-admixture" else None,
    )
    assert resolver.resolve_neural_admixture() == "/usr/bin/neural-admixture"


# ---------------------------------------------------------------------------
# _module_available success path
# ---------------------------------------------------------------------------


def test_module_available_true_when_named_in_output(resolver, monkeypatch):
    class R:
        stdout = "plink/2.00a5.8\n"
        stderr = ""

    monkeypatch.setattr("manifold_genetics.utils.tools.subprocess.run", lambda *a, **k: R())
    assert resolver._module_available("plink") is True


def test_try_load_module_true_on_zero_exit(resolver, monkeypatch):
    class R:
        returncode = 0

    monkeypatch.setattr("manifold_genetics.utils.tools.subprocess.run", lambda *a, **k: R())
    assert resolver._try_load_module("plink") is True


# ---------------------------------------------------------------------------
# _download_flashpca
# ---------------------------------------------------------------------------


def test_download_flashpca_extracts_and_cleans_up(resolver, tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlretrieve", _gz_writer())
    path = resolver._download_flashpca()
    assert Path(path) == tmp_path / "flashpca_x86-64"
    assert Path(path).exists()
    assert Path(path).stat().st_mode & 0o111  # executable bit set
    assert not (tmp_path / "flashpca_x86-64.gz").exists()  # gz removed


def test_download_flashpca_returns_existing(resolver, tmp_path, monkeypatch):
    binp = tmp_path / "flashpca_x86-64"
    binp.write_text("#!/bin/sh\n")
    binp.chmod(0o755)

    def _boom(url, dest):
        raise AssertionError("should not download when binary already present")

    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)
    assert resolver._download_flashpca() == str(binp)


def test_download_flashpca_wraps_failure(resolver, monkeypatch):
    def _boom(url, dest):
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)
    with pytest.raises(ToolNotFoundError, match="Failed to download FlashPCA"):
        resolver._download_flashpca()


# ---------------------------------------------------------------------------
# _download_plink2
# ---------------------------------------------------------------------------


def test_download_plink2_extracts_zip(resolver, tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlretrieve", _zip_writer({"plink2": "#!/bin/sh\n"}))
    path = resolver._download_plink2()
    assert Path(path) == tmp_path / "plink2"
    assert Path(path).exists()
    assert not (tmp_path / "plink2.zip").exists()


def test_download_plink2_tries_urls_in_order(resolver, tmp_path, monkeypatch):
    attempts = []
    good = _zip_writer({"plink2": "#!/bin/sh\n"})

    def _flaky(url, dest):
        attempts.append(url)
        if len(attempts) == 1:
            raise OSError("first mirror down")
        good(url, dest)

    monkeypatch.setattr(urllib.request, "urlretrieve", _flaky)
    resolver._download_plink2()
    assert len(attempts) == 2


def test_download_plink2_wraps_failure(resolver, monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda url, dest: (_ for _ in ()).throw(OSError("down")),
    )
    with pytest.raises(ToolNotFoundError, match="Failed to download PLINK2"):
        resolver._download_plink2()


# ---------------------------------------------------------------------------
# _download_plink1
# ---------------------------------------------------------------------------


def test_download_plink1_extracts_and_removes_extras(resolver, tmp_path, monkeypatch):
    members = {"plink": "#!/bin/sh\n", "LICENSE": "x", "toy.ped": "x", "prettify": "x"}
    monkeypatch.setattr(urllib.request, "urlretrieve", _zip_writer(members))
    path = resolver._download_plink1()
    assert Path(path) == tmp_path / "plink"
    assert Path(path).exists()
    assert not (tmp_path / "LICENSE").exists()
    assert not (tmp_path / "toy.ped").exists()
    assert not (tmp_path / "plink1.zip").exists()


def test_download_plink1_wraps_failure(resolver, monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlretrieve",
        lambda url, dest: (_ for _ in ()).throw(OSError("down")),
    )
    with pytest.raises(ToolNotFoundError, match="Failed to download PLINK v1.9"):
        resolver._download_plink1()


# ---------------------------------------------------------------------------
# install_tools / resolve_all
# ---------------------------------------------------------------------------


def test_install_tools_with_and_without_plink1(resolver, monkeypatch):
    monkeypatch.setattr(resolver, "_download_plink2", lambda: "p2")
    monkeypatch.setattr(resolver, "_download_flashpca", lambda: "fp")
    monkeypatch.setattr(resolver, "_download_plink1", lambda: "p1")

    assert resolver.install_tools(include_plink1=True) == {
        "plink2": "p2",
        "flashpca": "fp",
        "plink1": "p1",
    }
    assert set(resolver.install_tools(include_plink1=False)) == {"plink2", "flashpca"}


def test_resolve_all_aggregates(resolver, monkeypatch):
    monkeypatch.setattr(resolver, "resolve_plink2", lambda: "p2")
    monkeypatch.setattr(resolver, "resolve_flashpca", lambda: "fp")
    monkeypatch.setattr(resolver, "resolve_neural_admixture", lambda: "na")
    assert resolver.resolve_all() == {
        "plink2": "p2",
        "flashpca": "fp",
        "neural_admixture": "na",
    }


# ---------------------------------------------------------------------------
# Platforms other than Linux x86-64
#
# Added after the first macOS CI run, which failed four tests above. The product
# was correct in every case; these pin that correctness down so it is asserted
# rather than assumed.
# ---------------------------------------------------------------------------


def test_flashpca_refuses_on_macos_and_names_the_alternative(resolver, platform_is):
    """There is no macOS flashpca build, and that must not read as a failed download.

    A user on a Mac has no way to satisfy this dependency, so the error has to
    say so and point at the in-process backend -- which is the default and needs
    no binary -- rather than leaving them retrying a download that cannot work.
    """
    platform_is("Darwin", "arm64")

    with pytest.raises(ToolNotFoundError) as excinfo:
        resolver._download_flashpca()

    message = str(excinfo.value)
    assert "Darwin/arm64" in message, "the error must name the platform it refused"
    assert "--pca-backend python" in message, "the error must name the way forward"


def test_flashpca_refuses_on_linux_arm(resolver, platform_is):
    """Linux is not sufficient; the published build is x86-64 only.

    The message quotes the machine string as the platform reports it
    (``aarch64``), not the normalised alias used for the URL lookup, so that it
    matches what a user sees from ``uname -m``.
    """
    platform_is("Linux", "aarch64")

    with pytest.raises(ToolNotFoundError, match="Linux/aarch64"):
        resolver._download_flashpca()


@pytest.mark.parametrize(
    "system, machine, expected",
    [
        ("Darwin", "arm64", "plink2_mac_arm64"),
        ("Darwin", "x86_64", "plink2_mac_avx2"),
        ("Linux", "x86_64", "plink2_linux_x86_64"),
        ("Windows", "amd64", "plink2_win64"),
    ],
)
def test_plink2_downloads_the_build_for_the_platform(
    resolver, tmp_path, monkeypatch, platform_is, system, machine, expected
):
    """Every URL was once linux_x86_64, so a Mac fetched a Linux binary and then
    failed to execute it -- which reads as a corrupt download, not as a wrong
    build. ``amd64`` is here because Windows reports that rather than x86_64.
    """
    platform_is(system, machine)
    seen = []

    def _record(url, dest):
        seen.append(url)
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("plink2", "#!/bin/sh\n")

    monkeypatch.setattr(urllib.request, "urlretrieve", _record)
    resolver._download_plink2()

    assert expected in seen[0], f"fetched {seen[0]} on {system}/{machine}"
