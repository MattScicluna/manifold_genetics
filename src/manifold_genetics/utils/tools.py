"""
Tool path resolution for genomics pipeline.

Resolves paths to required tools (plink2, flashpca, neural-admixture) using
a fallback chain: download directory → environment variable → PATH → download/error.
"""

import gzip
import logging
import os
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Where downloaded binaries go, overriding every default.
TOOL_DIR_ENV = "MANIFOLD_GENETICS_TOOL_DIR"

# Release assets, per tool and platform.
#
# Only plink's Linux x86-64 builds have a stable `_latest` alias; the mac and
# Windows assets are dated, so these need bumping when upstream cuts a release.
# The error raised on a 404 points at the download page, which is the fallback.
#
# flashpca ships exactly one artefact, for Linux x86-64. There is nothing to
# fetch anywhere else -- which is what the in-process PCA backend is for, and
# what the error below says.
_PLINK2 = "https://s3.amazonaws.com/plink2-assets"
_PLINK1 = "https://s3.amazonaws.com/plink1-assets"
_PLINK2_DATE = "20260910"
_PLINK1_DATE = "20231211"

# Values are candidates, tried in order: the stable `_latest` alias where one
# exists, then a pinned build, so an upstream alias change is survivable.
_URLS = {
    "plink2": {
        ("Linux", "x86_64"): (
            f"{_PLINK2}/plink2_linux_x86_64_latest.zip",
            f"{_PLINK2}/alpha7/plink2_linux_x86_64_{_PLINK2_DATE}.zip",
        ),
        ("Darwin", "arm64"): (f"{_PLINK2}/alpha7/plink2_mac_arm64_{_PLINK2_DATE}.zip",),
        ("Darwin", "x86_64"): (f"{_PLINK2}/alpha7/plink2_mac_avx2_{_PLINK2_DATE}.zip",),
        ("Windows", "x86_64"): (f"{_PLINK2}/alpha7/plink2_win64_{_PLINK2_DATE}.zip",),
    },
    "plink1": {
        ("Linux", "x86_64"): (
            f"{_PLINK1}/plink_linux_x86_64_latest.zip",
            f"{_PLINK1}/plink_linux_x86_64_{_PLINK1_DATE}.zip",
        ),
        ("Darwin", "arm64"): (f"{_PLINK1}/plink_mac_{_PLINK1_DATE}.zip",),
        ("Darwin", "x86_64"): (f"{_PLINK1}/plink_mac_{_PLINK1_DATE}.zip",),
        ("Windows", "x86_64"): (f"{_PLINK1}/plink_win64_{_PLINK1_DATE}.zip",),
    },
    "flashpca": {
        ("Linux", "x86_64"): (
            "https://github.com/gabraham/flashpca/releases/download/v2.0/flashpca_x86-64.gz",
        ),
    },
}

_HOMEPAGES = {
    "plink2": "https://www.cog-genomics.org/plink/2.0/",
    "plink1": "https://www.cog-genomics.org/plink/1.9/",
    "flashpca": "https://github.com/gabraham/flashpca/releases",
}

# platform.machine() is not consistent across operating systems.
_MACHINE_ALIASES = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "x64": "x86_64",
    "arm64": "arm64",
    "aarch64": "arm64",
}


def fetch_url(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination``, falling back to curl and then wget.

    urllib verifies against Python's bundled certificate list; curl and wget
    verify against the operating system's. On a network with a TLS-intercepting
    proxy only the latter contains the proxy's root certificate, because an
    administrator put it there -- so urllib raises CERTIFICATE_VERIFY_FAILED on
    machines where downloading works perfectly well otherwise. Reported
    2026-09-13 from a managed network, first for the HGDP archive and then here,
    for plink2.

    Verification is never disabled. The trust decision is delegated to the
    machine's own configuration; it is not skipped. Do not "fix" a certificate
    error by turning the check off.

    Raises:
        OSError: every method failed. The urllib error is preserved, since it is
            usually the most descriptive.
    """
    try:
        urllib.request.urlretrieve(url, destination)
        return
    except OSError as urllib_error:
        logger.info("urllib could not fetch %s (%s); trying curl or wget", url, urllib_error)
        for tool, args in (
            ("curl", ["-fsSL", url, "-o", str(destination)]),
            ("wget", ["-q", url, "-O", str(destination)]),
        ):
            executable = shutil.which(tool)
            if not executable:
                continue
            try:
                subprocess.run([executable] + args, check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                logger.info("%s failed: %s", tool, exc)
                Path(destination).unlink(missing_ok=True)
                continue
            if Path(destination).exists() and Path(destination).stat().st_size > 0:
                logger.info("Downloaded with %s", tool)
                return
        raise urllib_error


class ToolNotFoundError(Exception):
    """Raised when a required tool cannot be found."""


def download_url(tool: str, *, system: str, machine: str) -> str:
    """The preferred release asset for ``tool`` on the given platform."""
    return download_candidates(tool, system=system, machine=machine)[0]


def download_candidates(tool: str, *, system: str, machine: str) -> Tuple[str, ...]:
    """Every release asset to try for ``tool`` on the given platform, in order.

    Raises:
        ToolNotFoundError: the tool is unknown, or upstream publishes no build
            for this platform. Both name the platform, because the previous
            behaviour -- fetching the Linux binary everywhere -- surfaced as a
            binary that would not execute, which reads as a corrupt download.
    """
    if tool not in _URLS:
        raise ToolNotFoundError(f"Unknown tool {tool!r}; choose from {', '.join(sorted(_URLS))}")

    key = (system, _MACHINE_ALIASES.get(machine.lower(), machine))
    urls = _URLS[tool].get(key)
    if urls is not None:
        return urls

    if tool == "flashpca":
        raise ToolNotFoundError(
            f"flashpca is published only for Linux x86-64, so there is no build for "
            f"{system}/{machine}. Use the in-process PCA backend instead, which is "
            f"the default and needs no binary: PCA() or --pca-backend python."
        )
    raise ToolNotFoundError(
        f"No {tool} build is known for {system}/{machine}. Download one from "
        f"{_HOMEPAGES[tool]} and point {TOOL_DIR_ENV} at the directory holding it."
    )


def _repository_bin() -> Optional[Path]:
    """The checkout's ``bin/``, or None when running from an installed package.

    ``manifold-genetics setup`` has always written there and the example scripts
    look there, so a development checkout keeps using it. Detected by the
    presence of ``pyproject.toml``: computing it from ``__file__`` alone is what
    made an installed package try to write beside site-packages.
    """
    root = Path(__file__).resolve().parents[3]
    return root / "bin" if (root / "pyproject.toml").exists() else None


def _user_cache_bin() -> Path:
    """Per-user cache directory for downloaded binaries."""
    try:
        from platformdirs import user_cache_dir
    except ImportError:  # pragma: no cover - platformdirs is a dependency
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return base / "manifold-genetics" / "bin"
    return Path(user_cache_dir("manifold-genetics")) / "bin"


class ToolResolver:
    """
    Resolve tool paths using a fallback chain.

    Priority order, for the downloadable binaries (plink2, plink v1.9, flashpca):
    1. The download directory (pre-fetched by ``manifold-genetics setup``)
    2. The tool's environment variable (PLINK_PATH, PLINK1_PATH, FLASHPCA_PATH)
    3. PATH lookup (shutil.which)
    4. Auto-download into the download directory

    neural-admixture is a Python package, so it has no download step: its
    environment variable, then the active virtual environment, then PATH.

    There is deliberately no Environment Modules step. A child shell cannot
    change this process's ``PATH``, so loading a module from here never made
    the binary visible; load it in the shell before running instead, and the
    PATH lookup finds it.
    """

    def __init__(self, download_dir: Optional[Path] = None):
        """
        Initialize tool resolver.

        Args:
            download_dir: Where downloaded binaries live. Defaults, in order, to
                ``$MANIFOLD_GENETICS_TOOL_DIR``, the checkout's ``bin/`` when
                running from one, and otherwise a per-user cache directory.

        The directory is *not* created here. This object is constructed by
        ordinary code paths -- ``PCA(backend="flashpca")`` among them -- and
        creating a directory as a side effect of that put a stray ``bin/``
        beside site-packages on every installed copy.
        """
        if download_dir is None:
            override = os.environ.get(TOOL_DIR_ENV)
            if override:
                download_dir = Path(override)
            else:
                download_dir = _repository_bin() or _user_cache_bin()

        self.download_dir = Path(download_dir).expanduser()

    def _ensure_download_dir(self) -> Path:
        """Create the download directory, at the point something is downloaded."""
        self.download_dir.mkdir(parents=True, exist_ok=True)
        return self.download_dir

    def _resolve_binary(
        self,
        *,
        names: Sequence[str],
        env_var: str,
        download: Callable[[], str],
        label: str,
        local_names: Optional[Sequence[str]] = None,
    ) -> str:
        """
        The four-step chain shared by every downloadable binary.

        Args:
            names: Executable names to look for on PATH, in order.
            env_var: Environment variable that may name the executable outright.
            download: Fetches the binary into ``download_dir`` and returns its path.
            label: How the tool is called in log and error messages.
            local_names: File names to accept in ``download_dir``. Defaults to
                ``names``; it is separate because what a download leaves behind
                is not always what PATH would be searched for.

        Returns:
            Path to the executable.

        Raises:
            ToolNotFoundError: ``env_var`` is set but does not name a valid
                executable (set-but-wrong is an error, never silently skipped),
                or nothing was found and the download failed.
        """
        # 1. Download directory first: what `setup` pre-fetched wins.
        for local_name in (local_names if local_names is not None else names):
            local = self.download_dir / local_name
            if local.exists() and self._validate_executable(str(local)):
                logger.debug(f"Using pre-downloaded {label}: {local}")
                return str(local)

        # 2. The tool's environment variable.
        if env_path := os.getenv(env_var):
            if self._validate_executable(env_path):
                return env_path
            raise ToolNotFoundError(f"{env_var} points to invalid executable: {env_path}")

        # 3. PATH.
        for name in names:
            if path := shutil.which(name):
                return path

        # 4. Auto-download (will fail on compute nodes without internet!)
        logger.warning(
            f"{label} not found in {self.download_dir}, {env_var}, or PATH. "
            "Attempting download (will fail on compute nodes without internet)..."
        )
        return download()

    def resolve_plink2(self) -> str:
        """
        Resolve plink2 path.

        Priority:
        1. Local bin directory (pre-downloaded during setup)
        2. PLINK_PATH environment variable
        3. PATH lookup (plink2, then plink)
        4. Auto-download to download_dir (will fail on compute nodes without internet)

        Returns:
            Path to plink2 executable

        Raises:
            ToolNotFoundError: If plink2 cannot be found
        """
        return self._resolve_binary(
            names=["plink2", "plink"],
            local_names=["plink2"],
            env_var="PLINK_PATH",
            download=self._download_plink2,
            label="plink2",
        )

    def resolve_plink1(self) -> str:
        """
        Resolve plink v1.9 path.

        Needed by the preprocessing shell for LD pruning and the WRayner checker,
        which plink2 does not implement the same way.

        Priority:
        1. Local bin directory (pre-downloaded during setup)
        2. PLINK1_PATH environment variable
        3. PATH lookup
        4. Auto-download to download_dir (will fail on compute nodes without internet)

        Returns:
            Path to plink (v1.9) executable

        Raises:
            ToolNotFoundError: If plink v1.9 cannot be found
        """
        return self._resolve_binary(
            names=["plink"],
            env_var="PLINK1_PATH",
            download=self._download_plink1,
            label="plink (v1.9)",
        )

    def resolve_flashpca(self) -> str:
        """
        Resolve FlashPCA path.

        Priority:
        1. Local bin directory (pre-downloaded during setup)
        2. FLASHPCA_PATH environment variable
        3. PATH lookup
        4. Auto-download to download_dir (will fail on compute nodes without internet)

        Returns:
            Path to flashpca executable

        Raises:
            ToolNotFoundError: If download fails
        """
        return self._resolve_binary(
            names=["flashpca", "flashpca_x86-64"],
            env_var="FLASHPCA_PATH",
            download=self._download_flashpca,
            label="FlashPCA",
        )

    def resolve_neural_admixture(self) -> str:
        """
        Resolve neural-admixture path.

        Priority:
        1. NEURAL_ADMIXTURE_PATH environment variable
        2. Check virtual environment (current or package .venv)
        3. PATH lookup
        4. Error

        Returns:
            Path to neural-admixture executable

        Raises:
            ToolNotFoundError: If neural-admixture cannot be found
        """
        # 1. Check NEURAL_ADMIXTURE_PATH env var
        if env_path := os.getenv("NEURAL_ADMIXTURE_PATH"):
            if self._validate_executable(env_path):
                return env_path
            else:
                raise ToolNotFoundError(
                    f"NEURAL_ADMIXTURE_PATH points to invalid executable: {env_path}"
                )

        # 2. Check virtual environment
        # Try current venv first
        venv_bin = Path(os.getenv("VIRTUAL_ENV", "")) / "bin" / "neural-admixture"
        if venv_bin.exists() and self._validate_executable(str(venv_bin)):
            return str(venv_bin)

        # 3. Check PATH
        if path := shutil.which("neural-admixture"):
            return path

        # 4. Not found
        raise ToolNotFoundError(
            "neural-admixture not found.\n\n"
            "To fix this, install neural-admixture with:\n"
            "  pip install neural-admixture\n\n"
            "Or manually set the path:\n"
            "  export NEURAL_ADMIXTURE_PATH=/path/to/neural-admixture"
        )

    def _validate_executable(self, path: str) -> bool:
        """
        Check if path points to a valid executable.

        Args:
            path: Path to check

        Returns:
            True if path exists and is executable
        """
        path_obj = Path(path)
        return path_obj.exists() and path_obj.is_file() and os.access(path, os.X_OK)

    def _download_flashpca(self) -> str:
        """
        Download FlashPCA v2.0 binary.

        Returns:
            Path to downloaded flashpca executable

        Raises:
            ToolNotFoundError: If download fails
        """
        url = download_url("flashpca", system=platform.system(), machine=platform.machine())
        self._ensure_download_dir()
        output_gz = self.download_dir / "flashpca_x86-64.gz"
        output_bin = self.download_dir / "flashpca_x86-64"

        # Check if already downloaded
        if output_bin.exists() and self._validate_executable(str(output_bin)):
            logger.info(f"FlashPCA already downloaded: {output_bin}")
            return str(output_bin)

        try:
            # Download gzipped binary
            logger.info(f"Downloading FlashPCA from {url}...")
            fetch_url(url, output_gz)

            # Decompress
            logger.info(f"Decompressing to {output_bin}...")
            with gzip.open(output_gz, "rb") as f_in:
                with open(output_bin, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Make executable
            output_bin.chmod(0o755)

            # Clean up gz file
            output_gz.unlink()

            logger.info(f"FlashPCA installed successfully: {output_bin}")
            return str(output_bin)

        except Exception as e:
            raise ToolNotFoundError(
                f"Failed to download FlashPCA from {url}: {e}\n"
                f"It is optional -- the default PCA backend needs no binary. To use it "
                f"anyway, download it and point {TOOL_DIR_ENV} at the directory."
            )

    def _download_plink2(self) -> str:
        """
        Download PLINK2 binary.

        Returns:
            Path to downloaded plink2 executable

        Raises:
            ToolNotFoundError: If download fails
        """
        urls = list(
            download_candidates("plink2", system=platform.system(), machine=platform.machine())
        )
        self._ensure_download_dir()
        output_zip = self.download_dir / "plink2.zip"
        output_bin = self.download_dir / "plink2"

        # Check if already downloaded
        if output_bin.exists() and self._validate_executable(str(output_bin)):
            logger.info(f"PLINK2 already downloaded: {output_bin}")
            return str(output_bin)

        try:
            # Download zip file (try URLs in order)
            last_error = None
            for url in urls:
                try:
                    logger.info(f"Downloading PLINK2 from {url}...")
                    fetch_url(url, output_zip)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    continue

            if last_error is not None:
                raise last_error

            # Extract binary
            logger.info(f"Extracting to {self.download_dir}...")
            import zipfile

            with zipfile.ZipFile(output_zip, "r") as zip_ref:
                zip_ref.extractall(self.download_dir)

            # Make executable
            output_bin.chmod(0o755)

            # Clean up zip file
            output_zip.unlink()

            logger.info(f"PLINK2 installed successfully: {output_bin}")
            return str(output_bin)

        except Exception as e:
            raise ToolNotFoundError(
                f"Failed to download PLINK2 from {urls[0]}: {e}\n"
                f"Download it from {_HOMEPAGES['plink2']} and either put it on PATH or "
                f"point {TOOL_DIR_ENV} at the directory holding it."
            )

    def _download_plink1(self) -> str:
        """
        Download PLINK v1.9 binary.

        Returns:
            Path to downloaded plink executable

        Raises:
            ToolNotFoundError: If download fails
        """
        url = download_url("plink1", system=platform.system(), machine=platform.machine())
        self._ensure_download_dir()
        output_zip = self.download_dir / "plink1.zip"
        output_bin = self.download_dir / "plink"

        # Check if already downloaded
        if output_bin.exists() and self._validate_executable(str(output_bin)):
            logger.info(f"PLINK (v1.9) already downloaded: {output_bin}")
            return str(output_bin)

        try:
            # Download zip file
            logger.info(f"Downloading PLINK v1.9 from {url}...")
            fetch_url(url, output_zip)

            # Extract binary
            logger.info(f"Extracting to {self.download_dir}...")
            import zipfile

            with zipfile.ZipFile(output_zip, "r") as zip_ref:
                zip_ref.extractall(self.download_dir)

            # Make executable
            output_bin.chmod(0o755)

            # Clean up zip file and extra files
            output_zip.unlink()
            for extra in [
                "LICENSE",
                "prettify",
                "toy.ped",
                "toy.map",
                "toy.fam",
                "toy.bed",
                "toy.bim",
            ]:
                extra_path = self.download_dir / extra
                if extra_path.exists():
                    extra_path.unlink()

            logger.info(f"PLINK v1.9 installed successfully: {output_bin}")
            return str(output_bin)

        except Exception as e:
            raise ToolNotFoundError(
                f"Failed to download PLINK v1.9 from {url}: {e}\n"
                f"Download it from {_HOMEPAGES['plink1']} and either put it on PATH or "
                f"point {TOOL_DIR_ENV} at the directory holding it."
            )

    def install_tools(self, include_plink1: bool = True) -> dict:
        """
        Install external tools to the package bin/ directory.

        Args:
            include_plink1: If True, download PLINK v1.9 as well.

        Returns:
            Dictionary with keys: plink2, flashpca, and optionally plink1
        """
        tools = {
            "plink2": self._download_plink2(),
            "flashpca": self._download_flashpca(),
        }
        if include_plink1:
            tools["plink1"] = self._download_plink1()
        return tools

    def resolve_all(self) -> dict:
        """
        Resolve all required tools and return paths.

        Returns:
            Dictionary with keys: plink2, flashpca, neural_admixture

        Raises:
            ToolNotFoundError: If any tool cannot be found
        """
        return {
            "plink2": self.resolve_plink2(),
            "flashpca": self.resolve_flashpca(),
            "neural_admixture": self.resolve_neural_admixture(),
        }
