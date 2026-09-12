"""
Tool path resolution for genomics pipeline.

Resolves paths to required tools (plink2, flashpca, neural-admixture) using
a fallback chain: environment variables → module system → PATH → download/error.
"""

import gzip
import logging
import os
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

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
    Resolve tool paths using fallback chain.

    Priority order:
    1. Environment variable (PLINK_PATH, FLASHPCA_PATH, NEURAL_ADMIXTURE_PATH)
    2. Module system (module load plink)
    3. PATH lookup (shutil.which)
    4. Auto-download (FlashPCA only) or error
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

    def resolve_plink2(self) -> str:
        """
        Resolve plink2 path.

        Priority:
        1. Local bin directory (pre-downloaded during setup)
        2. PLINK_PATH environment variable
        3. Module system (module load plink)
        4. PATH lookup
        5. Auto-download to download_dir (will fail on compute nodes without internet)

        Returns:
            Path to plink2 executable

        Raises:
            ToolNotFoundError: If plink2 cannot be found
        """
        # 1. Check local bin directory FIRST (pre-downloaded during setup)
        local_plink = self.download_dir / "plink2"
        if local_plink.exists() and self._validate_executable(str(local_plink)):
            logger.debug(f"Using pre-downloaded plink2: {local_plink}")
            return str(local_plink)

        # 2. Check PLINK_PATH env var
        if env_path := os.getenv("PLINK_PATH"):
            if self._validate_executable(env_path):
                return env_path
            else:
                raise ToolNotFoundError(f"PLINK_PATH points to invalid executable: {env_path}")

        # 3. Check module system (Compute Canada clusters)
        # Try multiple plink versions
        for version in ["plink/2.00-20231024-avx2", "plink/2.00a5.8", "plink"]:
            if self._try_load_module(version):
                # Module loaded successfully, check PATH
                for name in ["plink2", "plink"]:
                    if path := shutil.which(name):
                        logger.debug(f"Found plink via module {version}: {path}")
                        return path

        # 4. Check PATH
        for name in ["plink2", "plink"]:
            if path := shutil.which(name):
                return path

        # 5. Auto-download (will fail on compute nodes without internet!)
        logger.warning(
            "plink2 not found in bin/, PATH, or modules. "
            "Attempting download (will fail on compute nodes without internet)..."
        )
        return self._download_plink2()

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
        # 1. Check local bin directory FIRST (pre-downloaded during setup)
        for local_name in ["flashpca", "flashpca_x86-64"]:
            local_flashpca = self.download_dir / local_name
            if local_flashpca.exists() and self._validate_executable(str(local_flashpca)):
                logger.debug(f"Using pre-downloaded flashPCA: {local_flashpca}")
                return str(local_flashpca)

        # 2. Check FLASHPCA_PATH env var
        if env_path := os.getenv("FLASHPCA_PATH"):
            if self._validate_executable(env_path):
                return env_path
            else:
                raise ToolNotFoundError(f"FLASHPCA_PATH points to invalid executable: {env_path}")

        # 3. Check PATH
        for name in ["flashpca", "flashpca_x86-64"]:
            if path := shutil.which(name):
                return path

        # 4. Auto-download (will fail on compute nodes without internet!)
        logger.warning(
            "FlashPCA not found in bin/, PATH, or environment. "
            "Attempting download (will fail on compute nodes without internet)..."
        )
        return self._download_flashpca()

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

    def _module_available(self, module_name: str) -> bool:
        """
        Check if a module is available via module system.

        Args:
            module_name: Name of module to check

        Returns:
            True if module is available
        """
        try:
            # Check if module command exists
            result = subprocess.run(
                ["module", "avail", module_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # If module system works and module found
            return (
                module_name.lower() in result.stderr.lower()
                or module_name.lower() in result.stdout.lower()
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _try_load_module(self, module_name: str) -> bool:
        """
        Try to load a module via module system.

        Args:
            module_name: Name of module to load

        Returns:
            True if module loaded successfully
        """
        try:
            # Try to load the module
            result = subprocess.run(
                ["bash", "-c", f"module load {module_name} 2>&1"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

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
            urllib.request.urlretrieve(url, output_gz)

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
                    urllib.request.urlretrieve(url, output_zip)
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
            urllib.request.urlretrieve(url, output_zip)

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
