"""
Tool path resolution for genomics pipeline.

Resolves paths to required tools (plink2, flashpca, neural-admixture) using
a fallback chain: environment variables → module system → PATH → download/error.
"""

import gzip
import logging
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ToolNotFoundError(Exception):
    """Raised when a required tool cannot be found."""

    pass


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
            download_dir: Directory to download tools (default: package_root/bin)
        """
        if download_dir is None:
            # Default to package bin directory
            package_root = Path(__file__).resolve().parents[3]
            download_dir = package_root / "bin"

        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

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
        url = "https://github.com/gabraham/flashpca/releases/download/v2.0/flashpca_x86-64.gz"
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
                f"Failed to download FlashPCA: {e}\n" f"Please download manually from: {url}"
            )

    def _download_plink2(self) -> str:
        """
        Download PLINK2 binary.

        Returns:
            Path to downloaded plink2 executable

        Raises:
            ToolNotFoundError: If download fails
        """
        # PLINK2 release URLs (try most recent first, then fallback)
        urls = [
            "https://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_20260110.zip",
            "https://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_latest.zip",
            "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_x86_64_20231211.zip",
        ]
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
                f"Failed to download PLINK2: {e}\n"
                f"Please download manually from: https://www.cog-genomics.org/plink/2.0/"
            )

    def _download_plink1(self) -> str:
        """
        Download PLINK v1.9 binary.

        Returns:
            Path to downloaded plink executable

        Raises:
            ToolNotFoundError: If download fails
        """
        url = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip"
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
                f"Failed to download PLINK v1.9: {e}\n"
                f"Please download manually from: https://www.cog-genomics.org/plink/1.9/"
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
