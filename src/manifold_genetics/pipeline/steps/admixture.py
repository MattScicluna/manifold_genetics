"""In-process admixture step.

``run_admixture()`` is the seam shared by the ``manifold-genetics admixture``
subcommand and ``run_admixture_step()``. Both go through the ``NeuralAdmixture``
wrapper, which constructs a real ``NeuralAdmixtureBackend`` when no backend is
injected — so a test backend and the real one take the SAME path. That replaces
the orchestrator's old special-case branch, which called an injected backend
directly and used ``fit_transform`` for the fit cohort where the CLI path used
``transform``.

The real external tool still runs in its own process: ``NeuralAdmixtureBackend``
shells out to ``neural-admixture train`` / ``infer``, one child per K. That
boundary lives in L1 and is unchanged here.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from ...admixture import AdmixtureBackend, NeuralAdmixture
from ..config import AdmixtureConfig, IOConfig
from .paths import admixture_output_paths

logger = logging.getLogger(__name__)

__all__ = [
    "AdmixtureStepResult",
    "admixture_output_paths",
    "run_admixture",
    "run_admixture_step",
]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class AdmixtureStepResult:
    """Typed outputs of the admixture step.

    ``q_prefix`` is the project cohort's prefix — ``<prefix>.{k}.csv`` — which is
    what the metrics step consumes.
    """

    q_prefix: Path
    k_values: Tuple[int, ...]
    dir: Path
    checkpoints_dir: Path
    fit_prefix: Path
    fit_q_files: Dict[int, Path] = field(default_factory=dict)
    project_q_files: Dict[int, Path] = field(default_factory=dict)
    skipped: bool = False


def run_admixture(
    fit_plink: PathLike,
    project_plink: PathLike,
    *,
    checkpoints_dir: PathLike,
    fit_output: PathLike,
    project_output: PathLike,
    k_min: int = 2,
    k_max: int = 10,
    force: bool = False,
    threads: Optional[int] = None,
    num_gpus: Optional[int] = None,
    batch_size: Optional[int] = None,
    model_name: str = "fit",
    backend: Optional[AdmixtureBackend] = None,
) -> Tuple[Dict[int, Path], Dict[int, Path]]:
    """Train admixture models on the fit cohort and infer on both cohorts.

    ``backend`` is passed straight through to ``NeuralAdmixture``; ``None`` means
    the real ``NeuralAdmixtureBackend``. Values like ``threads=0`` / ``num_gpus=0``
    are forwarded as given — they are meaningful, not absent.

    Returns ``(fit_q_files, project_q_files)``, each mapping K to a CSV path.
    """
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    Path(fit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(project_output).parent.mkdir(parents=True, exist_ok=True)

    admix = NeuralAdmixture(
        k_min=k_min,
        k_max=k_max,
        force=force,
        threads=threads,
        num_gpus=num_gpus,
        batch_size=batch_size,
        backend=backend,
    )

    admix.fit(fit_plink, output_dir=checkpoints_dir, model_name=model_name)
    fit_q_files = admix.transform(fit_plink, output_prefix=fit_output)
    project_q_files = admix.transform(project_plink, output_prefix=project_output)

    return fit_q_files, project_q_files


def run_admixture_step(
    io: IOConfig,
    admix: AdmixtureConfig,
    *,
    backend: Optional[AdmixtureBackend] = None,
) -> AdmixtureStepResult:
    """Run admixture for the configured K range and report where it wrote."""
    paths = admixture_output_paths(io, admix)

    logger.info(f"Running admixture K={admix.k_min}..{admix.k_max}")
    run_admixture(
        io.fit_plink,
        io.project_plink,
        checkpoints_dir=paths["checkpoints_dir"],
        fit_output=paths["fit_prefix"],
        project_output=paths["project_prefix"],
        k_min=admix.k_min,
        k_max=admix.k_max,
        threads=admix.threads,
        num_gpus=admix.num_gpus,
        batch_size=admix.batch_size,
        backend=backend,
    )

    return AdmixtureStepResult(
        q_prefix=paths["project_prefix"],
        k_values=tuple(range(admix.k_min, admix.k_max + 1)),
        dir=paths["dir"],
        checkpoints_dir=paths["checkpoints_dir"],
        fit_prefix=paths["fit_prefix"],
        fit_q_files=paths["fit_q_files"],
        project_q_files=paths["project_q_files"],
    )
