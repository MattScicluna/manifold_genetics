"""
Command-line interface for manifold-genetics.

Provides commands for PCA, admixture, embeddings, visualization, and full pipeline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .admixture import NeuralAdmixture
from .embeddings import PHATE, TSNE, UMAP, DiffusionMap
from .metrics import compute_admixture_preservation, compute_geographic_preservation
from .pca import PCA
from .pipeline import run_pipeline
from .utils.io import read_colormap
from .utils.tools import ToolResolver
from .utils.validation import (
    validate_admixture_csv,
    validate_colormap_json,
    validate_column_in_csv,
    validate_embedding_csv,
    validate_geographic_csv,
    validate_label_column,
    validate_labels_colormap_match,
    validate_labels_csv,
    validate_sample_id_overlap,
)
from .visualization import (
    plot_admixture_bar_grid,
    plot_admixture_embedding_grid,
    plot_knn_composition,
    plot_pca_pairs,
    plot_projection,
    visualize,
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)-40s - %(levelname)s - %(message)s"
    )


def _resolve_k_values(
    k_min: Optional[int],
    k_max: Optional[int],
    ks: Optional[List[int]],
    q_prefix: Optional[Path] = None,
) -> List[int]:
    """Resolve list of K values from explicit list, range, or filesystem."""
    if ks:
        return sorted(set(ks))
    if k_min is not None and k_max is not None:
        return list(range(k_min, k_max + 1))
    if q_prefix is not None:
        prefix = Path(q_prefix)
        found = []
        for path in prefix.parent.glob(f"{prefix.name}.*.csv"):
            try:
                k = int(path.stem.split(".")[-1])
                found.append(k)
            except Exception:
                continue
        if found:
            return sorted(set(found))
    raise ValueError(
        "Please provide --ks or both --k-min/--k-max (or ensure <prefix>.<K>.csv files exist to auto-detect)."
    )


def cmd_pca(args):
    """Run PCA command."""
    setup_logging(args.verbose)

    # Resolve fit/project prefixes
    fit_prefix = args.fit_plink or args.input
    project_prefix = args.project_plink or fit_prefix

    if fit_prefix is None:
        raise ValueError("Please provide --input or --fit-plink for PCA fitting.")

    # Determine outputs
    fit_output = args.fit_output
    project_output = args.project_output or args.output

    if project_output is None:
        raise ValueError("Please provide --output or --project-output for PCA projection.")

    model_dir = Path(args.model_dir) if args.model_dir else None
    if args.flashpca_output_dir:
        model_dir = Path(args.flashpca_output_dir)

    pca = PCA(n_components=args.n_pcs, force=args.force)

    if args.project_plink:
        # Fit on one dataset, project another
        pca.fit(fit_prefix, output_dir=model_dir)
        if fit_output:
            pca.project(fit_prefix, output_path=fit_output)
        pca_coords = pca.project(project_prefix, output_path=project_output)
        print(f"PCA fit on {fit_prefix} and projected {project_prefix}")
        if fit_output:
            print(f"Fit PCA coords: {fit_output}")
        print(f"Projected PCA coords: {project_output}")
    else:
        # Fit and project on the same dataset
        pca_coords = pca.fit_transform(fit_prefix, output_path=project_output)
        print(f"PCA complete: {project_output}")

    # Report shape excluding sample_id column
    n_samples = pca_coords.shape[0]
    n_pcs = pca_coords.shape[1] - 1  # Exclude sample_id column
    print(f"Shape: ({n_samples}, {n_pcs}) [excluding sample_id column]")
    return 0


def cmd_admixture(args):
    """Run admixture command."""
    setup_logging(args.verbose)

    fit_prefix = args.fit_plink or args.input
    project_prefix = args.project_plink or fit_prefix

    if fit_prefix is None:
        raise ValueError("Please provide --input or --fit-plink for admixture fitting.")

    # Resolve checkpoint dir and Q output dir
    checkpoint_dir = Path(
        args.neuraladmixture_output_dir or args.output or Path.cwd() / "admixture_outputs"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not args.fit_output or not args.project_output:
        raise ValueError("Please provide --fit-output and --project-output for admixture outputs.")
    fit_output_path = Path(args.fit_output)
    project_output_path = Path(args.project_output)
    fit_output_path.parent.mkdir(parents=True, exist_ok=True)
    project_output_path.parent.mkdir(parents=True, exist_ok=True)

    admix = NeuralAdmixture(
        k_min=args.k_min,
        k_max=args.k_max,
        force=args.force,
        threads=args.threads,
        num_gpus=args.num_gpus,
        batch_size=getattr(args, "neuraladmixture_batch_size", None),
    )

    # Fit models
    admix.fit(fit_prefix, output_dir=checkpoint_dir, model_name=args.model_name)

    # Optional: infer on fit subset
    fit_q_files = admix.transform(
        fit_prefix,
        output_prefix=fit_output_path,
    )

    # Project/infer on project subset
    project_q_files = admix.transform(
        project_prefix,
        output_prefix=project_output_path,
    )

    print(f"Admixture fit on {fit_prefix} and projected {project_prefix}")
    print(f"Fit Q CSVs written to prefix: {fit_output_path}")
    print(f"Projected Q CSVs written to prefix: {project_output_path}")
    return 0


def cmd_plot_admixture(args):
    """Plot stacked admixture barplots across Ks."""
    setup_logging(args.verbose)

    q_prefix = Path(args.q_prefix)
    labels = args.labels
    group_col = args.group_column
    k_values = _resolve_k_values(args.k_min, args.k_max, args.ks, q_prefix=q_prefix)

    # Validate inputs
    validate_admixture_csv(str(q_prefix), k_values)
    validate_labels_csv(labels)
    validate_colormap_json(args.colormap)
    validate_label_column(group_col, labels, args.colormap)

    output_path = (
        Path(args.output) if args.output else q_prefix.parent / f"{q_prefix.name}_admixture.png"
    )

    group_order = None
    if args.colormap:
        cmap = read_colormap(args.colormap)
        # Try to pull order from the matching key (case-insensitive)
        for key in cmap:
            if key.lower() == group_col.lower():
                group_order = list(cmap[key].keys())
                break

    # Convert "none" to None for within_group_order
    within_group_order = None if args.within_group_order == "none" else args.within_group_order

    plot_admixture_bar_grid(
        q_prefix=q_prefix,
        labels=labels,
        group_column=group_col,
        k_values=k_values,
        output_path=output_path,
        subsample_per_group=args.subsample_per_group,
        group_order=group_order,
        colormap=args.colormap,
        within_group_order=within_group_order,
        component_colors_output=getattr(args, "component_colors_output", None),
    )
    print(f"Admixture bar plot written to: {output_path}")
    return 0


def cmd_plot_admixture_embedding(args):
    """Plot embedding colored by admixture components for multiple Ks."""
    setup_logging(args.verbose)

    q_prefix = Path(args.q_prefix)
    embedding = args.embedding
    k_values = _resolve_k_values(args.k_min, args.k_max, args.ks, q_prefix=q_prefix)

    # Validate inputs
    validate_embedding_csv(embedding)
    validate_admixture_csv(str(q_prefix), k_values)
    first_q = f"{q_prefix}.{k_values[0]}.csv"
    validate_sample_id_overlap(embedding, first_q, "embedding", "admixture")
    output_path = (
        Path(args.output)
        if args.output
        else q_prefix.parent / f"{q_prefix.name}_admixture_embedding.png"
    )

    plot_admixture_embedding_grid(
        embedding=embedding,
        q_prefix=q_prefix,
        k_values=k_values,
        output_path=output_path,
        pc_x=args.pc_x,
        pc_y=args.pc_y,
        subsample=args.subsample,
        component_colormap=getattr(args, "component_colormap", None),
    )
    print(f"Admixture-embedding plot written to: {output_path}")
    return 0


def cmd_plot_knn_composition(args):
    """Plot KNN label composition stacked bars for project individuals."""
    setup_logging(args.verbose)

    # Validate inputs
    validate_embedding_csv(args.fit_embedding)
    validate_embedding_csv(args.project_embedding)
    validate_labels_csv(args.fit_labels)
    validate_labels_csv(args.project_labels)
    validate_colormap_json(args.fit_colormap)
    validate_label_column(args.fit_label_column, args.fit_labels, args.fit_colormap)
    validate_column_in_csv(args.project_label_column, args.project_labels)
    validate_sample_id_overlap(
        args.fit_embedding,
        args.fit_labels,
        "fit embedding",
        "fit labels",
    )
    validate_sample_id_overlap(
        args.project_embedding,
        args.project_labels,
        "project embedding",
        "project labels",
    )

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        proj_emb = Path(args.project_embedding)
        output_path = proj_emb.parent / f"{proj_emb.stem}_knn_composition.png"

    result = plot_knn_composition(
        fit_embedding=args.fit_embedding,
        project_embedding=args.project_embedding,
        fit_labels=args.fit_labels,
        project_labels=args.project_labels,
        fit_colormap=args.fit_colormap,
        fit_label_column=args.fit_label_column,
        project_label_column=args.project_label_column,
        output_path=output_path,
        project_label_subset=args.project_label_subset,
        k=args.k,
        sort_by_dominant=args.sort_by_dominant,
        subsample_per_group=args.subsample_per_group,
        project_name=args.project_name,
    )
    print(f"KNN composition plot saved: {result}")
    return 0


def cmd_setup(args):
    """Download external tools (plink2, flashpca, optional plink v1.9)."""
    setup_logging(args.verbose)

    resolver = ToolResolver()
    tools = resolver.install_tools(include_plink1=not args.skip_plink1)

    print("External tools installed:")
    for name, path in tools.items():
        print(f"  - {name}: {path}")
    return 0


def cmd_embed(args):
    """Run embedding command."""
    setup_logging(args.verbose)

    fit_input = args.fit_input or args.input
    project_input = args.project_input or fit_input

    # Validate inputs (after resolving fit/project)
    if fit_input is not None:
        validate_embedding_csv(fit_input)
    if project_input is not None and project_input != fit_input:
        validate_embedding_csv(project_input)

    if fit_input is None:
        raise ValueError("Please provide --input or --fit-input for embedding fit.")
    if args.output is None and args.project_output is None:
        raise ValueError("Please provide --output or --project-output for embedding transform.")

    project_output = args.project_output or args.output
    fit_output = args.fit_output

    # Ensure output directories exist
    if fit_output:
        Path(fit_output).parent.mkdir(parents=True, exist_ok=True)
    if project_output:
        Path(project_output).parent.mkdir(parents=True, exist_ok=True)

    # Parse embedding-specific parameters
    embed_params = {}
    if args.method == "phate":
        # Allow t="auto" or a numeric string; ensure PHATE gets int for numbers
        t_param = args.t
        if isinstance(t_param, str) and t_param != "auto":
            t_param = int(t_param)

        n_landmark = None
        if args.n_landmark is not None:
            if isinstance(args.n_landmark, str):
                if args.n_landmark.lower() != "none":
                    n_landmark = int(args.n_landmark)
            else:
                n_landmark = args.n_landmark

        if args.random_landmarking and n_landmark is None:
            raise ValueError("random-landmarking requires --n-landmark to be set")
        embed_params = {
            "knn": args.knn,
            "t": t_param,
            "n_landmark": n_landmark,
            "random_landmarking": args.random_landmarking,
            "embed_batch_size": getattr(args, "embed_batch_size", None),
        }
        model = PHATE(n_components=2, **embed_params)
    elif args.method == "umap":
        embed_params = {"n_neighbors": args.n_neighbors, "min_dist": args.min_dist}
        model = UMAP(n_components=2, **embed_params)
    elif args.method == "tsne":
        embed_params = {"perplexity": args.perplexity}
        model = TSNE(n_components=2, **embed_params)
    elif args.method == "diffusion_map":
        embed_params = {"knn": args.knn}
        model = DiffusionMap(n_components=2, **embed_params)
    else:
        print(f"Unknown method: {args.method}")
        return 1

    # Fit on fit_input
    model.fit(fit_input)
    if fit_output:
        model.transform(fit_input).to_csv(fit_output, index=False)

    # Transform project_input
    embedding = model.transform(project_input)
    embedding.to_csv(project_output, index=False)

    print(f"Embedding fit on {fit_input} and projected {project_input}")
    if fit_output:
        print(f"Fit embedding saved to: {fit_output}")
    print(f"Projected embedding saved to: {project_output}")
    # Report shape excluding sample_id column
    n_samples = embedding.shape[0]
    n_dims = embedding.shape[1] - 1  # Exclude sample_id column
    print(f"Shape: ({n_samples}, {n_dims}) [excluding sample_id column]")
    return 0


def cmd_metrics_geographic(args):
    """Compute geographic preservation metrics."""
    setup_logging(args.verbose)

    # Validate inputs
    validate_embedding_csv(args.embedding)
    validate_geographic_csv(args.geographic, args.longitude_col, args.latitude_col)
    validate_sample_id_overlap(
        args.embedding, args.geographic, "embedding", "geographic coordinates"
    )

    result = compute_geographic_preservation(
        embedding=args.embedding,
        geographic_coords=args.geographic,
        longitude_col=args.longitude_col,
        latitude_col=args.latitude_col,
        num_samples=args.num_dists_sampled,
        ignore_missing=not args.keep_missing,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Geographic metrics saved to: {output_path}")
    return 0


def cmd_metrics_admixture(args):
    """Compute admixture preservation metrics."""
    setup_logging(args.verbose)

    admixture_prefix = Path(args.admixture_output)
    k_values = list(range(args.k_min, args.k_max + 1))

    # Validate inputs
    validate_embedding_csv(args.embedding)
    validate_admixture_csv(str(admixture_prefix), k_values)
    first_q = f"{admixture_prefix}.{args.k_min}.csv"
    validate_sample_id_overlap(args.embedding, first_q, "embedding", "admixture")

    # Build Q files dict from prefix: {k: path/to/prefix.k.csv}
    q_files = {}
    for k in k_values:
        q_files[k] = Path(f"{admixture_prefix}.{k}.csv")

    if not q_files:
        raise ValueError(f"No admixture files found for K={args.k_min} to {args.k_max}")

    metrics = compute_admixture_preservation(
        embedding=args.embedding,
        q_files=q_files,
        k_value=args.k_value,
        num_samples=args.num_dists_sampled,
        subsample=args.subsample,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Admixture metrics saved to: {output_path}")
    return 0


def cmd_plot(args):
    """Run visualization command."""
    setup_logging(args.verbose)

    # Validate inputs
    validate_embedding_csv(args.input)
    validate_labels_csv(args.labels)
    validate_colormap_json(args.colormap)
    validate_labels_colormap_match(args.labels, args.colormap)
    validate_sample_id_overlap(args.input, args.labels, "embedding", "labels")

    output_dir = Path(args.output).parent if args.output else Path.cwd()
    output_prefix = Path(args.output).stem if args.output else "embedding"

    figure_paths = visualize(
        embedding=args.input,
        labels=args.labels,
        colormap=args.colormap,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )

    print(f"Visualization complete:")
    for path in figure_paths:
        print(f"  {path}")
    return 0


def cmd_plot_pca(args):
    """Run PCA visualization command."""
    setup_logging(args.verbose)

    # Validate inputs
    validate_embedding_csv(args.input)
    validate_labels_csv(args.labels)
    validate_colormap_json(args.colormap)
    validate_labels_colormap_match(args.labels, args.colormap)
    validate_sample_id_overlap(args.input, args.labels, "PCA coordinates", "labels")

    output_dir = Path(args.output) if args.output else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load colormap to get label columns
    with open(args.colormap) as f:
        colormap_dict = json.load(f)

    figure_paths = []
    for label_col in colormap_dict.keys():
        output_path = output_dir / f"pca_pairs_by_{label_col}.png"
        plot_path = plot_pca_pairs(
            pca_coords=args.input,
            labels=args.labels,
            colormap=colormap_dict,
            output_path=output_path,
            label_column=label_col,
            n_pcs=args.n_pcs,
            title=f"PCA Pairs by {label_col}",
        )
        figure_paths.append(plot_path)

    print(f"PCA visualization complete:")
    for path in figure_paths:
        print(f"  {path}")
    return 0


def cmd_plot_projection(args):
    """Run projection plot command."""
    setup_logging(args.verbose)

    # Validate inputs
    validate_embedding_csv(args.fit_embedding)
    validate_embedding_csv(args.project_embedding)
    validate_labels_csv(args.fit_labels)
    validate_labels_csv(args.project_labels)
    validate_colormap_json(args.fit_colormap)
    validate_colormap_json(args.project_colormap)
    validate_label_column(args.fit_column, args.fit_labels, args.fit_colormap)
    validate_label_column(args.project_column, args.project_labels, args.project_colormap)
    validate_sample_id_overlap(args.fit_embedding, args.fit_labels, "fit embedding", "fit labels")
    validate_sample_id_overlap(
        args.project_embedding, args.project_labels, "project embedding", "project labels"
    )

    figure_path = plot_projection(
        fit_embedding=args.fit_embedding,
        project_embedding=args.project_embedding,
        fit_labels=args.fit_labels,
        project_labels=args.project_labels,
        fit_colormap=args.fit_colormap,
        project_colormap=args.project_colormap,
        output_path=args.output,
        fit_label_column=args.fit_column,
        project_label_column=args.project_column,
        point_size=args.point_size,
        alpha=args.alpha,
        linewidth=args.linewidth,
    )

    print(f"Projection plot saved: {figure_path}")
    return 0


def cmd_pipeline(args):
    """Run full pipeline command."""
    setup_logging(args.verbose)

    # Validate inputs that are provided
    if args.labels:
        validate_labels_csv(args.labels)
    if args.colormap:
        validate_colormap_json(args.colormap)
    if args.labels and args.colormap:
        validate_labels_colormap_match(args.labels, args.colormap)

    fit_labels = getattr(args, "fit_labels", None)
    project_labels = getattr(args, "project_labels", None)
    fit_colormap = getattr(args, "fit_colormap", None)
    project_colormap = getattr(args, "project_colormap", None)

    if fit_labels:
        validate_labels_csv(fit_labels)
    if project_labels:
        validate_labels_csv(project_labels)
    if fit_colormap:
        validate_colormap_json(fit_colormap)
    if project_colormap:
        validate_colormap_json(project_colormap)
    if fit_labels and fit_colormap:
        validate_labels_colormap_match(fit_labels, fit_colormap)
    if project_labels and project_colormap:
        validate_labels_colormap_match(project_labels, project_colormap)

    # Validate specific columns when provided
    admix_group_col = getattr(args, "admixture_group_column", None)
    proj_fit_col = getattr(args, "projection_plot_fit_column", None)
    proj_project_col = getattr(args, "projection_plot_project_column", None)

    if admix_group_col:
        # admixture group column uses shared labels/colormap or fit variants
        admix_labels = fit_labels or args.labels
        admix_cmap = fit_colormap or args.colormap
        if admix_labels and admix_cmap:
            validate_label_column(admix_group_col, admix_labels, admix_cmap)
    if proj_fit_col and (fit_labels or args.labels) and (fit_colormap or args.colormap):
        validate_label_column(
            proj_fit_col, fit_labels or args.labels, fit_colormap or args.colormap
        )
    if proj_project_col and (project_labels or args.labels) and (project_colormap or args.colormap):
        validate_label_column(
            proj_project_col,
            project_labels or args.labels,
            project_colormap or args.colormap,
        )

    if hasattr(args, "geographic") and args.geographic:
        validate_geographic_csv(args.geographic)

    # Parse embedding parameters
    embedding_params = {}
    if args.embedding == "phate":
        embedding_params["knn"] = args.knn
        # t can be "auto" or int-like
        t_param = args.t
        if isinstance(t_param, str) and t_param != "auto":
            t_param = int(t_param)
        embedding_params["t"] = t_param

        n_landmark = None
        if args.n_landmark is not None:
            if isinstance(args.n_landmark, str):
                if args.n_landmark.lower() != "none":
                    n_landmark = int(args.n_landmark)
            else:
                n_landmark = args.n_landmark
        if args.random_landmarking and n_landmark is None:
            raise ValueError("random-landmarking requires --n-landmark to be set")
        embedding_params["n_landmark"] = n_landmark
        embedding_params["random_landmarking"] = args.random_landmarking
        embedding_params["embed_batch_size"] = getattr(args, "embed_batch_size", None)
    elif args.embedding == "umap":
        embedding_params["n_neighbors"] = args.n_neighbors
    elif args.embedding == "tsne":
        embedding_params["perplexity"] = args.perplexity
    elif args.embedding == "diffusion_map":
        embedding_params["knn"] = args.knn

    # Handle separate labels/colormaps for cross-cohort analysis
    fit_labels = args.fit_labels if hasattr(args, "fit_labels") and args.fit_labels else None
    project_labels = (
        args.project_labels if hasattr(args, "project_labels") and args.project_labels else None
    )
    fit_colormap = (
        args.fit_colormap if hasattr(args, "fit_colormap") and args.fit_colormap else None
    )
    project_colormap = (
        args.project_colormap
        if hasattr(args, "project_colormap") and args.project_colormap
        else None
    )

    # Handle projection column args
    projection_plot_fit_column = getattr(args, "projection_plot_fit_column", None)
    projection_plot_project_column = getattr(args, "projection_plot_project_column", None)

    # Use the canonical run_pipeline function
    results = run_pipeline(
        fit_plink=args.fit_plink,
        project_plink=args.project_plink,
        output_dir=args.output,
        labels=args.labels,
        colormap=args.colormap,
        fit_labels=fit_labels,
        project_labels=project_labels,
        fit_colormap=fit_colormap,
        project_colormap=project_colormap,
        geographic_coords=args.geographic if hasattr(args, "geographic") else None,
        n_pcs=args.n_pcs,
        flashpca_output_dir=args.flashpca_output_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        admix_threads=args.threads,
        admix_gpus=args.num_gpus,
        admix_batch_size=getattr(args, "neuraladmixture_batch_size", None),
        embedding=args.embedding,
        embedding_params=embedding_params,
        embedding_input=args.embedding_input,
        admix_group_column=args.admixture_group_column,
        admix_within_group_order=(
            None
            if args.admixture_within_group_order == "none"
            else args.admixture_within_group_order
        ),
        projection_plot_fit_column=projection_plot_fit_column,
        projection_plot_project_column=projection_plot_project_column,
        skip_pca=args.skip_pca,
        skip_admixture=args.skip_admixture,
        skip_embedding=args.skip_embedding,
        skip_visualization=args.skip_embedding_visualization,
        skip_pca_visualization=args.skip_pca_visualization,
        skip_admixture_visualization=args.skip_admixture_visualization,
        skip_metrics=args.skip_metrics,
    )

    print(f"Pipeline complete!")
    print(f"Output directory: {args.output}")

    if "metrics" in results:
        print("\nMetrics:")
        if "geographic" in results["metrics"]:
            geo = results["metrics"]["geographic"]
            print(f"  Geographic preservation: {geo['correlation']:.4f} (p={geo['p_value']:.2e})")
        if "admixture" in results["metrics"]:
            print("  Admixture preservation:")
            for k, metrics in results["metrics"]["admixture"].items():
                print(f"    K={k}: {metrics['correlation']:.4f}")

    return 0


def main(argv: Optional[List[str]] = None):
    """Main CLI entry point.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]`` when None.
    """
    parser = argparse.ArgumentParser(
        prog="manifold-genetics",
        description=(
            "Genetic analysis with PCA, Neural Admixture, and manifold learning.\n\n"
            "Run 'manifold-genetics --help' for this list, or\n"
            "'manifold-genetics <subcommand> --help' (or with '-h' instead of '--help') for\n"
            "subcommand-specific options and usage examples."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # PCA command
    pca_parser = subparsers.add_parser(
        "pca",
        help="Run PCA",
        description=(
            "Run FlashPCA on a PLINK dataset.\n\n"
            "Fits a PCA model on the fit set and projects it onto the project set\n"
            "(or the same set when only --input is given). Outputs standardised CSV\n"
            "files with columns: sample_id, dim_1, dim_2, ..., dim_N."
        ),
        epilog=(
            "Examples:\n"
            "  # Fit and project on the same dataset\n"
            "  manifold-genetics pca --input data/all --output results/pca/pca_50.csv\n\n"
            "  # Fit on one set, project another\n"
            "  manifold-genetics pca \\\n"
            "      --fit-plink data/fit --project-plink data/project \\\n"
            "      --fit-output results/pca/fit_pca_50.csv \\\n"
            "      --project-output results/pca/project_pca_50.csv \\\n"
            "      --flashpca-output-dir results/pca/flashpca_outputs \\\n"
            "      --n-pcs 50"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pca_parser.add_argument(
        "--input", help="Input PLINK prefix — fits and projects on the same dataset"
    )
    pca_parser.add_argument("--fit-plink", help="PLINK prefix to fit PCA (defaults to --input)")
    pca_parser.add_argument(
        "--project-plink",
        help="PLINK prefix to project using the fitted PCA (defaults to fit prefix)",
    )
    pca_parser.add_argument(
        "--fit-output",
        help="CSV path for PCA coordinates of the fit subset (optional)",
    )
    pca_parser.add_argument(
        "--project-output",
        help="CSV path for projected subset (defaults to --output)",
    )
    pca_parser.add_argument(
        "--flashpca-output-dir",
        help="Directory for flashpca intermediate outputs (defaults to ./pca_outputs)",
    )
    pca_parser.add_argument("--output", help="Output CSV file (projected subset)")
    pca_parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCs")
    pca_parser.add_argument("--force", action="store_true", help="Force recomputation")
    pca_parser.add_argument(
        "--model-dir",
        help="Directory for flashpca intermediate outputs (default: ./pca_outputs)",
    )
    pca_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    pca_parser.set_defaults(func=cmd_pca)

    # Admixture command
    admix_parser = subparsers.add_parser(
        "admixture",
        help="Run neural admixture",
        description=(
            "Run Neural Admixture for K in [k-min, k-max].\n\n"
            "Fits models on the fit set and infers ancestry proportions (Q matrices) for\n"
            "both fit and project sets. Outputs one CSV per K with columns:\n"
            "sample_id, component_1, ..., component_K (values sum to 1.0)."
        ),
        epilog=(
            "Examples:\n"
            "  manifold-genetics admixture \\\n"
            "      --fit-plink data/fit --project-plink data/project \\\n"
            "      --neuraladmixture-output-dir results/admixture/checkpoints \\\n"
            "      --fit-output results/admixture/fit \\\n"
            "      --project-output results/admixture/project \\\n"
            "      --k-min 2 --k-max 10 --threads 8\n\n"
            "  # With GPU and large-dataset batch size\n"
            "  manifold-genetics admixture ... --num-gpus 1 --neuraladmixture-batch-size 400"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    admix_parser.add_argument("--input", help="Input PLINK prefix (fit + project)")
    admix_parser.add_argument(
        "--fit-plink", help="PLINK prefix to fit admixture (defaults to --input)"
    )
    admix_parser.add_argument(
        "--project-plink", help="PLINK prefix to project admixture (defaults to fit prefix)"
    )
    admix_parser.add_argument(
        "--output",
        help="Output directory (deprecated; use --neuraladmixture-output-dir and explicit --fit-output/--project-output)",
    )
    admix_parser.add_argument(
        "--model-name", default="fit", help="Model name prefix (default: fit)"
    )
    admix_parser.add_argument(
        "--fit-output", help="Prefix for fit admixture CSV outputs (required)"
    )
    admix_parser.add_argument(
        "--project-output", help="Prefix for projected admixture CSV outputs (required)"
    )
    admix_parser.add_argument(
        "--neuraladmixture-output-dir",
        help="Directory for neural admixture checkpoints/outputs (defaults to --output or ./admixture_outputs)",
    )
    admix_parser.add_argument("--k-min", type=int, default=2, help="Minimum K")
    admix_parser.add_argument("--k-max", type=int, default=10, help="Maximum K")
    admix_parser.add_argument("--force", action="store_true", help="Force retraining")
    admix_parser.add_argument("--threads", type=int, help="Threads for neural admixture")
    admix_parser.add_argument(
        "--num-gpus", type=int, help="GPUs for neural admixture (default: auto)"
    )
    admix_parser.add_argument(
        "--neuraladmixture-batch-size",
        type=int,
        help="Batch size for training and inference (helps avoid OOM on large datasets)",
    )
    admix_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    admix_parser.set_defaults(func=cmd_admixture)

    # Admixture visualization: stacked barplots
    plot_admix_parser = subparsers.add_parser(
        "plot-admixture",
        help="Plot stacked admixture barplots for Ks",
        description=(
            "Plot a grid of stacked ancestry-proportion bar charts, one panel per K.\n\n"
            "Samples are grouped by --group-column and ordered within groups by\n"
            "--within-group-order. Group order comes from the colormap JSON.\n"
            "Use --component-colors-output to save the component-colour assignments so\n"
            "that plot-admixture-embedding can use matching colours."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plot_admix_parser.add_argument(
        "--q-prefix", required=True, help="Prefix for admixture CSVs (<prefix>.<K>.csv)"
    )
    plot_admix_parser.add_argument(
        "--labels", required=True, help="Labels CSV with sample_id and grouping column"
    )
    plot_admix_parser.add_argument(
        "--group-column",
        required=True,
        help="Column in labels used to order bars (e.g., Population)",
    )
    plot_admix_parser.add_argument(
        "--colormap",
        required=True,
        help="Colormap JSON (used to derive group ordering for the chosen column)",
    )
    plot_admix_parser.add_argument("--k-min", type=int, help="Minimum K (if Ks not provided)")
    plot_admix_parser.add_argument("--k-max", type=int, help="Maximum K (if Ks not provided)")
    plot_admix_parser.add_argument(
        "--ks", type=int, nargs="+", help="Explicit K values (overrides k-min/k-max)"
    )
    plot_admix_parser.add_argument(
        "--subsample-per-group", type=int, help="Subsample each group to N samples (optional)"
    )
    plot_admix_parser.add_argument(
        "--within-group-order",
        choices=["chron", "tree", "none"],
        default="chron",
        help="Method for ordering samples within groups: 'chron' (sort by components), 'tree' (hierarchical clustering), 'none' (original order)",
    )
    plot_admix_parser.add_argument(
        "--output", help="Output PNG path (default: <prefix>_admixture.png)"
    )
    plot_admix_parser.add_argument(
        "--component-colors-output",
        help="Optional path to save component colors JSON (for use with plot-admixture-embedding).",
    )
    plot_admix_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_admix_parser.set_defaults(func=cmd_plot_admixture)

    # Admixture visualization: embedding colored by admixture components
    plot_admix_emb_parser = subparsers.add_parser(
        "plot-admixture-embedding",
        help="Plot embedding colored by admixture components",
        description=(
            "Plot a grid of 2-D embedding scatter plots coloured by admixture component\n"
            "proportion, one subplot per (K, component) combination.\n\n"
            "Pass --component-colormap (exported by plot-admixture) to use the same\n"
            "white-to-component-colour gradients as the bar chart."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plot_admix_emb_parser.add_argument(
        "--embedding", required=True, help="Embedding CSV (sample_id, dim_1, dim_2, ...)"
    )
    plot_admix_emb_parser.add_argument(
        "--q-prefix", required=True, help="Prefix for admixture CSVs (<prefix>.<K>.csv)"
    )
    plot_admix_emb_parser.add_argument("--k-min", type=int, help="Minimum K (if Ks not provided)")
    plot_admix_emb_parser.add_argument("--k-max", type=int, help="Maximum K (if Ks not provided)")
    plot_admix_emb_parser.add_argument(
        "--ks", type=int, nargs="+", help="Explicit K values (overrides k-min/k-max)"
    )
    plot_admix_emb_parser.add_argument(
        "--pc-x", type=int, default=1, help="PC for x-axis (default: 1)"
    )
    plot_admix_emb_parser.add_argument(
        "--pc-y", type=int, default=2, help="PC for y-axis (default: 2)"
    )
    plot_admix_emb_parser.add_argument(
        "--subsample", type=int, help="Optional subsample size for plotting"
    )
    plot_admix_emb_parser.add_argument(
        "--output", help="Output PNG path (default: <prefix>_admixture_embedding.png)"
    )
    plot_admix_emb_parser.add_argument(
        "--component-colormap",
        help="Path to component colors JSON exported by plot-admixture. When provided, each "
        "component subplot uses a white-to-component-color gradient matching the bar chart.",
    )
    plot_admix_emb_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_admix_emb_parser.set_defaults(func=cmd_plot_admixture_embedding)

    # KNN composition plot
    plot_knn_parser = subparsers.add_parser(
        "plot-knn-composition",
        help="Plot KNN label composition for project individuals (stacked bars)",
        description=(
            "For each project individual, find its K nearest neighbours in the fit embedding\n"
            "and plot the composition of their fine-grained labels as a stacked bar.\n\n"
            "Useful for assessing how well-characterised project samples are by the reference\n"
            "panel (e.g. projecting a biobank onto an HGDP reference).\n"
            "Panels are grouped by --project-label-column; use --project-label-subset to\n"
            "restrict to specific groups."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plot_knn_parser.add_argument("--fit-embedding", required=True, help="Fit embedding CSV")
    plot_knn_parser.add_argument("--project-embedding", required=True, help="Project embedding CSV")
    plot_knn_parser.add_argument("--fit-labels", required=True, help="Fit labels CSV")
    plot_knn_parser.add_argument(
        "--fit-label-column",
        required=True,
        help="Fine-grained label column in fit labels (e.g. Population)",
    )
    plot_knn_parser.add_argument("--project-labels", required=True, help="Project labels CSV")
    plot_knn_parser.add_argument(
        "--project-label-column",
        required=True,
        help="Coarse label column in project labels (e.g. self_described_ancestry)",
    )
    plot_knn_parser.add_argument(
        "--fit-colormap",
        required=True,
        help="Colormap JSON (must contain fit-label-column key)",
    )
    plot_knn_parser.add_argument(
        "--k", type=int, default=10, help="Number of nearest neighbors (default: 10)"
    )
    plot_knn_parser.add_argument(
        "--project-label-subset",
        nargs="+",
        help="Subset of project labels to include as panels (default: all)",
    )
    plot_knn_parser.add_argument(
        "--project-name",
        default="Project",
        help="Name for project dataset used in panel titles (default: Project)",
    )
    plot_knn_parser.add_argument(
        "--no-sort-by-dominant",
        dest="sort_by_dominant",
        action="store_false",
        help="Disable sorting bars by dominant label within each panel",
    )
    plot_knn_parser.set_defaults(sort_by_dominant=True)
    plot_knn_parser.add_argument(
        "--subsample-per-group",
        type=int,
        help="Randomly subsample each panel to this many individuals (optional)",
    )
    plot_knn_parser.add_argument(
        "--output", help="Output PNG path (default: <project_embedding_stem>_knn_composition.png)"
    )
    plot_knn_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_knn_parser.set_defaults(func=cmd_plot_knn_composition)

    # Embedding command
    embed_parser = subparsers.add_parser(
        "embed",
        help="Run manifold embedding",
        description=(
            "Fit a 2-D manifold embedding on a PCA coordinate CSV and project samples into it.\n\n"
            "Supported methods:\n"
            "  phate        -- PHATE (recommended for population structure).\n"
            "                  Key params: --knn, --t, --n-landmark, --random-landmarking\n"
            "  umap         -- UMAP. Key params: --n-neighbors, --min-dist\n"
            "  tsne         -- t-SNE. Key params: --perplexity\n"
            "  diffusion_map -- Diffusion Maps. Key params: --knn\n\n"
            "Input CSVs must have columns: sample_id, dim_1, dim_2, ... (standard pipeline output)."
        ),
        epilog=(
            "Examples:\n"
            "  # PHATE: fit on fit set, project the project set\n"
            "  manifold-genetics embed \\\n"
            "      --fit-input results/pca/fit_pca_50.csv \\\n"
            "      --project-input results/pca/project_pca_50.csv \\\n"
            "      --project-output results/embeddings/phate_2d.csv \\\n"
            "      --method phate --knn 100 --t 3\n\n"
            "  # PHATE with landmarking for large datasets (>10K samples)\n"
            "  manifold-genetics embed ... --n-landmark 10000 --random-landmarking\n\n"
            "  # UMAP\n"
            "  manifold-genetics embed \\\n"
            "      --fit-input results/pca/project_pca_50.csv \\\n"
            "      --project-output results/embeddings/umap_2d.csv \\\n"
            "      --method umap --n-neighbors 15 --min-dist 0.1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    embed_parser.add_argument("--input", help="Input CSV file (fit + project)")
    embed_parser.add_argument(
        "--fit-input", help="Input CSV to fit embedding (defaults to --input)"
    )
    embed_parser.add_argument(
        "--project-input", help="Input CSV to project embedding (defaults to fit input)"
    )
    embed_parser.add_argument("--fit-output", help="Output CSV for fit embedding (optional)")
    embed_parser.add_argument(
        "--project-output", help="Output CSV for projected embedding (defaults to --output)"
    )
    embed_parser.add_argument("--output", help="Output CSV for projected embedding")
    embed_parser.add_argument(
        "--method",
        choices=["phate", "umap", "tsne", "diffusion_map"],
        default="phate",
        help="Embedding method (default: phate). See description above for method-specific params.",
    )
    embed_parser.add_argument(
        "--knn", type=int, default=25, help="K nearest neighbors (PHATE/diffusion_map; default: 25)"
    )
    embed_parser.add_argument(
        "--t",
        default="auto",
        help="Diffusion time for PHATE: integer or 'auto' (PHATE chooses automatically; default: auto)",
    )
    embed_parser.add_argument(
        "--n-landmark",
        type=str,
        help='Number of landmarks for PHATE (use an integer or "None"/omit for all samples)',
    )
    embed_parser.add_argument(
        "--random-landmarking",
        action="store_true",
        help="Use random landmarking for PHATE (requires --n-landmark)",
    )
    embed_parser.add_argument("--n-neighbors", type=int, default=15, help="Neighbors (UMAP)")
    embed_parser.add_argument("--min-dist", type=float, default=0.1, help="Min dist (UMAP)")
    embed_parser.add_argument("--perplexity", type=float, default=30, help="Perplexity (t-SNE)")
    embed_parser.add_argument(
        "--embed-batch-size",
        type=int,
        help="Batch size for embedding transform (to avoid OOM on large datasets; None=no batching)",
    )
    embed_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    embed_parser.set_defaults(func=cmd_embed)

    # Plot command
    plot_parser = subparsers.add_parser(
        "plot",
        help="Visualize embeddings",
        description=(
            "Generate publication-ready scatter plots of a 2-D embedding CSV.\n\n"
            "One figure is produced per label column found in the colormap JSON.\n"
            "Input CSV must have columns: sample_id, dim_1, dim_2."
        ),
        epilog=(
            "Example:\n"
            "  manifold-genetics plot \\\n"
            "      --input results/embeddings/phate_2d.csv \\\n"
            "      --labels data/labels.csv \\\n"
            "      --colormap data/colormap.json \\\n"
            "      --output results/figures/phate.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plot_parser.add_argument("--input", required=True, help="Input embedding CSV")
    plot_parser.add_argument("--labels", required=True, help="Labels CSV file")
    plot_parser.add_argument("--colormap", required=True, help="Colormap JSON file")
    plot_parser.add_argument("--output", help="Output figure path")
    plot_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_parser.set_defaults(func=cmd_plot)

    # Plot PCA command
    plot_pca_parser = subparsers.add_parser(
        "plot-pca",
        help="Visualize PCA coordinates",
        description=(
            "Generate pair-plot grids of PCA coordinates, one figure per label column in\n"
            "the colormap JSON. Plots the first --n-pcs components in pairwise scatter grids."
        ),
    )
    plot_pca_parser.add_argument("--input", required=True, help="Input PCA CSV file")
    plot_pca_parser.add_argument("--labels", required=True, help="Labels CSV file")
    plot_pca_parser.add_argument("--colormap", required=True, help="Colormap JSON file")
    plot_pca_parser.add_argument("--output", help="Output directory for PCA figures")
    plot_pca_parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCs to plot")
    plot_pca_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_pca_parser.set_defaults(func=cmd_plot_pca)

    # Plot: projection (fit + project together)
    plot_proj_parser = subparsers.add_parser(
        "plot-projection",
        help="Plot fit and projection embeddings together",
        description=(
            "Overlay the fit and project embedding sets in a single scatter plot.\n\n"
            "Fit samples are drawn as filled circles; project samples as hollow markers.\n"
            "Each set is coloured by its own label column (--fit-column / --project-column)\n"
            "using separate colormaps, enabling cross-cohort comparison."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plot_proj_parser.add_argument("--fit-embedding", required=True, help="Fit embedding CSV file")
    plot_proj_parser.add_argument(
        "--project-embedding", required=True, help="Project embedding CSV file"
    )
    plot_proj_parser.add_argument("--fit-labels", required=True, help="Fit labels CSV file")
    plot_proj_parser.add_argument("--project-labels", required=True, help="Project labels CSV file")
    plot_proj_parser.add_argument("--fit-colormap", required=True, help="Fit colormap JSON file")
    plot_proj_parser.add_argument(
        "--project-colormap", required=True, help="Project colormap JSON file"
    )
    plot_proj_parser.add_argument(
        "--fit-column", required=True, help="Column from fit colormap to use for coloring"
    )
    plot_proj_parser.add_argument(
        "--project-column", required=True, help="Column from project colormap to use for coloring"
    )
    plot_proj_parser.add_argument("--output", required=True, help="Output figure path")
    plot_proj_parser.add_argument(
        "--point-size", type=float, default=4.0, help="Size of scatter points"
    )
    plot_proj_parser.add_argument("--alpha", type=float, default=0.6, help="Transparency of points")
    plot_proj_parser.add_argument(
        "--linewidth", type=float, default=0.8, help="Edge width for hollow markers"
    )
    plot_proj_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_proj_parser.set_defaults(func=cmd_plot_projection)

    # Setup command (download external tools)
    setup_parser = subparsers.add_parser(
        "setup",
        help="Download external tools (plink2, flashpca, optional plink v1.9)",
        description=(
            "Download the external command-line tools required by the pipeline.\n\n"
            "Tools are placed in the bin/ directory under the project root:\n"
            "  bin/plink2   (~20 MB)\n"
            "  bin/flashpca (~2 MB)\n"
            "  bin/plink    (~2 MB, plink v1.9 — skip with --skip-plink1)\n\n"
            "Requires internet access (run on a login node, not a compute node).\n"
            "This command does NOT manage the Python environment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    setup_parser.add_argument(
        "--skip-plink1",
        action="store_true",
        help="Skip downloading PLINK v1.9",
    )
    setup_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    setup_parser.set_defaults(func=cmd_setup)

    # Metrics: geographic
    geo_metrics_parser = subparsers.add_parser(
        "metrics-geographic",
        help="Compute geographic preservation",
        description=(
            "Compute a Spearman correlation between pairwise geographic distances and pairwise\n"
            "embedding distances across randomly sampled pairs of samples.\n\n"
            "Outputs a JSON file with keys: correlation, p_value, n_pairs."
        ),
    )
    geo_metrics_parser.add_argument("--embedding", required=True, help="Embedding CSV")
    geo_metrics_parser.add_argument(
        "--geographic", required=True, help="Geographic coordinates CSV"
    )
    geo_metrics_parser.add_argument("--output", required=True, help="Output JSON path")
    geo_metrics_parser.add_argument(
        "--longitude-col", default="longitude", help="Longitude column name"
    )
    geo_metrics_parser.add_argument(
        "--latitude-col", default="latitude", help="Latitude column name"
    )
    geo_metrics_parser.add_argument(
        "--num-dists-sampled", type=int, default=50000, help="Max pairwise distances to sample"
    )
    geo_metrics_parser.add_argument(
        "--keep-missing", action="store_true", help="Keep samples with missing coordinates"
    )
    geo_metrics_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    geo_metrics_parser.set_defaults(func=cmd_metrics_geographic)

    # Metrics: admixture
    admix_metrics_parser = subparsers.add_parser(
        "metrics-admixture",
        help="Compute admixture preservation",
        description=(
            "Compute a Spearman correlation between pairwise admixture distances and pairwise\n"
            "embedding distances across randomly sampled pairs of samples, for each K.\n\n"
            "Admixture distance is the L1 distance between Q vectors.\n"
            "Outputs a JSON file keyed by K."
        ),
    )
    admix_metrics_parser.add_argument("--embedding", required=True, help="Embedding CSV")
    admix_metrics_parser.add_argument(
        "--admixture-output",
        required=True,
        help="Prefix for admixture CSVs (<prefix>.K.csv, e.g., path/to/project)",
    )
    admix_metrics_parser.add_argument("--output", required=True, help="Output JSON path")
    admix_metrics_parser.add_argument(
        "--k-min", type=int, required=True, help="Minimum K to include"
    )
    admix_metrics_parser.add_argument(
        "--k-max", type=int, required=True, help="Maximum K to include"
    )
    admix_metrics_parser.add_argument("--k-value", type=int, help="Compute only a single K")
    admix_metrics_parser.add_argument(
        "--num-dists-sampled", type=int, default=50000, help="Max pairwise distances to sample"
    )
    admix_metrics_parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample individuals to this count before computing distances (applied consistently across all K values). Useful for large datasets.",
    )
    admix_metrics_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    admix_metrics_parser.set_defaults(func=cmd_metrics_admixture)

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline (recommended entry point)",
        description=(
            "Run the full manifold-genetics pipeline end-to-end:\n"
            "  PCA → Neural Admixture → Embedding → Visualization → Metrics\n\n"
            "Steps can be skipped with --skip-pca, --skip-admixture, --skip-embedding,\n"
            "--skip-metrics. Visualization sub-steps can be skipped individually.\n\n"
            "Outputs are written under --output in subfolders:\n"
            "  pca/          PCA coordinate CSVs\n"
            "  admixture/    Q-matrix CSVs and model checkpoints\n"
            "  embeddings/   2-D embedding CSVs\n"
            "  figures/      All visualisation plots (PNG)\n"
            "  metrics/      Preservation metric JSONs\n\n"
            "For cross-cohort analysis (e.g. HGDP reference → biobank projection), pass\n"
            "separate --fit-labels/--project-labels and --fit-colormap/--project-colormap."
        ),
        epilog=(
            "Examples:\n"
            "  # Same-cohort analysis\n"
            "  manifold-genetics pipeline \\\n"
            "      --fit-plink data/fit --project-plink data/project \\\n"
            "      --labels data/labels.csv --colormap data/colormap.json \\\n"
            "      --output results/ \\\n"
            "      --n-pcs 50 --k-min 2 --k-max 10 \\\n"
            "      --embedding phate --knn 100 --t 3 \\\n"
            "      --threads 8\n\n"
            "  # Large dataset: use landmarking and GPU\n"
            "  manifold-genetics pipeline ... \\\n"
            "      --n-landmark 10000 --random-landmarking \\\n"
            "      --neuraladmixture-batch-size 400 --num-gpus 1\n\n"
            "  # Cross-cohort (separate fit/project labels and colormaps)\n"
            "  manifold-genetics pipeline \\\n"
            "      --fit-plink data/hgdp --project-plink data/ukbb \\\n"
            "      --fit-labels data/hgdp_labels.csv --fit-colormap data/hgdp_colors.json \\\n"
            "      --project-labels data/ukbb_labels.csv --project-colormap data/ukbb_colors.json \\\n"
            "      --output results/ --embedding-input both"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pipeline_parser.add_argument(
        "--fit-plink",
        required=True,
        help="PLINK prefix for PCA/admixture reference set (fit subset)",
    )
    pipeline_parser.add_argument(
        "--project-plink",
        required=True,
        help="PLINK prefix to project/apply the fitted models (project subset)",
    )
    pipeline_parser.add_argument(
        "--labels",
        help="Labels CSV file (used for both fit and project if --fit-labels/--project-labels not provided)",
    )
    pipeline_parser.add_argument(
        "--colormap",
        help="Colormap JSON file (used for both fit and project if --fit-colormap/--project-colormap not provided)",
    )
    pipeline_parser.add_argument(
        "--fit-labels", help="Labels CSV file for fit dataset (for separate fit/project labels)"
    )
    pipeline_parser.add_argument(
        "--project-labels",
        help="Labels CSV file for project dataset (for separate fit/project labels)",
    )
    pipeline_parser.add_argument(
        "--fit-colormap",
        help="Colormap JSON file for fit dataset (optional, for cross-cohort analysis)",
    )
    pipeline_parser.add_argument(
        "--project-colormap",
        help="Colormap JSON file for project dataset (optional, for cross-cohort analysis)",
    )
    pipeline_parser.add_argument("--output", required=True, help="Output directory")
    pipeline_parser.add_argument("--geographic", help="Geographic coordinates CSV")
    pipeline_parser.add_argument(
        "--flashpca-output-dir",
        help="Directory for flashpca intermediate outputs (defaults to ./pca_outputs)",
    )
    pipeline_parser.add_argument(
        "--threads",
        type=int,
        help="Threads for neural admixture (defaults to SLURM/affinity detection)",
    )
    pipeline_parser.add_argument(
        "--num-gpus",
        type=int,
        help="GPUs for neural admixture (default: auto-detect if CUDA available)",
    )
    pipeline_parser.add_argument(
        "--neuraladmixture-batch-size",
        type=int,
        help="Batch size for neural admixture training and inference (helps avoid OOM on large datasets)",
    )
    pipeline_parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCs")
    pipeline_parser.add_argument("--k-min", type=int, default=2, help="Min K (admixture)")
    pipeline_parser.add_argument("--k-max", type=int, default=10, help="Max K (admixture)")
    pipeline_parser.add_argument(
        "--embedding",
        choices=["phate", "umap", "tsne", "diffusion_map"],
        default="phate",
        help="Embedding method",
    )
    pipeline_parser.add_argument("--knn", type=int, default=5, help="KNN (PHATE/DM, default 5)")
    pipeline_parser.add_argument("--t", default="auto", help="Diffusion time (PHATE)")
    pipeline_parser.add_argument(
        "--n-landmark",
        type=str,
        help='Number of landmarks for PHATE (use an integer or "None"/omit for all samples)',
    )
    pipeline_parser.add_argument(
        "--random-landmarking",
        action="store_true",
        help="Use random landmarking for PHATE (requires --n-landmark)",
    )
    pipeline_parser.add_argument("--n-neighbors", type=int, default=15, help="Neighbors (UMAP)")
    pipeline_parser.add_argument("--perplexity", type=float, default=30, help="Perplexity (t-SNE)")
    pipeline_parser.add_argument(
        "--embed-batch-size",
        type=int,
        help="Batch size for embedding transform (to avoid OOM on large datasets; None=no batching)",
    )
    pipeline_parser.add_argument("--skip-pca", action="store_true", help="Skip PCA")
    pipeline_parser.add_argument("--skip-admixture", action="store_true", help="Skip admixture")
    pipeline_parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding")
    pipeline_parser.add_argument(
        "--embedding-input",
        choices=["fit", "project", "both"],
        default="both",
        help="Which dataset to embed: 'fit' (fit+transform on fit set), 'project' (fit+transform on project set), 'both' (fit on fit, transform on project)",
    )
    pipeline_parser.add_argument(
        "--skip-embedding-visualization", action="store_true", help="Skip embedding visualization"
    )
    pipeline_parser.add_argument(
        "--skip-pca-visualization", action="store_true", help="Skip PCA visualization"
    )
    pipeline_parser.add_argument(
        "--skip-admixture-visualization",
        action="store_true",
        help="Skip admixture visualizations (bar/embedding)",
    )
    pipeline_parser.add_argument(
        "--admixture-group-column",
        help="Grouping column for admixture barplots (defaults to first colormap key; e.g., Genetic_region_merged)",
    )
    pipeline_parser.add_argument(
        "--admixture-within-group-order",
        choices=["chron", "tree", "none"],
        default="chron",
        help="Method for ordering samples within groups in admixture barplots: 'chron' (sort by components), 'tree' (hierarchical clustering), 'none' (original order)",
    )
    pipeline_parser.add_argument(
        "--projection-plot-fit-column",
        help="Column from fit colormap to use for projection plot (e.g., Genetic_region_merged)",
    )
    pipeline_parser.add_argument(
        "--projection-plot-project-column",
        help="Column from project colormap to use for projection plot (e.g., race_ethnicity)",
    )
    pipeline_parser.add_argument("--skip-metrics", action="store_true", help="Skip metrics")
    pipeline_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    # Parse and execute
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
