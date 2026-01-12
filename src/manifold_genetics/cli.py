"""
Command-line interface for manifold-genetics.

Provides commands for PCA, admixture, embeddings, visualization, and full pipeline.
"""

import argparse
import json
import logging
import sys
import re
from pathlib import Path
from typing import List, Optional

from .pca import PCA
from .admixture import NeuralAdmixture
from .embeddings import PHATE, UMAP, TSNE, DiffusionMap
from .visualization import (
    visualize,
    plot_pca_pairs,
    plot_projection,
    plot_admixture_bar_grid,
    plot_admixture_embedding_grid,
)
from .pipeline import Pipeline, run_pipeline
from .metrics import compute_geographic_preservation, compute_admixture_preservation
from .utils.io import read_colormap


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)-40s - %(levelname)s - %(message)s"
    )


def _resolve_k_values(k_min: Optional[int], k_max: Optional[int], ks: Optional[List[int]], q_prefix: Optional[Path] = None) -> List[int]:
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
    raise ValueError("Please provide --ks or both --k-min/--k-max (or ensure <prefix>.<K>.csv files exist to auto-detect).")


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
    checkpoint_dir = Path(args.neuraladmixture_output_dir or args.output or Path.cwd() / "admixture_outputs")
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
        batch_size=getattr(args, 'neuraladmixture_batch_size', None),
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

    output_path = Path(args.output) if args.output else q_prefix.parent / f"{q_prefix.name}_admixture.png"

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
    )
    print(f"Admixture bar plot written to: {output_path}")
    return 0


def cmd_plot_admixture_embedding(args):
    """Plot embedding colored by admixture components for multiple Ks."""
    setup_logging(args.verbose)

    q_prefix = Path(args.q_prefix)
    embedding = args.embedding
    k_values = _resolve_k_values(args.k_min, args.k_max, args.ks, q_prefix=q_prefix)
    output_path = Path(args.output) if args.output else q_prefix.parent / f"{q_prefix.name}_admixture_embedding.png"

    plot_admixture_embedding_grid(
        embedding=embedding,
        q_prefix=q_prefix,
        k_values=k_values,
        output_path=output_path,
        pc_x=args.pc_x,
        pc_y=args.pc_y,
        subsample=args.subsample,
    )
    print(f"Admixture-embedding plot written to: {output_path}")
    return 0


def cmd_embed(args):
    """Run embedding command."""
    setup_logging(args.verbose)

    fit_input = args.fit_input or args.input
    project_input = args.project_input or fit_input

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

    # Build Q files dict from prefix: {k: path/to/prefix.k.csv}
    q_files = {}
    for k in range(args.k_min, args.k_max + 1):
        q_file = Path(f"{admixture_prefix}.{k}.csv")
        if not q_file.exists():
            raise FileNotFoundError(f"Admixture file not found: {q_file}")
        q_files[k] = q_file

    if not q_files:
        raise ValueError(f"No admixture files found for K={args.k_min} to {args.k_max}")

    metrics = compute_admixture_preservation(
        embedding=args.embedding,
        q_files=q_files,
        k_value=args.k_value,
        num_samples=args.num_dists_sampled,
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

    output_dir = Path(args.output) if args.output else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load colormap to get label columns
    with open(args.colormap, "r") as f:
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

    figure_path = plot_projection(
        fit_embedding=args.fit_embedding,
        project_embedding=args.project_embedding,
        fit_labels=args.fit_labels,
        project_labels=args.project_labels,
        fit_colormap=args.fit_colormap,
        project_colormap=args.project_colormap,
        output_path=args.output,
        point_size=args.point_size,
        alpha=args.alpha,
        linewidth=args.linewidth,
    )

    print(f"Projection plot saved: {figure_path}")
    return 0


def cmd_pipeline(args):
    """Run full pipeline command."""
    setup_logging(args.verbose)

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
    project_labels = args.project_labels if hasattr(args, "project_labels") and args.project_labels else None
    fit_colormap = args.fit_colormap if hasattr(args, "fit_colormap") and args.fit_colormap else None
    project_colormap = args.project_colormap if hasattr(args, "project_colormap") and args.project_colormap else None

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
        admix_batch_size=getattr(args, 'neuraladmixture_batch_size', None),
        embedding=args.embedding,
        embedding_params=embedding_params,
        embedding_input=args.embedding_input,
        admix_group_column=args.admixture_group_column,
        admix_within_group_order=None if args.admixture_within_group_order == "none" else args.admixture_within_group_order,
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
            print(
                f"  Geographic preservation: {geo['correlation']:.4f} (p={geo['p_value']:.2e})"
            )
        if "admixture" in results["metrics"]:
            print("  Admixture preservation:")
            for k, metrics in results["metrics"]["admixture"].items():
                print(f"    K={k}: {metrics['correlation']:.4f}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="manifold-genetics",
        description="Genetic analysis with PCA, Admixture, and manifold learning",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # PCA command
    pca_parser = subparsers.add_parser("pca", help="Run PCA")
    pca_parser.add_argument("--input", help="Input PLINK prefix (fit + project)")
    pca_parser.add_argument(
        "--fit-plink", help="PLINK prefix to fit PCA (defaults to --input)"
    )
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
    admix_parser = subparsers.add_parser("admixture", help="Run neural admixture")
    admix_parser.add_argument("--input", help="Input PLINK prefix (fit + project)")
    admix_parser.add_argument("--fit-plink", help="PLINK prefix to fit admixture (defaults to --input)")
    admix_parser.add_argument("--project-plink", help="PLINK prefix to project admixture (defaults to fit prefix)")
    admix_parser.add_argument("--output", help="Output directory (deprecated; use --neuraladmixture-output-dir and explicit --fit-output/--project-output)")
    admix_parser.add_argument("--model-name", default="fit", help="Model name prefix (default: fit)")
    admix_parser.add_argument("--fit-output", help="Prefix for fit admixture CSV outputs (required)")
    admix_parser.add_argument("--project-output", help="Prefix for projected admixture CSV outputs (required)")
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
        "--neuraladmixture-batch-size", type=int, help="Batch size for training and inference (helps avoid OOM on large datasets)"
    )
    admix_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    admix_parser.set_defaults(func=cmd_admixture)

    # Admixture visualization: stacked barplots
    plot_admix_parser = subparsers.add_parser("plot-admixture", help="Plot stacked admixture barplots for Ks")
    plot_admix_parser.add_argument("--q-prefix", required=True, help="Prefix for admixture CSVs (<prefix>.<K>.csv)")
    plot_admix_parser.add_argument("--labels", required=True, help="Labels CSV with sample_id and grouping column")
    plot_admix_parser.add_argument("--group-column", required=True, help="Column in labels used to order bars (e.g., Population)")
    plot_admix_parser.add_argument("--colormap", required=True, help="Colormap JSON (used to derive group ordering for the chosen column)")
    plot_admix_parser.add_argument("--k-min", type=int, help="Minimum K (if Ks not provided)")
    plot_admix_parser.add_argument("--k-max", type=int, help="Maximum K (if Ks not provided)")
    plot_admix_parser.add_argument("--ks", type=int, nargs="+", help="Explicit K values (overrides k-min/k-max)")
    plot_admix_parser.add_argument("--subsample-per-group", type=int, help="Subsample each group to N samples (optional)")
    plot_admix_parser.add_argument("--within-group-order", choices=["chron", "tree", "none"], default="chron", help="Method for ordering samples within groups: 'chron' (sort by components), 'tree' (hierarchical clustering), 'none' (original order)")
    plot_admix_parser.add_argument("--output", help="Output PNG path (default: <prefix>_admixture.png)")
    plot_admix_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_admix_parser.set_defaults(func=cmd_plot_admixture)

    # Admixture visualization: embedding colored by admixture components
    plot_admix_emb_parser = subparsers.add_parser("plot-admixture-embedding", help="Plot embedding colored by admixture components")
    plot_admix_emb_parser.add_argument("--embedding", required=True, help="Embedding CSV (sample_id, dim_1, dim_2, ...)")
    plot_admix_emb_parser.add_argument("--q-prefix", required=True, help="Prefix for admixture CSVs (<prefix>.<K>.csv)")
    plot_admix_emb_parser.add_argument("--k-min", type=int, help="Minimum K (if Ks not provided)")
    plot_admix_emb_parser.add_argument("--k-max", type=int, help="Maximum K (if Ks not provided)")
    plot_admix_emb_parser.add_argument("--ks", type=int, nargs="+", help="Explicit K values (overrides k-min/k-max)")
    plot_admix_emb_parser.add_argument("--pc-x", type=int, default=1, help="PC for x-axis (default: 1)")
    plot_admix_emb_parser.add_argument("--pc-y", type=int, default=2, help="PC for y-axis (default: 2)")
    plot_admix_emb_parser.add_argument("--subsample", type=int, help="Optional subsample size for plotting")
    plot_admix_emb_parser.add_argument("--output", help="Output PNG path (default: <prefix>_admixture_embedding.png)")
    plot_admix_emb_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_admix_emb_parser.set_defaults(func=cmd_plot_admixture_embedding)

    # Embedding command
    embed_parser = subparsers.add_parser("embed", help="Run manifold embedding")
    embed_parser.add_argument("--input", help="Input CSV file (fit + project)")
    embed_parser.add_argument("--fit-input", help="Input CSV to fit embedding (defaults to --input)")
    embed_parser.add_argument("--project-input", help="Input CSV to project embedding (defaults to fit input)")
    embed_parser.add_argument("--fit-output", help="Output CSV for fit embedding (optional)")
    embed_parser.add_argument("--project-output", help="Output CSV for projected embedding (defaults to --output)")
    embed_parser.add_argument("--output", help="Output CSV for projected embedding")
    embed_parser.add_argument(
        "--method",
        choices=["phate", "umap", "tsne", "diffusion_map"],
        default="phate",
        help="Embedding method",
    )
    embed_parser.add_argument("--knn", type=int, default=25, help="K nearest neighbors (PHATE/DM)")
    embed_parser.add_argument("--t", default="auto", help="Diffusion time (PHATE)")
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
    plot_parser = subparsers.add_parser("plot", help="Visualize embeddings")
    plot_parser.add_argument("--input", required=True, help="Input embedding CSV")
    plot_parser.add_argument("--labels", required=True, help="Labels CSV file")
    plot_parser.add_argument("--colormap", required=True, help="Colormap JSON file")
    plot_parser.add_argument("--output", help="Output figure path")
    plot_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_parser.set_defaults(func=cmd_plot)

    # Plot PCA command
    plot_pca_parser = subparsers.add_parser("plot-pca", help="Visualize PCA coordinates")
    plot_pca_parser.add_argument("--input", required=True, help="Input PCA CSV file")
    plot_pca_parser.add_argument("--labels", required=True, help="Labels CSV file")
    plot_pca_parser.add_argument("--colormap", required=True, help="Colormap JSON file")
    plot_pca_parser.add_argument("--output", help="Output directory for PCA figures")
    plot_pca_parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCs to plot")
    plot_pca_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_pca_parser.set_defaults(func=cmd_plot_pca)

    # Plot: projection (fit + project together)
    plot_proj_parser = subparsers.add_parser("plot-projection", help="Plot fit and projection embeddings together")
    plot_proj_parser.add_argument("--fit-embedding", required=True, help="Fit embedding CSV file")
    plot_proj_parser.add_argument("--project-embedding", required=True, help="Project embedding CSV file")
    plot_proj_parser.add_argument("--fit-labels", required=True, help="Fit labels CSV file")
    plot_proj_parser.add_argument("--project-labels", required=True, help="Project labels CSV file")
    plot_proj_parser.add_argument("--fit-colormap", required=True, help="Fit colormap JSON file")
    plot_proj_parser.add_argument("--project-colormap", required=True, help="Project colormap JSON file")
    plot_proj_parser.add_argument("--output", required=True, help="Output figure path")
    plot_proj_parser.add_argument("--point-size", type=float, default=4.0, help="Size of scatter points")
    plot_proj_parser.add_argument("--alpha", type=float, default=0.6, help="Transparency of points")
    plot_proj_parser.add_argument("--linewidth", type=float, default=0.8, help="Edge width for hollow markers")
    plot_proj_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_proj_parser.set_defaults(func=cmd_plot_projection)

    # Metrics: geographic
    geo_metrics_parser = subparsers.add_parser("metrics-geographic", help="Compute geographic preservation")
    geo_metrics_parser.add_argument("--embedding", required=True, help="Embedding CSV")
    geo_metrics_parser.add_argument("--geographic", required=True, help="Geographic coordinates CSV")
    geo_metrics_parser.add_argument("--output", required=True, help="Output JSON path")
    geo_metrics_parser.add_argument("--longitude-col", default="longitude", help="Longitude column name")
    geo_metrics_parser.add_argument("--latitude-col", default="latitude", help="Latitude column name")
    geo_metrics_parser.add_argument("--num-dists-sampled", type=int, default=50000, help="Max pairwise distances to sample")
    geo_metrics_parser.add_argument("--keep-missing", action="store_true", help="Keep samples with missing coordinates")
    geo_metrics_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    geo_metrics_parser.set_defaults(func=cmd_metrics_geographic)

    # Metrics: admixture
    admix_metrics_parser = subparsers.add_parser("metrics-admixture", help="Compute admixture preservation")
    admix_metrics_parser.add_argument("--embedding", required=True, help="Embedding CSV")
    admix_metrics_parser.add_argument("--admixture-output", required=True, help="Prefix for admixture CSVs (<prefix>.K.csv, e.g., path/to/transform)")
    admix_metrics_parser.add_argument("--output", required=True, help="Output JSON path")
    admix_metrics_parser.add_argument("--k-min", type=int, required=True, help="Minimum K to include")
    admix_metrics_parser.add_argument("--k-max", type=int, required=True, help="Maximum K to include")
    admix_metrics_parser.add_argument("--k-value", type=int, help="Compute only a single K")
    admix_metrics_parser.add_argument("--num-dists-sampled", type=int, default=50000, help="Max pairwise distances to sample")
    admix_metrics_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    admix_metrics_parser.set_defaults(func=cmd_metrics_admixture)

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument(
        "--fit-plink",
        required=True,
        help="PLINK prefix for PCA/admixture reference set (fit subset)",
    )
    pipeline_parser.add_argument(
        "--project-plink",
        required=True,
        help="PLINK prefix to project/apply the fitted models (transform subset)",
    )
    pipeline_parser.add_argument("--labels", help="Labels CSV file (used for both fit and project if --fit-labels/--project-labels not provided)")
    pipeline_parser.add_argument("--colormap", help="Colormap JSON file (used for both fit and project if --fit-colormap/--project-colormap not provided)")
    pipeline_parser.add_argument("--fit-labels", help="Labels CSV file for fit dataset (for separate fit/project labels)")
    pipeline_parser.add_argument("--project-labels", help="Labels CSV file for project dataset (for separate fit/project labels)")
    pipeline_parser.add_argument("--fit-colormap", help="Colormap JSON file for fit dataset (optional, for cross-cohort analysis)")
    pipeline_parser.add_argument("--project-colormap", help="Colormap JSON file for project dataset (optional, for cross-cohort analysis)")
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
    pipeline_parser.add_argument("--n-landmark", type=str, help='Number of landmarks for PHATE (use an integer or "None"/omit for all samples)')
    pipeline_parser.add_argument("--random-landmarking", action="store_true", help="Use random landmarking for PHATE (requires --n-landmark)")
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
        help="Which dataset to embed: 'fit' (fit+transform on fit set), 'project' (fit+transform on project set), 'both' (fit on fit, transform on project)"
    )
    pipeline_parser.add_argument(
        "--skip-embedding-visualization", action="store_true", help="Skip embedding visualization"
    )
    pipeline_parser.add_argument(
        "--skip-pca-visualization", action="store_true", help="Skip PCA visualization"
    )
    pipeline_parser.add_argument(
        "--skip-admixture-visualization", action="store_true", help="Skip admixture visualizations (bar/embedding)"
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
    pipeline_parser.add_argument("--skip-metrics", action="store_true", help="Skip metrics")
    pipeline_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    # Parse and execute
    args = parser.parse_args()

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
