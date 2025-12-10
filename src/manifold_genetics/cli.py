"""
Command-line interface for manifold-genetics.

Provides commands for PCA, admixture, embeddings, visualization, and full pipeline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .pca import PCA
from .admixture import NeuralAdmixture
from .embeddings import PHATE, UMAP, TSNE, DiffusionMap
from .visualization import visualize
from .pipeline import Pipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)-40s - %(levelname)s - %(message)s"
    )


def cmd_pca(args):
    """Run PCA command."""
    setup_logging(args.verbose)

    pca = PCA(n_components=args.n_pcs, force=args.force)
    pca_coords = pca.fit_transform(args.input, output_path=args.output)

    print(f"PCA complete: {args.output}")
    print(f"Shape: {pca_coords.shape}")
    return 0


def cmd_admixture(args):
    """Run admixture command."""
    setup_logging(args.verbose)

    admix = NeuralAdmixture(
        k_min=args.k_min,
        k_max=args.k_max,
        force=args.force,
        threads=args.threads,
    )
    q_files = admix.fit_transform(args.input, output_dir=args.output)

    print(f"Admixture complete: {args.output}")
    print(f"Generated Q files for K={args.k_min} to {args.k_max}:")
    for k, path in sorted(q_files.items()):
        print(f"  K={k}: {path}")
    return 0


def cmd_embed(args):
    """Run embedding command."""
    setup_logging(args.verbose)

    # Parse embedding-specific parameters
    embed_params = {}
    if args.method == "phate":
        embed_params = {"knn": args.knn, "t": args.t}
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

    embedding = model.fit_transform(args.input, output_path=args.output)

    print(f"Embedding complete: {args.output}")
    print(f"Shape: {embedding.shape}")
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

    figure_paths = visualize(
        embedding=args.input,
        labels=args.labels,
        colormap=args.colormap,
        output_dir=output_dir,
        output_prefix="pca",
    )

    print(f"PCA visualization complete:")
    for path in figure_paths:
        print(f"  {path}")
    return 0


def cmd_pipeline(args):
    """Run full pipeline command."""
    setup_logging(args.verbose)

    # Parse embedding parameters
    embedding_params = {}
    if args.embedding == "phate":
        embedding_params["knn"] = args.knn
        embedding_params["t"] = args.t
        # Handle random landmarking
        if args.phate_random_landmark:
            embedding_params["n_landmark"] = None
        else:
            embedding_params["n_landmark"] = 2000
    elif args.embedding == "umap":
        embedding_params["n_neighbors"] = args.n_neighbors
    elif args.embedding == "tsne":
        embedding_params["perplexity"] = args.perplexity
    elif args.embedding == "diffusion_map":
        embedding_params["knn"] = args.knn

    pipeline = Pipeline(
        fit_plink_prefix=args.fit_plink,
        transform_plink_prefix=args.project_plink,
        labels=args.labels,
        colormap=args.colormap,
        output_dir=args.output,
        geographic_coords=args.geographic if hasattr(args, "geographic") else None,
    )

    results = pipeline.run(
        n_pcs=args.n_pcs,
        k_min=args.k_min,
        k_max=args.k_max,
        embedding=args.embedding,
        embedding_params=embedding_params,
        skip_pca=args.skip_pca,
        skip_admixture=args.skip_admixture,
        skip_embedding=args.skip_embedding,
        skip_visualization=args.skip_embedding_visualization,
        skip_pca_visualization=args.skip_pca_visualization,
        skip_metrics=args.skip_metrics,
        admix_threads=args.threads,
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
    pca_parser.add_argument("--input", required=True, help="Input PLINK prefix")
    pca_parser.add_argument("--output", required=True, help="Output CSV file")
    pca_parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCs")
    pca_parser.add_argument("--force", action="store_true", help="Force recomputation")
    pca_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    pca_parser.set_defaults(func=cmd_pca)

    # Admixture command
    admix_parser = subparsers.add_parser("admixture", help="Run neural admixture")
    admix_parser.add_argument("--input", required=True, help="Input PLINK prefix")
    admix_parser.add_argument("--output", required=True, help="Output directory")
    admix_parser.add_argument("--k-min", type=int, default=2, help="Minimum K")
    admix_parser.add_argument("--k-max", type=int, default=10, help="Maximum K")
    admix_parser.add_argument("--force", action="store_true", help="Force retraining")
    admix_parser.add_argument("--threads", type=int, help="Threads for neural admixture")
    admix_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    admix_parser.set_defaults(func=cmd_admixture)

    # Embedding command
    embed_parser = subparsers.add_parser("embed", help="Run manifold embedding")
    embed_parser.add_argument("--input", required=True, help="Input CSV file (PCA coords)")
    embed_parser.add_argument("--output", required=True, help="Output CSV file")
    embed_parser.add_argument(
        "--method",
        choices=["phate", "umap", "tsne", "diffusion_map"],
        default="phate",
        help="Embedding method",
    )
    embed_parser.add_argument("--knn", type=int, default=25, help="K nearest neighbors (PHATE/DM)")
    embed_parser.add_argument("--t", default="auto", help="Diffusion time (PHATE)")
    embed_parser.add_argument("--n-neighbors", type=int, default=15, help="Neighbors (UMAP)")
    embed_parser.add_argument("--min-dist", type=float, default=0.1, help="Min dist (UMAP)")
    embed_parser.add_argument("--perplexity", type=float, default=30, help="Perplexity (t-SNE)")
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
    plot_pca_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    plot_pca_parser.set_defaults(func=cmd_plot_pca)

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
    pipeline_parser.add_argument("--labels", required=True, help="Labels CSV file")
    pipeline_parser.add_argument("--colormap", required=True, help="Colormap JSON file")
    pipeline_parser.add_argument("--output", required=True, help="Output directory")
    pipeline_parser.add_argument("--geographic", help="Geographic coordinates CSV")
    pipeline_parser.add_argument(
        "--threads",
        type=int,
        help="Threads for neural admixture (defaults to SLURM/affinity detection)",
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
    pipeline_parser.add_argument("--t", type=int, default=5, help="Diffusion time (PHATE)")
    pipeline_parser.add_argument("--phate-random-landmark", action="store_true", help="Use random landmarking for PHATE")
    pipeline_parser.add_argument("--n-neighbors", type=int, default=15, help="Neighbors (UMAP)")
    pipeline_parser.add_argument("--perplexity", type=float, default=30, help="Perplexity (t-SNE)")
    pipeline_parser.add_argument("--skip-pca", action="store_true", help="Skip PCA")
    pipeline_parser.add_argument("--skip-admixture", action="store_true", help="Skip admixture")
    pipeline_parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding")
    pipeline_parser.add_argument(
        "--skip-embedding-visualization", action="store_true", help="Skip embedding visualization"
    )
    pipeline_parser.add_argument(
        "--skip-pca-visualization", action="store_true", help="Skip PCA visualization"
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
