"""A rotating video of a 3-D embedding, written alongside the interactive HTML.

An HTML figure cannot go into slides or supplementary material; a video can.
Frames are drawn with matplotlib's 3-D axes (no browser), stacked the way the
2-D and 3-D figures stack them -- Unknown at the bottom, the colormap's first
entries on top -- and the camera turns once around the vertical axis.

MP4 needs ``imageio-ffmpeg`` (a bundled ffmpeg binary, no system install), part
of the ``interactive`` extra with plotly; GIF needs only Pillow.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib
import numpy as np
import pandas as pd

from ..utils.io import read_colormap, read_labels_csv

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

VIDEO_FORMATS = ("mp4", "gif")


def _ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise ImportError(
            "An MP4 of the 3-D embedding needs imageio-ffmpeg: "
            "pip install 'manifold-genetics[interactive]' (or ask for a GIF, which does not)"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _writer(output_path: Path, fps: int):
    from matplotlib import animation

    fmt = output_path.suffix.lstrip(".").lower()
    if fmt == "mp4":
        matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg_path()
        return animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    if fmt == "gif":
        return animation.PillowWriter(fps=fps)
    raise ValueError(f"video format {fmt!r} is not one of {VIDEO_FORMATS}")


def plot_embedding_rotation(
    embedding: Union[pd.DataFrame, str, Path],
    labels: Union[pd.DataFrame, str, Path],
    colormap: Union[Dict, str, Path],
    output_path: Union[str, Path],
    label_column: Optional[str] = None,
    title: Optional[str] = None,
    point_size: float = 2.0,
    alpha: float = 0.6,
    max_points: Optional[int] = None,
    random_state: Optional[int] = 42,
    fps: int = 24,
    seconds: float = 12.0,
    elev: float = 20.0,
    dpi: int = 120,
    figsize: tuple = (7.0, 6.0),
) -> Path:
    """Write a video of the embedding turning once through 360 degrees.

    The format follows the suffix of ``output_path`` (``.mp4`` or ``.gif``).
    Points are stacked as in the 2-D and 3-D figures: samples with no colour
    first (grey, Unknown), then the colormap's groups in reverse, so its first
    entries are drawn on top. The legend lists the groups in colormap order.

    Args:
        max_points: Cap on points drawn (a random, seeded subsample); None keeps all.
        fps, seconds: Frame rate and length; ``fps * seconds`` frames in all.
        elev: Camera elevation in degrees; 0 looks along the plane of dims 1-2.

    Returns:
        Path to the written video.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Patch

    # Imported here: plotting imports this module's caller, not the reverse.
    from .plotting import UNKNOWN_COLOR, UNKNOWN_LABEL, _read_embedding_3d, uncoloured

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _writer(output_path, fps)  # fail on a missing encoder before drawing

    embedding_df = _read_embedding_3d(embedding)
    labels_df = read_labels_csv(labels) if isinstance(labels, (str, Path)) else labels
    if labels_df.index.name == "sample_id":
        labels_df = labels_df.reset_index()
    colormap_dict = read_colormap(colormap) if isinstance(colormap, (str, Path)) else colormap
    if label_column is None:
        label_column = next(iter(colormap_dict))
    if label_column not in colormap_dict:
        raise ValueError(f"Column {label_column!r} is not in the colormap")
    color_dict = colormap_dict[label_column]

    df = embedding_df.merge(labels_df, on="sample_id", how="inner")
    if label_column not in df.columns:
        raise ValueError(f"Column {label_column!r} not found in the labels file")
    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=random_state)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection="3d")
    # Keep the drawing order below; by default mplot3d re-sorts whole groups by
    # depth, which would undo the stacking the other figures use.
    ax.computed_zorder = False

    xyz = df[["dim_1", "dim_2", "dim_3"]].to_numpy()
    spans = np.ptp(xyz, axis=0)
    # Zoomed so the cloud fills the frame; the box corners it would clip are
    # empty with the axes hidden.
    aspect = np.where(spans > 0, spans, 1.0)
    try:
        ax.set_box_aspect(aspect, zoom=1.45)
    except TypeError:  # matplotlib < 3.7 has no zoom
        ax.set_box_aspect(aspect)
    ax.set_axis_off()

    missing = uncoloured(df[label_column], color_dict)
    if missing.any():
        part = df[missing]
        ax.scatter(
            part["dim_1"],
            part["dim_2"],
            part["dim_3"],
            s=point_size,
            c=UNKNOWN_COLOR,
            alpha=alpha * 0.5,
            linewidths=0,
            depthshade=False,
        )
    groups = [k for k in color_dict if k in set(df[label_column].dropna().astype(str))]
    for label in reversed(groups):
        part = df[df[label_column].astype(str) == label]
        ax.scatter(
            part["dim_1"],
            part["dim_2"],
            part["dim_3"],
            s=point_size,
            c=color_dict[label],
            alpha=alpha,
            linewidths=0,
            depthshade=False,
        )

    handles = [Patch(facecolor=color_dict[g], label=g) for g in groups]
    if missing.any():
        handles.append(Patch(facecolor=UNKNOWN_COLOR, label=UNKNOWN_LABEL))
    fig.legend(
        handles=handles, loc="lower center", ncol=min(3, len(handles)), fontsize=7, frameon=False
    )
    if title:
        fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(left=0, right=1, top=0.97, bottom=0.08)

    n_frames = max(1, int(round(fps * seconds)))

    def turn(i):
        ax.view_init(elev=elev, azim=-90 + 360.0 * i / n_frames)
        return ()

    anim = animation.FuncAnimation(fig, turn, frames=n_frames, blit=False)
    logger.info(f"Writing {n_frames} frames of {len(df)} points to {output_path}")
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path
