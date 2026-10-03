"""The rotating video written beside each 3-D HTML figure (#159)."""

import numpy as np
import pandas as pd
import pytest

from manifold_genetics.visualization import video
from manifold_genetics.visualization.plotting import UNKNOWN_COLOR

CMAP = {"Region": {"A": "#111111", "B": "#222222", "C": "#333333"}}


def _inputs(values, seed=0):
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in range(len(values))]
    emb = pd.DataFrame(
        {"sample_id": ids, **{f"dim_{k}": rng.normal(size=len(ids)) for k in (1, 2, 3)}}
    )
    return emb, pd.DataFrame({"sample_id": ids, "Region": values})


@pytest.fixture
def drawn(monkeypatch):
    """Colour of each scatter call, in drawing order."""
    from mpl_toolkits.mplot3d import Axes3D

    calls = []
    real = Axes3D.scatter

    def rec(self, xs, ys, *a, **kw):
        calls.append((kw.get("c"), len(xs)))
        return real(self, xs, ys, *a, **kw)

    monkeypatch.setattr(Axes3D, "scatter", rec)
    return calls


def test_writes_a_gif_with_fps_times_seconds_frames(tmp_path):
    from PIL import Image

    emb, labels = _inputs(list("AABBCC"))
    out = video.plot_embedding_rotation(emb, labels, CMAP, tmp_path / "e.gif", fps=4, seconds=1.5)

    assert out.exists() and out.stat().st_size > 0
    assert Image.open(out).n_frames == 6


def test_writes_an_mp4(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    emb, labels = _inputs(list("AABBCC"))

    out = video.plot_embedding_rotation(emb, labels, CMAP, tmp_path / "e.mp4", fps=4, seconds=0.5)

    assert out.exists() and out.stat().st_size > 0


def test_groups_stack_like_the_other_figures(tmp_path, drawn):
    """Unknown first (bottom), then the colormap in reverse, so A is on top."""
    emb, labels = _inputs(["A", "B", "C", "Z", None, "A"])

    video.plot_embedding_rotation(emb, labels, CMAP, tmp_path / "e.gif", fps=1, seconds=1)

    assert [c for c, _ in drawn] == [UNKNOWN_COLOR, "#333333", "#222222", "#111111"]
    assert drawn[0][1] == 2, "the unlisted Z and the empty label, both grey"


def test_every_point_is_drawn(tmp_path, drawn):
    emb, labels = _inputs(list("ABCABCZ"))

    video.plot_embedding_rotation(emb, labels, CMAP, tmp_path / "e.gif", fps=1, seconds=1)

    assert sum(n for _, n in drawn) == 7


def test_max_points_subsamples_reproducibly(tmp_path, drawn):
    emb, labels = _inputs(list("ABC") * 20)

    video.plot_embedding_rotation(
        emb, labels, CMAP, tmp_path / "e.gif", fps=1, seconds=1, max_points=12
    )

    assert sum(n for _, n in drawn) == 12


def test_an_unknown_format_is_refused_before_drawing(tmp_path, drawn):
    emb, labels = _inputs(list("ABC"))

    with pytest.raises(ValueError, match="mp4"):
        video.plot_embedding_rotation(emb, labels, CMAP, tmp_path / "e.avi")
    assert not drawn


def test_a_missing_encoder_names_the_extra(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "imageio_ffmpeg":
            raise ImportError("no imageio_ffmpeg")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    emb, labels = _inputs(list("ABC"))

    with pytest.raises(ImportError, match=r"manifold-genetics\[interactive\]"):
        video.plot_embedding_rotation(emb, labels, CMAP, tmp_path / "e.mp4")


class TestVisualize3d:
    """`plot-3d` / visualize_3d write the HTML and the video side by side."""

    def test_html_and_video_for_each_label_column(self, tmp_path):
        pytest.importorskip("plotly")
        from manifold_genetics.visualization.plotting import visualize_3d

        emb, labels = _inputs(list("AABBCC"))
        paths = visualize_3d(
            emb,
            labels,
            CMAP,
            output_dir=tmp_path,
            video_format="gif",
            video_kwargs={"fps": 2, "seconds": 1},
        )

        assert sorted(p.name for p in paths) == [
            "embedding_3d_by_Region.gif",
            "embedding_3d_by_Region.html",
        ]

    def test_no_video_writes_the_html_alone(self, tmp_path):
        pytest.importorskip("plotly")
        from manifold_genetics.visualization.plotting import visualize_3d

        emb, labels = _inputs(list("AABBCC"))
        paths = visualize_3d(emb, labels, CMAP, output_dir=tmp_path, video_format=None)

        assert [p.suffix for p in paths] == [".html"]

    def test_without_the_encoder_the_html_is_still_written(self, tmp_path, monkeypatch, caplog):
        pytest.importorskip("plotly")
        from manifold_genetics.visualization.plotting import visualize_3d

        def no_encoder(*a, **kw):
            raise ImportError("An MP4 needs imageio-ffmpeg")

        monkeypatch.setattr(video, "plot_embedding_rotation", no_encoder)
        emb, labels = _inputs(list("AABBCC"))

        paths = visualize_3d(emb, labels, CMAP, output_dir=tmp_path)

        assert [p.suffix for p in paths] == [".html"]
        assert "not the video" in caplog.text


def test_plot_3d_cli_passes_the_video_options(tmp_path, monkeypatch):
    from manifold_genetics import cli

    seen = {}
    monkeypatch.setattr(cli, "visualize_3d", lambda **kw: seen.update(kw) or [])
    for name in (
        "validate_embedding_csv",
        "validate_labels_csv",
        "validate_colormap_json",
        "validate_labels_colormap_match",
        "validate_sample_id_overlap",
    ):
        monkeypatch.setattr(cli, name, lambda *a, **k: None)

    base = ["plot-3d", "--input", "e.csv", "--labels", "l.csv", "--colormap", "c.json"]
    cli.main(base + ["--video-format", "gif", "--fps", "10", "--seconds", "3", "--elev", "5"])
    assert seen["video_format"] == "gif"
    assert seen["video_kwargs"] == {"fps": 10, "seconds": 3.0, "elev": 5.0}

    cli.main(base + ["--no-video"])
    assert seen["video_format"] is None
