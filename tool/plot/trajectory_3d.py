"""Animated 3D comparison plots for policy trajectories."""

import os
from pathlib import Path
import tempfile
from typing import Optional

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mini-umi-matplotlib"),
)

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


def _as_tf(name: str, value: np.ndarray, dimensions: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != dimensions or array.shape[-2:] != (4, 4):
        expected = "[T, 4, 4]" if dimensions == 3 else "[F, H, 4, 4]"
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _set_equal_axes(ax, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float((maximum - minimum).max()) / 2.0, 1e-3)
    radius *= 1.08
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def save_trajectory_comparison_gif(
    *,
    output_path: str | Path,
    ground_truth_tf_world: np.ndarray,
    prediction_tf_world: np.ndarray,
    target_tf_world: np.ndarray,
    frame_indices: np.ndarray,
    dataset_fps: float,
    gif_fps: float = 10.0,
    episode_index: Optional[int] = None,
    frame_position_rmse: Optional[np.ndarray] = None,
    orientation_stride: int = 3,
    orientation_axis_length_m: Optional[float] = None,
    dpi: int = 110,
) -> Path:
    """Save an animated world-frame trajectory comparison.

    ``ground_truth_tf_world`` is the complete episode action trajectory.
    ``prediction_tf_world[f]`` and ``target_tf_world[f]`` are the predicted and
    original future horizons at ``frame_indices[f]``. Transform rotation columns
    are rendered as red X, green Y, and blue Z axes.
    """
    ground_truth_tf_world = _as_tf(
        "ground_truth_tf_world",
        ground_truth_tf_world,
        dimensions=3,
    )
    prediction_tf_world = _as_tf(
        "prediction_tf_world",
        prediction_tf_world,
        dimensions=4,
    )
    target_tf_world = _as_tf(
        "target_tf_world",
        target_tf_world,
        dimensions=4,
    )
    if prediction_tf_world.shape != target_tf_world.shape:
        raise ValueError("prediction_tf_world and target_tf_world must have equal shapes")
    ground_truth_xyz = ground_truth_tf_world[..., :3, 3]
    prediction_xyz = prediction_tf_world[..., :3, 3]
    target_xyz = target_tf_world[..., :3, 3]
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if frame_indices.ndim != 1 or len(frame_indices) != len(prediction_xyz):
        raise ValueError("frame_indices must have one entry per prediction frame")
    if len(frame_indices) == 0:
        raise ValueError("At least one prediction frame is required")
    if np.any(frame_indices < 0) or np.any(frame_indices >= len(ground_truth_xyz)):
        raise ValueError("frame_indices must index ground_truth_xyz")
    if dataset_fps <= 0 or gif_fps <= 0:
        raise ValueError("dataset_fps and gif_fps must be positive")
    if orientation_stride <= 0:
        raise ValueError("orientation_stride must be positive")
    if frame_position_rmse is not None:
        frame_position_rmse = np.asarray(frame_position_rmse, dtype=np.float64)
        if frame_position_rmse.shape != (len(frame_indices),):
            raise ValueError("frame_position_rmse must have one entry per frame")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".gif":
        raise ValueError(f"output_path must end in .gif, got {output_path}")

    all_points = np.concatenate(
        [
            ground_truth_xyz,
            prediction_xyz.reshape(-1, 3),
            target_xyz.reshape(-1, 3),
        ],
        axis=0,
    )
    if orientation_axis_length_m is None:
        orientation_axis_length_m = max(
            float(np.ptp(all_points, axis=0).max()) * 0.055,
            0.01,
        )
    if orientation_axis_length_m <= 0:
        raise ValueError("orientation_axis_length_m must be positive")
    figure = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    ax.plot(
        ground_truth_xyz[:, 0],
        ground_truth_xyz[:, 1],
        ground_truth_xyz[:, 2],
        color="0.75",
        linewidth=1.5,
        label="ground truth (full)",
    )
    progress_line, = ax.plot(
        [], [], [],
        color="#1f77b4",
        linewidth=2.5,
        label="ground truth (elapsed)",
    )
    prediction_line, = ax.plot(
        [], [], [],
        color="#ff7f0e",
        linewidth=2.5,
        marker="o",
        markersize=2.5,
        label="prediction horizon",
    )
    current_point, = ax.plot(
        [], [], [],
        color="black",
        marker="o",
        markersize=6,
        linestyle="None",
        label="current time",
    )
    annotation = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, va="top")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=25, azim=-55)
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([0], [0], color="red", linewidth=2),
            Line2D([0], [0], color="green", linewidth=2),
            Line2D([0], [0], color="blue", linewidth=2),
        ]
    )
    labels.extend(["orientation X", "orientation Y", "orientation Z"])
    ax.legend(handles=handles, labels=labels, loc="lower left")
    _set_equal_axes(ax=ax, points=all_points)
    orientation_artists = []

    def draw_orientation(tf_world: np.ndarray, alpha: float) -> None:
        horizon_indices = list(range(0, len(tf_world), orientation_stride))
        if horizon_indices[-1] != len(tf_world) - 1:
            horizon_indices.append(len(tf_world) - 1)
        selected_tf_world = tf_world[horizon_indices]
        origins = selected_tf_world[:, :3, 3]
        rotations = selected_tf_world[:, :3, :3]
        for axis_index, color in enumerate(("red", "green", "blue")):
            directions = rotations[:, :, axis_index]
            orientation_artists.append(
                ax.quiver(
                    origins[:, 0],
                    origins[:, 1],
                    origins[:, 2],
                    directions[:, 0],
                    directions[:, 1],
                    directions[:, 2],
                    length=orientation_axis_length_m,
                    normalize=True,
                    arrow_length_ratio=0.3,
                    color=color,
                    alpha=alpha,
                    linewidth=1.0,
                )
            )

    def update(frame_number: int):
        for artist in orientation_artists:
            artist.remove()
        orientation_artists.clear()
        frame_index = int(frame_indices[frame_number])
        progress = ground_truth_xyz[:frame_index + 1]
        prediction = prediction_xyz[frame_number]
        current = ground_truth_xyz[frame_index]
        progress_line.set_data(progress[:, 0], progress[:, 1])
        progress_line.set_3d_properties(progress[:, 2])
        prediction_line.set_data(prediction[:, 0], prediction[:, 1])
        prediction_line.set_3d_properties(prediction[:, 2])
        current_point.set_data([current[0]], [current[1]])
        current_point.set_3d_properties([current[2]])
        draw_orientation(target_tf_world[frame_number], alpha=0.25)
        draw_orientation(prediction_tf_world[frame_number], alpha=0.9)

        episode_label = "" if episode_index is None else f"episode {episode_index} | "
        detail = f"{episode_label}frame {frame_index} | t={frame_index / dataset_fps:.2f}s"
        if frame_position_rmse is not None:
            detail += f"\nposition RMSE={frame_position_rmse[frame_number] * 1000.0:.1f} mm"
        detail += "\norientation: prediction opaque | ground truth faint"
        annotation.set_text(detail)
        return progress_line, prediction_line, current_point, annotation, *orientation_artists

    animation = FuncAnimation(
        fig=figure,
        func=update,
        frames=len(frame_indices),
        interval=1000.0 / gif_fps,
        blit=False,
    )
    try:
        animation.save(
            filename=str(output_path),
            writer=PillowWriter(fps=gif_fps),
            dpi=dpi,
        )
    finally:
        plt.close(figure)
    return output_path
