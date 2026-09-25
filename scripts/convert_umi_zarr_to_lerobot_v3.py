#!/usr/bin/env python3
"""Convert an absolute-pose UMI Zarr replay buffer to LeRobot Dataset v3.0.

For LeRobot v2.1 output use ``convert_umi_zarr_to_lerobot_v21.py``.  The two
formats deliberately have separate entry points so a destination can never
silently change format after a dependency upgrade.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import zarr
from tqdm import tqdm

from diffusion_policy.dataset.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.dataset.replay_buffer import ReplayBuffer


def _is_rgb_array(array) -> bool:
    return array.ndim == 4 and array.shape[-1] in (1, 3, 4) and array.dtype == np.uint8


def _feature_spec(array, dtype: str) -> dict:
    return {
        "dtype": dtype,
        "shape": tuple(array.shape[1:]),
        "names": None,
    }


def _get_absolute_action(replay_buffer) -> np.ndarray:
    if "action" in replay_buffer:
        return np.asarray(replay_buffer["action"][:], dtype=np.float32)

    action_suffixes = (
        "eef_pos",
        "eef_rot_axis_angle",
        "gripper_width",
    )
    robot_indices = sorted({
        int(key.split("_")[0].removeprefix("robot"))
        for key in replay_buffer.keys()
        if key.startswith("robot")
        and any(key.endswith(f"_{suffix}") for suffix in action_suffixes)
    })
    actions = []
    for robot_index in robot_indices:
        keys = [
            f"robot{robot_index}_{suffix}"
            for suffix in action_suffixes
            if f"robot{robot_index}_{suffix}" in replay_buffer
        ]
        actions.extend(np.asarray(replay_buffer[key][:]) for key in keys)
    if not actions:
        raise KeyError(
            "The Zarr replay buffer has neither an 'action' array nor robot EEF state arrays")
    return np.concatenate(actions, axis=-1).astype(np.float32, copy=False)


def convert_zarr_to_lerobot(
        zarr_path: Path,
        output_dir: Path,
        repo_id: str,
        fps: int,
        task: str,
        overwrite: bool=False,
        image_writer_threads: int=4,
        _episode_per_file: bool=False) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "The converter requires lerobot==0.4.3. Run `uv sync` in the "
            "repository root before starting the conversion."
        ) from exc

    zarr_path = zarr_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        info_path = output_dir / "meta" / "info.json"
        if info_path.is_file():
            with info_path.open() as file:
                existing_version = json.load(file).get("codebase_version")
            if existing_version not in (None, "v3.0"):
                raise ValueError(
                    f"Refusing to overwrite LeRobot {existing_version} data with v3.0. "
                    "Use a separate output directory."
                )
        shutil.rmtree(output_dir)

    register_codecs()
    with zarr.ZipStore(str(zarr_path), mode="r") as store:
        replay_buffer = ReplayBuffer.create_from_group(zarr.group(store=store))
        absolute_action = _get_absolute_action(replay_buffer)

        state_features = {}
        video_features = {}
        features = {
            "action": _feature_spec(absolute_action, "float32"),
        }
        for raw_key, array in replay_buffer.items():
            if raw_key == "action":
                continue
            if _is_rgb_array(array):
                feature_key = f"observation.images.{raw_key}"
                features[feature_key] = _feature_spec(array, "video")
                video_features[raw_key] = feature_key
            else:
                feature_key = f"observation.state.{raw_key}"
                features[feature_key] = _feature_spec(array, str(array.dtype))
                state_features[raw_key] = feature_key

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=output_dir,
            robot_type="umi",
            features=features,
            use_videos=bool(video_features),
            image_writer_threads=image_writer_threads,
        )
        if _episode_per_file:
            # v2.1 stores exactly one episode in every parquet and video file.
            # A zero size threshold makes the v3 writer roll over at each episode;
            # the v2.1 converter then rewrites the paths and metadata layout.
            dataset.meta.info["data_files_size_in_mb"] = 0
            dataset.meta.info["video_files_size_in_mb"] = 0

        episode_start = 0
        try:
            for episode_index, episode_end in enumerate(replay_buffer.episode_ends):
                episode_end = int(episode_end)
                for frame_index in tqdm(
                        range(episode_start, episode_end),
                        desc=f"episode {episode_index}"):
                    frame = {
                        "action": absolute_action[frame_index],
                        "task": task,
                    }
                    for raw_key, feature_key in state_features.items():
                        frame[feature_key] = np.asarray(replay_buffer[raw_key][frame_index])
                    for raw_key, feature_key in video_features.items():
                        frame[feature_key] = np.asarray(replay_buffer[raw_key][frame_index])
                    dataset.add_frame(frame)
                dataset.save_episode()
                episode_start = episode_end
        finally:
            dataset.finalize()

    schema = {
        "lerobot_dataset_version": "v3.0",
        "format": "umi-absolute-pose-v1",
        "action_key": "action",
        "action_semantics": "absolute_target_pose_axis_angle_and_gripper",
        "pose_frame": "slam_world",
        "position_unit": "meter",
        "rotation_format": "axis_angle",
        "transform_direction": "tf_world_eef",
        "state_features": state_features,
        "video_features": video_features,
    }
    schema_path = output_dir / "meta" / "umi_schema.json"
    with schema_path.open("w") as file:
        json.dump(schema, file, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Dedicated LeRobot v3.0 output directory (do not reuse for v2.1)",
    )
    parser.add_argument("--repo-id", default="local/umi_abs")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--task", default="perform the demonstrated manipulation task")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_zarr_to_lerobot(
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        task=args.task,
        overwrite=args.overwrite,
        image_writer_threads=args.image_writer_threads,
    )


if __name__ == "__main__":
    main()
