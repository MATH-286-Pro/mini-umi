#!/usr/bin/env python3
"""Convert an absolute-pose UMI Zarr replay buffer to LeRobot Dataset v3.0.

For LeRobot v2.1 output use ``convert_umi_zarr_to_lerobot_v21.py``.  The two
formats deliberately have separate entry points so a destination can never
silently change format after a dependency upgrade.
"""

import argparse
from contextlib import contextmanager
import json
import logging
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import zarr
from tqdm import tqdm

from diffusion_policy.dataset.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.dataset.replay_buffer import ReplayBuffer
from tool.linalg.pose import pose_to_mat
from tool.linalg.representation import validate_tf


@contextmanager
def _quiet_encoder_output():
    """Capture native encoder stderr; replay its tail only if encoding fails.

    SVT-AV1 and FFmpeg can bypass Python logging and write directly to fd 2.
    Keep this process-wide redirection limited to synchronous save/finalize
    calls so the outer tqdm continues writing to the original terminal.
    """
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+b") as captured:
        original_stderr = os.dup(2)
        failed = False
        try:
            os.dup2(captured.fileno(), 2)
            try:
                yield
            except BaseException:
                failed = True
                raise
        finally:
            try:
                sys.stderr.flush()
            finally:
                os.dup2(original_stderr, 2)
                os.close(original_stderr)
            if failed:
                captured.seek(0, os.SEEK_END)
                captured.seek(max(0, captured.tell() - 65536))
                with os.fdopen(os.dup(2), mode="wb") as restored:
                    shutil.copyfileobj(fsrc=captured, fdst=restored)


@contextmanager
def _quiet_dataset_progress():
    """Hide nested HF bars and routine logs, restoring caller settings on exit."""
    from datasets.utils import logging as datasets_logging

    progress_was_disabled = datasets_logging.is_progress_bar_enabled() is False
    previous_logging_disable = logging.root.manager.disable
    datasets_logging.disable_progress_bar()
    logging.disable(level=max(previous_logging_disable, logging.INFO))
    try:
        yield
    finally:
        logging.disable(level=previous_logging_disable)
        if not progress_was_disabled:
            datasets_logging.enable_progress_bar()


def _make_tf_states(replay_buffer):
    """Replace EEF pose aliases with [T,4,4] tf_world_eef (column vectors).

    Translation is in metres; input rotvec is in radians. No frame change is
    applied: p_world = tf_world_eef @ p_eef, with last row [0,0,0,1].
    """
    robot_ids = sorted({
        int(match.group(1)) for key in replay_buffer.keys()
        if (match := re.fullmatch(pattern=r'robot(\d+)_eef_(?:pos|tf|pose|rot_.*)', string=key))
    })
    extra_states, replaced_keys = {}, set()
    for robot_id in robot_ids:
        prefix = f'robot{robot_id}_eef'
        pos_key, rot_key, tf_key = f'{prefix}_pos', f'{prefix}_rot_axis_angle', f'{prefix}_tf'
        tf = None
        if pos_key in replay_buffer and rot_key in replay_buffer:
            pos = np.asarray(a=replay_buffer[pos_key][:], dtype=np.float32)
            rot = np.asarray(a=replay_buffer[rot_key][:], dtype=np.float32)
            if pos.shape != (replay_buffer.n_steps, 3) or rot.shape != pos.shape:
                raise ValueError(f'{prefix} requires matching [T,3] position and axis-angle states')
            if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
                raise ValueError(f'{prefix} pose must be finite')
            tf = pose_to_mat(pose=np.concatenate((pos, rot), axis=-1))
        if tf_key in replay_buffer:
            stored_tf = np.asarray(a=replay_buffer[tf_key][:], dtype=np.float32)
            if stored_tf.shape != (replay_buffer.n_steps, 4, 4):
                raise ValueError(f'{tf_key} must have shape [T,4,4]')
            validate_tf(tf=stored_tf)
            if tf is not None and not np.allclose(a=tf, b=stored_tf, atol=1e-5):
                raise ValueError(f'{tf_key} conflicts with position and axis-angle states')
            tf = stored_tf
        if tf is None:
            raise ValueError(f'{prefix} requires pos + rot_axis_angle or tf')
        extra_states[tf_key] = tf
        replaced_keys.update(key for key in replay_buffer.keys() if key in (pos_key, tf_key, f'{prefix}_pose') or key.startswith(f'{prefix}_rot_'))
    return extra_states, replaced_keys


def _is_rgb_array(array) -> bool:
    return array.ndim == 4 and array.shape[-1] in (1, 3, 4) and array.dtype == np.uint8


def _feature_spec(array, dtype: str) -> dict:
    return {
        "dtype": dtype,
        "shape": tuple(array.shape[1:]),
        "names": None,
    }


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

    register_codecs(verbose=False)
    with zarr.ZipStore(str(zarr_path), mode="r") as store:
        replay_buffer = ReplayBuffer.create_from_group(zarr.group(store=store))
        # LeRobot action is a placeholder; training targets come from future states.
        placeholder_action = np.zeros(shape=(1,), dtype=np.float32)

        extra_states, replaced_keys = _make_tf_states(replay_buffer=replay_buffer)
        if not extra_states:
            raise ValueError("State-derived actions require at least one robot EEF pose")
        state_features = {}
        video_features = {}
        features = {
            "action": {"dtype": "float32", "shape": (1,), "names": None},
        }
        for raw_key, array in replay_buffer.items():
            if raw_key == "action" or raw_key in replaced_keys:
                continue
            if _is_rgb_array(array):
                feature_key = f"observation.images.{raw_key}"
                features[feature_key] = _feature_spec(array, "video")
                video_features[raw_key] = feature_key
            else:
                feature_key = f"observation.state.{raw_key}"
                features[feature_key] = _feature_spec(array, str(array.dtype))
                state_features[raw_key] = feature_key

        for raw_key, array in extra_states.items():
            state_features[raw_key] = f"observation.state.{raw_key}"
            features[state_features[raw_key]] = _feature_spec(array=array, dtype="float32")

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
        with _quiet_dataset_progress(), tqdm(total=replay_buffer.n_episodes, desc="Converting LeRobot", unit="episode", dynamic_ncols=True) as progress:
            try:
                for episode_end in replay_buffer.episode_ends:
                    episode_end = int(episode_end)
                    for frame_index in range(episode_start, episode_end):
                        frame = {
                            "action": placeholder_action.copy(),
                            "task": task,
                        }
                        for raw_key, feature_key in state_features.items():
                            array = extra_states[raw_key] if raw_key in extra_states else replay_buffer[raw_key]
                            frame[feature_key] = np.asarray(a=array[frame_index])
                        for raw_key, feature_key in video_features.items():
                            frame[feature_key] = np.asarray(replay_buffer[raw_key][frame_index])
                        dataset.add_frame(frame=frame)
                    with _quiet_encoder_output():
                        dataset.save_episode()
                    episode_start = episode_end
                    progress.update(n=1)
            finally:
                with _quiet_encoder_output():
                    dataset.finalize()

    schema = {
        "lerobot_dataset_version": "v3.0",
        "format": "umi-tf-v1",
        "action_semantics": "future_state_tf_and_gripper",
        "pose_frame": "slam_world",
        "position_unit": "meter",
        "rotation_format": "rotation_matrix",
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
