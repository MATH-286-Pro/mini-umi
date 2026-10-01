"""Episode-level trajectory inference and visualization."""

import copy
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.dataloader_lerobot import UmiDatasetLeRobot
from diffusion_policy.dataset.dataloader_zarr import UmiDatasetZarr
from diffusion_policy.dataset.dataset_umi import UmiDatasetBase
from diffusion_policy.dataset.sampler import sample_values
from diffusion_policy.common.obs_schema import reference_source
from diffusion_policy.common.pose_encoding import decode_action


@dataclass(frozen=True)
class EpisodeTrajectoryResult:
    episode_index: int
    gif_path: Path
    metrics_path: Path
    metrics: dict[str, float]
    inference_frames: int


def resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {resolved} was requested, but CUDA is unavailable")
    return resolved


def load_policy_for_evaluation(
    *,
    checkpoint_path: str | Path,
    weights: str = "auto",
    device: str | torch.device = "auto",
):
    """Load a policy plus its resolved checkpoint config for inference."""
    if weights not in {"auto", "ema", "model"}:
        raise ValueError("weights must be one of: auto, ema, model")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(
        f=checkpoint_path,
        pickle_module=dill,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    cfg = copy.deepcopy(payload["cfg"])
    OmegaConf.resolve(cfg)

    policy_cfg = copy.deepcopy(cfg.policy)
    if "pretrained" in policy_cfg.obs_encoder:
        policy_cfg.obs_encoder.pretrained = False
    if "transforms" in policy_cfg.obs_encoder:
        del policy_cfg.obs_encoder.transforms
    if "image_augmentor" in policy_cfg:
        policy_cfg.image_augmentor = None
    policy = hydra.utils.instantiate(policy_cfg)

    state_dicts = payload["state_dicts"]
    if weights == "auto":
        state_key = "ema_model" if "ema_model" in state_dicts else "model"
    else:
        state_key = {"ema": "ema_model", "model": "model"}[weights]
    if state_key not in state_dicts:
        raise KeyError(
            f"Checkpoint has no {state_key!r}; available state dicts: {sorted(state_dicts)}"
        )
    policy.load_state_dict(state_dict=state_dicts[state_key])
    resolved_device = resolve_device(device=device)
    policy.to(device=resolved_device)
    policy.eval()
    del payload
    gc.collect()
    return policy, cfg, state_key


def load_dataset_for_evaluation(
    *,
    dataset_path: str | Path,
    shape_meta: dict | DictConfig,
    dataset_format: str = "auto",
    cache_dir: Optional[str | Path] = None,
) -> UmiDatasetBase:
    """Load either a LeRobot directory or legacy zipped Zarr dataset."""
    dataset_path = Path(dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    if dataset_format not in {"auto", "lerobot", "zarr"}:
        raise ValueError("dataset_format must be one of: auto, lerobot, zarr")
    if dataset_format == "auto":
        dataset_format = (
            "lerobot"
            if (dataset_path / "meta" / "info.json").is_file()
            else "zarr"
        )
    shape_meta = (
        OmegaConf.to_container(shape_meta, resolve=True)
        if OmegaConf.is_config(shape_meta)
        else copy.deepcopy(shape_meta)
    )
    common = {
        "shape_meta": shape_meta,
        "dataset_path": str(dataset_path),
        "action_padding": False,
        "temporally_independent_normalization": False,
        "episode_start_pose_noise_scale": 0.0,
        "seed": 42,
        "val_ratio": 0.0,
        "max_duration": None,
        "normalizer_num_workers": 0,
    }
    if dataset_format == "lerobot":
        return UmiDatasetLeRobot(
            **common,
            cache_dir=None,
            video_backend="pyav",
        )
    return UmiDatasetZarr(
        **common,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )


def _episode_dataset(dataset: UmiDatasetBase, episode_index: int) -> UmiDatasetBase:
    episode_count = int(dataset.replay_buffer.n_episodes)
    if not 0 <= episode_index < episode_count:
        raise IndexError(
            f"episode_index must be in [0, {episode_count - 1}], got {episode_index}"
        )
    episode_mask = np.zeros(episode_count, dtype=bool)
    episode_mask[episode_index] = True
    episode_dataset = copy.copy(dataset)
    episode_dataset.threadpool_limits_is_applied = False
    episode_dataset.sampler = dataset.sampler.with_episode_mask(episode_mask=episode_mask)
    episode_dataset.sampler.episode_start_pose_noise_scale = 0.0
    if len(episode_dataset) == 0:
        raise ValueError(
            f"Episode {episode_index} is too short for the configured action horizon"
        )
    return episode_dataset


def _sample_indices(
    *,
    sample_count: int,
    frame_stride: int,
    max_frames: Optional[int],
) -> list[int]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    indices = list(range(0, sample_count, frame_stride))
    if indices[-1] != sample_count - 1:
        indices.append(sample_count - 1)
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if len(indices) > max_frames:
            positions = np.linspace(0, len(indices) - 1, num=max_frames, dtype=np.int64)
            indices = [indices[int(position)] for position in positions]
    return indices


def _predict_actions(
    *,
    policy,
    episode_dataset: UmiDatasetBase,
    sample_indices: Iterable[int],
    device: torch.device,
    inference_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    selected_indices = list(sample_indices)
    predictions = []
    targets = []
    current_indices = []
    for start in range(0, len(selected_indices), inference_batch_size):
        batch_indices = selected_indices[start:start + inference_batch_size]
        samples = [episode_dataset[index] for index in batch_indices]
        obs = {
            key: torch.stack([sample["obs"][key] for sample in samples], dim=0)
            for key in samples[0]["obs"]
        }
        obs = dict_apply(obs, lambda value: value.to(device=device, non_blocking=True))
        with torch.inference_mode():
            prediction = policy.predict_action(obs_dict=obs)["action_pred"]
        predictions.append(prediction.detach().cpu().numpy())
        targets.append(
            torch.stack([sample["action"] for sample in samples], dim=0).numpy()
        )
        current_indices.extend(
            int(episode_dataset.sampler.indices[index][0])
            for index in batch_indices
        )
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.asarray(current_indices, dtype=np.int64),
    )


def _action_references(*, dataset: UmiDatasetBase, current_indices: np.ndarray, episode_index: int) -> dict:
    references = {}
    episode_slice = dataset.replay_buffer.get_episode_slice(idx=episode_index)
    episode_end = episode_slice.stop
    if dataset.sampler.max_duration is not None:
        episode_end = min(episode_end, episode_slice.start + int(dataset.sampler.max_duration * 60))
    for field in dataset.action_spec.fields:
        name = field.relative_to
        source = reference_source(relative_to=name)
        if source is None or name in references:
            continue
        if name.endswith('_at_episode_start'):
            references[name] = dataset.pose_data.episode_start_tf[source][episode_index]
        else:
            references[name] = sample_values(
                values=dataset.pose_data.observations[source],
                indices=current_indices + dataset.action_spec.reference_latency_steps,
                start=episode_slice.start,
                end=episode_end,
                is_tf=True,
            )[:, None]
    return references


def _action_tf_world(*, action: np.ndarray, action_spec, references: dict, robot_index: int) -> np.ndarray:
    """Decode the configured position and rotation fields into world transforms."""
    decoded = decode_action(values=action, action_spec=action_spec, references=references)
    tf = decoded.get(f'robot{robot_index}_eef_tf')
    if not isinstance(tf, np.ndarray):
        raise ValueError(f'Trajectory evaluation requires position and rotation actions for robot {robot_index}')
    return tf


def _rotation_angle_error(
    *,
    prediction_tf: np.ndarray,
    target_tf: np.ndarray,
) -> np.ndarray:
    prediction_rotation = prediction_tf[..., :3, :3]
    target_rotation = target_tf[..., :3, :3]
    relative_rotation = prediction_rotation @ np.swapaxes(target_rotation, -1, -2)
    cosine = (np.trace(relative_rotation, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def evaluate_episode_trajectory(
    *,
    policy,
    dataset: UmiDatasetBase,
    episode_index: int,
    output_path: str | Path,
    device: str | torch.device,
    dataset_fps: float,
    robot_index: int = 0,
    frame_stride: int = 10,
    max_frames: Optional[int] = 120,
    inference_batch_size: int = 4,
    gif_fps: float = 10.0,
    seed: int = 42,
) -> EpisodeTrajectoryResult:
    """Run one prerecorded episode and render future-horizon predictions."""
    if dataset_fps <= 0:
        raise ValueError("dataset_fps must be positive")
    if robot_index < 0:
        raise ValueError("robot_index must be non-negative")
    resolved_device = resolve_device(device=device)
    episode_dataset = _episode_dataset(
        dataset=dataset,
        episode_index=episode_index,
    )
    selected_indices = _sample_indices(
        sample_count=len(episode_dataset),
        frame_stride=frame_stride,
        max_frames=max_frames,
    )

    torch.manual_seed(seed=seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    was_training = policy.training
    policy.eval()
    try:
        prediction, target, current_indices = _predict_actions(
            policy=policy,
            episode_dataset=episode_dataset,
            sample_indices=selected_indices,
            device=resolved_device,
            inference_batch_size=inference_batch_size,
        )
    finally:
        policy.train(mode=was_training)

    replay_buffer = dataset.replay_buffer
    episode_slice = replay_buffer.get_episode_slice(idx=episode_index)
    episode_start = int(episode_slice.start)
    episode_end = int(episode_slice.stop)
    relative_frame_indices = current_indices - episode_start

    references = _action_references(
        dataset=episode_dataset, current_indices=current_indices, episode_index=episode_index,
    )
    prediction_tf_world = _action_tf_world(
        action=prediction, action_spec=dataset.action_spec, references=references, robot_index=robot_index,
    )
    target_tf_world = _action_tf_world(
        action=target, action_spec=dataset.action_spec, references=references, robot_index=robot_index,
    )

    prediction_xyz = prediction_tf_world[..., :3, 3]
    target_xyz = target_tf_world[..., :3, 3]
    position_squared_error = np.sum((prediction_xyz - target_xyz) ** 2, axis=-1)
    frame_position_rmse = np.sqrt(position_squared_error.mean(axis=-1))
    rotation_error = _rotation_angle_error(
        prediction_tf=prediction_tf_world,
        target_tf=target_tf_world,
    )
    robot_columns = []
    width_columns = []
    offset = 0
    for field in dataset.action_spec.fields:
        columns = list(range(offset, offset + field.output_dim))
        if field.robot_id == robot_index:
            robot_columns.extend(columns)
            if field.kind == 'width':
                width_columns.extend(columns)
        offset += field.output_dim
    action_error = prediction[..., robot_columns] - target[..., robot_columns]
    metrics = {
        "position_rmse_m": float(np.sqrt(position_squared_error.mean())),
        "rotation_rmse_rad": float(np.sqrt(np.square(rotation_error).mean())),
        "action_mse": float(np.square(action_error).mean()),
    }
    if width_columns:
        metrics['gripper_mae_m'] = float(np.abs(prediction[..., width_columns] - target[..., width_columns]).mean())

    ground_truth_tf_world = dataset.pose_data.actions[f'robot{robot_index}_eef_tf'][episode_start:episode_end]
    output_path = Path(output_path)
    from tool.plot import save_trajectory_comparison_gif

    gif_path = save_trajectory_comparison_gif(
        output_path=output_path,
        ground_truth_tf_world=ground_truth_tf_world,
        prediction_tf_world=prediction_tf_world,
        target_tf_world=target_tf_world,
        frame_indices=relative_frame_indices,
        dataset_fps=dataset_fps,
        gif_fps=gif_fps,
        episode_index=episode_index,
        frame_position_rmse=frame_position_rmse,
    )
    metrics_path = gif_path.with_suffix(".json")
    record = {
        "episode_index": episode_index,
        "robot_index": robot_index,
        "dataset_fps": dataset_fps,
        "inference_frames": len(selected_indices),
        "frame_stride": frame_stride,
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return EpisodeTrajectoryResult(
        episode_index=episode_index,
        gif_path=gif_path,
        metrics_path=metrics_path,
        metrics=metrics,
        inference_frames=len(selected_indices),
    )


def evaluate_configured_episodes(
    *,
    policy,
    dataset: UmiDatasetBase,
    evaluation_cfg: DictConfig,
    device: str | torch.device,
    output_dir: str | Path,
    epoch: int,
) -> list[EpisodeTrajectoryResult]:
    """Run the configured episode set from a training workspace."""
    epoch_directory = Path(output_dir) / "evaluation" / f"epoch_{epoch:04d}"
    results = []
    for episode_index in evaluation_cfg.episode_indices:
        episode_index = int(episode_index)
        results.append(
            evaluate_episode_trajectory(
                policy=policy,
                dataset=dataset,
                episode_index=episode_index,
                output_path=epoch_directory / f"episode_{episode_index:03d}.gif",
                device=device,
                dataset_fps=float(evaluation_cfg.dataset_fps),
                robot_index=int(evaluation_cfg.robot_index),
                frame_stride=int(evaluation_cfg.frame_stride),
                max_frames=(
                    None
                    if evaluation_cfg.max_frames is None
                    else int(evaluation_cfg.max_frames)
                ),
                inference_batch_size=int(evaluation_cfg.inference_batch_size),
                gif_fps=float(evaluation_cfg.gif_fps),
                seed=int(evaluation_cfg.seed),
            )
        )
    return results
