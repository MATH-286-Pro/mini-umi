import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

from diffusion_policy.dataset.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.dataloader_zarr import UmiDatasetZarr


class _LeRobotVideoArray:
    """Expose a LeRobot video feature through the slice API used by SequenceSampler."""

    def __init__(self, root: Path, info: dict, episodes: pd.DataFrame,
            episode_ends: np.ndarray, feature_key: str, shape: tuple):
        self.root = root
        self.info = info
        self.episodes = episodes
        self.episode_ends = episode_ends
        self.feature_key = feature_key
        self.shape = (int(episode_ends[-1]),) + tuple(shape)

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
        elif np.isscalar(index):
            indices = [int(index)]
        else:
            indices = np.asarray(index).reshape(-1).tolist()

        frames = [self._load_frame(frame_index) for frame_index in indices]
        result = np.stack(frames, axis=0)
        if np.isscalar(index):
            return result[0]
        return result

    def __len__(self):
        return self.shape[0]

    def _load_frame(self, index: int) -> np.ndarray:
        episode_index = int(np.searchsorted(self.episode_ends, index, side="right"))
        episode_start = 0 if episode_index == 0 else int(self.episode_ends[episode_index - 1])
        episode = self.episodes.iloc[episode_index]
        timestamp = float(episode[f"videos/{self.feature_key}/from_timestamp"])
        timestamp += (index - episode_start) / float(self.info["fps"])
        video_path = self.info["video_path"].format(
            video_key=self.feature_key,
            chunk_index=int(episode[f"videos/{self.feature_key}/chunk_index"]),
            file_index=int(episode[f"videos/{self.feature_key}/file_index"]),
        )
        return _decode_video_frame(self.root / video_path, timestamp)


def _decode_video_frame(path: Path, timestamp: float) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target_pts = int(timestamp / float(stream.time_base))
        container.seek(max(target_pts, 0), stream=stream, backward=True)
        closest = None
        closest_distance = float("inf")
        for frame in container.decode(stream):
            frame_timestamp = float(frame.pts * frame.time_base)
            distance = abs(frame_timestamp - timestamp)
            if distance < closest_distance:
                closest = frame.to_ndarray(format="rgb24")
                closest_distance = distance
            if frame_timestamp >= timestamp:
                break
    if closest is None:
        raise IndexError(f"Could not decode timestamp {timestamp:.6f} from {path}")
    return closest


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {path}")
    return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)


def _column_to_numpy(frames: pd.DataFrame, info: dict, feature_key: str) -> np.ndarray:
    shape = tuple(info["features"][feature_key]["shape"])
    values = frames[feature_key].to_numpy()
    if shape:
        values = np.stack(values).reshape((len(frames),) + shape)
    return values


def _load_schema(dataset_path: Path, info: dict) -> dict:
    schema_path = dataset_path / "meta" / "umi_schema.json"
    if schema_path.is_file():
        with schema_path.open() as file:
            return json.load(file)

    features = info["features"]
    return {
        "action_key": "action",
        "state_features": {
            key.removeprefix("observation.state."): key
            for key in features
            if key.startswith("observation.state.")
        },
        "video_features": {
            key.removeprefix("observation.images."): key
            for key, feature in features.items()
            if feature["dtype"] == "video"
        },
    }


class UmiDatasetLeRobot(UmiDatasetZarr):
    """Train UMI policies from absolute-pose LeRobot v3 datasets.

    LeRobot remains the storage layer. This adapter reconstructs the legacy
    ReplayBuffer interface so the existing timestamp interpolation, horizon
    sampling, SE(3) relative conversion, and normalization stay identical.
    """

    def __init__(self, shape_meta: dict, dataset_path: str, repo_id: str,
            cache_dir=None, pose_repr: dict={}, action_padding: bool=False,
            temporally_independent_normalization: bool=False,
            repeat_frame_prob: float=0.0, seed: int=42,
            val_ratio: float=0.0, max_duration=None, image_transform=None,
            normalizer_num_workers: int=0,
            video_backend: str="pyav"):
        if cache_dir is not None:
            raise ValueError("cache_dir is only supported by UmiDatasetZarr")
        if video_backend != "pyav":
            raise ValueError("The lightweight LeRobot loader currently supports video_backend='pyav' only")
        dataset_path = Path(dataset_path).expanduser().resolve()
        with (dataset_path / "meta" / "info.json").open() as file:
            info = json.load(file)
        episodes = _read_parquet_tree(dataset_path / "meta" / "episodes")
        episodes = episodes.sort_values("episode_index").reset_index(drop=True)
        episode_ends = episodes["length"].to_numpy(dtype=np.int64).cumsum()
        frames = _read_parquet_tree(dataset_path / "data")
        frames = frames.sort_values("index").reset_index(drop=True)
        schema = _load_schema(dataset_path, info)
        data = {
            "action": _column_to_numpy(frames, info, schema["action_key"]),
        }
        for raw_key, feature_key in schema["state_features"].items():
            data[raw_key] = _column_to_numpy(frames, info, feature_key)
        for raw_key, feature_key in schema["video_features"].items():
            feature_shape = tuple(info["features"][feature_key]["shape"])
            data[raw_key] = _LeRobotVideoArray(
                dataset_path, info, episodes, episode_ends, feature_key, feature_shape)

        replay_buffer = ReplayBuffer({
            "data": data,
            "meta": {"episode_ends": episode_ends},
        })
        self.lerobot_info = info
        self.lerobot_schema = schema
        super().__init__(
            shape_meta=shape_meta,
            dataset_path=str(dataset_path),
            pose_repr=pose_repr,
            action_padding=action_padding,
            temporally_independent_normalization=temporally_independent_normalization,
            repeat_frame_prob=repeat_frame_prob,
            seed=seed,
            val_ratio=val_ratio,
            max_duration=max_duration,
            image_transform=image_transform,
            normalizer_num_workers=normalizer_num_workers,
            replay_buffer=replay_buffer,
        )
