import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from diffusion_policy.dataset.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.dataset_umi import UmiDatasetBase


class _LeRobotVideoArray:
    """Expose a LeRobot video feature through the slice API used by SequenceSampler."""

    def __init__(self, root: Path, layout, episodes: pd.DataFrame,
            episode_ends: np.ndarray, feature_key: str, shape: tuple):
        self.root = root
        self.layout = layout
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
        video_path, timestamp = self.layout.locate_video_frame(
            episode=episode,
            feature_key=self.feature_key,
            frame_index=index - episode_start,
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
    # Concatenate before converting to pandas: HF matrix extension dtypes can
    # fail dtype equality during pandas concat across v2.1 episode files.
    table = pa.concat_tables(tables=[pq.read_table(source=file) for file in files])
    return table.to_pandas()


def _read_jsonlines(path: Path) -> pd.DataFrame:
    with path.open() as file:
        records = [json.loads(line) for line in file if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")
    return pd.DataFrame.from_records(records)


class _LeRobotLayoutV21:
    def __init__(self, info: dict):
        self.info = info

    def load_episodes(self, dataset_path: Path) -> pd.DataFrame:
        return _read_jsonlines(dataset_path / "meta" / "episodes.jsonl")

    def locate_video_frame(self, episode: pd.Series, feature_key: str,
            frame_index: int) -> tuple[str, float]:
        episode_index = int(episode["episode_index"])
        video_path = self.info["video_path"].format(
            video_key=feature_key,
            episode_chunk=episode_index // 1000,
            episode_index=episode_index,
        )
        return video_path, frame_index / float(self.info["fps"])


class _LeRobotLayoutV30:
    def __init__(self, info: dict):
        self.info = info

    def load_episodes(self, dataset_path: Path) -> pd.DataFrame:
        return _read_parquet_tree(dataset_path / "meta" / "episodes")

    def locate_video_frame(self, episode: pd.Series, feature_key: str,
            frame_index: int) -> tuple[str, float]:
        video_path = self.info["video_path"].format(
            video_key=feature_key,
            chunk_index=int(episode[f"videos/{feature_key}/chunk_index"]),
            file_index=int(episode[f"videos/{feature_key}/file_index"]),
        )
        timestamp = float(episode[f"videos/{feature_key}/from_timestamp"])
        timestamp += frame_index / float(self.info["fps"])
        return video_path, timestamp


def _make_lerobot_layout(info: dict):
    version = info.get("codebase_version")
    layouts = {
        "v2.1": _LeRobotLayoutV21,
        "v3.0": _LeRobotLayoutV30,
    }
    try:
        layout_class = layouts[version]
    except KeyError as error:
        supported = ", ".join(sorted(layouts))
        raise ValueError(
            f"Unsupported LeRobot dataset version {version!r}; expected one of: {supported}"
        ) from error
    return layout_class(info=info)


def _column_to_numpy(frames: pd.DataFrame, info: dict, feature_key: str) -> np.ndarray:
    shape = tuple(info["features"][feature_key]["shape"])
    values = frames[feature_key].to_numpy()
    if shape:
        # Parquet nested lists materialize as object arrays of matrix rows.
        values = np.stack(arrays=[np.asarray(a=value.tolist()) if isinstance(value, np.ndarray) and value.dtype == object else value for value in values]).reshape((len(frames),) + shape)
    return values


def _load_schema(dataset_path: Path, info: dict) -> dict:
    schema_path = dataset_path / "meta" / "umi_schema.json"
    if schema_path.is_file():
        with schema_path.open() as file:
            return json.load(file)

    features = info["features"]
    return {
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


class UmiDatasetLeRobot(UmiDatasetBase):
    """Train UMI policies from absolute-pose LeRobot v2.1 or v3.0 datasets.

    LeRobot remains the storage layer. The shared dataset decodes stored poses
    to homogeneous transforms before sampling and per-field model encoding.
    """

    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            cache_dir=None,
            action_padding: bool=False,
            temporally_independent_normalization: bool=False,
            episode_start_pose_noise_scale: float=0.05,
            seed: int=42,
            val_ratio: float=0.0,
            max_duration=None,
            normalizer_num_workers: int=0,
            video_backend: str="pyav"):

        if cache_dir is not None:
            raise ValueError("cache_dir is not supported by UmiDatasetLeRobot")
        if video_backend != "pyav":
            raise ValueError("The lightweight LeRobot loader currently supports video_backend='pyav' only")

        dataset_path = Path(dataset_path).expanduser().resolve()
        with (dataset_path / "meta" / "info.json").open() as file:
            info = json.load(file)
        layout = _make_lerobot_layout(info=info)
        episodes = layout.load_episodes(dataset_path=dataset_path)
        episodes = episodes.sort_values("episode_index").reset_index(drop=True)
        episode_ends = episodes["length"].to_numpy(dtype=np.int64).cumsum()
        frames = _read_parquet_tree(dataset_path / "data")
        frames = frames.sort_values("index").reset_index(drop=True)
        schema = _load_schema(dataset_path, info)
        data = {}
        # State-derived datasets deliberately omit action from the ReplayBuffer.
        # The shared dataset constructs canonical action poses from selected states.
        for raw_key, feature_key in schema["state_features"].items():
            data[raw_key] = _column_to_numpy(frames, info, feature_key)
        for raw_key, feature_key in schema["video_features"].items():
            feature_shape = tuple(info["features"][feature_key]["shape"])
            data[raw_key] = _LeRobotVideoArray(
                root=dataset_path,
                layout=layout,
                episodes=episodes,
                episode_ends=episode_ends,
                feature_key=feature_key,
                shape=feature_shape,
            )

        replay_buffer = ReplayBuffer({
            "data": data,
            "meta": {"episode_ends": episode_ends},
        })
        
        self.lerobot_info = info
        self.lerobot_schema = schema

        super().__init__(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            action_padding=action_padding,
            temporally_independent_normalization=temporally_independent_normalization,
            episode_start_pose_noise_scale=episode_start_pose_noise_scale,
            seed=seed,
            val_ratio=val_ratio,
            max_duration=max_duration,
            normalizer_num_workers=normalizer_num_workers,
        )
