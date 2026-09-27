#!/usr/bin/env python3
"""Convert an absolute-pose UMI Zarr replay buffer to LeRobot Dataset v2.1.

This is intentionally a separate command from the v3.0 converter.  A v2.1
dataset uses one parquet/video per episode and JSONL metadata, whereas v3.0
consolidates episodes into larger parquet/video files and parquet metadata.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

try:
    from scripts.convert_umi_zarr_to_lerobot_v3 import convert_zarr_to_lerobot
except ModuleNotFoundError:
    from convert_umi_zarr_to_lerobot_v3 import convert_zarr_to_lerobot


V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_jsonlines(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, default=_json_value, sort_keys=True) + "\n")


def _load_episode_metadata(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError("v3 staging dataset has no episode metadata")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _nested_episode_stats(row: dict) -> dict:
    stats = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        _, feature_key, statistic = key.split("/", maxsplit=2)
        stats.setdefault(feature_key, {})[statistic] = _json_value(value)
    return stats


def _move_unique_episode_files(root: Path, episodes: pd.DataFrame, video_keys: list[str]) -> None:
    data_locations = set()
    video_locations = {key: set() for key in video_keys}
    for row in episodes.to_dict(orient="records"):
        data_location = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if data_location in data_locations:
            raise RuntimeError("v2.1 staging invariant failed: multiple episodes share a data file")
        data_locations.add(data_location)
        for video_key in video_keys:
            location = (
                int(row[f"videos/{video_key}/chunk_index"]),
                int(row[f"videos/{video_key}/file_index"]),
            )
            if location in video_locations[video_key]:
                raise RuntimeError(
                    f"v2.1 staging invariant failed: multiple episodes share video {video_key}"
                )
            video_locations[video_key].add(location)

    for row in episodes.to_dict(orient="records"):
        episode_index = int(row["episode_index"])
        episode_chunk = episode_index // 1000
        source_data = root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"
        target_data = root / V21_DATA_PATH.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        target_data.parent.mkdir(parents=True, exist_ok=True)
        source_data.replace(target_data)

        for video_key in video_keys:
            source_video = root / "videos" / video_key / f"chunk-{int(row[f'videos/{video_key}/chunk_index']):03d}" / f"file-{int(row[f'videos/{video_key}/file_index']):03d}.mp4"
            target_video = root / V21_VIDEO_PATH.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            target_video.parent.mkdir(parents=True, exist_ok=True)
            source_video.replace(target_video)

    # All v3 video files now live at their v2.1 paths. Remove the empty v3
    # camera-first directory tree so the output contains only one layout.
    for video_key in video_keys:
        shutil.rmtree(root / "videos" / video_key)


def _convert_staging_v3_to_v21(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    with info_path.open() as file:
        info = json.load(file)
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Expected a LeRobot v3.0 staging dataset")

    episodes = _load_episode_metadata(root=root)
    video_keys = sorted(
        key for key, feature in info["features"].items() if feature["dtype"] == "video"
    )
    _move_unique_episode_files(root=root, episodes=episodes, video_keys=video_keys)

    episode_records = []
    episode_stats_records = []
    for row in episodes.to_dict(orient="records"):
        episode_index = int(row["episode_index"])
        episode_records.append({
            "episode_index": episode_index,
            "tasks": list(row["tasks"]),
            "length": int(row["length"]),
        })
        episode_stats_records.append({
            "episode_index": episode_index,
            "stats": _nested_episode_stats(row=row),
        })

    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_records = [
        {"task_index": int(row.task_index), "task": str(task)}
        for task, row in tasks.iterrows()
    ]
    _write_jsonlines(root / "meta" / "episodes.jsonl", episode_records)
    _write_jsonlines(root / "meta" / "episodes_stats.jsonl", episode_stats_records)
    _write_jsonlines(root / "meta" / "tasks.jsonl", task_records)

    shutil.rmtree(root / "meta" / "episodes")
    (root / "meta" / "tasks.parquet").unlink()
    info.update({
        "codebase_version": "v2.1",
        "data_path": V21_DATA_PATH,
        "video_path": V21_VIDEO_PATH if video_keys else None,
        "total_chunks": (int(info["total_episodes"]) + 999) // 1000,
    })
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    with info_path.open("w") as file:
        json.dump(info, file, indent=4)

    schema_path = root / "meta" / "umi_schema.json"
    with schema_path.open() as file:
        schema = json.load(file)
    schema["lerobot_dataset_version"] = "v2.1"
    with schema_path.open("w") as file:
        json.dump(schema, file, indent=2, sort_keys=True)


def convert_zarr_to_lerobot_v21(
        zarr_path: Path,
        output_dir: Path,
        repo_id: str,
        fps: int,
        task: str,
        overwrite: bool=False,
        image_writer_threads: int=4) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and overwrite:
        info_path = output_dir / "meta" / "info.json"
        if info_path.is_file():
            with info_path.open() as file:
                existing_version = json.load(file).get("codebase_version")
            if existing_version not in (None, "v2.1"):
                raise ValueError(
                    f"Refusing to overwrite LeRobot {existing_version} data with v2.1. "
                    "Use a separate output directory."
                )
        shutil.rmtree(output_dir)
    convert_zarr_to_lerobot(
        zarr_path=zarr_path,
        output_dir=output_dir,
        repo_id=repo_id,
        fps=fps,
        task=task,
        overwrite=False,
        image_writer_threads=image_writer_threads,
        _episode_per_file=True,
    )
    _convert_staging_v3_to_v21(root=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Dedicated LeRobot v2.1 output directory (do not reuse for v3.0)",
    )
    parser.add_argument("--repo-id", default="local/umi_abs_v21")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--task", default="perform the demonstrated manipulation task")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_zarr_to_lerobot_v21(
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
