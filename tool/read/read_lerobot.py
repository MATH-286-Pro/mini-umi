"""Print LeRobot feature names and per-frame shapes from meta/info.json.

Usage: python -m tool.read.read_lerobot DATASET_ROOT [DATASET_ROOT ...]
"""

import argparse
import json
from pathlib import Path


def read_lerobot(dataset_path: str | Path) -> None:
    """Inspect feature metadata without loading Parquet files or videos."""
    path = Path(dataset_path).expanduser()
    with (path / "meta" / "info.json").open(encoding="utf-8") as file:
        info = json.load(fp=file)
    print(f"Dataset: {path} (per-frame shapes)")
    for field, feature in sorted(info["features"].items()):
        print(f"{field}: shape={tuple(feature['shape'])}")
    if not info["features"]:
        print("(no feature fields)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_paths", nargs="+", help="LeRobot dataset root paths")
    args = parser.parse_args()
    for dataset_path in args.dataset_paths:
        read_lerobot(dataset_path=dataset_path)


if __name__ == "__main__":
    main()
