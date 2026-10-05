"""Print all array paths and shapes in Zarr v2 directories or ZIP stores.

Usage: python -m tool.read.read_zarr DATASET.zarr.zip [DATASET.zarr ...]
"""

import argparse
import json
from pathlib import Path

import zarr


def read_zarr(dataset_path: str | Path) -> None:
    """Inspect metadata only, including arrays using custom image codecs."""
    path = Path(dataset_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    store = (
        zarr.DirectoryStore(path=str(path))
        if path.is_dir()
        else zarr.ZipStore(path=str(path), mode="r")
    )
    try:
        # Reading .zarray directly avoids instantiating or decoding custom codecs.
        keys = sorted(key for key in store.keys() if key == ".zarray" or key.endswith("/.zarray"))
        if not keys and ".zgroup" not in store:
            raise ValueError(f"Not a Zarr v2 dataset: {path}")
        print(f"Dataset: {path}")
        for key in keys:
            metadata = json.loads(s=store[key])
            field = key.removesuffix(".zarray").rstrip("/") or "/"
            print(f"{field}: shape={tuple(metadata['shape'])}")
        if not keys:
            print("(no array fields)")
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_paths", nargs="+", help="Zarr v2 directory or ZIP paths")
    args = parser.parse_args()
    for dataset_path in args.dataset_paths:
        read_zarr(dataset_path=dataset_path)


if __name__ == "__main__":
    main()
