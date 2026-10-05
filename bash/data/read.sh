#!/usr/bin/env bash
set -eu

if [[ $# -ne 2 ]]; then
    echo "Usage: bash $0 {zarr|lerobot} DATASET_PATH" >&2
    exit 1
fi

FORMAT="$1"
DIR="$2"

case "$FORMAT" in
    zarr)
        READER="tool.read.read_zarr"
        ;;
    lerobot)
        READER="tool.read.read_lerobot"
        ;;
    *)
        echo "Unsupported format: $FORMAT. Choose zarr or lerobot." >&2
        exit 1
        ;;
esac

uv run python -m "$READER" "$DIR"
