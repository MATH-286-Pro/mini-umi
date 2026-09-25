#!/usr/bin/env bash
set -eu

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

DIR="${1:-./data/dataset/zarr/sample/pushing_2024_05_29_huy.zarr.zip}"
NAME="$(basename "$(dirname "$DIR")")"
OUTPUT_DIR="./data/dataset/lerobot_v3.0/${NAME}"

uv run python -m scripts.convert_umi_zarr_to_lerobot_v3 \
    "$DIR" \
    "$OUTPUT_DIR" \
    --repo-id local/umi_abs \
    --fps 60 \
    --task "place the cup at the demonstrated target"
