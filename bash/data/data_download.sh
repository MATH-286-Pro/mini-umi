#!/bin/bash

DATA_DIR="$(pwd)/data/dataset/zarr/sample"

# 下载文件到 ./data/sample
wget -P "$DATA_DIR" \
  "https://real.stanford.edu/umi-on-legs/pushing_2024_05_29_huy.zarr.zip"