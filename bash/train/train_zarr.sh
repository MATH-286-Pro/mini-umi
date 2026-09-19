#!/bin/bash

# WANDB__EXECUTABLE=$(which python) python train.py
# export WANDB_DISABLED=true

uv run train.py \
  --config-name=train/unet_timm_umi \
  dataset=zarr \
  dataset.dataset_path=./data/zarr/sample/pushing_2024_05_29_huy.zarr.zip \
  hydra.run.dir=data/policy/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name} \
  dataloader.batch_size=32 \
  val_dataloader.batch_size=32

