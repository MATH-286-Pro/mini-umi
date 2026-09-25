import os
from datetime import datetime
from pathlib import Path
import shutil
from typing import Optional

from filelock import FileLock
import zarr

from diffusion_policy.dataset.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.dataset.dataset_umi import UmiDatasetBase
from diffusion_policy.dataset.replay_buffer import ReplayBuffer


register_codecs()


def _load_replay_buffer_zarr(dataset_path: str, cache_dir: Optional[str]) -> ReplayBuffer:
    if cache_dir is None:
        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            return ReplayBuffer.copy_from_store(
                src_store=zip_store,
                store=zarr.MemoryStore(),
            )

    mod_time = os.path.getmtime(dataset_path)
    stamp = datetime.fromtimestamp(mod_time).isoformat()
    stem_name = os.path.basename(dataset_path).split('.')[0]
    cache_name = '_'.join([stem_name, stamp])
    cache_dir_path = Path(os.path.expanduser(cache_dir))
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir_path / f'{cache_name}.zarr.mdb'
    lock_path = cache_dir_path / f'{cache_name}.lock'

    print('Acquiring lock on cache.')
    with FileLock(lock_path):
        if not cache_path.exists():
            try:
                with zarr.LMDBStore(
                        str(cache_path), writemap=True, metasync=False, sync=False,
                        map_async=True, lock=False) as lmdb_store:
                    with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                        print(f'Copying data to {cache_path}')
                        ReplayBuffer.copy_from_store(
                            src_store=zip_store,
                            store=lmdb_store,
                        )
                print('Cache written to disk!')
            except Exception:
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                raise

    store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
    return ReplayBuffer.create_from_group(group=zarr.group(store))


class UmiDatasetZarr(UmiDatasetBase):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            cache_dir: Optional[str]=None,
            action_padding: bool=False,
            temporally_independent_normalization: bool=False,
            repeat_frame_prob: float=0.0,
            seed: int=42,
            val_ratio: float=0.0,
            max_duration: Optional[float]=None,
            image_transform=None,
            normalizer_num_workers: int=32):
        replay_buffer = _load_replay_buffer_zarr(dataset_path, cache_dir)
        super().__init__(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            action_padding=action_padding,
            temporally_independent_normalization=temporally_independent_normalization,
            repeat_frame_prob=repeat_frame_prob,
            seed=seed,
            val_ratio=val_ratio,
            max_duration=max_duration,
            image_transform=image_transform,
            normalizer_num_workers=normalizer_num_workers,
        )
