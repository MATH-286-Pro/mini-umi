import copy
from typing import Dict

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from diffusion_policy.dataset.normalization import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.action_schema import resolve_action_spec
from diffusion_policy.common.obs_schema import resolve_shape_meta, resolve_observation_fields, is_rgb_type
from diffusion_policy.dataset.pose_data import load_pose_data
from diffusion_policy.dataset.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class UmiDatasetBase(BaseDataset):
    def __init__(self, shape_meta, replay_buffer, action_padding=False,
                 temporally_independent_normalization=False,
                 episode_start_pose_noise_scale=0.05, seed=42, val_ratio=0.0,
                 max_duration=None, normalizer_num_workers=32):
        self.shape_meta = resolve_shape_meta(shape_meta=shape_meta)
        self.action_spec = resolve_action_spec(shape_meta=self.shape_meta)
        self.observation_fields = resolve_observation_fields(shape_meta=self.shape_meta)
        self.lowdim_keys = list(self.observation_fields)
        self.rgb_keys = [key for key, config in self.shape_meta['obs'].items() if is_rgb_type(type_name=config['type'])]
        self.replay_buffer = replay_buffer
        self.pose_data = load_pose_data(replay_buffer=replay_buffer, observation_fields=self.observation_fields, action_spec=self.action_spec)
        self.val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        if not np.isfinite(episode_start_pose_noise_scale) or episode_start_pose_noise_scale < 0:
            raise ValueError('episode_start_pose_noise_scale must be finite and nonnegative')
        self.sampler = SequenceSampler(shape_meta=self.shape_meta, pose_data=self.pose_data,
            rgb_arrays={key: replay_buffer[key] for key in self.rgb_keys}, episode_mask=~self.val_mask,
            action_padding=action_padding, max_duration=max_duration,
            episode_start_pose_noise_scale=episode_start_pose_noise_scale)
        self.temporally_independent_normalization = temporally_independent_normalization
        self.normalizer_num_workers = normalizer_num_workers
        self.threadpool_limits_is_applied = False

    def get_validation_dataset(self):
        val_set = copy.copy(x=self)
        val_set.sampler = self.sampler.with_episode_mask(episode_mask=self.val_mask)
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        if not len(self):
            raise ValueError('Cannot fit a normalizer on an empty dataset')
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        previous_ignore_rgb = self.sampler.ignore_rgb_is_applied
        self.sampler.ignore_rgb(apply=True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=self.normalizer_num_workers,
        )
        try:
            for batch in tqdm(iterable=dataloader, desc='iterating dataset to get normalization'):
                for key in self.lowdim_keys:
                    data_cache[key].append(copy.deepcopy(x=batch['obs'][key]))
                data_cache['action'].append(copy.deepcopy(x=batch['action']))
        finally:
            self.sampler.ignore_rgb(apply=previous_ignore_rgb)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        action_normalizers = list()
        offset = 0
        for field in self.action_spec.fields:
            stat = array_to_stats(arr=data_cache['action'][..., offset:offset + field.output_dim])
            # pos / gripper use range normalization; rot preserves its representation.
            factory = get_identity_normalizer_from_stat if field.kind == 'rot' else get_range_normalizer_from_stat
            action_normalizers.append(factory(stat=stat))
            offset += field.output_dim

        normalizer['action'] = concatenate_normalizer(normalizers=action_normalizers)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(arr=data_cache[key])

            factory = get_identity_normalizer_from_stat if self.observation_fields[key].kind == 'rot' else get_range_normalizer_from_stat
            this_normalizer = factory(stat=stat)
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(limits=1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx=idx)
        return dict_apply(x=data, func=lambda value: torch.from_numpy(value))
