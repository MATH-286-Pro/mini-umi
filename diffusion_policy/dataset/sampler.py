"""Episode-safe sampling of canonical poses and explicitly configured fields."""

import copy
import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.action_schema import resolve_action_spec
from diffusion_policy.common.obs_schema import resolve_observation_fields, reference_source
from diffusion_policy.common.pose_encoding import encode_field
from tool.linalg.representation import interpolate_tf


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(shape=n_episodes, dtype=bool)
    if val_ratio <= 0 or n_episodes < 2:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(a=n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def sample_values(values, indices, start, end, is_tf):
    indices = np.clip(a=indices, a_min=start, a_max=end - 1)
    if is_tf:
        return interpolate_tf(tf=values[start:end], indices=indices - start)
    left, right = np.floor(indices).astype(int), np.ceil(indices).astype(int)
    fraction = (indices - left)[..., None]
    return values[left] * (1 - fraction) + values[right] * fraction


class SequenceSampler:
    def __init__(self, shape_meta, pose_data, rgb_arrays, episode_mask=None,
                 action_padding=False, max_duration=None, episode_start_pose_noise_scale=0.05):
        self.shape_meta = shape_meta
        self.pose_data = pose_data
        self.rgb_arrays = rgb_arrays
        self.observation_fields = resolve_observation_fields(shape_meta=shape_meta)
        self.action_spec = resolve_action_spec(shape_meta=shape_meta)
        self.action_padding = action_padding
        self.max_duration = max_duration
        self.episode_start_pose_noise_scale = episode_start_pose_noise_scale
        self.ignore_rgb_is_applied = False
        self.set_episode_mask(episode_mask=episode_mask)

    def set_episode_mask(self, episode_mask):
        self.indices = []
        self.episode_indices = []
        start = 0
        for episode, stop in enumerate(self.pose_data.episode_ends):
            end = int(stop)
            # Preserve the dataset's existing max_duration convention (60 Hz).
            if self.max_duration is not None:
                end = min(end, start + int(self.max_duration * 60))
            if episode_mask is None or episode_mask[episode]:
                for current in range(start, end):
                    last_action = current + self.action_spec.latency_steps + (self.action_spec.horizon - 1) * self.action_spec.down_sample_steps
                    if not self.action_padding and last_action > end - 1:
                        continue
                    self.indices.append((current, start, end))
                    self.episode_indices.append(episode)
            start = int(stop)

    def with_episode_mask(self, episode_mask):
        sampler = copy.copy(x=self)
        sampler.set_episode_mask(episode_mask=episode_mask)
        return sampler

    def __len__(self):
        return len(self.indices)

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply

    def sample_sequence(self, idx):
        current, start, end = self.indices[idx]
        episode = self.episode_indices[idx]
        references = {}
        for field in [*self.observation_fields.values(), *self.action_spec.fields]:
            name = field.relative_to
            source = reference_source(relative_to=name)
            if source is None or name in references:
                continue
            if name.endswith('_at_episode_start'):
                tf = self.pose_data.episode_start_tf[source][episode].copy()
                if self.episode_start_pose_noise_scale:
                    # Perturb translation and rotation vector once per shared reference.
                    noise = np.random.normal(loc=0, scale=self.episode_start_pose_noise_scale, size=6)
                    tf[:3, 3] += noise[:3]
                    rotvec = Rotation.from_matrix(matrix=tf[:3, :3]).as_rotvec() + noise[3:]
                    tf[:3, :3] = Rotation.from_rotvec(rotvec=rotvec).as_matrix()
                references[name] = tf
            else:
                indices = np.asarray(a=[current + self.action_spec.reference_latency_steps])
                references[name] = sample_values(values=self.pose_data.observations[source], indices=indices, start=start, end=end, is_tf=True)[0]

        obs = {}
        for key, field in self.observation_fields.items():
            config = self.shape_meta['obs'][key]
            indices = current + config.get('latency_steps', 0) - np.arange(start=config['horizon'] - 1, stop=-1, step=-1) * config.get('down_sample_steps', 1)
            values = sample_values(values=self.pose_data.observations[field.source], indices=indices, start=start, end=end, is_tf=field.kind != 'width')
            obs[key] = encode_field(values=values, field=field, references=references).astype(np.float32)
        if not self.ignore_rgb_is_applied:
            for key, array in self.rgb_arrays.items():
                config = self.shape_meta['obs'][key]
                horizon, stride = config['horizon'], config.get('down_sample_steps', 1)
                # Keep lazy video/Zarr slice access; pad history with the first sampled image.
                valid = min(horizon, (current - start) // stride + 1)
                frames = array[current - (valid - 1) * stride:current + 1:stride]
                channels, height, width = config['shape']
                if frames.shape[1:] != (height, width, channels):
                    raise ValueError(f'{key}: type {config["type"]} requires stored HWC shape {(height, width, channels)}, got {frames.shape[1:]}')
                if valid < horizon:
                    frames = np.concatenate([np.repeat(a=frames[:1], repeats=horizon - valid, axis=0), frames], axis=0)
                obs[key] = np.moveaxis(a=frames, source=-1, destination=1).astype(np.float32) / 255.0

        indices = current + self.action_spec.latency_steps + np.arange(stop=self.action_spec.horizon) * self.action_spec.down_sample_steps
        actions = []
        for field in self.action_spec.fields:
            values = sample_values(values=self.pose_data.actions[field.source], indices=indices, start=start, end=end, is_tf=field.kind != 'width')
            actions.append(encode_field(values=values, field=field, references=references))
        return {'obs': obs, 'action': np.concatenate(actions, axis=-1).astype(np.float32)}
