from typing import Optional
from collections.abc import Sequence
import re
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
from diffusion_policy.dataset.replay_buffer import ReplayBuffer

def get_robot_ids_from_keys(keys) -> list[int]:
    """Return all robot IDs with EEF positions, sorted numerically."""
    return sorted({
        int(match.group(1))
        for key in keys
        if (match := re.fullmatch(pattern=r'robot(\d+)_eef_pos', string=key))
    })

def get_robot_ids(shape_meta: dict) -> list[int]:
    """Return configured observation robot IDs in numeric order."""
    return get_robot_ids_from_keys(keys=shape_meta['obs'])

def _parse_robot_names(names, field: str) -> list[int]:
    if not isinstance(names, Sequence) or isinstance(names, str):
        raise ValueError(f"{field} must be a list of robot names")
    ids = []
    for name in names:
        match = re.fullmatch(pattern=r'robot(0|[1-9]\d*)', string=name) if isinstance(name, str) else None
        if match is None:
            raise ValueError(f"Invalid {field} name: {name!r}; expected robot0, robot1, ...")
        ids.append(int(match.group(1)))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{field} must be nonempty and contain no duplicates")
    return ids


def get_action_robot_ids(shape_meta: dict) -> list[int]:
    """Resolve selected action blocks in YAML order, defaulting to observation IDs."""
    names = shape_meta['action'].get('robot_ids')
    ids = get_robot_ids(shape_meta=shape_meta) if names is None else _parse_robot_names(names=names, field='action.robot_ids')
    if not ids:
        raise ValueError("No action robots configured")
    for robot_id in ids:
        for suffix in ('eef_pos', 'eef_rot_axis_angle', 'gripper_width'):
            if f'robot{robot_id}_{suffix}' not in shape_meta['obs']:
                raise ValueError(f"Action robot{robot_id} requires observation robot{robot_id}_{suffix}")
    if list(shape_meta['action']['shape']) != [10 * len(ids)]:
        raise ValueError("action.shape must equal [10 * number of action robots]")
    return ids


def select_replay_action(replay_buffer: ReplayBuffer, shape_meta: dict) -> np.ndarray:
    """Select absolute 7D action blocks for either storage backend.

    source_robot_ids describes the stored action, while robot_ids describes the
    training output. Legacy full-state datasets use numeric robot order. Subset
    or differently ordered stored actions require an explicit source_robot_ids.
    Never infer the stored layout from the selected observation/action subset.
    """
    selected_ids = get_action_robot_ids(shape_meta=shape_meta)
    if 'action' not in replay_buffer:
        # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
        return np.concatenate([
            replay_buffer[f'robot{robot_id}_{suffix}'][:]
            for robot_id in selected_ids
            for suffix in ('eef_pos', 'eef_rot_axis_angle', 'gripper_width')
        ], axis=-1)

    raw_action = replay_buffer['action'][:]
    source_names = shape_meta['action'].get('source_robot_ids')
    source_ids = (
        get_robot_ids_from_keys(keys=replay_buffer.keys()) if source_names is None
        else _parse_robot_names(names=source_names, field='action.source_robot_ids')
    )
    if not source_ids or raw_action.ndim != 2 or raw_action.shape[-1] != 7 * len(source_ids):
        raise ValueError(
            f"Stored action shape {raw_action.shape} does not match 7D blocks for source robots {source_ids}. "
            "Set action.source_robot_ids to the exact robot order stored in the dataset."
        )
    missing_ids = set(selected_ids) - set(source_ids)
    if missing_ids:
        raise ValueError(f"Selected action robots {sorted(missing_ids)} are absent from action.source_robot_ids")
    blocks = [source_ids.index(robot_id) for robot_id in selected_ids]
    return raw_action.reshape(raw_action.shape[0], len(source_ids), 7)[:, blocks, :].reshape(raw_action.shape[0], 7 * len(selected_ids))


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class SequenceSampler:
    def __init__(self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,
        lowdim_keys: list,
        key_horizon: dict,
        key_latency_steps: dict,
        key_down_sample_steps: dict,
        episode_mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
        max_duration: Optional[float]=None
    ):
        episode_ends = replay_buffer.episode_ends[:]

        # create indices, including (current_idx, start_idx, end_idx)
        indices = list()
        for i in range(len(episode_ends)):
            if episode_mask is not None and not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)
            for current_idx in range(start_idx, end_idx):
                if not action_padding and end_idx < current_idx + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1:
                    continue
                indices.append((current_idx, start_idx, end_idx))
        
        # load low_dim to memory and keep rgb as compressed zarr array
        self.replay_buffer = dict()
        self.robot_ids = get_robot_ids(shape_meta=shape_meta)
        self.num_robot = len(self.robot_ids)
        self.action_robot_ids = get_action_robot_ids(shape_meta=shape_meta)
        for key in lowdim_keys:
            if key.endswith('pos_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                self.replay_buffer[key] = replay_buffer[key[:-4]][:, list(axis)]
            elif key.endswith('quat_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                # HACK for hybrid abs/relative proprioception
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_quat(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith('axis_angle_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_rotvec(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            else:
                self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        
        
        self.replay_buffer['action'] = select_replay_action(
            replay_buffer=replay_buffer, shape_meta=shape_meta)

        self.action_padding = action_padding
        self.indices = indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        
        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer

    def __len__(self):
        return len(self.indices)
    
    def sample_sequence(self, idx):
        current_idx, start_idx, end_idx = self.indices[idx]

        result = dict()

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        # observation
        for key in obs_keys:
            input_arr = self.replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_latency_steps = self.key_latency_steps[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            
            if key in self.rgb_keys:
                assert this_latency_steps == 0
                num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
                slice_start = current_idx - (num_valid - 1) * this_downsample_steps

                output = input_arr[slice_start: current_idx + 1: this_downsample_steps]
                assert output.shape[0] == num_valid
                
                # solve padding
                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
            else:
                idx_with_latency = np.array(
                    [current_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
                    dtype=np.float32)
                idx_with_latency = idx_with_latency[::-1]
                idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                if 'rot' in key:
                    # rotation
                    rot_preprocess, rot_postprocess = None, None
                    if key.endswith('quat'):
                        rot_preprocess = st.Rotation.from_quat
                        rot_postprocess = st.Rotation.as_quat
                    elif key.endswith('axis_angle'):
                        rot_preprocess = st.Rotation.from_rotvec
                        rot_postprocess = st.Rotation.as_rotvec
                    else:
                        raise NotImplementedError
                    slerp = st.Slerp(
                        times=np.arange(interpolation_start, interpolation_end),
                        rotations=rot_preprocess(input_arr[interpolation_start: interpolation_end]))
                    output = rot_postprocess(slerp(idx_with_latency))
                else:
                    interp = si.interp1d(
                        x=np.arange(interpolation_start, interpolation_end),
                        y=input_arr[interpolation_start: interpolation_end],
                        axis=0, assume_sorted=True)
                    output = interp(idx_with_latency)
                
            result[key] = output

        # aciton
        input_arr = self.replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_latency_steps = self.key_latency_steps['action']
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_idx: slice_end: action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output

        return result
    
    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply
