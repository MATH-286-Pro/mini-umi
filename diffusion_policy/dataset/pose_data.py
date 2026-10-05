"""Decode state poses to tf arrays and derive future-state training targets."""

from dataclasses import dataclass

import numpy as np

from diffusion_policy.common.obs_schema import reference_source
from tool.linalg.pose import pose_to_mat
from tool.linalg.representation import validate_tf


@dataclass
class PoseData:
    observations: dict
    actions: dict
    episode_ends: np.ndarray
    episode_start_tf: dict


def load_pose_data(replay_buffer, observation_fields, action_spec) -> PoseData:
    """Read tf or pos+rotvec states; action fields select those same trajectories."""
    ends = np.asarray(a=replay_buffer.episode_ends[:], dtype=np.int64)
    if ends.ndim != 1 or not len(ends) or np.any(np.diff(a=np.concatenate(([0], ends))) <= 0):
        raise ValueError('episode_ends must describe nonempty episodes')
    n_frames = int(ends[-1])
    observations, actions = {}, {}

    def read_array(key, shape):
        if key not in replay_buffer:
            raise ValueError(f'Missing dataset source: {key}')
        value = np.asarray(a=replay_buffer[key][:], dtype=np.float64)
        if value.shape != (n_frames, *shape) or not np.all(np.isfinite(value)):
            raise ValueError(f'{key} must be finite with shape {(n_frames, *shape)}')
        return value

    def read_state(source):
        if source in observations:
            return observations[source]
        if source.endswith('_tf'):
            if source in replay_buffer:
                tf = read_array(key=source, shape=(4, 4))
                validate_tf(tf=tf)
            else:
                prefix = source.removesuffix('_tf')
                pos = read_array(key=f'{prefix}_pos', shape=(3,))
                rot = read_array(key=f'{prefix}_rot_axis_angle', shape=(3,))
                tf = pose_to_mat(pose=np.concatenate([pos, rot], axis=-1))
            observations[source] = tf
        else:
            observations[source] = read_array(key=source, shape=(1,))
        return observations[source]

    for field in [*observation_fields.values(), *action_spec.fields]:
        reference = reference_source(relative_to=field.relative_to)
        if reference is not None:
            read_state(source=reference)
    for field in observation_fields.values():
        read_state(source=field.source)

    for field in action_spec.fields:
        actions[field.source] = read_state(source=field.source)

    starts = np.concatenate(([0], ends[:-1]))
    episode_start_tf = {}
    for source, tf in observations.items():
        if not source.endswith('_tf'):
            continue
        boundary_key = source.removesuffix('_eef_tf') + '_demo_start_pose'
        if boundary_key in replay_buffer:
            boundary = read_array(key=boundary_key, shape=(6,))
            episode_start_tf[source] = pose_to_mat(pose=boundary[starts])
        else:
            episode_start_tf[source] = tf[starts]
    return PoseData(observations=observations, actions=actions, episode_ends=ends, episode_start_tf=episode_start_tf)
