"""Metadata for deriving UMI and GR00T actions from named state features."""

import json
from pathlib import Path
import re

import numpy as np


def make_state_action_layout(replay_buffer, state_features: dict, video_features: dict):
    """Add xyz+rotvec aliases for GR00T's existing single-column EEF reader."""
    extra_states = {}
    groups = {}
    robot_ids = sorted({
        int(match.group(1)) for key in replay_buffer.keys()
        if (match := re.fullmatch(pattern=r'robot(\d+)_eef_pos', string=key))
    })
    if not robot_ids:
        raise ValueError('State-derived actions require at least one robot EEF pose')
    for robot_id in robot_ids:
        prefix = f'robot{robot_id}'
        pos_key, rot_key = f'{prefix}_eef_pos', f'{prefix}_eef_rot_axis_angle'
        if rot_key not in replay_buffer:
            raise ValueError(f'Missing state field {rot_key}')
        pos = replay_buffer[pos_key][:]
        rot = replay_buffer[rot_key][:]
        if pos.ndim != 2 or pos.shape[-1] != 3 or rot.shape != pos.shape:
            raise ValueError(f'{prefix} requires matching [T,3] position and axis-angle states')
        pose_key = f'{prefix}_eef_pose'
        pose = np.concatenate([pos, rot], axis=-1).astype(dtype=np.float32)
        if pose_key in replay_buffer:
            if not np.array_equal(a1=replay_buffer[pose_key][:], a2=pose):
                raise ValueError(f'{pose_key} conflicts with the derived xyz+rotvec alias')
        extra_states[pose_key] = pose
        state_features[pose_key] = f'observation.state.{pose_key}'
        groups[pose_key] = dict(original_key=state_features[pose_key], start=0, end=6)
        gripper_key = f'{prefix}_gripper_width'
        if gripper_key in replay_buffer:
            if replay_buffer[gripper_key].shape != (len(pos), 1):
                raise ValueError(f'{gripper_key} must have shape [T,1]')
            groups[gripper_key] = dict(original_key=state_features[gripper_key], start=0, end=1)
    modality = dict(
        state=groups,
        action={key: dict(value) for key, value in groups.items()},
        video={key: dict(original_key=value) for key, value in video_features.items()},
        annotation={'human.task_description': dict(original_key='task_index')},
    )
    return extra_states, modality


def write_gr00t_metadata(output_dir: Path, modality: dict) -> None:
    """Emit a v2.1 GR00T mapping and an editable modality selection config."""
    meta = output_dir / 'meta'
    (meta / 'modality.json').write_text(data=json.dumps(obj=modality, indent=2) + '\n')
    config = f'''# Generated UMI state-derived action config for the local GR00T API.
# Edit STATE_KEYS and ACTION_KEYS independently; keep each action's reference pose in STATE_KEYS.
# Each pose key is xyz+rotvec (6D); gripper keys are separate (1D).
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

STATE_KEYS = {list(modality['state'])!r}
ACTION_KEYS = {list(modality['action'])!r}
VIDEO_KEYS = {list(modality['video'])!r}
# Match UMI's default 16-step horizon with stride 3; adjust for your task.
ACTION_INDICES = list(range(0, 16 * 3, 3))

config = {{
    "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
    "action": ModalityConfig(
        delta_indices=ACTION_INDICES,
        modality_keys=ACTION_KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE if key.endswith("_eef_pose") else ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF if key.endswith("_eef_pose") else ActionType.NON_EEF,
                format=ActionFormat.XYZ_ROTVEC if key.endswith("_eef_pose") else ActionFormat.DEFAULT,
                state_key=key,
            ) for key in ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}}
if VIDEO_KEYS:
    config["video"] = ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS)
register_modality_config(config=config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''
    (meta / 'gr00t_config.py').write_text(data=config)
