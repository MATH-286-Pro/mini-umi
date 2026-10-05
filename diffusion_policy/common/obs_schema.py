"""Shared semantic fields for explicitly configured observations and actions."""

from collections.abc import Mapping
from dataclasses import dataclass
import copy
import math
import re

from tool.linalg.representation import REPRESENTATION_DIMS


@dataclass(frozen=True)
class PoseField:
    source: str
    type: str
    relative_to: str
    robot_id: int
    kind: str
    output_dim: int


def reference_source(relative_to: str) -> str | None:
    if relative_to == 'world':
        return None
    match = re.fullmatch(pattern=r'(robot(?:0|[1-9]\d*)_eef)_at_(obs_end|episode_start)', string=relative_to)
    if match is None:
        raise ValueError(f'Invalid relative_to: {relative_to!r}')
    return f'{match.group(1)}_tf'


def resolve_pose_field(config: Mapping) -> PoseField:
    removed = {'representation', 'raw_shape', 'rotation_rep', 'axis'} & set(config)
    if removed:
        raise ValueError(f'Unsupported field options: {sorted(removed)}; use type')
    source = config.get('source')
    field_type = config.get('type')
    match = re.fullmatch(pattern=r'robot(0|[1-9]\d*)_(eef_tf|gripper_width)', string=source) if isinstance(source, str) else None
    if match is None or not isinstance(field_type, str) or field_type not in REPRESENTATION_DIMS:
        raise ValueError(f'Invalid source/type: {source!r}, {field_type!r}')
    kind = 'width' if field_type == 'width1d' else ('pos' if field_type == 'pos3d_xyz' else 'rot')
    if (kind == 'width') != (match.group(2) == 'gripper_width'):
        raise ValueError(f'{source} cannot use type {field_type}')
    relative_to = config.get('relative_to', 'world' if kind == 'width' else None)
    if not isinstance(relative_to, str):
        raise ValueError(f'{source} requires explicit relative_to')
    reference_source(relative_to=relative_to)
    if kind == 'width' and relative_to != 'world':
        raise ValueError('width1d has no relative pose reference')
    dim = REPRESENTATION_DIMS[field_type]
    if 'shape' in config and list(config['shape']) != [dim]:
        raise ValueError(f'{field_type} requires shape [{dim}]')
    return PoseField(source=source, type=field_type, relative_to=relative_to, robot_id=int(match.group(1)), kind=kind, output_dim=dim)


def validate_sampling(config: Mapping, label: str) -> None:
    for key in ('horizon', 'down_sample_steps'):
        value = config.get(key, None if key == 'horizon' else 1)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{label}.{key} must be a positive integer')
    latency = config.get('latency_steps', 0)
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise ValueError(f'{label}.latency_steps must be finite and nonnegative')


def is_rgb_type(type_name: str) -> bool:
    """RGB is explicitly three channels; H and W must be positive integers."""
    return isinstance(type_name, str) and re.fullmatch(pattern=r'rgb3x[1-9]\d*x[1-9]\d*d', string=type_name) is not None


def type_shape(type_name: str) -> tuple[int, ...]:
    """Return CHW for rgb3xHxWd, or the vector shape for a pose/width type."""
    if is_rgb_type(type_name=type_name):
        return tuple(int(value) for value in type_name[3:-1].split('x'))
    if isinstance(type_name, str) and type_name in REPRESENTATION_DIMS:
        return (REPRESENTATION_DIMS[type_name],)
    raise ValueError(f'Unsupported type: {type_name!r}; use rgb3xHxWd or an explicit pose/width type')


def resolve_observation_fields(shape_meta: Mapping) -> dict[str, PoseField]:
    fields = {}
    for key, config in shape_meta['obs'].items():
        removed = {'representation', 'raw_shape', 'rotation_rep', 'axis', 'down_sampling_steps'} & set(config)
        if removed:
            raise ValueError(f'{key}: unsupported options {sorted(removed)}; use type and down_sample_steps')
        validate_sampling(config=config, label=key)
        shape = type_shape(type_name=config.get('type'))
        if 'shape' in config and tuple(config['shape']) != shape:
            raise ValueError(f'{key}: type {config["type"]} requires shape {list(shape)}')
        if is_rgb_type(type_name=config['type']):
            if config.get('latency_steps', 0) != 0:
                raise ValueError('RGB latency_steps must be zero')
            continue
        fields[key] = resolve_pose_field(config=config)
    return fields


def resolve_shape_meta(shape_meta: Mapping) -> dict:
    """Infer all observation shapes from type without modifying caller config."""
    resolve_observation_fields(shape_meta=shape_meta)
    result = copy.deepcopy(x=dict(shape_meta))
    result['obs'] = {key: dict(value) for key, value in shape_meta['obs'].items()}
    for config in result['obs'].values():
        config['shape'] = list(type_shape(type_name=config['type']))
    return result
