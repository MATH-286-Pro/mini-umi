"""Resolve ordered model action fields; only the explicit schema is accepted."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from diffusion_policy.common.obs_schema import PoseField, resolve_observation_fields, resolve_pose_field, validate_sampling


@dataclass(frozen=True)
class ActionSpec:
    fields: tuple[PoseField, ...]
    horizon: int
    latency_steps: float
    down_sample_steps: int
    reference_latency_steps: float

    @property
    def output_dim(self):
        return sum(field.output_dim for field in self.fields)


def resolve_action_spec(shape_meta: Mapping) -> ActionSpec:
    action = shape_meta['action']
    if not isinstance(action, Mapping) or not isinstance(action.get('basic'), Mapping):
        raise ValueError('Explicit action schema requires basic and details')
    basic = action['basic']
    unknown = set(action) - {'basic', 'details', 'shape', 'reference_latency_steps'}
    if unknown:
        raise ValueError(f'Unsupported action options: {sorted(unknown)}')
    unknown = set(basic) - {'horizon', 'latency_steps', 'down_sample_steps'}
    if unknown:
        raise ValueError(f'Unsupported action.basic options: {sorted(unknown)}')
    stride = basic.get('down_sample_steps', 1)
    validate_sampling(config=dict(horizon=basic.get('horizon'), down_sample_steps=stride, latency_steps=basic.get('latency_steps', 0)), label='action.basic')
    details = action.get('details')
    if not isinstance(details, Sequence) or isinstance(details, str) or not details:
        raise ValueError('action.details must be a nonempty list of field mappings')
    resolve_observation_fields(shape_meta=shape_meta)
    fields, seen = [], set()
    for config in details:
        if not isinstance(config, Mapping):
            raise ValueError('Explicit action.details entries require source and type')
        unknown = set(config) - {'source', 'type', 'relative_to', 'shape'}
        if unknown:
            raise ValueError(f'Unsupported action field options: {sorted(unknown)}')
        field = resolve_pose_field(config=config)
        identity = (field.source, field.kind)
        if identity in seen:
            raise ValueError(f'Duplicate action component: {identity}')
        seen.add(identity)
        fields.append(field)
    reference_latency = action.get('reference_latency_steps', 0)
    validate_sampling(config=dict(horizon=1, latency_steps=reference_latency), label='action.reference')
    spec = ActionSpec(fields=tuple(fields), horizon=basic['horizon'], latency_steps=float(basic.get('latency_steps', 0)), down_sample_steps=stride, reference_latency_steps=float(reference_latency))
    if 'shape' in action and list(action['shape']) != [spec.output_dim]:
        raise ValueError(f'action.shape must equal [{spec.output_dim}]')
    return spec
