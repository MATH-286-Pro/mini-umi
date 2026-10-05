"""Shared training/inference field encoding and prediction decoding."""

import numpy as np

from tool.linalg.representation import decode_rotation, encode_tf


def encode_field(values: np.ndarray, field, references: dict) -> np.ndarray:
    if field.kind == 'width':
        return values.copy()
    tf = values
    if field.relative_to != 'world':
        tf = np.linalg.inv(a=references[field.relative_to]) @ tf
    return encode_tf(tf=tf, representation=field.type)


def decode_action(values: np.ndarray, action_spec, references: dict) -> dict:
    """Decode unnormalized [...,D] predictions into absolute pose components.

    references maps each configured relative_to name to its world-from-reference
    [4,4] tf (or a broadcastable batch). Results are keyed by canonical source;
    pose entries contain pos/rot matrices independently, so partial action
    specifications never fabricate an unpredicted position or rotation.
    """
    values = np.asarray(a=values)
    if values.shape[-1] != action_spec.output_dim or not np.all(np.isfinite(values)):
        raise ValueError('Action values must be finite and match the configured dimension')
    result, offset = {}, 0
    for field in action_spec.fields:
        component = values[..., offset:offset + field.output_dim]
        offset += field.output_dim
        if field.kind == 'width':
            result[field.source] = component.copy()
            continue
        base_tf = np.eye(N=4) if field.relative_to == 'world' else references[field.relative_to]
        pose = result.setdefault(field.source, {})
        if field.kind == 'pos':
            pose['pos'] = (base_tf[..., :3, :3] @ component[..., None])[..., 0] + base_tf[..., :3, 3]
        else:
            pose['rot'] = base_tf[..., :3, :3] @ decode_rotation(values=component, representation=field.type)
    for source, pose in list(result.items()):
        if isinstance(pose, dict) and 'pos' in pose and 'rot' in pose:
            tf = np.broadcast_to(array=np.eye(N=4), shape=values.shape[:-1] + (4, 4)).copy()
            tf[..., :3, 3], tf[..., :3, :3] = pose['pos'], pose['rot']
            result[source] = tf
    return result
