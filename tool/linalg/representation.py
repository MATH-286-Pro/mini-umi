"""Explicit model representations for column-vector transforms p_world = tf @ p_local.

Transforms have shape [..., 4, 4], positions are metres, rotations are radians.
rot6d_row concatenates the first two rows; rot6d_col concatenates the first
two columns (not a row-major flatten of a 3x2 slice).
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tool.linalg.pose import mat_to_rot6d, rot6d_to_mat


REPRESENTATION_DIMS = {
    'pos3d_xyz': 3,
    'rot4d_quat_wxyz': 4,
    'rot4d_quat_xyzw': 4,
    'rot6d_row': 6,
    'rot6d_col': 6,
    'rot3d_axis_angle': 3,
    'width1d': 1,
}


def encode_tf(tf: np.ndarray, representation: str) -> np.ndarray:
    """Encode one component of [...,4,4] transforms, preserving batch axes."""
    if representation == 'pos3d_xyz':
        return tf[..., :3, 3].copy()
    mat = tf[..., :3, :3]
    if representation == 'rot6d_row':
        return mat_to_rot6d(mat=mat)
    if representation == 'rot6d_col':
        return mat_to_rot6d(mat=np.swapaxes(a=mat, axis1=-1, axis2=-2))
    rotation = Rotation.from_matrix(matrix=mat.reshape(-1, 3, 3))
    if representation == 'rot3d_axis_angle':
        return rotation.as_rotvec().reshape(tf.shape[:-2] + (3,))
    if representation in ('rot4d_quat_xyzw', 'rot4d_quat_wxyz'):
        quat_xyzw = rotation.as_quat(canonical=True).reshape(tf.shape[:-2] + (4,))
        return quat_xyzw if representation == 'rot4d_quat_xyzw' else quat_xyzw[..., [3, 0, 1, 2]]
    raise ValueError(f'Cannot encode tf as {representation!r}')


def decode_rotation(values: np.ndarray, representation: str) -> np.ndarray:
    """Decode [...,D] rotations to [...,3,3], normalizing quaternions/6D."""
    values = np.asarray(a=values)
    if not representation.startswith('rot') or values.shape[-1] != REPRESENTATION_DIMS.get(representation):
        raise ValueError(f'Invalid rotation shape {values.shape} for {representation!r}')
    if not np.all(np.isfinite(values)):
        raise ValueError('Rotation values must be finite')
    if representation in ('rot6d_row', 'rot6d_col'):
        first, second = values[..., :3], values[..., 3:]
        if np.any(np.linalg.norm(x=first, axis=-1) < 1e-12) or np.any(np.linalg.norm(x=np.cross(a=first, b=second), axis=-1) < 1e-12):
            raise ValueError('Degenerate 6D rotation: vectors must be nonzero and nonparallel')
        mat = rot6d_to_mat(rot6d_row=values)
        return mat if representation == 'rot6d_row' else np.swapaxes(a=mat, axis1=-1, axis2=-2)
    if representation == 'rot3d_axis_angle':
        rotation = Rotation.from_rotvec(rotvec=values.reshape(-1, 3))
    else:
        quat_xyzw = values if representation == 'rot4d_quat_xyzw' else values[..., [1, 2, 3, 0]]
        rotation = Rotation.from_quat(quat=quat_xyzw.reshape(-1, 4))
    return rotation.as_matrix().reshape(values.shape[:-1] + (3, 3))


def validate_tf(tf: np.ndarray) -> None:
    """Reject invalid stored rigid transforms instead of silently projecting them."""
    if tf.shape[-2:] != (4, 4) or not np.all(np.isfinite(tf)):
        raise ValueError('tf must be finite with shape [...,4,4]')
    mat = tf[..., :3, :3]
    if not np.allclose(a=tf[..., 3, :], b=[0, 0, 0, 1], atol=1e-5) or not np.allclose(a=np.swapaxes(a=mat, axis1=-1, axis2=-2) @ mat, b=np.eye(N=3), atol=1e-5) or not np.allclose(a=np.linalg.det(a=mat), b=1, atol=1e-5):
        raise ValueError('tf must contain an SO(3) rotation and bottom row [0,0,0,1]')


def interpolate_tf(tf: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Sample a [T,4,4] trajectory at clipped local frame indices using SLERP.

    Translation is linear; rotation follows the shortest SO(3) path. Callers
    must slice to one episode before invoking this function.
    """
    indices = np.clip(a=indices, a_min=0, a_max=len(tf) - 1)
    left = np.floor(indices).astype(int)
    right = np.ceil(indices).astype(int)
    if np.all(left == right):
        return tf[left].copy()
    result = np.broadcast_to(array=np.eye(N=4), shape=indices.shape + (4, 4)).copy()
    fraction = (indices - left)[..., None]
    result[..., :3, 3] = tf[left, :3, 3] * (1 - fraction) + tf[right, :3, 3] * fraction
    # Only materialize the requested interval, not the whole episode.
    start, end = int(left.min()), int(right.max()) + 1
    slerp = Slerp(times=np.arange(start=start, stop=end), rotations=Rotation.from_matrix(matrix=tf[start:end, :3, :3]))
    result[..., :3, :3] = slerp(indices).as_matrix()
    return result
