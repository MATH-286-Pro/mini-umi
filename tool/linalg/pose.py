"""NumPy pose and rotation representation conversions."""

import numpy as np
import scipy.spatial.transform as st


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """Convert ``[..., x, y, z, rx, ry, rz]`` axis-angle poses to ``[..., 4, 4]`` transforms."""
    pos = pose[..., :3]
    rot = st.Rotation.from_rotvec(pose[..., 3:]).as_matrix()
    tf = np.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype)
    tf[..., :3, :3] = rot
    tf[..., :3, 3] = pos
    tf[..., 3, 3] = 1
    return tf


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """Flatten the first two rows of ``[..., 3, 3]`` rotation matrices to row-based 6D rotations."""
    batch_shape = mat.shape[:-2]
    rot6d_row = mat[..., :2, :].copy().reshape(batch_shape + (6,))
    return rot6d_row


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


def rot6d_to_mat(rot6d_row: np.ndarray) -> np.ndarray:
    """Convert row-based ``[..., 6]`` rotations to ``[..., 3, 3]`` rotation matrices."""
    first_row = _normalize(rot6d_row[..., :3])
    second_row = rot6d_row[..., 3:]
    second_row = _normalize(
        second_row
        - np.sum(first_row * second_row, axis=-1, keepdims=True) * first_row
    )
    third_row = np.cross(first_row, second_row, axis=-1)
    return np.stack((first_row, second_row, third_row), axis=-2)


def mat_to_pose10d(tf: np.ndarray) -> np.ndarray:
    """Convert ``[..., 4, 4]`` transforms to position plus row-based rotation-6D.

    The returned shape is ``[..., 9]``. The legacy name refers to the common
    action representation after a tenth gripper dimension is appended.
    """
    pos = tf[..., :3, 3]
    rot6d_row = mat_to_rot6d(tf[..., :3, :3])
    return np.concatenate([pos, rot6d_row], axis=-1)


def convert_pose_mat_rep(
    pose_mat: np.ndarray,
    base_pose_mat: np.ndarray,
    pose_rep: str = "abs",
    backward: bool = False,
) -> np.ndarray:
    """Convert batched object transforms between absolute and base-relative forms.

    ``pose_mat`` has shape ``[..., 4, 4]`` and maps object coordinates into the
    reference frame. ``base_pose_mat`` has shape ``[4, 4]`` and maps the base
    frame into that same reference frame. ``relative`` uses proper homogeneous
    composition; ``rel`` intentionally preserves the legacy training behavior.
    """
    tf = pose_mat
    base_tf = base_pose_mat
    if not backward:
        if pose_rep == "abs":
            return tf
        if pose_rep == "rel":
            # Legacy implementation retained for checkpoint/data compatibility.
            pos = tf[..., :3, 3] - base_tf[:3, 3]
            rot = tf[..., :3, :3] @ np.linalg.inv(base_tf[:3, :3])
            out_tf = np.copy(tf)
            out_tf[..., :3, :3] = rot
            out_tf[..., :3, 3] = pos
            return out_tf
        if pose_rep == "relative":
            return np.linalg.inv(base_tf) @ tf
        if pose_rep == "delta":
            all_pos = np.concatenate([base_tf[None, :3, 3], tf[..., :3, 3]], axis=0)
            out_pos = np.diff(all_pos, axis=0)
            all_rot = np.concatenate([base_tf[None, :3, :3], tf[..., :3, :3]], axis=0)
            out_rot = tf[..., :3, :3] @ np.linalg.inv(all_rot[:-1])
            out_tf = np.copy(tf)
            out_tf[..., :3, :3] = out_rot
            out_tf[..., :3, 3] = out_pos
            return out_tf
        raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    if pose_rep == "abs":
        return tf
    if pose_rep == "rel":
        # Legacy implementation retained for checkpoint/data compatibility.
        pos = tf[..., :3, 3] + base_tf[:3, 3]
        rot = tf[..., :3, :3] @ base_tf[:3, :3]
        out_tf = np.copy(tf)
        out_tf[..., :3, :3] = rot
        out_tf[..., :3, 3] = pos
        return out_tf
    if pose_rep == "relative":
        return base_tf @ tf
    if pose_rep == "delta":
        out_pos = np.cumsum(tf[..., :3, 3], axis=0) + base_tf[:3, 3]
        out_rot = np.zeros_like(tf[..., :3, :3])
        current_tf = base_tf
        for idx in range(len(tf)):
            current_tf = tf[idx] @ current_tf
            out_rot[idx] = current_tf[:3, :3]
        out_tf = np.copy(tf)
        out_tf[..., :3, :3] = out_rot
        out_tf[..., :3, 3] = out_pos
        return out_tf
    raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")


def compute_relative_pose(
    pos: np.ndarray,
    rot: np.ndarray,
    base_pos: np.ndarray | None,
    base_rot_mat: np.ndarray,
    rot_transformer_to_mat,
    rot_transformer_to_target,
    backward: bool = False,
    delta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert position and rotation sequences to or from a base-relative form.

    ``rot_transformer_to_mat`` converts the input rotation representation to
    matrices. ``rot_transformer_to_target`` converts matrices to the output
    representation. With ``delta=True``, the first element is relative to the
    supplied base and subsequent elements are relative to their predecessor.
    """
    if not backward:
        if not delta:
            output_pos = pos if base_pos is None else pos - base_pos
            output_rot = rot_transformer_to_target.forward(
                rot_transformer_to_mat.forward(rot) @ np.linalg.inv(base_rot_mat)
            )
            return output_pos, output_rot

        all_pos = np.concatenate([base_pos[None, ...], pos], axis=0)
        output_pos = np.diff(all_pos, axis=0)
        rot_mat = rot_transformer_to_mat.forward(rot)
        all_rot_mat = np.concatenate([base_rot_mat[None, ...], rot_mat], axis=0)
        output_rot_mat = all_rot_mat[1:] @ np.linalg.inv(all_rot_mat[:-1])
        output_rot = rot_transformer_to_target.forward(output_rot_mat)
        return output_pos, output_rot

    if not delta:
        output_pos = pos if base_pos is None else pos + base_pos
        output_rot = rot_transformer_to_mat.inverse(
            rot_transformer_to_target.inverse(rot) @ base_rot_mat
        )
        return output_pos, output_rot

    output_pos = np.cumsum(pos, axis=0) + base_pos
    rot_mat = rot_transformer_to_target.inverse(rot)
    output_rot_mat = np.zeros_like(rot_mat)
    current_rot_mat = base_rot_mat
    for idx in range(len(rot_mat)):
        current_rot_mat = rot_mat[idx] @ current_rot_mat
        output_rot_mat[idx] = current_rot_mat
    output_rot = rot_transformer_to_mat.inverse(output_rot_mat)
    return output_pos, output_rot
