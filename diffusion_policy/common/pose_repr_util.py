import numpy as np

from tool.linalg import convert_pose_mat_rep


def compute_relative_pose(pos, rot, base_pos, base_rot_mat,
                          rot_transformer_to_mat,
                          rot_transformer_to_target,
                          backward=False,
                          delta=False):
    if not backward:
        # forward pass
        if not delta:
            output_pos = pos if base_pos is None else pos - base_pos
            output_rot = rot_transformer_to_target.forward(
                rot_transformer_to_mat.forward(rot) @ np.linalg.inv(base_rot_mat))
            return output_pos, output_rot
        else:
            all_pos = np.concatenate([base_pos[None,...], pos], axis=0)
            output_pos = np.diff(all_pos, axis=0)
            
            rot_mat = rot_transformer_to_mat.forward(rot)
            all_rot_mat = np.concatenate([base_rot_mat[None,...], rot_mat], axis=0)
            prev_rot = np.linalg.inv(all_rot_mat[:-1])
            curr_rot = all_rot_mat[1:]
            rot = np.matmul(curr_rot, prev_rot)
            output_rot = rot_transformer_to_target.forward(rot)
            return output_pos, output_rot
            
    else:
        # backward pass
        if not delta:
            output_pos = pos if base_pos is None else pos + base_pos
            output_rot = rot_transformer_to_mat.inverse(
                rot_transformer_to_target.inverse(rot) @ base_rot_mat)
            return output_pos, output_rot
        else:
            output_pos = np.cumsum(pos, axis=0) + base_pos
            
            rot_mat = rot_transformer_to_target.inverse(rot)
            output_rot_mat = np.zeros_like(rot_mat)
            curr_rot = base_rot_mat
            for i in range(len(rot_mat)):
                curr_rot = rot_mat[i] @ curr_rot
                output_rot_mat[i] = curr_rot
            output_rot = rot_transformer_to_mat.inverse(rot)
            return output_pos, output_rot

