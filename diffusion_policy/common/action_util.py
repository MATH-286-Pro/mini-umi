import torch
import torch.nn.functional as F


def action_mse(pred_action: torch.Tensor, gt_action: torch.Tensor, action_spec) -> dict:
    """Measure ordered action fields without inventing missing gripper targets."""
    indices = {'pos': [], 'rot': [], 'width': []}
    offset = 0
    for field in action_spec.fields:
        indices[field.kind].extend(range(offset, offset + field.output_dim))
        offset += field.output_dim
    if pred_action.shape != gt_action.shape or pred_action.shape[-1] != offset:
        raise ValueError('Action tensor shapes do not match configured robot dimensions')
    result = {'action_mse_error': F.mse_loss(input=pred_action, target=gt_action)}
    for name, columns in indices.items():
        if columns:
            result[f'action_mse_error_{name}'] = F.mse_loss(
                input=pred_action[..., columns], target=gt_action[..., columns])
    return result
