"""Export a policy as ONNX observation encoder and single-step denoiser.

The scheduler stays outside ONNX so runtimes can choose the denoising step count.
"""

import argparse
import copy
import json
from pathlib import Path

import dill
import hydra
import torch
from omegaconf import OmegaConf

from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy


class ObservationEncoderOnnx(torch.nn.Module):
    def __init__(self, policy, keys):
        super().__init__()
        self.normalizer  = policy.normalizer
        self.obs_encoder = policy.obs_encoder
        self.keys = tuple(keys)

    def forward(self, *values):
        observations = dict(zip(self.keys, values, strict=True))
        return self.obs_encoder(self.normalizer.normalize(observations))


class UnetDenoiserOnnx(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, noisy_action, timestep, observation_condition):
        return self.model(noisy_action, timestep, local_cond=None,
                          global_cond=observation_condition)


class TransformerDenoiserOnnx(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, noisy_action, timestep, observation_condition):
        return self.model(noisy_action, timestep, observation_condition)


def _json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def load_policy(checkpoint, weights):
    with checkpoint.open("rb") as stream:
        payload = torch.load(stream, pickle_module=dill, map_location="cpu",
                             weights_only=False)
    cfg = copy.deepcopy(payload["cfg"])
    OmegaConf.resolve(cfg)
    if "pretrained" in cfg.policy.obs_encoder:
        cfg.policy.obs_encoder.pretrained = False

    if "transforms" in cfg.policy.obs_encoder:
        del cfg.policy.obs_encoder.transforms

    # cfg.policy.obs_encoder.pop("transforms", None) # 删除不需要的 cfg 字段 功能同上

    policy = hydra.utils.instantiate(cfg.policy)
    states = payload["state_dicts"]
    state_key = ("ema_model" if "ema_model" in states else "model") if weights == "auto" else (
        "ema_model" if weights == "ema" else "model")
    if state_key not in states:
        raise KeyError(f"Checkpoint has no {state_key!r}; available: {sorted(states)}")
    policy.load_state_dict(states[state_key])
    policy.eval()
    return policy, state_key


def export_policy_onnx(checkpoint, output_dir, weights="auto", opset=17):
    policy, state_key = load_policy(Path(checkpoint), weights)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    obs_encoder = policy.obs_encoder
    keys = sorted(obs_encoder.rgb_keys + obs_encoder.low_dim_keys)
    shape_meta = obs_encoder.shape_meta["obs"]
    examples = tuple(torch.zeros(
        (1, int(shape_meta[key]["horizon"]), *shape_meta[key]["shape"]),
        dtype=torch.float32) for key in keys)

    obs_encoder_module = ObservationEncoderOnnx(policy, keys)
    torch.onnx.export(
        model = obs_encoder_module, 
        args  = examples, 
        f     = output_dir / "obs_encoder.onnx",
        input_names=keys, 
        output_names=["observation_condition"],
        dynamic_axes={**{key: {0: "batch"} for key in keys},
                      "observation_condition": {0: "batch"}},
        opset_version=opset, 
        do_constant_folding=True, 
        dynamo=False)
    
    with torch.no_grad():
        condition = obs_encoder_module(*examples)

    noisy_action = torch.zeros((1, policy.action_horizon, policy.action_dim))
    timestep = torch.tensor(0, dtype=torch.int64)


    # 判断策略类型
    match policy:
        case DiffusionUnetTimmPolicy():
            architecture = "unet"
            denoiser = UnetDenoiserOnnx(policy.model)

        case DiffusionTransformerTimmPolicy():
            architecture = "transformer"
            denoiser = TransformerDenoiserOnnx(policy.model)

        case _:
            raise TypeError(f"Unsupported policy type: {type(policy).__name__}")

    # 导出 onnx
    torch.onnx.export(
        model = denoiser, 
        args  = (noisy_action, timestep, condition), 
        f     = output_dir / "denoiser.onnx",
        input_names=[
            "noisy_action", 
            "timestep", 
            "observation_condition"
            ],
        output_names=[
            "model_output"
            ],
        dynamic_axes={
            "noisy_action":          {0: "batch"},
            "observation_condition": {0: "batch"},
            "model_output":          {0: "batch"}
            },
        opset_version=opset, 
        do_constant_folding=True, 
        dynamo=False
        )

    action_params = policy.normalizer.params_dict["action"]

    metadata = {
        "format_version": 1, "architecture": architecture, "weights": state_key,
        "files": {"observation_encoder": "observation_encoder.onnx",
                  "denoiser": "denoiser.onnx"},
        "observations": [{"name": key,
                          "shape": [int(shape_meta[key]["horizon"]),
                                    *map(int, shape_meta[key]["shape"])],
                          "type": shape_meta[key].get("type", "low_dim")}
                         for key in keys],
        "condition_shape": list(condition.shape[1:]),
        "action_shape": [policy.action_horizon, policy.action_dim],
        "action_normalizer": {
            "scale": action_params["scale"].detach().reshape(-1).tolist(),
            "offset": action_params["offset"].detach().reshape(-1).tolist(),
            "formula": "normalized = raw * scale + offset"},
        "scheduler": {
            "class": type(policy.noise_scheduler).__name__,
            "config": _json_value(dict(policy.noise_scheduler.config)),
            "default_num_inference_steps": int(policy.num_inference_steps),
            "step_kwargs": _json_value(policy.kwargs),
            "runtime_num_inference_steps": True},
        "onnx": {"opset": opset, "dynamic_batch": True}}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--weights", choices=("auto", "model", "ema"), default="auto")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    metadata = export_policy_onnx(
        args.checkpoint, 
        args.output_dir,
        args.weights, 
        args.opset
    )
    
    print(f"Exported {metadata['architecture']} policy to {args.output_dir} "
          f"using {metadata['weights']} weights.")


if __name__ == "__main__":
    main()
