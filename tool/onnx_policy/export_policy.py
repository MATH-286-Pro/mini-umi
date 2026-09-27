"""Export a complete diffusion policy to one ONNX file."""
import argparse
import copy
import json
from pathlib import Path
import dill
import hydra
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy

class ObservationEncoderOnnx(torch.nn.Module):
    def __init__(self, policy, keys):
        super().__init__()
        self.normalizer = policy.normalizer
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
        return self.model(noisy_action, timestep, local_cond=None, global_cond=observation_condition)


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
    checkpoint = Path(checkpoint)
    with checkpoint.open("rb") as stream:
        payload = torch.load(f=stream, pickle_module=dill, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(payload["cfg"])
    OmegaConf.resolve(cfg)
    if "pretrained" in cfg.policy.obs_encoder:
        cfg.policy.obs_encoder.pretrained = False
    if "transforms" in cfg.policy.obs_encoder:
        del cfg.policy.obs_encoder.transforms
    policy = hydra.utils.instantiate(cfg.policy)
    states = payload["state_dicts"]
    state_key = ("ema_model" if "ema_model" in states else "model") if weights == "auto" else ("ema_model" if weights == "ema" else "model")
    if state_key not in states:
        raise KeyError(f"Checkpoint has no {state_key!r}; available: {sorted(states)}")
    policy.load_state_dict(states[state_key])
    policy.eval()
    return policy, state_key

def export_policy_onnx(checkpoint, output_dir, weights="auto", opset=17, num_inference_steps=None):
    import tempfile
    import numpy as np
    import onnx
    from onnx import TensorProto, compose, helper, numpy_helper

    policy, state_key = load_policy(checkpoint=checkpoint, weights=weights)
    if not isinstance(policy.noise_scheduler, DDIMScheduler):
        raise TypeError("Single-file export currently requires DDIMScheduler")
    if policy.noise_scheduler.config.prediction_type != "epsilon":
        raise ValueError("Single-file export currently requires prediction_type=epsilon")
    if policy.noise_scheduler.config.thresholding or float(policy.kwargs.get("eta", 0.0)) != 0.0:
        raise ValueError("Single-file export requires thresholding=False and eta=0")

    output_dir = Path(output_dir)
    if output_dir.suffix.lower() != ".onnx":
        output_dir = output_dir / "policy.onnx"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    inference_steps = int(policy.num_inference_steps if num_inference_steps is None else num_inference_steps)
    train_steps = int(policy.noise_scheduler.config.num_train_timesteps)
    if not 1 <= inference_steps <= train_steps:
        raise ValueError(f"num_inference_steps must be in [1, {train_steps}], got {inference_steps}")

    obs_encoder = policy.obs_encoder
    keys = sorted(obs_encoder.rgb_keys + obs_encoder.low_dim_keys)
    shape_meta = obs_encoder.shape_meta["obs"]
    examples = tuple(torch.zeros((1, int(shape_meta[key]["horizon"]), *shape_meta[key]["shape"]), dtype=torch.float32) for key in keys)
    encoder = ObservationEncoderOnnx(policy=policy, keys=keys)
    with torch.no_grad():
        condition = encoder(*examples)
    noise = torch.zeros((1, policy.action_horizon, policy.action_dim), dtype=torch.float32)
    if isinstance(policy, DiffusionUnetTimmPolicy):
        architecture, denoiser = "unet", UnetDenoiserOnnx(model=policy.model)
    elif isinstance(policy, DiffusionTransformerTimmPolicy):
        architecture, denoiser = "transformer", TransformerDenoiserOnnx(model=policy.model)
    else:
        raise TypeError(f"Unsupported policy type: {type(policy).__name__}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        encoder_file = Path(temporary_directory) / "encoder.onnx"
        denoiser_file = Path(temporary_directory) / "denoiser.onnx"
        torch.onnx.export(model=encoder, args=examples, f=encoder_file, input_names=keys, output_names=["observation_condition"], dynamic_axes={**{key: {0: "batch"} for key in keys}, "observation_condition": {0: "batch"}}, opset_version=opset, do_constant_folding=True, dynamo=False, external_data=False)
        torch.onnx.export(model=denoiser, args=(noise, torch.tensor(0, dtype=torch.int64), condition), f=denoiser_file, input_names=["noisy_action", "timestep", "observation_condition"], output_names=["model_output"], dynamic_axes={"noisy_action": {0: "batch"}, "observation_condition": {0: "batch"}, "model_output": {0: "batch"}}, opset_version=opset, do_constant_folding=True, dynamo=False, external_data=False)
        encoder_model = compose.add_prefix(onnx.load(encoder_file), "enc_")
        denoiser_model = compose.add_prefix(onnx.load(denoiser_file), "den_")

    input_renames = {f"enc_{key}": key for key in keys}
    for node in encoder_model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = input_renames.get(name, name)
    for graph_input in encoder_model.graph.input:
        graph_input.name = input_renames.get(graph_input.name, graph_input.name)

    scheduler = copy.deepcopy(policy.noise_scheduler)
    scheduler.set_timesteps(num_inference_steps=inference_steps)
    timesteps = np.asarray([int(value) for value in scheduler.timesteps], dtype=np.int64)
    alpha_values = scheduler.alphas_cumprod.cpu().numpy()
    alphas = alpha_values[timesteps].astype(np.float32)
    previous = np.concatenate([timesteps[1:], np.asarray([-1], dtype=np.int64)])
    final_alpha = float(torch.as_tensor(scheduler.final_alpha_cumprod))
    previous_alphas = np.asarray([alpha_values[value] if value >= 0 else final_alpha for value in previous], dtype=np.float32)

    body_nodes = []
    for node in denoiser_model.graph.node:
        copied = type(node)()
        copied.CopyFrom(node)
        for index, name in enumerate(copied.input):
            copied.input[index] = {"den_noisy_action": "sample_in", "den_timestep": "loop_timestep", "den_observation_condition": "enc_observation_condition"}.get(name, name)
        body_nodes.append(copied)
    body_nodes.extend([
        helper.make_node("Gather", ["loop_timesteps", "iteration"], ["loop_timestep"], axis=0),
    ])
    # Gather must run before the denoiser nodes.
    body_nodes.insert(0, body_nodes.pop())
    body_nodes.extend([
        helper.make_node("Gather", ["loop_alphas", "iteration"], ["alpha_t"], axis=0),
        helper.make_node("Gather", ["loop_previous_alphas", "iteration"], ["alpha_previous"], axis=0),
        helper.make_node("Sub", ["one", "alpha_t"], ["beta_t"]),
        helper.make_node("Sqrt", ["beta_t"], ["sqrt_beta_t"]),
        helper.make_node("Mul", ["sqrt_beta_t", "den_model_output"], ["weighted_output"]),
        helper.make_node("Sub", ["sample_in", "weighted_output"], ["original_numerator"]),
        helper.make_node("Sqrt", ["alpha_t"], ["sqrt_alpha_t"]),
        helper.make_node("Div", ["original_numerator", "sqrt_alpha_t"], ["predicted_original_unclipped"]),
        helper.make_node("Clip", ["predicted_original_unclipped", "clip_min", "clip_max"], ["predicted_original"]),
        helper.make_node("Sqrt", ["alpha_previous"], ["sqrt_alpha_previous"]),
        helper.make_node("Mul", ["sqrt_alpha_previous", "predicted_original"], ["original_component"]),
        helper.make_node("Sub", ["one", "alpha_previous"], ["previous_beta"]),
        helper.make_node("Sqrt", ["previous_beta"], ["sqrt_previous_beta"]),
        helper.make_node("Mul", ["sqrt_previous_beta", "den_model_output"], ["epsilon_component"]),
        helper.make_node("Add", ["original_component", "epsilon_component"], ["sample_out"]),
        helper.make_node("Identity", ["condition_in"], ["condition_out"]),
    ])
    action_shape = ["batch", policy.action_horizon, policy.action_dim]
    body = helper.make_graph(body_nodes, "ddim_loop_body", [helper.make_tensor_value_info("iteration", TensorProto.INT64, []), helper.make_tensor_value_info("condition_in", TensorProto.BOOL, []), helper.make_tensor_value_info("sample_in", TensorProto.FLOAT, action_shape)], [helper.make_tensor_value_info("condition_out", TensorProto.BOOL, []), helper.make_tensor_value_info("sample_out", TensorProto.FLOAT, action_shape)], initializer=[*denoiser_model.graph.initializer, numpy_helper.from_array(timesteps, "loop_timesteps"), numpy_helper.from_array(alphas, "loop_alphas"), numpy_helper.from_array(previous_alphas, "loop_previous_alphas"), numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), "one"), numpy_helper.from_array(np.asarray(-float(scheduler.config.clip_sample_range), dtype=np.float32), "clip_min"), numpy_helper.from_array(np.asarray(float(scheduler.config.clip_sample_range), dtype=np.float32), "clip_max")])

    action_params = policy.normalizer.params_dict["action"]
    scale = action_params["scale"].detach().cpu().numpy().astype(np.float32)
    offset = action_params["offset"].detach().cpu().numpy().astype(np.float32)
    top_nodes = list(encoder_model.graph.node)
    top_nodes.extend([helper.make_node("Loop", ["trip_count", "initial_condition", "initial_noise"], ["normalized_action"], body=body), helper.make_node("Sub", ["normalized_action", "action_offset"], ["centered_action"]), helper.make_node("Div", ["centered_action", "action_scale"], ["action"])])
    graph = helper.make_graph(top_nodes, "mini_umi_policy", [*encoder_model.graph.input, helper.make_tensor_value_info("initial_noise", TensorProto.FLOAT, action_shape)], [helper.make_tensor_value_info("action", TensorProto.FLOAT, action_shape)], initializer=[*encoder_model.graph.initializer, numpy_helper.from_array(np.asarray(inference_steps, dtype=np.int64), "trip_count"), numpy_helper.from_array(np.asarray(True, dtype=np.bool_), "initial_condition"), numpy_helper.from_array(offset, "action_offset"), numpy_helper.from_array(scale, "action_scale")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="mini-umi")
    model.ir_version = max(encoder_model.ir_version, denoiser_model.ir_version)
    metadata = {"format_version": 2, "architecture": architecture, "weights": state_key, "num_inference_steps": inference_steps, "observations": [{"name": key, "shape": [int(shape_meta[key]["horizon"]), *map(int, shape_meta[key]["shape"])], "type": shape_meta[key].get("type", "low_dim")} for key in keys], "action_shape": [policy.action_horizon, policy.action_dim], "scheduler": {"class": type(policy.noise_scheduler).__name__, "config": _json_value(dict(policy.noise_scheduler.config))}, "onnx": {"opset": opset, "dynamic_batch": True}}
    metadata_property = model.metadata_props.add()
    metadata_property.key = "mini_umi_policy"
    metadata_property.value = json.dumps(metadata, ensure_ascii=False)
    onnx.checker.check_model(model)
    onnx.save_model(model, f=output_dir, save_as_external_data=False)
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--weights", choices=("auto", "model", "ema"), default="auto")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int)
    args = parser.parse_args()

    metadata = export_policy_onnx(checkpoint=args.checkpoint, output_dir=args.output_file, weights=args.weights, opset=args.opset, num_inference_steps=args.num_inference_steps)
    output_file = args.output_file if args.output_file.suffix.lower() == ".onnx" else args.output_file / "policy.onnx"
    print(f"Exported {metadata['architecture']} policy to {output_file} using {metadata['weights']} weights and {metadata['num_inference_steps']} steps.")

if __name__ == "__main__":
    main()
