"""Load and run an exported ONNX diffusion policy."""

import json
from pathlib import Path

import diffusers
import numpy as np
import onnxruntime as ort
import torch


class OnnxPolicy:
    """ONNX Runtime policy with an external Diffusers noise scheduler."""

    def __init__(self, model_dir, providers=None, num_inference_steps=None):
        self.model_dir = Path(model_dir)
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != 1:
            raise ValueError(
                f"Unsupported format version: {self.metadata.get('format_version')}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_kwargs = {"sess_options": session_options}
        if providers is not None:
            session_kwargs["providers"] = providers
        
        files = self.metadata["files"]
        self.obs_encoder = ort.InferenceSession(str(self.model_dir / files["obs_encoder"]), **session_kwargs)
        self.denoiser    = ort.InferenceSession(str(self.model_dir / files["denoiser"]), **session_kwargs)

        scheduler_meta = self.metadata["scheduler"]
        scheduler_class = getattr(diffusers, scheduler_meta["class"], None)
        if scheduler_class is None:
            raise ValueError(
                f"Unsupported scheduler class: {scheduler_meta['class']}")
        self.scheduler = scheduler_class.from_config(scheduler_meta["config"])

        self.infer_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else scheduler_meta["default_num_inference_steps"])
        self.train_steps = int(self.scheduler.config.num_train_timesteps)

        # 检查 inference 步数与 training 步数
        if self.infer_steps > self.train_steps:
            raise ValueError(
                f"num_inference_steps ({self.infer_steps}) exceeds num_train_timesteps "
                f"({self.train_steps})")

        # 设置 scheduler 步数
        self.scheduler.set_timesteps(self.infer_steps)


        # obs 与 action
        self.observation_specs = {item["name"]: item for item in self.metadata["observations"]}
        self.action_shape = tuple(self.metadata["action_shape"])
        
        # 归一化
        normalizer = self.metadata["action_normalizer"]
        self.action_scale  = np.asarray(normalizer["scale"],  dtype=np.float32)
        self.action_offset = np.asarray(normalizer["offset"], dtype=np.float32)

        self.step_kwargs = scheduler_meta.get("step_kwargs", {})

    def NORMALIZE(self, action):
        return (action - self.action_offset) / self.action_scale


    def __call__(self, obs: dict, seed=None) -> np.ndarray:
        return self.predict_action(self, obs, seed=None)

    def _prepare_observations(self, obs: dict):

        # 检查 obs dict
        missing = set(self.observation_specs) - set(obs)
        if missing:
            raise KeyError(f"Missing observations: {sorted(missing)}")
        
        inputs = {}
        batch_size = None

        for name, spec in self.observation_specs.items():
            value = np.ascontiguousarray(obs[name], dtype=np.float32)
            expected = tuple(spec["shape"])

            # 检查维度
            if value.ndim != len(expected) + 1 or tuple(value.shape[1:]) != expected:
                raise ValueError(
                    f"{name!r} must have shape (batch, {expected}), got {value.shape}")
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("All observations must use the same batch size")
            inputs[name] = value

        return inputs, batch_size

    def predict_action(self, obs: dict, seed=None) -> np.ndarray:

        # 构建观测
        encoder_inputs, batch_size = self._prepare_observations(obs)
        condition = self.obs_encoder.run(["observation_condition"], encoder_inputs)[0]

        # 采集 XT (action) 初始噪声
        rng = np.random.default_rng(seed)
        sample = rng.standard_normal((batch_size, *self.action_shape), dtype=np.float32)

        # 根据 scheduler 运行 action 去噪过程
        # 设置 scheduler 在 init 函数中
        for timestep in self.scheduler.timesteps:
            timestep_value = np.asarray(int(timestep), dtype=np.int64)
            model_output = self.denoiser.run(
                ["model_output"],
                {
                    "noisy_action": sample,
                    "timestep": timestep_value,
                    "observation_condition": condition,
                },
            )[0]
            sample_tensor = torch.from_numpy(sample)
            output_tensor = torch.from_numpy(model_output)
            sample = self.scheduler.step(
                output_tensor, timestep, sample_tensor,
                **self.step_kwargs).prev_sample.numpy()

        # 归一化 action
        action = self.NORMALIZE(sample)

        return action
