"""Load and run a single-file ONNX diffusion policy."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

class OnnxPolicy:
    """ONNX Runtime wrapper for an end-to-end exported policy."""
    def __init__(self, model_file, providers=None):
        self.model_file = Path(model_file)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_kwargs = {"sess_options": session_options}
        if providers is not None:
            session_kwargs["providers"] = providers
        self.session = ort.InferenceSession(str(self.model_file), **session_kwargs)
        custom_metadata = self.session.get_modelmeta().custom_metadata_map
        if "mini_umi_policy" not in custom_metadata:
            raise ValueError(f"{self.model_file} has no mini-UMI policy metadata")
        self.metadata = json.loads(custom_metadata["mini_umi_policy"])
        if self.metadata.get("format_version") != 2:
            raise ValueError(f"Unsupported format version: {self.metadata.get('format_version')}")
        self.observation_specs = {item["name"]: item for item in self.metadata["observations"]}
        self.action_shape = tuple(self.metadata["action_shape"])
        self.num_inference_steps = int(self.metadata["num_inference_steps"])

    def __call__(self, obs: dict, seed=None) -> np.ndarray:
        return self.predict_action(obs=obs, seed=seed)

    def _prepare_observations(self, obs):
        missing = set(self.observation_specs) - set(obs)
        if missing:
            raise KeyError(f"Missing observations: {sorted(missing)}")
        inputs = {}
        batch_size = None
        for name, spec in self.observation_specs.items():
            value = np.ascontiguousarray(obs[name], dtype=np.float32)
            expected = tuple(spec["shape"])
            if value.ndim != len(expected) + 1 or tuple(value.shape[1:]) != expected:
                raise ValueError(f"{name!r} must have shape (batch, {expected}), got {value.shape}")
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("All observations must use the same batch size")
            inputs[name] = value
        return inputs, batch_size

    def predict_action(self, obs: dict, seed=None) -> np.ndarray:
        inputs, batch_size = self._prepare_observations(obs=obs)
        rng = np.random.default_rng(seed=seed)
        inputs["initial_noise"] = rng.standard_normal((batch_size, *self.action_shape), dtype=np.float32)
        return self.session.run(["action"], inputs)[0]
