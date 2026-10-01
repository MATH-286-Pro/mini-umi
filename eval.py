"""Run episode-level diffusion policy evaluation and create trajectory GIFs."""

import argparse
import json
from pathlib import Path

from diffusion_policy.evaluation import (
    evaluate_episode_trajectory,
    load_dataset_for_evaluation,
    load_policy_for_evaluation,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a policy checkpoint on selected dataset episodes.",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--episode",
        type=int,
        action="append",
        default=None,
        help="Zero-based episode index. Repeat the option to evaluate multiple episodes.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("auto", "ema", "model"), default="auto")
    parser.add_argument("--dataset-format", choices=("auto", "lerobot", "zarr"), default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-fps", type=float, default=None)
    parser.add_argument("--robot-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--gif-fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    episodes = [0] if args.episode is None else args.episode
    output_dir = (
        Path("outputs/eval") / args.checkpoint.stem
        if args.output_dir is None
        else args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, cfg, state_key = load_policy_for_evaluation(
        checkpoint_path=args.checkpoint,
        weights=args.weights,
        device=args.device,
    )
    dataset = load_dataset_for_evaluation(
        dataset_path=args.dataset,
        shape_meta=cfg.policy.shape_meta,
        dataset_format=args.dataset_format,
        cache_dir=args.cache_dir,
    )
    dataset_fps = args.dataset_fps
    if dataset_fps is None:
        dataset_fps = float(getattr(dataset, "lerobot_info", {}).get("fps", 60.0))

    print(f"Loaded {state_key} weights on {policy.device}; dataset has {dataset.replay_buffer.n_episodes} episodes")
    for episode_index in episodes:
        result = evaluate_episode_trajectory(
            policy=policy,
            dataset=dataset,
            episode_index=episode_index,
            output_path=output_dir / f"episode_{episode_index:03d}.gif",
            device=policy.device,
            dataset_fps=dataset_fps,
            robot_index=args.robot_index,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            inference_batch_size=args.inference_batch_size,
            gif_fps=args.gif_fps,
            seed=args.seed,
        )
        print(f"episode {episode_index}: {result.gif_path}")
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
