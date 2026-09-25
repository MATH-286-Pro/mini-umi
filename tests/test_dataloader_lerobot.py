import tempfile
import unittest
from pathlib import Path

import pandas as pd

from diffusion_policy.dataset.dataloader_lerobot import (
    _LeRobotLayoutV21,
    _LeRobotLayoutV30,
    _make_lerobot_layout,
)


class TestLeRobotLayouts(unittest.TestCase):
    def test_v21_reads_jsonl_and_locates_episode_video(self):
        info = {
            "codebase_version": "v2.1",
            "fps": 20,
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
        }
        layout = _make_lerobot_layout(info=info)
        self.assertIsInstance(layout, _LeRobotLayoutV21)

        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory)
            metadata_path = dataset_path / "meta" / "episodes.jsonl"
            metadata_path.parent.mkdir()
            metadata_path.write_text(
                '{"episode_index": 1001, "length": 12, "tasks": ["pick"]}\n'
            )
            episodes = layout.load_episodes(dataset_path=dataset_path)

        video_path, timestamp = layout.locate_video_frame(
            episode=episodes.iloc[0],
            feature_key="observation.images.camera0_rgb",
            frame_index=5,
        )
        self.assertEqual(
            video_path,
            "videos/chunk-001/observation.images.camera0_rgb/episode_001001.mp4",
        )
        self.assertEqual(timestamp, 0.25)

    def test_v30_locates_frame_inside_shared_video(self):
        info = {
            "codebase_version": "v3.0",
            "fps": 20,
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
        layout = _make_lerobot_layout(info=info)
        self.assertIsInstance(layout, _LeRobotLayoutV30)
        episode = pd.Series({
            "videos/camera/chunk_index": 2,
            "videos/camera/file_index": 3,
            "videos/camera/from_timestamp": 1.5,
        })

        video_path, timestamp = layout.locate_video_frame(
            episode=episode,
            feature_key="camera",
            frame_index=5,
        )

        self.assertEqual(video_path, "videos/camera/chunk-002/file-003.mp4")
        self.assertEqual(timestamp, 1.75)

    def test_unknown_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported LeRobot dataset version"):
            _make_lerobot_layout(info={"codebase_version": "v1.0"})


if __name__ == "__main__":
    unittest.main()
