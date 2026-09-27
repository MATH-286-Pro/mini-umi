import unittest

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf


class TrainingConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OmegaConf.register_new_resolver("eval", eval, replace=True)

    def compose(self, config_name, overrides=None):
        with initialize(version_base=None, config_path="../diffusion_policy/config"):
            return compose(config_name=config_name, overrides=overrides or [])

    def test_default_config_groups_are_merged(self):
        cfg = self.compose("train/unet_timm_umi")
        resolved = OmegaConf.to_container(cfg, resolve=True)

        self.assertEqual(resolved["task"]["name"], "umi")
        self.assertEqual(resolved["dataset_name"], "zarr")
        self.assertEqual(resolved["augmentation"]["name"], "crop_color")
        self.assertEqual(resolved["network_name"], "unet_timm")
        self.assertEqual(
            resolved["policy"]["image_augmentor"]["_target_"],
            "diffusion_policy.model.vision.image_augmentation.BatchImageAugmentor",
        )

        augmentor = instantiate(cfg.policy.image_augmentor)
        self.assertEqual(augmentor.__class__.__name__, "BatchImageAugmentor")

    def test_config_groups_can_be_overridden_independently(self):
        cfg = self.compose(
            "train/unet_timm_umi",
            overrides=[
                "dataset=lerobot",
                "network=transformer_timm",
                "augmentation=none",
                "dataset.dataset_path=/tmp/umi_lerobot",
            ],
        )
        resolved = OmegaConf.to_container(cfg, resolve=True)

        self.assertEqual(resolved["dataset_name"], "lerobot")
        self.assertEqual(resolved["dataset"]["dataset_path"], "/tmp/umi_lerobot")
        self.assertEqual(resolved["network_name"], "transformer_timm")
        self.assertEqual(resolved["augmentation"]["name"], "none")
        self.assertIsNone(resolved["policy"]["image_augmentor"])
        self.assertEqual(
            resolved["_target_"],
            "diffusion_policy.workspace.train_diffusion_transformer_timm_workspace.TrainDiffusionTransformerTimmWorkspace",
        )

    def test_legacy_experiment_names_select_expected_groups(self):
        cases = {
            "train/unet_timm_umi_bimanual": ("zarr", "unet_timm", "crop_rotate_color"),
            "train/transformer_umi": ("zarr", "transformer_timm", "crop_rotate_color"),
            "train/transformer_umi_bimanual": ("zarr", "transformer_timm", "crop_rotate_color"),
        }
        for config_name, expected in cases.items():
            with self.subTest(config_name=config_name):
                cfg = self.compose(config_name)
                actual = (cfg.dataset_name, cfg.network_name, cfg.augmentation.name)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
