import unittest

import torch

from diffusion_policy.model.vision.image_augmentation import BatchImageAugmentor


class _PerCallOffset(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value + self.calls


class BatchImageAugmentorTest(unittest.TestCase):
    def test_batch_items_are_independent_and_history_frames_are_consistent(self):
        transform = _PerCallOffset()
        augmentor = BatchImageAugmentor(image_shape=(8, 8), transforms=[transform])
        image = torch.zeros(2, 3, 1, 8, 8)

        result = augmentor(image)

        self.assertEqual(transform.calls, 2)
        torch.testing.assert_close(result[0, 0], result[0, 2])
        torch.testing.assert_close(result[1, 0], result[1, 2])
        self.assertFalse(torch.equal(result[0], result[1]))

    def test_rejects_unbatched_history(self):
        augmentor = BatchImageAugmentor(image_shape=(8, 8), transforms=[])
        with self.assertRaisesRegex(ValueError, "B,T,C,H,W"):
            augmentor(torch.zeros(3, 1, 8, 8))


if __name__ == "__main__":
    unittest.main()
