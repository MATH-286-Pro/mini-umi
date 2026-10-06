import torch
import torchvision


class SequenceImageAugmentor(torch.nn.Module):
    """Apply one spatial/color transform consistently to an image history.

    Input tensors use ``T,C,H,W``. Torchvision treats the leading time axis as
    an additional batch dimension, so random parameters are shared by every
    frame in the observation history.
    """

    def __init__(self, image_shape: tuple, transforms: list):
        super().__init__()
        image_shape = tuple(image_shape)
        if len(image_shape) != 2:
            raise ValueError(f"image_shape must be (H, W), got {image_shape}")

        modules = list(transforms or [])
        if modules and not isinstance(modules[0], torch.nn.Module):
            if modules[0].type != 'RandomCrop':
                raise ValueError("The only supported declarative first transform is RandomCrop")
            ratio = float(modules[0].ratio)
            crop_size = tuple(max(1, int(size * ratio)) for size in image_shape)
            modules = [
                torchvision.transforms.RandomCrop(size=crop_size),
                torchvision.transforms.Resize(size=image_shape, antialias=True),
            ] + modules[1:]
        self.transform = torch.nn.Identity() if not modules else torch.nn.Sequential(*modules)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected T,C,H,W image history, got shape {tuple(image.shape)}")
        return self.transform(image)
