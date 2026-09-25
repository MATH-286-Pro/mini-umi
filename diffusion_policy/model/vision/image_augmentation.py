import torch
import torchvision


class BatchImageAugmentor(torch.nn.Module):
    """Augment batched image histories before visual feature extraction.

    Input tensors use ``B,T,C,H,W``. Each batch item draws independent random
    parameters while every frame in one observation history shares them.
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
        if image.ndim != 5:
            raise ValueError(f"Expected B,T,C,H,W image batch, got shape {tuple(image.shape)}")
        return torch.stack([self.transform(sequence) for sequence in image], dim=0)


def augment_observations(obs_dict: dict, rgb_keys: list[str], augmentor: BatchImageAugmentor) -> dict:
    result = dict(obs_dict)
    for key in rgb_keys:
        result[key] = augmentor(result[key])
    return result
