"""Synthetic image data used by the debugger harness."""

from __future__ import annotations

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class TinyImageDataset(Dataset):
    """Small class-clustered image dataset with train and feature transforms."""

    def __init__(
        self,
        num_classes: int,
        samples_per_class: int,
        image_size: int,
        seed: int,
    ) -> None:
        self.num_classes = int(num_classes)
        self.samples_per_class = int(samples_per_class)
        self.image_size = int(image_size)
        self.labels = [
            class_id
            for class_id in range(self.num_classes)
            for _ in range(self.samples_per_class)
        ]
        self.orig_labels = [100 + label for label in self.labels]
        self.images = [
            self._make_image(class_id=label, sample_id=index, seed=seed)
            for index, label in enumerate(self.labels)
        ]

        # Training views are deliberately stochastic so augmented batch
        # construction can be inspected. Graph embeddings use the stable
        # feature transform, matching the production extraction path.
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.feature_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def _make_image(self, class_id: int, sample_id: int, seed: int) -> Image.Image:
        source_size = self.image_size + 6
        rng = np.random.default_rng(seed + 1009 * sample_id)
        image = rng.normal(18.0, 5.0, size=(source_size, source_size, 3))

        # Each class has a distinct dominant channel and stripe location. The
        # model starts random, but these clusters make graph structure readable.
        dominant_channel = class_id % 3
        image[..., dominant_channel] += 145.0
        stripe = 2 + (class_id * 3) % max(3, source_size - 4)
        image[stripe : stripe + 3, :, :] += 45.0
        image[:, (stripe + sample_id) % source_size, :] += 35.0
        return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.images[int(index)].copy()
        if self.transform is not None:
            image = self.transform(image)
        return image, self.orig_labels[int(index)]
