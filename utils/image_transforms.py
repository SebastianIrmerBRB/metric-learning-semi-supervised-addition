"""Image preprocessing pipelines shared by every dataset the project loads.

``image_resize_mode`` selects the geometry that turns a source image into the
square tensor the DINOv2 backbone consumes.

``squash`` is the historical pipeline every study before 2026-08-25 ran: a
direct resize to 224x224 that ignores aspect ratio but never crops the object
out of frame.

``dinov2`` reproduces the resize-then-center-crop geometry of DINOv2's own
evaluation transform (short side to 256 with bicubic interpolation, then a
224 center crop), and gives training the scale jitter and horizontal flip that
DINOv2 pretraining saw. Both are closer to the backbone's pretraining
distribution; the crop is the trade, since a 224 center crop can cut off part
of an off-center object.

Only the deterministic transform matters for the frozen + cached runs this
project uses most: ``use_feature_transform_for_training`` replaces training
augmentation with ``feature_transform`` whenever backbone features are cached,
so the train pipeline below is reached only by unfrozen or
augmented-precompute runs.
"""

import torchvision.transforms as tfm
import torchvision.transforms.v2 as v2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_RESIZE_MODE_SQUASH = "squash"
IMAGE_RESIZE_MODE_DINOV2 = "dinov2"
IMAGE_RESIZE_MODES = (IMAGE_RESIZE_MODE_SQUASH, IMAGE_RESIZE_MODE_DINOV2)
DEFAULT_IMAGE_RESIZE_MODE = IMAGE_RESIZE_MODE_SQUASH

# DINOv2 splits its input into 14x14 patches and asserts the side length is a
# multiple of that patch size, so only 224, 336, 448 and 518 are usable sizes.
DINOV2_PATCH_SIZE = 14
DEFAULT_IMAGE_SIZE = 224
# DINOv2's released eval transform is Resize(256) -> CenterCrop(224); keeping
# the ratio rather than the literal 256 lets a larger crop scale with it.
DINOV2_RESIZE_RATIO = 256 / 224
# Scale jitter for the training crop. DINOv2 pretraining uses (0.32, 1.0) for
# its global crops; retrieval fine-tuning keeps more of the object in frame.
DINOV2_TRAIN_CROP_SCALE = (0.5, 1.0)
DINOV2_TRAIN_RAND_AUGMENT_OPS = 2
SQUASH_TRAIN_RAND_AUGMENT_OPS = 3


IMAGE_RESIZE_MODE_INFO = {
    IMAGE_RESIZE_MODE_SQUASH: (
        "resizes to a square 224x224 for train and evaluation, discarding "
        "aspect ratio but keeping the whole image; training adds RandAugment"
    ),
    IMAGE_RESIZE_MODE_DINOV2: (
        "matches DINOv2's own eval transform, resizing the short side to 256 "
        "with bicubic interpolation and taking a 224 center crop; training "
        "uses RandomResizedCrop plus a horizontal flip and RandAugment"
    ),
}


def describe_image_resize_modes():
    """Return one ``name: summary`` line per resize mode, in offered order."""

    return [f"{mode}: {IMAGE_RESIZE_MODE_INFO[mode]}" for mode in IMAGE_RESIZE_MODES]


def format_image_resize_mode_help():
    """Build the --image_resize_mode help text from the mode registry."""

    return (
        "image geometry applied before the backbone. "
        + " | ".join(describe_image_resize_modes())
    )


def validate_image_resize_mode(image_resize_mode):
    """Return the mode after checking it against the registry."""

    if image_resize_mode not in IMAGE_RESIZE_MODES:
        raise ValueError(
            f"image_resize_mode must be one of {IMAGE_RESIZE_MODES}: {image_resize_mode}"
        )
    return image_resize_mode


def validate_image_size(image_size):
    """Return the side length after checking the backbone can patch it."""

    image_size = int(image_size)
    if image_size <= 0 or image_size % DINOV2_PATCH_SIZE:
        raise ValueError(
            f"image_size must be a positive multiple of the DINOv2 patch size "
            f"{DINOV2_PATCH_SIZE}: {image_size}"
        )
    return image_size


def dinov2_resize_size(image_size=DEFAULT_IMAGE_SIZE):
    """Return the short-side resize that precedes a ``image_size`` center crop."""

    return int(round(validate_image_size(image_size) * DINOV2_RESIZE_RATIO))


def make_train_transform(
    image_resize_mode=DEFAULT_IMAGE_RESIZE_MODE,
    image_size=DEFAULT_IMAGE_SIZE,
):
    """Build the stochastic transform used while optimizing."""

    validate_image_resize_mode(image_resize_mode)
    image_size = validate_image_size(image_size)
    if image_resize_mode == IMAGE_RESIZE_MODE_DINOV2:
        geometry = [
            tfm.RandomResizedCrop(
                image_size,
                scale=DINOV2_TRAIN_CROP_SCALE,
                interpolation=tfm.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            tfm.RandomHorizontalFlip(),
            tfm.RandAugment(
                num_ops=DINOV2_TRAIN_RAND_AUGMENT_OPS,
                interpolation=tfm.InterpolationMode.BILINEAR,
            ),
        ]
    else:
        geometry = [
            tfm.Resize(size=(image_size, image_size), antialias=True),
            tfm.RandAugment(
                num_ops=SQUASH_TRAIN_RAND_AUGMENT_OPS,
                interpolation=tfm.InterpolationMode.BILINEAR,
            ),
        ]
    return tfm.Compose(
        [
            v2.RGB(),
            *geometry,
            tfm.ToTensor(),
            tfm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_test_transform(
    image_resize_mode=DEFAULT_IMAGE_RESIZE_MODE,
    image_size=DEFAULT_IMAGE_SIZE,
):
    """Build the deterministic transform used for validation, test, and features."""

    validate_image_resize_mode(image_resize_mode)
    image_size = validate_image_size(image_size)
    if image_resize_mode == IMAGE_RESIZE_MODE_DINOV2:
        geometry = [
            tfm.Resize(
                dinov2_resize_size(image_size),
                interpolation=tfm.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            tfm.CenterCrop(image_size),
        ]
    else:
        geometry = [tfm.Resize(size=(image_size, image_size), antialias=True)]
    return tfm.Compose(
        [
            v2.RGB(),
            *geometry,
            tfm.ToTensor(),
            tfm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
