"""Image augmentations used for PanNuke model training."""

import albumentations as A


def create_training_transforms() -> A.Compose:
    """Create safe augmentations for histology training images."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.08,
                hue=0.02,
                p=0.5,
            ),
        ]
    )


def create_validation_transforms() -> A.Compose:
    """Return validation transforms without random modifications."""
    return A.Compose([])
