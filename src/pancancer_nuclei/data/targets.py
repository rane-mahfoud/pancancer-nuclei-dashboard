"""Training-target generation for boundary-aware segmentation."""

import numpy as np
from scipy import ndimage

BACKGROUND_LABEL = 0
INTERIOR_LABEL = 1
BOUNDARY_LABEL = 2
IGNORE_LABEL = 255

NUMBER_OF_SPATIAL_CLASSES = 3


def create_boundary_target(
    instance_masks: np.ndarray,
    boundary_width: int = 2,
) -> np.ndarray:
    """Create background, nucleus-interior, and boundary labels.

    Labels:
        0: background
        1: nucleus interior
        2: nucleus boundary
        255: overlapping ambiguous annotation
    """
    instance_masks = np.asarray(instance_masks)

    if instance_masks.ndim != 3:
        raise ValueError("instance_masks must have shape (number_of_instances, height, width).")

    if boundary_width < 1:
        raise ValueError("boundary_width must be at least 1.")

    boolean_masks = instance_masks.astype(bool)
    _, height, width = boolean_masks.shape

    target = np.full(
        (height, width),
        BACKGROUND_LABEL,
        dtype=np.uint8,
    )

    if len(boolean_masks) == 0:
        return target

    foreground = np.any(boolean_masks, axis=0)
    overlap = boolean_masks.sum(axis=0) > 1
    boundary = np.zeros((height, width), dtype=bool)

    erosion_structure = np.ones((3, 3), dtype=bool)

    for instance_mask in boolean_masks:
        if not instance_mask.any():
            continue

        eroded_mask = ndimage.binary_erosion(
            instance_mask,
            structure=erosion_structure,
            iterations=boundary_width,
            border_value=0,
        )
        instance_boundary = instance_mask & ~eroded_mask
        boundary |= instance_boundary

    target[foreground] = INTERIOR_LABEL
    target[boundary] = BOUNDARY_LABEL
    target[overlap] = IGNORE_LABEL

    return target
