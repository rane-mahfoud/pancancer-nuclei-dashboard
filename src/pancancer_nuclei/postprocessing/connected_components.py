"""Convert semantic segmentation masks into individual nuclei."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

NUMBER_OF_SEMANTIC_CLASSES = 6
IGNORE_LABEL = 255


@dataclass(frozen=True)
class InstanceSegmentation:
    """Individual masks, categories, and labelled instance map."""

    instance_masks: np.ndarray
    categories: np.ndarray
    instance_map: np.ndarray

    @property
    def number_of_instances(self) -> int:
        """Return the number of detected nuclei."""
        return int(self.instance_masks.shape[0])


def semantic_to_instances(
    semantic_mask: np.ndarray,
    minimum_area: int = 10,
    connectivity: int = 2,
) -> InstanceSegmentation:
    """Split semantic regions into connected nucleus instances.

    Semantic labels use:
        0: background
        1: neoplastic
        2: inflammatory
        3: connective
        4: dead
        5: epithelial
        255: ignored pixel

    Returned categories use the original PanNuke convention from 0 to 4.
    """
    semantic_mask = np.asarray(semantic_mask)

    if semantic_mask.ndim != 2:
        raise ValueError("semantic_mask must have shape (height, width).")

    if minimum_area < 1:
        raise ValueError("minimum_area must be at least 1.")

    if connectivity not in (1, 2):
        raise ValueError("connectivity must be either 1 or 2.")

    valid_labels = set(range(NUMBER_OF_SEMANTIC_CLASSES))
    valid_labels.add(IGNORE_LABEL)

    observed_labels = set(np.unique(semantic_mask).tolist())
    invalid_labels = observed_labels - valid_labels

    if invalid_labels:
        raise ValueError(f"semantic_mask contains invalid labels: {sorted(invalid_labels)}")

    structure = ndimage.generate_binary_structure(
        rank=2,
        connectivity=connectivity,
    )

    masks: list[np.ndarray] = []
    categories: list[int] = []

    instance_map = np.zeros(
        semantic_mask.shape,
        dtype=np.int32,
    )
    next_instance_id = 1

    for semantic_label in range(1, NUMBER_OF_SEMANTIC_CLASSES):
        class_mask = semantic_mask == semantic_label
        component_map, number_of_components = ndimage.label(
            class_mask,
            structure=structure,
        )

        for component_id in range(1, number_of_components + 1):
            component_mask = component_map == component_id
            component_area = int(component_mask.sum())

            if component_area < minimum_area:
                continue

            masks.append(component_mask)
            categories.append(semantic_label - 1)

            instance_map[component_mask] = next_instance_id
            next_instance_id += 1

    if masks:
        instance_masks = np.stack(masks).astype(bool)
    else:
        height, width = semantic_mask.shape
        instance_masks = np.empty(
            (0, height, width),
            dtype=bool,
        )

    return InstanceSegmentation(
        instance_masks=instance_masks,
        categories=np.asarray(categories, dtype=np.int64),
        instance_map=instance_map,
    )
