"""Convert boundary-aware predictions into individual nuclei with watershed."""

import numpy as np
from scipy import ndimage

from pancancer_nuclei.postprocessing.connected_components import (
    IGNORE_LABEL,
    NUMBER_OF_SEMANTIC_CLASSES,
    InstanceSegmentation,
)

NUMBER_OF_SPATIAL_CLASSES = 3
INTERIOR_LABEL = 1
BOUNDARY_LABEL = 2
CONNECTED_COMPONENT_SEMANTIC_LABELS = (2,)


def _empty_segmentation(image_shape: tuple[int, int]) -> InstanceSegmentation:
    """Create an empty result with the expected shapes and dtypes."""
    return InstanceSegmentation(
        instance_masks=np.empty((0, *image_shape), dtype=bool),
        categories=np.empty(0, dtype=np.int64),
        instance_map=np.zeros(image_shape, dtype=np.int32),
    )


def _validate_inputs(
    semantic_mask: np.ndarray,
    spatial_probabilities: np.ndarray,
    seed_threshold: float,
    minimum_seed_area: int,
    minimum_instance_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize watershed inputs."""
    semantic_mask = np.asarray(semantic_mask)
    spatial_probabilities = np.asarray(
        spatial_probabilities,
        dtype=np.float32,
    )

    if semantic_mask.ndim != 2:
        raise ValueError("semantic_mask must have shape (height, width).")
    if spatial_probabilities.shape != (
        NUMBER_OF_SPATIAL_CLASSES,
        *semantic_mask.shape,
    ):
        raise ValueError("spatial_probabilities must have shape (3, height, width).")
    if not np.isfinite(spatial_probabilities).all():
        raise ValueError("spatial_probabilities must contain finite values.")
    if not 0.0 <= seed_threshold <= 1.0:
        raise ValueError("seed_threshold must be between 0 and 1.")
    if minimum_seed_area < 1:
        raise ValueError("minimum_seed_area must be at least 1.")
    if minimum_instance_area < 1:
        raise ValueError("minimum_instance_area must be at least 1.")

    valid_semantic_labels = set(range(NUMBER_OF_SEMANTIC_CLASSES))
    valid_semantic_labels.add(IGNORE_LABEL)
    invalid_labels = set(np.unique(semantic_mask).tolist()) - valid_semantic_labels
    if invalid_labels:
        raise ValueError(f"semantic_mask contains invalid labels: {sorted(invalid_labels)}")

    return semantic_mask, spatial_probabilities


def _create_markers(
    foreground: np.ndarray,
    interior_probability: np.ndarray,
    boundary_probability: np.ndarray,
    seed_threshold: float,
    minimum_seed_area: int,
    structure: np.ndarray,
) -> np.ndarray:
    """Create reliable interior markers and cover unseeded foreground regions."""
    seed_mask = (
        foreground
        & (interior_probability >= seed_threshold)
        & (interior_probability > boundary_probability)
    )
    raw_markers, number_of_markers = ndimage.label(
        seed_mask,
        structure=structure,
    )

    filtered_seed_mask = np.zeros(foreground.shape, dtype=bool)
    for marker_id in range(1, number_of_markers + 1):
        marker_mask = raw_markers == marker_id
        if int(marker_mask.sum()) >= minimum_seed_area:
            filtered_seed_mask[marker_mask] = True

    markers, _ = ndimage.label(filtered_seed_mask, structure=structure)
    foreground_components, number_of_components = ndimage.label(
        foreground,
        structure=structure,
    )

    next_marker = int(markers.max()) + 1
    for component_id in range(1, number_of_components + 1):
        component = foreground_components == component_id
        if np.any(markers[component] > 0):
            continue

        component_scores = np.where(component, interior_probability, -1.0)
        seed_position = np.unravel_index(
            int(np.argmax(component_scores)),
            component_scores.shape,
        )
        markers[seed_position] = next_marker
        next_marker += 1

    return markers.astype(np.int32)


def boundary_predictions_to_instances(
    semantic_mask: np.ndarray,
    spatial_probabilities: np.ndarray,
    seed_threshold: float = 0.5,
    minimum_seed_area: int = 15,
    minimum_instance_area: int = 100,
    connectivity: int = 2,
) -> InstanceSegmentation:
    """Use interior markers and boundary probabilities to separate nuclei.

    Semantic labels use background 0 and nucleus classes 1-5. Spatial
    probabilities are ordered as background, interior, and boundary.
    Returned categories follow the original PanNuke convention 0-4.
    """
    semantic_mask, spatial_probabilities = _validate_inputs(
        semantic_mask,
        spatial_probabilities,
        seed_threshold,
        minimum_seed_area,
        minimum_instance_area,
    )
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be either 1 or 2.")

    foreground = (semantic_mask > 0) & (semantic_mask < NUMBER_OF_SEMANTIC_CLASSES)
    if not np.any(foreground):
        return _empty_segmentation(tuple(semantic_mask.shape))

    structure = ndimage.generate_binary_structure(2, connectivity)
    interior_probability = spatial_probabilities[INTERIOR_LABEL]
    boundary_probability = spatial_probabilities[BOUNDARY_LABEL]
    elevation = boundary_probability + 0.25 * (1.0 - interior_probability)
    elevation = np.clip(elevation, 0.0, 1.25)
    elevation = np.rint(elevation / 1.25 * 255.0).astype(np.uint8)

    masks: list[np.ndarray] = []
    categories: list[int] = []
    instance_map = np.zeros(semantic_mask.shape, dtype=np.int32)

    connected_component_foreground = np.isin(
        semantic_mask,
        CONNECTED_COMPONENT_SEMANTIC_LABELS,
    )
    watershed_foreground = foreground & ~connected_component_foreground

    # The boundary head improves separation for most classes. Inflammatory
    # predictions are instead kept as connected components because Fold 2
    # diagnostics showed that watershed consistently over-split that class.
    if np.any(watershed_foreground):
        markers = _create_markers(
            foreground=watershed_foreground,
            interior_probability=interior_probability,
            boundary_probability=boundary_probability,
            seed_threshold=seed_threshold,
            minimum_seed_area=minimum_seed_area,
            structure=structure,
        )
        watershed_map = ndimage.watershed_ift(
            elevation,
            markers,
            structure=structure,
        )
        watershed_map[~watershed_foreground] = 0

        for marker_id in np.unique(watershed_map):
            if marker_id == 0:
                continue

            instance_mask = watershed_map == marker_id
            if int(instance_mask.sum()) < minimum_instance_area:
                continue

            semantic_counts = np.bincount(
                semantic_mask[instance_mask],
                minlength=NUMBER_OF_SEMANTIC_CLASSES,
            )
            semantic_label = int(np.argmax(semantic_counts[1:]) + 1)
            next_instance_id = len(masks) + 1
            masks.append(instance_mask)
            categories.append(semantic_label - 1)
            instance_map[instance_mask] = next_instance_id

    for semantic_label in CONNECTED_COMPONENT_SEMANTIC_LABELS:
        component_map, number_of_components = ndimage.label(
            semantic_mask == semantic_label,
            structure=structure,
        )
        for component_id in range(1, number_of_components + 1):
            instance_mask = component_map == component_id
            if int(instance_mask.sum()) < minimum_instance_area:
                continue

            next_instance_id = len(masks) + 1
            masks.append(instance_mask)
            categories.append(semantic_label - 1)
            instance_map[instance_mask] = next_instance_id

    if not masks:
        return _empty_segmentation(tuple(semantic_mask.shape))

    return InstanceSegmentation(
        instance_masks=np.stack(masks).astype(bool),
        categories=np.asarray(categories, dtype=np.int64),
        instance_map=instance_map,
    )
