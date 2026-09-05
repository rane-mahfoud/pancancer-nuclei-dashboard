"""Tests for boundary-aware training targets."""

import numpy as np
import pytest

from pancancer_nuclei.data.targets import (
    BACKGROUND_LABEL,
    BOUNDARY_LABEL,
    IGNORE_LABEL,
    INTERIOR_LABEL,
    create_boundary_target,
)


def test_single_nucleus_has_interior_and_boundary() -> None:
    """A square nucleus should produce a border and an interior."""
    instance_masks = np.zeros((1, 9, 9), dtype=bool)
    instance_masks[0, 2:7, 2:7] = True

    target = create_boundary_target(
        instance_masks,
        boundary_width=1,
    )

    assert target[4, 4] == INTERIOR_LABEL
    assert target[2, 2] == BOUNDARY_LABEL
    assert target[0, 0] == BACKGROUND_LABEL
    assert int((target == INTERIOR_LABEL).sum()) == 9
    assert int((target == BOUNDARY_LABEL).sum()) == 16


def test_touching_nuclei_have_separating_boundaries() -> None:
    """Touching instances should retain a boundary between them."""
    instance_masks = np.zeros((2, 7, 9), dtype=bool)
    instance_masks[0, 1:6, 1:4] = True
    instance_masks[1, 1:6, 4:7] = True

    target = create_boundary_target(
        instance_masks,
        boundary_width=1,
    )

    assert target[3, 3] == BOUNDARY_LABEL
    assert target[3, 4] == BOUNDARY_LABEL
    assert target[3, 2] == INTERIOR_LABEL
    assert target[3, 5] == INTERIOR_LABEL


def test_overlapping_annotations_are_ignored() -> None:
    """Pixels assigned to multiple nuclei should use the ignore label."""
    instance_masks = np.zeros((2, 7, 7), dtype=bool)
    instance_masks[0, 1:5, 1:5] = True
    instance_masks[1, 3:6, 3:6] = True

    target = create_boundary_target(
        instance_masks,
        boundary_width=1,
    )

    expected_overlap = instance_masks[0] & instance_masks[1]

    assert np.all(target[expected_overlap] == IGNORE_LABEL)


def test_empty_sample_is_entirely_background() -> None:
    """A patch without nuclei should contain only background."""
    instance_masks = np.empty((0, 8, 8), dtype=bool)

    target = create_boundary_target(instance_masks)

    assert target.shape == (8, 8)
    assert target.dtype == np.uint8
    assert np.all(target == BACKGROUND_LABEL)


def test_tiny_nucleus_can_be_all_boundary() -> None:
    """Very small nuclei may contain no eroded interior."""
    instance_masks = np.zeros((1, 5, 5), dtype=bool)
    instance_masks[0, 2, 2] = True

    target = create_boundary_target(instance_masks)

    assert target[2, 2] == BOUNDARY_LABEL
    assert not np.any(target == INTERIOR_LABEL)


@pytest.mark.parametrize("boundary_width", [0, -1])
def test_invalid_boundary_width_is_rejected(
    boundary_width: int,
) -> None:
    """Boundary width must be a positive integer."""
    instance_masks = np.zeros((1, 5, 5), dtype=bool)

    with pytest.raises(
        ValueError,
        match="boundary_width",
    ):
        create_boundary_target(
            instance_masks,
            boundary_width=boundary_width,
        )


def test_invalid_mask_dimensions_are_rejected() -> None:
    """Instance masks require instance, height, and width dimensions."""
    invalid_masks = np.zeros((8, 8), dtype=bool)

    with pytest.raises(
        ValueError,
        match="number_of_instances",
    ):
        create_boundary_target(invalid_masks)
