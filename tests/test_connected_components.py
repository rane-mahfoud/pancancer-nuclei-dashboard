"""Tests for semantic-to-instance post-processing."""

import numpy as np
import pytest

from pancancer_nuclei.postprocessing.connected_components import (
    semantic_to_instances,
)


def test_disconnected_regions_become_separate_instances() -> None:
    """Disconnected regions of one class should become separate nuclei."""
    semantic_mask = np.zeros((8, 8), dtype=np.uint8)
    semantic_mask[1:3, 1:3] = 1
    semantic_mask[5:7, 5:7] = 1

    result = semantic_to_instances(
        semantic_mask,
        minimum_area=1,
    )

    assert result.number_of_instances == 2
    np.testing.assert_array_equal(
        result.categories,
        np.array([0, 0]),
    )
    assert set(np.unique(result.instance_map)) == {0, 1, 2}


def test_adjacent_different_classes_remain_separate() -> None:
    """Touching regions with different classes should remain separate."""
    semantic_mask = np.zeros((6, 8), dtype=np.uint8)
    semantic_mask[1:5, 1:4] = 1
    semantic_mask[1:5, 4:7] = 2

    result = semantic_to_instances(
        semantic_mask,
        minimum_area=1,
    )

    assert result.number_of_instances == 2
    np.testing.assert_array_equal(
        result.categories,
        np.array([0, 1]),
    )
    assert not np.any(result.instance_masks[0] & result.instance_masks[1])


def test_small_regions_are_removed() -> None:
    """Regions smaller than the area threshold should be discarded."""
    semantic_mask = np.zeros((8, 8), dtype=np.uint8)
    semantic_mask[1, 1] = 5
    semantic_mask[4:6, 4:6] = 5

    result = semantic_to_instances(
        semantic_mask,
        minimum_area=2,
    )

    assert result.number_of_instances == 1
    np.testing.assert_array_equal(
        result.categories,
        np.array([4]),
    )
    assert int(result.instance_masks[0].sum()) == 4


def test_empty_and_ignored_mask_produces_no_instances() -> None:
    """Background and ignored pixels should not become nuclei."""
    semantic_mask = np.zeros((5, 5), dtype=np.uint8)
    semantic_mask[1:3, 1:3] = 255

    result = semantic_to_instances(
        semantic_mask,
        minimum_area=1,
    )

    assert result.number_of_instances == 0
    assert result.instance_masks.shape == (0, 5, 5)
    assert result.categories.shape == (0,)
    assert not result.instance_map.any()


def test_invalid_semantic_label_is_rejected() -> None:
    """Unexpected labels should produce a clear error."""
    semantic_mask = np.zeros((5, 5), dtype=np.int64)
    semantic_mask[2, 2] = 17

    with pytest.raises(
        ValueError,
        match="invalid labels",
    ):
        semantic_to_instances(semantic_mask)


@pytest.mark.parametrize("connectivity", [0, 3])
def test_invalid_connectivity_is_rejected(
    connectivity: int,
) -> None:
    """Only four- or eight-neighbour connectivity is accepted."""
    semantic_mask = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match="connectivity",
    ):
        semantic_to_instances(
            semantic_mask,
            connectivity=connectivity,
        )
