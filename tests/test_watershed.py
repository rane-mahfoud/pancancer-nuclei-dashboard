"""Tests for boundary-aware watershed post-processing."""

import numpy as np
import pytest

from pancancer_nuclei.postprocessing.watershed import (
    boundary_predictions_to_instances,
)


def create_probabilities(height: int, width: int) -> np.ndarray:
    """Create background-dominant spatial probabilities."""
    probabilities = np.zeros((3, height, width), dtype=np.float32)
    probabilities[0] = 0.9
    probabilities[1] = 0.05
    probabilities[2] = 0.05
    return probabilities


def test_empty_semantic_mask_returns_no_instances() -> None:
    """Background-only predictions should yield an empty result."""
    semantic = np.zeros((16, 16), dtype=np.int64)
    probabilities = create_probabilities(16, 16)

    result = boundary_predictions_to_instances(semantic, probabilities)

    assert result.number_of_instances == 0
    assert result.instance_map.shape == (16, 16)
    assert not result.instance_map.any()


def test_boundary_separates_touching_nuclei() -> None:
    """Two interior markers should split one touching semantic region."""
    semantic = np.zeros((20, 20), dtype=np.int64)
    semantic[4:16, 3:17] = 3
    probabilities = create_probabilities(20, 20)

    probabilities[:, 4:16, 3:17] = 0.05
    probabilities[1, 5:15, 4:9] = 0.9
    probabilities[1, 5:15, 11:16] = 0.9
    probabilities[2, 4:16, 9:11] = 0.95

    result = boundary_predictions_to_instances(
        semantic,
        probabilities,
        seed_threshold=0.5,
        minimum_seed_area=5,
        minimum_instance_area=10,
    )

    assert result.number_of_instances == 2
    assert result.categories.tolist() == [2, 2]
    assert set(np.unique(result.instance_map).tolist()) == {0, 1, 2}


def test_inflammatory_regions_use_connected_components() -> None:
    """Inflammatory regions should not be over-split by spatial seeds."""
    semantic = np.zeros((20, 20), dtype=np.int64)
    semantic[4:16, 3:17] = 2
    probabilities = create_probabilities(20, 20)

    probabilities[:, 4:16, 3:17] = 0.05
    probabilities[1, 5:15, 4:9] = 0.9
    probabilities[1, 5:15, 11:16] = 0.9
    probabilities[2, 4:16, 9:11] = 0.95

    result = boundary_predictions_to_instances(
        semantic,
        probabilities,
        seed_threshold=0.5,
        minimum_seed_area=5,
        minimum_instance_area=10,
    )

    assert result.number_of_instances == 1
    assert result.categories.tolist() == [1]


def test_touching_semantic_classes_keep_separate_identities() -> None:
    """Watershed basins must not cross predicted class boundaries."""
    semantic = np.zeros((14, 14), dtype=np.int64)
    semantic[3:11, 2:7] = 1
    semantic[3:11, 7:12] = 2
    probabilities = create_probabilities(14, 14)
    probabilities[:, 3:11, 2:12] = 0.05
    probabilities[1, 4:10, 3:11] = 0.9

    result = boundary_predictions_to_instances(
        semantic,
        probabilities,
        seed_threshold=0.5,
        minimum_seed_area=5,
        minimum_instance_area=10,
    )

    assert result.number_of_instances == 2
    assert result.categories.tolist() == [0, 1]


def test_small_instances_are_removed() -> None:
    """The final area filter should remove tiny predicted nuclei."""
    semantic = np.zeros((12, 12), dtype=np.int64)
    semantic[2:5, 2:5] = 1
    probabilities = create_probabilities(12, 12)
    probabilities[1, 2:5, 2:5] = 0.9

    result = boundary_predictions_to_instances(
        semantic,
        probabilities,
        seed_threshold=0.5,
        minimum_seed_area=1,
        minimum_instance_area=10,
    )

    assert result.number_of_instances == 0


@pytest.mark.parametrize(
    ("semantic_shape", "probability_shape"),
    [
        ((2, 8, 8), (3, 8, 8)),
        ((8, 8), (2, 8, 8)),
        ((8, 8), (3, 7, 8)),
    ],
)
def test_invalid_shapes_are_rejected(
    semantic_shape: tuple[int, ...],
    probability_shape: tuple[int, ...],
) -> None:
    """Both inputs must use their documented dimensions."""
    semantic = np.zeros(semantic_shape, dtype=np.int64)
    probabilities = np.zeros(probability_shape, dtype=np.float32)

    with pytest.raises(ValueError):
        boundary_predictions_to_instances(semantic, probabilities)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("seed_threshold", -0.1),
        ("seed_threshold", 1.1),
        ("minimum_seed_area", 0),
        ("minimum_instance_area", 0),
        ("connectivity", 3),
    ],
)
def test_invalid_parameters_are_rejected(keyword: str, value: float) -> None:
    """Invalid watershed settings should fail clearly."""
    semantic = np.zeros((8, 8), dtype=np.int64)
    probabilities = create_probabilities(8, 8)

    with pytest.raises(ValueError):
        boundary_predictions_to_instances(
            semantic,
            probabilities,
            **{keyword: value},
        )
