"""Post-processing tools for nucleus-instance segmentation."""

from pancancer_nuclei.postprocessing.connected_components import (
    InstanceSegmentation,
    semantic_to_instances,
)

__all__ = [
    "InstanceSegmentation",
    "semantic_to_instances",
]
