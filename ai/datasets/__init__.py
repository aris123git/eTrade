"""
ai/datasets - Production dataset construction for AI models.

VERSION: 1.0.0
"""

from ai.datasets.alignment import (
    AlignedFeatureBlock,
    align_feature_blocks,
    align_timeframe_to_primary,
    asof_indices,
    merge_aligned_features,
)
from ai.datasets.batching import BatchIterator, TimeSeriesBatch
from ai.datasets.builder import DatasetBuilder, FeatureEngineLike, LabelGeneratorLike, create_dataset_builder
from ai.datasets.schema import DatasetBundle, empty_bundle
from ai.datasets.walk_forward import WalkForwardDataset, WalkForwardFold
from ai.datasets.windows import (
    SequenceData,
    WindowSpec,
    apply_stride,
    generate_sequences,
    iter_sliding_windows,
    make_sliding_windows,
    sliding_window_indices,
)

__all__ = [
    "AlignedFeatureBlock",
    "align_feature_blocks",
    "align_timeframe_to_primary",
    "asof_indices",
    "merge_aligned_features",
    "BatchIterator",
    "TimeSeriesBatch",
    "DatasetBuilder",
    "FeatureEngineLike",
    "LabelGeneratorLike",
    "create_dataset_builder",
    "DatasetBundle",
    "empty_bundle",
    "WalkForwardDataset",
    "WalkForwardFold",
    "SequenceData",
    "WindowSpec",
    "apply_stride",
    "generate_sequences",
    "iter_sliding_windows",
    "make_sliding_windows",
    "sliding_window_indices",
]
