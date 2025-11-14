"""
Hypergraph Fusion Module - 支持 Padding + Masking

多模态超图融合网络，用于情感分类任务
"""

from .dataloader import (
    PaddedEmotionDataset,
    padded_collate_fn,
    create_dataloaders,
    load_mosei_data,
    load_meld_data
)

from .modules import (
    HypergraphInitializer,
    HypergraphAugmentation,
    HypergraphConvolution,
    GraphContrastiveLearning,
    BottleneckLayer,
    MultimodalHypergraphLayer
)

from .network import (
    UnimodalEncoder,
    MultimodalHypergraphFusion,
    HypergraphEmotionClassifier
)

__all__ = [
    # DataLoader
    'PaddedEmotionDataset',
    'padded_collate_fn',
    'create_dataloaders',
    'load_mosei_data',
    'load_meld_data',

    # Modules
    'HypergraphInitializer',
    'HypergraphAugmentation',
    'HypergraphConvolution',
    'GraphContrastiveLearning',
    'BottleneckLayer',
    'MultimodalHypergraphLayer',

    # Network
    'UnimodalEncoder',
    'MultimodalHypergraphFusion',
    'HypergraphEmotionClassifier'
]

__version__ = '1.0.0'
