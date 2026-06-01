"""mechinterp-samples: runnable, readable mechanistic-interpretability demos.

This package is the public, showcase-quality counterpart to a private research
vault. It prioritises **readability**: classes over loose functions, clear
encapsulation, and polymorphism where it genuinely earns its keep (concept
datasets and probes are the two axes you are most likely to extend).

The first demo (`samples/h001_linear_probe/`) asks a classic question: is a simple
concept (sentiment) a **linearly decodable direction** in a language model's
residual stream, and if so, *where*? The honest answer it surfaces, that a naive
setup leaks the concept at the embedding layer, is itself the teaching point.
"""
from __future__ import annotations

from .datasets import ConceptDataset, SentimentDataset, DatasetSplit
from .capture import ActivationCapturer, MeanPooler, LastTokenPooler
from .probes import LinearProbe, LayerProbeSweep, LayerResult, SweepReport
from .plotting import LayerCurvePlotter

__all__ = [
    "ConceptDataset",
    "SentimentDataset",
    "DatasetSplit",
    "ActivationCapturer",
    "MeanPooler",
    "LastTokenPooler",
    "LinearProbe",
    "LayerProbeSweep",
    "LayerResult",
    "SweepReport",
    "LayerCurvePlotter",
]

__version__ = "0.1.0"
