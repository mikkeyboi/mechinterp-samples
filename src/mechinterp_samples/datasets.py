"""Concept datasets for probing demos.

The probing question is always "is *this concept* linearly readable", so the
concept is the natural polymorphic axis. `ConceptDataset` is an abstract base;
each concrete concept (sentiment here, others later) implements `build()` and
declares how its train/test split is formed.

A central design choice lives here: the split is **vocabulary-disjoint** by
default. Words used in test prompts never appear in training prompts, which
forces a probe to generalise the *concept direction* rather than memorise a
lexical shortcut. This is what makes the eventual "it leaked at layer 0 anyway"
finding meaningful: even a vocab-disjoint split does not stop the embedding layer
from carrying the concept.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    """An immutable train/test split over a parallel (texts, labels) corpus.

    `train_idx` / `test_idx` index into `texts` and `labels`. Keeping the corpus
    flat and carrying index arrays (rather than four separate lists) means the
    activation capturer runs the model **once** over every text and the probe
    then slices the cached activations, so capture cost does not double.
    """

    texts: list[str]
    labels: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def n_train(self) -> int:
        return int(len(self.train_idx))

    @property
    def n_test(self) -> int:
        return int(len(self.test_idx))

    def texts_for(self, which: str) -> list[str]:
        idx = self.train_idx if which == "train" else self.test_idx
        return [self.texts[i] for i in idx]


class ConceptDataset(ABC):
    """Abstract base for a binary concept-probing dataset.

    Subclass to add a concept: implement `build()` to return a `DatasetSplit`.
    The label convention is 1 = concept present (e.g. positive sentiment),
    0 = absent. Keep prompts short; activation capture cost scales with tokens.
    """

    #: Human-readable concept name, used in reports and figure titles.
    name: str = "concept"

    def __init__(self, seed: int = 0):
        self.seed = seed

    @abstractmethod
    def build(self) -> DatasetSplit:
        """Construct and return the dataset split. Deterministic given `seed`."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, seed={self.seed})"


class SentimentDataset(ConceptDataset):
    """Sentiment (positive vs negative) from polar-adjective templates.

    Sentiment is unambiguous, well studied as a probe target, and short prompts
    keep capture cheap on a small GPU. Labels are noise-free by construction;
    the question is whether the *representation* is linearly separable, not
    whether the task is hard. The split holds out a quarter of the adjectives in
    each class so test prompts use **unseen words**.
    """

    name = "sentiment"

    POS_ADJ = [
        "wonderful", "fantastic", "delightful", "superb", "excellent", "brilliant",
        "lovely", "amazing", "gorgeous", "marvelous", "outstanding", "charming",
        "terrific", "splendid", "magnificent", "joyful",
    ]
    NEG_ADJ = [
        "terrible", "awful", "horrible", "dreadful", "miserable", "disgusting",
        "atrocious", "appalling", "hideous", "lousy", "abysmal", "vile",
        "repugnant", "horrendous", "gloomy", "wretched",
    ]
    TEMPLATES = [
        "The movie was absolutely {adj}.",
        "Honestly, that meal felt {adj} to me.",
        "What a {adj} experience that turned out to be.",
        "I found the whole trip rather {adj}.",
        "Everyone agreed the show was {adj}.",
        "My day has been completely {adj}.",
    ]

    def build(self) -> DatasetSplit:
        rng = np.random.default_rng(self.seed)

        n_hold = len(self.POS_ADJ) // 4  # 4 of 16 adjectives held out per class
        pos_train, pos_test = self.POS_ADJ[:-n_hold], self.POS_ADJ[-n_hold:]
        neg_train, neg_test = self.NEG_ADJ[:-n_hold], self.NEG_ADJ[-n_hold:]

        texts: list[str] = []
        labels: list[int] = []
        split: list[str] = []

        def emit(adjs: list[str], label: int, which: str) -> None:
            for adj in adjs:
                for tmpl in self.TEMPLATES:
                    texts.append(tmpl.format(adj=adj))
                    labels.append(label)
                    split.append(which)

        emit(pos_train, 1, "train")
        emit(neg_train, 0, "train")
        emit(pos_test, 1, "test")
        emit(neg_test, 0, "test")

        labels_arr = np.array(labels, dtype=np.int64)
        split_arr = np.array(split)
        train_idx = np.where(split_arr == "train")[0]
        test_idx = np.where(split_arr == "test")[0]
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        return DatasetSplit(
            texts=texts, labels=labels_arr,
            train_idx=train_idx, test_idx=test_idx,
        )
