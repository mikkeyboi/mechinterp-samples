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


class NegationSentimentDataset(ConceptDataset):
    """Negation-composed sentiment: a concept that cannot be read off one word.

    `SentimentDataset` turned out to be *too easy*: polar adjectives are already
    linearly separable in embedding space, so a probe scores at the ceiling from
    layer 0, before the transformer computes anything. That tells us sentiment is
    linearly present but nothing about *where* it is built.

    This dataset closes that shortcut by making the label a **composition** of two
    tokens rather than a property of either:

        net_sentiment = adjective_polarity  XOR  negated

    so "wonderful" (positive) and "not wonderful" (negative) share an adjective
    but carry opposite labels, and "terrible" / "not terrible" likewise. Now:

    * Both classes contain both polar adjective sets, so the adjective alone is
      uninformative and a probe on the raw embedding has no lexical shortcut.
    * A bag-of-tokens baseline sees the "not" token and the adjective token, but
      the target is their XOR, which a linear unigram model cannot represent, so
      that baseline sits at chance too.
    * Only a representation that has actually *composed* negation with the
      adjective encodes net sentiment linearly. That composition is built across
      depth, so we expect the rise-then-plateau accuracy-by-layer curve: chance
      at the embedding, rising to a ceiling by the mid stack.

    Same probe protocol as `SentimentDataset` (per-layer logistic probe,
    shuffled-label control, vocab-disjoint split), so only the concept changes.
    This is the honest, informative companion to the lexical case.
    """

    name = "sentiment-negation-xor"

    # Reuse the lexical adjective sets verbatim so the *only* change from the
    # easy concept is the negation composition, not a different vocabulary.
    POS_ADJ = SentimentDataset.POS_ADJ
    NEG_ADJ = SentimentDataset.NEG_ADJ

    # Affirmative and negated carrier sentences kept parallel, so the only
    # systematic difference between an affirmative prompt and its negation is the
    # negation itself (which flips the adjective's polarity).
    AFFIRM_TEMPLATES = [
        "The movie was absolutely {adj}.",
        "Honestly, that meal felt {adj} to me.",
        "What a {adj} experience that turned out to be.",
        "Everyone agreed the show was {adj}.",
    ]
    NEGATED_TEMPLATES = [
        "The movie was not {adj} at all.",
        "Honestly, that meal did not feel {adj} to me.",
        "That experience was not {adj} in the slightest.",
        "Nobody agreed the show was {adj}.",
    ]

    def build(self) -> DatasetSplit:
        rng = np.random.default_rng(self.seed)

        n_hold = len(self.POS_ADJ) // 4  # 4 of 16 adjectives held out per class
        pos_train, pos_test = self.POS_ADJ[:-n_hold], self.POS_ADJ[-n_hold:]
        neg_train, neg_test = self.NEG_ADJ[:-n_hold], self.NEG_ADJ[-n_hold:]

        texts: list[str] = []
        labels: list[int] = []
        split: list[str] = []

        def emit(adjs: list[str], polarity_pos: bool, which: str) -> None:
            """Emit affirmative and negated prompts for a set of adjectives.

            `polarity_pos` is True for positive adjectives. Affirmative keeps the
            polarity as the label; negation flips it (the XOR).
            """
            for adj in adjs:
                for tmpl in self.AFFIRM_TEMPLATES:
                    texts.append(tmpl.format(adj=adj))
                    labels.append(1 if polarity_pos else 0)
                    split.append(which)
                for tmpl in self.NEGATED_TEMPLATES:
                    texts.append(tmpl.format(adj=adj))
                    labels.append(0 if polarity_pos else 1)
                    split.append(which)

        emit(pos_train, True, "train")
        emit(neg_train, False, "train")
        emit(pos_test, True, "test")
        emit(neg_test, False, "test")

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
