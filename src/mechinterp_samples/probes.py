"""Probes and the per-layer sweep that drives the demo.

`Probe` is an abstract base so the *probe family* is the polymorphic axis: a
linear (logistic-regression) probe ships here, and a reader can drop in an MLP or
a different linear model by subclassing without touching the sweep logic.

The sweep is built around one discipline that separates a real interpretability
claim from a fooled one: the **control task** (Hewitt & Liang 2019; Belinkov
2022). For every layer we train a second probe of identical capacity on
*shuffled* labels and score it on the *real* held-out labels. A genuine concept
direction leaves the control near chance, so **selectivity** (real minus control)
is large. If selectivity is small, the probe's accuracy comes from capacity or
leakage, not from a linearly decodable concept, and the claim is not supported.

A bag-of-tokens baseline guards the other flank: if a residual-stream probe
cannot beat logistic regression on raw token counts, we have shown only that the
*words* differ, not that the *representation* encodes the concept.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@dataclass
class LayerResult:
    """Probe outcome for a single layer, including its control-task score."""

    layer: int
    train_acc: float
    test_acc: float
    control_test_acc: float = -1.0

    @property
    def selectivity(self) -> float:
        """Real held-out accuracy minus the shuffled-label control's accuracy.

        The honest 'is the concept *linearly readable here*' number: it rewards a
        layer only to the extent it beats a same-capacity probe with no signal.
        """
        return self.test_acc - self.control_test_acc


@dataclass
class SweepReport:
    """Serialisable summary of a full per-layer sweep; what a writeup needs."""

    concept: str
    model: str
    n_train: int
    n_test: int
    d_model: int
    n_layers: int
    seed: int
    baseline_test_acc: float
    chance_acc: float
    per_layer: list[LayerResult] = field(default_factory=list)

    @property
    def best_layer(self) -> LayerResult:
        return max(self.per_layer, key=lambda r: r.test_acc)

    @property
    def best_selectivity_layer(self) -> LayerResult:
        return max(self.per_layer, key=lambda r: r.selectivity)

    def to_json(self, path) -> None:
        d = asdict(self)
        d["best_layer"] = asdict(self.best_layer)
        bs = self.best_selectivity_layer
        d["best_selectivity_layer"] = {**asdict(bs), "selectivity": bs.selectivity}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=2))


class Probe(ABC):
    """Abstract probe: fit on (X, y), predict on X. Subclass to add a family."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Probe":
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(accuracy_score(y, self.predict(x)))


class LinearProbe(Probe):
    """L2-regularised logistic regression. The standard linear probe.

    `C` is the inverse regularisation strength; lowering it shrinks capacity,
    which is the knob a selectivity-guided capacity sweep would turn.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 0):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._clf = LogisticRegression(max_iter=max_iter, C=C, random_state=seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearProbe":
        self._clf.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._clf.predict(x)


class LayerProbeSweep:
    """Run a probe + control across every captured layer, plus baselines.

    Parameters
    ----------
    probe_factory:
        A zero-arg callable returning a fresh `Probe`. Defaults to `LinearProbe`.
        Swap it to sweep a different probe family with no other changes.
    """

    def __init__(self, probe_factory=None, seed: int = 0):
        self.probe_factory = probe_factory or (lambda: LinearProbe(seed=seed))
        self.seed = seed

    def _layer_result(self, x_tr, y_tr, x_te, y_te) -> LayerResult:
        probe = self.probe_factory().fit(x_tr, y_tr)

        # Control: same-capacity probe on permuted labels, scored on real labels.
        rng = np.random.default_rng(self.seed)
        y_shuf = rng.permutation(y_tr)
        ctrl = self.probe_factory().fit(x_tr, y_shuf)

        return LayerResult(
            layer=-1,  # set by caller
            train_acc=probe.score(x_tr, y_tr),
            test_acc=probe.score(x_te, y_te),
            control_test_acc=ctrl.score(x_te, y_te),
        )

    @staticmethod
    def _bag_of_tokens_baseline(texts_tr, y_tr, texts_te, y_te, seed=0) -> float:
        vec = CountVectorizer(lowercase=True)
        xb_tr = vec.fit_transform(texts_tr)
        xb_te = vec.transform(texts_te)
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(xb_tr, y_tr)
        return float(accuracy_score(y_te, clf.predict(xb_te)))

    @staticmethod
    def _chance(y) -> float:
        _, counts = np.unique(y, return_counts=True)
        return float(counts.max() / counts.sum())

    def run(self, acts_by_layer, split, concept: str, model: str) -> SweepReport:
        """Execute the sweep. `split` is a datasets.DatasetSplit."""
        labels = split.labels
        tr, te = split.train_idx, split.test_idx
        y_tr, y_te = labels[tr], labels[te]

        layers = sorted(acts_by_layer)
        per_layer: list[LayerResult] = []
        for layer in layers:
            acts = acts_by_layer[layer]
            res = self._layer_result(acts[tr], y_tr, acts[te], y_te)
            res.layer = layer
            per_layer.append(res)

        baseline = self._bag_of_tokens_baseline(
            split.texts_for("train"), y_tr, split.texts_for("test"), y_te,
            seed=self.seed,
        )
        d_model = acts_by_layer[layers[0]].shape[1]

        return SweepReport(
            concept=concept,
            model=model,
            n_train=split.n_train,
            n_test=split.n_test,
            d_model=int(d_model),
            n_layers=len(layers),
            seed=self.seed,
            baseline_test_acc=baseline,
            chance_acc=self._chance(y_te),
            per_layer=per_layer,
        )
