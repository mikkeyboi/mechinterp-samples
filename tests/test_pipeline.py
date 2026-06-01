"""Model-free tests for the probing pipeline.

None of these load a language model. We synthesise activations with a known
linear structure so the probe machinery, the control task, and the report can be
verified deterministically on CPU in well under a second. This mirrors the
project rule that analysis is decoupled from capture and always unit-testable.
"""
from __future__ import annotations

import numpy as np

from mechinterp_samples import (
    SentimentDataset,
    LinearProbe,
    LayerProbeSweep,
)
from mechinterp_samples.datasets import DatasetSplit


def _synthetic_split(n=200, d=32, seed=0) -> tuple[dict[int, np.ndarray], DatasetSplit]:
    """Two layers: layer 0 is pure noise, layer 1 has a planted linear direction."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    direction = rng.normal(size=d)

    noise_layer = rng.normal(size=(n, d))
    signal_layer = rng.normal(size=(n, d)) + (y[:, None] * 3.0) * direction

    idx = np.arange(n)
    rng.shuffle(idx)
    split = DatasetSplit(
        texts=[f"example {i}" for i in range(n)],
        labels=y,
        train_idx=idx[: n // 2],
        test_idx=idx[n // 2:],
    )
    return {0: noise_layer, 1: signal_layer}, split


def test_dataset_split_is_vocab_disjoint():
    split = SentimentDataset(seed=0).build()
    train_words = " ".join(split.texts_for("train")).lower()
    # held-out adjectives must not appear in any training prompt
    for adj in SentimentDataset.NEG_ADJ[-4:] + SentimentDataset.POS_ADJ[-4:]:
        assert adj not in train_words
    assert split.n_train > 0 and split.n_test > 0


def test_linear_probe_separates_planted_direction():
    acts, split = _synthetic_split()
    probe = LinearProbe(seed=0).fit(acts[1][split.train_idx], split.labels[split.train_idx])
    acc = probe.score(acts[1][split.test_idx], split.labels[split.test_idx])
    assert acc > 0.9


def test_sweep_control_and_selectivity():
    acts, split = _synthetic_split()
    report = LayerProbeSweep(seed=0).run(acts, split, concept="synthetic", model="none")

    # noise layer: probe near chance, low selectivity.
    # signal layer: high accuracy, control near chance, high selectivity.
    by_layer = {r.layer: r for r in report.per_layer}
    assert by_layer[1].test_acc > 0.9
    assert by_layer[1].selectivity > 0.3
    assert by_layer[1].control_test_acc < 0.75
    assert report.best_layer.layer == 1
    assert report.best_selectivity_layer.layer == 1


def test_report_roundtrips_to_json(tmp_path):
    acts, split = _synthetic_split()
    report = LayerProbeSweep(seed=0).run(acts, split, concept="synthetic", model="none")
    out = tmp_path / "report.json"
    report.to_json(out)
    assert out.exists()
    import json
    d = json.loads(out.read_text())
    assert d["concept"] == "synthetic"
    assert "best_selectivity_layer" in d
