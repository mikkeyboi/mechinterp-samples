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
    NegationSentimentDataset,
    LinearProbe,
    LayerProbeSweep,
    DepthProfileContrastPlotter,
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


def test_negation_dataset_is_xor_and_balanced():
    """The composed label must be adjective_polarity XOR negation, and balanced.

    Each adjective contributes equal affirmative (keeps polarity) and negated
    (flips polarity) prompts, so both classes contain both polar adjective sets:
    that is what removes the lexical shortcut the easy concept had.
    """
    ds = NegationSentimentDataset(seed=0)
    split = ds.build()
    pos = int((split.labels == 1).sum())
    neg = int((split.labels == 0).sum())
    assert pos == neg  # perfectly balanced by construction

    # An affirmative prompt with a positive adjective is net-positive; its negated
    # twin with the same adjective is net-negative. Find one such pair by text.
    texts = split.texts
    labels = split.labels
    affirm_pos = [i for i, t in enumerate(texts)
                  if t == "The movie was absolutely wonderful."]
    negate_pos = [i for i, t in enumerate(texts)
                  if t == "The movie was not wonderful at all."]
    assert affirm_pos and negate_pos
    assert labels[affirm_pos[0]] == 1   # "wonderful" affirmative -> positive
    assert labels[negate_pos[0]] == 0   # "not wonderful" -> negative (XOR flip)


def test_negation_dataset_vocab_disjoint():
    """Held-out adjectives must never appear in any training prompt."""
    ds = NegationSentimentDataset(seed=0)
    split = ds.build()
    train_words = " ".join(split.texts_for("train")).lower()
    for adj in ds.NEG_ADJ[-4:] + ds.POS_ADJ[-4:]:
        assert adj not in train_words
    assert split.n_train > 0 and split.n_test > 0


def test_negation_bag_of_tokens_cannot_solve_xor():
    """A linear unigram model cannot represent XOR, so it should sit near chance.

    This is the load-bearing property: the composed concept is *not* a lexical
    lookup, so the bag-of-tokens baseline (logistic regression on token counts)
    must fail where a residual-stream probe on a composed representation succeeds.
    """
    ds = NegationSentimentDataset(seed=0)
    split = ds.build()
    y = split.labels
    # Reach the same baseline the sweep reports, via its static helper.
    baseline = LayerProbeSweep._bag_of_tokens_baseline(
        split.texts_for("train"), y[split.train_idx],
        split.texts_for("test"), y[split.test_idx], seed=0,
    )
    assert baseline < 0.75  # near chance; XOR is not linearly separable in tokens


def test_contrast_plotter_writes_figure(tmp_path):
    """The two-curve contrast figure renders from two reports without a model."""
    acts, split = _synthetic_split()
    rep_a = LayerProbeSweep(seed=0).run(acts, split, concept="A", model="synthetic")
    rep_b = LayerProbeSweep(seed=0).run(acts, split, concept="B", model="synthetic")
    out = tmp_path / "contrast.png"
    path = DepthProfileContrastPlotter().plot(rep_a, rep_b, out)
    assert path.exists() and path.stat().st_size > 0
