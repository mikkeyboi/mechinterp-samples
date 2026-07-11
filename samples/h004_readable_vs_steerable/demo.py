"""H004 demo: a readable direction need not be an output-aligned lever.

The offline default builds a planted depth profile. A linear readout becomes
perfect by layer 2, but its score is reversed in the output-token basis there.
The output-basis score becomes aligned only deeper. This reproduces the shape of
the real observation without claiming that a selected site validates a layer
selector.

    python demo.py
    python demo.py --real --model google/gemma-3-1b-it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import mechinterp_samples  # noqa: F401
except ModuleNotFoundError:
    _SRC = Path(__file__).resolve().parents[2] / "src"
    if _SRC.is_dir():
        sys.path.insert(0, str(_SRC))

from mechinterp_samples import (
    AccessibilityPlotter,
    ActivationCapturer,
    MeanPooler,
    OutputAccessibilityAnalyzer,
)

HERE = Path(__file__).parent


def _balanced_split(n: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], n // 2)
    idx = rng.permutation(n)
    midpoint = n // 2
    return labels, idx[:midpoint], idx[midpoint:]


def run_synthetic(seed: int = 0):
    """Analyze a planted readable-early, output-aligned-late profile."""
    rng = np.random.default_rng(seed)
    labels, train_idx, test_idx = _balanced_split(240, seed)
    signed = 2 * labels - 1
    probe_strength = [0.0, 0.8, 2.5, 3.0, 3.2, 3.4]
    output_strength = [-0.2, -0.5, -2.0, -0.4, 1.2, 2.5]
    probe_scores = {
        layer: strength * signed + rng.normal(scale=0.45, size=len(labels))
        for layer, strength in enumerate(probe_strength)
    }
    lens_scores = {
        layer: strength * signed + rng.normal(scale=0.55, size=len(labels))
        for layer, strength in enumerate(output_strength)
    }
    return OutputAccessibilityAnalyzer().analyze(
        probe_scores,
        lens_scores,
        labels,
        train_idx,
        test_idx,
        model="synthetic planted profile",
        concept="composed polarity",
        selected_steering_layer=5,
    )


def _concessive_dataset(seed: int = 0):
    """Token-matched order pairs whose label is the asserted clause polarity."""
    positive = [
        "wonderful", "excellent", "delightful", "beautiful", "enjoyable",
        "superb", "charming", "splendid",
    ]
    negative = [
        "terrible", "awful", "dreary", "ugly", "tedious", "dismal", "boring", "grim",
    ]
    texts, labels = [], []
    for pos, neg in zip(positive, negative):
        texts.extend(
            [
                f"Despite being {neg}, the movie was {pos}.",
                f"Despite being {pos}, the movie was {neg}.",
            ]
        )
        labels.extend([1, 0])
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    pair_order = rng.permutation(len(positive))
    train_pairs = set(pair_order[: len(positive) // 2])
    train_idx, test_idx = [], []
    for pair in range(len(positive)):
        target = train_idx if pair in train_pairs else test_idx
        target.extend([2 * pair, 2 * pair + 1])
    return texts, labels, np.asarray(train_idx), np.asarray(test_idx)


def _first_token_ids(tokenizer, words):
    ids = []
    for word in words:
        pieces = tokenizer.encode(" " + word, add_special_tokens=False)
        if not pieces:
            raise ValueError(f"tokenizer produced no id for {word!r}")
        ids.append(pieces[0])
    return ids


def run_real(model_name: str, seed: int = 0):
    """Compute probe and logit-lens profiles in one model forward pass per batch.

    The model's final hidden state must already be post-final-norm. This is checked
    against the emitted logits before intermediate raw residuals are normalized
    once and passed through the unembedding.
    """
    import torch
    from sklearn.linear_model import LogisticRegression

    texts, labels, train_idx, test_idx = _concessive_dataset(seed)
    capturer = ActivationCapturer(model_name, pooler=MeanPooler(), batch_size=8)
    acts = capturer.capture(texts)
    model, tokenizer = capturer.model, capturer.tokenizer
    final_norm = getattr(model.model, "norm", None)
    if final_norm is None:
        raise RuntimeError("this demo expects the model's final norm at model.model.norm")
    output = model.get_output_embeddings()

    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        capturer.device
    )
    with torch.no_grad():
        result = model(**encoded, output_hidden_states=True)
        emitted = result.logits[:, -1, :].float()
        reconstructed = output(result.hidden_states[-1][:, -1, :]).float()
        correlation = float(
            np.corrcoef(
                emitted.detach().cpu().numpy().ravel(),
                reconstructed.detach().cpu().numpy().ravel(),
            )[0, 1]
        )
        if not np.isfinite(correlation) or correlation < 0.999:
            raise RuntimeError(
                "logit-lens invariant failed: unembedding the final hidden state "
                f"does not reconstruct logits (Pearson r={correlation:.6f})"
            )
        mask = encoded["attention_mask"].unsqueeze(-1)
        lens_scores = {}
        pos_ids = _first_token_ids(tokenizer, ["positive", "good", "yes"])
        neg_ids = _first_token_ids(tokenizer, ["negative", "bad", "no"])
        for layer in sorted(acts):
            hidden = result.hidden_states[layer]
            weights = mask.to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
            logits = output(final_norm(pooled)).float()
            lens_scores[layer] = (
                torch.logsumexp(logits[:, pos_ids], dim=-1)
                - torch.logsumexp(logits[:, neg_ids], dim=-1)
            ).cpu().numpy()

    probe_scores = {}
    for layer, values in acts.items():
        classifier = LogisticRegression(max_iter=2000, random_state=seed)
        classifier.fit(values[train_idx], labels[train_idx])
        probe_scores[layer] = classifier.decision_function(values)

    report = OutputAccessibilityAnalyzer().analyze(
        probe_scores,
        lens_scores,
        labels,
        train_idx,
        test_idx,
        model=model_name,
        concept="concessive net sentiment",
        selected_steering_layer=None,
    )
    print(f"logit-lens invariant Pearson r={correlation:.6f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="run a real model (GPU recommended)")
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=str(HERE / "figures" / "readable_vs_output_aligned.png"),
    )
    args = parser.parse_args()

    report = run_real(args.model, args.seed) if args.real else run_synthetic(args.seed)
    for line in report.summary_lines():
        print(line)
    json_path = report.to_json(HERE / "figures" / "readable_vs_output_aligned.json")
    figure_path = AccessibilityPlotter().plot(report, args.out)
    print(f"wrote report -> {json_path}")
    print(f"wrote figure -> {figure_path}")


if __name__ == "__main__":
    main()
