"""Output accessibility diagnostics for readable-versus-steerable directions.

A trained probe can read any linear basis it learns. A logit lens asks a stricter
question: is the same concept already aligned with the model's own output basis?
This module keeps that comparison model-free and testable. The real-model demo
only supplies per-example probe scores and logit-lens scores.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _midranks(values: np.ndarray) -> np.ndarray:
    """Zero-indexed average ranks with exact tie handling."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def rank_auc(scores, labels) -> float:
    """P(a random positive scores above a random negative), ties worth one half."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("scores and labels must be same-length one-dimensional arrays")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("rank AUC needs both positive and negative examples")
    ranks = _midranks(scores)
    rank_sum_pos = float(ranks[labels == 1].sum()) + n_pos
    mann_whitney = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(mann_whitney / (n_pos * n_neg))


@dataclass(frozen=True)
class LayerAccessibility:
    """Probe readability and output accessibility at one residual-stream layer."""

    layer: int
    probe_accuracy: float
    logit_lens_auc: float
    regime: str


@dataclass
class AccessibilityReport:
    """Serializable depth profile with a deliberately bounded interpretation."""

    model: str
    concept: str
    chance_auc: float
    layers: list[LayerAccessibility]
    readable_onset_layer: int
    selected_steering_layer: int | None
    selected_site_is_descriptive: bool = True

    @property
    def readable_onset(self) -> LayerAccessibility:
        return next(x for x in self.layers if x.layer == self.readable_onset_layer)

    @property
    def selected_site(self) -> LayerAccessibility | None:
        if self.selected_steering_layer is None:
            return None
        return next(x for x in self.layers if x.layer == self.selected_steering_layer)

    def to_json(self, path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2))
        return out

    def summary_lines(self) -> list[str]:
        onset = self.readable_onset
        lines = [
            f"readable onset L{onset.layer}: probe={onset.probe_accuracy:.3f}, "
            f"logit-lens AUC={onset.logit_lens_auc:.3f} ({onset.regime})"
        ]
        site = self.selected_site
        if site is not None:
            lines.append(
                f"selected steering site L{site.layer}: probe={site.probe_accuracy:.3f}, "
                f"logit-lens AUC={site.logit_lens_auc:.3f} ({site.regime})"
            )
            lines.append(
                "This onset/site contrast is descriptive because the steering site "
                "was selected using steering measurements."
            )
        return lines


class OutputAccessibilityAnalyzer:
    """Compare a learned probe readout with the model's output-basis readout.

    ``probe_scores_by_layer`` and ``lens_scores_by_layer`` map residual-stream
    layer indices to one scalar per example. The analyzer fits a threshold and
    orientation for the probe score on a training split, evaluates held-out probe
    accuracy, and computes a held-out rank AUC for the logit-lens score.
    """

    def __init__(self, present_accuracy: float = 0.75, aligned_auc: float = 0.75):
        self.present_accuracy = present_accuracy
        self.aligned_auc = aligned_auc

    @staticmethod
    def _fit_probe_threshold(scores, labels) -> tuple[float, int]:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        unique = np.unique(scores)
        if len(unique) == 1:
            thresholds = unique
        else:
            thresholds = np.r_[
                unique[0] - 1.0,
                0.5 * (unique[:-1] + unique[1:]),
                unique[-1] + 1.0,
            ]
        best = (-1.0, 0.0, 1)
        for orientation in (1, -1):
            for threshold in thresholds:
                pred = (orientation * scores >= orientation * threshold).astype(int)
                acc = float((pred == labels).mean())
                candidate = (acc, -abs(float(threshold)), orientation)
                if candidate > (best[0], -abs(best[1]), best[2]):
                    best = (acc, float(threshold), orientation)
        return best[1], best[2]

    def _regime(self, probe_accuracy: float, lens_auc: float) -> str:
        if probe_accuracy < self.present_accuracy:
            return "not-readable"
        if lens_auc >= self.aligned_auc:
            return "readable-and-output-aligned"
        if lens_auc < 0.5:
            return "readable-but-output-anti-aligned"
        return "readable-but-not-output-aligned"

    def analyze(
        self,
        probe_scores_by_layer: dict[int, np.ndarray],
        lens_scores_by_layer: dict[int, np.ndarray],
        labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        *,
        model: str,
        concept: str,
        readable_ceiling: float = 0.95,
        selected_steering_layer: int | None = None,
    ) -> AccessibilityReport:
        layers = sorted(probe_scores_by_layer)
        if layers != sorted(lens_scores_by_layer):
            raise ValueError("probe and logit-lens profiles must cover identical layers")
        labels = np.asarray(labels, dtype=int)
        rows: list[LayerAccessibility] = []
        for layer in layers:
            probe_scores = np.asarray(probe_scores_by_layer[layer], dtype=float)
            lens_scores = np.asarray(lens_scores_by_layer[layer], dtype=float)
            if probe_scores.shape != labels.shape or lens_scores.shape != labels.shape:
                raise ValueError("every layer must have one probe and lens score per label")
            threshold, orientation = self._fit_probe_threshold(
                probe_scores[train_idx], labels[train_idx]
            )
            pred = (
                orientation * probe_scores[test_idx] >= orientation * threshold
            ).astype(int)
            accuracy = float((pred == labels[test_idx]).mean())
            auc = rank_auc(lens_scores[test_idx], labels[test_idx])
            rows.append(
                LayerAccessibility(layer, accuracy, auc, self._regime(accuracy, auc))
            )

        readable = [row.layer for row in rows if row.probe_accuracy >= readable_ceiling]
        if not readable:
            raise ValueError("no layer reaches the requested readable ceiling")
        if selected_steering_layer is not None and selected_steering_layer not in layers:
            raise ValueError("selected steering layer is absent from the supplied profile")
        return AccessibilityReport(
            model=model,
            concept=concept,
            chance_auc=0.5,
            layers=rows,
            readable_onset_layer=min(readable),
            selected_steering_layer=selected_steering_layer,
        )
