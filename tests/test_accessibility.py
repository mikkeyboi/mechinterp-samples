"""Model-free tests for output accessibility and readable-versus-steerable claims."""
from __future__ import annotations

import numpy as np
import pytest

from mechinterp_samples import AccessibilityPlotter, OutputAccessibilityAnalyzer, rank_auc


def _profile(seed=0):
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], 100)
    signed = 2 * labels - 1
    train_idx = np.arange(0, 100)
    test_idx = np.arange(100, 200)
    probe = {
        0: rng.normal(size=200),
        1: 3.0 * signed + rng.normal(scale=0.3, size=200),
        2: 3.0 * signed + rng.normal(scale=0.3, size=200),
    }
    lens = {
        0: rng.normal(size=200),
        1: -2.0 * signed + rng.normal(scale=0.3, size=200),
        2: 2.0 * signed + rng.normal(scale=0.3, size=200),
    }
    report = OutputAccessibilityAnalyzer().analyze(
        probe,
        lens,
        labels,
        train_idx,
        test_idx,
        model="synthetic",
        concept="polarity",
        selected_steering_layer=2,
    )
    return report


def test_rank_auc_handles_ties():
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 1, 0, 1])
    assert rank_auc(scores, labels) == pytest.approx(0.5)


def test_analyzer_separates_readability_from_output_alignment():
    report = _profile()
    assert report.readable_onset_layer == 1
    assert report.readable_onset.probe_accuracy == 1.0
    assert report.readable_onset.logit_lens_auc < 0.5
    assert report.readable_onset.regime == "readable-but-output-anti-aligned"
    site = report.selected_site
    assert site is not None
    assert site.logit_lens_auc > 0.75
    assert site.regime == "readable-and-output-aligned"
    assert report.selected_site_is_descriptive is True


def test_analyzer_rejects_mismatched_profiles():
    labels = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="identical layers"):
        OutputAccessibilityAnalyzer().analyze(
            {0: labels},
            {1: labels},
            labels,
            np.array([0, 1]),
            np.array([2, 3]),
            model="x",
            concept="y",
        )


def test_report_and_plot_roundtrip(tmp_path):
    report = _profile()
    json_path = report.to_json(tmp_path / "report.json")
    figure_path = AccessibilityPlotter().plot(report, tmp_path / "profile.png")
    assert json_path.exists() and json_path.stat().st_size > 0
    assert figure_path.exists() and figure_path.stat().st_size > 0
    lines = report.summary_lines()
    assert any("descriptive" in line for line in lines)
