"""Narrative plotting for the layer sweep.

`LayerCurvePlotter` turns a `SweepReport` into the figure that *is* the demo's
argument: accuracy as a function of residual-stream depth, with the control,
bag-of-tokens baseline, and chance drawn as reference lines. The shape of this
curve is the whole point. A concept the model *computes* should be hard to read
at the embedding layer and rise with depth; a concept that is merely *lexically
present* is readable from layer 0. The plot makes that distinction visible.
"""
from __future__ import annotations

from pathlib import Path

from .probes import SweepReport


class LayerCurvePlotter:
    """Render and save the accuracy-by-layer curve for a SweepReport."""

    def __init__(self, dpi: int = 140, figsize=(8, 5)):
        self.dpi = dpi
        self.figsize = figsize

    def plot(self, report: SweepReport, out_path) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        layers = [r.layer for r in report.per_layer]
        test_acc = [r.test_acc for r in report.per_layer]
        train_acc = [r.train_acc for r in report.per_layer]
        control_acc = [r.control_test_acc for r in report.per_layer]

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(layers, test_acc, "o-", label="probe (held-out)")
        ax.plot(layers, train_acc, "o--", alpha=0.4, label="probe (train)")
        ax.plot(layers, control_acc, "s-", color="tab:purple", alpha=0.6,
                label="control (shuffled labels)")
        ax.axhline(report.baseline_test_acc, color="tab:red", ls=":",
                   label=f"bag-of-tokens ({report.baseline_test_acc:.2f})")
        ax.axhline(report.chance_acc, color="gray", ls="--",
                   label=f"chance ({report.chance_acc:.2f})")
        ax.set_xlabel("residual-stream layer (0 = embedding)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Linear decodability by layer: {report.concept} ({report.model})")
        ax.legend(loc="lower right")
        fig.tight_layout()

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out
