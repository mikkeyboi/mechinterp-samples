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


class DepthProfileContrastPlotter:
    """Overlay two concepts' accuracy-by-layer curves to show the *shape* contrast.

    This is the figure that carries the H001 depth-profile argument: a lexical
    concept (separable at the embedding, flat at the ceiling) against a composed
    concept (chance at the embedding, rising to a ceiling by the mid stack). Same
    model, same probe, same 1.000 ceiling at the top, opposite meaning. The top
    panel is held-out accuracy; the bottom panel is selectivity (real minus the
    shuffled-label control), which confirms the rise is representation, not probe
    capacity.

    Pass the two `SweepReport`s in the order (lexical, composed); the labels and
    annotations assume that order.
    """

    #: red for the lexical concept, blue for the composed concept (colourblind-safe).
    C_LEXICAL = "#c44e52"
    C_COMPOSED = "#4c72b0"

    def __init__(self, dpi: int = 150, figsize=(8, 6.4)):
        self.dpi = dpi
        self.figsize = figsize

    @staticmethod
    def _curve(report: SweepReport):
        layers = [r.layer for r in report.per_layer]
        acc = [r.test_acc for r in report.per_layer]
        sel = [r.selectivity for r in report.per_layer]
        return layers, acc, sel

    def plot(self, lexical: SweepReport, composed: SweepReport, out_path) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        L1, acc1, sel1 = self._curve(lexical)
        L2, acc2, sel2 = self._curve(composed)

        fig, (ax, axs) = plt.subplots(
            2, 1, figsize=self.figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
        )

        # Top: held-out accuracy by layer.
        ax.plot(L1, acc1, "-o", color=self.C_LEXICAL, ms=4, lw=2,
                label="Lexical sentiment: pos vs neg")
        ax.plot(L2, acc2, "-o", color=self.C_COMPOSED, ms=4, lw=2,
                label="Composed sentiment: polarity XOR negation")
        ax.axhline(composed.chance_acc, color="gray", ls="--", lw=1,
                   label=f"Chance ({composed.chance_acc:.3f})")
        ax.set_ylabel("Held-out probe accuracy")
        ax.set_ylim(0.45, 1.03)
        ax.set_title(f"Where a concept becomes linearly readable ({composed.model})")
        ax.legend(loc="center right", fontsize=9, framealpha=0.95)
        ax.grid(alpha=0.25)

        # Annotate the two shapes (use each curve's own best layer for the arrow).
        ax.annotate("flat at ceiling from the embedding\n(no computation needed)",
                    xy=(lexical.best_layer.layer, lexical.best_layer.test_acc),
                    xytext=(3.5, 0.74), fontsize=8.5, color=self.C_LEXICAL,
                    arrowprops=dict(arrowstyle="->", color=self.C_LEXICAL, lw=1))
        ax.annotate("chance at the embedding,\nrises then plateaus",
                    xy=(composed.best_layer.layer, composed.best_layer.test_acc),
                    xytext=(max(composed.best_layer.layer + 2, 10), 0.62),
                    fontsize=8.5, color=self.C_COMPOSED,
                    arrowprops=dict(arrowstyle="->", color=self.C_COMPOSED, lw=1))

        # Bottom: selectivity (real minus shuffled-label control).
        axs.plot(L1, sel1, "-o", color=self.C_LEXICAL, ms=3, lw=1.5)
        axs.plot(L2, sel2, "-o", color=self.C_COMPOSED, ms=3, lw=1.5)
        axs.axhline(0.0, color="gray", ls="--", lw=1)
        axs.set_ylabel("Selectivity\n(real - shuffled)", fontsize=9)
        axs.set_xlabel("Residual-stream layer (0 = input embedding)")
        axs.grid(alpha=0.25)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out
