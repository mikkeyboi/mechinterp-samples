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
from .refusal import ReproductionReport


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

    @staticmethod
    def _validate_comparable(lexical: SweepReport, composed: SweepReport) -> None:
        """Fail fast if the two reports are not on the same axis.

        The contrast figure only means something if both sweeps were run over the
        same residual-stream layers (and, in practice, the same model). If a caller
        passes mismatched reports the overlay would silently mislead, so we refuse
        rather than draw it.
        """
        l_layers = [r.layer for r in lexical.per_layer]
        c_layers = [r.layer for r in composed.per_layer]
        if l_layers != c_layers:
            raise ValueError(
                "Cannot contrast sweeps over different layers: "
                f"lexical has {len(l_layers)} layers {l_layers[:3]}..., "
                f"composed has {len(c_layers)} layers {c_layers[:3]}.... "
                "Both sweeps must cover the same residual-stream points."
            )
        if lexical.model != composed.model:
            raise ValueError(
                "Refusing to contrast sweeps from different models "
                f"({lexical.model!r} vs {composed.model!r}); the figure would "
                "compare apples to oranges. Re-run both on the same model."
            )

    def plot(self, lexical: SweepReport, composed: SweepReport, out_path) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._validate_comparable(lexical, composed)
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


class RefusalReproductionPlotter:
    """Two-panel bar figure for the single-direction refusal reproduction.

    Left panel (suppression, on harmful prompts): baseline refusal, refusal after
    ablating the real direction, and refusal after ablating a norm-matched random
    control. Right panel (induction, on harmless prompts): baseline, after adding
    the real direction, after adding the control. The story the figure has to tell
    is the gap between the real bar and the control bar: a single-direction result
    is one where erasing/adding the real direction moves refusal hard while the
    equal-norm random direction does not.
    """

    C_BASELINE = "#777777"
    C_REAL_SUPPRESS = "#1f77b4"
    C_CTRL_SUPPRESS = "#aec7e8"
    C_REAL_INDUCE = "#d62728"
    C_CTRL_INDUCE = "#ff9896"

    def __init__(self, dpi: int = 140, figsize=(11, 4.5)):
        self.dpi = dpi
        self.figsize = figsize

    def plot(self, report: ReproductionReport, out_path) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        real, control = report.real, report.control

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Left: suppression on the harmful set.
        labels = ["baseline", "ablation\n(real dir)", "ablation\n(random ctrl)"]
        vals = [
            real.baseline_harmful_refusal,
            real.ablation_harmful_refusal,
            control.ablation_harmful_refusal,
        ]
        ax1.bar(labels, vals, color=[self.C_BASELINE, self.C_REAL_SUPPRESS, self.C_CTRL_SUPPRESS])
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("refusal rate (harmful prompts)")
        ax1.set_title("Suppression: erase the refusal direction")
        ax1.axhline(real.baseline_harmful_refusal, ls="--", c=self.C_BASELINE, lw=1)
        for i, val in enumerate(vals):
            ax1.text(i, val + 0.02, f"{val:.2f}", ha="center", va="bottom")

        # Right: induction on the harmless set.
        labels2 = ["baseline", "addition\n(real dir)", "addition\n(random ctrl)"]
        vals2 = [
            real.baseline_harmless_refusal,
            real.addition_harmless_refusal,
            control.addition_harmless_refusal,
        ]
        ax2.bar(labels2, vals2, color=[self.C_BASELINE, self.C_REAL_INDUCE, self.C_CTRL_INDUCE])
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("refusal rate (harmless prompts)")
        ax2.set_title("Induction: add the refusal direction")
        ax2.axhline(real.baseline_harmless_refusal, ls="--", c=self.C_BASELINE, lw=1)
        for i, val in enumerate(vals2):
            ax2.text(i, val + 0.02, f"{val:.2f}", ha="center", va="bottom")

        fig.suptitle(
            f"Refusal as a single direction ({report.model}): {report.verdict}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=self.dpi)
        plt.close(fig)
        return out
