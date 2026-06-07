"""H001 depth-profile contrast: a probe that lies at layer 0, and one that does not.

This is the companion to the blog post "A probe at layer 0 is a lie detector for
your experiment." It runs the **same** probe pipeline on **two** concepts and
overlays their accuracy-by-layer curves, because the teaching point is the
*shape* contrast, not any single number:

  * Lexical sentiment (positive vs negative adjectives): separable at the ceiling
    from layer 0, the input embedding, before the transformer computes anything.
    The probe is reading the dictionary. Flat at the top is a *warning*, not a win.
  * Negation-composed sentiment (net = adjective_polarity XOR negation): chance at
    the embedding, rising to a ceiling across the early-to-mid stack. The label
    cannot be read off any single token, so the model has to *build* the concept,
    and the rising flank is where it does.

Same model, same probe, same 1.000 ceiling at the top, opposite meaning.

    python contrast_demo.py            # synthetic, instant, no GPU or download
    python contrast_demo.py --real     # real Gemma 3 1B capture (GPU + gated access)

Synthetic mode plants the two shapes by hand (lexical = separable from layer 0;
composed = chance at layer 0, rising with depth) so the figure and the pipeline
are visible in seconds with no model. Pass `--real` to reproduce the genuine
Gemma 3 1B curves; that is the run the blog figure is made from.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

# The per-layer logistic probes run on unscaled ~1000-dim residual activations, so
# lbfgs sometimes hits max_iter without fully converging. Held-out accuracy is
# unaffected (it matches the private reference run), so we silence the cosmetic
# warning here at the demo boundary rather than in the library.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from mechinterp_samples import (
    SentimentDataset,
    NegationSentimentDataset,
    ActivationCapturer,
    MeanPooler,
    LayerProbeSweep,
    DepthProfileContrastPlotter,
)
from mechinterp_samples.datasets import ConceptDataset, DatasetSplit
from mechinterp_samples.probes import SweepReport

HERE = Path(__file__).parent
MODEL_NAME = "google/gemma-3-1b-it"
N_LAYERS_SYNTH = 27  # match the real Gemma 3 1B residual-point count (incl. embedding)


def synthetic_lexical(split: DatasetSplit, d_model: int = 64,
                      seed: int = 0) -> dict[int, np.ndarray]:
    """Activations where the concept is *already separable at layer 0*.

    Mirrors the real lexical-sentiment finding: polar adjectives are linearly
    separable in embedding space, so the probe scores at the ceiling from the
    embedding onward with no depth dependence.
    """
    rng = np.random.default_rng(seed)
    y = split.labels
    n = len(y)
    direction = rng.normal(size=d_model)
    acts = {}
    for layer in range(N_LAYERS_SYNTH):
        base = rng.normal(size=(n, d_model))
        acts[layer] = base + (y[:, None] * 3.0) * direction  # strong at every layer
    return acts


def synthetic_composed(split: DatasetSplit, d_model: int = 64,
                       seed: int = 1) -> dict[int, np.ndarray]:
    """Activations where the concept is *built across depth*.

    Mirrors the negation-XOR finding: chance at the embedding, rising to a ceiling
    by the mid stack. The signal strength ramps from 0 at layer 0.
    """
    rng = np.random.default_rng(seed)
    y = split.labels
    n = len(y)
    direction = rng.normal(size=d_model)
    acts = {}
    for layer in range(N_LAYERS_SYNTH):
        # 0 at the embedding, saturating by ~layer 8, matching the real curve.
        strength = 3.0 * min(layer / 8.0, 1.0)
        base = rng.normal(size=(n, d_model))
        acts[layer] = base + (y[:, None] * strength) * direction
    return acts


def run_one(dataset: ConceptDataset, real: bool, capturer: ActivationCapturer | None,
            synth_fn, seed: int) -> tuple[SweepReport, DatasetSplit]:
    """Build a dataset, get activations (real or synthetic), run the probe sweep."""
    split = dataset.build()
    print(f"  {dataset.name}: n_train={split.n_train} n_test={split.n_test}")
    if real:
        assert capturer is not None
        acts = capturer.capture(split.texts)
        model_label = MODEL_NAME
    else:
        acts = synth_fn(split, seed=seed)
        model_label = "synthetic"
    report = LayerProbeSweep(seed=seed).run(
        acts, split, concept=dataset.name, model=model_label
    )
    return report, split


def summarise(tag: str, report: SweepReport) -> None:
    best = report.best_layer
    l0 = report.per_layer[0]
    print(f"  [{tag}] layer-0 acc {l0.test_acc:.3f} | best L{best.layer} "
          f"acc {best.test_acc:.3f} | bag-of-tokens {report.baseline_test_acc:.3f} "
          f"| chance {report.chance_acc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="capture real Gemma 3 1B activations (GPU + gated access)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "figures" / "depth_profile_contrast.png"))
    args = ap.parse_args()

    if args.real:
        print(f"capturing real activations from {MODEL_NAME} (mean-pooled) ...")
        capturer = ActivationCapturer(MODEL_NAME, pooler=MeanPooler())
    else:
        print("synthetic mode (no model). pass --real for genuine Gemma capture.")
        capturer = None

    print("\nExperiment A: lexical sentiment (the easy concept)")
    lex_report, _ = run_one(
        SentimentDataset(seed=args.seed), args.real, capturer,
        synthetic_lexical, args.seed,
    )
    summarise("lexical", lex_report)

    print("\nExperiment B: negation-composed sentiment (polarity XOR negation)")
    comp_report, _ = run_one(
        NegationSentimentDataset(seed=args.seed), args.real, capturer,
        synthetic_composed, args.seed,
    )
    summarise("composed", comp_report)

    # Persist both reports so every number in the figure is traceable.
    figdir = HERE / "figures"
    lex_report.to_json(figdir / "contrast_lexical_report.json")
    comp_report.to_json(figdir / "contrast_composed_report.json")

    fig_out = DepthProfileContrastPlotter().plot(lex_report, comp_report, args.out)
    print(f"\nwrote reports -> {figdir}/contrast_*.json")
    print(f"wrote figure  -> {fig_out}")
    print("\nThe shape is the finding: red is flat at the ceiling from layer 0 "
          "(lexical, no computation needed); blue starts at chance and is built "
          "across depth (composed).")


if __name__ == "__main__":
    main()
