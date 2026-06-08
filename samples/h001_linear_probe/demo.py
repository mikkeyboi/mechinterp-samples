"""H001 demo: is sentiment a linearly decodable direction, and *where*?

Run this top to bottom. By default it runs in **synthetic mode** (no model, no
GPU, no download) so anyone can see the full pipeline and the figure in seconds.
Pass `--real` to capture genuine Gemma 3 1B activations instead (needs the
`capture` extra, a GPU, and HuggingFace access to the gated model).

    python demo.py            # synthetic, instant, no dependencies beyond core
    python demo.py --real     # real Gemma 3 1B capture (GPU + gated access)

The teaching point is the *shape* of the accuracy-by-layer curve, not a single
number. See README.md for how to read it, including the honest "it leaked at the
embedding layer" result the real run produces.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the sample runnable from a fresh clone without `pip install` first: add the
# repo's src/ to the path if the package is not already importable. (Installing the
# package, per the README, also works and takes precedence.)
try:
    import mechinterp_samples  # noqa: F401
except ModuleNotFoundError:
    _SRC = Path(__file__).resolve().parents[2] / "src"
    if _SRC.is_dir():
        sys.path.insert(0, str(_SRC))

from mechinterp_samples import (
    SentimentDataset,
    ActivationCapturer,
    MeanPooler,
    LayerProbeSweep,
    LayerCurvePlotter,
)
from mechinterp_samples.datasets import DatasetSplit

HERE = Path(__file__).parent
MODEL_NAME = "google/gemma-3-1b-it"


def synthetic_activations(split: DatasetSplit, n_layers: int = 12, d_model: int = 64,
                          seed: int = 0) -> dict[int, np.ndarray]:
    """Stand-in activations with a *depth-dependent* planted signal.

    To make the demo pedagogically honest without a model, we plant the concept
    so it strengthens with depth: barely present at layer 0, clear by the middle
    layers. This is the curve a genuinely *computed* concept would produce, and a
    useful contrast to the real Gemma run, where sentiment turns out to be
    readable from layer 0 (a lexical/embedding effect). The README walks through
    both.
    """
    rng = np.random.default_rng(seed)
    y = split.labels
    n = len(y)
    direction = rng.normal(size=d_model)
    acts = {}
    for layer in range(n_layers):
        strength = 2.5 * (layer / (n_layers - 1))  # 0 at embedding -> strong deep
        base = rng.normal(size=(n, d_model))
        acts[layer] = base + (y[:, None] * strength) * direction
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="capture real Gemma 3 1B activations (GPU + gated access)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "figures" / "h001_accuracy_by_layer.png"))
    args = ap.parse_args()

    dataset = SentimentDataset(seed=args.seed)
    split = dataset.build()
    print(f"dataset: {dataset.name}  n_train={split.n_train}  n_test={split.n_test}")

    if args.real:
        print(f"capturing real activations from {MODEL_NAME} ...")
        capturer = ActivationCapturer(MODEL_NAME, pooler=MeanPooler())
        acts = capturer.capture(split.texts)
        model_label = MODEL_NAME
    else:
        print("synthetic mode (no model). pass --real for genuine Gemma capture.")
        acts = synthetic_activations(split, seed=args.seed)
        model_label = "synthetic"

    report = LayerProbeSweep(seed=args.seed).run(
        acts, split, concept=dataset.name, model=model_label
    )

    best = report.best_layer
    best_sel = report.best_selectivity_layer
    print(f"\nbaseline (bag-of-tokens) test acc: {report.baseline_test_acc:.3f}")
    print(f"chance                           : {report.chance_acc:.3f}")
    print(f"best layer {best.layer}: test acc {best.test_acc:.3f}")
    print(f"best selectivity layer {best_sel.layer}: "
          f"sel {best_sel.selectivity:.3f} (test {best_sel.test_acc:.3f}, "
          f"control {best_sel.control_test_acc:.3f})")

    json_out = HERE / "figures" / "h001_report.json"
    report.to_json(json_out)
    fig_out = LayerCurvePlotter().plot(report, args.out)
    print(f"\nwrote report -> {json_out}")
    print(f"wrote figure -> {fig_out}")


if __name__ == "__main__":
    main()
