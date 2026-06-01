# H001: Is a concept a linearly decodable direction, and *where*?

A runnable demonstration of **linear probing** of a language model's residual
stream. We ask whether a simple concept (sentiment) is a linear direction in the
model's internal representation, and if so, at what depth it becomes readable.

## TL;DR

- Train a logistic-regression **probe** on the residual-stream activations at each
  layer of a small transformer, and plot accuracy as a function of depth.
- Guard the claim with two controls: a **shuffled-label control task**
  (selectivity = real accuracy minus control accuracy) and a **bag-of-tokens
  baseline** (does the *representation* beat the raw *words*?).
- The **shape** of the curve is the result, not any single number.

## Run it

```bash
pip install -e ".[dev]"          # from the repo root
python samples/h001_linear_probe/demo.py          # synthetic, instant, no GPU
python samples/h001_linear_probe/demo.py --real   # real Gemma 3 1B (GPU + gated)
```

Or open `notebook.ipynb` for the narrated version.

Synthetic mode plants a concept that **strengthens with depth** (weak at the
embedding layer, clear by the middle), so you see what a genuinely *computed*
concept looks like: accuracy rises with depth while the control stays near
chance. No model or download required.

## How to read the figure

![accuracy by layer](figures/h001_accuracy_by_layer.png)

- **probe (held-out)**: probe accuracy on unseen test prompts at each layer.
- **control (shuffled labels)**: a same-capacity probe trained on permuted labels.
  It should sit near chance; the gap up to the real probe is the **selectivity**.
- **bag-of-tokens**: logistic regression on raw token counts. The probe must beat
  this to show the concept lives in the *representation*, not just the words.
- **chance**: majority-class accuracy, the real floor.

A concept the model **computes** should be hard to read at layer 0 (the raw
embedding) and become readable with depth: a rise-then-plateau curve.

## The honest result (real Gemma 3 1B run)

When we run `--real` on Gemma 3 1B, held-out accuracy is near-perfect at **every**
layer, **including layer 0 (the embedding, before any transformer computation).**
That is a cautionary result, and it is the point of including it:

- The concept is **linearly present**, but the signal is **lexical /
  embedding-space**. Polar adjectives are separable in embedding space before the
  model computes anything.
- A vocabulary-disjoint split (test words never seen in training) blocks
  *memorising exact words*, but does **not** block *embedding-space leakage* of
  sentiment.
- So the apparatus works and the concept is linearly decodable, but this setup
  does **not** tell us *where the model computes* sentiment. The predicted
  rise-with-depth profile does not appear.

The fix (a follow-up demo) is to make the concept one that **cannot** be read off
the embedding, e.g. a relation that depends on composing several tokens, so depth
has to do real work. Fuller research notes are available on request.

## Files

```
h001_linear_probe/
  demo.py        runnable end to end (synthetic by default, --real for Gemma)
  notebook.ipynb narrated walkthrough
  figures/       figures + report JSON the demo writes
  README.md      this file
```

The reusable machinery (`SentimentDataset`, `ActivationCapturer`, `LinearProbe`,
`LayerProbeSweep`, `LayerCurvePlotter`) lives in the shared package
`src/mechinterp_samples/` so later demos reuse it. See the top-level README for
the design.
