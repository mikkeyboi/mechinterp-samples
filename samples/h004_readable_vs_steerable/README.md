# Readable is not necessarily steerable

This sample separates two questions that are easy to conflate:

1. Can a learned linear probe read the concept at this layer?
2. Is the concept already aligned with the model's own output-token basis?

A probe learns whatever linear basis makes the labels separable. The logit-lens
score is stricter: it applies the model's final normalization and unembedding,
then asks whether positive examples rank above negative examples in that output
basis. A readable direction can therefore be output-anti-aligned.

## Run it

```bash
python demo.py
```

The default is deterministic, offline, and model-free. It plants a concept that
becomes perfectly readable at layer 2, while its output-basis AUC is below chance
there. Output alignment emerges only in later layers. The script writes a JSON
report and a figure under `figures/`.

For an optional real-model read:

```bash
pip install -e ".[capture]"
python demo.py --real --model google/gemma-3-1b-it
```

The real path builds token-matched concessive-sentiment pairs, captures all
residual-stream layers in one pass per batch, and checks a critical invariant:
unembedding the final hidden state must reconstruct the emitted logits before it
applies the final norm once to intermediate states. It reports a depth profile
without selecting a steering site, because selecting one requires a separate
causal steering sweep.

## Honest takeaway

The synthetic demo proves that perfect linear readability does not imply output
alignment. On the measured research runs, the readable onset was output-
anti-aligned and a separately measured deep steering site was output-aligned on
three small transformers. That onset/site comparison is descriptive, not a
validated layer selector, because the deep sites were selected using steering
measurements. A held-out prediction requires choosing a layer from accessibility
alone, then measuring steering there on a new concept or model.

The method follows Billa (2026), [Predicting Where Steering Vectors
Succeed](https://arxiv.org/abs/2604.15557), while keeping the selected-site claim
explicitly bounded.
