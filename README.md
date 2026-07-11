# mechinterp-samples

Runnable, readable **mechanistic-interpretability** demonstrations. Each sample is
a small, self-contained study you can clone and run: a notebook plus a shared,
class-based library. The goal is to *teach the method*, not just show a result.

These are the public, showcase-quality slices of an ongoing learning project. Not
every research artifact is here; the ones that are, prioritise clarity. Fuller
notes are available on request.

## Quick start

```bash
git clone https://github.com/mikkeyboi/mechinterp-samples
cd mechinterp-samples
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

pytest                                          # model-free tests, CPU, seconds
python samples/h001_linear_probe/demo.py        # first demo, synthetic mode
```

Every demo runs in a **synthetic / no-model mode by default** so a stranger can
see the full pipeline and figure with zero GPU and zero downloads. Real
model-capture runs are opt-in (`--real`) and need the `capture` extra.

## Samples

| Sample | Question | Status |
|---|---|---|
| [`h001_linear_probe`](samples/h001_linear_probe/) | Is sentiment a linearly decodable direction in the residual stream, and at what depth? | runnable; includes a cautionary "leaks at the embedding layer" finding |
| [`h004_readable_vs_steerable`](samples/h004_readable_vs_steerable/) | Does linear readability imply alignment with the model's output basis? | runnable; synthetic offline default plus an optional real-model logit-lens profile |

More samples land as the project progresses; the table grows with them.

## Design

The two axes you are most likely to extend are **polymorphic** by design:

- **Concept** -> subclass `ConceptDataset` (`datasets.py`). Implement `build()`,
  return a `DatasetSplit`. `SentimentDataset` is the worked example.
- **Probe family** -> subclass `Probe` (`probes.py`). Implement `fit`/`predict`.
  `LinearProbe` (logistic regression) ships; an MLP probe is a drop-in.

The rest is composition:

```
src/mechinterp_samples/
  datasets.py   ConceptDataset (ABC), SentimentDataset, DatasetSplit
  capture.py    ActivationCapturer + Pooler hierarchy (MeanPooler, LastTokenPooler)
  probes.py     Probe (ABC), LinearProbe, LayerProbeSweep, LayerResult, SweepReport
  plotting.py   LayerCurvePlotter
```

Two disciplines are baked in because they separate a real interpretability claim
from a fooled one:

1. **Control task** (Hewitt & Liang 2019; Belinkov 2022). Every layer also gets a
   same-capacity probe trained on shuffled labels. **Selectivity** = real minus
   control. High accuracy with low selectivity means the probe is exploiting
   capacity, not reading a concept.
2. **No quantization of interpreted activations, and never Ollama.** Activation
   capture loads the model in HuggingFace `transformers` with
   `output_hidden_states`; 4-bit weights corrupt residual-stream geometry, and
   completion servers expose no hook into it at all.

`hidden_states[0]` is the embedding output (pre-transformer); layers `1..n` are
post-block residual streams. Keeping all of them is what lets the
accuracy-by-layer curve reveal whether a concept is *computed deep* or merely
*present at the embedding*.

## Repo layout

```
mechinterp-samples/
  src/mechinterp_samples/   reusable, class-based library
  samples/<hNNN>_<slug>/    one self-contained demo each (notebook + README + figures)
  tests/                    model-free pytest suite
  pyproject.toml            core deps; [capture] extra for real model runs
```

The `samples/hNNN_<slug>/` naming **mirrors** the private project's
hypothesis-oriented structure, so a given hypothesis maps to a given public demo
without guesswork.

## Contributing / conventions

- **Conventional commits** (`feat:`, `fix:`, `docs:`, ...); semantic-release
  derives versions from messages.
- Classes over loose functions; encapsulate state; use polymorphism where it
  earns its keep (concepts, probes, poolers).
- Every sample must run **end to end from a fresh clone** in synthetic mode, and
  state its honest takeaway, including cautionary results.

## License

MIT. See [LICENSE](LICENSE).
