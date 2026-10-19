# Contributing

These are showcase demonstrations. The bar is **runnable and readable**, not
exhaustive.

## Conventions

- **Conventional commits** (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
  semantic-release derives versions from the history.
- **Classes over loose functions.** Encapsulate state. Use polymorphism where it
  earns its keep; in this repo the established axes are:
  - a concept -> subclass `ConceptDataset` (`src/mechinterp_samples/datasets.py`)
  - a probe family -> subclass `Probe` (`src/mechinterp_samples/probes.py`)
  - a token pooling strategy -> subclass `Pooler` (`src/mechinterp_samples/capture.py`)
- **Every sample runs end to end from a fresh clone** in a no-model synthetic mode.
  Real model runs are opt-in and must degrade gracefully without a GPU.
- **State the honest takeaway**, including cautionary results. A demo that shows a
  method's failure mode is as valuable as one that shows a clean win.

## Adding a sample

1. Put reusable logic in `src/mechinterp_samples/` as classes; keep the
   `samples/<hNNN>_<slug>/` directory thin (notebook + `demo.py` + README + figures).
2. Mirror the private project's hypothesis id in the directory name (`hNNN_<slug>`).
3. Add or extend a test in `tests/` that exercises the new logic **without** a
   model (synthesise activations with known structure, as the existing tests do).
4. `pytest` must pass on CPU in seconds.

## What does *not* belong here

No private research notes, hypotheses text, task tracking, or personal details.
This repo is the public showcase; the lab notebook lives elsewhere. When fuller
detail would help a reader, write "available on request" rather than pasting notes.
