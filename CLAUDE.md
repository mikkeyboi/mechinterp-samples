# Working in this repo

This is a personal showcase for a mechanistic-interpretability learning project:
small, runnable, readable demos backed by a class-based library. It does not take
outside contributions. The bar is **runnable and readable**, not exhaustive.

## Scope of a session

One session does one unit of work, and one unit of work is one pull request.
Typical units:

- **Add a sample.** One new `samples/<hNNN>_<slug>/` directory, the library code
  and tests it needs, and its row in the README's Samples table.
- **Fix or improve one sample.** Changes stay inside that sample's directory and
  the library code it uses.
- **Repo maintenance.** CI, tooling, or docs, when that is what was asked for.

Where a session's work starts:

- From the ask. Identify which unit above it is and which sample (`hNNN`) it
  touches before editing.
- From the current default branch, on a fresh branch.

Where it stops:

- When the checklist below passes locally for that unit and a PR is open with CI
  green. Polishing beyond that is a new unit.
- At the edge of the unit. Do not touch other samples, restructure the shared
  library, change CI, or bump dependencies unless the ask needs it. If you notice
  something worth fixing elsewhere, say so in the PR description instead of
  fixing it.

## Conventions

- **Conventional commits** (`feat(hNNN): ...`, `fix:`, `docs:`, `refactor:`,
  `test:`). semantic-release derives versions from the history.
- **Classes over loose functions.** Encapsulate state. Use polymorphism where it
  earns its keep; the established axes are:
  - a concept -> subclass `ConceptDataset` (`src/mechinterp_samples/datasets.py`)
  - a probe family -> subclass `Probe` (`src/mechinterp_samples/probes.py`)
  - a token pooling strategy -> subclass `Pooler` (`src/mechinterp_samples/capture.py`)
- **Every sample runs end to end from a fresh clone** in a no-model synthetic mode.
  Real model runs are opt-in and must degrade gracefully without a GPU.
- **State the honest takeaway**, including cautionary results. A demo that shows a
  method's failure mode is as valuable as one that shows a clean win.

## Adding a sample

A sample is one directory, `samples/<hNNN>_<slug>/`, plus whatever shared logic it
needs in `src/mechinterp_samples/`. Items marked **(CI)** are enforced by CI.

**Layout**

- [ ] Directory named `hNNN_<slug>`, mirroring the private project's hypothesis id.
- [ ] `README.md` with the question, how to run it, and the honest takeaway. **(CI)**
- [ ] `demo.py` with an `if __name__ == "__main__":` entry point. **(CI)**
- [ ] `figures/` holding the figure and JSON report the demo writes.
- [ ] Optional: `notebook.ipynb` as a narrated walkthrough (see below).

**Runs from a fresh clone**

- [ ] The default mode is synthetic: no model, no GPU, no network. Real-model runs
      sit behind an opt-in flag such as `--real`. **(CI)**
- [ ] The demo bootstraps `src/` onto `sys.path` itself (copy the try/except block
      at the top of an existing `demo.py`), so it runs without `pip install`. **(CI)**
- [ ] If the demo takes `--seed`, it accepts `--seed 0`. **(CI)**
- [ ] It finishes inside the CI budget (`SAMPLE_RUN_TIMEOUT`, 420 s).

**Library and tests**

- [ ] Reusable logic lives in `src/mechinterp_samples/` as classes, extending an
      existing axis where one fits; the sample directory stays thin.
- [ ] New public names are exported from `src/mechinterp_samples/__init__.py`.
- [ ] A test in `tests/` exercises the new logic **without** a model, using
      synthetic activations with known structure. `pytest` passes on CPU in
      seconds. **(CI)**

**Repo surface**

- [ ] A row for the sample in the README's Samples table.
- [ ] All checks below pass.

### Notebooks are optional

Each sample needs a README and a runnable `demo.py`; that is what CI checks. A
notebook is worth adding when a narrated, cell-by-cell walkthrough teaches more
than the script does (`h001_linear_probe` has one). If a sample has one, it must
run top to bottom in synthetic mode from the sample directory and be committed
with outputs cleared. CI lints notebooks with ruff but does not execute them, so
run it before committing.

## What stays out

- Research notes, hypothesis text, task tracking, and personal details. The lab
  notebook lives in a separate private repo; do not copy from it. When fuller
  detail would help a reader, write "available on request".
- Assistant or delegation voice. Public prose reads as one author, first person
  or impersonal. `scripts/check_privacy.py` enforces part of this in CI.
- Turnkey misuse paths. Safety-relevant samples (for example `h005`) ship a
  synthetic default and keep sensitive data steps out of this repo.

## Checks to run before pushing

```bash
pip install -e ".[dev]"
ruff check src samples scripts tests
python scripts/check_privacy.py
python -m pytest -q
python scripts/validate_samples.py
```

These mirror the CI jobs in `.github/workflows/ci.yml`.
