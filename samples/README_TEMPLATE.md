# HNNN: <The question this sample asks, phrased as a question?>

One short paragraph of context: what method this demonstrates, what it is
reproducing or testing, and why the question is worth asking.

<!--
Shared template for every samples/hNNN_<slug>/README.md.

scripts/validate_samples.py checks that each sample README has:
  - a title of the form "# HNNN: <question>?" whose id matches the directory;
  - the six required "## " sections below, spelled exactly, in this order;
  - at least one image under figures/ that exists on disk.

You may add extra sections (a follow-up demo, a Files listing, a "why this
matters" note) anywhere after the title, as long as the required ones stay in
order. Keep the honest result in Takeaway and the boundaries in Limits, so a
reader skimming any sample finds them in the same place.
-->

## TL;DR

- Two to four bullets: the method, the control that guards the claim, and what
  shape of result to look for.

## Run it

```bash
pip install -e ".[dev]"                     # from the repo root
python samples/hNNN_<slug>/demo.py          # synthetic, instant, no GPU
python samples/hNNN_<slug>/demo.py --real   # optional real-model run
```

Say what the synthetic default plants and what the real path needs.

## How to read the figure

![short alt text](figures/<figure>.png)

- One bullet per series or panel: what it is and where it should sit.
- One sentence on what a positive result looks like.

## Takeaway

The honest result, including on a real model if one was run. Cautionary or
null results belong here as much as clean wins.

## Limits

- What the result does **not** show (e.g. readable is not causal, sufficient is
  not stored, a selected site is descriptive).
- Scope choices the sample makes deliberately.
