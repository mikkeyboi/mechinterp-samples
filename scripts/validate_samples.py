#!/usr/bin/env python3
"""Validate every sample in samples/ so the repo stays runnable as it grows.

This is the scaling guard for the showcase repo: rather than hand-maintaining a
per-sample CI list, we *discover* samples and assert the invariants every sample
must satisfy. A new sample added tomorrow is validated automatically.

Invariants enforced (per sample directory under samples/):
  1. It has a README.md (a stranger needs to know what it is and how to run it),
     and that README follows samples/README_TEMPLATE.md: a "# HNNN: <question>?"
     title matching the directory id, then the required "## " sections in order,
     and at least one rendered figure under figures/ that exists on disk. The
     README is parsed as CommonMark (markdown-it-py), so code, comments and
     quoted examples never count as structure or figures.
  2. It exposes at least one runnable demo: a *.py with an `if __name__ == "__main__"`.
  3. Every demo runs **standalone from a fresh clone**: no install, no PYTHONPATH.
     We invoke it in a subprocess with an empty PYTHONPATH and a synthetic/offline
     flag, and require exit 0. This is the blindspot that bit us once (the package
     was only importable via pytest's pythonpath, not when run directly).

Demos are expected to have a no-network/no-GPU default mode (synthetic). If a demo
genuinely cannot run without a model, mark it by adding the literal string
`# ci: skip-run` near the top; it will still be checked for structure, just not
executed. Keep that escape hatch rare and justified.

Exit code is non-zero if any invariant fails, with a precise report.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "samples"
# Per-demo wall-clock budget. The probe sweeps fit many logistic models on CPU.
RUN_TIMEOUT = int(os.environ.get("SAMPLE_RUN_TIMEOUT", "300"))
# The shared README shape (see samples/README_TEMPLATE.md). Extra sections may sit
# anywhere after the title; these must all appear, spelled exactly, in this order.
REQUIRED_SECTIONS = [
    "TL;DR",
    "Run it",
    "How to read the figure",
    "Takeaway",
    "Limits",
]
TITLE_RX = re.compile(r"^(H\d{3}): .+\?$")


def find_samples() -> list[Path]:
    if not SAMPLES.is_dir():
        return []
    return sorted(p for p in SAMPLES.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


def readme_outline(text: str) -> tuple[list[tuple[int, str]], list[str]]:
    """Parse a README as CommonMark and return what it renders.

    Returns its top-level (not quoted or listed) h1/h2 headings as (level, text)
    in order, and the decoded local path of every rendered image with a relative
    destination (URLs with a scheme or host are skipped). Anything inside code,
    HTML or a comment is never a heading or an image.
    """
    tokens = MarkdownIt("commonmark").parse(text)
    headings: list[tuple[int, str]] = []
    images: list[str] = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.level == 0 and tok.tag in ("h1", "h2"):
            headings.append((int(tok.tag[1]), tokens[i + 1].content.strip()))
        for child in tok.children or []:
            if child.type == "image":
                url = urlsplit(str(child.attrGet("src") or ""))
                if not url.scheme and not url.netloc:  # local relative file only
                    images.append(unquote(url.path))
    return headings, images


def check_readme(sample: Path) -> list[str]:
    """Return template violations for a sample's README (empty list means ok)."""
    readme = sample / "README.md"
    if not readme.is_file():
        return ["missing README.md"]
    headings, images = readme_outline(readme.read_text(encoding="utf-8", errors="replace"))

    problems: list[str] = []
    title_at = next((i for i, (level, _) in enumerate(headings) if level == 1), None)
    title = headings[title_at][1] if title_at is not None else None
    m = TITLE_RX.match(title or "")
    if not m:
        shown = f"# {title}" if title is not None else None
        problems.append(f"title must read '# HNNN: <question>?', got {shown!r}")
    elif not sample.name.lower().startswith(m.group(1).lower() + "_"):
        problems.append(f"title id {m.group(1)} does not match directory {sample.name}")

    sections = [(i, text) for i, (level, text) in enumerate(headings) if level == 2]
    names = [text for _, text in sections]
    missing = [s for s in REQUIRED_SECTIONS if s not in names]
    if missing:
        problems.append("missing section(s): " + ", ".join(f"'## {s}'" for s in missing))
    else:
        at = [sections[names.index(s)][0] for s in REQUIRED_SECTIONS]
        if at != sorted(at):
            problems.append("sections out of order; expected " + " -> ".join(REQUIRED_SECTIONS))
        if title_at is not None and title_at > at[0]:
            problems.append("title must come before the required sections")

    fig_dir = (sample / "figures").resolve()
    figures = []
    for src in images:
        if not src.startswith("figures/"):
            continue
        if (sample / src).resolve().is_relative_to(fig_dir):
            figures.append(src)
        else:
            problems.append(f"figure outside figures/: {src}")
    if not figures:
        problems.append("no figure: embed at least one ![...](figures/...) image")
    for fig in figures:
        if not (sample / fig).is_file():
            problems.append(f"figure not found: {fig}")
    return problems


def demos_in(sample: Path) -> list[Path]:
    out = []
    for py in sorted(sample.glob("*.py")):
        text = py.read_text(encoding="utf-8", errors="replace")
        if '__main__' in text:
            out.append(py)
    return out


def run_standalone(demo: Path) -> tuple[bool, str]:
    """Run a demo as a fresh-clone stranger would: clean env, repo root cwd."""
    text = demo.read_text(encoding="utf-8", errors="replace")
    if "# ci: skip-run" in text:
        return True, "skipped (marked # ci: skip-run)"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)          # force the sample to bootstrap its own path
    env["MPLBACKEND"] = "Agg"            # never try to open a display
    env["HF_HUB_OFFLINE"] = "1"          # a stray model fetch should fail loud, not hang
    env["TRANSFORMERS_OFFLINE"] = "1"
    cmd = [sys.executable, str(demo.relative_to(REPO))]
    # Pass --seed 0 when supported; harmless arg parsers ignore unknowns? No, argparse
    # errors on unknown args, so only add flags we know the convention uses.
    if "--seed" in text:
        cmd += ["--seed", "0"]
    try:
        p = subprocess.run(
            cmd, cwd=REPO, env=env, timeout=RUN_TIMEOUT,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {RUN_TIMEOUT}s"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-8:]
        return False, "exit %d\n    %s" % (p.returncode, "\n    ".join(tail))
    return True, "ok"


def main() -> int:
    samples = find_samples()
    if not samples:
        print("No samples/ directory or no samples found; nothing to validate.")
        return 0

    failures: list[str] = []
    print(f"Validating {len(samples)} sample(s) under {SAMPLES.relative_to(REPO)}/\n")
    for s in samples:
        rel = s.relative_to(REPO)
        print(f"== {rel} ==")

        problems = check_readme(s)
        if problems:
            for prob in problems:
                failures.append(f"{rel}/README.md: {prob}")
                print(f"  [FAIL] README.md: {prob}")
        else:
            print("  [ok] README.md follows the template")

        demos = demos_in(s)
        if not demos:
            failures.append(f"{rel}: no runnable demo (no *.py with __main__)")
            print("  [FAIL] no runnable demo")
            continue

        for demo in demos:
            ok, msg = run_standalone(demo)
            tag = "ok" if ok else "FAIL"
            print(f"  [{tag}] run {demo.name}: {msg}")
            if not ok:
                failures.append(f"{demo.relative_to(REPO)}: {msg}")
        print()

    if failures:
        print("SAMPLE VALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All samples valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
