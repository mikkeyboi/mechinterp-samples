#!/usr/bin/env python3
"""Validate every sample in samples/ so the repo stays runnable as it grows.

This is the scaling guard for the showcase repo: rather than hand-maintaining a
per-sample CI list, we *discover* samples and assert the invariants every sample
must satisfy. A new sample added tomorrow is validated automatically.

Invariants enforced (per sample directory under samples/):
  1. It has a README.md (a stranger needs to know what it is and how to run it),
     and that README follows samples/README_TEMPLATE.md: a "# HNNN: <question>?"
     title matching the directory id, the required "## " sections in order, and
     at least one figure under figures/ that exists on disk.
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
TITLE_RX = re.compile(r"^# (H\d{3}): .+\?\s*$")
# An image is "![alt](figures/...)". Whether a backslash escapes it is decided in
# figure_paths(), so the match itself consumes nothing before the "!" and
# adjacent images are each found.
FIGURE_RX = re.compile(r"!\[[^\]]*\]\((figures/[^)\s]+)\)")
# CommonMark fences: up to three spaces of indent, then three or more backticks or
# tildes. A fence closes on a line of the same character, at least as long.
FENCE_OPEN_RX = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
# An HTML comment block: a line opening with "<!--" after up to three spaces.
COMMENT_OPEN_RX = re.compile(r"^ {0,3}<!--")
# Inline code spans and inline HTML comments are found by blank_inline_hidden(),
# which scans left to right so escapes and whichever span opens first win.
BLANK_LINE_RX = re.compile(r"\n[ \t]*\n")
ASCII_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def find_samples() -> list[Path]:
    if not SAMPLES.is_dir():
        return []
    return sorted(p for p in SAMPLES.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


def strip_hidden_blocks(text: str) -> str:
    """Blank every line of a fenced code block or an HTML comment block.

    One pass handles both, so a fence marker inside a comment stays hidden and a
    comment marker inside a code block stays literal. A fence closes on the same
    character, at least as long, so a "~~~" block, or a four-backtick block
    holding a triple-backtick line, is removed whole. A comment block starts on a
    line opening with "<!--" and ends on the line holding "-->". Unclosed blocks
    run to the end of the document, as they render.
    """
    out: list[str] = []
    fence: tuple[str, int] | None = None
    in_comment = False
    for line in text.splitlines():
        if in_comment:
            in_comment = "-->" not in line
            out.append("")
        elif fence is not None:
            char, length = fence
            if re.match(r"^ {0,3}%s{%d,}\s*$" % (re.escape(char), length), line):
                fence = None
            out.append("")
        elif COMMENT_OPEN_RX.match(line):
            in_comment = "-->" not in line.split("<!--", 1)[1]
            out.append("")
        else:
            m = FENCE_OPEN_RX.match(line)
            # A backtick fence's info string may not contain a backtick.
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                fence = (m.group(1)[0], len(m.group(1)))
                out.append("")
            else:
                out.append(line)
    return "\n".join(out)


def blank_inline_hidden(body: str) -> str:
    """Blank inline code spans and inline HTML comments, keeping line breaks.

    Scans left to right like CommonMark: a backslash before ASCII punctuation
    escapes it (so "\\`" or "\\<!--" opens nothing), a backtick run is closed by
    the next run of the same length, a comment by the next "-->", and neither
    crosses a blank line. Text inside an open code span is literal, backslashes
    included. Unclosed openers stay as text. Escapes are kept in the output so
    figure_paths() can still see them.
    """
    out: list[str] = []
    i, n = 0, len(body)

    def paragraph_end(start: int) -> int:
        m = BLANK_LINE_RX.search(body, start)
        return m.start() if m else n

    while i < n:
        ch = body[i]
        if ch == "\\" and i + 1 < n and body[i + 1] in ASCII_PUNCT:
            out.append(body[i : i + 2])
            i += 2
        elif ch == "`":
            j = i
            while j < n and body[j] == "`":
                j += 1
            run = j - i
            close = re.compile(r"(?<!`)`{%d}(?!`)" % run).search(body, j, paragraph_end(j))
            if close:
                out.append(" " + "\n" * body.count("\n", i, close.end()))
                i = close.end()
            else:
                out.append(body[i:j])
                i = j
        elif body.startswith("<!--", i):
            end = body.find("-->", i + 4, paragraph_end(i))
            if end != -1:
                out.append(" " + "\n" * body.count("\n", i, end + 3))
                i = end + 3
            else:
                out.append(ch)
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def figure_paths(body: str) -> list[str]:
    """Destinations of rendered figures: skip any "!" behind an odd run of
    backslashes (escaped); an even run is literal backslashes, so it renders."""
    out = []
    for m in FIGURE_RX.finditer(body):
        start = m.start()
        run = len(body[:start]) - len(body[:start].rstrip("\\"))
        if run % 2 == 0:
            out.append(m.group(1))
    return out


def check_readme(sample: Path) -> list[str]:
    """Return template violations for a sample's README (empty list means ok)."""
    readme = sample / "README.md"
    if not readme.is_file():
        return ["missing README.md"]
    text = readme.read_text(encoding="utf-8", errors="replace")
    # Ignore fenced code and HTML comments so a "# comment" in a bash block or a
    # heading inside a note is not mistaken for structure.
    body = strip_hidden_blocks(text)
    # Blank inline code and inline comments too, keeping line breaks so the
    # block structure (titles, sections) is untouched. A figure quoted in code
    # is then not counted, and a quoted "<!--" cannot hide what follows it.
    body = blank_inline_hidden(body)
    lines = body.splitlines()

    problems: list[str] = []
    title = next((ln for ln in lines if ln.startswith("# ")), None)
    m = TITLE_RX.match(title or "")
    if not m:
        problems.append(f"title must read '# HNNN: <question>?', got {title!r}")
    elif not sample.name.lower().startswith(m.group(1).lower() + "_"):
        problems.append(f"title id {m.group(1)} does not match directory {sample.name}")

    sections = [ln[3:].strip() for ln in lines if ln.startswith("## ")]
    missing = [s for s in REQUIRED_SECTIONS if s not in sections]
    if missing:
        problems.append("missing section(s): " + ", ".join(f"'## {s}'" for s in missing))
    else:
        order = [sections.index(s) for s in REQUIRED_SECTIONS]
        if order != sorted(order):
            problems.append("sections out of order; expected " + " -> ".join(REQUIRED_SECTIONS))

    figures = figure_paths(body)
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
