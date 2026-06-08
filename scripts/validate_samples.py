#!/usr/bin/env python3
"""Validate every sample in samples/ so the repo stays runnable as it grows.

This is the scaling guard for the showcase repo: rather than hand-maintaining a
per-sample CI list, we *discover* samples and assert the invariants every sample
must satisfy. A new sample added tomorrow is validated automatically.

Invariants enforced (per sample directory under samples/):
  1. It has a README.md (a stranger needs to know what it is and how to run it).
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
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "samples"
# Per-demo wall-clock budget. The probe sweeps fit many logistic models on CPU.
RUN_TIMEOUT = int(os.environ.get("SAMPLE_RUN_TIMEOUT", "300"))


def find_samples() -> list[Path]:
    if not SAMPLES.is_dir():
        return []
    return sorted(p for p in SAMPLES.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


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

        if not (s / "README.md").is_file():
            failures.append(f"{rel}: missing README.md")
            print("  [FAIL] no README.md")
        else:
            print("  [ok] README.md")

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
