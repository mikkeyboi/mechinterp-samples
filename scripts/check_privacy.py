#!/usr/bin/env python3
"""Fail CI if tracked, shipping files leak identity or agent/delegation voice.

This is a public showcase repo. Per the project's privacy rule, the public
collaboration surface must not carry Mike's real-world identity, and must read as
one author's voice (no AI-agent/delegation chatter). This script greps the tracked
source for those tells so a leak cannot merge.

Scope: shipping files only (code, docs, notebooks). It deliberately skips the
package author metadata and license, and skips itself. It does NOT scan git
history or commit metadata (that is enforced separately).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Identity + delegation-voice tells. Word-boundaried to avoid false hits like
# "claudication"; case-insensitive.
PATTERNS = [
    r"\bmichael\b", r"\bleung\b", r"\bmike\b",
    r"mw\.leung", r"\bgmail\b",
    # agent / delegation voice that should never appear in public prose
    r"\bopus\b", r"\bcodex\b", r"\bcopilot\b", r"gpt-5",
    r"claude opus", r"cloud[- ]agent", r"\bthe maintainer\b",
]
RX = re.compile("|".join(PATTERNS), re.IGNORECASE)

# Extensions worth scanning (shipping text). Binary/figure files are skipped.
SCAN_EXT = {".py", ".md", ".ipynb", ".txt", ".rst", ".cfg", ".toml", ".yml", ".yaml"}

# Files/paths exempt from the scan, with a reason.
EXEMPT = {
    "scripts/check_privacy.py",     # this file lists the patterns
    "LICENSE",
}
# Lines that legitimately mention an allowed token (e.g. the package author handle
# "mikkeyboi" is fine; we only ban the *real name* and agent voice). We allow the
# GitHub handle explicitly.
ALLOW_LINE = re.compile(r"mikkeyboi", re.IGNORECASE)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return out


def main() -> int:
    hits: list[str] = []
    for rel in tracked_files():
        if rel in EXEMPT:
            continue
        p = REPO / rel
        if p.suffix.lower() not in SCAN_EXT or not p.is_file():
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            m = RX.search(line)
            if not m:
                continue
            # Allow the bare GitHub handle "mikkeyboi" even though it contains "mike".
            if m.group(0).lower() == "mike" and ALLOW_LINE.search(line):
                continue
            hits.append(f"{rel}:{n}: matched {m.group(0)!r}: {line.strip()[:120]}")

    if hits:
        print("PRIVACY CHECK FAILED -- identity/delegation tells in shipping files:")
        for h in hits:
            print(f"  - {h}")
        print("\nPublic repo text must stay first-person and free of real-name / "
              "agent-delegation voice. See the project's privacy rule.")
        return 1
    print("Privacy check passed: no identity/delegation tells in tracked shipping files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
