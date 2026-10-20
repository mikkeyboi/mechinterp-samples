"""Tests for the README template check in scripts/validate_samples.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_samples.py"
_spec = importlib.util.spec_from_file_location("validate_samples", _SCRIPT)
validate_samples = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_samples)
check_readme = validate_samples.check_readme

IMAGE = "![plot](figures/result.png)"


def _readme(image: str = IMAGE) -> str:
    return f"""# H001: Is this a valid sample?

Intro.

## TL;DR

- point

## Run it

```bash
python demo.py
```

## How to read the figure

{image}

## Takeaway

Result.

## Limits

- limit
"""


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "h001_example"
    (root / "figures").mkdir(parents=True)
    (root / "figures" / "result.png").write_bytes(b"png")
    return root


def _check(sample: Path, text: str) -> list[str]:
    (sample / "README.md").write_text(text, encoding="utf-8")
    return check_readme(sample)


def test_valid_readme_passes(sample):
    assert _check(sample, _readme()) == []


@pytest.mark.parametrize(
    "wrapped",
    [
        "~~~markdown\n{body}~~~\n",
        "````markdown\n{body}````\n",
        "~~~~\n{body}",  # unclosed fence runs to the end of the document
    ],
)
def test_fenced_readme_does_not_satisfy_template(sample, wrapped):
    problems = _check(sample, wrapped.format(body=_readme()))
    assert any(p.startswith("title") for p in problems)
    assert any(p.startswith("missing section") for p in problems)
    assert any(p.startswith("no figure") for p in problems)


@pytest.mark.parametrize("fence", ["~~~", "```", "````"])
def test_fenced_shell_comment_before_title_is_ignored(sample, fence):
    preamble = f"{fence}bash\n# install the dependencies\n{fence}\n\n"
    assert _check(sample, preamble + _readme()) == []


@pytest.mark.parametrize(
    "image",
    [
        f"`{IMAGE}`",
        f"``{IMAGE}``",
        "\\" + IMAGE,
        "\\\\\\" + IMAGE,
    ],
)
def test_unrendered_image_is_not_a_figure(sample, image):
    assert "no figure: embed at least one ![...](figures/...) image" in _check(
        sample, _readme(image)
    )


@pytest.mark.parametrize(
    "image",
    [
        f"`{IMAGE}` renders as {IMAGE}",
        "\\\\" + IMAGE,  # an escaped backslash, then a real image
        f"Inline: {IMAGE}",
        f"A stray ` backtick.\n\n{IMAGE}\n\nAnother ` later.",  # spans stop at blank lines
    ],
)
def test_rendered_image_counts(sample, image):
    assert _check(sample, _readme(image)) == []


@pytest.mark.parametrize(
    "preamble",
    [
        "<!--\n```\n-->\n\n",  # a fence opener inside a comment stays hidden
        "<!-- a note with ~~~ inline -->\n\n",
        "```html\n<!-- an unclosed comment, shown as code\n```\n\n",  # stays literal
    ],
)
def test_comments_and_fences_do_not_swallow_each_other(sample, preamble):
    assert _check(sample, preamble + _readme()) == []


def test_template_inside_comment_does_not_satisfy_contract(sample):
    problems = _check(sample, "<!--\n" + _readme() + "-->\n")
    assert any(p.startswith("title") for p in problems)


def test_missing_figure_file_fails(sample):
    (sample / "figures" / "result.png").unlink()
    assert _check(sample, _readme()) == ["figure not found: figures/result.png"]


def test_shipped_samples_follow_template():
    for s in validate_samples.find_samples():
        assert check_readme(s) == [], s.name
