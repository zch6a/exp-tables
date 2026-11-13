"""A doctest on a docstring's first line passes on Python 3.13 (which dedents docstrings) and fails on 3.10: none."""

from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_no_doctest_on_a_docstrings_first_line():
    bad = [
        f"{p.relative_to(SRC)}:{n}"
        for p in SRC.rglob("*.py")
        for n, line in enumerate(p.read_text(encoding="utf-8").split("\n"), 1)
        if '""">>>' in line
    ]
    assert bad == [], bad
