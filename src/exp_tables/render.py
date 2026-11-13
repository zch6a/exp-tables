"""The table as Markdown (a README, a pull request) and as HTML (a page, a notebook): the same marks as the LaTeX
table, with a legend saying what they mean."""

from __future__ import annotations

from html import escape

from .table import Cell, Table, _spread

ARROWS = {"higher": " ↑", "lower": " ↓"}


def _value(c: Cell, decimals: int, spread: str) -> tuple[str, str]:
    """(mean, spread) as text; ("--", "") for a missing result."""
    if c.summary is None:
        return "--", ""
    mean = f"{c.summary.mean:.{decimals}f}"
    if spread == "none" or c.summary.n < 2:
        return mean, ""
    return mean, f"{_spread(c.summary, spread):.{decimals}f}"


def legend(t: Table) -> str:
    """What the marks and the spread mean, in a sentence."""
    spread = {"std": "standard deviation", "sem": "standard error", "ci": "95% confidence interval"}.get(t.spread)
    parts = [f"mean ± {spread} over seeds" if spread else "mean over seeds"]
    if t.test != "none":
        parts.append(f"bold: the best, and any not significantly different from it ({_test_name(t)})")
    else:
        parts.append("bold: the best")
    parts.append("underlined: the second best")
    return "; ".join(parts) + "."


def _test_name(t: Table) -> str:
    names = sorted({c.test for c in t.comparisons})
    alpha = t.comparisons[0].alpha if t.comparisons else 0.05
    return f"{', '.join(names) or 'no test possible'}, α = {alpha}, corrected per column"


def _md(s: str) -> str:
    return s.replace("|", "\\|").replace("*", "\\*").replace("_", "\\_")


def markdown(t: Table) -> str:
    """A GitHub Markdown table: best in bold, second best underlined (``<ins>``), the spread after ±; column groups
    joined to the metric's name ("cifar10 · accuracy ↑"); then the legend."""
    label = lambda s: _md(t.labels.get(s) or s)  # noqa: E731
    heads = [(f"{label(g)} · " if g else "") + label(m) + ARROWS.get(d, "") for g, m, d in zip(t.groups, t.metrics, t.directions, strict=True)]
    lines = ["| " + label(t.row_title) + " | " + " | ".join(heads) + " |", "|---|" + "---:|" * len(heads)]
    for name, row in zip(t.rows, t.cells, strict=True):
        cells = []
        for c, d in zip(row, t.decimals, strict=True):
            mean, spread = _value(c, d, t.spread)
            mean = f"**{mean}**" if c.best else f"<ins>{mean}</ins>" if c.second else mean
            cells.append(f"{mean} ± {spread}" if spread else mean)
        lines.append("| " + label(name) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\n" + legend(t) + "\n"


def html(t: Table) -> str:
    """An HTML table (a fragment, to put in a page): column groups spanning their metrics, best in bold, second best
    underlined, the spread small; every label escaped; then the legend."""
    e = lambda s: escape(t.labels.get(s) or s)  # noqa: E731
    out = ['<table class="exp-tables">', "<thead>"]
    if any(t.groups):
        spans: list[tuple[str, int]] = []
        for g in t.groups:
            if spans and spans[-1][0] == g:
                spans[-1] = (g, spans[-1][1] + 1)
            else:
                spans.append((g, 1))
        out.append("<tr><th></th>" + "".join(f'<th colspan="{n}">{e(g)}</th>' for g, n in spans) + "</tr>")
    out.append(
        f"<tr><th>{e(t.row_title)}</th>"
        + "".join(f"<th>{e(m)}{ARROWS.get(d, '')}</th>" for m, d in zip(t.metrics, t.directions, strict=True))
        + "</tr>"
    )
    out += ["</thead>", "<tbody>"]
    for name, row in zip(t.rows, t.cells, strict=True):
        tds = []
        for c, d in zip(row, t.decimals, strict=True):
            mean, spread = _value(c, d, t.spread)
            mean = f"<b>{mean}</b>" if c.best else f"<u>{mean}</u>" if c.second else mean
            tds.append(f"<td>{mean}" + (f" <small>± {spread}</small>" if spread else "") + "</td>")
        out.append(f"<tr><th>{e(name)}</th>" + "".join(tds) + "</tr>")
    out += ["</tbody>", f'<caption style="caption-side: bottom">{escape(legend(t))}</caption>', "</table>"]
    return "\n".join(out) + "\n"
