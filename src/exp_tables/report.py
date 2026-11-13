"""What the table's marks rest on, in words: each column's best against the other rows, with the test, the p value
before and after the correction, the effect sizes, and how many seeds a difference of that size would need."""

from __future__ import annotations

from .table import Comparison, Table


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def comparison_line(c: Comparison, decimals: int = 2) -> str:
    """One comparison in a line.

    >>> from exp_tables.table import Comparison
    >>> line = comparison_line(Comparison("cifar10", "accuracy", "ours", "baseline", 0.3, "paired t", 0.2, 0.4, 0.8, 0.44, (5, 5), 22))
    >>> line.startswith("cifar10 · accuracy: ours vs baseline +0.30 (paired t, p = 0.200, corrected p = 0.400;")
    True
    >>> line.endswith("5 and 5 seeds): within the noise; about 22 seeds each would tell")
    True
    """
    where = f"{c.group} · {c.metric}" if c.group else c.metric
    head = (
        f"{where}: {c.best} vs {c.other} {c.difference:+.{decimals}f} ({c.test}, {_p(c.p)}, corrected {_p(c.p_adjusted)}; "
        f"Hedges g {c.hedges_g:.2f}, Cliff delta {c.cliffs_delta:.2f}, {c.seeds[0]} and {c.seeds[1]} seeds)"
    )
    if c.significant:
        return f"{head}: significant"
    need = f"; about {c.seeds_needed} seeds each would tell" if c.seeds_needed else ""
    return f"{head}: within the noise{need}"


def comparisons(t: Table) -> str:
    """Every comparison of the table, column by column; a note when no test was made."""
    if t.test == "none":
        return "No significance tests (--test none): the best is marked as printed.\n"
    if not t.comparisons:
        return "No comparisons: a column needs a best and another row with two or more seeds each, and a known direction.\n"
    decimals = {(g, m): d for g, m, d in zip(t.groups, t.metrics, t.decimals, strict=True)}
    return "\n".join(comparison_line(c, decimals.get((c.group, c.metric), 2)) for c in t.comparisons) + "\n"
