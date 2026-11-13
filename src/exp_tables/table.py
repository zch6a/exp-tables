"""From runs to a table: rows (methods), column groups (datasets) and metrics, each cell the summary of a metric
over seeds; the best and second best of each column marked where the metric's direction is known; rendered as
booktabs LaTeX."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .data import SEED, Run
from .significance import Result, benjamini_hochberg, bootstrap, cliffs_delta, hedges_g, holm, paired, seeds_needed, welch, wilcoxon
from .stats import Summary

LOWER = re.compile(
    r"\b(?:loss|err(?:or)?s?|perplexity|ppl|fid|kid|wer|cer|ter|mse|mae|rmse|nll|bpd|bpc|eer|latency|time|runtime|seconds?|ms|"
    r"params?|parameters|flops|macs|memory|mem|cost|distance|regret|lpips|chamfer|ece|mce|brier|calibration)\b",
    re.I,
)
HIGHER = re.compile(
    r"\b(?:acc(?:uracy)?|top-?[15]|bleu|rouge(?:-\w+)?|meteor|chrf|f1|f-?score|auc|auroc|ap|map|ap50|ap75|iou|miou|precision|"
    r"recall|psnr|ssim|ndcg|mrr|hits?@?\d*|pass@?\d*|win ?rate|success|em|exact match|spearman|pearson|r2|reward|return|"
    r"bertscore|cider|spice|throughput|score)\b",
    re.I,
)
ROW_FIELDS = ("method", "model", "algorithm", "approach", "system", "baseline", "name", "run", "experiment")


def direction(metric: str) -> str:
    """'higher', 'lower' or '' for a metric's name (its last part: "eval/accuracy" is an accuracy).

    >>> direction("eval/accuracy"), direction("val_loss"), direction("test FID"), direction("tokens")
    ('higher', 'lower', 'lower', '')
    """
    name = re.sub(r"[_./-]+", " ", metric.rsplit("/", 1)[-1])
    if re.search(r"↑|\(\+\)|higher", metric, re.I):
        return "higher"
    if re.search(r"↓|lower", metric, re.I):
        return "lower"
    lo, hi = LOWER.search(name), HIGHER.search(name)
    return "lower" if lo and not hi else "higher" if hi and not lo else ""


@dataclass
class Spec:
    """What the table shows. Empty lists are worked out from the runs."""

    rows: list[str] = field(default_factory=list)  # fields naming a row (a method)
    columns: list[str] = field(default_factory=list)  # fields grouping columns (a dataset)
    metrics: list[str] = field(default_factory=list)  # metrics shown, in order
    seed: str = ""  # the field telling seeds apart
    better: dict[str, str] = field(default_factory=dict)  # metric -> "higher" or "lower" (else from its name)
    spread: str = "std"  # "std", "sem", "ci" (95%) or "none"
    decimals: dict[str, int] = field(default_factory=dict)  # metric -> decimals (else from its spread)
    labels: dict[str, str] = field(default_factory=dict)  # a metric, field value or field name -> how the table prints it
    test: str = "auto"  # "auto" (paired t when the seeds pair up, else Welch's), "welch", "paired", "wilcoxon", "bootstrap", "none"
    alpha: float = 0.05
    correction: str = "holm"  # "holm", "bh" or "none": for the comparisons of a column taken together


@dataclass
class Cell:
    summary: Summary | None
    best: bool = False  # the best, or not significantly different from it
    second: bool = False
    seeds: list[str] = field(default_factory=list)  # each value's seed, in order
    tied: bool = False  # marked best because it is not significantly different from the best


@dataclass
class Comparison:
    """The best of a column against another row."""

    group: str
    metric: str
    best: str
    other: str
    difference: float  # best minus other, in the metric's units
    test: str
    p: float
    p_adjusted: float
    hedges_g: float
    cliffs_delta: float
    seeds: tuple[int, int]
    seeds_needed: int | None  # per method, for a difference of this size to be found (power 0.8)
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.p_adjusted < self.alpha


@dataclass
class Table:
    rows: list[str]  # row labels
    row_title: str  # the header over the row labels
    groups: list[str]  # each column's group ("" without groups)
    metrics: list[str]  # each column's metric
    cells: list[list[Cell]]  # [row][column]
    directions: list[str]
    decimals: list[int]
    spread: str
    labels: dict[str, str] = field(default_factory=dict)
    comparisons: list[Comparison] = field(default_factory=list)
    test: str = "none"


def _varying(runs: list[Run], name: str) -> bool:
    return len({r.fields.get(name, "") for r in runs}) > 1


def resolve(runs: list[Run], spec: Spec) -> Spec:
    """The spec with what was left empty worked out: the seed field (by its name), the row field (method, model, …,
    else the first field that varies), the column fields (the other fields that vary, but not one that names every
    run apart, like a run id), and every metric."""
    names = list(dict.fromkeys(k for r in runs for k in r.fields))
    seed = spec.seed or next((n for n in names if SEED.match(n.rsplit("/", 1)[-1]) and n not in ("run", "run_id")), "")
    varying = [n for n in names if n != seed and _varying(runs, n)]
    ids = {n for n in varying if len({r.fields.get(n, "") for r in runs}) == len(runs) and len(runs) > 2}
    rows = spec.rows or next(([n] for pref in ROW_FIELDS for n in varying if n.rsplit("/", 1)[-1].lower() == pref and n not in ids), [])
    if not rows:
        rows = next(([n] for n in varying if n not in ids), [])
    columns = spec.columns or [n for n in varying if n not in rows and n not in ids and not n.lower().startswith(("config/", "params/"))][:1]
    metrics = spec.metrics or list(dict.fromkeys(m for r in runs for m in r.metrics))
    return Spec(rows, columns, metrics, seed, spec.better, spec.spread, spec.decimals, spec.labels, spec.test, spec.alpha, spec.correction)


def _auto_decimals(summaries: list[Summary]) -> int:
    """Two significant digits of the spread; with no spread (one seed), as many decimals as the values have (<= 4)."""
    spreads = [s.std for s in summaries if s.n > 1 and s.std > 0]
    if spreads:
        return max(0, min(4, 1 - math.floor(math.log10(min(spreads)))))
    shown = [len(f"{v:.4f}".rstrip("0").split(".")[1]) for s in summaries for v in s.values]
    return min(4, max(shown, default=1))


def build(runs: list[Run], spec: Spec | None = None) -> Table:
    """The table of `runs` as `spec` asks (worked out where it does not say)."""
    spec = resolve(runs, spec or Spec())
    key = lambda r, names: " / ".join(r.fields.get(n, "") for n in names)  # noqa: E731
    row_keys = list(dict.fromkeys(key(r, spec.rows) for r in runs))
    group_keys = list(dict.fromkeys(key(r, spec.columns) for r in runs)) if spec.columns else [""]
    values: dict[tuple[str, str, str], list[float]] = {}
    seeds: dict[tuple[str, str, str], list[str]] = {}
    for r in runs:
        for m in spec.metrics:
            if m in r.metrics:
                k = (key(r, spec.rows), key(r, spec.columns) if spec.columns else "", m)
                values.setdefault(k, []).append(r.metrics[m])
                seeds.setdefault(k, []).append(r.fields.get(spec.seed, ""))
    columns = [(g, m) for g in group_keys for m in spec.metrics if any((rk, g, m) in values for rk in row_keys)]
    cells = [
        [Cell(Summary(values[(rk, g, m)]), seeds=seeds[(rk, g, m)]) if (rk, g, m) in values else Cell(None) for g, m in columns] for rk in row_keys
    ]
    directions = [spec.better.get(m) or direction(m) for _, m in columns]
    decimals = []
    for j, (_, m) in enumerate(columns):
        present = [c.summary for row in cells if (c := row[j]).summary is not None]
        decimals.append(spec.decimals.get(m, _auto_decimals(present)))
    for j, way in enumerate(directions):
        if way not in ("higher", "lower"):
            continue
        ranked = sorted({round(c.summary.mean, decimals[j]) for row in cells if (c := row[j]).summary is not None}, reverse=way == "higher")
        for row in cells:
            c = row[j]
            if c.summary is not None and ranked:
                c.best = round(c.summary.mean, decimals[j]) == ranked[0]  # equal as printed: both are the best
    table = Table(row_keys, " / ".join(spec.labels.get(n, n) for n in spec.rows), [g for g, _ in columns], [m for _, m in columns], cells,
                  directions, decimals, spec.spread, spec.labels, test=spec.test)  # fmt: skip
    if spec.test != "none":
        _significance(table, spec)
    for j in range(len(columns)):  # the second best: after a single best (with several marked best, the next is third), among three or more
        rest = [row[j] for row in cells if row[j].summary is not None and not row[j].best]
        single_best = sum(row[j].best for row in cells) == 1
        if directions[j] in ("higher", "lower") and single_best and len([row[j] for row in cells if row[j].summary is not None]) > 2 and rest:
            sign = 1 if directions[j] == "higher" else -1
            top = max(round(sign * c.summary.mean, decimals[j]) for c in rest if c.summary is not None)
            for c in rest:
                c.second = c.summary is not None and round(sign * c.summary.mean, decimals[j]) == top
    return table


def _pairs(a: Cell, b: Cell) -> bool:
    """Whether two cells have values for the same seeds, each seed once."""
    return bool(a.seeds) and all(a.seeds) and sorted(a.seeds) == sorted(b.seeds) and len(set(a.seeds)) == len(a.seeds)


def _needed(a: Cell, b: Cell, r: Result, alpha: float) -> int | None:
    """Seeds needed, the way the test went: paired values by their differences."""
    assert a.summary is not None and b.summary is not None
    if r.name in ("paired t", "Wilcoxon"):
        order = dict(zip(b.seeds, b.summary.values, strict=True))
        return seeds_needed(a.summary.values, [order[s] for s in a.seeds], alpha, paired=True)
    return seeds_needed(a.summary.values, b.summary.values, alpha)


def _test(spec_test: str, a: Cell, b: Cell) -> Result:
    """The test of two cells' values: paired when asked, or (auto) when both have the same seeds."""
    assert a.summary is not None and b.summary is not None
    x, y = a.summary.values, b.summary.values
    pairs = _pairs(a, b)
    if pairs:  # put b's values in a's seed order
        order = {s: v for s, v in zip(b.seeds, y, strict=True)}
        y_paired = [order[s] for s in a.seeds]
    if spec_test == "bootstrap":
        return bootstrap(x, y)
    if spec_test == "wilcoxon" and pairs:
        return wilcoxon(x, y_paired)
    if spec_test in ("paired", "auto") and pairs:
        return paired(x, y_paired)
    return welch(x, y)


def _significance(table: Table, spec: Spec) -> None:
    """Each column's best against the other rows (with two or more seeds each); the p values of a column corrected
    together; the rows not significantly different from the best marked best as well (tied)."""
    for j, way in enumerate(table.directions):
        if way not in ("higher", "lower"):
            continue
        column = [(table.rows[i], row[j]) for i, row in enumerate(table.cells) if row[j].summary is not None]
        leaders = [(name, c) for name, c in column if c.best]
        if not leaders:
            continue
        sign = 1 if way == "higher" else -1
        best_name, best = max(leaders, key=lambda nc: sign * nc[1].summary.mean)  # type: ignore[union-attr]
        others = [(name, c) for name, c in column if c is not best and c.summary is not None and c.summary.n >= 2]
        if best.summary is None or best.summary.n < 2 or not others:
            continue
        results = [_test(spec.test, best, c) for _, c in others]
        ps = [r.p for r in results]
        adjusted = holm(ps) if spec.correction == "holm" else benjamini_hochberg(ps) if spec.correction == "bh" else ps
        for (name, c), r, p_adj in zip(others, results, adjusted, strict=True):
            assert c.summary is not None
            if p_adj >= spec.alpha and not c.best:
                c.best, c.tied = True, True
            table.comparisons.append(
                Comparison(
                    table.groups[j],
                    table.metrics[j],
                    best_name,
                    name,
                    best.summary.mean - c.summary.mean,
                    r.name,
                    r.p,
                    p_adj,
                    hedges_g(best.summary.values, c.summary.values),
                    cliffs_delta(best.summary.values, c.summary.values),
                    (best.summary.n, c.summary.n),
                    _needed(best, c, r, spec.alpha),
                    spec.alpha,
                )  # fmt: skip
            )


# ---------------------------------------------------------------- LaTeX

LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                 "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}  # fmt: skip


def tex(s: str) -> str:
    r"""Text as LaTeX prints it: the special characters escaped.

    >>> tex("acc_top-1 (%) & F#")
    'acc\\_top-1 (\\%) \\& F\\#'
    """
    return "".join(LATEX_SPECIAL.get(c, c) for c in s)


def _spread(s: Summary, how: str) -> float:
    return {"std": s.std, "sem": s.sem, "ci": s.ci()}.get(how, 0.0)


def latex_cell(c: Cell, decimals: int, spread: str) -> str:
    r"""A cell: the mean (bold for the best, underlined for the second) and its spread, small.

    >>> latex_cell(Cell(Summary([93.0, 93.4, 92.9]), best=True), 1, "std")
    '\\textbf{93.1}\\,{\\scriptsize$\\pm$0.3}'
    """
    if c.summary is None:
        return "--"
    mean = f"{c.summary.mean:.{decimals}f}"
    mean = rf"\textbf{{{mean}}}" if c.best else rf"\underline{{{mean}}}" if c.second else mean
    if spread == "none" or c.summary.n < 2:
        return mean
    return rf"{mean}\,{{\scriptsize$\pm${_spread(c.summary, spread):.{decimals}f}}}"


def latex(t: Table) -> str:
    """The table as booktabs LaTeX (``\\usepackage{booktabs}``): a header row of groups when there are any, with
    rules under each group, then the metrics with an arrow for which way is better."""

    def label(s: str) -> str:
        return tex(t.labels.get(s) or s)

    arrow = {"higher": r" $\uparrow$", "lower": r" $\downarrow$"}
    lines = [rf"\begin{{tabular}}{{l{'r' * len(t.metrics)}}}", r"\toprule"]
    if any(t.groups):
        spans: list[tuple[str, int, int]] = []  # (group, first column, last column), 1-based after the label column
        for j, g in enumerate(t.groups, start=2):
            if spans and spans[-1][0] == g:
                spans[-1] = (g, spans[-1][1], j)
            else:
                spans.append((g, j, j))
        lines.append(" & " + " & ".join(rf"\multicolumn{{{b - a + 1}}}{{c}}{{{label(g)}}}" for g, a, b in spans) + r" \\")
        lines.append(" ".join(rf"\cmidrule(lr){{{a}-{b}}}" for _, a, b in spans))
    lines.append(label(t.row_title) + " & " + " & ".join(label(m) + arrow.get(d, "") for m, d in zip(t.metrics, t.directions, strict=True)) + r" \\")
    lines.append(r"\midrule")
    for name, row in zip(t.rows, t.cells, strict=True):
        cells = [latex_cell(c, d, t.spread) for c, d in zip(row, t.decimals, strict=True)]
        lines.append(label(name) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
