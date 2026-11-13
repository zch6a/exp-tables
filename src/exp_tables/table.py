"""From runs to a table: rows (methods), column groups (datasets) and metrics, each cell the summary of a metric
over seeds; the best and second best of each column marked where the metric's direction is known; rendered as
booktabs LaTeX."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .data import SEED, Run
from .stats import Summary

LOWER = re.compile(
    r"\b(?:loss|err(?:or)?s?|perplexity|ppl|fid|kid|wer|cer|ter|mse|mae|rmse|nll|bpd|bpc|eer|latency|time|runtime|seconds?|ms|"
    r"params?|parameters|flops|macs|memory|mem|cost|distance|regret|lpips|chamfer)\b",
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


@dataclass
class Cell:
    summary: Summary | None
    best: bool = False
    second: bool = False


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
    return Spec(rows, columns, metrics, seed, spec.better, spec.spread, spec.decimals, spec.labels)


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
    for r in runs:
        for m in spec.metrics:
            if m in r.metrics:
                values.setdefault((key(r, spec.rows), key(r, spec.columns) if spec.columns else "", m), []).append(r.metrics[m])
    columns = [(g, m) for g in group_keys for m in spec.metrics if any((rk, g, m) in values for rk in row_keys)]
    cells = [[Cell(Summary(values[(rk, g, m)]) if (rk, g, m) in values else None) for g, m in columns] for rk in row_keys]
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
                shown = round(c.summary.mean, decimals[j])
                c.best = shown == ranked[0]  # equal as printed: both are the best
                c.second = len(ranked) > 2 and shown == ranked[1]  # a second best only among three or more
    title = " / ".join(spec.labels.get(n, n) for n in spec.rows)
    return Table(row_keys, title, [g for g, _ in columns], [m for _, m in columns], cells, directions, decimals, spec.spread, spec.labels)


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
