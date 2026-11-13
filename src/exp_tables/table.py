"""From runs to a table: rows (methods), column groups (datasets) and metrics, each cell the summary of a metric
over seeds; the best and second best of each column marked where the metric's direction is known; rendered as
booktabs LaTeX."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .data import SEED, Run
from .significance import Result, benjamini_hochberg, bootstrap, cliffs_delta, hedges_g, holm, paired, seeds_needed, welch, wilcoxon
from .stats import Summary

LOWER = re.compile(
    r"\b(?:loss|err(?:or)?s?|perplexity|ppl|fid|kid|wer|cer|ter|mse|mae|rmse|nll|bpd|bpc|eer|latency|time|runtime|seconds?|ms|"
    r"params?|parameters|flops|macs|memory|mem|cost|distance|regret|lpips|chamfer|hausdorff|ece|mce|brier|calibration|mape|smape|"
    r"fpr|fnr|far|frr|crps|failures?|retries|violations?)\b",
    re.I,
)
HIGHER = re.compile(
    r"\b(?:acc(?:uracy)?|top[- ]?[15]|bleu|rouge(?:-\w+)?|meteor|chrf|f1|f-?score|auc|auroc|ap|map|ap50|ap75|iou|miou|precision|"
    r"recall|psnr|ssim|ndcg|mrr|hits?@?\d*|pass@?\d*|win ?rate|success|em|exact match|spearman|pearson|r2|reward|return|"
    r"bertscore|cider|spice|throughput|score|auprc|aupr|auroc|specificity|sensitivity|tpr|tnr|mcc|kappa|dice|elo|elbo|r²|efficiency|speedup)(?!\w)",
    re.I,
)
GENERIC = re.compile(r"score|top[- ]?[15]", re.I)  # "brier score", "top-1 error": the other word decides
# a field whose values all look like "seed 42", "s1", "trial-3": the seed, whatever the column is called
SEED_VALUE = re.compile(r"^(?:seed|s|trial|run|rep|repeat|fold)[ _=-]?\d+$", re.I)
ROW_FIELDS = ("method", "model", "algorithm", "approach", "system", "baseline", "name", "run", "experiment")


def direction(metric: str) -> str:
    """'higher', 'lower' or '' for a metric's name (its last part: "eval/accuracy" is an accuracy).

    >>> direction("eval/accuracy"), direction("val_loss"), direction("test FID"), direction("tokens")
    ('higher', 'lower', 'lower', '')
    >>> direction("AUPRC"), direction("mcc"), direction("specificity"), direction("threshold")
    ('higher', 'higher', 'higher', '')
    >>> direction("acc/flowers102"), direction("elbo_lower_bound"), direction("brier_score"), direction("top-1 error")
    ('higher', 'higher', 'lower', 'lower')
    >>> direction("cost (higher is better)"), direction("reward ↓")
    ('higher', 'lower')
    >>> direction("cost_efficiency"), direction("cost_per_success"), direction("failure_rate"), direction("fid_score")
    ('higher', 'lower', 'lower', 'lower')
    """
    return _direction(metric.rsplit("/", 1)[-1]) or _direction(metric)  # "acc/flowers102": the dataset is last


def _direction(metric: str) -> str:
    name = re.sub(r"[_./]+", " ", metric)
    if re.search(r" per ", name, re.I):  # "cost per success" is a cost
        name = re.split(r" per ", name, flags=re.I)[0]
    if re.search(r"↑|\(\+\)|higher[ _-]is[ _-]better|\(higher\)", metric, re.I):
        return "higher"
    if re.search(r"↓|\(-\)|lower[ _-]is[ _-]better|\(lower\)", metric, re.I):
        return "lower"
    lo, hi = LOWER.search(name), HIGHER.search(name)
    if lo and hi:  # both: a generic word ("score", "top-1") gives way, else the last word decides ("cost efficiency")
        if GENERIC.fullmatch(hi.group(0)):
            return "lower"
        last_lo = max(m.start() for m in LOWER.finditer(name))
        last_hi = max(m.start() for m in HIGHER.finditer(name))
        return "lower" if last_lo > last_hi else "higher"
    return "lower" if lo else "higher" if hi else ""


@dataclass
class Spec:
    """What the table shows. Empty lists (and columns None) are worked out from the runs."""

    rows: list[str] = field(default_factory=list)  # fields naming a row (a method)
    columns: list[str] | None = None  # fields grouping columns (a dataset); None: worked out; []: no groups
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
    correction: str = "holm"
    notes: list[str] = field(default_factory=list)  # what the reader should know about the runs (e.g. runs sharing a seed)


def _varying(runs: list[Run], name: str) -> bool:
    return len({r.fields.get(name, "") for r in runs}) > 1


def _crosses(runs: list[Run], rows: list[str], name: str) -> bool:
    """Whether a field takes several values within some row, so that it can group columns (a field fixed by the row,
    like a second name of the method, cannot: it would make a diagonal table)."""
    seen: dict[tuple[str, ...], set[str]] = {}
    for r in runs:
        seen.setdefault(tuple(r.fields.get(n, "") for n in rows), set()).add(r.fields.get(name, ""))
    return any(len(v) > 1 for v in seen.values())


def _setting(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def as_fields(runs: list[Run], names: list[str]) -> list[Run]:
    """The runs with these metrics made fields: a numeric setting (a learning rate, a data ratio) is read as a metric,
    and naming it as rows, columns or the seed says it is part of what the run is.

    >>> as_fields([Run({"method": "a"}, {"ratio": 0.1, "acc": 90.0})], ["ratio"])[0].fields
    {'method': 'a', 'ratio': '0.1'}
    """
    names = [n for n in names if n]
    if not any(n in r.metrics for r in runs for n in names):
        return runs
    out = []
    for r in runs:
        moved = {n: _setting(r.metrics[n]) for n in names if n in r.metrics}
        out.append(Run({**r.fields, **moved}, {k: v for k, v in r.metrics.items() if k not in moved}, r.source))
    return out


def resolve(runs: list[Run], spec: Spec) -> Spec:
    """The spec with what was left empty worked out: the seed field (by its name, else by values like "seed 42"),
    the row field (method, model, …, else the first field that varies), the column fields (another field that varies within a row, but not one that
    names every run apart, like a run id), and every metric."""
    names = list(dict.fromkeys(k for r in runs for k in r.fields))
    seed = spec.seed or next((n for n in names if SEED.match(n.rsplit("/", 1)[-1]) and n not in ("run", "run_id")), "")
    seed = seed or next((n for n in names if _varying(runs, n) and all(SEED_VALUE.match(r.fields.get(n, "")) for r in runs)), "")
    varying = [n for n in names if n != seed and n not in (spec.columns or []) and _varying(runs, n)]
    ids = {n for n in varying if len({r.fields.get(n, "") for r in runs}) == len(runs) and len(runs) > 2}
    # a field naming every run apart is a run id when there are seeds, but the method itself when each has one run
    allowed = [n for n in varying if n not in ids or not (seed and _varying(runs, seed))]
    rows = spec.rows or next(([n] for pref in ROW_FIELDS for n in allowed if n.rsplit("/", 1)[-1].lower() == pref), [])
    if not rows:
        rows = next(([n] for n in varying if n not in ids), []) or allowed[:1]
    columns = (
        spec.columns
        if spec.columns is not None
        else [
            n
            for n in varying
            if n not in rows and n not in ids and not n.lower().startswith(("config/", "params/")) and (not rows or _crosses(runs, rows, n))
        ][:1]
    )
    metrics = spec.metrics or list(dict.fromkeys(m for r in runs for m in r.metrics))
    return Spec(rows, columns, metrics, seed, spec.better, spec.spread, spec.decimals, spec.labels, spec.test, spec.alpha, spec.correction)


def _auto_decimals(summaries: list[Summary]) -> int:
    """Two significant digits of the column's typical (median) spread, so that one very steady row does not stretch
    every number of its column; with no spread (one seed each, or equal values), the decimals the values
    have, up to four significant digits. At most 10, so that 1e-5 is not printed as 0.0000.

    >>> _auto_decimals([Summary([1.2e-5, 1.3e-5]), Summary([3.1e-5, 3.0e-5])]), _auto_decimals([Summary([93.125]), Summary([91.5])])
    (8, 2)
    """
    spreads = [s.std for s in summaries if s.n > 1 and math.isfinite(s.std) and s.std > 0]
    if spreads:
        typical = sorted(spreads)[(len(spreads) - 1) // 2]
        return max(0, min(10, 1 - math.floor(math.log10(typical) + 1e-9)))  # 0.01 computed as 0.00999… is still 0.01
    shown = []
    for v in (v for s in summaries for v in s.values if math.isfinite(v) and v != 0):
        has = len(f"{v:.10f}".rstrip("0").split(".")[1])
        shown.append(min(has, max(0, 3 - math.floor(math.log10(abs(v)) + 1e-9))))
    return min(10, max(shown, default=0))


def build(runs: list[Run], spec: Spec | None = None) -> Table:
    """The table of `runs` as `spec` asks (worked out where it does not say)."""
    spec = spec or Spec()
    runs = as_fields(runs, [*spec.rows, *(spec.columns or []), spec.seed])
    spec = resolve(runs, spec)
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
    if not columns:
        raise ValueError("no metric to show: no column of the runs holds numbers (a metric's values must be numbers or missing)")
    cells = [
        [Cell(Summary(values[(rk, g, m)]), seeds=seeds[(rk, g, m)]) if (rk, g, m) in values else Cell(None) for g, m in columns] for rk in row_keys
    ]
    directions = [spec.better.get(m) or direction(m) for _, m in columns]
    decimals = []
    for j, (_, m) in enumerate(columns):
        present = [c.summary for row in cells if (c := row[j]).summary is not None]
        decimals.append(spec.decimals.get(m, _auto_decimals(present)))
    for j, way in enumerate(directions):
        if way not in ("higher", "lower") or sum(row[j].summary is not None for row in cells) < 2:
            continue  # no direction, or nothing to compare with
        ranked = sorted({round(c.summary.mean, decimals[j]) for row in cells if (c := row[j]).summary is not None}, reverse=way == "higher")
        for row in cells:
            c = row[j]
            if c.summary is not None and ranked:
                c.best = round(c.summary.mean, decimals[j]) == ranked[0]  # equal as printed: both are the best
    table = Table(row_keys, " / ".join(spec.labels.get(n, n) for n in spec.rows), [g for g, _ in columns], [m for _, m in columns], cells,
                  directions, decimals, spec.spread, spec.labels, test=spec.test, correction=spec.correction)  # fmt: skip
    table.notes += _shared_seeds(runs, spec, seeds)
    if spec.test != "none":
        _significance(table, spec)
        table.notes += _test_notes(table, spec)
    for j in range(len(columns)):  # the second best: after a single best (with several marked best, the next is third), among three or more
        rest = [row[j] for row in cells if row[j].summary is not None and not row[j].best]
        single_best = sum(row[j].best for row in cells) == 1
        if directions[j] in ("higher", "lower") and single_best and len([row[j] for row in cells if row[j].summary is not None]) > 2 and rest:
            sign = 1 if directions[j] == "higher" else -1
            top = max(round(sign * c.summary.mean, decimals[j]) for c in rest if c.summary is not None)
            for c in rest:
                c.second = c.summary is not None and round(sign * c.summary.mean, decimals[j]) == top
    return table


def _shared_seeds(runs: list[Run], spec: Spec, seeds: dict[tuple[str, str, str], list[str]]) -> list[str]:
    """A note when runs of one cell share a seed: they are averaged as if they were seeds, which is right only for
    repeats; more often another setting tells them apart, read as a metric when it is a number."""
    if not spec.seed:
        return []
    shared = {k: s for k, v in seeds.items() if (s := next((x for x, n in Counter(v).items() if x and n > 1), None))}
    if not shared:
        return []
    (row, group, metric), seed = next(iter(shared.items()))
    where = " / ".join(x for x in (row, group, metric) if x)
    count = seeds[(row, group, metric)].count(seed)
    metrics = list(dict.fromkeys(m for r in runs for m in r.metrics))
    settings = [m for m in metrics if 1 < len({r.metrics.get(m) for r in runs if m in r.metrics}) <= len(runs) // 3]
    hint = f" (a number read as a metric, maybe {', '.join(settings[:3])})" if settings else ""
    return [
        f"{len(shared)} cell(s) have several runs with the same seed (e.g. {where}: {spec.seed} {seed} {count} times), "
        f"averaged as if they were different seeds; if another setting tells them apart{hint}, name it with --rows, --columns or --filter"
    ]


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


def _test(spec_test: str, a: Cell, b: Cell, alpha: float = 0.05) -> Result:
    """The test of two cells' values: paired when asked, or (auto) when both have the same seeds."""
    assert a.summary is not None and b.summary is not None
    x, y = a.summary.values, b.summary.values
    pairs = _pairs(a, b)
    if pairs:  # put b's values in a's seed order
        order = {s: v for s, v in zip(b.seeds, y, strict=True)}
        y_paired = [order[s] for s in a.seeds]
    if spec_test == "bootstrap":
        return bootstrap(x, y, alpha=alpha)
    if spec_test == "wilcoxon" and pairs:
        return wilcoxon(x, y_paired)
    if spec_test in ("paired", "auto") and pairs:
        return paired(x, y_paired)
    return welch(x, y)


def _test_notes(table: Table, spec: Spec) -> list[str]:
    """What the tests could not do as asked: Wilcoxon with too few pairs to reach alpha, unpaired seeds."""
    notes = []
    pairs = sorted({c.seeds[1] for c in table.comparisons if c.test == "Wilcoxon"})
    few = [n for n in pairs if 2 / 2**n >= spec.alpha]
    if few:
        notes.append(f"with {few[-1]} pairs the Wilcoxon test cannot reach p < {spec.alpha} (its smallest p is {2 / 2 ** few[-1]:.3g}): "
                     "those rows are within the noise whatever the values; more seeds, or the paired t test, can tell")  # fmt: skip
    if spec.test in ("paired", "wilcoxon") and any(c.test == "Welch's t" for c in table.comparisons):
        notes.append(f"some rows' seeds do not pair up with the best's: Welch's t test instead of the {spec.test} test for them")
    return notes


def _significance(table: Table, spec: Spec) -> None:
    """Each column's best against the other rows (with two or more seeds each); the p values of a column corrected
    together; the rows not significantly different from the best marked best as well (tied)."""
    for j, way in enumerate(table.directions):
        if way not in ("higher", "lower"):
            continue
        column = [(table.rows[i], row[j]) for i, row in enumerate(table.cells) if row[j].summary is not None]
        if len(column) < 2:
            continue
        sign = 1 if way == "higher" else -1
        best_name, best = max(column, key=lambda nc: sign * nc[1].summary.mean)  # type: ignore[union-attr]
        others = [(name, c) for name, c in column if c is not best and c.summary is not None and c.summary.n >= 2]
        if best.summary is None or best.summary.n < 2 or not others:
            continue  # untested: the marks stay as printed
        results = [_test(spec.test, best, c, spec.alpha) for _, c in others]
        ps = [r.p for r in results]
        adjusted = holm(ps) if spec.correction == "holm" else benjamini_hochberg(ps) if spec.correction == "bh" else ps
        for (name, c), r, p_adj in zip(others, results, adjusted, strict=True):
            assert c.summary is not None
            # the test decides, also against "equal as printed": bold says "not significantly worse"
            c.best, c.tied = p_adj >= spec.alpha, p_adj >= spec.alpha
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
