# exp-tables — development and polishing log

## Development

### 2026-11-13 — runs, summaries, the LaTeX table

- **No dependencies.** The statistics are written out with the standard library: Student's t distribution from the
  regularized incomplete beta function (continued fraction), its quantiles by bisection. Checked against the
  standard tables: t(0.975) = 4.303, 3.182, 2.571, 2.042, 1.984 for 2, 3, 5, 30, 100 degrees of freedom.
- **Reading** (`data.py`): every source becomes runs with text fields and numeric metrics.
  - CSV/TSV (delimiter detected; wandb's runs export is one): a column is a metric when all its values are numbers
    (`93.1%`, `1,234`, `1e-3` read too) and its name is not a seed's, a configuration's or a tracker's bookkeeping
    (runtime, timestamps, state).
  - JSON (a list of runs, `{"runs": [...]}`, or runs by name) and JSON Lines, nested objects flattened with `/`;
    numbers under `config/`, `params/`, `args/` are fields, not metrics.
  - MLflow `mlruns` folders: parameters and tags as fields (the run's name, the experiment's name), each metric's
    value at its last step.
- **Summaries** (`stats.py`): mean, sample standard deviation, standard error, the t confidence interval's
  half-width.
- **The table** (`table.py`): worked out from the runs when not specified: the seed field by its name, the row
  field (method, model, …, else the first field that varies), a column group (another varying field, not one that
  names every run apart), every metric. Which way is better from the metric's name (accuracy, BLEU, F1 up; loss,
  error, FID, perplexity down) or given. The best of each column bold, the second underlined (only among three or
  more, and compared as printed, so equal values are both marked). Decimals: two significant digits of the spread.
- **LaTeX**: booktabs, a header row of column groups with `\cmidrule`s, arrows for the direction, the spread small
  after the mean, `--` for a missing result, labels escaped.

Tests: 19 (synthetic results written by the tests), Python 3.10 and 3.14; ruff, mypy clean.

Next: significance tests (Welch and paired t, Wilcoxon, bootstrap), multiple-comparison corrections, effect sizes,
"within noise" marks and how many seeds; TensorBoard event files; Markdown and HTML; a config file; the CLI.
