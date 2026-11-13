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

### 2026-11-13 — significance

- **Tests** (`significance.py`, standard library only): Welch's t (unequal variances), the paired t test, the
  Wilcoxon signed-rank test (exact by dynamic programming up to 25 pairs without tied differences, the normal
  approximation with the tie correction beyond), a seeded bootstrap of the difference in means; Holm's and
  Benjamini-Hochberg's corrections; Hedges' g and Cliff's delta.
- **Checked by hand and independently**: the t distribution at fractional degrees of freedom against a numerical
  integration of its density (agreement to six decimals; Welch's example p = 0.1075); Wilcoxon's exact p for
  all-positive differences (2/2^5) and for one negative (4/2^5); the corrections never lower a p value. Three of my
  own doctest expectations were wrong (a paired t without the square root of n, two hand roundings) and the code
  right; each was worked through before changing it.
- **How many seeds**: first a normal approximation, which said "about 1 seed" for a comparison that already had 2
  and was not significant. Now the t-based power at n's own degrees of freedom, and for paired seeds the spread of
  the differences: "3 seeds each".
- **The table**: in each column with a direction, the best against every other row with two or more seeds (paired t
  when the seeds pair up, else Welch's; or the test asked for), the column's p values corrected together; rows not
  significantly different from the best are marked best too ("tied"), and a second best is underlined only after a
  single best. With `test="none"`, the best as printed, as before.
- **The report** (`report.py`): each comparison in a line: the difference, the test, p before and after the
  correction, the effect sizes, the seeds, and "significant" or "within the noise; about N seeds each would tell".

Tests: 40, Python 3.10 and 3.14; ruff, mypy clean (the commit waits for ruff: it caught two issues this time).

Next: TensorBoard event files; Markdown and HTML tables; a config file that reproduces a table; the CLI.
