---
name: exp-tables
description: Turn experiment results into a paper's table with exp-tables. Reads runs from CSV/TSV (a wandb export too), JSON, JSON Lines, TensorBoard logs, MLflow mlruns folders or lm-evaluation-harness results; averages each metric over seeds with its spread; tests each method against the best of its column (paired or Welch's t, Wilcoxon, bootstrap, corrected per column) and marks as best only what the seeds support; writes booktabs LaTeX, Markdown, HTML or JSON, and a config file that makes the same table again. Use when the user asks for a results table, wants to know whether an improvement is significant or within the noise of the seeds, how many seeds they need, or wants their paper's tables kept in step with their results.
license: MIT
compatibility: Needs Python 3.10+ and uv. Runs exp-tables with uvx from GitHub (git+https://github.com/zch6a/exp-tables) unless an exp-tables command is installed. No dependencies; nothing is sent anywhere.
metadata:
  homepage: https://github.com/zch6a/exp-tables
---

# Results to tables with exp-tables

```bash
uvx --from git+https://github.com/zch6a/exp-tables exp-tables results.csv              # a LaTeX table
uvx --from git+https://github.com/zch6a/exp-tables exp-tables runs/ --format markdown   # TensorBoard logs or mlruns
```

Use `exp-tables` instead of the `uvx ...` prefix if it is installed. The table is worked out from the runs (the
method field for rows, a dataset-like field for column groups, the seed field by its name, every metric); say it
when the guess is wrong: `--rows model --columns task --seed trial --metrics acc,loss`. Other options:
`--filter split=test`, `--better metric=lower`, `--label accuracy="Acc. (%)"`, `--decimals accuracy=1`,
`--spread std|sem|ci|none`, `--test auto|welch|paired|wilcoxon|bootstrap|none`, `--format latex|markdown|html|json`.

- `--report` prints what each mark rests on: for every column, the best against each other method, with the test,
  the p value before and after the correction, the effect sizes, and "significant" or "within the noise; about N
  seeds each would tell". Read it before claiming an improvement; quote it when the user asks whether a gain is
  real.
- `--save-config table.json` writes a config that makes the same table again; `exp-tables --config table.json
  --check paper/table.tex` says whether the table in the paper still matches the results (exit 1 and the lines
  that differ when it does not).

## Reading the table

- **Bold** is the best and every method not significantly different from it; a noisy method can be bold where a
  steadier one, slightly better on average, is not. That is the point: say so rather than "fixing" it.
- Underlined is the second best, shown only after a single best.
- With one seed per method there is no spread and no test: tell the user the table cannot say whether differences
  are real, and suggest more seeds (the report's estimate says how many). Single results with a standard error
  (lm-evaluation-harness reports one, or a CSV column `acc_stderr`) are tested by it instead (a z test).
- Notes on stderr say what looks wrong in the runs (several runs of one cell sharing a seed, untested cells): pass
  them on, and fix the layout they point at (`--columns`, `--filter`) rather than ignoring them.
