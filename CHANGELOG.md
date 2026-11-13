# Changelog

## 0.1.0

The first release: experiment results to a paper's table, with the best marked only where the seeds support it.

- **Reads** CSV and TSV (a wandb runs export too), JSON, JSON Lines, MLflow `mlruns` folders and TensorBoard logs
  (its own reader: PyTorch's and TF2's scalars), into runs with fields and metrics.
- **Works out the table** (rows, column groups, the seed field, the metrics) unless told; which way is better from
  each metric's name unless told.
- **Summarises** each metric over seeds: mean with the standard deviation, standard error or a 95% t interval.
- **Tests** each column's best against the other methods (paired t when the seeds pair up, Welch's otherwise, or
  Wilcoxon or a bootstrap), corrected per column (Holm or Benjamini-Hochberg); marks as best every method not
  significantly different from the best; reports effect sizes and how many seeds a difference would need.
- **Writes** booktabs LaTeX, Markdown, HTML and JSON; a config file that makes the same table again; `--check` that
  says whether a committed table still matches the results, and a GitHub Action around it.
- **Says so** when the runs look wrong for the table: several runs of one cell sharing a seed, a Wilcoxon test that
  cannot reach the significance level with so few pairs, seeds that do not pair up for a paired test.
- Reads **lm-evaluation-harness** results files (one run per model and task), and **tests single results by their
  standard errors** (the harness's, or a CSV column `acc_stderr`): a z test, the error shown as the spread.
- A **Python API**: `exp_tables.read`, `build`, `Spec`, `latex`, `markdown`, `html`, `comparisons`.
- Tested on ten public multi-seed results files from GitHub (layouts, names and missing values as people write
  them), on MultiBERTs fine-tuning TensorBoard logs and Open LLM Leaderboard results from the Hugging Face Hub, and
  on 100,000 runs; two independent reviews.
- No dependencies: the statistics are in the standard library, checked against independent values.
