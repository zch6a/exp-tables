# Changelog

## 0.1.0

The first release: experiment results to a paper's table, with the best marked only where the seeds support it.

- **Reads** CSV and TSV (a wandb runs export too), JSON, JSON Lines, MLflow `mlruns` folders, TensorBoard logs
  (its own reader: PyTorch's and TF2's scalars, no TensorFlow) and lm-evaluation-harness results files, into runs
  with fields, metrics and, where reported, standard errors. Missing values (`nan`, `n/a`, `-`) are missing.
- **Works out the table** (rows, column groups that vary within a row, the seed field, the metrics) unless told;
  which way is better from each metric's name unless told; logged settings and error columns are not metrics.
- **Summarises** each metric over seeds: mean with the standard deviation, standard error or a 95% t interval;
  decimals from the column's typical spread.
- **Tests** each column's best against the other methods (paired t when the seeds pair up, Welch's otherwise, or
  Wilcoxon, exact with ties, or a bootstrap), corrected per column (Holm or Benjamini-Hochberg); single results
  with standard errors by a z test. Bold is every method not significantly different from the best; a cell that
  cannot be tested is left unmarked. The report gives each test, p before and after the correction, effect sizes
  and how many seeds a difference within the noise would need.
- **Writes** booktabs LaTeX (escaped, Greek letters as math), Markdown, HTML and strict JSON; a config file that
  makes the same table again; `--check` that says whether a committed table still matches the results, and a GitHub
  Action around it.
- **Says so** on stderr (and in the JSON) when the runs look wrong for the table: runs of one cell sharing a seed,
  cells that could not be tested, a Wilcoxon test with too few pairs to be significant, a test that could not be
  used as asked, columns read as standard errors.
- A **Python API**: `exp_tables.read`, `build`, `Spec`, `latex`, `markdown`, `html`, `comparisons`.
- Tested on ten public multi-seed results files from GitHub (layouts, names and missing values as people write
  them), MultiBERTs fine-tuning TensorBoard logs and Open LLM Leaderboard results from the Hugging Face Hub, and
  100,000 runs; three independent reviews (over thirty defects found and fixed, each with a test).
- No dependencies: the statistics are in the standard library, checked against independent values.
