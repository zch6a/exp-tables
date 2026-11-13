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

### 2026-11-13 — TensorBoard, Markdown, HTML

- **TensorBoard** (`tensorboard.py`), without TensorFlow: the TFRecord framing with both checksums (CRC32C, checked
  against the standard check value; a record whose checksum fails, or that is cut short by a run still writing,
  ends the file), and the protobuf fields a scalar needs: Event (wall time, step, summary), Summary.Value (tag,
  `simple_value`, tensor), TensorProto (dtype; `tensor_content` or packed values) — so PyTorch's SummaryWriter and
  TF2's `tf.summary.scalar` are both read. Each folder holding event files is a run; `ours/lr=0.1/seed_2` gives
  method `ours`, field `lr` = 0.1, seed 2. Each tag's value at its last step.
- **Markdown**: best in bold, second best `<ins>`, the spread after ±, column groups joined to the metric's name,
  then a legend saying what the marks and the spread mean (and which test, α, correction).
- **HTML**: a fragment with spanning group headers, `<b>` / `<u>`, the spread `<small>`, every label escaped, the
  legend as the caption.
- Tests write the event files byte by byte (a varint and field encoder, the TFRecord framing), both kinds of scalar,
  a cut record and a corrupt checksum.

Tests: 46, Python 3.10 and 3.14; ruff, mypy clean.

Next: the CLI, and a config file that reproduces a table (the spec as worked out, the inputs, the version).

### 2026-11-13 — the command line and the config file

- `exp-tables INPUT...`: CSV/TSV/JSON/JSONL files, TensorBoard folders, mlruns folders; several inputs are told apart
  by the field `file`. Options for everything the table works out (`--rows`, `--columns`, `--metrics`, `--seed`),
  `--filter FIELD=VALUE`, `--better`, `--label`, `--decimals`, `--spread`, `--test`, `--alpha`, `--correction`;
  `--format latex|markdown|html|json`; `--report` (on stdout when the table goes to a file, else on stderr).
- **A config that makes the table again**: `--save-config FILE` writes the inputs (relative to the config, with
  `..` when needed, so the two can move together), the filters, the spec as worked out (rows, columns, seed,
  metrics, every option), the format and the version; `--config FILE` makes the same table, byte for byte (a test
  checks it), options given beside it overriding it.
- **JSON**: each column (group, metric, direction, decimals), each cell's values with their seeds, the summary and
  the marks, every comparison with "significant".
- **Errors** name what there is ("no metric 'bleu' in the runs; their metrics: accuracy, loss"); exit 2 when no
  table can be made, also on an unexpected error.
- Found while testing: inputs outside the config's folder were saved as absolute paths (tying the config to one
  machine); now relative.

Tests: 53, Python 3.10 and 3.14; ruff, mypy clean.

Next: README (English, Chinese) with example data, CI, the GitHub Action, the Skill, a demo image; then the release
checks and the five polishing rounds.

### 2026-11-13 — --check, the example, the Action, the docs

- **`--check FILE`**: makes the table and compares it with a committed one; exit 0 when they match, 1 with a unified
  diff of the lines that differ, 2 when the file cannot be read. The epilog lists the exit statuses.
- **The example**: `scripts/make_example.py` writes `examples/results.csv` from a seeded generator (made-up results
  of a made-up paper: four methods, CIFAR-10/100, five seeds, accuracy and ECE), with `examples/table.json` (the
  config, labels included) and `examples/table.tex` (the table it makes). A test checks that the committed table
  still matches, so the example cannot drift from the code.
- **GitHub Action** (`action.yml`): `config` and `table` in, `matches` out; runs `exp-tables --config ... --check`;
  the job summary gets the diff, the Markdown table and the report; `fail: false` to only report. CI runs it on
  the example (matches) and on a copy with one number changed (does not match, with `fail: false`).
- **CI** on three systems with Python 3.10 and 3.13; **release workflow** copied from arxiv-diff (the fixed one
  that handles lightweight tags), publishing in a job that runs no project code.
- **Skill** (`plugin/`), SECURITY (no network at all), CONTRIBUTING (a module map), CHANGELOG, READMEs in English
  and Chinese with the example's real table and report, `docs/demo.png` (rendered from exp-tables' own HTML with
  headless Chrome, no network) and `docs/social-preview.png` (from `scripts/social-card.html`).
- Found while writing the docs:
  - ECE had no direction (no arrow, no test): `ece`, `mce`, `brier`, `calibration` are now lower-is-better.
  - The report printed raw names (`cifar100 · accuracy`) beside a table with labels; it now uses the labels.
  - HTML group headers were right-aligned over their columns; now centred.
  - The example shows the point of the tool: on CIFAR-10 ECE, ViT-S (2.61 ± 0.52) is bold and the ablation
    (2.60 ± 0.21) is not, because bold means "not significantly worse" and only the ablation's seeds show it is
    worse. The READMEs explain this instead of hiding it.

Tests: 55, Python 3.10 and 3.13; ruff, mypy clean.

Next: the public repository and CI; then five polishing rounds (independent reviews, real public result data),
then v0.1.0.

### 2026-11-13 — public

- Checked before publishing: commits by zch6a with the noreply address only, no attribution lines, no personal paths
  or names in the tree.
- https://github.com/zch6a/exp-tables (public, main), with a description and topics. The first CI run is green:
  three systems × Python 3.10/3.13, and the action job (the example matches; after one number is changed it does not,
  with `matches=false`).

Next: polishing round 1.

### 2026-11-13 — polishing round 1

**Real data.** Ten public multi-seed results CSVs from GitHub repositories (found by code search; fetched to a
temporary folder, never committed; layouts kept as synthetic tests in `tests/test_real_shapes.py`). What they showed:

- A second name of the method (`model_key` beside `model`) was taken as the column groups: a diagonal table with
  every cell bold. Column groups must now vary within a row; the chest X-ray results came out by data budget
  (10%, 20%, 100%) as they should.
- A table of one row bolded everything. A column needs two rows to have a best.
- `AUPRC`, `specificity`, `mcc`, `cost_efficiency`, `failure_rate` had no direction or the wrong one. More names,
  and when a name has words of both kinds the last decides (`cost efficiency` up; `cost per success` down; a
  generic `score` or `top-1` gives way: `brier score`, `top-1 error` down).
- Runs at several data ratios with the same seed were averaged as seeds, silently. Now a note says so and names the
  likely setting (a number read as a metric); such a number can be named in `--rows`, `--columns`, `--filter`.
- Hyperparameter columns of a wandb export (`lr`, `batch_size`) were shown as metrics; now they are fields.
- Decimals: a spread of 0.01 computed as 0.00999… gave a digit too many; a constant column (0.1 three times) had a
  spread of 1e-17 and four decimals; one very steady row stretched its whole column. Now: exactly 0 for equal values,
  the median spread, and up to 10 decimals so that 1e-5 is not 0.0000.

**Independent review** (a separate agent read every module and ran inputs against it; 13 confirmed defects, all
fixed, each with a regression test in `tests/test_review_round1.py`):

1. One run per method (no seeds) became one unnamed row averaging the methods: a field naming every run apart is
   now the method when there are no seeds, a run id only when there are; seed fields are also found by their values
   (`seed 42`, `s1`).
2. One `NaN` (JSON allows it) pinned every corrected p of its column to 1, so every row was bold; non-finite values
   are missing now.
3. `direction()` matched "lower" inside `flowers102` and `elbo_lower_bound`: only explicit markers count now
   ("lower is better", "(lower)", ↓), and the dataset after the metric (`acc/flowers102`) no longer hides it.
4. Metrics around 1e-5 printed as `0.0000`, all bold.
5. One `nan`/`n/a`/`-` in a column made it text, and text can become column groups; these are missing values now.
6. No metric at all gave exit 0 and broken LaTeX; now exit 2 with "no metric to show".
7. A pandas index column (empty header) was a metric; it is left out.
8. Rows equal as printed but significantly worse were bold, against the legend. The tests decide now.
9. `--format json` wrote `Infinity`; now `null`, with `allow_nan=False` as a guarantee.
10. A single-seed `NaN` crashed the decimals.
11. MLflow ignored tags (the README promised them) and read deleted runs.
12. `--filter` could not reach a number read as a metric, though the new note suggested it; numbers compare as
    numbers (`0.10` is `0.1`).
13. `--columns ""` could not turn groups off (it means none now; `Spec.columns=None` means worked out); `--test`,
    `--correction`, `--alpha` (0 < α < 1), `--decimals` (0–12) are checked, also from a config; the legend says
    "uncorrected" with `--correction none`.

Also from the review's "plausible" list: Markdown labels escape `\`, backticks, brackets and HTML; `--check` reads
UTF-16 (PowerShell 5's `>`), says so on other encodings (exit 2), and ignores `tool_version` in JSON; the Action
writes `matches` before the summary, puts stderr in the summary and makes the summary best-effort; Wilcoxon is now
exact with ties too (doubled average ranks; differences compared at 12 significant digits, so 90.1 − 80.1 and
90.2 − 80.2 tie) and notes when its pairs are too few to reach α; a paired test that falls back to Welch's says so.

**Speed.** 100,000 runs (20 methods × 10 datasets × 500 seeds × 3 metrics) took 133 s, nearly all in the power
calculation's t quantiles: the quantile is now cached, bracketed by doubling, and the normal one past 10⁶ degrees of
freedom; Cliff's delta sorts instead of comparing every pair. Now 3.4 s (Wilcoxon 2.8 s). The bootstrap stops early
once P is four standard errors from α; it is still the slow test (124 s on this extreme file), as the README says.

**The rest of the round.** `--help`, errors and exit statuses read again; the legend only states marks the table
has ("one run per cell: no spread" when there is none; "untested" when the best is marked without a test). SECURITY:
no network, no dependencies, nothing to audit beyond the standard library. README (both languages) and CHANGELOG
updated with what changed.

Tests: 78 (were 55), Python 3.10 and 3.13; ruff, mypy clean.

Next: round 2.
