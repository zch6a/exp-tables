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

### 2026-11-13 — polishing round 2

Round 1's CI run was green on every job.

**Real logs.** Fetched to a temporary folder from the Hugging Face Hub, deleted after; their shapes kept as
synthetic tests in `tests/test_round2.py`.

- MultiBERTs fine-tunes (3 pre-training seeds × 5 fine-tuning seeds, 15 TF2/Flax event files of 5.4 MB): read
  right (tensor scalars, last step), and the table's choice (rows: the fine-tuning run; seeds: the pre-training
  seed) is sensible; `--seed ft --columns none` gives the other view. Found:
  - Reading took 20 s for 80 MB. Most varints are one byte (a fast path), and only the records whose values are
    kept need their CRC32C checked (if one fails, the file is read again checking every record, so a corrupt record
    still ends it): 10 s, the same output byte for byte.
  - The logged learning rate was a metric, with ten decimals of float noise. Logged settings (learning rate, epoch,
    step) are left out of the default metrics; a spread below 10⁻⁹ of the mean is noise.
- Open LLM Leaderboard results (lm-evaluation-harness `results_*.json`, three Llama 2 models): read as one run with
  hundreds of "metrics" from its config, and 429 "shared seed" notes. Now its own reader: one run per model and
  task, metrics by name with their filter (`exact_match (strict-match)`), standard errors left out; the model from
  `model_name` or `model_args`. `--filter task=a,b` keeps either.
- `--columns ""` cannot be typed in PowerShell 5 (it drops empty arguments): `--columns none`.

**Second independent review** (statistics against independent values, the marks, rendering, the API; 9 confirmed
defects, all fixed, tests in `tests/test_review_round2.py`). It confirmed the t distribution (published quantiles,
exact formulas for df 1–3, integration at fractional df), Welch's and the paired t, Wilcoxon (all 2ⁿ sign patterns,
1,440 cases), Holm and BH (311 cases), Hedges' g, Cliff's delta, and seeds needed (simulated power 0.79–0.91 at the
n given, below 0.8 at n − 1). Fixed:

1. LaTeX rows starting with `[` did not compile (read as `\\[...]`), and a leading `*` vanished: such labels get `{}`.
2. The bootstrap's early stop looked at α, but after Holm the decision is at α/k…α: it now stops only when P is far
   from every level the correction can hold it to (two of 150 random tables had a different bold before).
3. Cells that could not be tested (one seed) were bold or not as printed under a legend saying bold means tested.
   Now the column's best is chosen among the rows with two or more seeds, every other row is tested against it, and
   a row with one seed is left unmarked, with a note counting such cells. (A table with no tests at all still marks
   the best as printed, and its legend says "untested".)
4. "About N seeds would tell" could name N already there: it now uses the level the corrected p is held to, the
   pairs Wilcoxon needs to reach it at all (2/2ⁿ < level), and is always more than there are.
5. `import exp_tables` gave nothing, `read()` wanted a `Path`, and `build()` took any `Spec` (`spread="CI"`,
   `test="Wilcoxon"`, a misspelt row field) and made a wrong table. The package exports the API (with a doctest),
   `read` takes a string, and `build` checks the spec (the checks moved from the CLI to `table.check`).
6. A line break in a label broke the Markdown row.
7. Wilcoxon missed ties among large values (rounded at 12 digits of the difference): now relative to the values.
8. Greek letters stopped pdflatex: written as math (`\ensuremath{\beta}`; capitals that look Latin as Latin).
   `<`, `>`, `|` escaped too. Other scripts (CJK) need the paper's own setup, as the README says.
9. Corrected p values were rounded to 12 decimals, so 0.04999999999999 became 0.05: now 15.

Also: a metric named `reward ↓` printed two arrows.

The example and the README's table and report are unchanged. LaTeX with the nastiest labels compiles with pdflatex
(local MiKTeX, packages already installed, the installer disabled).

Tests: 94 (were 78), Python 3.10 and 3.13; ruff, mypy clean.

Next: round 3. Worth doing: lm-evaluation-harness gives a standard error per metric, so a single-run comparison of
models could be tested with it (a z test on the difference) instead of being untested.

### 2026-11-13 — polishing round 3

Round 2's CI first failed on `ruff format --check`: ruff also formats Python blocks inside the READMEs, and my commit
gate only ran `ruff check`. Fixed (the README's comments as ruff aligns them), and the gate now runs both; CI green.

**Single results tested by their standard errors.** LLM evaluations rarely have seeds, so round 2's leaderboard
tables were all "untested". lm-evaluation-harness reports a standard error per metric (over its test items); so do
many CSVs (`acc_stderr`, `acc_se`). Now:

- runs carry `errors` (a metric's standard error): the harness reader keeps them, and any `X_stderr` / `X_se` /
  `X_sem` / `X standard error` metric beside `X` becomes `X`'s error rather than a column of its own;
- two single results are compared by a z test (difference over the root of the summed squared errors; independent
  errors, conservative when both were measured on the same items), corrected per column like the rest;
- the error is printed as the single result's ±, in every format, with decimals from it; the legend says so ("one
  run per cell, ± its standard error as the evaluation reported it", or the same as an aside beside seeds), the
  report line gives both errors instead of seed counts and effect sizes, the JSON has each cell's `error`;
- single results beside rows with seeds are not tested (the two kinds of spread are not comparable), and the note
  about untested cells says so.

On the round-2 leaderboard shape (two models, harness errors 0.01): 0.58 vs 0.53 is now significant (z = 3.5) and
only the better model is bold, where before both were "untested".

**The Action on a realistic repository** (a made-up paper repository: results/, tables/main.json from
`--save-config`, paper/tables/main.tex), run locally through Git Bash with GitHub's variables (`GITHUB_OUTPUT`,
`GITHUB_STEP_SUMMARY`, `RUNNER_TEMP`): matching (exit 0, `matches=true`), stale with `fail: true` (exit 1, the diff in
the summary) and `fail: false` (exit 0, `matches=false`), and a missing config (exit 2). Found and fixed:

- uv's "Building … Installed" lines went into the summary's diff block: `uvx -q`.
- Errors about the runs or the config printed the whole usage text before the message (in the summary too): now one
  line, "exp-tables: error: … (exp-tables --help lists the options)"; argparse's own errors keep the usage.
- Paths in messages printed with backslashes on Windows: POSIX form.

**The rest.** The Skill mentions the harness, the z test and the notes (pass them on, fix the layout they point at);
README and CHANGELOG updated. `--help`, first run and error messages re-read.

Tests: 98 (were 94), Python 3.10 and 3.13; ruff (check and format), mypy clean.

Next: round 4.

### 2026-11-13 — polishing round 4

Round 3's CI was green on every job.

**Docs and packaging.** The README opens with one sentence of what it is for and says what the CI check does;
CONTRIBUTING has the module map as it now is and how the tests are laid out (by round, one test per review finding);
the package metadata names every reader, `py.typed` ships, the sdist leaves out the GitHub-only images (164 KB → 112
KB); SECURITY lists what is escaped and that JSON output is strict. Checked: no network, subprocess, eval or pickle
in the package; every workflow `contents: read` except the release job's publish step; every action pinned by commit;
the wheel holds only the package.

**Third independent review** (the standard errors, the z test, the untested-cell rule, the Action, the README; it
confirmed the z test, Holm/BH over it, the harness reader's filters and "N/A" errors, the legend in every mix, the
CLI's error paths and the Action's exit logic). Fixed, with tests in `tests/test_review_round3.py`:

1. **The command line failed on every CSV with an `X_stderr` column** (round 3 tested only `build()`): its check and
   layout saw the error columns as metrics, which `build()` then removed. Error columns are now left out by `resolve()`
   and turned into errors only by `build()`, which notes which columns it read as errors; `--metrics acc_stderr`
   says why it is not a metric. `--save-config` and the Action work on such files.
2. **Untested cells were underlined as the second best**, even above the bold one: the second best is now chosen
   among tested cells only.
3. **A column with one row with seeds bolded that row untested**, even when it was the worst: a column is tested
   among rows with seeds when two or more have them, else among single results with errors when two or more have
   them, else not at all, and then it is left unmarked as a whole (counted in the note).
4. **The README's report said "about 9 seeds"; the tool says 10** since round 2 counts the correction. Fixed, and a
   test now checks every line of both READMEs' report against the tool's output.
5. An error column for a run that lacked the metric became a column of its own (with an arrow): error columns are
   never metrics.
6. `read_lm_eval` reused `name` inside its loop, so runs' `source` named a metric, not the file.
7. Smaller: z tests report no effect sizes (null in JSON; the report line gives the errors); negative errors are
   ignored; the "no comparisons" message and the legend mention standard errors; `--test welch` (or another) on
   single results notes that the z test was used; the remaining backslash paths in messages.

From its "plausible" list: the Action treated uvx's own failures (exit 1) as a stale table; now exit 1 counts as "does
not match" only when exp-tables says so, else it is an error (2). The legend says "reported standard error" rather
than "as the evaluation reported it", since a CSV's `acc_se` may come from anywhere.

Tests: 107 (were 98), Python 3.10 and 3.13; ruff (check and format), mypy clean.

Next: round 5, then v0.1.0.

### 2026-11-13 — polishing round 5

Round 4's CI was green on every job.

**As a new user would.** `uvx --from git+https://github.com/zch6a/exp-tables exp-tables` with an empty uv cache:
installs and runs (version, the example's table). In PowerShell 5: `> table.tex` writes a UTF-8 file with a BOM (and
UTF-16 elsewhere) and `--check` reads both; labels with spaces, `%` and Chinese, and `--filter "method=TinyNet
(ours),ResNet-18"`, come through. Found:

- **Wildcards.** PowerShell and cmd pass `results_*.json` to the program as it is, so "no such file". A pattern that
  is not a file is now expanded by the tool (sorted; one that matches nothing still says "no such file").
- **`--help`** lagged behind: the lm-evaluation-harness input, `a,b` filters, `--check`, the notes, what the default
  metrics leave out. Rewritten with an example per use.

**Everything once more.** Eight input kinds (CSV, TSV, JSON list, JSON by name, JSON Lines, MLflow, TensorBoard,
harness files by wildcard) × four formats, each made with `--report` and `--save-config`, then remade from the config
with `--check` against itself: 32 of 32 made, matched and marked.

**The rest of the round.** CHANGELOG rewritten for the release as the tool now is; README unchanged (its report is
under test since round 4); ruff check and format, mypy, tests on 3.10 and 3.13.

Tests: 108 (were 107).

Next: v0.1.0.

### 2026-11-13 — v0.1.0

Round 5's CI was green on every job. Tagged `v0.1.0` (annotated) on cb4e89d and pushed; the release workflow's build
job (tests, build, read-only) and publish job (gh only) both passed: the GitHub release "exp-tables 0.1.0" has the
CHANGELOG section, the wheel and the source archive, and `v0` points at the release for `zch6a/exp-tables@v0`.
Not published to PyPI (the user's decision).

Five rounds: ten public results files, MultiBERTs TensorBoard logs and Open LLM Leaderboard results tried; three
independent reviews; 55 tests at the first commit, 108 at the release.

### 2026-11-14 — Chinese README: AI tone

- 4 places: a pair of decorative dashes in the statistics paragraph made plain sentences, two semicolons joining independent sentences split into full stops (intro and the hand-made-tables paragraph), a 3-item 顿号 list in a sentence reworded (the job summary line).
- The checks follow the findings of lieflat-less-ai-tone (shiujan, CC BY-NC 4.0); its findings applied, its text not copied.
- Left as they are: 加粗的意思是…不是“接近” (a real, demonstrated misreading), the supported-format enumeration, colon-led lists, table cells, the code blocks the README test reads.