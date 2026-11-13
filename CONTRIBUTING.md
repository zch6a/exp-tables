# Contributing

The most useful contributions are **results it reads wrongly** (an export from a tracker it does not know, a layout
it guesses badly) and **statistics it gets wrong** (with the reference value and where it comes from).

## Development

```bash
uv sync
uv run pytest -q          # no test reaches the network: results and event files are written by the tests
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

A statistical function comes with a doctest against a value checked independently (a table, a hand calculation, a
numerical integration), and says where the value comes from. The example is made by `scripts/make_example.py`
(seeded); after a change to the output, remake `examples/table.tex` with
`uv run exp-tables --config examples/table.json -o examples/table.tex`.

## Where things are

| | |
|---|---|
| `__init__.py` | the Python API (`read`, `build`, `Spec`, `latex`, `markdown`, `html`, `comparisons`) |
| `data.py` | reading CSV/TSV, JSON, JSON Lines, MLflow, lm-evaluation-harness results into runs (fields, metrics, errors) |
| `tensorboard.py` | TensorBoard event files (TFRecord, protobuf) |
| `stats.py` | summaries over seeds, Student's t distribution |
| `significance.py` | tests (t, Wilcoxon, bootstrap, z on standard errors), corrections, effect sizes, seeds needed |
| `table.py` | the table: layout worked out, spec checks, standard errors, marks and tests per column, notes; LaTeX |
| `render.py` | Markdown and HTML, the legend |
| `report.py` | the comparisons in words |
| `cli.py` | the command line, filters, several inputs, the config file, `--check`, JSON |
| `action.yml` | the GitHub Action around `--check` |
| `scripts/` | the example's data, the demo image, the social card |

The tests are by round: `test_real_shapes.py`, `test_round2.py`, `test_round3.py` rebuild shapes seen in public
results files; `test_review_round1.py` and `test_review_round2.py` hold one test per defect an independent review
found.
