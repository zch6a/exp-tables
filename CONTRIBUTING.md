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
| `data.py` | reading CSV/TSV, JSON, JSON Lines, MLflow into runs |
| `tensorboard.py` | TensorBoard event files (TFRecord, protobuf) |
| `stats.py` | summaries over seeds, Student's t distribution |
| `significance.py` | tests, corrections, effect sizes, seeds needed |
| `table.py` | the table: layout, marks, significance per column; LaTeX |
| `render.py` | Markdown and HTML |
| `report.py` | the comparisons in words |
| `cli.py` | the command line, the config file, JSON |
