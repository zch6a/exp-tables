"""Polishing round 3: single results tested with the standard errors their evaluation reports."""

from __future__ import annotations

import pytest

from exp_tables.data import read_csv
from exp_tables.render import legend, markdown
from exp_tables.report import comparisons
from exp_tables.table import build, latex


def test_stderr_columns_in_a_csv_test_single_results() -> None:
    runs = read_csv("model,acc,acc_stderr,f1,f1_se\nbig,0.80,0.010,0.70,0.02\nsmall,0.77,0.011,0.69,0.02\ntiny,0.60,0.012,0.50,0.02\n", "r.csv")
    t = build(runs)
    assert t.metrics == ["acc", "f1"]  # the errors are not columns
    marks = [[c.best for c in row] for row in t.cells]
    assert marks == [[True, True], [False, True], [False, False]]  # f1: 0.01 / 0.028 = 0.35, within the noise
    acc = {c.other: c for c in t.comparisons if c.metric == "acc"}
    # acc: z = 0.03 / sqrt(0.010^2 + 0.011^2) = 2.02, p = 0.044; Holm takes tiny's p (about 0) first, small's times 1
    assert acc["small"].p == pytest.approx(0.0436, abs=1e-3) and acc["small"].significant
    assert "± 0.010" in markdown(t) and r"\textbf{0.800}\,{\scriptsize$\pm$0.010}" in latex(t)
    assert legend(t).startswith("one run per cell, ± its standard error as the evaluation reported it; bold: the best")
    line = next(x for x in comparisons(t).splitlines() if "acc: big vs small" in x)
    assert line == "acc: big vs small +0.030 (z (standard errors), p = 0.044, corrected p = 0.044; standard errors 0.010 and 0.011): significant"


def test_single_results_beside_rows_with_seeds_are_not_tested() -> None:
    runs = read_csv("method,seed,acc,acc_stderr\nours,1,90.0,\nours,2,90.4,\nbase,1,85.0,\nbase,2,85.2,\nllm,,89.0,0.5\n", "r.csv")
    t = build(runs)
    assert [row[0].best for row in t.cells] == [True, False, False]
    assert any("could not be tested" in n for n in t.notes)
    assert "(a single run: ± its standard error, as reported)" in legend(t)
