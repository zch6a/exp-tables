"""Regression tests for what the round-4 review (the third) found in the standard-error and untested-cell code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp_tables.cli import main
from exp_tables.data import read_csv, read_lm_eval
from exp_tables.report import comparisons
from exp_tables.table import build

SE_CSV = "model,acc,acc_stderr,f1,f1_se\nbig,0.80,0.010,0.70,0.02\nsmall,0.77,0.011,0.69,0.02\ntiny,0.60,0.012,0.50,0.02\n"


def test_the_command_line_reads_error_columns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "se.csv"
    p.write_text(SE_CSV, encoding="utf-8")
    assert main([str(p), "--format", "markdown", "--save-config", str(tmp_path / "t.json")]) == 0
    out = capsys.readouterr()
    assert out.out.splitlines()[0] == "| model | acc ↑ | f1 ↑ |" and "| big | **0.800** ± 0.010 |" in out.out
    assert "read as standard errors (of the metric they are named after): acc_stderr, f1_se" in out.err
    assert json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))["spec"]["metrics"] == ["acc", "f1"]
    assert main(["--config", str(tmp_path / "t.json"), "--format", "markdown"]) == 0


def test_untested_cells_are_not_underlined_either() -> None:
    t = build(read_csv("model,acc,acc_stderr\nbig,0.80,\nmid,0.78,0.01\nsmall,0.70,0.01\ntiny,0.60,0.01\n", "r.csv"))
    big, mid, small, tiny = (row[0] for row in t.cells)
    assert mid.best and not big.best and not big.second  # big has no error: untested, unmarked
    assert not small.best and not tiny.best
    rows = "method,seed,acc\n" + "".join(f"{m},{s},{v + s / 10}\n" for m, v in (("A", 90), ("B", 80)) for s in range(3)) + "C,0,95\n"
    t = build(read_csv(rows, "r.csv"))
    assert [(row[0].best, row[0].second) for row in t.cells] == [(True, False), (False, True), (False, False)]  # B tested: second; C not


def test_a_column_with_nothing_to_test_against_is_left_unmarked() -> None:
    rows = "method,seed,acc,acc_stderr,f1\nA,1,50.0,,1\nA,2,50.2,,2\nB,1,,,1.5\nB,2,,,2.5\nC,,90.0,0.3,\nD,,91.0,0.3,\n"
    t = build(read_csv(rows, "r.csv"))
    acc = [row[0] for row in t.cells]
    # one row with seeds, two single results: the singles are tested against each other (D beats C? 1.0 / 0.42 = 2.4)
    assert [c.best for c in acc] == [False, False, False, True] and any(c.metric == "acc" for c in t.comparisons)
    rows = "method,seed,acc,f1\nA,1,50.0,1\nA,2,50.2,2\nB,1,,1.5\nB,2,,2.5\nC,,90.0,\nD,,91.0,\n"
    t = build(read_csv(rows, "r.csv"))
    assert not any(row[0].best or row[0].second for row in t.cells)  # acc: nothing testable at all
    assert any("could not be tested" in n for n in t.notes)


def test_an_orphan_error_column_is_not_a_metric() -> None:
    t = build(read_csv("model,acc,acc_stderr\nbig,0.80,0.01\nmid,,0.02\nsmall,0.70,0.01\n", "r.csv"))
    assert t.metrics == ["acc"]


def test_negative_errors_are_ignored_and_z_tests_have_no_effect_sizes() -> None:
    t = build(read_csv("model,acc,acc_stderr\nbig,0.80,-0.01\nsmall,0.70,0.01\nmid,0.75,0.01\n", "r.csv"))
    assert t.cells[0][0].error is None
    assert all(c.hedges_g is None and c.cliffs_delta is None for c in t.comparisons)
    assert "Hedges" not in comparisons(t)


def test_lm_eval_sources_name_the_file() -> None:
    runs = read_lm_eval({"results": {"a": {"acc,none": 0.5}, "b": {"acc,none": 0.6}}, "versions": {}, "model_name": "m"}, "r.json")
    assert [r.source for r in runs] == ["r.json:a", "r.json:b"]


@pytest.mark.parametrize("readme", ["README.md", "README.zh-CN.md"])
def test_the_readme_report_is_what_the_tool_says(readme: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = Path(__file__).resolve().parent.parent
    text = (root / readme).read_text(encoding="utf-8")
    block = text.split("```text\n", 1)[1].split("```", 1)[0]
    shown = [line.strip() for line in block.replace("\n  ", " ").splitlines() if line.strip()]  # wrapped lines joined
    args = [str(root / "examples" / "results.csv"), "--label", "accuracy=Accuracy (%)", "--label", "ece=ECE (%)", "--label", "method=Method"]
    assert main([*args, "--report", "-o", str(tmp_path / "table.tex")]) == 0
    report = capsys.readouterr().out.splitlines()
    assert shown and all(line in report for line in shown), [line for line in shown if line not in report]


def test_a_wildcard_the_shell_left_alone_is_expanded(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for i, value in enumerate((0.5, 0.6)):
        (tmp_path / f"results_{i}.json").write_text(json.dumps({"model": f"m{i}", "acc": value}), encoding="utf-8")
    assert main([str(tmp_path / "results_*.json"), "--format", "markdown"]) == 0  # PowerShell passes the pattern as it is
    out = capsys.readouterr().out
    assert "| m0 |" in out and "| m1 |" in out
    assert main([str(tmp_path / "nothing_*.json")]) == 2
    assert "no such file or folder" in capsys.readouterr().err


def test_other_tests_on_single_results_say_so() -> None:
    t = build(read_csv(SE_CSV, "r.csv"), __import__("exp_tables").Spec(test="welch"))
    assert any("not by the welch test" in n for n in t.notes)
