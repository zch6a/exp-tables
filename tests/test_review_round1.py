"""Regression tests for what the round-1 review found, each on the smallest input that showed it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp_tables.cli import main
from exp_tables.data import read_csv, read_json, read_mlflow
from exp_tables.render import legend
from exp_tables.table import Spec, build


def run(args: list[object], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_one_run_per_method_is_one_row_per_method() -> None:
    t = build(read_csv("method,acc\nA,90.1\nB,91.2\nC,89.3\n", "r.csv"))
    assert t.rows == ["A", "B", "C"] and t.cells[1][0].best
    t = build(read_json('{"ours": {"acc": 90.1}, "base": {"acc": 88.0}, "abl": {"acc": 89.0}}', "r.json"))
    assert t.rows == ["ours", "base", "abl"]
    # with seeds, a field naming every run apart is a run id, not the method
    t = build(read_csv("name,seed,acc\nr1,1,90\nr2,2,91\nr3,3,92\n", "r.csv"))
    assert t.rows == [""] and t.cells[0][0].summary is not None and t.cells[0][0].summary.n == 3


def test_seed_values_like_seed_42_make_the_seed_field() -> None:
    t = build(read_csv("run,test_top1\nseed 42,0.90\nseed 1,0.89\nseed 2,0.91\n", "r.csv"))
    assert t.rows == [""] and t.cells[0][0].summary is not None and t.cells[0][0].summary.n == 3


def test_nan_and_infinities_are_missing() -> None:
    runs = read_json('[{"method": "A", "seed": 1, "acc": NaN}, {"method": "A", "seed": 2, "acc": 90}, {"method": "A", "seed": 3, "acc": 90.2},'
                     ' {"method": "B", "seed": 1, "acc": 89}, {"method": "B", "seed": 2, "acc": 89.5},'
                     ' {"method": "C", "seed": 1, "acc": 80}, {"method": "C", "seed": 2, "acc": 80.5},'
                     ' {"method": "D", "seed": 1, "acc": Infinity}]', "r.json")  # fmt: skip
    t = build(runs)
    assert t.rows == ["A", "B", "C", "D"] and t.cells[0][0].summary.values == [90.0, 90.2]  # type: ignore[union-attr]
    assert t.cells[3][0].summary is None
    c = next(c for c in t.comparisons if c.other == "C")
    assert c.significant and not t.cells[2][0].best  # one NaN no longer pins every corrected p to 1


def test_missing_markers_keep_a_column_numeric_and_an_index_column_is_dropped() -> None:
    runs = read_csv(",method,seed,acc,loss\n0,A,1,90,0.5\n1,A,2,91,nan\n2,B,1,80,n/a\n3,B,2,81,0.6\n", "r.csv")
    assert runs[1].metrics == {"acc": 91.0} and "loss" not in runs[1].fields and "" not in runs[0].fields
    t = build(runs)
    assert t.metrics == ["acc", "loss"] and t.groups == ["", ""]


def test_no_metric_is_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "r.csv"
    p.write_text("method,seed,note\nA,1,good\nB,1,bad\n", encoding="utf-8")
    code, out, err = run([p], capsys)
    assert code == 2 and not out and "no metric to show" in err


def test_tiny_metrics_get_enough_decimals() -> None:
    rows = ["method,seed,mse"] + [f"A,{s},{v}" for s, v in enumerate((1.0e-5, 1.1e-5, 0.9e-5))]
    rows += [f"B,{s},{v}" for s, v in enumerate((3.0e-5, 3.2e-5, 2.9e-5))]
    t = build(read_csv("\n".join(rows) + "\n", "r.csv"))
    assert t.decimals == [7] and t.cells[0][0].best and not t.cells[1][0].best


def test_equal_as_printed_but_significantly_worse_is_not_bold() -> None:
    rows = ["method,seed,acc"] + [f"A,{s},{v}" for s, v in enumerate((93.14, 93.141, 93.139))]
    rows += [f"B,{s},{v}" for s, v in enumerate((93.06, 93.061, 93.059))]
    t = build(read_csv("\n".join(rows) + "\n", "r.csv"), Spec(decimals={"acc": 1}))
    assert t.comparisons[0].significant and t.cells[0][0].best and not t.cells[1][0].best


def test_json_output_is_strict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "r.csv"
    p.write_text("method,seed,acc\nA,1,91\nA,2,91\nB,1,90\nB,2,90\nC,1,80\nC,2,80\n", encoding="utf-8")
    code, out, _ = run([p, "--format", "json"], capsys)

    def reject(name: str) -> None:
        raise ValueError(name)

    data = json.loads(out, parse_constant=reject)  # no Infinity, no NaN
    assert code == 0 and data["comparisons"][0]["hedges_g"] is None


def test_mlflow_tags_and_deleted_runs(tmp_path: Path) -> None:
    exp = tmp_path / "mlruns" / "1"
    for run_id, method, value, stage in (("a", "ours", 0.9, "active"), ("b", "base", 0.8, "active"), ("c", "base", 0.99, "deleted")):
        d = exp / run_id
        (d / "metrics").mkdir(parents=True)
        (d / "tags").mkdir()
        (d / "metrics" / "acc").write_text(f"1 {value} 0\n", encoding="utf-8")
        (d / "tags" / "method").write_text(method, encoding="utf-8")
        (d / "tags" / "mlflow.source.name").write_text("train.py", encoding="utf-8")
        (d / "meta.yaml").write_text(f"run_id: {run_id}\nlifecycle_stage: {stage}\n", encoding="utf-8")
    runs = read_mlflow(tmp_path / "mlruns")
    assert [(r.fields["method"], r.metrics["acc"]) for r in runs] == [("ours", 0.9), ("base", 0.8)]
    assert all("mlflow.source.name" not in r.fields for r in runs)


def test_a_numeric_setting_can_be_filtered_on(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "r.csv"
    p.write_text("method,seed,ratio,acc\nA,1,0.1,90\nA,1,0.5,80\nB,1,0.1,85\nB,1,0.5,70\n", encoding="utf-8")
    code, out, err = run([p, "--filter", "ratio=0.10", "--format", "markdown"], capsys)
    assert code == 0 and "| A | **90** |" in out and "ratio" not in out.splitlines()[0]


def test_options_are_checked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "r.csv"
    p.write_text("method,dataset,seed,acc\nA,a,1,90\nA,a,2,91\nB,a,1,80\nB,a,2,81\nA,b,1,70\nB,b,1,60\n", encoding="utf-8")
    code, out, _ = run([p, "--columns", "", "--format", "markdown"], capsys)
    assert code == 0 and out.splitlines()[0] == "| method | acc ↑ |"  # no groups when asked for none
    for bad in (["--alpha", "5"], ["--decimals", "acc=-1"]):
        assert main([str(p), *bad]) == 2
    cfg = tmp_path / "t.json"
    cfg.write_text(json.dumps({"inputs": ["r.csv"], "spec": {"correction": "Holm"}}), encoding="utf-8")
    assert main(["--config", str(cfg)]) == 2
    assert "correction is holm, bh or none, not 'Holm'" in capsys.readouterr().err
    code, out, _ = run([p, "--correction", "none", "--format", "markdown", "--columns", ""], capsys)
    assert "uncorrected" in out


def test_check_reads_utf16_and_ignores_the_version_in_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "r.csv"
    p.write_text("method,seed,acc\nA,1,90\nA,2,91\nB,1,80\nB,2,81\n", encoding="utf-8")
    code, tex, _ = run([p], capsys)
    (tmp_path / "t.tex").write_text(tex, encoding="utf-16")  # PowerShell 5's > writes this
    assert run([p, "--check", tmp_path / "t.tex"], capsys)[0] == 0
    code, js, _ = run([p, "--format", "json"], capsys)
    (tmp_path / "t.json").write_text(js.replace('"tool_version": "', '"tool_version": "0.0.0-'), encoding="utf-8")
    assert run([p, "--format", "json", "--check", tmp_path / "t.json"], capsys)[0] == 0
    (tmp_path / "bad.tex").write_bytes(b"\xff\x00\xc3")
    assert run([p, "--check", tmp_path / "bad.tex"], capsys)[0] == 2


def test_wilcoxon_with_five_pairs_says_it_cannot_reach_alpha() -> None:
    rows = ["method,seed,acc"] + [f"A,{s},{90 + s / 10}" for s in range(5)] + [f"B,{s},{80 + s / 10}" for s in range(5)]
    t = build(read_csv("\n".join(rows) + "\n", "r.csv"), Spec(test="wilcoxon"))
    assert t.cells[1][0].best and any("cannot reach p < 0.05" in n for n in t.notes)
    assert "Wilcoxon" in legend(t)
