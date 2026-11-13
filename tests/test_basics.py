"""Readers, summaries and the LaTeX table, on synthetic results written by the tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from exp_tables.data import read, read_csv, read_json, read_jsonl
from exp_tables.stats import Summary
from exp_tables.table import Spec, build, latex, resolve

CSV = """method,dataset,seed,accuracy,loss
ours,cifar10,0,93.0,0.31
ours,cifar10,1,93.4,0.29
ours,cifar10,2,92.9,0.30
baseline,cifar10,0,91.2,0.35
baseline,cifar10,1,91.0,0.36
baseline,cifar10,2,91.5,0.34
ablation,cifar10,0,92.1,0.33
ablation,cifar10,1,92.4,0.32
ablation,cifar10,2,92.0,0.33
ours,cifar100,0,74.1,1.10
ours,cifar100,1,73.8,1.12
baseline,cifar100,0,70.2,1.30
baseline,cifar100,1,70.6,1.28
"""


def test_csv_fields_and_metrics() -> None:
    runs = read_csv(CSV, "results.csv")
    assert len(runs) == 13
    assert runs[0].fields == {"method": "ours", "dataset": "cifar10", "seed": "0"}  # the seed is a field, not a metric
    assert runs[0].metrics == {"accuracy": 93.0, "loss": 0.31}
    assert runs[0].source == "results.csv:2"


def test_tsv_percent_signs_and_empty_cells() -> None:
    runs = read_csv("model\tseed\tacc\tF1\nA\t1\t91.5%\t\nB\t1\t90.0%\t0.8\n", "r.tsv")
    assert runs[0].metrics == {"acc": 91.5} and runs[1].metrics == {"acc": 90.0, "F1": 0.8}


def test_json_shapes() -> None:
    nested = [{"method": "ours", "seed": 1, "config": {"lr": 0.1}, "metrics": {"acc": 0.93}}]
    (r,) = read_json(json.dumps(nested))
    assert r.fields == {"method": "ours", "seed": "1", "config/lr": "0.1"} and r.metrics == {"metrics/acc": 0.93}
    by_name = read_json(json.dumps({"ours-s1": {"acc": 0.9}, "ours-s2": {"acc": 0.91}}))
    assert [x.fields["run"] for x in by_name] == ["ours-s1", "ours-s2"]
    wrapped = read_json(json.dumps({"runs": [{"method": "a", "acc": 1}], "version": 2}))
    assert wrapped[0].metrics == {"acc": 1.0}
    lines = read_jsonl('{"method": "a", "acc": 0.5}\n\n{"method": "b", "acc": 0.6}\n')
    assert [x.fields["method"] for x in lines] == ["a", "b"]


def test_mlflow_folder(tmp_path: Path) -> None:
    exp = tmp_path / "mlruns" / "1"
    exp.mkdir(parents=True)
    (exp / "meta.yaml").write_text("experiment_id: '1'\nname: cifar\n", encoding="utf-8")
    for run_id, method, seed, accs in (("aaa", "ours", "0", [80, 92, 93.1]), ("bbb", "baseline", "0", [70, 90, 91.0])):
        run = exp / run_id
        (run / "metrics").mkdir(parents=True)
        (run / "params").mkdir()
        (run / "tags").mkdir()
        (run / "params" / "method").write_text(method, encoding="utf-8")
        (run / "params" / "seed").write_text(seed, encoding="utf-8")
        (run / "tags" / "mlflow.runName").write_text(f"{method}-{seed}", encoding="utf-8")
        (run / "metrics" / "accuracy").write_text("".join(f"16990000{k} {a} {k}\n" for k, a in enumerate(accs)), encoding="utf-8")
    runs = read(tmp_path)
    assert {r.fields["method"]: r.metrics["accuracy"] for r in runs} == {"ours": 93.1, "baseline": 91.0}  # the last step
    assert runs[0].fields["experiment"] == "cifar" and runs[0].fields["run"] == "ours-0"


def test_summary() -> None:
    s = Summary([93.0, 93.4, 92.9])
    assert math.isclose(s.mean, 93.1) and math.isclose(s.std, 0.2645751311, rel_tol=1e-9) and s.n == 3
    assert Summary([5.0]).std == 0 and Summary([5.0]).ci() == 0


def test_the_table_is_worked_out_from_the_runs() -> None:
    runs = read_csv(CSV, "results.csv")
    spec = resolve(runs, Spec())
    assert (spec.rows, spec.columns, spec.seed, spec.metrics) == (["method"], ["dataset"], "seed", ["accuracy", "loss"])
    t = build(runs)
    assert t.rows == ["ours", "baseline", "ablation"]
    assert t.groups == ["cifar10", "cifar10", "cifar100", "cifar100"] and t.directions == ["higher", "lower", "higher", "lower"]
    acc10 = [row[0] for row in t.cells]
    assert [c.best for c in acc10] == [True, False, False] and [c.second for c in acc10] == [False, False, True]
    assert t.cells[2][2].summary is None  # no ablation on cifar100
    assert not any(row[2].second for row in t.cells)  # two methods: no second best


def test_latex() -> None:
    t = build(read_csv(CSV, "results.csv"), Spec(labels={"accuracy": "Acc. (%)", "method": "Method"}))
    tex = latex(t)
    assert tex.startswith("\\begin{tabular}{lrrrr}\n\\toprule\n")
    assert " & \\multicolumn{2}{c}{cifar10} & \\multicolumn{2}{c}{cifar100} \\\\" in tex
    assert "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}" in tex
    assert "Method & Acc. (\\%) $\\uparrow$ & loss $\\downarrow$ & Acc. (\\%) $\\uparrow$ & loss $\\downarrow$ \\\\" in tex
    assert "ours & \\textbf{93.10}\\,{\\scriptsize$\\pm$0.26}" in tex  # two significant digits of the spread
    assert "ablation & \\underline{92.17}" in tex and tex.rstrip().endswith("\\end{tabular}")
    assert "& -- \\\\" in tex.splitlines()[-3]  # ablation has no cifar100 result


@pytest.mark.parametrize("spread", ["sem", "ci", "none"])
def test_spreads(spread: str) -> None:
    tex = latex(build(read_csv(CSV, "results.csv"), Spec(spread=spread)))
    first = next(line for line in tex.splitlines() if line.startswith("ours"))
    if spread == "none":
        assert "\\pm" not in first
    else:
        assert "\\pm" in first
