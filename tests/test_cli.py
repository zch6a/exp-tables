"""The command line, on result files written by the tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_basics import CSV

from exp_tables.cli import main


@pytest.fixture
def results(tmp_path: Path) -> Path:
    p = tmp_path / "results.csv"
    p.write_text(CSV, encoding="utf-8")
    return p


def run(args: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_latex_by_default(results: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run([results], capsys)
    assert code == 0 and out.startswith("\\begin{tabular}") and "\\textbf{93.10}" in out


def test_formats_and_options(results: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run([results, "--format", "markdown", "--metrics", "accuracy", "--label", "accuracy=Acc. (%)", "--decimals", "accuracy=1"], capsys)
    assert code == 0 and out.splitlines()[0] == "| method | cifar10 · Acc. (%) ↑ | cifar100 · Acc. (%) ↑ |"
    assert "| ours | **93.1** ± 0.3 |" in out
    code, out, _ = run([results, "--format", "html", "--spread", "none", "--test", "none"], capsys)
    assert code == 0 and "<table" in out and "±" not in out.split("<caption")[0]


def test_json(results: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run([results, "--format", "json"], capsys)
    data = json.loads(out)
    assert code == 0 and data["tool"] == "exp-tables" and [c["metric"] for c in data["columns"]] == ["accuracy", "loss", "accuracy", "loss"]
    ours = data["rows"][0]
    assert ours["row"] == "ours" and ours["cells"][0]["values"] == [93.0, 93.4, 92.9] and ours["cells"][0]["best"]
    assert data["rows"][2]["cells"][2] is None  # no ablation on cifar100
    assert all("significant" in c and "p_adjusted" in c for c in data["comparisons"])


def test_filters_and_report(results: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    out_file = tmp_path / "t.tex"
    code, out, _ = run([results, "--filter", "dataset=cifar10", "--report", "-o", out_file], capsys)
    table = out_file.read_text(encoding="utf-8")
    assert code == 0 and "cifar100" not in table and "cifar10" not in table  # one dataset: no column groups left
    assert "ours vs baseline" in out and "significant" in out  # the report, on stdout as the table went to the file


def test_errors_say_what_there_is(results: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(results), "--metrics", "bleu"]) == 2
    assert "no metric 'bleu' in the runs; their metrics: accuracy, loss" in capsys.readouterr().err
    assert main([str(results), "--rows", "model"]) == 2
    assert "no field 'model' in the runs; their fields: method, dataset, seed" in capsys.readouterr().err
    assert main([str(results), "--filter", "dataset=imagenet"]) == 2
    assert "no run matches the filters" in capsys.readouterr().err
    assert main([str(results), "--better", "loss=down"]) == 2
    assert "takes higher or lower" in capsys.readouterr().err
    assert main(["missing.csv"]) == 2
    assert "no such file or folder" in capsys.readouterr().err


def test_several_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "ours.json").write_text(json.dumps([{"seed": s, "acc": 90 + s / 10} for s in range(3)]), encoding="utf-8")
    (tmp_path / "base.json").write_text(json.dumps([{"seed": s, "acc": 85 + s / 10} for s in range(3)]), encoding="utf-8")
    code, out, _ = run([tmp_path / "ours.json", tmp_path / "base.json", "--format", "markdown"], capsys)
    assert code == 0 and out.splitlines()[0].startswith("| file |") and "| ours | **90.10** ± 0.10 |" in out


def test_a_config_makes_the_same_table_again(results: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "tables" / "main.json"
    args = [results, "--label", "accuracy=Acc.", "--spread", "sem", "--format", "markdown", "--save-config", config]
    code, first, _ = run(args, capsys)
    saved = json.loads(config.read_text(encoding="utf-8"))
    assert code == 0 and saved["inputs"] == ["../results.csv"] and saved["format"] == "markdown"
    assert saved["spec"]["rows"] == ["method"] and saved["spec"]["seed"] == "seed" and saved["spec"]["spread"] == "sem"
    code, again, _ = run(["--config", config], capsys)
    assert code == 0 and again == first
    code, latex, _ = run(["--config", config, "--format", "latex"], capsys)  # an option given overrides the config
    assert latex.startswith("\\begin{tabular}")


def test_check_the_committed_table(results: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    committed = tmp_path / "table.tex"
    assert main([str(results), "-o", str(committed)]) == 0
    capsys.readouterr()
    assert main([str(results), "--check", str(committed)]) == 0
    assert "matches the results" in capsys.readouterr().out
    committed.write_text(committed.read_text(encoding="utf-8").replace("93.10", "93.40") + "  \n", encoding="utf-8")
    assert main([str(results), "--check", str(committed)]) == 1
    out = capsys.readouterr().out
    assert "does not match the results" in out and "-ours & \\textbf{93.40}" in out and "+ours & \\textbf{93.10}" in out
    assert main([str(results), "--check", str(tmp_path / "none.tex")]) == 2


def test_the_example_matches_its_results(capsys: pytest.CaptureFixture[str]) -> None:
    examples = Path(__file__).parent.parent / "examples"
    assert main(["--config", str(examples / "table.json"), "--check", str(examples / "table.tex")]) == 0
