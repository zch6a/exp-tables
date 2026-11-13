"""Polishing round 2: real TensorBoard logs (MultiBERTs fine-tunes) and lm-evaluation-harness results, rebuilt as
synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_tensorboard_render import event, tfrecord

from exp_tables.cli import main
from exp_tables.data import read
from exp_tables.table import build
from exp_tables.tensorboard import _last_values


def test_only_the_kept_records_need_their_checksum() -> None:
    data = bytearray(tfrecord(event(1, "loss", 0.9), event(2, "loss", 0.5), event(3, "loss", 0.3)))
    first_body_end = len(tfrecord(event(1, "loss", 0.9))) - 4
    data[first_body_end - 1] ^= 0xFF  # corrupt an early record's data: its value is not kept, the last one is
    assert _last_values(bytes(data))["loss"][2] == pytest.approx(0.3)
    data = bytearray(tfrecord(event(1, "loss", 0.9), event(2, "loss", 0.5), event(3, "loss", 0.3)))
    data[-5] ^= 0xFF  # corrupt the last record: read again, checking every record; the value before it is kept
    assert _last_values(bytes(data))["loss"][2] == pytest.approx(0.5)


def test_a_logged_learning_rate_is_not_a_default_metric(tmp_path: Path) -> None:
    for seed, loss in ((1, 0.2), (2, 0.25)):
        d = tmp_path / "runs" / f"seed_{seed}"
        d.mkdir(parents=True)
        (d / "events.out.tfevents.1.host").write_bytes(
            tfrecord(event(9, "train_loss", loss), event(9, "train_learning_rate", 1.7e-7), event(9, "train/epoch", 3))
        )
    t = build(read(tmp_path / "runs"))
    assert t.metrics == ["train_loss"]


HARNESS = {
    "results": {
        "arc": {"alias": "arc", "acc,none": 0.53, "acc_stderr,none": 0.01, "acc_norm,none": 0.57, "acc_norm_stderr,none": "N/A"},
        "gsm8k": {"exact_match,strict-match": 0.14, "exact_match,flexible-extract": 0.15, "exact_match_stderr,strict-match": 0.009},
    },
    "versions": {"arc": 1.0},
    "n-shot": {"arc": 25},
    "configs": {"arc": {"num_fewshot": 25, "metadata": {"version": 1.0}}},
    "config": {"model": "hf", "model_args": "pretrained=org/model-7b,dtype=float16", "random_seed": 0, "batch_size": 8},
    "date": 1739400000.0,
}


def test_lm_evaluation_harness_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = tmp_path / "a.json"
    a.write_text(json.dumps(HARNESS), encoding="utf-8")
    other = json.loads(json.dumps(HARNESS))
    other["config"]["model_args"] = "pretrained=org/model-13b"
    other["results"]["arc"]["acc,none"] = 0.58
    b = tmp_path / "b.json"
    b.write_text(json.dumps(other), encoding="utf-8")
    runs = read(a)
    assert [(r.fields, r.metrics) for r in runs] == [
        ({"model": "org/model-7b", "task": "arc", "seed": "0"}, {"acc": 0.53, "acc_norm": 0.57}),
        ({"model": "org/model-7b", "task": "gsm8k", "seed": "0"}, {"exact_match (strict-match)": 0.14, "exact_match (flexible-extract)": 0.15}),
    ]
    assert main([str(a), str(b), "--format", "markdown", "--filter", "task=arc,mmlu", "--metrics", "acc"]) == 0
    out = capsys.readouterr()
    assert out.out.splitlines()[0] == "| model | acc ↑ |"
    assert "| org/model-13b | **0.58** |" in out.out and "untested" in out.out and "note" not in out.err


def test_columns_none(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "r.csv"
    p.write_text("method,dataset,seed,acc\nA,a,1,90\nA,a,2,91\nB,a,1,80\nB,a,2,81\nA,b,1,70\nB,b,1,60\n", encoding="utf-8")
    assert main([str(p), "--columns", "none", "--format", "markdown"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "| method | acc ↑ |"
