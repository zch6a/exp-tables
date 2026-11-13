"""Shapes found in real results files (polishing round 1, on public multi-seed CSVs), rebuilt as synthetic data."""

from __future__ import annotations

from exp_tables.data import read_csv
from exp_tables.render import legend, markdown
from exp_tables.table import Spec, build, direction, resolve


def test_a_second_name_of_the_method_does_not_group_columns() -> None:
    # model_key and model name the same thing; the budget is what varies within a model
    rows = ["model_key,model,budget,seed,f1"]
    for key, name, base in (("r50", "ResNet-50", 0.80), ("vit", "ViT-S", 0.70)):
        for budget, add in (("10pct", 0.0), ("100pct", 0.1)):
            rows += [f"{key},{name},{budget},{s},{base + add + s / 1000}" for s in range(3)]
    runs = read_csv("\n".join(rows) + "\n", "r.csv")
    spec = resolve(runs, Spec())
    assert spec.rows == ["model"] and spec.columns == ["budget"]
    t = build(runs)
    assert t.groups == ["10pct", "100pct"] and all(c.summary is not None for row in t.cells for c in row)  # no diagonal


def test_one_row_is_not_marked_best() -> None:
    runs = read_csv("seed,AUPRC,f1\n1,0.70,0.60\n2,0.65,0.62\n3,0.72,0.61\n", "r.csv")
    t = build(runs)
    assert not any(c.best or c.second for row in t.cells for c in row)
    assert legend(t) == "mean ± standard deviation over seeds."
    assert "**" not in markdown(t)


def test_runs_sharing_a_seed_get_a_note_and_a_numeric_setting_can_be_named() -> None:
    rows = ["seed,ratio,RL,Rule"]
    for s in (1, 2, 3):
        for ratio in (0.1, 0.5, 1.0):
            rows.append(f"{s},{ratio},{0.5 - ratio / 4 + s / 100},{0.2 + s / 1000}")
    runs = read_csv("\n".join(rows) + "\n", "r.csv")
    t = build(runs)
    assert len(t.notes) == 1 and "seed 1 3 times" in t.notes[0] and "maybe ratio" in t.notes[0]
    t = build(runs, Spec(columns=["ratio"], metrics=["RL", "Rule"]))
    assert not t.notes
    assert t.groups == ["0.1", "0.1", "0.5", "0.5", "1", "1"] and len(t.rows) == 1
    assert [c.summary.n for c in t.cells[0] if c.summary] == [3] * 6


def test_directions_of_common_metric_names() -> None:
    higher = ["AUPRC", "auroc", "Macro-F1", "specificity", "sensitivity", "mcc", "cohen_kappa", "dice", "test_top1", "pr_auc", "recall@10"]
    lower = ["test_ece", "MAPE", "fpr", "train_time", "num_params", "hausdorff_distance"]
    assert [direction(m) for m in higher] == ["higher"] * len(higher)
    assert [direction(m) for m in lower] == ["lower"] * len(lower)
    assert [direction(m) for m in ("threshold", "ratio", "lift", "best_epoch")] == [""] * 4


def test_a_wandb_export_with_plain_hyperparameter_columns() -> None:
    head = '"Name","State","Runtime","dataset","lr","batch_size","model","seed","_step","val/acc","val/loss"'
    lines = [head] + [
        f'"{m}-{s}","{"crashed" if s == 2 else "finished"}","3600","cifar10","0.1","128","{m}","{s}","999","{acc}","{loss}"'
        for m, vals in (("ours", [(93.1, 0.30), (93.4, 0.29), (92.8, 0.31)]), ("baseline", [(91.2, 0.35), (91.0, 0.36), (91.5, 0.34)]))
        for s, (acc, loss) in enumerate(vals)
    ]
    runs = read_csv("\n".join(lines) + "\n", "wandb.csv")
    assert runs[0].fields["lr"] == "0.1" and "batch_size" not in runs[0].metrics  # settings, not results
    t = build(runs)
    assert t.rows == ["ours", "baseline"] and t.metrics == ["val/acc", "val/loss"] and t.groups == ["", ""]
    assert t.decimals == [2, 3]  # a spread of 0.01, computed as 0.00999…, is still two significant digits at 3 decimals


def test_untested_marks_say_so() -> None:
    runs = read_csv("method,acc\na,90\nb,80\n", "r.csv")  # one seed each: the best as printed, untested
    t = build(runs)
    assert t.cells[0][0].best and not t.comparisons
    assert legend(t) == "one run per cell: no spread; bold: the best (untested: that needs two or more seeds in each row, or standard errors)."
