"""Significance in the table: the best marked only where the seeds support it."""

from __future__ import annotations

import pytest
from test_basics import CSV

from exp_tables.data import read_csv
from exp_tables.significance import benjamini_hochberg, cliffs_delta, holm, wilcoxon
from exp_tables.table import Spec, build, latex


def runs_of(rows: dict[str, list[float]], metric: str = "accuracy") -> str:
    lines = [f"method,seed,{metric}"]
    for method, values in rows.items():
        lines += [f"{method},{k},{v}" for k, v in enumerate(values)]
    return "\n".join(lines) + "\n"


def test_within_noise_both_are_marked() -> None:
    """Two methods whose seeds overlap: the second is not significantly worse, so it is marked best too."""
    runs = read_csv(
        runs_of({"ours": [93.1, 92.6, 93.4, 92.8, 93.0], "baseline": [92.9, 93.2, 92.5, 93.0, 92.7], "weak": [88.0, 88.4, 87.9, 88.3, 88.1]})
    )
    t = build(runs)
    best, baseline, weak = (row[0] for row in t.cells)
    assert best.best and not best.tied
    assert baseline.best and baseline.tied  # within the noise of the seeds
    assert not weak.best and not weak.second  # clearly worse; and no second best: the two bold ones are not it
    by_other = {c.other: c for c in t.comparisons}
    assert by_other["baseline"].test == "paired t" and not by_other["baseline"].significant
    assert by_other["weak"].significant and by_other["weak"].hedges_g > 5 and by_other["weak"].cliffs_delta == 1.0


def test_without_tests_only_the_best_as_printed() -> None:
    runs = read_csv(
        runs_of({"ours": [93.1, 92.6, 93.4, 92.8, 93.0], "baseline": [92.9, 93.2, 92.5, 93.0, 92.7], "weak": [88.0, 88.4, 87.9, 88.3, 88.1]})
    )
    t = build(runs, Spec(test="none"))
    assert [row[0].best for row in t.cells] == [True, False, False] and [row[0].second for row in t.cells] == [False, True, False]
    assert not t.comparisons


def test_seeds_that_do_not_pair_up_use_welch() -> None:
    text = "method,seed,accuracy\nours,0,90.0\nours,1,90.4\nours,2,90.2\nbase,7,85.0\nbase,8,85.3\nbase,9,85.1\n"
    t = build(read_csv(text))
    (c,) = t.comparisons
    assert c.test == "Welch's t" and c.significant and c.seeds == (3, 3)


def test_two_seeds_are_rarely_enough(tmp_path: object) -> None:
    """The example's cifar100 column has two seeds a method: a difference of 3.5 points is still not significant."""
    t = build(read_csv(CSV, "results.csv"))
    acc100 = [row[2] for row in t.cells]
    assert acc100[0].best and acc100[1].best and acc100[1].tied
    c = next(c for c in t.comparisons if c.group == "cifar100" and c.metric == "accuracy")
    assert c.p > 0.05 and c.seeds_needed is not None and c.seeds_needed <= 3  # but a third seed would likely do


@pytest.mark.parametrize("how", ["welch", "paired", "wilcoxon", "bootstrap"])
def test_every_test_can_be_asked_for(how: str) -> None:
    runs = read_csv(runs_of({"ours": [93.1, 92.6, 93.4, 92.8, 93.0, 93.2], "weak": [88.0, 88.4, 87.9, 88.3, 88.1, 88.2]}))
    (c,) = build(runs, Spec(test=how)).comparisons
    expected = {"welch": "Welch's t", "paired": "paired t", "wilcoxon": "Wilcoxon", "bootstrap": "bootstrap"}[how]
    assert c.test == expected and c.p < 0.05


def test_lower_is_better_and_the_latex_marks() -> None:
    runs = read_csv(runs_of({"ours": [0.30, 0.31, 0.29], "close": [0.31, 0.30, 0.32], "far": [0.50, 0.52, 0.51]}, metric="loss"))
    tex = latex(build(runs))
    assert "ours & \\textbf{0.300}" in tex and "close & \\textbf{0.310}" in tex and "far & 0.510" in tex


def test_wilcoxon_mixed_signs_and_ties() -> None:
    # differences -1, 2, 3, 4, 5: W- = 1, P(W <= 1) = 2/32, two-sided 0.125
    assert wilcoxon([0, 4, 6, 8, 10], [1, 2, 3, 4, 5]).p == 0.125
    tied = wilcoxon([2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6])  # all differences 1: the normal approximation, ties corrected
    assert 0 < tied.p < 0.05


def test_corrections_never_lower_a_p_value() -> None:
    ps = [0.001, 0.02, 0.03, 0.2, 0.5]
    for adjusted in (holm(ps), benjamini_hochberg(ps)):
        assert all(a >= p for a, p in zip(adjusted, ps, strict=True)) and all(a <= 1 for a in adjusted)
    assert cliffs_delta([1, 2], [1, 2]) == 0.0
