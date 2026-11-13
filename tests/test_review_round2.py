"""Regression tests for what the round-2 review found (statistics, marks, rendering, the Python API)."""

from __future__ import annotations

import pytest

import exp_tables
from exp_tables.data import Run, read_csv
from exp_tables.render import legend, markdown
from exp_tables.report import comparisons
from exp_tables.significance import benjamini_hochberg, bootstrap, holm, wilcoxon
from exp_tables.table import Spec, build, latex, tex


def csv(rows: dict[str, list[float]], metric: str = "acc") -> list[Run]:
    lines = [f"method,seed,{metric}"] + [f'"{m}",{s},{v}' for m, vs in rows.items() for s, v in enumerate(vs)]
    return read_csv("\n".join(lines) + "\n", "r.csv")


def test_latex_rows_starting_with_a_bracket_or_a_star() -> None:
    out = latex(build(csv({"ours": [90, 91], "[CLS] pooling": [80, 81], "*ours": [70, 71]})))
    assert "\n{}[CLS] pooling & " in out and "\n{}*ours & " in out


def test_bootstrap_stops_early_only_away_from_every_corrected_level() -> None:
    rows = {
        "m0": [89.7292, 89.387, 90.6246, 89.1462, 91.053],
        "m1": [91.4623, 88.8475, 90.3646, 87.6877, 88.5058],
        "m2": [90.6684, 87.9356, 88.5078, 86.3757, 87.9908],
        "m3": [87.9129, 89.9028, 88.1643, 91.2494, 87.6016],
        "m4": [88.5665, 86.2347, 87.834, 89.4558, 88.1304],
    }
    t = build(csv(rows), Spec(test="bootstrap"))
    m2 = next(c for c in t.comparisons if c.other == "m2")
    assert m2.significant and not t.cells[2][0].best  # m2 is significantly worse, as 10,000 resamples say (corrected p 0.045)
    a, b = rows["m0"], rows["m2"]
    assert bootstrap(a, b, levels=[0.05 / 4, 0.05 / 3, 0.05 / 2, 0.05]).p == pytest.approx(bootstrap(a, b).p, abs=0.004)


def test_a_cell_that_cannot_be_tested_is_not_marked_and_a_note_says_so() -> None:
    rows = {"ours": [90.0, 90.4, 90.2], "tied": [89.0, 90.4, 89.6], "one": [90.2], "one2": [90.1], "worse": [85.0, 85.1, 85.2]}
    t = build(csv(rows))
    marks = {name: c.best for name, (c,) in zip(t.rows, t.cells, strict=True)}
    assert marks == {"ours": True, "tied": True, "one": False, "one2": False, "worse": False}
    assert any("2 cell(s) could not be tested" in n for n in t.notes)
    # the highest mean has one seed: the best of the rows with seeds is tested against the others, the lucky one unmarked
    t = build(csv({"ours": [81.0], "abl": [80.85, 80.95, 80.9], "base": [80.0, 80.1, 80.2]}))
    assert [row[0].best for row in t.cells] == [False, True, False] and [c.other for c in t.comparisons] == ["base"]


def test_seeds_needed_is_more_than_there_are_and_follows_the_correction() -> None:
    ours = [90.0, 90.5, 91.0, 90.2, 90.8]
    rows = {"ours": ours} | {f"b{i}": [v - 0.9 + i * 1e-4 for v in ours] for i in range(11)}
    t = build(csv(rows), Spec(test="welch"))
    within = [c for c in t.comparisons if not c.significant]
    assert within and all(c.seeds_needed is None or c.seeds_needed > 5 for c in within)
    t = build(csv({"ours": [90.0, 90.6, 89.8, 90.3], "base": [89.0, 89.3, 89.0, 89.2]}), Spec(test="wilcoxon"))
    assert t.comparisons[0].seeds_needed is not None and t.comparisons[0].seeds_needed >= 6  # 2 / 2^n < 0.05 needs 6 pairs
    assert "about 3 seeds" not in comparisons(t)


def test_the_python_api() -> None:
    runs = exp_tables.read_csv("method,seed,acc\nours,1,90.1\nours,2,90.3\nbase,1,88.0\nbase,2,88.4\n")
    t = exp_tables.build(runs)
    assert "\\textbf{90.20}" in exp_tables.latex(t) and "<b>90.20</b>" in exp_tables.html(t)
    for bad in (Spec(spread="CI"), Spec(correction="BH"), Spec(test="Wilcoxon"), Spec(better={"acc": "up"}), Spec(alpha=5), Spec(rows=["methd"])):
        with pytest.raises(ValueError):
            exp_tables.build(runs, bad)


def test_markdown_labels_with_line_breaks_keep_the_row() -> None:
    out = markdown(build(csv({"two\n\nparagraphs": [1, 2], "b": [0, 0.5]})))
    assert "| two paragraphs | " in out


def test_wilcoxon_ties_at_large_values() -> None:
    a = [1000000.3, 2000000.4, 1500000.5, 1234567.9, 7654321.1, 3000000.6]
    b = [x - 0.1 for x in a]
    a[0], b[0] = b[0], a[0]
    assert wilcoxon(a, b).p == pytest.approx(0.21875)


def test_greek_letters_and_comparisons_in_latex() -> None:
    assert tex("β-VAE Ω α<1|2") == r"\ensuremath{\beta}-VAE \ensuremath{\Omega} \ensuremath{\alpha}\textless{}1\textbar{}2"
    assert tex("ΑΒΡ") == "ABP"


def test_corrections_do_not_round_p_up_to_alpha() -> None:
    assert holm([0.04999999999999])[0] < 0.05 and benjamini_hochberg([0.04999999999999])[0] < 0.05
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_no_second_arrow_when_the_name_has_one() -> None:
    t = build(csv({"a": [1, 2], "b": [3, 4]}, "reward ↓"))
    assert "reward ↓ |" in markdown(t) and "↓ ↓" not in markdown(t) and "reward ↓ \\\\" in latex(t)
    assert legend(t)
