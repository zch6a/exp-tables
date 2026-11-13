"""exp-tables: experiment results to a paper's table.

exp-tables results.csv                         a table worked out from the runs, as LaTeX
exp-tables runs/ --format markdown             TensorBoard logs or an mlruns folder
exp-tables a.json b.json --rows file           several files (the field "file" tells them apart)
exp-tables results.csv --report                and what each mark rests on
exp-tables results.csv --save-config table.json   then: exp-tables --config table.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__, render
from .data import Run, number, read
from .report import comparisons
from .table import TESTS, Spec, Table, as_fields, build, latex, resolve
from .table import check as check_spec

SPEC_KEYS = ("rows", "columns", "metrics", "seed", "better", "spread", "decimals", "labels", "test", "alpha", "correction")


class UsageError(ValueError):
    """What was asked cannot be done as asked (exit status 2)."""


def _pairs(items: list[str], what: str) -> dict[str, str]:
    """key=value options into a dict ("loss=lower", 'accuracy=Acc. (%)')."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise UsageError(f"--{what} takes key=value, not {item!r}")
        out[key.strip()] = value.strip()
    return out


def _list(text: str | None) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()] if text else []


def _matches(value: str | None, wanted: str) -> bool:
    """A field's value against a filter's: as text, or as numbers ("0.10" is 0.1); "a,b" is either."""
    if value == wanted:
        return True
    for one in wanted.split(","):
        one = one.strip()
        a, b = number(value or ""), number(one)
        if value == one or (a is not None and a == b):
            return True
    return False


def load_runs(inputs: list[Path], filters: dict[str, str], settings: list[str] | None = None) -> list[Run]:
    """The runs of every input (each run told apart by the field "file" when there are several inputs), with the
    named settings (and the filtered fields) made fields when they were read as metrics, keeping those whose fields
    match every filter."""
    runs: list[Run] = []
    for path in inputs:
        if not path.exists():
            raise UsageError(f"{path}: no such file or folder")
        for run in read(path):
            if len(inputs) > 1:
                run.fields.setdefault("file", path.stem if path.is_file() else path.name)
            runs.append(run)
    runs = as_fields(runs, [*(settings or []), *filters])  # a numeric setting, read as a metric, can be filtered on
    kept = [r for r in runs if all(_matches(r.fields.get(k), v) for k, v in filters.items())]
    if runs and not kept:
        raise UsageError(f"no run matches the filters {filters}")
    if not kept:
        raise UsageError("no runs in the inputs")
    return kept


def check(runs: list[Run], spec: Spec) -> None:
    """The spec against the runs (table.check), its complaints as usage errors."""
    try:
        check_spec(runs, spec)
    except ValueError as e:
        raise UsageError(str(e)) from e


def to_json(t: Table) -> dict[str, Any]:
    """The table as data: each column (group, metric, direction, decimals), each cell's values and summary and marks,
    and every comparison."""
    columns = [
        {"group": g, "metric": m, "direction": d, "decimals": k} for g, m, d, k in zip(t.groups, t.metrics, t.directions, t.decimals, strict=True)
    ]
    rows = []
    for name, row in zip(t.rows, t.cells, strict=True):
        cells: list[dict[str, Any] | None] = []
        for c in row:
            if c.summary is None:
                cells.append(None)
                continue
            s = c.summary
            cells.append({"values": s.values, "seeds": c.seeds, "n": s.n, "mean": s.mean, "std": s.std, "sem": s.sem, "ci95": s.ci(),
                          "error": c.error, "best": c.best, "second": c.second, "tied": c.tied})  # fmt: skip
        rows.append({"row": name, "cells": cells})
    return {
        "tool": "exp-tables",
        "tool_version": __version__,
        "row_title": t.row_title,
        "spread": t.spread,
        "test": t.test,
        "columns": columns,
        "rows": rows,
        "comparisons": [asdict(c) | {"significant": c.significant} for c in t.comparisons],
        "notes": t.notes,
    }


def _finite(data: Any) -> Any:
    """The data with infinities and NaN as null: strict JSON has neither (an effect size is infinite when neither
    side has any spread)."""
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    if isinstance(data, dict):
        return {k: _finite(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_finite(v) for v in data]
    return data


def save_config(path: Path, inputs: list[Path], filters: dict[str, str], spec: Spec, fmt: str) -> None:
    """The table as a file to make it again: the inputs (relative to the file), the filters, the spec as worked out,
    the format and the version that made it."""
    base = path.resolve().parent

    def rel(p: Path) -> str:
        try:  # relative, with "..", so the config and its data can move together
            return Path(os.path.relpath(p.resolve(), base)).as_posix()
        except ValueError:  # another drive (Windows)
            return p.resolve().as_posix()

    data = {
        "tool": "exp-tables",
        "tool_version": __version__,
        "inputs": [rel(p) for p in inputs],
        "filters": filters,
        "format": fmt,
        "spec": {k: getattr(spec, k) for k in SPEC_KEYS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise UsageError(f"{path.as_posix()}: not a readable config ({getattr(e, 'strerror', None) or e})") from e
    if not isinstance(data, dict) or not isinstance(data.get("spec", {}), dict):
        raise UsageError(f"{path.as_posix()}: not an exp-tables config")
    unknown = set(data.get("spec", {})) - set(SPEC_KEYS)
    if unknown:
        raise UsageError(f"{path.as_posix()}: unknown spec keys {sorted(unknown)}")
    base = path.resolve().parent
    data["inputs"] = [base / p for p in data.get("inputs", [])]
    return data


def _check(path: Path, text: str) -> int:
    """The table made now against the one in `path`: 0 when they are the same (line endings and trailing spaces aside),
    1 with the lines that differ."""
    try:
        raw = path.read_bytes()
        committed = raw.decode("utf-16") if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else raw.decode("utf-8-sig")
    except OSError as e:
        print(f"exp-tables: {path.as_posix()}: {e.strerror or e}", file=sys.stderr)
        return 2
    except UnicodeDecodeError:
        print(f"exp-tables: {path.as_posix()}: not UTF-8 or UTF-16 text", file=sys.stderr)
        return 2
    version = re.compile(r'^\s*"tool_version": ".*",?$')  # JSON: another version of exp-tables is not a difference in the results
    norm = lambda t: [line.rstrip() for line in t.strip().splitlines() if not version.match(line)]  # noqa: E731
    if norm(committed) == norm(text):
        print(f"exp-tables: {path.as_posix()} matches the results")
        return 0
    diff = difflib.unified_diff(norm(committed), norm(text), f"{path.as_posix()} (in the paper)", "made from the results", lineterm="")
    print(f"exp-tables: {path.as_posix()} does not match the results:", *diff, sep="\n")
    return 1


def _write(output: Path | None, text: str) -> None:
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """The command line. Exit status: 0 done, 2 when it could not make the table (and on any unexpected error)."""
    try:
        return _main(argv)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # noqa: BLE001 - the last line of defence for the exit status
        print(f"exp-tables: {e.__class__.__name__}: {e}", file=sys.stderr)
        return 2


def _main(argv: list[str] | None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="exp-tables", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="exit status: 0 done, 1 when --check finds a difference, 2 when it could not make the table")  # fmt: skip
    ap.add_argument("inputs", nargs="*", type=Path, metavar="INPUT", help="CSV, TSV, JSON or JSONL files, TensorBoard log folders, mlruns folders")
    ap.add_argument("--rows", help="the field(s) naming a row, comma-separated (default: method, model, … or the first that varies)")
    ap.add_argument("--columns", help="the field(s) grouping columns, or none (default: another field that varies within a row)")
    ap.add_argument("--metrics", help="the metrics shown, in order (default: all)")
    ap.add_argument("--seed", help="the field telling seeds apart (default: by its name: seed, trial, fold, …)")
    ap.add_argument("--filter", action="append", default=[], metavar="FIELD=VALUE", help="keep only the runs with this value (repeatable)")
    ap.add_argument("--better", action="append", default=[], metavar="METRIC=higher|lower", help="which way is better (default: from its name)")
    ap.add_argument("--label", action="append", default=[], metavar="NAME=TEXT", help="how the table prints a metric, field or value")
    ap.add_argument("--decimals", action="append", default=[], metavar="METRIC=N", help="decimals (default: two significant digits of the spread)")
    ap.add_argument(
        "--spread", choices=["std", "sem", "ci", "none"], help="after the mean: standard deviation (default), standard error, 95%% CI, nothing"
    )
    ap.add_argument("--test", choices=TESTS,
                    help="significance test against each column's best (default auto: paired t when the seeds pair up, else Welch's)")  # fmt: skip
    ap.add_argument("--alpha", type=float, help="significance level (default 0.05)")
    ap.add_argument("--correction", choices=["holm", "bh", "none"], help="for a column's comparisons together (default holm)")
    ap.add_argument("--format", choices=["latex", "markdown", "html", "json"], help="the output (default latex)")
    ap.add_argument("--report", action="store_true", help="also what each mark rests on: tests, p values, effect sizes, seeds needed")
    ap.add_argument("-o", "--output", type=Path, help="write the table to this file")
    ap.add_argument("--config", type=Path, help="make the table a config file describes (options given here override it)")
    ap.add_argument("--save-config", type=Path, metavar="FILE", help="write a config that makes this table again")
    ap.add_argument(
        "--check", type=Path, metavar="FILE", help="compare with the table in FILE (e.g. the one in the paper): exit 1 and show the lines that differ"
    )
    ap.add_argument("--version", action="version", version=f"exp-tables {__version__}")
    args = ap.parse_args(argv)

    try:
        config = read_config(args.config) if args.config else {}
        inputs = args.inputs or config.get("inputs", [])
        if not inputs:
            ap.error("give the results to read (files or folders), or --config")
        filters = {**config.get("filters", {}), **_pairs(args.filter, "filter")}
        given = dict(config.get("spec", {}))
        if args.rows:
            given["rows"] = _list(args.rows)
        if args.columns is not None:  # "none" (or ""; PowerShell 5 drops an empty argument): no groups
            given["columns"] = [] if args.columns.strip().lower() in ("", "none") else _list(args.columns)
        if args.metrics:
            given["metrics"] = _list(args.metrics)
        for key in ("seed", "spread", "test", "alpha", "correction"):
            if getattr(args, key) is not None:
                given[key] = getattr(args, key)
        for key, opt in (("better", args.better), ("labels", args.label)):
            if opt:
                given[key] = {**given.get(key, {}), **_pairs(opt, key)}
        if args.decimals:
            try:
                given["decimals"] = {**given.get("decimals", {}), **{k: int(v) for k, v in _pairs(args.decimals, "decimals").items()}}
            except ValueError as e:
                raise UsageError("--decimals takes METRIC=N with N a whole number") from e
        fmt = args.format or config.get("format", "latex")
        spec = Spec(**{k: v for k, v in given.items() if k in SPEC_KEYS})
        runs = load_runs([Path(p) for p in inputs], filters, [*spec.rows, *(spec.columns or []), spec.seed])
        check(runs, spec)
        spec = resolve(runs, spec)
        table = build(runs, spec)
    except UsageError as e:  # what the runs or the config lack: the message, not the whole usage
        print(f"exp-tables: error: {e} (exp-tables --help lists the options)", file=sys.stderr)
        return 2
    except (ValueError, OSError) as e:
        print(f"exp-tables: {e}", file=sys.stderr)
        return 2

    for note in table.notes:
        print(f"exp-tables: note: {note}", file=sys.stderr)
    out = {"latex": latex, "markdown": render.markdown, "html": render.html}.get(fmt)
    text = out(table) if out else json.dumps(_finite(to_json(table)), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.check:
        return _check(args.check, text)
    _write(args.output, text)
    if args.report:
        (sys.stdout if args.output else sys.stderr).write(comparisons(table))
    if args.save_config:
        save_config(args.save_config, [Path(p) for p in inputs], filters, spec, fmt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
