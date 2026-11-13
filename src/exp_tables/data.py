"""Reading experiment results: every source becomes a list of runs, each with its text fields (method, dataset,
seed, configuration) and its numeric metrics.

Sources: CSV and TSV (one run per row: wandb's runs export is one), JSON (a list of runs, ``{"runs": [...]}``, or
``{name: run}``), JSON Lines, and MLflow's ``mlruns`` folders. Nested objects are flattened with ``/``
(``{"config": {"lr": 0.1}}`` is the field ``config/lr``).
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# columns that are part of what a run is, even when they hold numbers
SEED = re.compile(r"^(?:seed|random[_ ]?seed|run[_ ]?seed|trial|repeat|rep|fold|replicate|run[_ ]?id|run)$", re.I)
FIELDISH = re.compile(r"(?:^|/)(?:config|params?|hparams?|hyperparameters|args|settings|cfg)(?:/|$)|^(?:id|name|step|epoch|epochs)$", re.I)
# a tracker's bookkeeping, not a result
META = re.compile(r"^(?:_?runtime|_?timestamp|_?step|_wandb.*|created|state|user|notes|tags|sweep|duration|start[_ ]?time|end[_ ]?time)$", re.I)
NUMBER = re.compile(r"^\s*([-+−]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(%?)\s*$")


@dataclass
class Run:
    fields: dict[str, str] = field(default_factory=dict)  # what the run is: method, dataset, seed, configuration
    metrics: dict[str, float] = field(default_factory=dict)  # what it measured
    source: str = ""  # where it was read: file and line, or a run's folder


def number(text: str) -> float | None:
    """A value as a number: "93.1", "93.1%", "1,234", "1e-3"; None for anything else.

    >>> number("93.1%"), number(" 1,234 "), number("1e-3"), number("n/a"), number("")
    (93.1, 1234.0, 0.001, None, None)
    """
    m = NUMBER.match(text)
    return float(m.group(1).replace("−", "-").replace(",", "")) if m else None


def _flat(obj: object, prefix: str = "") -> dict[str, object]:
    if isinstance(obj, dict):
        out: dict[str, object] = {}
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}{k}/"))
        return out
    return {prefix[:-1]: obj}


def _split(values: dict[str, object], source: str) -> Run:
    """A run from its flattened values: numbers are metrics unless their name says they are part of what the run is
    (a seed, a configuration value); text and booleans are fields."""
    run = Run(source=source)
    for k, v in values.items():
        if v is None or META.match(k.rsplit("/", 1)[-1]):
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or SEED.match(k.rsplit("/", 1)[-1]) or FIELDISH.search(k):
            run.fields[k] = json.dumps(v) if isinstance(v, (list, bool)) else str(v)
        else:
            run.metrics[k] = float(v)
    return run


def read_csv(text: str, name: str = "") -> list[Run]:
    """Runs from CSV or TSV (the delimiter is detected). A column is a metric when every value in it is a number
    (or empty) and its name is not a seed's or a configuration's; the others are fields."""
    sample = text[:20_000]
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    if not rows:
        return []
    columns = [c for c in rows[0] if c is not None]
    numeric = {
        c
        for c in columns
        if not SEED.match(c) and not FIELDISH.search(c) and not META.match(c)
        and any((r.get(c) or "").strip() for r in rows)
        and all(number(r.get(c) or "") is not None for r in rows if (r.get(c) or "").strip())
    }  # fmt: skip
    runs = []
    for i, r in enumerate(rows, start=2):
        run = Run(source=f"{name}:{i}")
        for c in columns:
            value = (r.get(c) or "").strip()
            if META.match(c) or not value:
                continue
            if c in numeric:
                run.metrics[c] = number(value)  # type: ignore[assignment]  # every value of the column is one
            else:
                run.fields[c] = value
        runs.append(run)
    return runs


def read_json(text: str, name: str = "") -> list[Run]:
    """Runs from JSON: a list of runs, an object with a list under "runs" (or "results", "experiments"), or an object
    of runs by name (the name becomes the field "run")."""
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("runs", "results", "experiments", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:  # runs by name, or a single run
            data = [{"run": k, **v} for k, v in data.items()] if data and all(isinstance(v, dict) for v in data.values()) else [data]
    if not isinstance(data, list):
        raise ValueError(f"{name}: no runs in it (a list of objects, or an object of them)")
    return [_split(_flat(item), f"{name}:{i}") for i, item in enumerate(data, start=1) if isinstance(item, dict)]


def read_jsonl(text: str, name: str = "") -> list[Run]:
    """Runs from JSON Lines: one object a line (blank lines skipped)."""
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(_split(_flat(item), f"{name}:{i}"))
    return out


def _mlflow_name(meta: Path) -> str:
    try:
        m = re.search(r"^name:\s*(.+)$", meta.read_text(encoding="utf-8"), re.M)
    except OSError:
        return ""
    return m.group(1).strip().strip("'\"") if m else ""


def read_mlflow(folder: Path) -> list[Run]:
    """Runs from an MLflow tracking folder (``mlruns``): each run's parameters and tags become fields (its name
    "run", its experiment's name "experiment"), each metric its last logged value (by step)."""
    runs = []
    for metrics_dir in sorted(folder.glob("*/*/metrics")):
        run_dir = metrics_dir.parent
        run = Run(source=run_dir.relative_to(folder).as_posix())
        experiment = _mlflow_name(run_dir.parent / "meta.yaml")
        if experiment:
            run.fields["experiment"] = experiment
        for p in sorted((run_dir / "params").glob("*")) if (run_dir / "params").is_dir() else []:
            run.fields[p.name] = p.read_text(encoding="utf-8").strip()
        name = run_dir / "tags" / "mlflow.runName"
        if name.is_file():
            run.fields["run"] = name.read_text(encoding="utf-8").strip()
        for m in sorted(p for p in metrics_dir.rglob("*") if p.is_file()):
            last: tuple[float, float] | None = None  # (step, value)
            for line in m.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2 and number(parts[1]) is not None:
                    step = float(parts[2]) if len(parts) >= 3 and number(parts[2]) is not None else float("inf")
                    if last is None or step >= last[0]:
                        last = (step, number(parts[1]))  # type: ignore[assignment]
            if last is not None:
                run.metrics[m.relative_to(metrics_dir).as_posix()] = last[1]
        runs.append(run)
    return runs


def read(path: Path) -> list[Run]:
    """Runs from a file or an MLflow folder, by what it is."""
    if path.is_dir():
        if any(path.glob("*/*/metrics")):
            return read_mlflow(path)
        if (path / "mlruns").is_dir():
            return read_mlflow(path / "mlruns")
        if any(path.rglob("events.out.tfevents.*")):
            from .tensorboard import read_tensorboard

            return read_tensorboard(path)
        raise ValueError(f"{path}: a folder, but neither MLflow's (mlruns/<experiment>/<run>/metrics) nor TensorBoard logs (events.out.tfevents.*)")
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_json(text, path.name)
    if suffix in (".jsonl", ".ndjson"):
        return read_jsonl(text, path.name)
    if suffix in (".csv", ".tsv", ".txt"):
        return read_csv(text, path.name)
    raise ValueError(f"{path}: not a file it reads (.csv, .tsv, .json, .jsonl, or an mlruns folder)")
