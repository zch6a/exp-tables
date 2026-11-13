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
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# columns that are part of what a run is, even when they hold numbers
SEED = re.compile(r"^(?:seed|random[_ ]?seed|run[_ ]?seed|trial|repeat|rep|fold|replicate|run[_ ]?id|run)$", re.I)
FIELDISH = re.compile(
    r"(?:^|/)(?:config|params?|hparams?|hyperparameters|args|settings|cfg)(?:/|$)|^(?:id|name|step|epoch|epochs|num[_ ]epochs|"
    r"lr|learning[_ ]rate|batch[_ ]size|bs|weight[_ ]decay|wd|momentum|dropout|warmup(?:[_ ]steps)?|max[_ ]steps|grad(?:ient)?[_ ]clip|"
    r"(?:num|n)[_ ]layers|hidden[_ ](?:size|dim)|d[_ ]model)$",
    re.I,
)
# a tracker's bookkeeping, not a result
META = re.compile(r"^(?:_?runtime|_?timestamp|_?step|_wandb.*|created|state|user|notes|tags|sweep|duration|start[_ ]?time|end[_ ]?time)$", re.I)
# what a results file writes for a missing value; read as missing, so one failed run does not turn a metric into text
MISSING = re.compile(r"^(?:nan|na|n/a|none|null|missing|-|--|—|[-+]?inf(?:inity)?|#n/a|#div/0!)$", re.I)
INDEX = re.compile(r"^(?:|unnamed: ?\d+|index)$", re.I)  # a data frame's row index, written by to_csv()
NUMBER = re.compile(r"^\s*([-+−]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(%?)\s*$")


@dataclass
class Run:
    fields: dict[str, str] = field(default_factory=dict)  # what the run is: method, dataset, seed, configuration
    metrics: dict[str, float] = field(default_factory=dict)  # what it measured
    source: str = ""  # where it was read: file and line, or a run's folder
    errors: dict[str, float] = field(default_factory=dict)  # a metric's standard error, when the run reports one (acc_stderr)


def number(text: str) -> float | None:
    """A value as a number: "93.1", "93.1%", "1,234", "1e-3"; None for anything else.

    >>> number("93.1%"), number(" 1,234 "), number("1e-3"), number("n/a"), number(""), number("1e999")
    (93.1, 1234.0, 0.001, None, None, None)
    """
    m = NUMBER.match(text)
    value = float(m.group(1).replace("−", "-").replace(",", "")) if m else None
    return value if value is not None and math.isfinite(value) else None


def _missing(text: str) -> bool:
    return not text or bool(MISSING.match(text))


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
        if v is None or META.match(k.rsplit("/", 1)[-1]) or (isinstance(v, float) and not math.isfinite(v)):
            continue  # NaN and infinities (JSON allows them): missing
        if isinstance(v, bool) or not isinstance(v, (int, float)) or SEED.match(k.rsplit("/", 1)[-1]) or FIELDISH.search(k):
            run.fields[k] = json.dumps(v) if isinstance(v, (list, bool)) else str(v)
        else:
            run.metrics[k] = float(v)
    return run


def read_csv(text: str, name: str = "") -> list[Run]:
    """Runs from CSV or TSV (the delimiter is detected). A column is a metric when every value in it is a number
    (or missing: empty, "nan", "n/a", "-", …) and its name is not a seed's or a configuration's; the others are fields.
    A data frame's unnamed index column is left out."""
    sample = text[:20_000]
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    if not rows:
        return []
    columns = [c for c in rows[0] if c is not None and not INDEX.match(c.strip())]
    present = {c: [v for r in rows if not _missing(v := (r.get(c) or "").strip())] for c in columns}
    numeric = {
        c
        for c in columns
        if not SEED.match(c) and not FIELDISH.search(c) and not META.match(c)
        and present[c] and all(number(v) is not None for v in present[c])
    }  # fmt: skip
    runs = []
    for i, r in enumerate(rows, start=2):
        run = Run(source=f"{name}:{i}")
        for c in columns:
            value = (r.get(c) or "").strip()
            if META.match(c) or _missing(value):
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
    if _is_lm_eval(data):
        return read_lm_eval(data, name)
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


def _is_lm_eval(data: object) -> bool:
    """Whether a JSON document is lm-evaluation-harness's results file: "results" by task, and its bookkeeping."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("results"), dict)
        and bool(data["results"])
        and all(isinstance(v, dict) for v in data["results"].values())
        and any(k in data for k in ("versions", "n-shot", "configs", "config"))
    )


def read_lm_eval(data: dict[str, object], name: str = "") -> list[Run]:
    """Runs from an lm-evaluation-harness results file: one run per task, with the fields "model" and "task" (and
    "seed" when the harness recorded one); each metric under its name ("acc", "exact_match"; with its filter when it
    is not "none": "exact_match (strict-match)"); its standard error (``acc_stderr``) as the metric's error.

    >>> runs = read_lm_eval({"results": {"arc": {"acc,none": 0.5, "acc_stderr,none": 0.01, "alias": "arc"}},
    ...                      "versions": {}, "model_name": "m"})
    >>> runs[0].fields, runs[0].metrics, runs[0].errors
    ({'model': 'm', 'task': 'arc'}, {'acc': 0.5}, {'acc': 0.01})
    """
    config = data.get("config") if isinstance(data.get("config"), dict) else {}
    assert isinstance(config, dict)
    model = str(data.get("model_name") or "")
    if not model:
        args = str(config.get("model_args") or "")
        found = re.search(r"(?:pretrained|model|engine)=([^,]+)", args)
        model = found.group(1) if found else str(config.get("model") or Path(name).stem)
    seed = config.get("random_seed")
    runs = []
    results = data["results"]
    assert isinstance(results, dict)
    for task, values in results.items():
        run = Run({"model": model, "task": str(task)}, source=f"{name}:{task}")
        if isinstance(seed, int) and not isinstance(seed, bool):
            run.fields["seed"] = str(seed)
        for key, value in values.items():
            metric, _, how = str(key).partition(",")
            if metric == "alias" or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            error = metric.endswith("_stderr")
            metric = metric.removesuffix("_stderr")
            label = metric if how in ("", "none") else f"{metric} ({how})"
            if not error:
                run.metrics[label] = float(value)
            elif value >= 0:
                run.errors[label] = float(value)
        run.errors = {k: v for k, v in run.errors.items() if k in run.metrics}
        if run.metrics:
            runs.append(run)
    return runs


def read_jsonl(text: str, name: str = "") -> list[Run]:
    """Runs from JSON Lines: one object a line (blank lines skipped)."""
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(_split(_flat(item), f"{name}:{i}"))
    return out


def _mlflow_meta(meta: Path, key: str) -> str:
    try:
        m = re.search(rf"^{key}:\s*(.+)$", meta.read_text(encoding="utf-8"), re.M)
    except OSError:
        return ""
    return m.group(1).strip().strip("'\"") if m else ""


def read_mlflow(folder: Path) -> list[Run]:
    """Runs from an MLflow tracking folder (``mlruns``): each run's parameters and tags become fields (its name
    "run", its experiment's name "experiment"; MLflow's own ``mlflow.*`` tags left out), each metric its last logged
    value (by step). Deleted runs and experiments are left out."""
    runs = []
    for metrics_dir in sorted(folder.glob("*/*/metrics")):
        run_dir = metrics_dir.parent
        if "deleted" in (_mlflow_meta(run_dir / "meta.yaml", "lifecycle_stage"), _mlflow_meta(run_dir.parent / "meta.yaml", "lifecycle_stage")):
            continue
        run = Run(source=run_dir.relative_to(folder).as_posix())
        experiment = _mlflow_meta(run_dir.parent / "meta.yaml", "name")
        if experiment:
            run.fields["experiment"] = experiment
        for p in sorted((run_dir / "params").glob("*")) if (run_dir / "params").is_dir() else []:
            run.fields[p.name] = p.read_text(encoding="utf-8").strip()
        for p in sorted((run_dir / "tags").glob("*")) if (run_dir / "tags").is_dir() else []:
            if p.is_file() and not p.name.startswith("mlflow."):
                run.fields.setdefault(p.name, p.read_text(encoding="utf-8").strip())
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


def read(path: str | Path) -> list[Run]:
    """Runs from a file or a folder (MLflow's, or TensorBoard logs), by what it is."""
    path = Path(path)
    if path.is_dir():
        if any(path.glob("*/*/metrics")):
            return read_mlflow(path)
        if (path / "mlruns").is_dir():
            return read_mlflow(path / "mlruns")
        if any(path.rglob("events.out.tfevents.*")):
            from .tensorboard import read_tensorboard

            return read_tensorboard(path)
        raise ValueError(
            f"{path.as_posix()}: a folder, but neither MLflow's (mlruns/<experiment>/<run>/metrics) nor TensorBoard logs (events.out.tfevents.*)"
        )
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_json(text, path.name)
    if suffix in (".jsonl", ".ndjson"):
        return read_jsonl(text, path.name)
    if suffix in (".csv", ".tsv", ".txt"):
        return read_csv(text, path.name)
    raise ValueError(f"{path.as_posix()}: not a file it reads (.csv, .tsv, .json, .jsonl, or an mlruns folder)")
