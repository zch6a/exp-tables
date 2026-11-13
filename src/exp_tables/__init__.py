"""Experiment results to paper tables.

>>> import exp_tables
>>> runs = exp_tables.read_csv("method,seed,acc\\nours,1,90.1\\nours,2,90.3\\nbase,1,88.0\\nbase,2,88.4\\n")
>>> table = exp_tables.build(runs, exp_tables.Spec(labels={"acc": "Acc. (%)"}))
>>> print(exp_tables.markdown(table).splitlines()[2])
| ours | **90.20** ± 0.14 |
"""

__version__ = "0.1.0"

from .data import Run, read, read_csv, read_json, read_jsonl  # noqa: E402
from .render import html, markdown  # noqa: E402
from .report import comparisons  # noqa: E402
from .table import Spec, Table, build, check, direction, latex, resolve  # noqa: E402

__all__ = [
    "Run",
    "Spec",
    "Table",
    "__version__",
    "build",
    "check",
    "comparisons",
    "direction",
    "html",
    "latex",
    "markdown",
    "read",
    "read_csv",
    "read_json",
    "read_jsonl",
    "resolve",
]
