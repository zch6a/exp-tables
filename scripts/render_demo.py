"""docs/demo.png: the example's real table (exp-tables' own HTML) and three lines of its report, rendered with
headless Chrome. Nothing on the screen is captured; nothing is sent anywhere.

    uv run python scripts/render_demo.py [path-to-chrome]
"""

import html
import subprocess
import sys
import tempfile
from pathlib import Path

from exp_tables.cli import load_runs, read_config
from exp_tables.render import html as table_html
from exp_tables.report import comparisons
from exp_tables.table import Spec, build

ROOT = Path(__file__).resolve().parent.parent
CHROME = sys.argv[1] if len(sys.argv) > 1 else r"C:\Program Files\Google\Chrome\Application\chrome.exe"

config = read_config(ROOT / "examples" / "table.json")
table = build(load_runs(config["inputs"], config["filters"]), Spec(**config["spec"]))
report = [line for line in comparisons(table).splitlines() if "CIFAR-100 · Accuracy" in line][:2]
report += [line for line in comparisons(table).splitlines() if "CIFAR-100 · ECE" in line][:1]
lines = [
    html.escape(line)
    .replace("within the noise", '<span class="noise">within the noise</span>')
    .replace(": significant", ': <span class="ok">significant</span>')
    for line in report
]
page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#f6f7f9;font:15px/1.5 'Segoe UI',system-ui,sans-serif;color:#1f2328}}
.card{{margin:24px;background:#fff;border:1px solid #d0d7de;border-radius:12px;padding:22px 26px;box-shadow:0 10px 30px rgba(0,0,0,.08)}}
h1{{font-size:15px;margin:0 0 14px;color:#57606a;font-weight:600}} code{{font:13px Consolas,monospace;color:#0550ae}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%}}
th,td{{padding:6px 12px;text-align:right;border-bottom:1px solid #eaeef2}} th{{font-weight:600}}
tbody th,thead tr:last-child th:first-child{{text-align:left;font-weight:500}} thead tr:first-child th{{border-bottom:1px solid #8c959f}}
small{{color:#6e7781}} caption{{font-size:12.5px;color:#57606a;text-align:left;padding-top:10px}}
pre{{margin:18px 0 0;padding:12px 14px;background:#0d1117;color:#c9d1d9;border-radius:8px;font:12.5px/1.6 Consolas,monospace;white-space:pre-wrap}}
.ok{{color:#7ee787}} .noise{{color:#e3b341}}
</style></head><body><div class="card">
<h1><code>$ exp-tables examples/results.csv --format html --report</code></h1>
{table_html(table)}
<pre>{"<br>".join(lines)}</pre>
</div></body></html>"""
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "demo.html"
    path.write_text(page, encoding="utf-8")
    target = ROOT / "docs" / "demo.png"
    target.parent.mkdir(exist_ok=True)
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-proxy-server", "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
         "--hide-scrollbars", "--window-size=1180,535", f"--screenshot={target}", path.as_uri()],
        check=True, capture_output=True,
    )  # fmt: skip
print(f"wrote {target}")
