"""TensorBoard event files (written here, byte by byte) and the Markdown and HTML tables."""

from __future__ import annotations

import struct
from pathlib import Path

from test_basics import CSV

from exp_tables.data import read, read_csv
from exp_tables.render import html, markdown
from exp_tables.table import Spec, build
from exp_tables.tensorboard import masked_crc, scalars

# ---------------------------------------------------------------- a minimal event writer


def varint(n: int) -> bytes:
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def field(number: int, wire: int, payload: bytes) -> bytes:
    key = varint(number << 3 | wire)
    return key + (varint(len(payload)) + payload if wire == 2 else payload)


def event(step: int, tag: str, value: float, *, tensor: bool = False, wall: float = 1.7e9) -> bytes:
    if tensor:  # TF2: a scalar tensor (dtype float, the four bytes in tensor_content)
        proto = field(1, 0, varint(1)) + field(2, 2, b"") + field(4, 2, struct.pack("<f", value))
        v = field(1, 2, tag.encode()) + field(8, 2, proto)
    else:  # simple_value
        v = field(1, 2, tag.encode()) + field(2, 5, struct.pack("<f", value))
    return field(1, 1, struct.pack("<d", wall + step)) + field(2, 0, varint(step)) + field(5, 2, field(1, 2, v))


def tfrecord(*bodies: bytes) -> bytes:
    out = b""
    for body in bodies:
        header = struct.pack("<Q", len(body))
        out += header + struct.pack("<I", masked_crc(header)) + body + struct.pack("<I", masked_crc(body))
    return out


def write_run(folder: Path, accs: list[float], loss: float, *, tensor: bool = False, cut: bool = False) -> None:
    folder.mkdir(parents=True)
    data = tfrecord(
        field(3, 2, b"brain.Event:2"), *(event(k, "eval/accuracy", a, tensor=tensor) for k, a in enumerate(accs)), event(9, "train/loss", loss)
    )
    if cut:
        data += tfrecord(event(10, "eval/accuracy", 0.0))[:-7]  # a run still writing: its last record is incomplete
    (folder / "events.out.tfevents.1700000000.host.123.0").write_bytes(data)


def test_scalars_both_ways_and_a_cut_record(tmp_path: Path) -> None:
    write_run(tmp_path / "logs" / "ours" / "seed_0", [0.80, 0.91, 0.930], 0.31)
    write_run(tmp_path / "logs" / "ours" / "seed_1", [0.79, 0.90, 0.934], 0.30, tensor=True)
    write_run(tmp_path / "logs" / "baseline" / "seed_0", [0.70, 0.85, 0.910], 0.36, cut=True)
    write_run(tmp_path / "logs" / "baseline" / "seed_1", [0.71, 0.86, 0.912], 0.35)
    runs = read(tmp_path / "logs")
    by = {(r.fields["method"], r.fields["seed"]): r for r in runs}
    assert set(by) == {("ours", "0"), ("ours", "1"), ("baseline", "0"), ("baseline", "1")}
    assert round(by["ours", "1"].metrics["eval/accuracy"], 3) == 0.934  # TF2's tensor scalar, the last step
    assert round(by["baseline", "0"].metrics["eval/accuracy"], 3) == 0.910  # the cut record is not read as 0.0
    assert by["ours", "0"].source == "ours/seed_0" and by["ours", "0"].fields["run"] == "ours/seed_0"
    t = build(runs)
    assert t.rows == ["baseline", "ours"] or t.rows == ["ours", "baseline"]
    assert t.directions == ["higher", "lower"]


def test_a_corrupt_record_ends_the_file() -> None:
    good = tfrecord(event(1, "acc", 0.5), event(2, "acc", 0.6))
    bad = bytearray(good)
    bad[-3] ^= 0xFF  # the second record's checksum no longer holds
    assert [round(v, 2) for *_, v in scalars(bytes(bad))] == [0.5]


def test_markdown() -> None:
    md = markdown(build(read_csv(CSV, "results.csv"), Spec(labels={"accuracy": "Acc."})))
    first, rule, ours = md.splitlines()[:3]
    assert first == "| method | cifar10 · Acc. ↑ | cifar10 · loss ↓ | cifar100 · Acc. ↑ | cifar100 · loss ↓ |"
    assert rule == "|---|---:|---:|---:|---:|"
    assert ours.startswith("| ours | **93.10** ± 0.26 |")
    assert "| ablation | <ins>92.17</ins> ± 0.21" in md
    assert "bold: the best, and any not significantly different from it (paired t, α = 0.05, corrected per column)" in md


def test_html_is_escaped() -> None:
    runs = read_csv("method,seed,acc\n<b>x</b>,0,90\n<b>x</b>,1,91\nplain,0,80\nplain,1,81\n")
    page = html(build(runs, Spec(labels={"acc": "acc <script>"})))
    assert "<b>x</b>" not in page.replace("<b>90.5</b>", "") and "&lt;b&gt;x&lt;/b&gt;" in page
    assert "acc &lt;script&gt; ↑" in page and "<script>" not in page
    assert page.count("<tr>") == 3 and "<small>± 0.71</small>" in page
