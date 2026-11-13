"""TensorBoard event files, read without TensorFlow: the TFRecord framing (each record's length and data checked
by CRC32C), and the few protobuf messages a scalar needs (Event, Summary, Summary.Value, TensorProto). Scalars
written as ``simple_value`` (PyTorch's SummaryWriter, TF1) and as tensors (TF2's ``tf.summary.scalar``) are read;
images, histograms and the rest are skipped. A last record cut short (a run still writing) ends the file quietly.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Iterator
from pathlib import Path

from .data import Run

# ---------------------------------------------------------------- CRC32C (Castagnoli), as TFRecord uses it


def _crc_table() -> list[int]:
    table = []
    for n in range(256):
        c = n
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        table.append(c)
    return table


CRC_TABLE = _crc_table()


def crc32c(data: bytes) -> int:
    """CRC-32C of `data`.

    >>> hex(crc32c(b"123456789"))  # the standard check value
    '0xe3069283'
    """
    c = 0xFFFFFFFF
    for byte in data:
        c = CRC_TABLE[(c ^ byte) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def masked_crc(data: bytes) -> int:
    c = crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _frames(data: bytes) -> Iterator[tuple[bytes, int]]:
    """Each record's data and its checksum, the length checked (a bad length, or a record cut short, ends the file)."""
    i = 0
    while i + 12 <= len(data):
        header = data[i : i + 8]
        (length,) = struct.unpack("<Q", header)
        (length_crc,) = struct.unpack("<I", data[i + 8 : i + 12])
        if masked_crc(header) != length_crc or i + 12 + length + 4 > len(data):
            return
        (body_crc,) = struct.unpack("<I", data[i + 12 + length : i + 16 + length])
        yield data[i + 12 : i + 12 + length], body_crc
        i += 16 + length


def records(data: bytes) -> Iterator[bytes]:
    """The records of a TFRecord file; a record whose checksums do not hold, or that is cut short, ends it."""
    for body, crc in _frames(data):
        if masked_crc(body) != crc:
            return
        yield body


# ---------------------------------------------------------------- protobuf, as far as scalars go


def _varint(data: bytes, i: int) -> tuple[int, int]:
    if i < len(data) and data[i] < 0x80:  # one byte: most keys and lengths
        return data[i], i + 1
    shift = value = 0
    while True:
        if i >= len(data):
            raise ValueError("truncated varint")
        b = data[i]
        i += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """A protobuf message's fields as (number, wire type, value): an int for varints and fixed-width numbers (their
    bits), bytes for length-delimited ones."""
    i = 0
    while i < len(data):
        key, i = _varint(data, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(data, i)
            yield number, wire, value
        elif wire == 1:
            yield number, wire, struct.unpack("<Q", data[i : i + 8])[0]
            i += 8
        elif wire == 2:
            length, i = _varint(data, i)
            yield number, wire, data[i : i + length]
            i += length
        elif wire == 5:
            yield number, wire, struct.unpack("<I", data[i : i + 4])[0]
            i += 4
        else:
            raise ValueError(f"wire type {wire} not read")


def _signed(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_HALF = 1, 2, 3, 9, 19


def _tensor_scalar(data: bytes) -> float | None:
    """The single number of a TensorProto (a scalar summary in TF2), else None."""
    dtype, content, values = 0, b"", []
    for number, wire, value in fields(data):
        if number == 1 and wire == 0:
            dtype = int(value)  # type: ignore[arg-type]
        elif number == 4 and wire == 2:
            content = value  # type: ignore[assignment]
        elif number in (5, 6, 7, 10) and wire == 2:  # packed float, double, int32, int64 values
            raw: bytes = value  # type: ignore[assignment]
            if number == 5:
                values += list(struct.unpack(f"<{len(raw) // 4}f", raw))
            elif number == 6:
                values += list(struct.unpack(f"<{len(raw) // 8}d", raw))
            else:
                j = 0
                while j < len(raw):
                    v, j = _varint(raw, j)
                    values.append(_signed(v))
        elif number == 5 and wire == 5:
            values.append(struct.unpack("<f", struct.pack("<I", value))[0])
        elif number == 6 and wire == 1:
            values.append(struct.unpack("<d", struct.pack("<Q", value))[0])
    if content:
        fmt = {DT_FLOAT: "<f", DT_DOUBLE: "<d", DT_INT32: "<i", DT_INT64: "<q", DT_HALF: "<e"}.get(dtype)
        if fmt and len(content) == struct.calcsize(fmt):
            return float(struct.unpack(fmt, content)[0])
        return None
    return float(values[0]) if len(values) == 1 else None


def scalars(data: bytes) -> Iterator[tuple[str, int, float, float]]:
    """(tag, step, wall time, value) of every scalar in an event file's bytes. A record's checksum is checked when it
    holds scalars (a corrupt one ends the file); the others (images, histograms, graphs: most of the bytes) are only
    skipped, so they are not checked: CRC32C in Python is the slow part of reading.
    """
    for record, crc in _frames(data):
        found = list(_record_scalars(record))
        if found and masked_crc(record) != crc:
            return
        yield from found


def _record_scalars(record: bytes) -> Iterator[tuple[str, int, float, float]]:
    """The scalars of one Event record; none when it is not an event as expected."""
    step, wall, summary = 0, 0.0, b""
    out = []
    try:
        for number, wire, value in fields(record):
            if number == 1 and wire == 1:
                wall = struct.unpack("<d", struct.pack("<Q", value))[0]
            elif number == 2 and wire == 0:
                step = _signed(int(value))  # type: ignore[arg-type]
            elif number == 5 and wire == 2:
                summary = value  # type: ignore[assignment]
        for number, wire, value in fields(summary):
            if number != 1 or wire != 2:
                continue
            tag, scalar = "", None
            for n, w, v in fields(value):  # type: ignore[arg-type]
                if n == 1 and w == 2:
                    tag = v.decode("utf-8", "replace")  # type: ignore[union-attr]
                elif n == 2 and w == 5:
                    scalar = struct.unpack("<f", struct.pack("<I", v))[0]
                elif n == 8 and w == 2:
                    scalar = _tensor_scalar(v)  # type: ignore[arg-type]
            if tag and scalar is not None:
                out.append((tag, step, wall, scalar))
    except (ValueError, struct.error):
        return iter(())  # a record that is not an event as expected: skipped
    return iter(out)


# ---------------------------------------------------------------- runs

SEED_PART = re.compile(r"^(?:seed|s|rep|run|trial)[_=-]?(\d+)$", re.I)


def run_fields(relative: str) -> dict[str, str]:
    """What a run is, from its folder: ``key=value`` parts as fields, a ``seed3`` / ``seed_3`` part as the seed, the
    first other part as the method; and the whole path as "run".

    >>> run_fields("ours/lr=0.1/seed_2")
    {'run': 'ours/lr=0.1/seed_2', 'lr': '0.1', 'seed': '2', 'method': 'ours'}
    """
    out = {"run": relative}
    rest = []
    for part in relative.split("/"):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k] = v
        elif (m := SEED_PART.match(part)) is not None:
            out["seed"] = m.group(1)
        elif part not in (".", ""):
            rest.append(part)
    if rest:
        out["method"] = rest[0]
        if len(rest) > 1:
            out["variant"] = "/".join(rest[1:])
    return out


def _last_values(data: bytes) -> dict[str, tuple[int, float, float]]:
    """Each tag's (step, wall time, value) at its last step in one file. Only the records whose values are kept have
    their checksum checked (CRC32C in Python is slow, and a log holds hundreds of thousands of records); if one of them
    does not hold, the file is read again checking every record, a corrupt one ending it."""
    last: dict[str, tuple[int, float, float, bytes, int]] = {}
    for body, crc in _frames(data):
        for tag, step, wall, value in _record_scalars(body):
            if tag not in last or (step, wall) >= last[tag][:2]:
                last[tag] = (step, wall, value, body, crc)
    kept = {id(v[3]): v for v in last.values()}
    if all(masked_crc(body) == crc for *_, body, crc in kept.values()):
        return {tag: v[:3] for tag, v in last.items()}
    checked: dict[str, tuple[int, float, float]] = {}
    for tag, step, wall, value in scalars(data):
        if tag not in checked or (step, wall) >= checked[tag][:2]:
            checked[tag] = (step, wall, value)
    return checked


def read_tensorboard(folder: Path) -> list[Run]:
    """Runs from a folder of TensorBoard logs: each folder holding event files is a run (its fields from its path),
    each tag a metric at its last step (the latest written, when a step repeats)."""
    by_run: dict[Path, list[Path]] = {}
    for f in sorted(folder.rglob("events.out.tfevents.*")):
        by_run.setdefault(f.parent, []).append(f)
    runs = []
    for run_dir, files in by_run.items():
        relative = run_dir.relative_to(folder).as_posix() if run_dir != folder else run_dir.name
        last: dict[str, tuple[int, float, float]] = {}  # tag -> (step, wall time, value)
        for f in files:
            for tag, (step, wall, value) in _last_values(f.read_bytes()).items():
                if tag not in last or (step, wall) >= last[tag][:2]:
                    last[tag] = (step, wall, value)
        if last:
            runs.append(Run(run_fields(relative), {tag: v for tag, (_, _, v) in last.items()}, relative))
    return runs
