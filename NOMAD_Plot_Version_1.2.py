#!/usr/bin/env python3
"""NOMAD Plot: standalone desktop plotting for Arc One and Keysight DC sweeps.

Dependencies: pip install numpy pandas matplotlib
Keysight: metadata followed by V1,I1 (volts, amperes) is detected automatically.
I-V points are connected in acquisition order, never sorted by voltage.
Choose Signed current / Linear for the reference-style SET/RESET plot.
PNG/PDF exports use IEEE-style serif fonts and 3.5-inch single-column figures.

Run the desktop app:
    python NOMAD_Plot_Version_1.0.py

Run without the GUI (useful for repeatable batch plots):
    python NOMAD_Plot_Version_1.0.py --batch data.csv --instrument arc-one --test iv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import threading
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.figure import Figure


APP_NAME = "NOMAD Plot"
VERSION = "1.5.5"

# IEEE-style defaults; Times New Roman when installed, portable serif fallback.
PLOT_COLORS = ("blue", "red", "black")
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.linewidth": 0.9, "xtick.direction": "in", "ytick.direction": "in",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

INSTRUMENT_AUTO = "Auto-detect"
INSTRUMENT_ARC = "Arc One"
INSTRUMENT_KEYSIGHT = "Keysight / Generic CSV"

TEST_IV = "Curve Tracer (I–V)"
TEST_RETENTION = "Retention"
TEST_PULSE = "Pulse / Resistance"
TEST_XY = "Custom X–Y"
TEST_PD = "Potentiation / Depression"
TEST_FORMING = "Forming"
TEST_ENDURANCE = "Endurance"

LAYOUT_OVERLAY = "One graph (overlay)"
LAYOUT_GRID = "Grid of subplots"
LAYOUT_SEPARATE = "Separate plot files"

ALL_TESTS = (TEST_IV, TEST_FORMING, TEST_ENDURANCE, TEST_RETENTION, TEST_PULSE, TEST_PD)
ALL_LAYOUTS = (LAYOUT_OVERLAY,)
LEGEND_POSITIONS = ("Below plot", "Best", "Upper right", "Upper left", "Lower right", "Lower left", "Center right", "Center left", "Upper center", "Lower center", "Center", "Hidden")


@dataclass
class DataSet:
    path: str
    frame: pd.DataFrame
    instrument: str
    metadata: list[str]

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)


@dataclass(frozen=True)
class DeviceRef:
    dataset_index: int
    wordline: Optional[object]
    bitline: Optional[object]
    label: str
    section: Optional[int] = None


@dataclass
class Curve:
    label: str
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str
    kind: str = "raw"
    legend_label: Optional[str] = None


def normalized_name(value: object) -> str:
    text = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
    return re.sub(r"[^a-z0-9]+", "", text)


def find_column(
    frame: pd.DataFrame,
    exact: Sequence[str] = (),
    contains: Sequence[str] = (),
) -> Optional[str]:
    normalized = {normalized_name(column): column for column in frame.columns}
    for candidate in exact:
        if normalized_name(candidate) in normalized:
            return normalized[normalized_name(candidate)]
    for column in frame.columns:
        key = normalized_name(column)
        if any(normalized_name(fragment) in key for fragment in contains):
            return column
    return None


def wordline_column(frame: pd.DataFrame) -> Optional[str]:
    return find_column(frame, exact=("Wordline", "WL", "Word line"))


def bitline_column(frame: pd.DataFrame) -> Optional[str]:
    return find_column(frame, exact=("Bitline", "BL", "Bit line"))


def resistance_column(frame: pd.DataFrame) -> Optional[str]:
    return find_column(
        frame,
        exact=("Resistance", "Resistance (Ohm)", "Resistance (Ω)", "R"),
        contains=("resistance",),
    )


def voltage_column(frame: pd.DataFrame) -> Optional[str]:
    return find_column(
        frame,
        exact=(
            "Amplitude (V)",
            "Voltage (V)",
            "Voltage",
            "ReadVoltage",
            "V",
            "V1",
        ),
        contains=("voltage",),
    )


def current_column(frame: pd.DataFrame) -> Optional[str]:
    return find_column(
        frame,
        exact=("Current (A)", "Current", "I (A)", "I", "I1"),
        contains=("current",),
    )


def pulse_width_column(frame: pd.DataFrame) -> Optional[str]:
    return find_column(
        frame,
        exact=("Pulse width (s)", "Pulse Width", "PulseWidth"),
        contains=("pulsewidth",),
    )


def time_column(frame: pd.DataFrame) -> Optional[str]:
    exact = (
        "Retention time (s)",
        "Retention time",
        "Elapsed time (s)",
        "Elapsed time",
        "Time (s)",
        "Time",
    )
    found = find_column(frame, exact=exact)
    if found:
        return found
    for column in frame.columns:
        key = normalized_name(column)
        if "time" in key and "pulse" not in key:
            return column
    return None


def tag_column(frame: pd.DataFrame) -> Optional[str]:
    return find_column(frame, exact=("Tag", "TestTag", "Test Tag"))


def _read_rows(path: str, limit: int = 80) -> list[list[str]]:
    rows: list[list[str]] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader):
            rows.append(row)
            if row_number + 1 >= limit:
                break
    return rows


def find_arc_one_header(path: str) -> int:
    for index, row in enumerate(_read_rows(path)):
        names = {normalized_name(value) for value in row}
        required = {"wordline", "bitline", "resistance"}
        if required.issubset(names) and ("amplitudev" in names or "readvoltage" in names):
            return index
    raise ValueError(
        "Arc One header not found. Expected Wordline, Bitline, Resistance, "
        "and Amplitude (V) or ReadVoltage."
    )


def find_keysight_header(path: str) -> Optional[int]:
    """Locate the actual V1/I1 table, not the descriptive I/V Sweep metadata."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.reader(handle)):
            if index >= 1000:
                break
            names = {normalized_name(value) for value in row}
            if "i1" in names and ("v1" in names or "time" in names):
                return index
    return None


def load_keysight_csv(path: str, header_index: int) -> DataSet:
    """Read repeated Keysight metadata/header/data blocks in original order."""
    records = []
    metadata = []
    active = False
    block = -1
    indices = None
    x_name = None
    known_metadata = {"ivtsampling", "ivsweep", "recordtime", "deviceid", "count", "flag", "remarks"}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or not any(cell.strip() for cell in row):
                continue
            names = [normalized_name(cell) for cell in row]
            if "i1" in names and ("v1" in names or "time" in names):
                next_x = "V1" if "v1" in names else "Time"
                if x_name is not None and x_name != next_x:
                    raise ValueError("Mixed voltage/time Keysight tables require separate files.")
                x_name = next_x
                indices = (names.index(normalized_name(x_name)), names.index("i1"))
                block += 1
                active = True
                continue
            key = names[0]
            if key in known_metadata or key.startswith("combinedcsweeps"):
                metadata.append(f"Line {line_number}: " + ", ".join(row))
                active = False
                continue
            if line_number <= header_index:
                metadata.append(", ".join(row))
                continue
            if not active or indices is None:
                raise ValueError(f"Keysight: expected V1,I1 or Time,I1 header before CSV line {line_number}.")
            try:
                voltage, current = (float(row[index].strip()) for index in indices)
                if not np.isfinite(voltage) or not np.isfinite(current):
                    raise ValueError("non-finite measurement")
            except (ValueError, IndexError) as error:
                raise ValueError(
                    f"Keysight: invalid Keysight measurement at CSV line {line_number}: "
                    + ", ".join(row)
                ) from error
            records.append((voltage, current, block))
    if not records:
        raise ValueError("The Keysight file contains no measurements.")
    frame = pd.DataFrame(records, columns=[x_name, "I1", "__keysight_block"])
    return DataSet(os.path.abspath(path), frame, INSTRUMENT_KEYSIGHT, metadata)


def find_generic_header(path: str) -> int:
    rows = _read_rows(path)
    for index, row in enumerate(rows):
        if len(row) < 2:
            continue
        nonempty = [value.strip() for value in row if value.strip()]
        if len(nonempty) >= 2 and sum(any(char.isalpha() for char in value) for value in nonempty) >= 2:
            return index
    return 0


def _metadata_before_header(path: str, header_index: int) -> list[str]:
    metadata: list[str] = []
    for row in _read_rows(path, limit=max(header_index, 1))[:header_index]:
        text = ", ".join(value.strip() for value in row if value.strip())
        if text:
            metadata.append(text)
    return metadata


def load_arc_one(path: str) -> DataSet:
    header_index = find_arc_one_header(path)
    frame = pd.read_csv(path, skiprows=header_index, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        if normalized_name(column) in {
            "wordline",
            "bitline",
            "resistance",
            "amplitudev",
            "pulsewidths",
            "readvoltage",
            "retentiontimes",
            "times",
        }:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.empty:
        raise ValueError("The Arc One file contains a header but no data rows.")
    return DataSet(
        path=os.path.abspath(path),
        frame=frame,
        instrument=INSTRUMENT_ARC,
        metadata=_metadata_before_header(path, header_index),
    )


def load_wgfmu_csv(path: str) -> DataSet:
    """Parse each WGFMU table with its own header and independent sample counts."""
    frames, records, metadata = [], [], []
    columns = None
    section_name = "Measurements"
    section = 0

    def flush():
        if columns is not None and records:
            frame = pd.DataFrame(records, columns=columns)
            frame["__section"] = section
            frame["__section_name"] = section_name
            frames.append(frame)

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row or not any(cell.strip() for cell in row):
                continue
            first = normalized_name(row[0])
            if first in {"potentiationdepression", "wgfmupatterneditor",
                         "wgfmupatterneditorchilddatadisplay"}:
                flush()
                records, columns = [], None
                section += 1
                section_name = row[1].strip() if len(row) > 1 else row[0].strip()
                metadata.append(f"Section {section}: " + ", ".join(row))
                continue
            if first in {"recordtime", "deviceid", "count", "flag", "remarks"}:
                metadata.append(", ".join(row))
                continue
            if columns is None:
                if not all(re.fullmatch(
                        r"(Time|MeasResult\d+_(value|time)|Waveform\d+_(time|voltage)|MeasTiming_(time|flag))",
                        c.strip()) for c in row):
                    raise ValueError(f"Unrecognized WGFMU header at line {line}.")
                columns = [c.strip() for c in row]
                continue
            if len(row) != len(columns):
                raise ValueError(f"WGFMU line {line}: expected {len(columns)} fields, got {len(row)}.")
            values = []
            for cell in row:
                if not cell.strip():
                    values.append(np.nan)
                else:
                    try:
                        number = float(cell)
                        if not np.isfinite(number):
                            raise ValueError("non-finite")
                        values.append(number)
                    except ValueError as error:
                        raise ValueError(f"WGFMU line {line}: invalid measurement {cell!r}.") from error
            records.append(values)
    flush()
    if not frames:
        raise ValueError("No WGFMU measurement tables found.")
    return DataSet(os.path.abspath(path), pd.concat(frames, ignore_index=True),
                   INSTRUMENT_KEYSIGHT, metadata)


def load_generic_csv(path: str) -> DataSet:
    first_rows = _read_rows(path, limit=10)
    if any(row and normalized_name(row[0]) in
           {"potentiationdepression", "wgfmupatterneditor", "wgfmupatterneditorchilddatadisplay"}
           for row in first_rows):
        return load_wgfmu_csv(path)
    keysight_header = find_keysight_header(path)
    if keysight_header is not None:
        return load_keysight_csv(path, keysight_header)
    header_index = find_generic_header(path)
    try:
        frame = pd.read_csv(path, skiprows=header_index, encoding="utf-8-sig")
    except Exception:
        frame = pd.read_csv(
            path,
            skiprows=header_index,
            encoding="utf-8-sig",
            sep=None,
            engine="python",
        )
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("The file does not contain a usable two-column data table.")
    return DataSet(
        path=os.path.abspath(path),
        frame=frame,
        instrument=INSTRUMENT_KEYSIGHT,
        metadata=_metadata_before_header(path, header_index),
    )


def detect_instrument(path: str) -> str:
    try:
        find_arc_one_header(path)
        return INSTRUMENT_ARC
    except ValueError:
        return INSTRUMENT_KEYSIGHT


def load_dataset(path: str, instrument: str) -> DataSet:
    selected = detect_instrument(path) if instrument == INSTRUMENT_AUTO else instrument
    if selected == INSTRUMENT_ARC:
        return load_arc_one(path)
    return load_generic_csv(path)


def _tag_values(frame: pd.DataFrame) -> pd.Series:
    column = tag_column(frame)
    if column is None:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip().str.upper()


def rows_for_test(frame: pd.DataFrame, test: str) -> pd.Series:
    if test == TEST_XY:
        return pd.Series(True, index=frame.index)
    if "__section" in frame.columns:
        if test in (TEST_PD, TEST_ENDURANCE):
            column = find_column(frame, exact=("MeasResult2_value", "MeasResult1_value"))
            return frame[column].notna() if column else pd.Series(False, index=frame.index)
        return pd.Series(False, index=frame.index)
    if test == TEST_PD:
        return pd.Series(False, index=frame.index)
    tags = _tag_values(frame)
    if test in (TEST_IV, TEST_FORMING):
        tagged = tags.str.startswith("CT")
        if tagged.any():
            return tagged
        voltage = voltage_column(frame)
        resistance = resistance_column(frame)
        current = current_column(frame)
        if voltage and (resistance or current):
            return pd.to_numeric(frame[voltage], errors="coerce").notna()
        return pd.Series(False, index=frame.index)
    if test == TEST_RETENTION:
        tagged = tags.str.startswith("RET")
        if tagged.any():
            return tagged
        time = time_column(frame)
        resistance = resistance_column(frame)
        if time and (resistance or current_column(frame)):
            return pd.to_numeric(frame[time], errors="coerce").notna()
        return pd.Series(False, index=frame.index)
    if test == TEST_PULSE:
        pulse = pulse_width_column(frame)
        pulse_mask = pd.Series(False, index=frame.index)
        if pulse:
            pulse_mask = pd.to_numeric(frame[pulse], errors="coerce").fillna(0).ne(0)
        tagged = tags.str.startswith("P")
        return pulse_mask | tagged
    return pd.Series(False, index=frame.index)


def format_device_value(value: object) -> str:
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return f"{int(number):02d}"
    except (TypeError, ValueError):
        pass
    return str(value)


def build_device_catalog(datasets: Sequence[DataSet], test: str) -> list[DeviceRef]:
    catalog: list[DeviceRef] = []
    use_prefix = len(datasets) > 1
    for dataset_index, dataset in enumerate(datasets):
        frame = dataset.frame
        mask = rows_for_test(frame, test)
        available = frame.loc[mask]
        if available.empty:
            continue
        if "__section" in frame.columns:
            for section_id, section_frame in available.groupby("__section", sort=False):
                name = str(section_frame["__section_name"].iloc[0])
                label = f"{dataset.filename} | {name} [{int(section_id)}]"
                catalog.append(DeviceRef(dataset_index, None, None, label, int(section_id)))
            continue
        wl_column = wordline_column(frame)
        bl_column = bitline_column(frame)
        prefix = f"{dataset.filename} | " if use_prefix else ""
        if wl_column and bl_column:
            device_rows = available[[wl_column, bl_column]].dropna().drop_duplicates()
            for wl, bl in device_rows.itertuples(index=False, name=None):
                label = f"{prefix}WL{format_device_value(wl)}–BL{format_device_value(bl)}"
                catalog.append(DeviceRef(dataset_index, wl, bl, label))
        else:
            catalog.append(DeviceRef(dataset_index, None, None, dataset.filename))
    return catalog


def subset_for_device(dataset: DataSet, ref: DeviceRef) -> pd.DataFrame:
    frame = dataset.frame
    if ref.section is not None:
        frame = frame.loc[frame["__section"] == ref.section]
    if ref.wordline is None or ref.bitline is None:
        return frame
    wl_column = wordline_column(frame)
    bl_column = bitline_column(frame)
    if not wl_column or not bl_column:
        return frame
    return frame.loc[(frame[wl_column] == ref.wordline) & (frame[bl_column] == ref.bitline)]


def current_to_amperes(values: pd.Series, column_name: str) -> np.ndarray:
    data = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    name = str(column_name).lower().replace("µ", "u").replace("μ", "u")
    if "(ma" in name or "[ma" in name:
        data *= 1e-3
    elif "(ua" in name or "[ua" in name:
        data *= 1e-6
    elif "(na" in name or "[na" in name:
        data *= 1e-9
    elif "(pa" in name or "[pa" in name:
        data *= 1e-12
    return data


def _valid_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _split_iv_runs(frame: pd.DataFrame) -> Iterable[tuple[int, pd.DataFrame]]:
    if "__keysight_block" in frame.columns:
        for block, run in frame.groupby("__keysight_block", sort=False):
            yield int(block) + 1, run
        return
    tags = _tag_values(frame)
    start = tags.str.startswith("CT_S")
    if start.any():
        run_ids = start.cumsum()
        for run_id, run in frame.groupby(run_ids, sort=False):
            yield int(run_id), run
    else:
        yield 1, frame


def _iv_curves(dataset: DataSet, ref: DeviceRef, current_mode: str) -> list[Curve]:
    frame = subset_for_device(dataset, ref)
    frame = frame.loc[rows_for_test(frame, TEST_IV)]
    voltage = voltage_column(frame)
    if not voltage:
        raise ValueError(f"{dataset.filename}: no voltage column was found.")
    resistance = resistance_column(frame)
    measured_current = current_column(frame)
    if not resistance and not measured_current:
        raise ValueError(f"{dataset.filename}: no current or resistance column was found.")

    output: list[Curve] = []
    runs = list(_split_iv_runs(frame))
    for run_number, run in runs:
        x = pd.to_numeric(run[voltage], errors="coerce").to_numpy(dtype=float)
        if measured_current:
            y = current_to_amperes(run[measured_current], measured_current)
        else:
            resistance_values = pd.to_numeric(run[resistance], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.divide(
                    x,
                    resistance_values,
                    out=np.full_like(x, np.nan, dtype=float),
                    where=resistance_values != 0,
                )
            # V / infinite resistance is physically a zero-current point and is retained.
            y[np.isinf(resistance_values) & np.isfinite(x)] = 0.0
        if current_mode == "Magnitude |I|":
            y = np.abs(y)
        x, y = _valid_xy(x, y)
        if not len(x):
            continue
        label = ref.label if len(runs) == 1 else f"{ref.label} / sweep {run_number}"
        output.append(Curve(label, x, y, "Voltage (V)", "Current (A)", "current_a"))
    if "__keysight_block" in frame.columns and len(output) > 1:
        # One file remains one colored trace. NaN separators preserve each
        # block's acquisition order without inventing inter-block segments.
        x_parts, y_parts = [], []
        for index, curve in enumerate(output):
            if index:
                x_parts.append(np.array([np.nan]))
                y_parts.append(np.array([np.nan]))
            x_parts.append(curve.x)
            y_parts.append(curve.y)
        return [Curve(ref.label, np.concatenate(x_parts), np.concatenate(y_parts),
                      "Voltage (V)", "Current (A)", "current_a")]
    return output


def _contiguous_retention_runs(frame: pd.DataFrame, mask: pd.Series) -> list[pd.DataFrame]:
    positions = np.flatnonzero(mask.to_numpy())
    if not len(positions):
        return []
    split_at = np.flatnonzero(np.diff(positions) > 1) + 1
    groups = np.split(positions, split_at)
    return [frame.iloc[group] for group in groups if len(group)]


def _retention_curves(dataset: DataSet, ref: DeviceRef, current_mode: str = "Signed current") -> list[Curve]:
    frame = subset_for_device(dataset, ref)
    mask = rows_for_test(frame, TEST_RETENTION)
    runs = _contiguous_retention_runs(frame, mask)
    if "__keysight_block" in frame.columns:
        runs = [run for _, run in frame.loc[mask].groupby("__keysight_block", sort=False)]
    resistance = resistance_column(frame)
    measured_current = current_column(frame)
    if not resistance and not measured_current:
        raise ValueError(f"{dataset.filename}: no resistance or measured-current column was found.")
    time = time_column(frame)
    output: list[Curve] = []
    for index, run in enumerate(runs, start=1):
        if time:
            x = pd.to_numeric(run[time], errors="coerce").to_numpy(dtype=float)
            x_label = "Time (s)" if "__keysight_block" in frame.columns else str(time)
        else:
            x = np.arange(len(run), dtype=float)
            x_label = "Reading index"
        if resistance:
            y = pd.to_numeric(run[resistance], errors="coerce").to_numpy(dtype=float)
            y_label, kind = "Resistance (Ω)", "resistance_ohm"
        else:
            y = current_to_amperes(run[measured_current], measured_current)
            if current_mode == "Magnitude |I|":
                y = np.abs(y)
            y_label, kind = "Current (A)", "current_a"
        x, y = _valid_xy(x, y)
        if not len(x):
            continue
        label = ref.label if len(runs) == 1 else f"{ref.label} / run {index}"
        output.append(Curve(label, x, y, x_label, y_label, kind))
    return output


def _pulse_curves(dataset: DataSet, ref: DeviceRef) -> list[Curve]:
    frame = subset_for_device(dataset, ref)
    frame = frame.loc[rows_for_test(frame, TEST_PULSE)]
    pulse = pulse_width_column(frame)
    resistance = resistance_column(frame)
    if not pulse or not resistance:
        raise ValueError(
            f"{dataset.filename}: pulse plotting needs Pulse width and Resistance columns."
        )
    x = pd.to_numeric(frame[pulse], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[resistance], errors="coerce").to_numpy(dtype=float)
    x, y = _valid_xy(x, y)
    if not len(x):
        return []
    return [Curve(ref.label, x, y, str(pulse), "Resistance (Ω)", "resistance_ohm")]


def _xy_curves(dataset: DataSet, ref: DeviceRef, x_column: str, y_column: str) -> list[Curve]:
    frame = subset_for_device(dataset, ref)
    if x_column not in frame.columns or y_column not in frame.columns:
        return []
    x = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)
    x, y = _valid_xy(x, y)
    if not len(x):
        return []
    return [Curve(ref.label, x, y, x_column, y_column, "raw")]


def _endurance_curves(dataset: DataSet, ref: DeviceRef, order: str) -> list[Curve]:
    """Each consecutive pair is one cycle; never sort readings by magnitude."""
    frame = subset_for_device(dataset, ref)
    if "MeasResult2_value" not in frame:
        raise ValueError("Endurance requires a WGFMU MeasResult2_value current channel (A).")
    y = pd.to_numeric(frame["MeasResult2_value"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("Endurance contains missing current readings; pairing would be ambiguous.")
    labels = {"Read 1 / Read 2": ("Read 1", "Read 2"),
              "HRS / LRS": ("HRS", "LRS"), "LRS / HRS": ("LRS", "HRS")}[order]
    result = []
    for phase, label in enumerate(labels):
        values = np.abs(y[phase::2])
        if len(values):
            result.append(Curve(f"{label} | {ref.label}",
                np.arange(1, len(values) + 1, dtype=float), values,
                "Cycle Number", "|Current| (A)", f"endurance_{phase}"))
    return result


def _scaled_curves(curves: list[Curve]) -> list[Curve]:
    if not curves:
        return curves
    kinds = {curve.kind for curve in curves}
    if len(kinds) != 1:
        return curves
    kind = next(iter(kinds))
    finite_values = [np.abs(curve.y[np.isfinite(curve.y)]) for curve in curves]
    finite_values = [values for values in finite_values if len(values)]
    if not finite_values:
        return curves
    maximum = float(max(np.max(values) for values in finite_values))
    if kind == "current_a":
        if maximum >= 1:
            factor, label = 1.0, "Current (A)"
        elif maximum >= 1e-3:
            factor, label = 1e3, "Current (mA)"
        elif maximum >= 1e-6:
            factor, label = 1e6, "Current (µA)"
        elif maximum >= 1e-9:
            factor, label = 1e9, "Current (nA)"
        else:
            factor, label = 1e12, "Current (pA)"
        return [replace(curve, y=curve.y * factor, y_label=label) for curve in curves]
    if kind == "resistance_ohm":
        if maximum >= 1e9:
            factor, label = 1e-9, "Resistance (GΩ)"
        elif maximum >= 1e6:
            factor, label = 1e-6, "Resistance (MΩ)"
        elif maximum >= 1e3:
            factor, label = 1e-3, "Resistance (kΩ)"
        else:
            factor, label = 1.0, "Resistance (Ω)"
        return [replace(curve, y=curve.y * factor, y_label=label) for curve in curves]
    return curves


def build_curves(
    datasets: Sequence[DataSet],
    references: Sequence[DeviceRef],
    test: str,
    current_mode: str = "Signed current",
    x_column: str = "",
    y_column: str = "",
    endurance_order: str = "Read 1 / Read 2",
) -> list[Curve]:
    curves: list[Curve] = []
    for ref in references:
        dataset = datasets[ref.dataset_index]
        if test in (TEST_IV, TEST_FORMING):
            curves.extend(_iv_curves(dataset, ref, current_mode))
        elif test == TEST_ENDURANCE:
            curves.extend(_endurance_curves(dataset, ref, endurance_order))
        elif test == TEST_RETENTION:
            curves.extend(_retention_curves(dataset, ref, current_mode))
        elif test == TEST_PULSE:
            curves.extend(_pulse_curves(dataset, ref))
        elif test == TEST_PD:
            frame = subset_for_device(dataset, ref)
            yc = y_column if y_column in frame.columns else "MeasResult2_value"
            xc = x_column if x_column in frame.columns else yc.replace("_value", "_time")
            if xc not in frame.columns or not frame[xc].notna().any():
                xc = "Time"
            curves.extend(_xy_curves(dataset, ref, xc, yc))
        elif test == TEST_XY:
            curves.extend(_xy_curves(dataset, ref, x_column, y_column))
    curves = [curve for curve in curves if len(curve.x) and len(curve.y)]
    if not curves:
        raise ValueError("No plottable numeric data were found for the current selection.")
    if test == TEST_FORMING:
        return [replace(c, y=c.y * 1e3, y_label="Current (mA)") for c in curves]
    return _scaled_curves(curves)


def _resolved_scale(requested: str, test: str) -> str:
    if requested != "Auto":
        return requested.lower()
    return "log" if test in (TEST_RETENTION, TEST_ENDURANCE) else "linear"


def _apply_scale(axis, curves: Sequence[Curve], requested: str, test: str) -> None:
    if len({curve.y_label for curve in curves}) > 1:
        raise ValueError("These traces have different Y quantities or units. Export separate plots or select matching traces.")
    scale = _resolved_scale(requested, test)
    if requested == "Auto" and test == TEST_RETENTION and all(c.kind == "current_a" for c in curves):
        scale = "linear"
    values = np.concatenate([curve.y[np.isfinite(curve.y)] for curve in curves])
    if scale == "log":
        if np.any(values <= 0):
            raise ValueError("Log scale requires positive Y values. Use Linear or SymLog.")
        axis.set_yscale("log")
    elif scale == "symlog":
        nonzero = np.abs(values[values != 0])
        threshold = float(np.min(nonzero) * 5) if len(nonzero) else 1e-12
        axis.set_yscale("symlog", linthresh=threshold)


def _style_axis(axis, curves: Sequence[Curve], test: str, scale: str) -> None:
    axis.set_xlabel(curves[0].x_label)
    axis.set_ylabel(curves[0].y_label)
    axis.grid(True, axis="y", which="major", color="0.7",
              linestyle="--", linewidth=0.5, alpha=0.8)
    axis.tick_params(which="both", direction="in")
    axis.minorticks_on()
    _apply_scale(axis, curves, scale, test)
    if test == TEST_RETENTION and all(c.kind == "current_a" for c in curves):
        values = np.concatenate([c.y[np.isfinite(c.y)] for c in curves])
        if axis.get_yscale() == "linear" and len(values) and np.all(values >= 0):
            axis.set_ylim(bottom=0)
    if test in (TEST_IV, TEST_FORMING):
        axis.axhline(0, color="#6B7280", linewidth=0.7, zorder=0)
        axis.axvline(0, color="#6B7280", linewidth=0.7, zorder=0)


def _plot_curve(axis, curve: Curve, color: Optional[object] = None) -> None:
    marker = ("o" if curve.kind == "endurance_0" else "s") if curve.kind.startswith("endurance_") else None
    axis.plot(
        curve.x,
        curve.y,
        label=curve.label,
        color=color,
        linewidth=1.35,
        linestyle="None" if curve.kind.startswith("endurance_") else "-",
        marker=marker,
        markersize=2.6,
        markeredgewidth=0,
    )


def overlay_figure(curves: Sequence[Curve], title: str, test: str, scale: str,
                   legend_size: float = 6.5, legend_position: str = "Below plot",
                   tick_size: float = 8, axis_label_size: float = 9,
                   tick_bold: bool = False, axis_label_bold: bool = False,
                   x_label: str = "", y_label: str = "",
                   title_size: float = 10, title_bold: bool = False) -> Figure:
    if not np.isfinite(legend_size) or not 1 <= legend_size <= 72:
        raise ValueError("Legend font size must be between 1 and 72 points.")
    if legend_position not in LEGEND_POSITIONS:
        raise ValueError("Choose a valid legend position.")
    for name, size in (("Tick number", tick_size), ("Axis label", axis_label_size), ("Title", title_size)):
        if not np.isfinite(size) or not 1 <= size <= 72:
            raise ValueError(f"{name} font size must be between 1 and 72 points.")
    show_legend = bool(curves) and legend_position != "Hidden"
    below = show_legend and legend_position == "Below plot"
    legend_height = (legend_size / 72 * 1.9 * len(curves) + 0.1) if below else 0
    figure = Figure(figsize=(3.5, 2.65 + legend_height), dpi=110,
                    facecolor="white", constrained_layout=True)
    if below:
        grid = figure.add_gridspec(2, 1, height_ratios=(2.65, legend_height))
        axis = figure.add_subplot(grid[0])
        legend_axis = figure.add_subplot(grid[1])
        legend_axis.set_axis_off()
    else:
        axis = figure.add_subplot(111)
    for index, curve in enumerate(curves):
        _plot_curve(axis, curve, PLOT_COLORS[index % 3])
    _style_axis(axis, curves, test, scale)
    axis.tick_params(axis="both", which="both", labelsize=tick_size)
    for coordinate in (axis.xaxis, axis.yaxis):
        coordinate.label.set_fontsize(axis_label_size)
        coordinate.label.set_fontweight("bold" if axis_label_bold else "normal")
        coordinate.get_offset_text().set_fontsize(tick_size)
        coordinate.get_offset_text().set_fontweight("bold" if tick_bold else "normal")
        for tick in coordinate.get_major_ticks() + coordinate.get_minor_ticks():
            for label in (tick.label1, tick.label2):
                label.set_fontsize(tick_size)
                label.set_fontweight("bold" if tick_bold else "normal")
    if x_label.strip():
        axis.set_xlabel(x_label, fontsize=axis_label_size,
                        fontweight="bold" if axis_label_bold else "normal")
    if y_label.strip():
        axis.set_ylabel(y_label, fontsize=axis_label_size,
                        fontweight="bold" if axis_label_bold else "normal")
    axis.set_title(textwrap.fill(title, 48), pad=6, fontsize=title_size,
                   fontweight="bold" if title_bold else "normal")
    if show_legend:
        labels = []
        for index, curve in enumerate(curves, 1):
            if curve.legend_label is not None:
                labels.append(curve.legend_label)
                continue
            name = curve.label.replace(".csv", "")
            # Retain section names for WGFMU without repeating long filenames.
            if " | " in name:
                filename, section = name.split(" | ", 1)
                name = filename[:18] + " / " + section
            if len(name) > 44:
                name = name[:28] + "…" + name[-15:]
            labels.append(f"{index}. {name}")
        target = legend_axis if below else axis
        target.legend(axis.lines[:len(curves)], labels,
                      loc="center" if below else legend_position.lower(),
                      fontsize=legend_size, frameon=not below, handlelength=2)
    return figure


def grid_figures(
    curves: Sequence[Curve],
    title: str,
    test: str,
    scale: str,
    per_page: int = 12,
) -> list[Figure]:
    figures: list[Figure] = []
    page_count = math.ceil(len(curves) / per_page)
    for page_index in range(page_count):
        page = curves[page_index * per_page : (page_index + 1) * per_page]
        columns = min(3, max(1, math.ceil(math.sqrt(len(page)))))
        rows = math.ceil(len(page) / columns)
        figure = Figure(
            figsize=(3.5 if columns == 1 else 7.16, 2.65 * rows),
            dpi=105,
            facecolor="white",
            constrained_layout=True,
        )
        axes = figure.subplots(rows, columns, squeeze=False)
        for curve_index, (axis, curve) in enumerate(zip(axes.flat, page)):
            color = PLOT_COLORS[(page_index * per_page + curve_index) % 3]
            _plot_curve(axis, curve, color)
            _style_axis(axis, [curve], test, scale)
            axis.set_title(textwrap.fill(curve.label, 32), fontsize=7)
        for axis in axes.flat[len(page) :]:
            axis.remove()
        suffix = f" — page {page_index + 1}/{page_count}" if page_count > 1 else ""
        figure.suptitle(title + suffix, fontsize=10)
        figures.append(figure)
    return figures


def single_figure(curve: Curve, title: str, test: str, scale: str) -> Figure:
    figure = Figure(figsize=(3.5, 2.65), dpi=110, facecolor="white", constrained_layout=True)
    axis = figure.add_subplot(111)
    _plot_curve(axis, curve, PLOT_COLORS[0])
    _style_axis(axis, [curve], test, scale)
    axis.set_title(textwrap.fill(title, 48), pad=6)
    return figure


def preview_figure(
    curves: Sequence[Curve], layout: str, title: str, test: str, scale: str,
    legend_size: float = 6.5, legend_position: str = "Below plot",
    tick_size: float = 8, axis_label_size: float = 9,
    tick_bold: bool = False, axis_label_bold: bool = False,
    x_label: str = "", y_label: str = "",
    title_size: float = 10, title_bold: bool = False
) -> Figure:
    if layout == LAYOUT_OVERLAY:
        return overlay_figure(curves, title, test, scale, legend_size, legend_position,
                              tick_size, axis_label_size, tick_bold, axis_label_bold,
                              x_label, y_label, title_size, title_bold)
    # Grid is also the clearest contact-sheet preview for separate-file export.
    return grid_figures(curves, title, test, scale, per_page=12)[0]


def safe_name(value: str, fallback: str = "plot") -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", value).strip(" ._")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", cleaned, re.I):
        cleaned = "_" + cleaned
    return cleaned[:120] or fallback


def unique_output_directory(parent: str, title: str) -> str:
    base = os.path.join(parent, safe_name(title, "Research Plot"))
    candidate = base
    number = 2
    while os.path.exists(candidate):
        candidate = f"{base}_{number}"
        number += 1
    os.makedirs(candidate, exist_ok=False)
    return candidate


def export_curves(
    curves: Sequence[Curve],
    layout: str,
    title: str,
    test: str,
    scale: str,
    formats: Sequence[str],
    dpi: int,
    output_parent: str,
    source_paths: Sequence[str],
    instrument: str,
    legend_size: float = 6.5,
    legend_position: str = "Below plot",
    tick_size: float = 8, axis_label_size: float = 9,
    tick_bold: bool = False, axis_label_bold: bool = False,
    x_label: str = "", y_label: str = "",
    title_size: float = 10, title_bold: bool = False,
) -> tuple[str, list[str]]:
    output_directory = unique_output_directory(output_parent, title)
    saved: list[str] = []

    def save(figure: Figure, stem: str) -> None:
        for extension in formats:
            path = os.path.join(output_directory, f"{safe_name(stem)}.{extension}")
            figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            saved.append(path)
        figure.clear()

    if layout == LAYOUT_OVERLAY:
        save(overlay_figure(curves, title, test, scale, legend_size, legend_position,
                              tick_size, axis_label_size, tick_bold, axis_label_bold,
                              x_label, y_label, title_size, title_bold), title)
    elif layout == LAYOUT_GRID:
        figures = grid_figures(curves, title, test, scale, per_page=12)
        for index, figure in enumerate(figures, start=1):
            suffix = f"_page_{index:02d}" if len(figures) > 1 else ""
            save(figure, title + suffix)
    else:
        used: dict[str, int] = {}
        for curve in curves:
            stem = safe_name(curve.label)
            used[stem] = used.get(stem, 0) + 1
            if used[stem] > 1:
                stem = f"{stem}_{used[stem]}"
            save(single_figure(curve, curve.label, test, scale), stem)

    settings = {
        "app": APP_NAME,
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [os.path.abspath(path) for path in source_paths],
        "instrument_selection": instrument,
        "test": test,
        "layout": layout,
        "scale": scale,
        "formats": list(formats),
        "dpi": dpi,
        "x_label_override": x_label,
        "y_label_override": y_label,
        "title_size": title_size,
        "title_bold": title_bold,
        "tick_size": tick_size,
        "axis_label_size": axis_label_size,
        "tick_bold": tick_bold,
        "axis_label_bold": axis_label_bold,
        "legend_size": legend_size,
        "legend_position": legend_position,
        "trace_count": len(curves),
        "traces": [curve.label for curve in curves],
        "legend_labels": [curve.legend_label for curve in curves],
    }
    settings_path = os.path.join(output_directory, "plot_settings.json")
    with open(settings_path, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2, ensure_ascii=False)
    saved.append(settings_path)
    return output_directory, saved


def numeric_columns(datasets: Sequence[DataSet]) -> list[str]:
    names: list[str] = []
    for dataset in datasets:
        for column in dataset.frame.columns:
            if str(column).startswith("__"):
                continue
            converted = pd.to_numeric(dataset.frame[column], errors="coerce")
            if converted.notna().any() and column not in names:
                names.append(column)
    return names


def scan_directory(path):
    """List one level only; never parse CSV contents or recursively walk a tree."""
    with os.scandir(path) as entries:
        items = [(entry.name, entry.path, entry.is_dir(follow_symlinks=False))
                 for entry in entries]
    return sorted(items, key=lambda item: (not item[2], item[0].casefold()))


def load_sources_worker(paths, folder, instrument, test, cancel, messages):
    """Only filesystem/data operations here. Tk widgets stay on the main thread."""
    try:
        paths = list(paths)
        if folder:
            messages.put(("progress", "Scanning selected folder (no subfolders)…"))
            found = []
            with os.scandir(folder) as entries:
                for entry in entries:
                    if cancel.is_set():
                        return
                    if entry.name.lower().endswith(".csv") and entry.is_file():
                        found.append(os.path.abspath(entry.path))
            paths = list(dict.fromkeys(paths + sorted(found)))
            if not found:
                messages.put(("empty", None))
                return
        loaded, errors = [], []
        for index, path in enumerate(paths, 1):
            if cancel.is_set():
                return
            messages.put(("progress", f"Reading {index}/{len(paths)}: {os.path.basename(path)}"))
            try:
                loaded.append(load_dataset(path, instrument))
            except Exception as error:
                errors.append(f"{os.path.basename(path)}: {error}")
        if cancel.is_set():
            return
        messages.put(("progress", "Preparing column and device lists…"))
        columns = numeric_columns(loaded)
        catalog = build_device_catalog(loaded, test)
        if not cancel.is_set():
            messages.put(("done", (paths, loaded, errors, columns, catalog, test)))
    except Exception as error:
        messages.put(("error", str(error)))


class ResearchPlotterApp:
    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title(f"{APP_NAME} {VERSION}")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{min(1400, screen_width - 60)}x{min(850, screen_height - 100)}")
        self.root.minsize(min(900, screen_width - 60), min(480, screen_height - 100))

        self._load_cancel = threading.Event()
        self._load_generation = 0
        self._explorer_generation = 0
        self._explorer_nodes = {}
        self._explorer_loaded = set()
        self._explorer_loading = set()
        self._explorer_root = None
        self._explorer_selection_job = None
        self._preview_after_load = False
        self._comparison_paths = []
        self.file_paths: list[str] = []
        self.datasets: list[DataSet] = []
        self.catalog: list[DeviceRef] = []
        self.canvas = None
        self.toolbar = None
        self.last_output_directory = ""

        self.instrument_var = tk.StringVar(value=INSTRUMENT_AUTO)
        self.test_var = tk.StringVar(value=TEST_IV)
        self.endurance_order_var = tk.StringVar(value="Read 1 / Read 2")
        self.layout_var = tk.StringVar(value=LAYOUT_OVERLAY)
        self.current_mode_var = tk.StringVar(value="Signed current")
        self.scale_var = tk.StringVar(value="Linear")
        self.x_label_var = tk.StringVar(value="")
        self.y_label_var = tk.StringVar(value="")
        self.title_size_var = tk.DoubleVar(value=10)
        self.title_bold_var = tk.BooleanVar(value=False)
        self.tick_size_var = tk.DoubleVar(value=8)
        self.axis_label_size_var = tk.DoubleVar(value=9)
        self.tick_bold_var = tk.BooleanVar(value=False)
        self.axis_label_bold_var = tk.BooleanVar(value=False)
        self.legend_names = {}
        self.show_legend_var = tk.BooleanVar(value=True)
        self.legend_size_var = tk.DoubleVar(value=6.5)
        self.legend_position_var = tk.StringVar(value="Below plot")
        self.title_var = tk.StringVar(value="I–V Characteristics")
        self.x_var = tk.StringVar(value="")
        self.y_var = tk.StringVar(value="")
        self.png_var = tk.BooleanVar(value=True)
        self.pdf_var = tk.BooleanVar(value=True)
        self.jpg_var = tk.BooleanVar(value=False)
        self.svg_var = tk.BooleanVar(value=False)
        self.dpi_var = tk.IntVar(value=300)
        self.status_var = tk.StringVar(value="Add one or more CSV files to begin.")

        self._configure_style()
        self._build_ui()

    def _configure_style(self) -> None:
        style = self.ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk
        view_bar = ttk.Frame(self.root, padding=(8, 4, 8, 0))
        view_bar.pack(fill=tk.X)
        self.explorer_toggle = ttk.Button(view_bar, text="Hide Explorer",
                                         command=self.toggle_explorer)
        self.explorer_toggle.pack(side=tk.LEFT)
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        self.outer_panes = outer
        self.explorer_visible = True
        self.explorer_width = 260
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        explorer = ttk.Frame(outer, width=260, padding=(4, 2))
        self.explorer_panel = explorer
        outer.add(explorer, weight=0)
        ttk.Label(explorer, text="Explorer", style="Title.TLabel").pack(anchor="w")
        explorer_buttons = ttk.Frame(explorer)
        explorer_buttons.pack(fill=tk.X, pady=(7, 5))
        ttk.Button(explorer_buttons, text="Open folder", command=self.add_folder).pack(side=tk.LEFT)
        ttk.Button(explorer_buttons, text="Refresh", command=self.refresh_explorer).pack(side=tk.LEFT, padx=4)
        tree_holder = ttk.Frame(explorer)
        tree_holder.pack(fill=tk.BOTH, expand=True)
        self.file_tree = ttk.Treeview(tree_holder, show="tree", selectmode="extended")
        self.file_tree.column("#0", width=245, minwidth=120, stretch=False)
        tree_y = ttk.Scrollbar(tree_holder, orient=tk.VERTICAL, command=self.file_tree.yview)
        tree_x = ttk.Scrollbar(tree_holder, orient=tk.HORIZONTAL, command=self.file_tree.xview)
        self.file_tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        self.file_tree.grid(row=0, column=0, sticky="nsew")
        tree_y.grid(row=0, column=1, sticky="ns")
        tree_x.grid(row=1, column=0, sticky="ew")
        tree_holder.rowconfigure(0, weight=1)
        tree_holder.columnconfigure(0, weight=1)
        self.file_tree.tag_configure("other", foreground="#888888")
        self.file_tree.bind("<<TreeviewOpen>>", self._expand_explorer)
        self.file_tree.bind("<Double-1>", self._double_click_explorer)
        self.file_tree.bind("<Return>", lambda event: self.add_explorer_selection())
        ttk.Button(explorer, text="Add selected CSVs to current plot",
                   command=self.add_explorer_selection).pack(fill=tk.X, pady=(7, 3))
        ttk.Label(explorer, text="Double-click a CSV to add it to the current plot.\n"
                  "Single-click selects. Ctrl/Shift-click selects multiple; use Add selected CSVs. Expand folders with the arrow.",
                  wraplength=250, justify=tk.LEFT).pack(anchor="w")

        # Reserve a fixed bottom row for actions. Only the settings scroll,
        # so a short window or Windows display scaling cannot hide the buttons.
        control_shell = ttk.Frame(outer, width=390, padding=(4, 2))
        control_shell.rowconfigure(0, weight=1)
        control_shell.columnconfigure(0, weight=1)
        settings_view = ttk.Frame(control_shell)
        settings_view.grid(row=0, column=0, sticky="nsew")
        settings_view.rowconfigure(0, weight=1)
        settings_view.columnconfigure(0, weight=1)
        settings_canvas = tk.Canvas(settings_view, width=370, height=300,
                                    highlightthickness=0, borderwidth=0)
        settings_scroll = ttk.Scrollbar(settings_view, orient=tk.VERTICAL,
                                        command=settings_canvas.yview)
        settings_canvas.configure(yscrollcommand=settings_scroll.set)
        settings_canvas.grid(row=0, column=0, sticky="nsew")
        settings_scroll.grid(row=0, column=1, sticky="ns")
        controls = ttk.Frame(settings_canvas, padding=(0, 0, 4, 0))
        settings_window = settings_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda event: settings_canvas.configure(
            scrollregion=settings_canvas.bbox("all")))
        settings_canvas.bind("<Configure>", lambda event: settings_canvas.itemconfigure(
            settings_window, width=event.width))
        self.settings_canvas = settings_canvas
        self.settings_controls = controls
        preview = ttk.Frame(outer, padding=(8, 2))
        outer.add(control_shell, weight=0)
        outer.add(preview, weight=1)

        ttk.Label(controls, text=APP_NAME, style="Title.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(
            controls,
            text="Arc One and Keysight-ready research plotting",
            foreground="#4B5563",
        ).pack(anchor="w", pady=(0, 9))

        source_box = ttk.Labelframe(
            controls, text="1. Files in current plot", style="Section.TLabelframe", padding=8
        )
        source_box.pack(fill=tk.X, pady=(0, 7))
        list_frame = ttk.Frame(source_box)
        list_frame.pack(fill=tk.X)
        self.file_list = tk.Listbox(list_frame, height=4, font=("Segoe UI", 9))
        file_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=file_scroll.set)
        self.file_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        source_buttons = ttk.Frame(source_box)
        source_buttons.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(source_buttons, text="Add CSV files", command=self.add_files).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(source_buttons, text="Open folder", command=self.add_folder).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(source_buttons, text="Clear", command=self.clear_files).pack(side=tk.LEFT)
        ttk.Button(source_box, text="Cancel loading", command=self.cancel_loading).pack(
            anchor="w", pady=(5, 0))
        ttk.Label(source_box, text="Double-click CSVs in Explorer to add them here.",
                  foreground="#4B5563").pack(anchor="w", pady=(3, 0))

        setup_box = ttk.Labelframe(
            controls, text="2. Plot setup", style="Section.TLabelframe", padding=8
        )
        setup_box.pack(fill=tk.X, pady=(0, 7))
        setup_box.columnconfigure(1, weight=1)
        self._combo_row(
            setup_box,
            0,
            "Instrument",
            self.instrument_var,
            (INSTRUMENT_AUTO, INSTRUMENT_ARC, INSTRUMENT_KEYSIGHT),
            self.on_instrument_changed,
        )
        self.test_combo = self._combo_row(
            setup_box, 1, "Test type", self.test_var, ALL_TESTS, self.on_test_changed
        )
        self._combo_row(setup_box, 2, "Layout", self.layout_var, ALL_LAYOUTS)
        self._combo_row(
            setup_box,
            3,
            "Current",
            self.current_mode_var,
            ("Signed current", "Magnitude |I|"),
        )
        self._combo_row(
            setup_box, 4, "Y scale", self.scale_var, ("Linear",)
        )
        ttk.Label(setup_box, text="Title").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Entry(setup_box, textvariable=self.title_var).grid(
            row=5, column=1, sticky="ew", padx=(7, 0), pady=3
        )
        self.x_combo = self._combo_row(setup_box, 6, "X column", self.x_var, ())
        self.y_combo = self._combo_row(setup_box, 7, "Y column", self.y_var, ())

        self._combo_row(setup_box, 8, "Endurance read order", self.endurance_order_var,
                        ("Read 1 / Read 2", "HRS / LRS", "LRS / HRS"))
        ttk.Label(setup_box, text="Endurance: consecutive read pairs = cycles; |MeasResult2| in A.\nRead 1 = odd rows; Read 2 = even rows. Y scale = linear.",
                  wraplength=330).grid(row=9, column=0, columnspan=2, sticky="w")

        ttk.Label(setup_box, text="Legend size (pt)").grid(row=10, column=0, sticky="w", pady=3)
        ttk.Spinbox(setup_box, from_=1, to=72, increment=0.5, width=7,
                    textvariable=self.legend_size_var).grid(row=10, column=1, sticky="ew", padx=(7, 0))
        self._combo_row(setup_box, 11, "Legend position", self.legend_position_var, LEGEND_POSITIONS)

        for row, label, size_var, bold_var in (
            (12, "Tick number size (pt)", self.tick_size_var, self.tick_bold_var),
            (13, "Axis label size (pt)", self.axis_label_size_var, self.axis_label_bold_var),
            (14, "Title size (pt)", self.title_size_var, self.title_bold_var),
        ):
            ttk.Label(setup_box, text=label).grid(row=row, column=0, sticky="w", pady=3)
            font_options = ttk.Frame(setup_box)
            font_options.grid(row=row, column=1, sticky="ew", padx=(7, 0))
            ttk.Spinbox(font_options, from_=1, to=72, increment=0.5, width=6,
                        textvariable=size_var).pack(side=tk.LEFT)
            ttk.Checkbutton(font_options, text="Bold", variable=bold_var).pack(side=tk.LEFT, padx=(5, 0))

        for row, label, variable in ((15, "X-axis label", self.x_label_var),
                                      (16, "Y-axis label", self.y_label_var)):
            ttk.Label(setup_box, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(setup_box, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(7, 0))
        ttk.Label(setup_box, text="Blank axis labels use the automatic labels and units.\nCustom labels change text only; include the plotted units.",
                  wraplength=330).grid(row=17, column=0, columnspan=2, sticky="w", pady=3)

        legend_actions = ttk.Frame(setup_box)
        legend_actions.grid(row=18, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Checkbutton(legend_actions, text="Show legend", variable=self.show_legend_var).pack(side=tk.LEFT)
        ttk.Button(legend_actions, text="Edit legend names", command=self.edit_legend_names).pack(side=tk.LEFT, padx=(5, 0))

        device_box = ttk.Labelframe(
            controls, text="3. Devices / traces", style="Section.TLabelframe", padding=8
        )
        device_box.pack(fill=tk.BOTH, expand=True, pady=(0, 7))
        device_frame = ttk.Frame(device_box)
        device_frame.pack(fill=tk.BOTH, expand=True)
        self.device_list = tk.Listbox(
            device_frame,
            height=9,
            selectmode=tk.EXTENDED,
            exportselection=False,
            font=("Segoe UI", 9),
        )
        device_scroll = ttk.Scrollbar(
            device_frame, orient=tk.VERTICAL, command=self.device_list.yview
        )
        device_scroll_x = ttk.Scrollbar(
            device_frame, orient=tk.HORIZONTAL, command=self.device_list.xview
        )
        self.device_list.configure(yscrollcommand=device_scroll.set,
                                   xscrollcommand=device_scroll_x.set)
        device_frame.rowconfigure(0, weight=1)
        device_frame.columnconfigure(0, weight=1)
        self.device_list.grid(row=0, column=0, sticky="nsew")
        device_scroll.grid(row=0, column=1, sticky="ns")
        device_scroll_x.grid(row=1, column=0, sticky="ew")
        device_buttons = ttk.Frame(device_box)
        device_buttons.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(device_buttons, text="Select all", command=self.select_all_devices).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(device_buttons, text="Clear selection", command=self.clear_devices).pack(
            side=tk.LEFT
        )

        action_box = ttk.Labelframe(
            control_shell, text="4. Preview and export", style="Section.TLabelframe", padding=8
        )
        action_box.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        options = ttk.Frame(action_box)
        options.pack(fill=tk.X, pady=(0, 6))
        ttk.Checkbutton(options, text="PNG", variable=self.png_var).pack(side=tk.LEFT)
        ttk.Checkbutton(options, text="PDF", variable=self.pdf_var).pack(side=tk.LEFT, padx=(7, 0))
        ttk.Checkbutton(options, text="JPG", variable=self.jpg_var).pack(side=tk.LEFT)
        ttk.Checkbutton(options, text="SVG", variable=self.svg_var).pack(side=tk.LEFT)
        dpi_options = ttk.Frame(action_box)
        dpi_options.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(dpi_options, text="DPI").pack(side=tk.LEFT, padx=(15, 4))
        ttk.Spinbox(dpi_options, from_=100, to=1200, increment=50, width=6, textvariable=self.dpi_var).pack(
            side=tk.LEFT
        )
        actions = ttk.Frame(action_box)
        actions.pack(fill=tk.X)
        ttk.Button(
            actions, text="Preview", command=self.preview_plot, style="Primary.TButton"
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(
            actions, text="Export plots", command=self.export_plots, style="Primary.TButton"
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(action_box, text="Open last output folder", command=self.open_last_output).pack(
            anchor="w", pady=(6, 0)
        )
        # ttk.Label has no "height" option; tk.Label supports a fixed
        # line count, keeping long status messages from pushing actions away.
        tk.Label(
            action_box,
            textvariable=self.status_var,
            foreground="#374151",
            background=ttk.Style(self.root).lookup("TFrame", "background") or "#f0f0f0",
            wraplength=340,
            height=3,
            anchor="w",
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(7, 0))

        ttk.Label(preview, text="Live preview", style="Title.TLabel").pack(anchor="w")
        self.preview_holder = ttk.Frame(preview, relief=tk.SUNKEN, borderwidth=1)
        self.preview_holder.pack(fill=tk.BOTH, expand=True, pady=(7, 0))
        self.preview_placeholder = ttk.Label(
            self.preview_holder,
            text="Your plot preview will appear here.\n\nAdd CSV files, choose a test, and press Preview.",
            anchor="center",
            justify=tk.CENTER,
            foreground="#6B7280",
            font=("Segoe UI", 12),
        )
        self.preview_placeholder.pack(fill=tk.BOTH, expand=True)
        self._update_custom_state()
        self._bind_settings_scroll(controls)
        settings_canvas.bind("<MouseWheel>", self._scroll_settings)
        settings_canvas.bind("<Button-4>", self._scroll_settings)
        settings_canvas.bind("<Button-5>", self._scroll_settings)

    def _bind_settings_scroll(self, widget) -> None:
        if widget.winfo_class() not in ("Listbox", "Treeview", "TCombobox", "TSpinbox"):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(sequence, self._scroll_settings, add="+")
        for child in widget.winfo_children():
            self._bind_settings_scroll(child)

    def _scroll_settings(self, event):
        bounds = self.settings_canvas.bbox("all")
        if not bounds or bounds[3] <= self.settings_canvas.winfo_height():
            return
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            delta = getattr(event, "delta", 0)
            if not delta:
                return
            units = -max(1, int(abs(delta) / 120)) if delta > 0 else max(1, int(abs(delta) / 120))
        self.settings_canvas.yview_scroll(units, "units")
        return "break"

    def _combo_row(self, parent, row, label, variable, values, command=None):
        ttk = self.ttk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", padx=(7, 0), pady=3)
        if command:
            combo.bind("<<ComboboxSelected>>", command)
        return combo

    def set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.root.update_idletasks()

    def add_files(self) -> None:
        from tkinter import filedialog

        paths = filedialog.askopenfilenames(
            title="Select research CSV files",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        self._add_paths(paths)

    def add_folder(self) -> None:
        from tkinter import filedialog

        folder = filedialog.askdirectory(title="Open a folder in Explorer")
        if folder:
            self._open_explorer(folder)

    def toggle_explorer(self) -> None:
        """Unmanage the pane without destroying its tree or loaded folder state."""
        if self.explorer_visible:
            self.explorer_width = self.outer_panes.sashpos(0)
            self.outer_panes.forget(self.explorer_panel)
            self.explorer_visible = False
            self.explorer_toggle.configure(text="Show Explorer")
        else:
            self.outer_panes.insert(0, self.explorer_panel, weight=0)
            self.explorer_visible = True
            self.explorer_toggle.configure(text="Hide Explorer")
            self.root.after_idle(self._restore_explorer_width)

    def _restore_explorer_width(self) -> None:
        if self.explorer_visible:
            available = self.outer_panes.winfo_width()
            self.outer_panes.sashpos(0, min(max(120, self.explorer_width),
                                          max(120, available // 2)))

    def _open_explorer(self, folder) -> None:
        # Opening a workspace never loads measurement data.
        self._explorer_generation += 1
        self._cancel_explorer_selection()
        self.file_tree.column("#0", width=245)
        self._explorer_root = os.path.abspath(folder)
        self._explorer_nodes.clear()
        self._explorer_loaded.clear()
        self._explorer_loading.clear()
        roots = self.file_tree.get_children()
        if roots:
            self.file_tree.delete(*roots)
        node = self.file_tree.insert("", "end", text=os.path.basename(folder) or folder,
                                     open=True)
        self._explorer_nodes[node] = (self._explorer_root, True)
        self._load_explorer_directory(node)
        self.set_status("Folder opened. Double-click a CSV to add it to the current plot.")

    def refresh_explorer(self) -> None:
        if self._explorer_root:
            self._open_explorer(self._explorer_root)

    def _expand_explorer(self, _event=None) -> None:
        self._load_explorer_directory(self.file_tree.focus())

    def _load_explorer_directory(self, node) -> None:
        info = self._explorer_nodes.get(node)
        if not info or not info[1] or node in self._explorer_loaded or node in self._explorer_loading:
            return
        self._explorer_loading.add(node)
        for child in self.file_tree.get_children(node):
            self.file_tree.delete(child)
        self.file_tree.insert(node, "end", text="Loading…")
        messages = queue.Queue()
        generation = self._explorer_generation

        def scan():
            try:
                messages.put((scan_directory(info[0]), None))
            except Exception as error:
                messages.put(([], str(error)))

        threading.Thread(target=scan, daemon=True).start()
        self.root.after(50, lambda: self._poll_explorer(node, generation, messages))

    def _poll_explorer(self, node, generation, messages) -> None:
        if generation != self._explorer_generation or not self.file_tree.exists(node):
            return
        try:
            entries, error = messages.get_nowait()
        except queue.Empty:
            self.root.after(50, lambda: self._poll_explorer(node, generation, messages))
            return
        children = self.file_tree.get_children(node)
        if children:
            self.file_tree.delete(*children)
        if error:
            self._explorer_loading.discard(node)
            self.file_tree.insert(node, "end", text="Cannot read folder (collapse and retry)")
            self.set_status("Cannot read folder: " + error)
            return
        # Insert in small batches so very large directories do not block Tk.
        self._insert_explorer_entries(node, generation, entries, 0)

    def _insert_explorer_entries(self, node, generation, entries, offset) -> None:
        if generation != self._explorer_generation or not self.file_tree.exists(node):
            return
        from tkinter import font as tkfont
        font = tkfont.nametofont("TkDefaultFont")
        depth, parent = 1, node
        while self.file_tree.parent(parent):
            depth += 1
            parent = self.file_tree.parent(parent)
        required_width = int(self.file_tree.column("#0", "width"))
        for name, path, is_directory in entries[offset:offset + 150]:
            required_width = max(required_width, font.measure(name) + 24 * (depth + 1) + 32)
            tags = () if is_directory or name.lower().endswith(".csv") else ("other",)
            child = self.file_tree.insert(node, "end", text=name, tags=tags)
            self._explorer_nodes[child] = (path, is_directory)
            if is_directory:
                self.file_tree.insert(child, "end", text="Expand to browse")
        self.file_tree.column("#0", width=required_width)
        if offset + 150 < len(entries):
            self.root.after(1, lambda: self._insert_explorer_entries(
                node, generation, entries, offset + 150))
        else:
            self._explorer_loading.discard(node)
            self._explorer_loaded.add(node)
            if not entries:
                self.file_tree.insert(node, "end", text="(empty folder)")

    def _cancel_explorer_selection(self) -> None:
        if self._explorer_selection_job is not None:
            self.root.after_cancel(self._explorer_selection_job)
            self._explorer_selection_job = None

    def _selected_explorer_csvs(self):
        paths = []
        for node in self.file_tree.selection():
            info = self._explorer_nodes.get(node)
            if info and not info[1] and info[0].lower().endswith(".csv"):
                paths.append(info[0])
        return paths

    def _double_click_explorer(self, event):
        # Use the clicked row, not a stale focus or the entire multiselection.
        node = self.file_tree.identify_row(event.y)
        info = self._explorer_nodes.get(node)
        if not info or info[1]:
            return  # Keep native folder expand/collapse behavior.
        if info[0].lower().endswith(".csv"):
            self._add_paths([info[0]])
        return "break"

    def add_explorer_selection(self) -> None:
        self._add_paths(self._selected_explorer_csvs())

    def _add_paths(self, paths: Iterable[str]) -> None:
        changed = False
        for path in paths:
            absolute = os.path.abspath(path)
            if absolute not in self.file_paths:
                self.file_paths.append(absolute)
                changed = True
        if not changed:
            return
        self._comparison_paths = list(self.file_paths)
        self.file_list.delete(0, self.tk.END)
        for path in self.file_paths:
            self.file_list.insert(self.tk.END, os.path.basename(path))
        self.reload_sources()

    def clear_files(self) -> None:
        self._cancel_explorer_selection()
        self.cancel_loading()
        self.file_paths.clear()
        self._comparison_paths.clear()
        self.datasets.clear()
        self.catalog.clear()
        self.file_list.delete(0, self.tk.END)
        self.device_list.delete(0, self.tk.END)
        self.set_status("Add one or more CSV files to begin.")

    def on_instrument_changed(self, _event=None) -> None:
        if self.file_paths:
            self.reload_sources()

    def on_test_changed(self, _event=None) -> None:
        default_titles = {
            TEST_IV: "I–V Characteristics",
            TEST_RETENTION: "Retention Characteristics",
            TEST_PULSE: "Pulse Response",
            TEST_XY: "Research Data Plot",
            TEST_PD: "Potentiation / Depression",
            TEST_FORMING: "Forming Characteristics",
            TEST_ENDURANCE: "Endurance Characteristics",
        }
        self.title_var.set(default_titles[self.test_var.get()])
        self._update_custom_state()
        self.refresh_catalog()

    def _update_custom_state(self) -> None:
        state = "readonly" if self.test_var.get() in (TEST_XY, TEST_PD) else "disabled"
        self.x_combo.configure(state=state)
        self.y_combo.configure(state=state)

    def cancel_loading(self) -> None:
        self._load_cancel.set()
        self._load_generation += 1
        self.set_status("Loading cancelled. Previously loaded data are still available.")

    def reload_sources(self) -> None:
        self._start_loading()

    def _start_loading(self, folder=None, preview=False) -> None:
        self._preview_after_load = preview
        self._load_cancel.set()
        self._load_generation += 1
        generation = self._load_generation
        self._load_cancel = threading.Event()
        messages = queue.Queue()
        self.set_status("Starting background import…")
        threading.Thread(
            target=load_sources_worker,
            args=(tuple(self.file_paths), folder, self.instrument_var.get(),
                  self.test_var.get(), self._load_cancel, messages),
            daemon=True,
        ).start()
        self.root.after(75, lambda: self._poll_loading(generation, messages))

    def _poll_loading(self, generation, messages) -> None:
        from tkinter import messagebox

        if generation != self._load_generation:
            return  # Cancelled/replaced jobs cannot overwrite newer UI state.
        latest_progress = None
        for _ in range(100):
            try:
                kind, payload = messages.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                latest_progress = payload
            elif kind == "done":
                self._finish_loading(*payload)
                return
            elif kind == "empty":
                self.set_status("No CSV files found directly in the selected folder.")
                return
            elif kind == "error":
                self.set_status("Import failed: " + payload)
                messagebox.showerror(APP_NAME, payload)
                return
        if latest_progress:
            self.set_status(latest_progress)
        self.root.after(75, lambda: self._poll_loading(generation, messages))

    def _finish_loading(self, paths, loaded, errors, columns, catalog, test) -> None:
        from tkinter import messagebox

        self.file_paths = paths
        self.file_list.delete(0, self.tk.END)
        if paths:
            self.file_list.insert(self.tk.END, *[os.path.basename(p) for p in paths])
        self.datasets = loaded
        if not loaded:
            self.catalog.clear()
            self.device_list.delete(0, self.tk.END)
            self.set_status("No files could be loaded.")
            if errors:
                messagebox.showerror(APP_NAME, "\n\n".join(errors[:10]))
            return
        if self.test_var.get() == TEST_IV and all(
                "__keysight_block" in d.frame.columns and time_column(d.frame)
                and current_column(d.frame) and not voltage_column(d.frame) for d in loaded):
            self.test_var.set(TEST_RETENTION)
            self.title_var.set("Retention Characteristics")
            self._update_custom_state()
            test = TEST_RETENTION
            catalog = build_device_catalog(loaded, TEST_RETENTION)
        if self.test_var.get() == TEST_IV:
            detected = None
            if all(any("endurance" in line.lower() for line in d.metadata[:1]) for d in loaded):
                detected = TEST_ENDURANCE
            elif all(any("forming" in line.lower() for line in d.metadata[:1]) for d in loaded):
                detected = TEST_FORMING
            if detected:
                self.test_var.set(detected)
                self.title_var.set(detected + " Characteristics")
                test = detected
                catalog = build_device_catalog(loaded, test)
                self._update_custom_state()
        if all("__section" in dataset.frame.columns for dataset in loaded) and self.test_var.get() == TEST_IV:
            self.test_var.set(TEST_PD)
            self.title_var.set("Potentiation / Depression")
            self._update_custom_state()
            test = TEST_PD
            catalog = build_device_catalog(loaded, TEST_PD)
        self.x_combo.configure(values=columns)
        self.y_combo.configure(values=columns)
        if columns:
            preferred_x = next(
                (column for column in columns if column == voltage_column(loaded[0].frame)), columns[0]
            )
            preferred_y = next(
                (column for column in columns if column == resistance_column(loaded[0].frame)),
                columns[min(1, len(columns) - 1)],
            )
            if self.test_var.get() == TEST_PD:
                preferred_x = "MeasResult2_time" if "MeasResult2_time" in columns else "Time"
                preferred_y = "MeasResult2_value"
            self.x_var.set(preferred_x)
            self.y_var.set(preferred_y)
        if test != self.test_var.get():
            self.reload_sources()
            return
        self.catalog = catalog
        self.device_list.delete(0, self.tk.END)
        if catalog:
            self.device_list.insert(self.tk.END, *[ref.label for ref in catalog])
            if test == TEST_ENDURANCE:
                seen = set()
                for i, ref in enumerate(catalog):
                    if ref.dataset_index not in seen:
                        self.device_list.selection_set(i)
                        seen.add(ref.dataset_index)
            elif test == TEST_PD:
                selected = [i for i, ref in enumerate(catalog)
                            if "Depression_Pulse" in ref.label or "Potentiation_Pulse" in ref.label]
                for i in selected or [0]:
                    self.device_list.selection_set(i)
            else:
                self.device_list.selection_set(0, self.tk.END)
        instruments = sorted({dataset.instrument for dataset in loaded})
        message = (
            f"Loaded {len(loaded)} file(s), {len(self.catalog)} selectable device(s). "
            f"Detected: {', '.join(instruments)}."
        )
        if errors:
            message += f" {len(errors)} file(s) were skipped."
        self.set_status(message)
        # Explicit preview avoids starting expensive rendering during folder import.
        self.set_status(message + " Select traces and press Preview.")
        if self._preview_after_load and catalog:
            generation = self._load_generation
            self.root.after(20, lambda: self.preview_plot()
                            if generation == self._load_generation else None)

    def refresh_catalog(self) -> None:
        self.catalog = build_device_catalog(self.datasets, self.test_var.get())
        self.device_list.delete(0, self.tk.END)
        for ref in self.catalog:
            self.device_list.insert(self.tk.END, ref.label)
        if self.catalog:
            self.device_list.selection_set(0)
        elif self.datasets:
            self.set_status(
                f"No {self.test_var.get()} rows were detected. Choose another test type."
            )

    def select_all_devices(self) -> None:
        if self.catalog:
            self.device_list.selection_set(0, self.tk.END)

    def clear_devices(self) -> None:
        self.device_list.selection_clear(0, self.tk.END)

    def _selected_references(self) -> list[DeviceRef]:
        indices = list(self.device_list.curselection())
        if not indices and self.catalog:
            indices = [0]
            self.device_list.selection_set(0)
        return [self.catalog[index] for index in indices]

    def _curves(self) -> list[Curve]:
        if not self.datasets:
            raise ValueError("Add at least one CSV file first.")
        references = self._selected_references()
        if not references:
            raise ValueError("No devices or traces are available for this test.")
        curves = build_curves(
            self.datasets,
            references,
            self.test_var.get(),
            current_mode=self.current_mode_var.get(),
            x_column=self.x_var.get(),
            y_column=self.y_var.get(),
            endurance_order=self.endurance_order_var.get(),
        )
        return [replace(c, legend_label=self.legend_names.get((self.test_var.get(), c.label)))
                for c in curves]

    def edit_legend_names(self) -> None:
        from tkinter import messagebox
        try:
            curves = self._curves()
        except Exception as error:
            messagebox.showinfo(APP_NAME, str(error))
            return
        tk, ttk = self.tk, self.ttk
        test = self.test_var.get()
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit legend names")
        dialog.geometry("650x450")
        ttk.Label(dialog, text="Enter a name for each plotted trace. Blank restores its automatic name.",
                  wraplength=600).pack(anchor="w", padx=10, pady=10)
        holder = ttk.Frame(dialog)
        holder.pack(fill=tk.BOTH, expand=True, padx=10)
        canvas = tk.Canvas(holder, highlightthickness=0)
        scroll = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rows = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=rows, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        entries = []
        for i, curve in enumerate(curves, 1):
            ttk.Label(rows, text=f"{i}. {curve.label}", wraplength=570).pack(anchor="w", pady=(7, 2))
            value = tk.StringVar(value=curve.legend_label or "")
            ttk.Entry(rows, textvariable=value).pack(fill=tk.X, padx=(0, 10))
            entries.append((curve.label, value))
        def apply():
            for original, value in entries:
                key = (test, original)
                if value.get().strip():
                    self.legend_names[key] = value.get().strip()
                else:
                    self.legend_names.pop(key, None)
            dialog.destroy()
            self.preview_plot()
        buttons = ttk.Frame(dialog)
        buttons.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(buttons, text="Apply", command=apply).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons, text="Reset names", command=lambda: [v.set("") for _, v in entries]).pack(side=tk.LEFT)
        dialog.transient(self.root)
        dialog.grab_set()

    def preview_plot(self) -> None:
        from tkinter import messagebox
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        try:
            curves = self._curves()
            figure = preview_figure(
                curves,
                self.layout_var.get(),
                self.title_var.get().strip() or "Research Plot",
                self.test_var.get(),
                self.scale_var.get(),
                legend_size=float(self.legend_size_var.get()),
                legend_position=self.legend_position_var.get() if self.show_legend_var.get() else "Hidden",
                tick_size=float(self.tick_size_var.get()),
                axis_label_size=float(self.axis_label_size_var.get()),
                tick_bold=self.tick_bold_var.get(),
                axis_label_bold=self.axis_label_bold_var.get(),
                x_label=self.x_label_var.get(),
                y_label=self.y_label_var.get(),
                title_size=float(self.title_size_var.get()),
                title_bold=self.title_bold_var.get(),
            )
        except Exception as error:
            self.set_status(str(error))
            if not isinstance(error, ValueError):
                messagebox.showerror(APP_NAME, str(error))
            return

        for child in self.preview_holder.winfo_children():
            child.destroy()
        self.canvas = FigureCanvasTkAgg(figure, master=self.preview_holder)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.preview_holder, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=self.tk.BOTTOM, fill=self.tk.X)
        self.canvas.get_tk_widget().pack(side=self.tk.TOP, fill=self.tk.BOTH, expand=True)
        detail = "first page shown" if len(curves) > 12 and self.layout_var.get() != LAYOUT_OVERLAY else ""
        self.set_status(f"Previewing {len(curves)} trace(s){'; ' + detail if detail else ''}.")

    def export_plots(self) -> None:
        from tkinter import messagebox

        formats: list[str] = []
        if self.png_var.get():
            formats.append("png")
        if self.pdf_var.get():
            formats.append("pdf")
        if self.jpg_var.get():
            formats.append("jpg")
        if self.svg_var.get():
            formats.append("svg")
        if not formats:
            messagebox.showinfo(APP_NAME, "Choose at least one export format: PNG, JPG, PDF, or SVG.")
            return
        try:
            curves = self._curves()
            title = self.title_var.get().strip() or "Research Plot"
            parent = os.path.dirname(self.datasets[0].path)
            self.set_status("Exporting plots…")
            directory, saved = export_curves(
                curves,
                self.layout_var.get(),
                title,
                self.test_var.get(),
                self.scale_var.get(),
                formats,
                int(self.dpi_var.get()),
                parent,
                self.file_paths,
                self.instrument_var.get(),
                legend_size=float(self.legend_size_var.get()),
                legend_position=self.legend_position_var.get() if self.show_legend_var.get() else "Hidden",
                tick_size=float(self.tick_size_var.get()),
                axis_label_size=float(self.axis_label_size_var.get()),
                tick_bold=self.tick_bold_var.get(),
                axis_label_bold=self.axis_label_bold_var.get(),
                x_label=self.x_label_var.get(),
                y_label=self.y_label_var.get(),
                title_size=float(self.title_size_var.get()),
                title_bold=self.title_bold_var.get(),
            )
            self.last_output_directory = directory
            self.set_status(f"Saved {len(saved) - 1} plot file(s) in {directory}")
            messagebox.showinfo(APP_NAME, f"Plots saved to:\n{directory}")
        except Exception as error:
            self.set_status(str(error))
            messagebox.showerror(APP_NAME, str(error))

    def open_last_output(self) -> None:
        from tkinter import messagebox

        if not self.last_output_directory or not os.path.isdir(self.last_output_directory):
            messagebox.showinfo(APP_NAME, "Export a plot first.")
            return
        try:
            if os.name == "nt":
                os.startfile(self.last_output_directory)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self.last_output_directory])
            else:
                subprocess.Popen(["xdg-open", self.last_output_directory])
        except Exception as error:
            messagebox.showerror(APP_NAME, str(error))


def run_batch(arguments) -> int:
    instrument_map = {
        "auto": INSTRUMENT_AUTO,
        "arc-one": INSTRUMENT_ARC,
        "keysight": INSTRUMENT_KEYSIGHT,
    }
    test_map = {"iv": TEST_IV, "retention": TEST_RETENTION, "pulse": TEST_PULSE, "pd": TEST_PD, "xy": TEST_XY, "forming": TEST_FORMING, "endurance": TEST_ENDURANCE}
    layout_map = {
        "overlay": LAYOUT_OVERLAY,
        "grid": LAYOUT_GRID,
        "separate": LAYOUT_SEPARATE,
    }
    instrument = instrument_map[arguments.instrument]
    test = test_map[arguments.test]
    layout = layout_map[arguments.layout]
    datasets = [load_dataset(path, instrument) for path in arguments.batch]
    references = build_device_catalog(datasets, test)
    if not references:
        raise ValueError(f"No {test} rows were detected in the selected files.")
    curves = build_curves(
        datasets,
        references,
        test,
        current_mode=arguments.current,
        x_column=arguments.x,
        y_column=arguments.y,
        endurance_order=arguments.endurance_order,
    )
    title = arguments.title or test
    output_parent = arguments.output_parent or os.path.dirname(os.path.abspath(arguments.batch[0]))
    directory, saved = export_curves(
        curves,
        layout,
        title,
        test,
        arguments.scale,
        arguments.formats,
        arguments.dpi,
        output_parent,
        arguments.batch,
        instrument,
        legend_size=arguments.legend_size,
        legend_position=arguments.legend_position,
        tick_size=arguments.tick_size,
        axis_label_size=arguments.axis_label_size,
        tick_bold=arguments.tick_bold,
        axis_label_bold=arguments.axis_label_bold,
        x_label=arguments.x_label,
        y_label=arguments.y_label,
        title_size=arguments.title_size,
        title_bold=arguments.title_bold,
    )
    print(f"Saved {len(saved) - 1} plot file(s) to: {directory}")
    return 0


def parse_arguments(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", nargs="+", metavar="CSV", help="Plot files without opening the GUI")
    parser.add_argument(
        "--instrument", choices=("auto", "arc-one", "keysight"), default="auto"
    )
    parser.add_argument("--test", choices=("iv", "retention", "pulse", "pd", "forming", "endurance"), default="iv")
    parser.add_argument("--endurance-order", choices=("Read 1 / Read 2", "HRS / LRS", "LRS / HRS"), default="Read 1 / Read 2")
    parser.add_argument("--layout", choices=("overlay",), default="overlay")
    parser.add_argument("--title", default="")
    parser.add_argument("--x", default="", help="X column for potentiation/depression")
    parser.add_argument("--y", default="", help="Y column for potentiation/depression")
    parser.add_argument(
        "--current", choices=("Signed current", "Magnitude |I|"), default="Signed current"
    )
    parser.add_argument("--scale", choices=("Linear",), default="Linear")
    parser.add_argument("--formats", nargs="+", choices=("png", "jpg", "pdf", "svg"), default=("png", "pdf"))
    parser.add_argument("--legend-size", type=float, default=6.5)
    parser.add_argument("--legend-position", choices=LEGEND_POSITIONS, default="Below plot")
    parser.add_argument("--tick-size", type=float, default=8)
    parser.add_argument("--axis-label-size", type=float, default=9)
    parser.add_argument("--tick-bold", action="store_true")
    parser.add_argument("--axis-label-bold", action="store_true")
    parser.add_argument("--x-label", default="")
    parser.add_argument("--y-label", default="")
    parser.add_argument("--title-size", type=float, default=10)
    parser.add_argument("--title-bold", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-parent", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    if arguments.batch:
        return run_batch(arguments)
    import tkinter as tk

    root = tk.Tk()
    ResearchPlotterApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
