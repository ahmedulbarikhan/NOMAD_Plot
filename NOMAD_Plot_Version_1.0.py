#!/usr/bin/env python3
"""Research Plotter: a small Windows desktop app for Arc One and lab CSV data.

Run the desktop app:
    python research_plotter.py

Run without the GUI (useful for repeatable batch plots):
    python research_plotter.py --batch data.csv --instrument arc-one --test iv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.figure import Figure


APP_NAME = "NOMAD_Plot"
VERSION = "0.1.0"

INSTRUMENT_AUTO = "Auto-detect"
INSTRUMENT_ARC = "Arc One"
INSTRUMENT_KEYSIGHT = "Keysight"

TEST_IV = "Curve Tracer (I–V)"
TEST_RETENTION = "Retention"
TEST_PULSE = "Pulse / Resistance"
TEST_XY = "Custom X–Y"

LAYOUT_OVERLAY = "One graph (overlay)"
LAYOUT_GRID = "Grid of subplots"
LAYOUT_SEPARATE = "Separate plot files"

ALL_TESTS = (TEST_IV, TEST_RETENTION, TEST_PULSE, TEST_XY)
ALL_LAYOUTS = (LAYOUT_OVERLAY, LAYOUT_GRID, LAYOUT_SEPARATE)


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


@dataclass
class Curve:
    label: str
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str
    kind: str = "raw"


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
        ),
        contains=("voltage",),
    )


def current_column(frame: pd.DataFrame) -> Optional[str]:
    return find_column(
        frame,
        exact=("Current (A)", "Current", "I (A)", "I"),
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


def load_generic_csv(path: str) -> DataSet:
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
    tags = _tag_values(frame)
    if test == TEST_IV:
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
        if time and resistance:
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
        wl_column = wordline_column(frame)
        bl_column = bitline_column(frame)
        prefix = f"{dataset.filename} | " if use_prefix else ""
        if wl_column and bl_column:
            device_rows = available[[wl_column, bl_column]].dropna().drop_duplicates()
            for wl, bl in device_rows.itertuples(index=False, name=None):
                label = f"{prefix}WL{format_device_value(wl)}–BL{format_device_value(bl)}"
                catalog.append(DeviceRef(dataset_index, wl, bl, label))
        else:
            catalog.append(DeviceRef(dataset_index, None, None, prefix + dataset.filename))
    return catalog


def subset_for_device(dataset: DataSet, ref: DeviceRef) -> pd.DataFrame:
    frame = dataset.frame
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
    return output


def _contiguous_retention_runs(frame: pd.DataFrame, mask: pd.Series) -> list[pd.DataFrame]:
    positions = np.flatnonzero(mask.to_numpy())
    if not len(positions):
        return []
    split_at = np.flatnonzero(np.diff(positions) > 1) + 1
    groups = np.split(positions, split_at)
    return [frame.iloc[group] for group in groups if len(group)]


def _retention_curves(dataset: DataSet, ref: DeviceRef) -> list[Curve]:
    frame = subset_for_device(dataset, ref)
    mask = rows_for_test(frame, TEST_RETENTION)
    runs = _contiguous_retention_runs(frame, mask)
    resistance = resistance_column(frame)
    if not resistance:
        raise ValueError(f"{dataset.filename}: no resistance column was found.")
    time = time_column(frame)
    output: list[Curve] = []
    for index, run in enumerate(runs, start=1):
        if time:
            x = pd.to_numeric(run[time], errors="coerce").to_numpy(dtype=float)
            x_label = str(time)
        else:
            x = np.arange(len(run), dtype=float)
            x_label = "Reading index"
        y = pd.to_numeric(run[resistance], errors="coerce").to_numpy(dtype=float)
        x, y = _valid_xy(x, y)
        if not len(x):
            continue
        label = ref.label if len(runs) == 1 else f"{ref.label} / run {index}"
        output.append(Curve(label, x, y, x_label, "Resistance (Ω)", "resistance_ohm"))
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
) -> list[Curve]:
    curves: list[Curve] = []
    for ref in references:
        dataset = datasets[ref.dataset_index]
        if test == TEST_IV:
            curves.extend(_iv_curves(dataset, ref, current_mode))
        elif test == TEST_RETENTION:
            curves.extend(_retention_curves(dataset, ref))
        elif test == TEST_PULSE:
            curves.extend(_pulse_curves(dataset, ref))
        elif test == TEST_XY:
            curves.extend(_xy_curves(dataset, ref, x_column, y_column))
    curves = [curve for curve in curves if len(curve.x) and len(curve.y)]
    if not curves:
        raise ValueError("No plottable numeric data were found for the current selection.")
    return _scaled_curves(curves)


def _resolved_scale(requested: str, test: str) -> str:
    if requested != "Auto":
        return requested.lower()
    return "log" if test == TEST_RETENTION else "linear"


def _apply_scale(axis, curves: Sequence[Curve], requested: str, test: str) -> None:
    scale = _resolved_scale(requested, test)
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
    axis.grid(True, which="major", color="#C7CCD1", linewidth=0.65, alpha=0.75)
    axis.grid(True, which="minor", color="#E4E7EA", linewidth=0.45, alpha=0.65)
    axis.minorticks_on()
    _apply_scale(axis, curves, scale, test)
    if test == TEST_IV:
        axis.axhline(0, color="#6B7280", linewidth=0.7, zorder=0)
        axis.axvline(0, color="#6B7280", linewidth=0.7, zorder=0)


def _plot_curve(axis, curve: Curve, color: Optional[object] = None) -> None:
    marker = "o" if len(curve.x) <= 120 else None
    axis.plot(
        curve.x,
        curve.y,
        label=curve.label,
        color=color,
        linewidth=1.35,
        marker=marker,
        markersize=2.6,
        markeredgewidth=0,
    )


def overlay_figure(curves: Sequence[Curve], title: str, test: str, scale: str) -> Figure:
    figure = Figure(figsize=(9.2, 6.2), dpi=110, facecolor="white", constrained_layout=True)
    axis = figure.add_subplot(111)
    colors = colormaps["tab20"](np.linspace(0, 1, max(1, len(curves))))
    for curve, color in zip(curves, colors):
        _plot_curve(axis, curve, color)
    _style_axis(axis, curves, test, scale)
    axis.set_title(title, fontweight="semibold", pad=10)
    if len(curves) <= 24:
        axis.legend(
            loc="best" if len(curves) <= 6 else "upper left",
            bbox_to_anchor=None if len(curves) <= 6 else (1.01, 1),
            fontsize=7.5,
            frameon=False,
        )
    else:
        axis.text(
            1.01,
            1.0,
            f"{len(curves)} traces\n(legend hidden)",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
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
            figsize=(4.2 * columns, 3.2 * rows),
            dpi=105,
            facecolor="white",
            constrained_layout=True,
        )
        axes = figure.subplots(rows, columns, squeeze=False)
        for axis, curve in zip(axes.flat, page):
            _plot_curve(axis, curve, "#0065A8")
            _style_axis(axis, [curve], test, scale)
            axis.set_title(curve.label, fontsize=9, fontweight="semibold")
        for axis in axes.flat[len(page) :]:
            axis.remove()
        suffix = f" — page {page_index + 1}/{page_count}" if page_count > 1 else ""
        figure.suptitle(title + suffix, fontsize=13, fontweight="semibold")
        figures.append(figure)
    return figures


def single_figure(curve: Curve, title: str, test: str, scale: str) -> Figure:
    figure = Figure(figsize=(7.2, 5.2), dpi=110, facecolor="white", constrained_layout=True)
    axis = figure.add_subplot(111)
    _plot_curve(axis, curve, "#0065A8")
    _style_axis(axis, [curve], test, scale)
    axis.set_title(title, fontweight="semibold", pad=10)
    return figure


def preview_figure(
    curves: Sequence[Curve], layout: str, title: str, test: str, scale: str
) -> Figure:
    if layout == LAYOUT_OVERLAY:
        return overlay_figure(curves, title, test, scale)
    # Grid is also the clearest contact-sheet preview for separate-file export.
    return grid_figures(curves, title, test, scale, per_page=12)[0]


def safe_name(value: str, fallback: str = "plot") -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", value).strip(" ._")
    cleaned = re.sub(r"\s+", " ", cleaned)
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
        save(overlay_figure(curves, title, test, scale), title)
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
        "trace_count": len(curves),
        "traces": [curve.label for curve in curves],
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
            converted = pd.to_numeric(dataset.frame[column], errors="coerce")
            if converted.notna().any() and column not in names:
                names.append(column)
    return names


class ResearchPlotterApp:
    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title(f"{APP_NAME} {VERSION}")
        self.root.geometry("1320x840")
        self.root.minsize(1040, 680)

        self.file_paths: list[str] = []
        self.datasets: list[DataSet] = []
        self.catalog: list[DeviceRef] = []
        self.canvas = None
        self.toolbar = None
        self.last_output_directory = ""

        self.instrument_var = tk.StringVar(value=INSTRUMENT_ARC)
        self.test_var = tk.StringVar(value=TEST_IV)
        self.layout_var = tk.StringVar(value=LAYOUT_OVERLAY)
        self.current_mode_var = tk.StringVar(value="Signed current")
        self.scale_var = tk.StringVar(value="Auto")
        self.title_var = tk.StringVar(value="Arc One I–V Curves")
        self.x_var = tk.StringVar(value="")
        self.y_var = tk.StringVar(value="")
        self.png_var = tk.BooleanVar(value=True)
        self.pdf_var = tk.BooleanVar(value=True)
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
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        controls = ttk.Frame(outer, width=375, padding=(4, 2))
        preview = ttk.Frame(outer, padding=(8, 2))
        outer.add(controls, weight=0)
        outer.add(preview, weight=1)

        ttk.Label(controls, text=APP_NAME, style="Title.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(
            controls,
            text="Arc One and Keysight-ready research plotting",
            foreground="#4B5563",
        ).pack(anchor="w", pady=(0, 9))

        source_box = ttk.Labelframe(
            controls, text="1. Data files", style="Section.TLabelframe", padding=8
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
        ttk.Button(source_buttons, text="Add folder", command=self.add_folder).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(source_buttons, text="Clear", command=self.clear_files).pack(side=tk.LEFT)

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
            setup_box, 4, "Y scale", self.scale_var, ("Auto", "Linear", "Log", "SymLog")
        )
        ttk.Label(setup_box, text="Title").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Entry(setup_box, textvariable=self.title_var).grid(
            row=5, column=1, sticky="ew", padx=(7, 0), pady=3
        )
        self.x_combo = self._combo_row(setup_box, 6, "Custom X", self.x_var, ())
        self.y_combo = self._combo_row(setup_box, 7, "Custom Y", self.y_var, ())

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
        self.device_list.configure(yscrollcommand=device_scroll.set)
        self.device_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        device_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        device_buttons = ttk.Frame(device_box)
        device_buttons.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(device_buttons, text="Select all", command=self.select_all_devices).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(device_buttons, text="Clear selection", command=self.clear_devices).pack(
            side=tk.LEFT
        )

        action_box = ttk.Labelframe(
            controls, text="4. Preview and export", style="Section.TLabelframe", padding=8
        )
        action_box.pack(fill=tk.X)
        options = ttk.Frame(action_box)
        options.pack(fill=tk.X, pady=(0, 6))
        ttk.Checkbutton(options, text="PNG", variable=self.png_var).pack(side=tk.LEFT)
        ttk.Checkbutton(options, text="PDF", variable=self.pdf_var).pack(side=tk.LEFT, padx=(7, 0))
        ttk.Label(options, text="DPI").pack(side=tk.LEFT, padx=(15, 4))
        ttk.Spinbox(options, from_=100, to=1200, increment=50, width=6, textvariable=self.dpi_var).pack(
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
        ttk.Label(
            action_box,
            textvariable=self.status_var,
            foreground="#374151",
            wraplength=340,
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
        from tkinter import filedialog, messagebox

        folder = filedialog.askdirectory(title="Choose a folder containing CSV files")
        if not folder:
            return
        found: list[str] = []
        for current, directories, files in os.walk(folder):
            directories.sort()
            for filename in sorted(files):
                if filename.lower().endswith(".csv"):
                    found.append(os.path.join(current, filename))
        if not found:
            messagebox.showinfo(APP_NAME, "No CSV files were found in that folder or its subfolders.")
            return
        self._add_paths(found)

    def _add_paths(self, paths: Iterable[str]) -> None:
        changed = False
        for path in paths:
            absolute = os.path.abspath(path)
            if absolute not in self.file_paths:
                self.file_paths.append(absolute)
                changed = True
        if not changed:
            return
        self.file_list.delete(0, self.tk.END)
        for path in self.file_paths:
            self.file_list.insert(self.tk.END, os.path.basename(path))
        self.reload_sources()

    def clear_files(self) -> None:
        self.file_paths.clear()
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
        }
        self.title_var.set(default_titles[self.test_var.get()])
        self._update_custom_state()
        self.refresh_catalog()

    def _update_custom_state(self) -> None:
        state = "readonly" if self.test_var.get() == TEST_XY else "disabled"
        self.x_combo.configure(state=state)
        self.y_combo.configure(state=state)

    def reload_sources(self) -> None:
        from tkinter import messagebox

        self.set_status("Reading data files…")
        loaded: list[DataSet] = []
        errors: list[str] = []
        for path in self.file_paths:
            try:
                loaded.append(load_dataset(path, self.instrument_var.get()))
            except Exception as error:
                errors.append(f"{os.path.basename(path)}: {error}")
        self.datasets = loaded
        if not loaded:
            self.catalog.clear()
            self.device_list.delete(0, self.tk.END)
            self.set_status("No files could be loaded.")
            if errors:
                messagebox.showerror(APP_NAME, "\n\n".join(errors))
            return
        columns = numeric_columns(loaded)
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
            self.x_var.set(preferred_x)
            self.y_var.set(preferred_y)
        self.refresh_catalog()
        instruments = sorted({dataset.instrument for dataset in loaded})
        message = (
            f"Loaded {len(loaded)} file(s), {len(self.catalog)} selectable device(s). "
            f"Detected: {', '.join(instruments)}."
        )
        if errors:
            message += f" {len(errors)} file(s) were skipped."
        self.set_status(message)
        self.root.after(80, self.preview_plot)

    def refresh_catalog(self) -> None:
        self.catalog = build_device_catalog(self.datasets, self.test_var.get())
        self.device_list.delete(0, self.tk.END)
        for ref in self.catalog:
            self.device_list.insert(self.tk.END, ref.label)
        if self.catalog:
            self.device_list.selection_set(0)
        elif self.datasets:
            self.set_status(
                f"No {self.test_var.get()} rows were detected. Choose another test or Custom X–Y."
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
        return build_curves(
            self.datasets,
            references,
            self.test_var.get(),
            current_mode=self.current_mode_var.get(),
            x_column=self.x_var.get(),
            y_column=self.y_var.get(),
        )

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
        if not formats:
            messagebox.showinfo(APP_NAME, "Choose PNG, PDF, or both.")
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
    test_map = {"iv": TEST_IV, "retention": TEST_RETENTION, "pulse": TEST_PULSE, "xy": TEST_XY}
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
    )
    print(f"Saved {len(saved) - 1} plot file(s) to: {directory}")
    return 0


def parse_arguments(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", nargs="+", metavar="CSV", help="Plot files without opening the GUI")
    parser.add_argument(
        "--instrument", choices=("auto", "arc-one", "keysight"), default="auto"
    )
    parser.add_argument("--test", choices=("iv", "retention", "pulse", "xy"), default="iv")
    parser.add_argument("--layout", choices=("overlay", "grid", "separate"), default="overlay")
    parser.add_argument("--title", default="")
    parser.add_argument("--x", default="", help="X column for --test xy")
    parser.add_argument("--y", default="", help="Y column for --test xy")
    parser.add_argument(
        "--current", choices=("Signed current", "Magnitude |I|"), default="Signed current"
    )
    parser.add_argument("--scale", choices=("Auto", "Linear", "Log", "SymLog"), default="Auto")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
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
