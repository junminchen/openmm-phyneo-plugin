#!/usr/bin/env python3
"""Plot per-component scan curves from run_li_dmc_reference_compare.py outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

COMMON_COMPONENTS = [
    "ADMPPmeForce",
    "QqTtDampingForce",
    "SlaterExForce",
    "SlaterSrEsForce",
    "SlaterSrPolForce",
    "SlaterSrDispForce",
    "SlaterDhfForce",
    "DampedDispersionForce",
    "ShortRangeTotal",
    "PhyNEOTotalComparable",
]

AGGREGATES = [
    "espol",
    "disp_total",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="conf_060_Li_DMC")
    parser.add_argument("--batch", default="000")
    parser.add_argument(
        "--compare-csv",
        default=str(OUTPUT_DIR / "li_dmc_reference_compare_batch-000_n12.csv"),
        help="CSV from run_li_dmc_reference_compare.py",
    )
    parser.add_argument(
        "--output-prefix",
        default="li_dmc_component_scans_batch-000_n12",
        help="Prefix for generated plot/report files.",
    )
    return parser.parse_args()


def load_compare_rows(csv_path: Path, batch: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["batch"] != batch:
                continue
            parsed: dict[str, float | str] = {"batch": row["batch"]}
            for key, value in row.items():
                if key == "batch":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows found for batch {batch} in {csv_path}")
    return rows


def stats(delta: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(delta)),
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def build_series(rows: list[dict[str, float | str]]) -> dict[str, dict[str, np.ndarray]]:
    series: dict[str, dict[str, np.ndarray]] = {}
    shift = np.array([row["shift"] for row in rows], dtype=float)
    series["shift"] = {"x": shift}

    for name in COMMON_COMPONENTS:
        series[name] = {
            "dmff": np.array([row[f"dmff_{name}"] for row in rows], dtype=float),
            "phyneo": np.array([row[f"phyneo_{name}"] for row in rows], dtype=float),
        }

    series["espol"] = {
        "dmff": np.array([row["dmff_espol"] for row in rows], dtype=float),
        "phyneo": np.array([row["phyneo_espol"] for row in rows], dtype=float),
        "sapt": np.array([row["sapt_espol"] for row in rows], dtype=float),
    }
    series["disp_total"] = {
        "dmff": np.array([row["dmff_disp_total"] for row in rows], dtype=float),
        "phyneo": np.array([row["phyneo_disp_total"] for row in rows], dtype=float),
        "sapt": np.array([row["sapt_disp"] for row in rows], dtype=float),
    }
    return series


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_compare_rows(Path(args.compare_csv), args.batch)
    series = build_series(rows)
    shift = series["shift"]["x"]

    plot_order = COMMON_COMPONENTS + AGGREGATES
    ncols = 3
    nrows = int(np.ceil(len(plot_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    report: dict[str, dict[str, float] | str | int] = {
        "pair": args.pair,
        "batch": args.batch,
        "points": len(rows),
        "source_compare_csv": str(Path(args.compare_csv)),
    }

    for ax, name in zip(axes, plot_order):
        dmff = series[name]["dmff"]
        phyneo = series[name]["phyneo"]
        ax.plot(shift, dmff, marker="o", label="DMFF")
        ax.plot(shift, phyneo, marker="s", label="PhyNEO")
        if "sapt" in series[name]:
            ax.plot(shift, series[name]["sapt"], marker="x", linestyle="--", label="SAPT")
        ax.set_title(name)
        ax.grid(True, linestyle="--", alpha=0.35)
        report[name] = stats(phyneo - dmff)
        if "sapt" in series[name]:
            report[f"{name}_phyneo_vs_sapt"] = stats(phyneo - series[name]["sapt"])
            report[f"{name}_dmff_vs_sapt"] = stats(dmff - series[name]["sapt"])

    for ax in axes[len(plot_order):]:
        ax.axis("off")

    for idx in range(max(0, len(axes) - ncols), len(axes)):
        axes[idx].set_xlabel("Shift")
    for idx in range(0, len(axes), ncols):
        axes[idx].set_ylabel("kJ/mol")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle(f"{args.pair} batch {args.batch}: per-component scan comparison", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plot_path = OUTPUT_DIR / f"{args.output_prefix}.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report_path = OUTPUT_DIR / f"{args.output_prefix}.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote plot: {plot_path}")
    print(f"Wrote report: {report_path}")
    for name in plot_order:
        item = report[name]
        if isinstance(item, dict):
            print(
                f"{name:24s} mean={item['mean']:+.6f} "
                f"max_abs={item['max_abs']:.6f} rmse={item['rmse']:.6f}"
            )


if __name__ == "__main__":
    main()
