#!/usr/bin/env python3
"""Plot one batch against SAPT-derived reference components from the scan data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from example_fixture import DATA_FILE, OUTPUT_DIR  # noqa: E402


def patch_jax_pickle_compat() -> None:
    from jax._src import core as jax_core

    if getattr(jax_core.ShapedArray.update, "_phyneo_pickle_compat", False):
        return

    original_update = jax_core.ShapedArray.update

    def compat_update(self, shape=None, dtype=None, weak_type=None, **kwargs):
        kwargs.pop("named_shape", None)
        return original_update(self, shape=shape, dtype=dtype, weak_type=weak_type, **kwargs)

    compat_update._phyneo_pickle_compat = True  # type: ignore[attr-defined]
    jax_core.ShapedArray.update = compat_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="conf_060_Li_DMC")
    parser.add_argument("--batch", default="000")
    parser.add_argument(
        "--compare-csv",
        default=str(OUTPUT_DIR / "li_dmc_reference_compare_batch-000_n12.csv"),
        help="CSV from run_li_dmc_reference_compare.py.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(OUTPUT_DIR / "li_dmc_reference_summary_batch-000_n12.json"),
        help="Summary JSON from run_li_dmc_reference_compare.py.",
    )
    parser.add_argument(
        "--output-prefix",
        default="li_dmc_reference_vs_sapt_batch-000_n12",
        help="Prefix for generated plot/report files.",
    )
    return parser.parse_args()


def load_reference_scan(pair: str, batch: str) -> dict:
    patch_jax_pickle_compat()
    with open(DATA_FILE, "rb") as handle:
        data = pickle.load(handle)
    return data[pair][batch]


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


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "max_abs": float(np.max(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    compare_csv = Path(args.compare_csv)
    summary_json = Path(args.summary_json)
    rows = load_compare_rows(compare_csv, args.batch)
    ref = load_reference_scan(args.pair, args.batch)
    with open(summary_json) as handle:
        summary = json.load(handle)

    npts = len(rows)
    shift = np.array([row["shift"] for row in rows], dtype=float)

    ref_shift = np.asarray(ref["shift"][:npts], dtype=float)
    ref_sapt_es = np.asarray(ref["es"][:npts], dtype=float) + np.asarray(ref["lr_es"][:npts], dtype=float)
    ref_sapt_pol = np.asarray(ref["pol"][:npts], dtype=float) + np.asarray(ref["lr_pol"][:npts], dtype=float)
    ref_sapt_disp = np.asarray(ref["disp"][:npts], dtype=float) + np.asarray(ref["lr_disp"][:npts], dtype=float)
    ref_sapt_espol = ref_sapt_es + ref_sapt_pol
    ref_sapt_total = ref_sapt_espol + ref_sapt_disp

    dmff_pme = np.array([row["dmff_ADMPPmeForce"] for row in rows], dtype=float)
    phyneo_pme = np.array([row["phyneo_ADMPPmeForce"] for row in rows], dtype=float)
    dmff_qqtt = np.array([row["dmff_QqTtDampingForce"] for row in rows], dtype=float)
    phyneo_qqtt = np.array([row["phyneo_QqTtDampingForce"] for row in rows], dtype=float)
    dmff_sr_es = np.array([row["dmff_SlaterSrEsForce"] for row in rows], dtype=float)
    phyneo_sr_es = np.array([row["phyneo_SlaterSrEsForce"] for row in rows], dtype=float)
    dmff_sr_pol = np.array([row["dmff_SlaterSrPolForce"] for row in rows], dtype=float)
    phyneo_sr_pol = np.array([row["phyneo_SlaterSrPolForce"] for row in rows], dtype=float)
    dmff_sr_disp = np.array([row["dmff_SlaterSrDispForce"] for row in rows], dtype=float)
    phyneo_sr_disp = np.array([row["phyneo_SlaterSrDispForce"] for row in rows], dtype=float)
    dmff_damped_disp = np.array([row["dmff_DampedDispersionForce"] for row in rows], dtype=float)
    phyneo_damped_disp = np.array([row["phyneo_DampedDispersionForce"] for row in rows], dtype=float)

    dmff_espol = dmff_pme + dmff_qqtt + dmff_sr_es + dmff_sr_pol
    phyneo_espol = phyneo_pme + phyneo_qqtt + phyneo_sr_es + phyneo_sr_pol
    dmff_disp_total = dmff_damped_disp + dmff_sr_disp
    phyneo_disp_total = phyneo_damped_disp + phyneo_sr_disp
    dmff_total_like = dmff_espol + dmff_disp_total
    phyneo_total_like = phyneo_espol + phyneo_disp_total

    checks = {
        "shift_vs_scan": stats(shift - ref_shift),
        "dmff_espol_vs_sapt": stats(dmff_espol - ref_sapt_espol),
        "phyneo_espol_vs_sapt": stats(phyneo_espol - ref_sapt_espol),
        "dmff_disp_vs_sapt": stats(dmff_disp_total - ref_sapt_disp),
        "phyneo_disp_vs_sapt": stats(phyneo_disp_total - ref_sapt_disp),
        "dmff_total_like_vs_sapt": stats(dmff_total_like - ref_sapt_total),
        "phyneo_total_like_vs_sapt": stats(phyneo_total_like - ref_sapt_total),
        "phyneo_vs_dmff_espol": stats(phyneo_espol - dmff_espol),
        "phyneo_vs_dmff_disp": stats(phyneo_disp_total - dmff_disp_total),
    }

    print(f"Pair: {args.pair}")
    print(f"Batch: {args.batch}")
    print(f"Points: {npts}")
    print("\nReference consistency checks against SAPT-derived scan data")
    for key in ["shift_vs_scan", "dmff_espol_vs_sapt", "phyneo_espol_vs_sapt", "dmff_disp_vs_sapt", "phyneo_disp_vs_sapt"]:
        item = checks[key]
        print(f"  {key:28s} mean={item['mean']:+.6e} max_abs={item['max_abs']:.6e} rmse={item['rmse']:.6e}")

    print("\nAggregate total-like checks")
    for key in ["dmff_total_like_vs_sapt", "phyneo_total_like_vs_sapt", "phyneo_vs_dmff_espol", "phyneo_vs_dmff_disp"]:
        item = checks[key]
        print(f"  {key:28s} mean={item['mean']:+.6f} max_abs={item['max_abs']:.6f} rmse={item['rmse']:.6f}")

    print("\nPer-point values")
    print("  point  shift      sapt_espol    dmff_espol    phyneo_espol   sapt_disp   dmff_disp   phyneo_disp")
    for i in range(npts):
        print(
            f"  {i:>4d}  {shift[i]:>7.4f}  {ref_sapt_espol[i]:>12.6f}  {dmff_espol[i]:>12.6f}  "
            f"{phyneo_espol[i]:>12.6f}  {ref_sapt_disp[i]:>10.6f}  {dmff_disp_total[i]:>10.6f}  {phyneo_disp_total[i]:>11.6f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    ax = axes[0, 0]
    ax.plot(shift, ref_sapt_espol, marker="x", linestyle="--", label="SAPT es+pol")
    ax.plot(shift, dmff_espol, marker="o", label="DMFF es+pol")
    ax.plot(shift, phyneo_espol, marker="s", label="PhyNEO es+pol")
    ax.set_title("ES + Pol")
    ax.set_ylabel("kJ/mol")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(shift, ref_sapt_disp, marker="x", linestyle="--", label="SAPT disp")
    ax.plot(shift, dmff_disp_total, marker="o", label="DMFF disp")
    ax.plot(shift, phyneo_disp_total, marker="s", label="PhyNEO disp")
    ax.set_title("Dispersion")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(shift, phyneo_espol - ref_sapt_espol, marker="s", label="PhyNEO-SAPT es+pol")
    ax.plot(shift, dmff_espol - ref_sapt_espol, marker="o", label="DMFF-SAPT es+pol")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("ES + Pol Delta")
    ax.set_xlabel("Shift")
    ax.set_ylabel("kJ/mol")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(shift, phyneo_disp_total - ref_sapt_disp, marker="s", label="PhyNEO-SAPT disp")
    ax.plot(shift, dmff_disp_total - ref_sapt_disp, marker="o", label="DMFF-SAPT disp")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Dispersion Delta")
    ax.set_xlabel("Shift")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.suptitle(f"{args.pair} batch {args.batch}: validation vs SAPT-derived scan data", fontsize=14)
    fig.tight_layout()

    plot_path = OUTPUT_DIR / f"{args.output_prefix}.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = {
        "pair": args.pair,
        "batch": args.batch,
        "points": npts,
        "source_compare_csv": str(compare_csv),
        "source_summary_json": str(summary_json),
        "summary_from_compare": summary,
        "checks": checks,
    }
    report_path = OUTPUT_DIR / f"{args.output_prefix}.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\nWrote plot: {plot_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
