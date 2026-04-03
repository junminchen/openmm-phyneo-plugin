#!/usr/bin/env python3
"""Compare plugin dispersion energies (LRC/native PME) against DMFF PME on the 100 DMC box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from compare_plugin_vs_dmff import BOX_PDB, CUTOFF_NM, FF_XML, DMFFBoxCalculator, PluginBoxCalculator, box_density_g_per_ml
import openmm.app as app


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=BOX_PDB)
    parser.add_argument("--xml", type=Path, default=FF_XML)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--cutoff", type=float, default=CUTOFF_NM)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    return parser.parse_args()


def plot_dispersion(plot_path: Path, rows: list[dict[str, float | str]]) -> None:
    labels = [str(row["label"]) for row in rows]
    values = [float(row["energy_kj_mol"]) for row in rows]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Dispersion energy (kJ/mol)")
    ax.set_title("100 DMC box: dispersion comparison")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdb = app.PDBFile(str(args.pdb))
    density = box_density_g_per_ml(pdb)
    molecule_count = sum(1 for _ in pdb.topology.residues())
    atom_count = sum(1 for _ in pdb.topology.atoms())

    plugin_lrc_terms, _ = PluginBoxCalculator(args.xml, args.pdb, args.cutoff, args.platform, "lrc").compute()
    plugin_native_terms, _ = PluginBoxCalculator(args.xml, args.pdb, args.cutoff, args.platform, "native_pme").compute()
    _, dmff_raw = DMFFBoxCalculator(args.xml, args.pdb, args.cutoff).compute_comparable("native_pme")

    plugin_lrc_disp = float(plugin_lrc_terms.get("DampedDispersionForce", 0.0))
    plugin_native_disp = float(plugin_native_terms.get("ADMPPmeForce", 0.0) - plugin_lrc_terms.get("ADMPPmeForce", 0.0))
    dmff_disp_pme = float(dmff_raw["ADMPDispPmeForce"])

    rows = [
        {"label": "Plugin LRC", "energy_kj_mol": plugin_lrc_disp, "delta_vs_dmff_kj_mol": plugin_lrc_disp-dmff_disp_pme},
        {"label": "Plugin native PME", "energy_kj_mol": plugin_native_disp, "delta_vs_dmff_kj_mol": plugin_native_disp-dmff_disp_pme},
        {"label": "DMFF PME", "energy_kj_mol": dmff_disp_pme, "delta_vs_dmff_kj_mol": 0.0},
    ]

    payload = {
        "pdb": str(args.pdb),
        "xml": str(args.xml),
        "platform": args.platform,
        "cutoff_nm": args.cutoff,
        "atom_count": atom_count,
        "molecule_count": molecule_count,
        "box_density_g_ml": density,
        "plugin_lrc_admp_kj_mol": float(plugin_lrc_terms.get("ADMPPmeForce", 0.0)),
        "plugin_native_admp_kj_mol": float(plugin_native_terms.get("ADMPPmeForce", 0.0)),
        "note": "Plugin native PME dispersion is extracted as ADMPPmeForce(native_pme) - ADMPPmeForce(lrc).",
        "rows": rows,
    }

    json_path = args.output_dir / "dmc_100mol_dispersion_modes.json"
    plot_path = args.output_dir / "dmc_100mol_dispersion_modes.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_dispersion(plot_path, rows)

    print(f"pdb: {args.pdb}")
    print(f"atom_count: {atom_count}")
    print(f"molecule_count: {molecule_count}")
    print(f"box_density_g_ml: {density:.6f}")
    print("\nDispersion comparison (kJ/mol):")
    for row in rows:
        print(
            f"  {row['label']:20s} "
            f"Energy={row['energy_kj_mol']:16.8f} "
            f"Delta_vs_DMFF={row['delta_vs_dmff_kj_mol']:16.8f}"
        )
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
