#!/usr/bin/env python3
"""Compare PME and NoCutoff single-point term energies on a packed DMC box."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_mpid84")

import matplotlib.pyplot as plt
import openmm.app as app
import openmm.unit as unit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
FIXTURE_ROOT = HERE.parent / "li_dmc_compare" / "fixtures"
BOX_PDB = HERE / "dmc_100mol_box.pdb"
FF_XML = FIXTURE_ROOT / "phyneo_ecl.xml"
PARAMS_DIR = FIXTURE_ROOT / "params_results"

import sys

for candidate in (REPO_ROOT, REPO_ROOT / "python"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from openmmtool import PhyNEOCalculator  # noqa: E402

DMC_MOLAR_MASS_G_PER_MOL = 90.07794
AVOGADRO = 6.02214076e23


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=BOX_PDB)
    parser.add_argument("--xml", type=Path, default=FF_XML)
    parser.add_argument("--params-dir", type=Path, default=PARAMS_DIR)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--cutoff", type=float, default=1.2)
    parser.add_argument("--periodic", action="store_true", default=True)
    parser.add_argument("--enable-intra", action="store_true", default=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    return parser.parse_args()


def strip_cryst1(pdb_path: Path) -> Path:
    lines = pdb_path.read_text(encoding="utf-8").splitlines()
    stripped = [line for line in lines if not line.startswith("CRYST1")]
    handle = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False, encoding="utf-8")
    with handle:
        handle.write("\n".join(stripped) + "\n")
    return Path(handle.name)


def box_density_g_per_ml(pdb: app.PDBFile) -> float:
    residues = list(pdb.topology.residues())
    dimensions = pdb.topology.getUnitCellDimensions()
    if dimensions is None:
        raise ValueError("Periodic box vectors are required to compute density.")
    a_nm, b_nm, c_nm = dimensions.value_in_unit(unit.nanometer)
    volume_nm3 = float(a_nm * b_nm * c_nm)
    volume_cm3 = volume_nm3 * 1.0e-21
    mass_g = len(residues) * DMC_MOLAR_MASS_G_PER_MOL / AVOGADRO
    return float(mass_g / volume_cm3)


def compute_terms(
    pdb_path: Path,
    xml_path: Path,
    *,
    platform: str,
    cutoff: float,
    use_periodic: bool,
    enable_intra: bool,
    params_dir: Path,
) -> tuple[dict[str, float], float]:
    calc = PhyNEOCalculator(
        pdb_path,
        xml_path,
        platform_name=platform,
        separate_terms=True,
        use_periodic=use_periodic,
        cutoff=cutoff,
        enable_intra=enable_intra,
        intra_params_dir=params_dir,
    )
    terms, _ = calc.get_separate_terms()
    total = calc.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilocalorie_per_mole
    )
    return {name: float(val) for name, val in terms.items()}, float(total)


def write_csv(csv_path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = ["term", "pme_kcal_mol", "nocutoff_kcal_mol", "delta_kcal_mol", "delta_per_molecule_kcal_mol"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_terms(plot_path: Path, rows: list[dict[str, float | str]]) -> None:
    terms = [str(row["term"]) for row in rows]
    deltas = [float(row["delta_kcal_mol"]) for row in rows]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    x = range(len(terms))
    pme = [float(row["pme_kcal_mol"]) for row in rows]
    nocut = [float(row["nocutoff_kcal_mol"]) for row in rows]

    ax0.plot(x, pme, marker="o", label="PME")
    ax0.plot(x, nocut, marker="s", label="NoCutoff")
    ax0.set_xticks(list(x))
    ax0.set_xticklabels(terms, rotation=45, ha="right")
    ax0.set_ylabel("Energy (kcal/mol)")
    ax0.set_title("100 DMC box: term energies")
    ax0.legend()

    ax1.bar(list(x), deltas)
    ax1.axhline(0.0, color="black", linewidth=1.0)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(terms, rotation=45, ha="right")
    ax1.set_ylabel("PME - NoCutoff (kcal/mol)")
    ax1.set_title("100 DMC box: term deltas")

    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.pdb.exists():
        raise FileNotFoundError(f"Missing input PDB: {args.pdb}. Run make_dmc_box.py first.")

    periodic_pdb = app.PDBFile(str(args.pdb))
    density = box_density_g_per_ml(periodic_pdb)
    atom_count = sum(1 for _ in periodic_pdb.topology.atoms())
    molecule_count = sum(1 for _ in periodic_pdb.topology.residues())

    boxless_pdb = strip_cryst1(args.pdb)
    try:
        pme_terms, pme_total = compute_terms(
            args.pdb,
            args.xml,
            platform=args.platform,
            cutoff=args.cutoff,
            use_periodic=True,
            enable_intra=args.enable_intra,
            params_dir=args.params_dir,
        )
        nocut_terms, nocut_total = compute_terms(
            boxless_pdb,
            args.xml,
            platform=args.platform,
            cutoff=args.cutoff,
            use_periodic=False,
            enable_intra=args.enable_intra,
            params_dir=args.params_dir,
        )
    finally:
        boxless_pdb.unlink(missing_ok=True)

    all_terms = sorted(set(pme_terms) | set(nocut_terms) | {"TotalPotentialEnergy"})
    rows: list[dict[str, float | str]] = []
    for term in all_terms:
        pme_val = pme_total if term == "TotalPotentialEnergy" else pme_terms.get(term, 0.0)
        nocut_val = nocut_total if term == "TotalPotentialEnergy" else nocut_terms.get(term, 0.0)
        delta = pme_val - nocut_val
        rows.append(
            {
                "term": term,
                "pme_kcal_mol": pme_val,
                "nocutoff_kcal_mol": nocut_val,
                "delta_kcal_mol": delta,
                "delta_per_molecule_kcal_mol": delta / molecule_count,
            }
        )

    json_path = args.output_dir / "dmc_100mol_pme_vs_nocutoff.json"
    csv_path = args.output_dir / "dmc_100mol_pme_vs_nocutoff.csv"
    plot_path = args.output_dir / "dmc_100mol_pme_vs_nocutoff.png"

    payload = {
        "pdb": str(args.pdb),
        "xml": str(args.xml),
        "platform": args.platform,
        "cutoff_nm": args.cutoff,
        "enable_intra": args.enable_intra,
        "atom_count": atom_count,
        "molecule_count": molecule_count,
        "box_density_g_ml": density,
        "note": "This is a same-geometry periodic PME vs nonperiodic NoCutoff comparison, not an absolute bulk truth benchmark.",
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    plot_terms(plot_path, rows)

    print(f"pdb: {args.pdb}")
    print(f"atom_count: {atom_count}")
    print(f"molecule_count: {molecule_count}")
    print(f"box_density_g_ml: {density:.6f}")
    print("\nTerm comparison (kcal/mol):")
    for row in rows:
        print(
            f"  {row['term']:28s} "
            f"PME={row['pme_kcal_mol']:14.6f} "
            f"NoCutoff={row['nocutoff_kcal_mol']:14.6f} "
            f"Delta={row['delta_kcal_mol']:14.6f}"
        )
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
