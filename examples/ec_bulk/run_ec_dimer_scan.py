#!/usr/bin/env python
"""Run an EC dimer distance scan comparing DMFF ADMP and OpenMM plugin energies."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
from openmm.app import PDBFile

from compare_dmff_openmm_energy import (
    DEFAULT_DMFF_XML,
    DEFAULT_OPENMM_XML,
    dmff_energies,
    enable_lrc,
    ensure_box,
    openmm_energies,
    convert_phyneo_to_dmff,
)
from run_neutral_screen import ensure_template_bonds, load_templates


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DEFAULT_MONOMER = PROJECT_ROOT / "1_training_slater_nb" / "pdb_bank" / "EC.pdb"
DEFAULT_OUTPUT = HERE / "dimer_scan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monomer", type=Path, default=DEFAULT_MONOMER)
    parser.add_argument("--dmff-xml", type=Path, default=DEFAULT_DMFF_XML)
    parser.add_argument("--openmm-xml", type=Path, default=DEFAULT_OPENMM_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--distances-a", type=float, nargs="*", default=[3.2, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0])
    parser.add_argument("--box-a", type=float, default=80.0)
    parser.add_argument("--cutoff-nm", type=float, default=2.5)
    parser.add_argument("--platform", default="Reference")
    return parser.parse_args()


def read_atoms(path: Path) -> list[dict[str, object]]:
    atoms = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atoms.append(
            {
                "name": line[12:16],
                "resname": line[17:20],
                "element": (line[76:78].strip() or line[12:16].strip()[0]).rjust(2),
                "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float),
            }
        )
    if not atoms:
        raise ValueError(f"No atoms found in {path}")
    return atoms


def centered_xyz(atoms: list[dict[str, object]]) -> np.ndarray:
    xyz = np.array([atom["xyz"] for atom in atoms], dtype=float)
    return xyz - xyz.mean(axis=0)


def write_pdb(path: Path, atoms: list[dict[str, object]], xyz_blocks: list[np.ndarray], box_a: float) -> None:
    with path.open("w") as handle:
        handle.write(f"CRYST1{box_a:9.3f}{box_a:9.3f}{box_a:9.3f}  90.00  90.00  90.00 P 1           1\n")
        serial = 1
        center = np.array([box_a / 2, box_a / 2, box_a / 2])
        for resid, xyz in enumerate(xyz_blocks, start=1):
            for atom, coord in zip(atoms, xyz + center):
                name = atom["name"]
                resname = atom["resname"]
                element = atom["element"]
                handle.write(
                    f"ATOM  {serial:5d} {name}{resname:>4s} A{resid:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {element}\n"
                )
                serial += 1
        handle.write("END\n")


def min_inter_distance(xyz_a: np.ndarray, xyz_b: np.ndarray) -> float:
    delta = xyz_a[:, None, :] - xyz_b[None, :, :]
    return float(np.sqrt((delta * delta).sum(axis=-1)).min())


def numeric(value: float | str) -> float:
    if isinstance(value, str):
        raise ValueError(value)
    return value


def summarize_terms(rows: dict[str, float | str], engine: str) -> dict[str, float]:
    if engine == "dmff":
        slater_terms = (
            "SlaterExForce",
            "SlaterSrEsForce",
            "SlaterSrPolForce",
            "SlaterSrDispForce",
            "SlaterDhfForce",
            "QqTtDampingForce",
            "SlaterDampingForce",
        )
        electro = numeric(rows["ADMPPmeForce"])
        disp = numeric(rows["ADMPDispPmeForce"])
        slater = sum(numeric(rows[name]) for name in slater_terms)
        return {"electro": electro, "dispersion": disp, "slater": slater, "nonbonded": electro + disp + slater}

    electro = numeric(rows["0:PhyNEOForce"])
    disp = numeric(rows["9:CustomNonbondedForce:pure_disp_lrc"])
    slater = numeric(rows["8:CustomNonbondedForce:slater_short"]) + numeric(rows["10:CustomNonbondedForce:pure_disp_cancel"])
    return {"electro": electro, "dispersion": disp, "slater": slater, "nonbonded": electro + disp + slater}


def interaction(full: dict[str, float], mono_a: dict[str, float], mono_b: dict[str, float]) -> dict[str, float]:
    return {key: full[key] - mono_a[key] - mono_b[key] for key in full}


def run_scan(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atoms = read_atoms(args.monomer)
    xyz0 = centered_xyz(atoms)
    templates, _ = load_templates(args.openmm_xml.resolve())

    with tempfile.TemporaryDirectory(prefix="ec_dimer_scan_") as tmp:
        tmpdir = Path(tmp)
        dmff_xml = convert_phyneo_to_dmff(args.dmff_xml.resolve(), tmpdir / "dmff_admp.xml")
        mono_a_pdb = tmpdir / "ec_a.pdb"
        mono_b_pdb = tmpdir / "ec_b.pdb"
        write_pdb(mono_a_pdb, atoms, [xyz0], args.box_a)
        rows = []
        for distance_a in args.distances_a:
            shift = np.array([distance_a, 0.0, 0.0])
            xyz_a = xyz0 - 0.5 * shift
            xyz_b = xyz0 + 0.5 * shift
            write_pdb(mono_a_pdb, atoms, [xyz_a], args.box_a)
            write_pdb(mono_b_pdb, atoms, [xyz_b], args.box_a)
            dimer_pdb = tmpdir / f"ec_dimer_{distance_a:.2f}.pdb"
            write_pdb(dimer_pdb, atoms, [xyz_a, xyz_b], args.box_a)

            energy_sets = {}
            for label, pdb_path in (("full", dimer_pdb), ("mono_a", mono_a_pdb), ("mono_b", mono_b_pdb)):
                pdb = PDBFile(str(pdb_path))
                ensure_box(pdb, args.box_a * 0.1)
                ensure_template_bonds(pdb.topology, templates)
                energy_sets[("dmff", label)] = summarize_terms(dmff_energies(dmff_xml, pdb, args.cutoff_nm), "dmff")
                energy_sets[("openmm", label)] = summarize_terms(openmm_energies(args.openmm_xml.resolve(), pdb, args.cutoff_nm, args.platform), "openmm")

            dmff_i = interaction(energy_sets[("dmff", "full")], energy_sets[("dmff", "mono_a")], energy_sets[("dmff", "mono_b")])
            openmm_i = interaction(energy_sets[("openmm", "full")], energy_sets[("openmm", "mono_a")], energy_sets[("openmm", "mono_b")])
            row = {
                "com_distance_a": distance_a,
                "min_atom_distance_a": min_inter_distance(xyz_a, xyz_b),
            }
            for term in ("electro", "dispersion", "slater", "nonbonded"):
                row[f"dmff_{term}"] = dmff_i[term]
                row[f"openmm_{term}"] = openmm_i[term]
                row[f"diff_{term}"] = openmm_i[term] - dmff_i[term]
            rows.append(row)

    out = args.output_dir / "ec_dimer_scan.tsv"
    columns = [
        "com_distance_a",
        "min_atom_distance_a",
        "dmff_electro",
        "openmm_electro",
        "diff_electro",
        "dmff_dispersion",
        "openmm_dispersion",
        "diff_dispersion",
        "dmff_slater",
        "openmm_slater",
        "diff_slater",
        "dmff_nonbonded",
        "openmm_nonbonded",
        "diff_nonbonded",
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> None:
    out = run_scan(parse_args())
    print(out)


if __name__ == "__main__":
    main()
