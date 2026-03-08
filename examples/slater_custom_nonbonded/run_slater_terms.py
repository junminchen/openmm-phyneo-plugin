#!/usr/bin/env python3
"""Evaluate DMFF Slater/QqTt short-range terms with OpenMM custom forces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openmm as mm
from openmm import unit

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from dmff_sr_custom_forces import (  # noqa: E402
    TERM_ORDER,
    build_positions,
    build_system,
    default_bonds,
    default_types,
    dmff_reference_energies,
    energy_by_group,
    load_pdb_types_and_bonds,
    parse_dmff_sr_xml,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default="examples/slater_custom_nonbonded/slater_terms.xml", help="DMFF-style XML containing the short-range sections.")
    parser.add_argument("--topology", choices=["dimer", "chain7"], default="chain7", help="Built-in topology used by the example.")
    parser.add_argument("--pdb", help="Optional PDB file for a real-structure evaluation. Requires residue templates in the XML.")
    parser.add_argument("--types", nargs="+", help="Override atom types; must match the chosen topology atom count.")
    parser.add_argument("--r-start", type=float, default=0.20, help="Start spacing in nm.")
    parser.add_argument("--r-stop", type=float, default=0.50, help="Stop spacing in nm.")
    parser.add_argument("--r-step", type=float, default=0.05, help="Step size in nm.")
    parser.add_argument("--compare-dmff", action="store_true", help="Also evaluate the same terms with DMFF kernels and print deltas.")
    args = parser.parse_args()

    force_data = parse_dmff_sr_xml(args.xml)
    if args.pdb:
        atom_types, bonds, fixed_positions_nm = load_pdb_types_and_bonds(args.pdb, args.xml)
    else:
        atom_types = args.types if args.types is not None else default_types(args.topology)
        bonds = default_bonds(args.topology)
        expected_atoms = 2 if args.topology == "dimer" else 7
        if len(atom_types) != expected_atoms:
            raise ValueError(f"Topology {args.topology} expects {expected_atoms} atom types, got {len(atom_types)}")
        fixed_positions_nm = None

    system = build_system(atom_types, bonds, force_data)
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))

    header = (["frame"] if args.pdb else ["r_nm"]) + TERM_ORDER
    if args.compare_dmff:
        header += [name + "_delta" for name in TERM_ORDER]
    print(" ".join(header))

    if args.pdb:
        context.setPositions(unit.Quantity(fixed_positions_nm, unit.nanometer))
        omm_energies = {name: energy_by_group(context, idx) for idx, name in enumerate(TERM_ORDER)}
        row = [Path(args.pdb).name] + [f"{omm_energies[name]:.12f}" for name in TERM_ORDER]
        if args.compare_dmff:
            ref = dmff_reference_energies(force_data, atom_types, bonds, fixed_positions_nm)
            row += [f"{(omm_energies[name] - ref[name]):.12e}" for name in TERM_ORDER]
        print(" ".join(row))
    else:
        r_value = args.r_start
        while r_value <= args.r_stop + 1e-12:
            positions_nm = build_positions(args.topology, r_value)
            context.setPositions(unit.Quantity(positions_nm, unit.nanometer))
            omm_energies = {name: energy_by_group(context, idx) for idx, name in enumerate(TERM_ORDER)}
            row = [f"{r_value:.3f}"] + [f"{omm_energies[name]:.12f}" for name in TERM_ORDER]
            if args.compare_dmff:
                ref = dmff_reference_energies(force_data, atom_types, bonds, positions_nm)
                row += [f"{(omm_energies[name] - ref[name]):.12e}" for name in TERM_ORDER]
            print(" ".join(row))
            r_value += args.r_step


if __name__ == "__main__":
    main()
