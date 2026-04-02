#!/usr/bin/env python3
"""Smoke test for the optional intra builder using the local Li-DMC fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import openmm as omm
import openmm.app as app
import openmm.unit as unit

from example_fixture import (  # noqa: E402
    CUTOFF_NM,
    FF_XML,
    PAIR,
    PARAMS_DIR,
    add_local_python_paths,
    build_box,
    resolve_pair_paths,
)

add_local_python_paths(include_workspace=True)

from openmmtool import IntraGromacsForceBuilder, PhyNEOCalculator  # noqa: E402


DEFAULT_PDB = resolve_pair_paths(PAIR)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--xml", type=Path, default=FF_XML)
    parser.add_argument("--params-dir", type=Path, default=PARAMS_DIR)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--cutoff", type=float, default=CUTOFF_NM)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    return parser.parse_args()


def compute_standalone_intra_energy(
    builder: IntraGromacsForceBuilder,
    pdb: app.PDBFile,
    platform_name: str,
) -> tuple[float, dict[str, float], list[dict[str, object]]]:
    intra_system, matched = builder.build_system(pdb.topology, box_vectors=build_box())
    expected_terms = {}
    for index in range(intra_system.getNumForces()):
        force = intra_system.getForce(index)
        force.setForceGroup(index)
    integrator = omm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = omm.Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(pdb.topology, intra_system, integrator, platform)
    simulation.context.setPeriodicBoxVectors(*build_box())
    simulation.context.setPositions(pdb.positions)
    energy = 0.0
    for index in range(intra_system.getNumForces()):
        force = intra_system.getForce(index)
        if not isinstance(force, builder._BONDED_FORCE_TYPES):
            continue
        state = simulation.context.getState(getEnergy=True, groups=2**index)
        force_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        expected_terms[f"Intra{force.getName() or type(force).__name__}"] = force_energy
        energy += force_energy
    return energy, expected_terms, matched


def main() -> None:
    args = parse_args()
    pdb = app.PDBFile(str(args.pdb))
    builder = IntraGromacsForceBuilder(args.params_dir)
    standalone_energy, standalone_terms, matched = compute_standalone_intra_energy(builder, pdb, args.platform)

    calculator = PhyNEOCalculator(
        args.pdb,
        args.xml,
        platform_name=args.platform,
        separate_terms=True,
        use_periodic=True,
        box_vectors=build_box(),
        cutoff=args.cutoff,
        enable_intra=True,
        intra_params_dir=args.params_dir,
    )
    terms, _ = calculator.get_separate_terms()
    intra_terms = {name: energy for name, energy in terms.items() if name.startswith("Intra")}
    intra_energy = float(sum(intra_terms.values()))

    print("Matched components:")
    for component in matched:
        entry = component["entry"]
        print(
            f"  {entry.stem}: residues={component['residue_names']} "
            f"natoms={len(component['atom_indices'])} "
            f"1-4={len(component['pair_recs']['1-4'])} 1-5={len(component['pair_recs']['1-5'])} "
            f"1-6={len(component['pair_recs']['1-6'])}"
        )

    print("\nCombined-system intra terms:")
    for name, energy in sorted(intra_terms.items()):
        print(f"  {name:28s} {energy:16.8f} kcal/mol")

    print("\nStandalone bonded/intra terms:")
    for name, energy in sorted(standalone_terms.items()):
        print(f"  {name:28s} {energy:16.8f} kcal/mol")

    print(f"\nStandalone intra energy: {standalone_energy:.8f} kcal/mol")
    print(f"Combined-system intra sum: {intra_energy:.8f} kcal/mol")
    delta = abs(standalone_energy - intra_energy)
    print(f"Absolute delta: {delta:.8e} kcal/mol")

    if not intra_terms:
        raise RuntimeError("No Intra* force terms were found in the combined system.")
    if delta > args.tolerance:
        raise RuntimeError(
            f"Intra energy mismatch exceeds tolerance: {delta:.8e} > {args.tolerance:.8e}"
        )


if __name__ == "__main__":
    main()
