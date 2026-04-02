#!/usr/bin/env python3
"""System-level smoke test for short MD with enable_intra=True on the local Li-DMC fixture."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
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

from openmmtool import PhyNEOCalculator  # noqa: E402


DEFAULT_PDB = resolve_pair_paths(PAIR)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB)
    parser.add_argument("--xml", type=Path, default=FF_XML)
    parser.add_argument("--params-dir", type=Path, default=PARAMS_DIR)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--cutoff", type=float, default=CUTOFF_NM)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--min-displacement", type=float, default=1.0e-8)
    return parser.parse_args()


def total_energy_kcal(calc: PhyNEOCalculator) -> float:
    state = calc.simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)


def finite_force_norm(calc: PhyNEOCalculator) -> float:
    state = calc.simulation.context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole / unit.angstroms)
    if not np.isfinite(forces).all():
        raise RuntimeError("Encountered non-finite forces during intra MD smoke test.")
    return float(np.linalg.norm(forces))


def main() -> None:
    args = parse_args()
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

    initial_positions = calculator.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    initial_positions = np.asarray(initial_positions.value_in_unit(unit.angstrom))
    initial_energy = total_energy_kcal(calculator)
    initial_force_norm = finite_force_norm(calculator)
    terms_before, _ = calculator.get_separate_terms()
    intra_terms_before = {k: v for k, v in terms_before.items() if k.startswith("Intra")}
    if not intra_terms_before:
        raise RuntimeError("No Intra* force terms were found before MD.")

    calculator.simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)
    calculator.step(args.steps)

    final_state = calculator.simulation.context.getState(getPositions=True, getEnergy=True)
    final_positions = np.asarray(final_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    final_force_norm = finite_force_norm(calculator)
    terms_after, _ = calculator.get_separate_terms()
    intra_terms_after = {k: v for k, v in terms_after.items() if k.startswith("Intra")}

    if not math.isfinite(initial_energy) or not math.isfinite(final_energy):
        raise RuntimeError("Encountered non-finite potential energy during intra MD smoke test.")

    max_displacement = float(np.max(np.linalg.norm(final_positions - initial_positions, axis=1)))
    if max_displacement <= args.min_displacement:
        raise RuntimeError(
            f"MD positions did not change enough: max displacement {max_displacement:.3e} A "
            f"<= threshold {args.min_displacement:.3e} A"
        )

    print("Intra terms before MD:")
    for name, energy in sorted(intra_terms_before.items()):
        print(f"  {name:28s} {energy:16.8f} kcal/mol")

    print("\nIntra terms after MD:")
    for name, energy in sorted(intra_terms_after.items()):
        print(f"  {name:28s} {energy:16.8f} kcal/mol")

    print("\nSystem-level MD smoke summary:")
    print(f"  platform              {args.platform}")
    print(f"  steps                 {args.steps}")
    print(f"  initial_energy        {initial_energy:.8f} kcal/mol")
    print(f"  final_energy          {final_energy:.8f} kcal/mol")
    print(f"  initial_force_norm    {initial_force_norm:.8f} kcal/mol/A")
    print(f"  final_force_norm      {final_force_norm:.8f} kcal/mol/A")
    print(f"  max_displacement      {max_displacement:.8e} A")
    print(f"  n_intra_terms_before  {len(intra_terms_before)}")
    print(f"  n_intra_terms_after   {len(intra_terms_after)}")


if __name__ == "__main__":
    main()
