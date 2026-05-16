#!/usr/bin/env python
"""Run the EC density example with the local force-field XML."""

from __future__ import annotations

import argparse
import builtins
import math
from pathlib import Path
from sys import stdout

import phyneoplugin  # Registers the PhyNEO plugin with OpenMM.
from openmm import *
from openmm.app import *
from openmm.unit import *


HERE = Path(__file__).resolve().parent
EC_BONDS = (
    (0, 1),  # O00-C01
    (1, 2),  # C01-O02
    (1, 5),  # C01-O05
    (2, 3),  # O02-C03
    (5, 4),  # O05-C04
    (3, 4),  # C03-C04
    (3, 6),  # C03-H06
    (3, 7),  # C03-H07
    (4, 8),  # C04-H08
    (4, 9),  # C04-H09
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=HERE / "ec_init.pdb")
    parser.add_argument("--forcefield", type=Path, default=HERE / "ec_forcefield.xml")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--report-interval", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K")
    parser.add_argument("--pressure", type=float, default=1.0, help="Pressure in bar")
    parser.add_argument("--friction", type=float, default=1.0, help="Friction in 1/ps")
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--cutoff-nm", type=float, default=1.0)
    parser.add_argument("--minimize-iterations", type=int, default=2000)
    parser.add_argument("--platform", default="auto", help="auto, CUDA, CPU, or Reference")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--precision", default="mixed", choices=("single", "mixed", "double"))
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def resolve_existing(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ensure_ec_bonds(topology: Topology) -> int:
    existing = {frozenset((a.index, b.index)) for a, b in topology.bonds()}
    added = 0

    for residue in topology.residues():
        if residue.name != "ECA":
            continue
        atoms = list(residue.atoms())
        if len(atoms) != 10:
            raise ValueError(f"Residue {residue} has {len(atoms)} atoms; expected 10 for ECA")

        for first, second in EC_BONDS:
            atom1 = atoms[first]
            atom2 = atoms[second]
            key = frozenset((atom1.index, atom2.index))
            if key not in existing:
                topology.addBond(atom1, atom2)
                existing.add(key)
                added += 1

    return added


def select_platform(name: str, device_index: str, precision: str) -> tuple[Platform, dict[str, str]]:
    requested = [name] if name != "auto" else ["CUDA", "CPU", "Reference"]
    errors = []

    for platform_name in requested:
        try:
            platform = Platform.getPlatformByName(platform_name)
        except Exception as exc:
            errors.append(f"{platform_name}: {exc}")
            continue

        if platform_name == "CUDA":
            return platform, {"DeviceIndex": device_index, "Precision": precision}
        return platform, {}

    raise RuntimeError("No requested OpenMM platform is available: " + "; ".join(errors))


def density_g_per_ml(system: System, state: State) -> float:
    vectors = state.getPeriodicBoxVectors()
    volume = vectors[0][0] * vectors[1][1] * vectors[2][2]
    total_mass = builtins.sum(system.getParticleMass(i).value_in_unit(dalton) for i in range(system.getNumParticles()))
    mass = total_mass * 1.66053906660e-27 * kilogram
    density = mass / volume
    return density.value_in_unit(kilogram / nanometer**3) * 1.0e24


def check_finite_state(context: Context, label: str) -> State:
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole / nanometer)
    if not math.isfinite(energy):
        raise RuntimeError(f"{label} energy is not finite: {energy}")
    if not math.isfinite(float(forces.max())) or not math.isfinite(float(forces.min())):
        raise RuntimeError(f"{label} forces contain non-finite values")
    return state


def main() -> None:
    args = parse_args()
    pdb_path = resolve_existing(args.pdb)
    forcefield_path = resolve_existing(args.forcefield)

    pdb = PDBFile(str(pdb_path))
    added_bonds = ensure_ec_bonds(pdb.topology)
    forcefield = ForceField(str(forcefield_path))

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=args.cutoff_nm * nanometer,
        constraints=HBonds,
    )
    system.addForce(MonteCarloBarostat(args.pressure * bar, args.temperature * kelvin))

    integrator = LangevinMiddleIntegrator(
        args.temperature * kelvin,
        args.friction / picosecond,
        args.timestep_fs * femtoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)

    platform, properties = select_platform(args.platform, args.device_index, args.precision)
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(args.temperature * kelvin, args.seed)

    print("EC density example")
    print(f"PDB: {pdb_path}")
    print(f"Force field: {forcefield_path}")
    print(f"Atoms: {system.getNumParticles()}")
    print(f"Added topology bonds: {added_bonds}")
    print(f"Platform: {platform.getName()}")
    print(f"Steps: {args.steps}")

    if args.minimize_iterations > 0:
        print("Minimizing energy...")
        simulation.minimizeEnergy(maxIterations=args.minimize_iterations)
        check_finite_state(simulation.context, "minimized")

    if args.report_interval > 0:
        simulation.reporters.append(
            StateDataReporter(
                stdout,
                args.report_interval,
                step=True,
                potentialEnergy=True,
                temperature=True,
                density=True,
                speed=True,
            )
        )

    print("Running NPT simulation...")
    if args.steps > 0:
        simulation.step(args.steps)

    final_state = check_finite_state(simulation.context, "final")
    final_state = simulation.context.getState(getEnergy=True)
    final_energy = final_state.getPotentialEnergy()
    final_density = density_g_per_ml(system, final_state)
    print("Run complete")
    print(f"Final energy: {final_energy}")
    print(f"Final density: {final_density:.6f} g/mL")


if __name__ == "__main__":
    main()
