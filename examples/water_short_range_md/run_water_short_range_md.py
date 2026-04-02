#!/usr/bin/env python3
"""Run a small water MD simulation with DMFF-style short-range custom forces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openmm as mm
import openmm.app as app
from openmm import unit

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from dmff_sr_custom_forces import TERM_ORDER, add_dmff_short_range_forces_from_xml  # noqa: E402


def choose_platform(name: str):
    if name == "CUDA":
        return mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"}
    if name == "Reference":
        return mm.Platform.getPlatformByName("Reference"), {}
    if name == "Auto":
        try:
            return mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"}
        except Exception:
            return mm.Platform.getPlatformByName("Reference"), {}
    raise ValueError(f"Unsupported platform {name}")


def report_short_range_energies(context):
    energies = {}
    for group, name in enumerate(TERM_ORDER):
        state = context.getState(getEnergy=True, groups={group})
        energies[name] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return energies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="Auto", choices=["Auto", "Reference", "CUDA"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K.")
    args = parser.parse_args()

    pdb = app.PDBFile("examples/water_short_range_md/water_short_range_dimer.pdb")
    ff = app.ForceField("examples/parameters/tip3p.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.8 * unit.nanometer,
        constraints=app.HBonds,
    )
    add_dmff_short_range_forces_from_xml(
        system,
        pdb.topology,
        "examples/water_short_range_md/water_short_range.xml",
        start_group=0,
    )

    integrator = mm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        1.0 / unit.picosecond,
        1.0 * unit.femtosecond,
    )
    platform, properties = choose_platform(args.platform)
    try:
        simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    except Exception as exc:
        if args.platform == "CUDA":
            raise
        print(f"Platform {platform.getName()} initialization failed, falling back to Reference: {exc}")
        platform = mm.Platform.getPlatformByName("Reference")
        properties = {}
        simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, 1234)

    print(f"Platform: {simulation.context.getPlatform().getName()}")
    before = report_short_range_energies(simulation.context)
    for name in TERM_ORDER:
        print(f"before {name}: {before[name]: .12f} kJ/mol")

    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            200,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            speed=True,
        )
    )
    simulation.step(args.steps)

    after = report_short_range_energies(simulation.context)
    for name in TERM_ORDER:
        print(f"after  {name}: {after[name]: .12f} kJ/mol")


if __name__ == "__main__":
    main()
