#!/usr/bin/env python3
"""CUDA NPT simulation of 512 water molecules using MPID polarizable force field.

Uses the ForceField XML parser to build the system from DMFF's
ADMPPmeForce/ADMPDispPmeForce parameters, adds short-range custom forces
and undamped C6/C8/C10 dispersion, then runs NPT with trajectory output.

Usage:
    python run_cuda_npt.py                         # default 200 ps
    python run_cuda_npt.py --steps 100000          # 100 ps
    python run_cuda_npt.py --platform Reference    # CPU reference platform
"""

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

import mpidplugin  # noqa: E402
from dmff_sr_custom_forces import (  # noqa: E402
    add_dmff_short_range_forces_from_xml,
    add_undamped_dispersion_force,
)

EXAMPLE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=200000, help="Number of MD steps (default: 200000 = 200 ps)")
    parser.add_argument("--platform", default="CUDA", choices=["CUDA", "Reference", "CPU"],
                        help="OpenMM platform (default: CUDA)")
    parser.add_argument("--dt", type=float, default=1.0, help="Timestep in fs (default: 1.0)")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K (default: 300)")
    parser.add_argument("--pressure", type=float, default=1.0, help="Pressure in atm (default: 1.0)")
    parser.add_argument("--output-dcd", default="traj.dcd", help="DCD trajectory file (default: traj.dcd)")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save trajectory every N steps (default: 1000)")
    parser.add_argument("--report-interval", type=int, default=10000, help="Report energy every N steps (default: 10000)")
    args = parser.parse_args()

    dmff_xml = EXAMPLE_DIR / "ff.backend_amoeba_total1000_classical_intra.xml"
    pdb_path = EXAMPLE_DIR / "02molL_init.pdb"
    residues_xml = EXAMPLE_DIR / "residues.xml"

    app.Topology.loadBondDefinitions(str(residues_xml))
    pdb = app.PDBFile(str(pdb_path))

    # ── Build system from DMFF XML via ForceField parser ──
    ff = app.ForceField(str(dmff_xml))
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.6 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        polarization="extrapolated",
    )
    vectors = pdb.topology.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*vectors)

    # ── Add short-range custom forces (Slater exchange, damping, etc.) ──
    sr_start_group = system.getNumForces()
    add_dmff_short_range_forces_from_xml(
        system, pdb.topology, str(dmff_xml), start_group=sr_start_group,
    )

    # ── Add undamped C6/C8/C10 dispersion with long-range correction ──
    add_undamped_dispersion_force(
        system, pdb.topology, str(dmff_xml), cutoff_nm=0.6,
    )

    # Assign each force to its own group for energy breakdown
    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(min(i, 31))

    # ── Single-point energy ──
    print("=" * 60)
    print("  Single-Point Energy")
    print("=" * 60)

    if args.platform == "CUDA":
        platform = mm.Platform.getPlatformByName("CUDA")
        properties = {"Precision": "mixed"}
    elif args.platform == "Reference":
        platform = mm.Platform.getPlatformByName("Reference")
        properties = {}
    else:
        platform = mm.Platform.getPlatformByName("CPU")
        properties = {}

    integrator_sp = mm.VerletIntegrator(0.001)
    sim_sp = app.Simulation(pdb.topology, system, integrator_sp, platform, properties)
    sim_sp.context.setPositions(pdb.positions)

    state = sim_sp.context.getState(getEnergy=True)
    total_e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    print(f"\nPlatform: {sim_sp.context.getPlatform().getName()}")
    print(f"Number of forces: {system.getNumForces()}")
    for i in range(min(system.getNumForces(), 32)):
        f = system.getForce(i)
        st = sim_sp.context.getState(getEnergy=True, groups={i})
        e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f"  Force {i:2d} ({type(f).__name__:30s}): {e:16.4f} kJ/mol")

    print(f"\n  TOTAL:          {total_e:16.4f} kJ/mol")
    print(f"  DMFF reference: {-10599.4996:16.4f} kJ/mol")
    print(f"  Delta:          {total_e - (-10599.4996):16.4f} kJ/mol")
    del sim_sp

    # ── NPT Simulation ──
    print("\n" + "=" * 60)
    print(f"  NPT Simulation ({args.temperature} K, {args.pressure} atm, {args.steps} steps)")
    print("=" * 60)

    dt = args.dt * unit.femtosecond
    temperature = args.temperature * unit.kelvin
    pressure = args.pressure * unit.atmosphere

    integrator = mm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, dt)
    barostat = mm.MonteCarloBarostat(pressure, temperature, 25)
    system.addForce(barostat)

    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    print("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=200)
    state = simulation.context.getState(getEnergy=True)
    min_e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f"  After minimization: {min_e:.4f} kJ/mol")

    total_time_ps = args.steps * args.dt / 1000.0
    print(f"\nRunning {args.steps} steps ({total_time_ps:.1f} ps)...")

    # Reporters
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout, args.report_interval,
            step=True, time=True,
            potentialEnergy=True, kineticEnergy=True,
            temperature=True, volume=True,
            density=True, speed=True,
        )
    )
    simulation.reporters.append(
        app.DCDReporter(str(EXAMPLE_DIR / args.output_dcd), args.save_interval)
    )

    simulation.step(args.steps)

    state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    box = state.getPeriodicBoxVectors()
    vol = (box[0][0] * box[1][1] * box[2][2]).value_in_unit(unit.nanometer**3)
    print(f"\nFinal energy:     {final_e:.4f} kJ/mol")
    print(f"Final box volume: {vol:.4f} nm^3")
    print(f"Trajectory saved: {args.output_dcd}")
    print("Done.")


if __name__ == "__main__":
    main()
