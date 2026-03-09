#!/usr/bin/env python3
"""CUDA NPT simulation of 100 ethanol molecules using MPID polarizable force field.

Usage:
    python make_box.py                     # generate initial PDB first
    python run_cuda_npt.py                 # default 200 ps
    python run_cuda_npt.py --steps 10000   # quick 5 ps test
    python run_cuda_npt.py --platform CPU  # CPU reference
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
FF_XML      = EXAMPLE_DIR.parent / "water_mpid_npt" / "ff.backend_amoeba_total1000_classical_intra.xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps",           type=int,   default=400000, help="MD steps (default: 400000 = 200 ps at 0.5 fs)")
    parser.add_argument("--platform",        default="CUDA", choices=["CUDA", "Reference", "CPU"])
    parser.add_argument("--dt",              type=float, default=0.5,   help="Timestep in fs (default: 0.5)")
    parser.add_argument("--temperature",     type=float, default=298.15,help="Temperature in K (default: 298.15)")
    parser.add_argument("--pressure",        type=float, default=1.0,   help="Pressure in atm (default: 1.0)")
    parser.add_argument("--pdb",             default="ethanol_100mol.pdb", help="Input PDB file")
    parser.add_argument("--output-dcd",      default="traj.dcd")
    parser.add_argument("--save-interval",   type=int,   default=1000)
    parser.add_argument("--report-interval", type=int,   default=10000)
    args = parser.parse_args()

    pdb_path = EXAMPLE_DIR / args.pdb
    if not pdb_path.exists():
        sys.exit(f"PDB not found: {pdb_path}\n  Run:  python make_box.py  first.")

    residues_xml = EXAMPLE_DIR / "residues.xml"
    app.Topology.loadBondDefinitions(str(residues_xml))
    pdb = app.PDBFile(str(pdb_path))

    # ── Build system from DMFF XML ──
    ff = app.ForceField(str(FF_XML))
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        polarization="extrapolated",
    )
    vectors = pdb.topology.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*vectors)

    # ── Short-range custom forces (Slater exchange + hardcore, damping, etc.) ──
    sr_start_group = system.getNumForces()
    add_dmff_short_range_forces_from_xml(
        system, pdb.topology, str(FF_XML), start_group=sr_start_group, s12=0.169,
    )

    # ── Tang-Toennies damped C6/C8/C10 dispersion ──
    cutoff_nm = 0.9
    add_undamped_dispersion_force(
        system, pdb.topology, str(FF_XML), cutoff_nm=cutoff_nm,
    )

    # Assign each force to its own group for energy breakdown
    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(min(i, 31))

    # ── Platform ──
    if args.platform == "CUDA":
        platform   = mm.Platform.getPlatformByName("CUDA")
        properties = {"Precision": "mixed"}
    elif args.platform == "Reference":
        platform   = mm.Platform.getPlatformByName("Reference")
        properties = {}
    else:
        platform   = mm.Platform.getPlatformByName("CPU")
        properties = {}

    # ── Single-point energy breakdown ──
    print("=" * 60)
    print("  Single-Point Energy")
    print("=" * 60)
    integrator_sp = mm.VerletIntegrator(0.001)
    sim_sp = app.Simulation(pdb.topology, system, integrator_sp, platform, properties)
    sim_sp.context.setPositions(pdb.positions)

    state    = sim_sp.context.getState(getEnergy=True)
    total_e  = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f"\nPlatform: {sim_sp.context.getPlatform().getName()}")
    print(f"Number of forces: {system.getNumForces()}")
    for i in range(min(system.getNumForces(), 32)):
        f  = system.getForce(i)
        st = sim_sp.context.getState(getEnergy=True, groups={i})
        e  = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f"  Force {i:2d} ({type(f).__name__:35s}): {e:16.4f} kJ/mol")
    print(f"\n  TOTAL:  {total_e:16.4f} kJ/mol")
    del sim_sp

    # ── NPT Simulation ──
    print("\n" + "=" * 60)
    print(f"  NPT Simulation ({args.temperature} K, {args.pressure} atm, {args.steps} steps)")
    print("=" * 60)

    dt          = args.dt * unit.femtosecond
    temperature = args.temperature * unit.kelvin
    pressure    = args.pressure * unit.atmosphere

    integrator = mm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, dt)
    barostat   = mm.MonteCarloBarostat(pressure, temperature, 25)
    system.addForce(barostat)

    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)

    print("Minimizing energy...")
    # Use tight tolerance and unlimited iterations to resolve close inter-molecular
    # contacts from the PACKMOL initial geometry.
    simulation.minimizeEnergy(tolerance=1.0)  # kJ/mol/nm, converge fully
    state = simulation.context.getState(getEnergy=True)
    min_e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f"  After minimization: {min_e:.4f} kJ/mol")

    # Set velocities AFTER minimization; use 10 K for gentle pre-equilibration
    # to avoid crashing with large forces from residual close contacts.
    simulation.context.setVelocitiesToTemperature(10 * unit.kelvin)

    # Short pre-equilibration ramp: 10 K → 100 K → 298 K, all at small dt
    print("  Pre-equilibrating (10K→298K)...")
    integrator.setStepSize(0.0001 * unit.picosecond)
    for T_ramp in [10, 50, 100, 200, 298.15]:
        integrator.setTemperature(T_ramp * unit.kelvin)
        simulation.context.setVelocitiesToTemperature(T_ramp * unit.kelvin)
        simulation.step(500)
    integrator.setTemperature(temperature)
    integrator.setStepSize(dt)
    state = simulation.context.getState(getEnergy=True)
    pre_e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f"  After pre-equilibration: {pre_e:.4f} kJ/mol")

    total_time_ps = args.steps * args.dt / 1000.0
    print(f"\nRunning {args.steps} steps ({total_time_ps:.1f} ps)...")

    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout, args.report_interval,
            step=True, time=True,
            potentialEnergy=True, kineticEnergy=True,
            temperature=True, volume=True, density=True, speed=True,
        )
    )
    simulation.reporters.append(
        app.DCDReporter(str(EXAMPLE_DIR / args.output_dcd), args.save_interval)
    )

    simulation.step(args.steps)

    state  = simulation.context.getState(getEnergy=True, getPositions=True)
    final_e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    box    = state.getPeriodicBoxVectors()
    vol    = (box[0][0] * box[1][1] * box[2][2]).value_in_unit(unit.nanometer**3)
    dens   = (100 * 46.07 / 1000) / (vol * 6.022e23 * 1e-27)
    print(f"\nFinal energy:     {final_e:.4f} kJ/mol")
    print(f"Final box volume: {vol:.4f} nm^3  ({dens:.4f} g/mL)")
    print(f"Trajectory saved: {args.output_dcd}")
    print("Done.")


if __name__ == "__main__":
    main()
