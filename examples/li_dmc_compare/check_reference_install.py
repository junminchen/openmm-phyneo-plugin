#!/usr/bin/env python3
"""Check that the Reference-platform PhyNEOForce installation is usable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.app.forcefield as ffmod
from openmm import unit

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from example_fixture import (  # noqa: E402
    DEFAULT_THOLE_WIDTH,
    FF_XML,
    PAIR,
    add_local_python_paths,
    build_box,
    load_local_plugins,
    resolve_pair_paths,
)

add_local_python_paths()

import phyneoforceplugin  # noqa: E402

def main() -> None:
    load_local_plugins()
    print("Package import check")
    print(f"  phyneoforceplugin: {phyneoforceplugin.__file__}")
    print(f"  MPIDForce alias available: {hasattr(phyneoforceplugin, 'MPIDForce')}")
    print(f"  ADMPPmeForce available: {hasattr(phyneoforceplugin, 'ADMPPmeForce')}")

    print("\nParser registration check")
    print(f"  ADMPPmeForce parser: {'ADMPPmeForce' in ffmod.parsers}")
    print(f"  ADMPDispPmeForce parser: {'ADMPDispPmeForce' in ffmod.parsers}")
    print(f"  MPIDForce parser alias: {'MPIDForce' in ffmod.parsers}")

    pdb_path, _, _ = resolve_pair_paths(PAIR)
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(str(FF_XML))
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=2.5 * unit.nanometer,
        polarization="extrapolated",
        defaultTholeWidth=DEFAULT_THOLE_WIDTH,
    )
    system.setDefaultPeriodicBoxVectors(*build_box())
    forces = [system.getForce(i) for i in range(system.getNumForces())]
    print("\nSystem construction check")
    print(f"  force_count: {len(forces)}")
    print(f"  has_admp_pme: {any(phyneoforceplugin.ADMPPmeForce.isinstance(f) for f in forces)}")
    print(f"  force_types: {[type(f).__name__ for f in forces]}")

    platform = mm.Platform.getPlatformByName("Reference")
    integrator = mm.VerletIntegrator(0.001)
    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPeriodicBoxVectors(*build_box())
    sim.context.setPositions(pdb.positions)
    state = sim.context.getState(getEnergy=True)
    print("\nReference platform check")
    print(f"  energy_kj_mol: {state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.8f}")


if __name__ == "__main__":
    main()
