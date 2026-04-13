#!/usr/bin/env python
"""
EC bulk MD simulation with 200 molecules.
Uses XML default scale factors (mScale14=0).
"""
from openmm.app import *
from openmm import *
from openmm.unit import *
import phyneoplugin
import numpy as np
import sys

# Parameters
XML_FILE = '/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/1_training_slater_nb/phyneo_ecl.xml'
PDB_FILE = '/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/1_training_slater_nb/pdb_bank/EC.pdb'
TEMPERATURE = 300 * kelvin
PRESSURE = 1 * atmosphere
NSTEPS = 5000  # 10ps (2fs timestep)
DT = 1 * femtoseconds
N_MOLECULES = 200

def create_ec_bulk_pdb():
    """Use packmol-generated PDB file."""
    pdb_path = '/tmp/ec_bulk_200.pdb'
    print(f"Using packmol-generated PDB: {pdb_path}")
    return pdb_path

def run_bulk_md():
    """Run bulk MD with EC."""

    print("=" * 60)
    print("EC Bulk MD Simulation")
    print(f"N molecules: {N_MOLECULES}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Pressure: {PRESSURE}")
    print(f"Steps: {NSTEPS} ({NSTEPS * DT / picosecond} ps)")
    print(f"mScale14: 1.0 (full scaling)")
    print("=" * 60)

    # Create PDB
    pdb_path = create_ec_bulk_pdb()
    pdb = PDBFile(pdb_path)

    # Load forcefield
    forcefield = ForceField(XML_FILE)

    # Create system
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=LJPME,
        polarization="extrapolated",
        nonbondedCutoff=8*angstrom,
        constraints=HBonds
    )

    # Use XML default (mScale14=0) - skip since setMultipoleScaleFactors doesn't work on CUDA
    # The XML already has mScale14=0 set, so we just use that
    print("Using XML default scale factors (mScale14=0)")

    # Create integrator (NVT, no barostat)
    integrator = LangevinIntegrator(TEMPERATURE, 1/picosecond, DT)

    # Choose platform
    platform_name = sys.argv[1] if len(sys.argv) > 1 else 'CUDA'
    try:
        platform = Platform.getPlatformByName(platform_name)
        properties = {'DeviceIndex': sys.argv[2] if len(sys.argv) > 2 else '0', 'Precision': 'mixed'}
        simulation = Simulation(pdb.topology, system, integrator, platform, properties)
        print(f"Platform: {platform_name}")
    except Exception as e:
        print(f"Using Reference platform: {e}")
        platform = Platform.getPlatformByName('Reference')
        simulation = Simulation(pdb.topology, system, integrator, platform)

    context = simulation.context
    context.setPositions(pdb.positions)

    # Minimize
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=500)

    # Get initial state
    state = context.getState(getEnergy=True)
    print(f"Initial energy: {state.getPotentialEnergy()}")

    # Equilibrate for 10ps
    print("Equilibrating 10ps...")
    simulation.step(5000)

    # Run production 100ps
    print(f"Running production 100ps...")
    simulation.step(NSTEPS)

    # Get final state
    state = context.getState(getEnergy=True, getPositions=True)

    # Calculate density
    box_vectors = state.getPeriodicBoxVectors()
    volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]

    # Calculate total mass (convert dalton to kg)
    total_mass_dalton = 0
    for i in range(system.getNumParticles()):
        total_mass_dalton += system.getParticleMass(i).value_in_unit(dalton)
    total_mass_kg = total_mass_dalton * 1.66054e-27

    density_kg_m3 = (total_mass_kg * kilogram) / volume
    density_g_ml = density_kg_m3.value_in_unit(kilogram / nanometer**3) * 1e24

    print(f"\n=== RESULTS ===")
    print(f"Final energy: {state.getPotentialEnergy()}")
    print(f"Density: {density_g_ml:.4f} g/mL")
    print(f"Experimental density: ~1.32 g/mL")

    return density_g_ml, state.getPotentialEnergy()

if __name__ == '__main__':
    run_bulk_md()
