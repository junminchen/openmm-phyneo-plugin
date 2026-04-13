#!/usr/bin/env python
"""
EC single molecule MD test - verify scale factor settings.
Tests two different scale settings on a single EC molecule:
1. Default water-like: mScale14=0.5, pScale14=0.5, dScale14=0.5
2. XML setting: mScale14=0.0, pScale14=0.0, dScale14=0.0
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
NSTEPS = 50000  # 100ps (2fs timestep)
DT = 2 * femtoseconds

def run_simulation(scale_name, mScale14, pScale14, dScale14):
    """Run simulation with given scale settings."""

    print(f"\n>>> Testing {scale_name}: mScale14={mScale14}, pScale14={pScale14}, dScale14={dScale14}")

    # Load PDB
    pdb = PDBFile(PDB_FILE)

    # Load forcefield
    forcefield = ForceField(XML_FILE)

    # Create system with LJPME (for periodic boundary)
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=LJPME,
        polarization="extrapolated",
        nonbondedCutoff=10*angstrom,
        constraints=HBonds
    )

    # Find and modify PhyNEOForce scales
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, phyneoplugin.PhyNEOForce):
            # Create new scale arrays
            mScales = np.array([0.0, 0.0, mScale14, 1.0, 1.0, 1.0])
            pScales = np.array([0.0, 0.0, pScale14, 1.0, 1.0, 1.0])
            dScales = np.array([1.0, 1.0, dScale14, 1.0, 1.0, 1.0])

            f.setMultipoleScaleFactors(mScales, pScales, dScales)
            print(f"  Scales: mScale14={mScale14}, pScale14={pScale14}, dScale14={dScale14}")
            break

    # Create integrator (no barostat for single molecule)
    integrator = LangevinIntegrator(TEMPERATURE, 1/picosecond, DT)

    # Choose platform
    platform_name = sys.argv[1] if len(sys.argv) > 1 else 'CUDA'
    try:
        platform = Platform.getPlatformByName(platform_name)
        properties = {'DeviceIndex': sys.argv[2] if len(sys.argv) > 2 else '0', 'Precision': 'mixed'}
        simulation = Simulation(pdb.topology, system, integrator, platform, properties)
        print(f"  Platform: {platform_name}")
    except Exception as e:
        print(f"  Using Reference platform: {e}")
        platform = Platform.getPlatformByName('Reference')
        simulation = Simulation(pdb.topology, system, integrator, platform)

    context = simulation.context
    context.setPositions(pdb.positions)

    # Set periodic box (single molecule in large box)
    box_vec = 5.0 * nanometer
    context.setPeriodicBoxVectors(
        Vec3(box_vec, 0*nanometer, 0*nanometer),
        Vec3(0*nanometer, box_vec, 0*nanometer),
        Vec3(0*nanometer, 0*nanometer, box_vec)
    )

    # Minimize
    print("  Minimizing...")
    simulation.minimizeEnergy(maxIterations=1000)

    # Get initial energy
    state = context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy()

    # Run production 100ps
    print(f"  Running 100ps...")
    simulation.step(NSTEPS)

    # Get final energy
    state = context.getState(getEnergy=True)
    final_energy = state.getPotentialEnergy()

    return initial_energy, final_energy

def main():
    print("=" * 60)
    print("EC Single Molecule MD - mScale14 Comparison")
    print("=" * 60)
    print(f"Temperature: {TEMPERATURE}")
    print(f"Steps: {NSTEPS} ({NSTEPS * DT / picosecond} ps)")
    print(f"Box size: 5nm (single molecule)")
    print()

    results = []

    # Test 1: XML setting (mScale14=0)
    try:
        initial, final = run_simulation("XML (mScale14=0)", 0.0, 0.0, 0.0)
        results.append(("XML (mScale14=0)", initial, final))
        print(f"  Initial Energy: {initial}")
        print(f"  Final Energy: {final}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Water-like setting (mScale14=0.5)
    try:
        initial, final = run_simulation("Water-like (mScale14=0.5)", 0.5, 0.5, 0.5)
        results.append(("Water-like (mScale14=0.5)", initial, final))
        print(f"  Initial Energy: {initial}")
        print(f"  Final Energy: {final}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - Energy (kJ/mol)")
    print("=" * 60)
    print(f"{'Setting':<30} {'Initial':<20} {'Final':<20}")
    print("-" * 70)
    for name, initial, final in results:
        init_val = initial.value_in_unit(kilojoule_per_mole)
        final_val = final.value_in_unit(kilojoule_per_mole)
        print(f"{name:<30} {init_val:<20.4f} {final_val:<20.4f}")

if __name__ == '__main__':
    main()
