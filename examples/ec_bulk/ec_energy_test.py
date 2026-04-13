#!/usr/bin/env python
"""
EC single molecule energy test - verify scale factor settings.
Tests two different scale settings on a single EC molecule.
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

def compute_energy(scale_name, mScale14, pScale14, dScale14):
    """Compute energy with given scale settings."""

    print(f"\n>>> Testing {scale_name}: mScale14={mScale14}, pScale14={pScale14}, dScale14={dScale14}")

    # Load PDB
    pdb = PDBFile(PDB_FILE)

    # Load forcefield
    forcefield = ForceField(XML_FILE)

    # Create system with NoCutoff first to test the force directly
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

    # Create integrator
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2*femtoseconds)

    # Use Reference platform for stability
    platform = Platform.getPlatformByName('Reference')
    simulation = Simulation(pdb.topology, system, integrator, platform)

    context = simulation.context
    context.setPositions(pdb.positions)

    # Set periodic box
    box_vec = 5.0 * nanometer
    context.setPeriodicBoxVectors(
        Vec3(box_vec, 0*nanometer, 0*nanometer),
        Vec3(0*nanometer, box_vec, 0*nanometer),
        Vec3(0*nanometer, 0*nanometer, box_vec)
    )

    # Compute initial energy
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy()

def main():
    print("=" * 60)
    print("EC Single Molecule - Energy vs Scale Factor")
    print("=" * 60)

    results = []

    # Test 1: XML setting (mScale14=0)
    try:
        energy = compute_energy("XML (mScale14=0)", 0.0, 0.0, 0.0)
        results.append(("XML (mScale14=0)", energy))
        print(f"  Energy: {energy}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Water-like setting (mScale14=0.5)
    try:
        energy = compute_energy("Water-like (mScale14=0.5)", 0.5, 0.5, 0.5)
        results.append(("Water-like (mScale14=0.5)", energy))
        print(f"  Energy: {energy}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: mScale14=1.0
    try:
        energy = compute_energy("Full (mScale14=1.0)", 1.0, 1.0, 1.0)
        results.append(("Full (mScale14=1.0)", energy))
        print(f"  Energy: {energy}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - Potential Energy (kJ/mol)")
    print("=" * 60)
    print(f"{'Setting':<30} {'Energy':<20}")
    print("-" * 50)
    for name, energy in results:
        val = energy.value_in_unit(kilojoule_per_mole)
        print(f"{name:<30} {val:<20.4f}")

if __name__ == '__main__':
    main()
