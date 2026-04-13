#!/usr/bin/env python
"""
EC single molecule energy test - CUDA version
"""
from openmm.app import *
from openmm import *
from openmm.unit import *
import phyneoplugin
import numpy as np
import sys

XML_FILE = '/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/1_training_slater_nb/phyneo_ecl.xml'
PDB_FILE = '/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/1_training_slater_nb/pdb_bank/EC.pdb'

def compute_energy_cuda(scale_name, mScale14, pScale14, dScale14):
    print(f"\n>>> Testing {scale_name}: mScale14={mScale14}")

    pdb = PDBFile(PDB_FILE)
    forcefield = ForceField(XML_FILE)

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=LJPME,
        polarization="extrapolated",
        nonbondedCutoff=10*angstrom,
        constraints=HBonds
    )

    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, phyneoplugin.PhyNEOForce):
            mScales = np.array([0.0, 0.0, mScale14, 1.0, 1.0, 1.0])
            pScales = np.array([0.0, 0.0, pScale14, 1.0, 1.0, 1.0])
            dScales = np.array([1.0, 1.0, dScale14, 1.0, 1.0, 1.0])
            f.setMultipoleScaleFactors(mScales, pScales, dScales)
            break

    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2*femtoseconds)

    platform = Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)

    context = simulation.context

    box_vec = 5.0 * nanometer
    context.setPeriodicBoxVectors(
        Vec3(box_vec, 0*nanometer, 0*nanometer),
        Vec3(0*nanometer, box_vec, 0*nanometer),
        Vec3(0*nanometer, 0*nanometer, box_vec)
    )
    context.setPositions(pdb.positions)

    # Reinitialize after setting scale factors to upload them to GPU
    context.reinitialize()
    context.setPositions(pdb.positions)

    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy()

def main():
    print("=" * 60)
    print("EC Single Molecule - Energy vs Scale Factor (CUDA)")
    print("=" * 60)

    results = []

    for name, m in [("mScale14=0", 0.0), ("mScale14=0.5", 0.5), ("mScale14=1.0", 1.0)]:
        try:
            energy = compute_energy_cuda(name, m, m, m)
            results.append((name, energy))
            print(f"  Energy: {energy}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY - Potential Energy (kJ/mol)")
    print("=" * 60)
    for name, energy in results:
        val = energy.value_in_unit(kilojoule_per_mole)
        print(f"{name:<20}: {val:.4f}")

if __name__ == '__main__':
    main()
