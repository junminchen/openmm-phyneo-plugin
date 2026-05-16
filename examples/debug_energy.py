#!/usr/bin/env python
"""
Debug script to compare Reference vs CUDA energy components for PhyNEOForce.
This helps identify which part of the energy calculation differs between platforms.
"""
from openmm.app import *
from openmm import *
from openmm.unit import *
import phyneoplugin
import numpy as np
import sys

# Test system: simple water box for quick debugging
PDB_FILE = '/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/init_mpid_plugin/OpenMMPhyNEOPlugin/examples/waterbox/waterbox_31ang.pdb'
XML_FILE = '/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/init_mpid_plugin/OpenMMPhyNEOPlugin/examples/waterbox/mpidwater_lmax2.xml'

def run_on_platform(platform_name, device_id='0'):
    """Run simulation on specified platform and return energy."""
    print(f"\n{'='*60}")
    print(f"Running on {platform_name}")
    print(f"{'='*60}")

    # Load PDB and forcefield
    pdb = PDBFile(PDB_FILE)
    forcefield = ForceField(XML_FILE)

    # Create system with PhyNEOForce
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=LJPME,
        polarization="extrapolated",
        nonbondedCutoff=8*angstrom,
        constraints=HBonds,
        defaultTholeWidth=8
    )

    # Set mScale14=0 to match typical settings
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, phyneoplugin.PhyNEOForce):
            f.set14ScaleFactor(0.0)
            print(f"Set mScale14=0 on PhyNEOForce")
            break

    # Create integrator
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2*femtoseconds)

    # Create simulation
    try:
        if platform_name == 'CUDA':
            platform = Platform.getPlatformByName('CUDA')
            properties = {'DeviceIndex': device_id, 'Precision': 'mixed'}
            simulation = Simulation(pdb.topology, system, integrator, platform, properties)
        elif platform_name == 'Reference':
            platform = Platform.getPlatformByName('Reference')
            simulation = Simulation(pdb.topology, system, integrator, platform)
        else:
            raise ValueError(f"Unknown platform: {platform_name}")
    except Exception as e:
        print(f"Failed to create simulation: {e}")
        return None

    context = simulation.context
    context.setPositions(pdb.positions)

    # Get initial state with energy decomposition
    state = context.getState(getEnergy=True, getForces=True)

    print(f"Platform: {context.getPlatform().getName()}")
    print(f"Total potential energy: {state.getPotentialEnergy()}")

    # Use ForceGroups to get per-force energies
    print("\n--- Force Group Analysis ---")
    for i in range(system.getNumForces()):
        system.getForce(i).setForceGroup(i)

    # Re-create context to apply force groups
    if platform_name == 'CUDA':
        simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    else:
        simulation = Simulation(pdb.topology, system, integrator, platform)
    context = simulation.context
    context.setPositions(pdb.positions)

    for i in range(system.getNumForces()):
        state = context.getState(getEnergy=True, groups={i})
        energy = state.getPotentialEnergy()
        force_type = type(system.getForce(i)).__name__
        print(f"Force {i} ({force_type}): {energy}")

    return state.getPotentialEnergy()

if __name__ == '__main__':
    # Run on Reference
    ref_energy = run_on_platform('Reference')

    # Run on CUDA
    cuda_energy = run_on_platform('CUDA', sys.argv[1] if len(sys.argv) > 1 else '0')

    if ref_energy is not None and cuda_energy is not None:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"Reference: {ref_energy}")
        print(f"CUDA:      {cuda_energy}")
        diff = cuda_energy - ref_energy
        print(f"Difference: {diff}")
        print(f"Relative:   {diff/ref_energy*100:.4f}%")