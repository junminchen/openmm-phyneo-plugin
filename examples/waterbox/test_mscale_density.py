#!/usr/bin/env python
"""
Test different mScale14 values for water density.
Runs 100ps NPT simulation and measures density.
"""
from openmm.app import *
from openmm import *
from openmm.unit import *
import phyneoplugin
import numpy as np
import sys

# Parameters
XML_FILE = 'mpidwater_lmax2_test14.xml'
PDB_FILE = 'waterbox_31ang.pdb'
TEMPERATURE = 300 * kelvin
PRESSURE = 1 * atmosphere
NSTEPS = 50000  # 100ps (2fs timestep)
DT = 2 * femtoseconds

# Test different mScale14 values
mScale14_values = [0.0, 0.25, 0.5, 0.75, 1.0]

def run_simulation(mScale14_value):
    """Run 100ps simulation with given mScale14 and return final density."""

    # Load PDB and forcefield
    pdb = PDBFile(PDB_FILE)
    forcefield = ForceField(XML_FILE)

    # Create system with LJPME and extrapolated polarization
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=LJPME,
        polarization="extrapolated",
        nonbondedCutoff=8*angstrom,
        constraints=HBonds,
        defaultTholeWidth=8
    )

    # Find and modify PhyNEOForce mScale14
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, phyneoplugin.PhyNEOForce):
            # Get current scales (default from XML)
            # mScale indices: 0=12, 1=13, 2=14, 3=15, 4=16, 5=...
            mScales = np.array([0.0, 0.0, mScale14_value, 1.0, 1.0, 1.0])
            pScales = np.array([0.0, 0.0, mScale14_value, 1.0, 1.0, 1.0])
            dScales = np.array([1.0, 1.0, mScale14_value, 1.0, 1.0, 1.0])

            f.setMultipoleScaleFactors(mScales, pScales, dScales)
            print(f"  mScale14 set to: {mScale14_value}")
            break

    # Create integrator and barostat
    integrator = LangevinIntegrator(TEMPERATURE, 1/picosecond, DT)
    system.addForce(MonteCarloBarostat(PRESSURE, TEMPERATURE, 25))

    # Choose platform
    platform_name = sys.argv[1] if len(sys.argv) > 1 else 'CUDA'
    try:
        platform = Platform.getPlatformByName(platform_name)
        properties = {'DeviceIndex': sys.argv[2] if len(sys.argv) > 2 else '0', 'Precision': 'mixed'}
        simulation = Simulation(pdb.topology, system, integrator, platform, properties)
        print(f"  Platform: {platform_name}")
    except:
        print("  Using Reference platform")
        platform = Platform.getPlatformByName('Reference')
        simulation = Simulation(pdb.topology, system, integrator, platform)

    context = simulation.context
    context.setPositions(pdb.positions)

    # Set periodic box vectors
    if pdb.topology.getPeriodicBoxVectors():
        context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Minimize
    print("  Minimizing...")
    simulation.minimizeEnergy(maxIterations=1000)

    # Equilibrate for 10ps first
    print("  Equilibrating 10ps...")
    simulation.step(5000)

    # Run production 100ps
    print(f"  Running production 100ps...")
    simulation.step(NSTEPS)

    # Get final density
    state = context.getState(getEnergy=True, getPositions=True)
    box_vectors = state.getPeriodicBoxVectors()
    volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]

    # Calculate total mass
    total_mass = 0
    for i in range(system.getNumParticles()):
        total_mass += system.getParticleMass(i).value_in_unit(dalton)

    density = (total_mass * dalton) / volume
    # Convert: dalton/nanometer^3 to gram/milliliter
    # 1 dalton = 1.66054e-24 gram
    # 1 nanometer^3 = 1e-21 milliliter
    # So 1 dalton/nanometer^3 = 1.66054e-3 gram/milliliter
    density_g_ml = density.value_in_unit(dalton/nanometer**3) * 1.66054e-3

    return density_g_ml

def main():
    print("=" * 60)
    print("Testing different mScale14 values - 100ps NPT simulation")
    print("=" * 60)
    print(f"Temperature: {TEMPERATURE}")
    print(f"Pressure: {PRESSURE}")
    print(f"Steps: {NSTEPS} ({NSTEPS * DT / picosecond} ps)")
    print()

    results = []
    for mScale14 in mScale14_values:
        print(f"\n>>> Testing mScale14 = {mScale14}")
        try:
            density = run_simulation(mScale14)
            results.append((mScale14, density))
            print(f"  Final density: {density:.4f} g/mL")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((mScale14, None))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Density vs mScale14")
    print("=" * 60)
    print(f"{'mScale14':<12} {'Density (g/mL)':<20}")
    print("-" * 32)
    for mScale14, density in results:
        if density is not None:
            print(f"{mScale14:<12.2f} {density:<20.4f}")
        else:
            print(f"{mScale14:<12.2f} {'FAILED':<20}")

if __name__ == '__main__':
    main()
