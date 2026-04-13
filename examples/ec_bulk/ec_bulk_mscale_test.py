#!/usr/bin/env python
"""
EC (ethylene carbonate) bulk MD simulation test.
Tests two different scale settings:
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
PRESSURE = 1 * atmosphere
NSTEPS = 50000  # 100ps (2fs timestep)
DT = 2 * femtoseconds
N_MOLECULES = 27  # 3x3x3

def create_ec_pdb():
    """Create a PDB file with multiple EC molecules."""
    pdb_in = PDBFile(PDB_FILE)
    original_positions = list(pdb_in.positions)

    # Box size: ~50 Angstrom to get reasonable density
    box_size = 3.5 * nanometer  # ~35 Angstrom

    box_angstrom = box_size / angstrom
    lines = [
        "HEADER    EC BULK TEST\n",
        "TITLE     EC BULK N=%d\n" % N_MOLECULES,
        "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n" % (box_angstrom, box_angstrom, box_angstrom, 90.0, 90.0, 90.0),
    ]

    atom_idx = 1
    lines_per_mol = 10  # 10 atoms per EC

    for i in range(N_MOLECULES):
        # Simple cubic grid
        offset_x = (i % 3) * box_size / 3
        offset_y = ((i // 3) % 3) * box_size / 3
        offset_z = (i // 9) * box_size / 3

        for j, pos in enumerate(original_positions):
            # Get atom name from original
            atom_name = ['O00', 'C01', 'O02', 'C03', 'C04', 'O05', 'H06', 'H07', 'H08', 'H09'][j]
            elem = ['O', 'C', 'O', 'C', 'C', 'O', 'H', 'H', 'H', 'H'][j]

            x = pos.x / 10 + offset_x / nanometer
            y = pos.y / 10 + offset_y / nanometer
            z = pos.z / 10 + offset_z / nanometer

            lines.append("ATOM  %5d  %-4s  ECA  %4d     %8.3f %8.3f %8.3f  1.00  0.00      %-2s\n" % (
                atom_idx, atom_name, i+1, x, y, z, elem))
            atom_idx += 1

    lines.append("END\n")

    # Write to temp PDB
    with open('/tmp/ec_bulk.pdb', 'w') as f:
        f.writelines(lines)

    return '/tmp/ec_bulk.pdb'

def run_simulation(scale_name, mScale14, pScale14, dScale14):
    """Run simulation with given scale settings."""

    print(f"\n>>> Testing {scale_name}: mScale14={mScale14}, pScale14={pScale14}, dScale14={dScale14}")

    # Create PDB with multiple molecules
    pdb_path = create_ec_pdb()
    pdb = PDBFile(pdb_path)

    # Load forcefield
    forcefield = ForceField(XML_FILE)

    # Create system
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
            # Default scales: mScale12=0, mScale13=0, mScale14=X, mScale15=1, mScale16=1
            mScales = np.array([0.0, 0.0, mScale14, 1.0, 1.0, 1.0])
            pScales = np.array([0.0, 0.0, pScale14, 1.0, 1.0, 1.0])
            dScales = np.array([1.0, 1.0, dScale14, 1.0, 1.0, 1.0])

            f.setMultipoleScaleFactors(mScales, pScales, dScales)
            print(f"  Scales set: mScale14={mScale14}, pScale14={pScale14}, dScale14={dScale14}")
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
    except Exception as e:
        print(f"  Using Reference platform: {e}")
        platform = Platform.getPlatformByName('Reference')
        simulation = Simulation(pdb.topology, system, integrator, platform)

    context = simulation.context
    context.setPositions(pdb.positions)

    # Set periodic box vectors
    box_vec = 3.5 * nanometer
    context.setPeriodicBoxVectors(box_vec, box_vec, box_vec)

    # Minimize
    print("  Minimizing...")
    simulation.minimizeEnergy(maxIterations=1000)

    # Equilibrate for 10ps
    print("  Equilibrating 10ps...")
    simulation.step(5000)

    # Run production 100ps
    print(f"  Running production 100ps...")
    simulation.step(NSTEPS)

    # Get final state
    state = context.getState(getEnergy=True, getPositions=True)

    # Calculate density
    box_vectors = state.getPeriodicBoxVectors()
    volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]

    # Calculate total mass (EC molecule: C3H4O3 = 88.06 g/mol)
    total_mass_kg = 0
    for i in range(system.getNumParticles()):
        total_mass_kg += system.getParticleMass(i).value_in_unit(kilogram)

    density_kg_m3 = (total_mass_kg * kilogram) / volume
    density_g_ml = density_kg_m3.value_in_unit(kilogram / nanometer**3) * 1e24

    energy = state.getPotentialEnergy()

    return density_g_ml, energy

def main():
    print("=" * 60)
    print("EC Bulk MD Simulation - mScale14 Comparison")
    print("=" * 60)
    print(f"Temperature: {TEMPERATURE}")
    print(f"Pressure: {PRESSURE}")
    print(f"Steps: {NSTEPS} ({NSTEPS * DT / picosecond} ps)")
    print(f"N molecules: {N_MOLECULES}")
    print()

    results = []

    # Test 1: XML setting (mScale14=0)
    try:
        density, energy = run_simulation("XML (mScale14=0)", 0.0, 0.0, 0.0)
        results.append(("XML (mScale14=0)", density, energy))
        print(f"  Density: {density:.4f} g/mL")
        print(f"  Energy: {energy}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Water-like setting (mScale14=0.5)
    try:
        density, energy = run_simulation("Water-like (mScale14=0.5)", 0.5, 0.5, 0.5)
        results.append(("Water-like (mScale14=0.5)", density, energy))
        print(f"  Density: {density:.4f} g/mL")
        print(f"  Energy: {energy}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Setting':<30} {'Density (g/mL)':<15} {'Energy':<20}")
    print("-" * 65)
    for name, density, energy in results:
        print(f"{name:<30} {density:<15.4f} {str(energy):<20}")
    print()
    print("Note: Experimental EC density at 300K is ~1.32 g/mL")

if __name__ == '__main__':
    main()
