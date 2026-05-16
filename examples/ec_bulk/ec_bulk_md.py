#!/usr/bin/env python
"""
EC bulk MD simulation with 200 molecules.
Uses the PhyNEO XML multipole scale factors.
"""
from openmm.app import *
from openmm import *
from openmm.unit import *
import phyneoplugin
import numpy as np
import math
import sys

# Parameters
XML_FILE = '/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/1_training_slater_nb/phyneo_ecl.xml'
TEMPERATURE = 300 * kelvin
EQUIL_STEPS = 1000
NSTEPS = 5000
DT = 1 * femtoseconds
N_MOLECULES = 200
BOX_SIZE = 80 * angstrom
NONBONDED_CUTOFF = 8 * angstrom

LJ_PARAMS = {
    'H': (0.25*nanometer, 0.10*kilojoule_per_mole),
    'C': (0.34*nanometer, 0.35*kilojoule_per_mole),
    'O': (0.30*nanometer, 0.65*kilojoule_per_mole),
}

def create_ec_bulk_pdb():
    """Use packmol-generated PDB file."""
    pdb_path = '/tmp/ec_bulk_200.pdb'
    print(f"Using packmol-generated PDB: {pdb_path}")
    return pdb_path

def _position_array(positions):
    """Return positions as a plain numpy array in nm."""
    return np.array([[p.x, p.y, p.z] for p in positions.value_in_unit(nanometer)])

def add_example_stabilizers(system, topology, positions):
    """Add simple bonded geometry and excluded-volume terms for this MD example.

    The PhyNEO XML used here contributes the multipole/polarization force only.
    These helper forces keep the packed EC molecules from collapsing during the
    CUDA smoke MD run; they are not a replacement for a production EC force field.
    """
    pos = _position_array(positions)
    bonds = [(a.index, b.index) for a, b in topology.bonds()]
    if not bonds:
        raise ValueError("The EC bulk PDB must contain CONECT bonds for the MD stabilizers.")

    bond_force = HarmonicBondForce()
    bond_force.setUsesPeriodicBoundaryConditions(False)
    for i, j in bonds:
        length = float(np.linalg.norm(pos[i]-pos[j]))
        bond_force.addBond(i, j, length, 300000.0)
    system.addForce(bond_force)

    neighbors = [set() for _ in range(system.getNumParticles())]
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)

    angle_force = HarmonicAngleForce()
    angle_force.setUsesPeriodicBoundaryConditions(False)
    for center, bonded in enumerate(neighbors):
        bonded = sorted(bonded)
        for first in range(len(bonded)):
            for second in range(first+1, len(bonded)):
                i = bonded[first]
                k = bonded[second]
                v1 = pos[i]-pos[center]
                v2 = pos[k]-pos[center]
                cos_theta = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                angle_force.addAngle(i, center, k, theta, 500.0)
    system.addForce(angle_force)

    lj_force = NonbondedForce()
    lj_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    lj_force.setCutoffDistance(NONBONDED_CUTOFF)
    lj_force.setUseSwitchingFunction(True)
    lj_force.setSwitchingDistance(0.7*nanometer)
    lj_force.setUseDispersionCorrection(False)
    for atom in topology.atoms():
        sigma, epsilon = LJ_PARAMS.get(atom.element.symbol, (0.30*nanometer, 0.20*kilojoule_per_mole))
        lj_force.addParticle(0.0, sigma, epsilon)
    lj_force.createExceptionsFromBonds(bonds, 0.0, 0.0)
    system.addForce(lj_force)

    print(f"Added MD stabilizers: {len(bonds)} bonds, "
          f"{angle_force.getNumAngles()} angles, {lj_force.getNumParticles()} LJ particles")

def check_state(context, label):
    """Print and validate finite energy/force diagnostics."""
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy()
    energy_value = energy.value_in_unit(kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer)
    max_force = float(np.max(np.abs(forces)))
    rms_force = float(np.sqrt(np.mean(forces*forces)))
    print(f"{label} energy: {energy}")
    print(f"{label} force max/rms: {max_force:.3f} / {rms_force:.3f} kJ/mol/nm")
    if not math.isfinite(energy_value) or not np.all(np.isfinite(forces)):
        raise RuntimeError(f"Non-finite energy or forces after {label}")
    return energy

def run_bulk_md():
    """Run bulk MD with EC."""

    print("=" * 60)
    print("EC Bulk MD Simulation")
    print(f"N molecules: {N_MOLECULES}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Box: {BOX_SIZE}")
    print(f"Equilibration steps: {EQUIL_STEPS} ({EQUIL_STEPS * DT / picosecond} ps)")
    print(f"Production steps: {NSTEPS} ({NSTEPS * DT / picosecond} ps)")
    print("=" * 60)

    # Create PDB
    pdb_path = create_ec_bulk_pdb()
    pdb = PDBFile(pdb_path)

    # Use Modeller to add periodic box
    modeller = Modeller(pdb.topology, pdb.positions)
    box_length = BOX_SIZE.value_in_unit(nanometer)
    box_vec = Vec3(box_length, 0, 0) * nanometer, Vec3(0, box_length, 0) * nanometer, Vec3(0, 0, box_length) * nanometer
    modeller.topology.setPeriodicBoxVectors(box_vec)

    # Load forcefield
    forcefield = ForceField(XML_FILE)

    # Create system
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=LJPME,
        polarization="extrapolated",
        nonbondedCutoff=NONBONDED_CUTOFF,
        constraints=None
    )
    add_example_stabilizers(system, modeller.topology, modeller.positions)

    print("Using PhyNEO XML multipole scale factors")

    # Create integrator (NVT, no barostat)
    integrator = LangevinIntegrator(TEMPERATURE, 1/picosecond, DT)

    # Choose platform
    platform_name = sys.argv[1] if len(sys.argv) > 1 else 'CUDA'
    try:
        platform = Platform.getPlatformByName(platform_name)
        properties = {'DeviceIndex': sys.argv[2] if len(sys.argv) > 2 else '0', 'Precision': 'mixed'}
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        print(f"Platform: {platform_name}")
    except Exception as e:
        print(f"Using Reference platform: {e}")
        platform = Platform.getPlatformByName('Reference')
        simulation = Simulation(modeller.topology, system, integrator, platform)

    context = simulation.context
    context.setPositions(modeller.positions)

    # Minimize
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=200)
    check_state(context, "Minimized")
    context.setVelocitiesToTemperature(TEMPERATURE, 2026)

    print("Equilibrating...")
    simulation.step(EQUIL_STEPS)
    check_state(context, "Post-equilibration")

    print("Running production...")
    simulation.step(NSTEPS)

    # Get final state
    state = context.getState(getEnergy=True, getPositions=True)
    final_energy = check_state(context, "Final")

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
    print(f"Final energy: {final_energy}")
    print(f"Density: {density_g_ml:.4f} g/mL")

    return density_g_ml, final_energy

if __name__ == '__main__':
    run_bulk_md()
