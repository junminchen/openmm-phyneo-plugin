from openmm.app import *
from openmm import *
from openmm.unit import *
import mpidplugin
from sys import stdout

# Load PDB and ForceField
pdb = PDBFile('ec_init.pdb')
topology = pdb.topology
# Explicitly add bonds for each residue to match XML template
# ECA: O00(0)-C01(1), C01(1)-O02(2), C01(1)-O05(5), O02(2)-C03(3), O05(5)-C04(4), C03(3)-C04(4), 
#      C03(3)-H06(6), C03(3)-H07(7), C04(4)-H08(8), C04(4)-H09(9)
for res in topology.residues():
    atoms = list(res.atoms())
    topology.addBond(atoms[0], atoms[1]) # O00-C01
    topology.addBond(atoms[1], atoms[2]) # C01-O02
    topology.addBond(atoms[1], atoms[5]) # C01-O05
    topology.addBond(atoms[2], atoms[3]) # O02-C03
    topology.addBond(atoms[5], atoms[4]) # O05-C04
    topology.addBond(atoms[3], atoms[4]) # C03-C04
    topology.addBond(atoms[3], atoms[6]) # C03-H06
    topology.addBond(atoms[3], atoms[7]) # C03-H07
    topology.addBond(atoms[4], atoms[8]) # C04-H08
    topology.addBond(atoms[4], atoms[9]) # C04-H09

forcefield = ForceField('ec_forcefield.xml')

# Create System
# Note: CustomNonbondedForce requires NoCutoff or CutoffPeriodic
system = forcefield.createSystem(pdb.topology, nonbondedMethod=CutoffPeriodic,
                                 nonbondedCutoff=1.0*nanometer, constraints=HBonds)

# Add Barostat for NPT
system.addForce(MonteCarloBarostat(1.0*bar, 300.0*kelvin))

# Integrator
integrator = LangevinMiddleIntegrator(300.0*kelvin, 1.0/picosecond, 0.001*picosecond)

# Platform
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'mixed'}

simulation = Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300.0*kelvin)

print("Minimizing energy...")
simulation.minimizeEnergy(maxIterations=2000)

print("Running NPT simulation...")
simulation.reporters.append(StateDataReporter(stdout, 500, step=True,
    potentialEnergy=True, temperature=True, density=True, speed=True))

simulation.step(20000)
