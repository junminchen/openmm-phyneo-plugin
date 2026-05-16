#!/usr/bin/env python
"""Debug script to check covalent flags for EC 1-4 pairs."""
from openmm.app import *
from openmm import *
from openmm.unit import *
import phyneoplugin
from phyneoplugin import PhyNEOForce
import numpy as np

# Parameters
XML_FILE = '/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/1_training_slater_nb/phyneo_ecl.xml'
PDB_FILE = '/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/1_training_slater_nb/pdb_bank/EC.pdb'

def debug_scale_factors():
    """Check covalent flags for EC system."""
    print("Loading EC system...")

    # Load PDB
    pdb = PDBFile(PDB_FILE)
    print(f"Number of atoms: {pdb.topology.getNumAtoms()}")

    # Use Modeller to add periodic box
    modeller = Modeller(pdb.topology, pdb.positions)
    box_vec = Vec3(80.0, 0, 0) * angstrom, Vec3(0, 80.0, 0) * angstrom, Vec3(0, 0, 80.0) * angstrom
    modeller.topology.setPeriodicBoxVectors(box_vec)

    # Load forcefield
    forcefield = ForceField(XML_FILE)

    # Create system
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=LJPME,
        polarization="extrapolated",
        nonbondedCutoff=8*angstrom,
        constraints=HBonds
    )

    # Get PhyNEO force
    phyneo_force = None
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if 'PhyNEO' in type(f).__name__:
            phyneo_force = f
            print(f"Found PhyNEOForce!")

    if phyneo_force is None:
        print("ERROR: PhyNEOForce not found!")
        return

    # Get covalent information for first few atoms
    print(f"\nChecking covalent flags for atoms 0-4...")
    covalent_types = ['Covalent12', 'Covalent13', 'Covalent14']
    for atom in range(5):
        print(f"\nAtom {atom}:")
        for cov_type in covalent_types:
            try:
                enum_val = getattr(PhyNEOForce, cov_type)
                atoms = phyneo_force.getCovalentMap(atom, enum_val)
                print(f"  {cov_type}: {atoms}")
            except Exception as e:
                print(f"  {cov_type}: Error - {e}")

if __name__ == '__main__':
    debug_scale_factors()