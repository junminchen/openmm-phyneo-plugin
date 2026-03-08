#!/usr/bin/env python3
"""Final single-point energy check: MPID + Bridge + SR terms vs DMFF Total."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

import openmm as mm
import openmm.app as app
from openmm import unit

import mpidplugin
from dmff_sr_custom_forces import TERM_ORDER, add_dmff_short_range_forces_from_xml
from dispersion_pme_bridge import DispersionPMEBridgeModel

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

def choose_platform(name: str):
    if name == "Reference":
        return mm.Platform.getPlatformByName("Reference"), {}
    return mm.Platform.getPlatformByName("CPU"), {}

def main():
    dmff_xml = EXAMPLE_DIR / "ff.backend_amoeba_total1000_classical_intra.xml"
    pdb_path = EXAMPLE_DIR / "02molL_init.pdb"
    residues_xml = EXAMPLE_DIR / "residues.xml"

    app.Topology.loadBondDefinitions(str(residues_xml))
    pdb = app.PDBFile(str(pdb_path))
    
    # 1. Create System
    ff = app.ForceField(str(dmff_xml))
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    for i in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(i), mm.NonbondedForce):
            system.removeForce(i)

    vectors = pdb.topology.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*vectors)
    
    root = ET.parse(dmff_xml).getroot()
    admp = root.find("ADMPPmeForce")
    mpid_force = mpidplugin.MPIDForce()
    mpid_force.setNonbondedMethod(mpidplugin.MPIDForce.PME)
    mpid_force.setCutoffDistance(0.6)
    mpid_force.setPMEParameters(4.8640384, 51, 51, 51)
    mpid_force.setPolarizationType(mpidplugin.MPIDForce.Mutual)
    mpid_force.setMutualInducedMaxIterations(20)
    mpid_force.setMutualInducedTargetEpsilon(1e-8)
    
    scales = [0.0]*5
    mpid_force.setMScales(scales)
    mpid_force.setPScales(scales)
    mpid_force.setDScales(scales)
    
    type_map = {}
    polar_map = {}
    for child in admp:
        if child.tag == "Atom": type_map[child.get("type")] = child.attrib
        elif child.tag == "Polarize": polar_map[child.get("type")] = child.attrib

    for residue in pdb.topology.residues():
        atoms = list(residue.atoms())
        o_atom = next(a for a in atoms if a.name == "O")
        h_atoms = [a for a in atoms if "H" in a.name]
        h1, h2 = h_atoms[0], h_atoms[1]
        
        # PERFECT NEUTRALITY
        o_charge = np.float64(type_map["380"]['c0'])
        h_charge = -o_charge / 2.0
        
        res_map = {o_atom.index: "380", h1.index: "381", h2.index: "381"}
        for atom in atoms:
            i = atom.index
            t = res_map[i]
            p = type_map[t]
            pol = polar_map[t]
            charge = float(o_charge if atom.name == "O" else h_charge)
            # TEST: DO NOT SCALE UNITS
            dipole = [float(p['dX']), float(p['dY']), float(p['dZ'])]
            qxx, qxy, qxz = np.float64(p['qXX']), np.float64(p['qXY']), np.float64(p['qXZ'])
            qyy, qyz, qzz = np.float64(p['qYY']), np.float64(p['qYZ']), np.float64(p['qZZ'])
            trace_third = (qxx + qyy + qzz) / 3.0
            quad = [float(qxx-trace_third), float(qxy), float(qyy-trace_third), 
                    float(qxz), float(qyz), float(qzz-trace_third)]
            if atom.name == "O": axis_type, z, x, y = 4, h1.index, h2.index, -1
            else: axis_type, z, x, y = 0, o_atom.index, (h2.index if atom == h1 else h1.index), -1
            mpid_force.addMultipole(charge, dipole, quad, [0.0]*10, axis_type, z, x, y, 
                                    float(pol['thole']), [float(pol['polarizabilityXX'])]*3)
            mpid_force.setCovalentMap(i, mpidplugin.MPIDForce.Covalent12, [idx for idx in [o_atom.index, h1.index, h2.index] if idx != i])

    system.addForce(mpid_force)
    sr_start_group = system.getNumForces()
    add_dmff_short_range_forces_from_xml(system, pdb.topology, str(dmff_xml), start_group=sr_start_group)

    # 3. Dispersion Bridge
    box_nm = np.asarray(vectors.value_in_unit(unit.nanometer), dtype=np.float64)
    pos_nm = np.asarray(pdb.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    bridge = DispersionPMEBridgeModel(pdb.topology, str(dmff_xml), 0.6, box_nm, ethresh=1e-4, lpme=True)

    # 4. Context
    platform, props = choose_platform("Reference")
    simulation = app.Simulation(pdb.topology, system, mm.VerletIntegrator(1.0), platform, props)
    simulation.context.setPositions(pdb.positions)
    
    omm_total = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    disp_energy = float(bridge.eval(pos_nm))
    
    print(f"--- FINAL ATTEMPT ---")
    print(f"TOTAL SYSTEM ENERGY: {omm_total + disp_energy:.12f} kJ/mol")
    print(f"REFERENCE DMFF TOTAL: -10599.499605695088 kJ/mol")
    print(f"RESIDUAL DELTA: {omm_total + disp_energy - (-10599.499605695088):.12f} kJ/mol")

if __name__ == "__main__":
    main()
