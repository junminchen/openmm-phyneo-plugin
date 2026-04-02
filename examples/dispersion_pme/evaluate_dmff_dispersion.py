#!/usr/bin/env python3
"""Evaluate DMFF ADMPDispPmeForce for one OpenMM frame."""

from __future__ import annotations

import argparse

import openmm.app as app
import openmm.unit as unit

from dispersion_pme_bridge import DispersionPMEBridgeModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True, help="Input PDB file.")
    parser.add_argument("--dmff-xml", required=True, help="DMFF XML containing ADMPDispPmeForce.")
    parser.add_argument("--cutoff-nm", type=float, required=True, help="Nonbonded cutoff in nm.")
    parser.add_argument("--ethresh", type=float, default=5e-4, help="DMFF Ewald threshold.")
    parser.add_argument(
        "--no-lpme",
        action="store_true",
        help="Disable long-range PME and use the cutoff-periodic real-space form.",
    )
    args = parser.parse_args()

    pdb = app.PDBFile(args.pdb)
    if pdb.topology.getPeriodicBoxVectors() is None:
        raise ValueError("Input PDB must define periodic box vectors for dispersion PME evaluation.")

    box_vectors = pdb.topology.getPeriodicBoxVectors()
    box_nm = [[vec[i].value_in_unit(unit.nanometer) for i in range(3)] for vec in box_vectors]
    positions_nm = pdb.positions.value_in_unit(unit.nanometer)

    model = DispersionPMEBridgeModel(
        topology=pdb.topology,
        dmff_xml=args.dmff_xml,
        cutoff_nm=args.cutoff_nm,
        box_nm=box_nm,
        ethresh=args.ethresh,
        lpme=not args.no_lpme,
    )
    energy_kj_mol = model.eval(positions_nm)
    print(f"ADMPDispPmeForce energy: {energy_kj_mol:.12f} kJ/mol")


if __name__ == "__main__":
    main()
