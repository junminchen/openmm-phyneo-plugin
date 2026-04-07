#!/usr/bin/env python3
"""
Run the legacy MPID water XML on the currently available plugin wrapper.

This script keeps the old water force field definition from MPIDOpenMMPlugin
(`mpidwater.xml`) while reusing the shared H2O_H2O_dimer.pdb geometry.  It
isolates the multipole force so the reported energy can be compared directly
against run_new_mpidwater_lmax2.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit


CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parents[2]
OLD_EXAMPLE_DIR = PROJECT_ROOT / "MPIDOpenMMPlugin" / "examples" / "water_dimer"
FF_XML = CURRENT_SCRIPT_DIR / "mpidwater.xml"
# PDB_FILE = CURRENT_SCRIPT_DIR / "H2O_H2O_dimer.pdb"
PDB_FILE = CURRENT_SCRIPT_DIR / "waterbox_31ang.pdb"

for build_dir in sorted((PROJECT_ROOT / "OpenMMPhyNEOPlugin").glob("build*/")):
    ref_lib = build_dir / "platforms" / "reference" / "libOpenMMPhyNEOForceReference.dylib"
    if ref_lib.exists():
        mm.Platform.loadPluginLibrary(str(ref_lib))
        break

try:
    import phyneoforceplugin  # noqa: F401, E402
except ImportError as exc:
    raise SystemExit(
        "Could not import phyneoforceplugin from the current Python environment."
    ) from exc


def shift_mol2(base_pos_nm: np.ndarray, dy_angstrom: float) -> np.ndarray:
    pos = np.array(base_pos_nm, dtype=np.float64)
    pos[3:, 1] += dy_angstrom * 0.1
    return pos


def roo(pos_nm: np.ndarray) -> float:
    return float(np.linalg.norm(pos_nm[3] - pos_nm[0]) * 10.0)


class OldMPIDWaterCalc:
    def __init__(self, ff_xml: Path, pdb_file: Path, method: str, thole_width: float):
        self.pdb = app.PDBFile(str(pdb_file))
        ff = app.ForceField(str(ff_xml))
        nonbonded_method = app.PME if method == "PME" else app.NoCutoff
        cutoff = 0.8 * unit.nanometer
        system = ff.createSystem(
            self.pdb.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=cutoff,
            constraints=None,
            polarization="extrapolated",
            defaultTholeWidth=thole_width,
        )

        for index in range(system.getNumForces() - 1, -1, -1):
            xml = mm.XmlSerializer.serialize(system.getForce(index))
            if "ADMPPmeForce" not in xml and "MPIDForce" not in xml and "Multipole" not in xml:
                system.removeForce(index)

        integrator = mm.VerletIntegrator(1e-6 * unit.femtoseconds)
        platform = mm.Platform.getPlatformByName("Reference")
        self.sim = app.Simulation(self.pdb.topology, system, integrator, platform)
        self.base_pos = np.array([[v.x, v.y, v.z] for v in self.pdb.positions], dtype=np.float64)

    def energy(self, pos_nm: np.ndarray | None = None) -> float:
        self.sim.context.setPositions(self.base_pos if pos_nm is None else pos_nm)
        state = self.sim.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["PME", "NoCutoff"],
        default="PME",
        help="OpenMM nonbonded method for the legacy XML.",
    )
    parser.add_argument(
        "--default-thole-width",
        type=float,
        default=8.0,
        help="defaultTholeWidth passed to ForceField.createSystem().",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Also run a Y-shift scan of the second water molecule.",
    )
    parser.add_argument(
        "--scan-start",
        type=float,
        default=-1.0,
        help="Starting Y shift in angstrom for the second water.",
    )
    parser.add_argument(
        "--scan-stop",
        type=float,
        default=8.5,
        help="Final Y shift in angstrom for the second water.",
    )
    parser.add_argument(
        "--scan-step",
        type=float,
        default=0.5,
        help="Step size in angstrom for the Y-shift scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calc = OldMPIDWaterCalc(FF_XML, PDB_FILE, args.method, args.default_thole_width)

    print("=" * 70)
    print("  Legacy MPID water XML on current plugin")
    print("=" * 70)
    print(f"  PDB    : {PDB_FILE}")
    print(f"  XML    : {FF_XML}")
    print(f"  Method : {args.method}")
    print(f"  defaultTholeWidth : {args.default_thole_width}")
    print(f"  Module : {Path(phyneoforceplugin.__file__).resolve()}")
    print(f"  R_OO   : {roo(calc.base_pos):.3f} Å")
    print(f"  Energy : {calc.energy():.12f} kJ/mol")

    if not args.scan:
        return

    print("\n" + "=" * 70)
    print("  Distance scan — shift 2nd water along Y")
    print("=" * 70)
    print(f"\n  {'shift(Å)':>10}  {'R_OO(Å)':>8}  {'Energy(kJ/mol)':>18}")
    print("  " + "-" * 42)

    shifts = np.arange(args.scan_start, args.scan_stop + 0.5 * args.scan_step, args.scan_step)
    for dy in shifts:
        pos = shift_mol2(calc.base_pos, dy)
        print(f"  {dy:10.3f}  {roo(pos):8.3f}  {calc.energy(pos):18.12f}")


if __name__ == "__main__":
    main()
