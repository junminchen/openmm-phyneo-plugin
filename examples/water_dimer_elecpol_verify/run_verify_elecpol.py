#!/usr/bin/env python3
"""
Verify that MPID (OpenMM plugin) and DMFF compute the same
electrostatics + polarization energy for a water–water dimer.

Performs:
  1) Single-point comparison at equilibrium geometry (PME and NoCutoff).
  2) Distance scan: shift the second water along Y and compare total energies.

Usage:
    conda run -n mpid python run_verify_elecpol.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]                       # MPIDForceplugin_repo
sys.path.insert(0, str(REPO_ROOT / "python"))

PHYNEO_ROOT = Path(
    "/home/am3-peichenzhong-group/Documents/project/project_h2o_ea/phyneo-water-ethanol"
)
DMFF_ROOT = PHYNEO_ROOT / "3_MD" / "vendor" / "DMFF"
if str(DMFF_ROOT) not in sys.path:
    sys.path.insert(0, str(DMFF_ROOT))

FF_XML = str(
    PHYNEO_ROOT
    / "2_PES"
    / "training_backend_amoeba_damping"
    / "checks"
    / "latest"
    / "trained_params.xml"
)
PDB_FILE = str(SCRIPT_DIR / "H2O_H2O_dimer.pdb")

# ── DMFF imports (must come before OpenMM to ensure JAX x64 is on) ────
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import numpy as np

# Patch jax.config into sys.modules so older DMFF code finds it
import types as _types
if "jax.config" not in sys.modules:
    shim = _types.ModuleType("jax.config")
    shim.config = jax.config
    sys.modules["jax.config"] = shim

from dmff.api import Hamiltonian            # type: ignore
from dmff.common import nblist              # type: ignore

# ── OpenMM / MPID imports ─────────────────────────────────────────────
import openmm as mm
import openmm.app as app
from openmm import unit
import mpidplugin


# ======================================================================
#  1.  MPID (OpenMM plugin) – reusable context
# ======================================================================
class MPIDCalculator:
    """Thin wrapper around an OpenMM context that keeps only MPIDForce."""

    def __init__(self, pdb_path: str, ff_xml: str, use_pme: bool = True):
        self.pdb = app.PDBFile(pdb_path)
        ff = app.ForceField(ff_xml)

        method = app.PME if use_pme else app.NoCutoff
        system = ff.createSystem(
            self.pdb.topology,
            nonbondedMethod=method,
            nonbondedCutoff=1.2 * unit.nanometer,
            polarization="extrapolated",
            defaultTholeWidth=8,
        )

        # Remove everything except MPIDForce.
        # SWIG wraps MPIDForce as generic openmm.Force; identify via XML.
        forces_to_remove = []
        mpid_found = False
        for i in range(system.getNumForces()):
            xml_str = mm.XmlSerializer.serialize(system.getForce(i))
            if "MPIDForce" in xml_str or "Multipole" in xml_str:
                mpid_found = True
            else:
                forces_to_remove.append(i)
        assert mpid_found, "MPIDForce not found in system"
        for i in sorted(forces_to_remove, reverse=True):
            system.removeForce(i)

        platform = mm.Platform.getPlatformByName("Reference")
        integrator = mm.VerletIntegrator(0.001 * unit.femtoseconds)
        self.sim = app.Simulation(self.pdb.topology, system, integrator, platform)
        self.base_positions = np.array(
            [[v.x, v.y, v.z] for v in self.pdb.positions]
        )  # nm

    def energy(self, positions_nm=None):
        if positions_nm is None:
            positions_nm = self.base_positions
        self.sim.context.setPositions(positions_nm)
        state = self.sim.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


# ======================================================================
#  2.  DMFF – reusable calculator
# ======================================================================
class DMFFCalculator:
    """Thin wrapper around DMFF ADMPPmeForce potential."""

    def __init__(self, pdb_path: str, ff_xml: str, use_pme: bool = True,
                 box_nm: float = None, cutoff_nm: float = None):
        self.pdb = app.PDBFile(pdb_path)

        hamiltonian = Hamiltonian(ff_xml)

        if use_pme:
            method = app.PME
            self.cutoff_nm = cutoff_nm or 1.2
        else:
            method = app.CutoffPeriodic
            self.cutoff_nm = cutoff_nm or 2.5

        if box_nm is not None:
            self.box_nm = jnp.eye(3) * box_nm
        else:
            box_vecs = self.pdb.topology.getPeriodicBoxVectors()
            self.box_nm = jnp.array(box_vecs._value)

        potentials = hamiltonian.createPotential(
            self.pdb.topology,
            nonbondedCutoff=self.cutoff_nm * unit.nanometer,
            nonbondedMethod=method,
            ethresh=1e-4,
            step_pol=None,
        )

        self.admp_fn = potentials.dmff_potentials["ADMPPmeForce"]
        self.params = hamiltonian.getParameters()
        self.cov_map = potentials.meta["cov_map"]
        self.base_positions = jnp.array(self.pdb.positions._value)

    def energy(self, positions_nm=None):
        if positions_nm is None:
            positions_nm = self.base_positions
        pos = jnp.array(positions_nm) if not isinstance(positions_nm, jnp.ndarray) else positions_nm
        nbl = nblist.NeighborListFreud(
            self.box_nm, self.cutoff_nm, self.cov_map, padding=False
        )
        nbl.allocate(pos, self.box_nm)
        return float(self.admp_fn(pos, self.box_nm, nbl.pairs, self.params))


# ======================================================================
#  3.  Distance scan utilities
# ======================================================================
def shift_second_water(base_pos_nm, dy_angstrom):
    """
    Shift the second water molecule (atoms 3,4,5) along Y by dy_angstrom
    relative to the original PDB geometry.
    The PDB has O-O distance = 3.5 A along Y.
    """
    pos = np.array(base_pos_nm, dtype=np.float64)
    dy_nm = dy_angstrom * 0.1
    pos[3:, 1] += dy_nm
    return pos


# ======================================================================
#  4.  Main
# ======================================================================
def main():
    print("=" * 72)
    print(" Water-water dimer: MPID vs DMFF electrostatics + polarization")
    print("=" * 72)
    print(f"PDB : {PDB_FILE}")
    print(f"FF  : {FF_XML}")
    print(f"MPID polarization: extrapolated")
    print(f"DMFF polarization: iterative (converged)")

    # ── single-point comparison ──
    for label, use_pme in [("PME", True), ("NoCutoff", False)]:
        print(f"\n--- {label} mode ---")
        mpid = MPIDCalculator(PDB_FILE, FF_XML, use_pme=use_pme)
        if use_pme:
            dmff = DMFFCalculator(PDB_FILE, FF_XML, use_pme=True)
        else:
            dmff = DMFFCalculator(PDB_FILE, FF_XML, use_pme=False,
                                  box_nm=6.0, cutoff_nm=2.5)

        e_mpid = mpid.energy()
        e_dmff = dmff.energy()
        delta = e_mpid - e_dmff
        print(f"  MPID  = {e_mpid:12.6f} kJ/mol")
        print(f"  DMFF  = {e_dmff:12.6f} kJ/mol")
        print(f"  delta = {delta:+.6f} kJ/mol")

    # ── distance scan (PME) ──
    print("\n" + "=" * 72)
    print(" Distance scan (PME, shift 2nd water along Y)")
    print("=" * 72)

    mpid_calc = MPIDCalculator(PDB_FILE, FF_XML, use_pme=True)
    dmff_calc = DMFFCalculator(PDB_FILE, FF_XML, use_pme=True)

    shifts = np.arange(-1.0, 8.1, 0.5)  # angstrom shift from PDB geometry
    print(f"\n{'R_OO(A)':>8s}  {'E_MPID':>12s}  {'E_DMFF':>12s}  {'delta':>12s}")
    print("-" * 50)

    deltas_phys = []  # deltas for R >= 3.0 A (physically relevant)
    for dy in shifts:
        pos = shift_second_water(mpid_calc.base_positions, dy)
        r_oo = np.linalg.norm(pos[3] - pos[0]) * 10.0

        e_m = mpid_calc.energy(pos)
        e_d = dmff_calc.energy(pos)
        delta = e_m - e_d

        marker = ""
        if r_oo >= 3.0:
            deltas_phys.append(delta)
        else:
            marker = "  (short range)"

        print(f"{r_oo:8.3f}  {e_m:12.6f}  {e_d:12.6f}  {delta:+12.6f}{marker}")

    deltas_phys = np.array(deltas_phys)

    print("-" * 50)
    print(f"\nFor R_OO >= 3.0 A ({len(deltas_phys)} points):")
    print(f"  Mean delta     = {np.mean(deltas_phys):+.6f} kJ/mol")
    print(f"  Max |delta|    = {np.max(np.abs(deltas_phys)):.6f} kJ/mol")
    print(f"  Std dev(delta) = {np.std(deltas_phys):.6f} kJ/mol")

    max_abs = np.max(np.abs(deltas_phys))
    if max_abs < 0.1:
        print(f"\nPASS: MPID and DMFF agree within {max_abs:.4f} kJ/mol for R >= 3.0 A.")
    elif max_abs < 1.0:
        print(f"\nCLOSE: MPID and DMFF agree within {max_abs:.4f} kJ/mol for R >= 3.0 A.")
    else:
        print(f"\nWARNING: max discrepancy = {max_abs:.4f} kJ/mol for R >= 3.0 A.")

    print("\nNote: Short-range (R < 3 A) discrepancies are expected due to")
    print("      different polarization solvers (extrapolated vs iterative).")


if __name__ == "__main__":
    main()
