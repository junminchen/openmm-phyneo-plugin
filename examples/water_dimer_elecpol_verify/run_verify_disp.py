#!/usr/bin/env python3
"""
Verify that MPID and DMFF compute the same damped dispersion energy
for a water–water dimer.

MPID dispersion:
    add_damped_dispersion_force() — a single CustomNonbondedForce with
    Tang-Toennies damped C6/C8/C10 (CutoffPeriodic).

DMFF dispersion:
    ADMPDispPmeForce  (undamped -C_n/r^n, PME or cutoff)
  + SlaterDampingForce (damping correction: (1 - f_n)*C_n/r^n)
  = -f_n * C_n/r^n  (same as MPID)

Usage:
    conda run -n mpid python run_verify_disp.py
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

# ── DMFF imports ──────────────────────────────────────────────────────
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import numpy as np

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
from dmff_sr_custom_forces import add_damped_dispersion_force


# ======================================================================
#  1.  MPID side: damped dispersion via CustomNonbondedForce
# ======================================================================
class MPIDDispCalculator:
    """OpenMM system with only the damped dispersion CustomNonbondedForce."""

    def __init__(self, pdb_path: str, ff_xml: str, cutoff_nm: float = 1.2):
        self.pdb = app.PDBFile(pdb_path)

        # Build a bare system (no forces at all)
        system = mm.System()
        for atom in self.pdb.topology.atoms():
            system.addParticle(atom.element.mass)

        # Set periodic box from PDB
        box_vecs = self.pdb.topology.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vecs)

        # Add only the damped dispersion force
        add_damped_dispersion_force(
            system, self.pdb.topology, ff_xml, cutoff_nm=cutoff_nm
        )

        platform = mm.Platform.getPlatformByName("Reference")
        integrator = mm.VerletIntegrator(0.001 * unit.femtoseconds)
        self.sim = app.Simulation(self.pdb.topology, system, integrator, platform)
        self.base_positions = np.array(
            [[v.x, v.y, v.z] for v in self.pdb.positions]
        )

    def energy(self, positions_nm=None):
        if positions_nm is None:
            positions_nm = self.base_positions
        self.sim.context.setPositions(positions_nm)
        state = self.sim.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


# ======================================================================
#  2.  DMFF side: ADMPDispPmeForce + SlaterDampingForce
# ======================================================================
class DMFFDispCalculator:
    """DMFF potential using ADMPDispPmeForce + SlaterDampingForce."""

    def __init__(self, pdb_path: str, ff_xml: str, use_pme: bool = True,
                 cutoff_nm: float = 1.2):
        self.pdb = app.PDBFile(pdb_path)

        hamiltonian = Hamiltonian(ff_xml)

        if use_pme:
            method = app.PME
        else:
            method = app.CutoffPeriodic

        self.cutoff_nm = cutoff_nm
        box_vecs = self.pdb.topology.getPeriodicBoxVectors()
        self.box_nm = jnp.array(box_vecs._value)

        potentials = hamiltonian.createPotential(
            self.pdb.topology,
            nonbondedCutoff=cutoff_nm * unit.nanometer,
            nonbondedMethod=method,
            ethresh=1e-4,
        )

        self.disp_fn = potentials.dmff_potentials.get("ADMPDispPmeForce")
        self.damp_fn = potentials.dmff_potentials.get("SlaterDampingForce")
        self.params = hamiltonian.getParameters()
        self.cov_map = potentials.meta["cov_map"]
        self.base_positions = jnp.array(self.pdb.positions._value)

        available = list(potentials.dmff_potentials.keys())
        if self.disp_fn is None:
            raise RuntimeError(f"ADMPDispPmeForce not found. Available: {available}")
        if self.damp_fn is None:
            raise RuntimeError(f"SlaterDampingForce not found. Available: {available}")

    def energy(self, positions_nm=None, components=False):
        if positions_nm is None:
            positions_nm = self.base_positions
        pos = jnp.array(positions_nm)

        nbl = nblist.NeighborListFreud(
            self.box_nm, self.cutoff_nm, self.cov_map, padding=False
        )
        nbl.allocate(pos, self.box_nm)
        pairs = nbl.pairs

        e_disp = float(self.disp_fn(pos, self.box_nm, pairs, self.params))
        e_damp = float(self.damp_fn(pos, self.box_nm, pairs, self.params))

        if components:
            return e_disp, e_damp, e_disp + e_damp
        return e_disp + e_damp


# ======================================================================
#  3.  Utilities
# ======================================================================
def shift_second_water(base_pos_nm, dy_angstrom):
    pos = np.array(base_pos_nm, dtype=np.float64)
    pos[3:, 1] += dy_angstrom * 0.1
    return pos


# ======================================================================
#  4.  Main
# ======================================================================
def main():
    print("=" * 72)
    print(" Water-water dimer: MPID vs DMFF damped dispersion (PME)")
    print("=" * 72)
    print(f"PDB : {PDB_FILE}")
    print(f"FF  : {FF_XML}")
    print()
    print("MPID: add_damped_dispersion_force (CustomNonbondedForce, CutoffPeriodic)")
    print("DMFF: ADMPDispPmeForce(PME) + SlaterDampingForce = total damped disp")

    cutoff_nm = 1.2

    # ── single-point with component breakdown ──
    print(f"\n--- Single-point at equilibrium (R_OO = 3.5 A, cutoff = {cutoff_nm} nm) ---")

    mpid = MPIDDispCalculator(PDB_FILE, FF_XML, cutoff_nm=cutoff_nm)
    dmff_pme = DMFFDispCalculator(PDB_FILE, FF_XML, use_pme=True, cutoff_nm=cutoff_nm)
    dmff_cut = DMFFDispCalculator(PDB_FILE, FF_XML, use_pme=False, cutoff_nm=cutoff_nm)

    e_mpid = mpid.energy()
    e_disp_pme, e_damp_pme, e_total_pme = dmff_pme.energy(components=True)
    e_disp_cut, e_damp_cut, e_total_cut = dmff_cut.energy(components=True)

    print(f"\n  MPID (damped disp)         = {e_mpid:12.6f} kJ/mol")
    print(f"  DMFF (PME):")
    print(f"    ADMPDispPmeForce         = {e_disp_pme:12.6f} kJ/mol  (undamped)")
    print(f"    SlaterDampingForce       = {e_damp_pme:12.6f} kJ/mol  (damping correction)")
    print(f"    total                    = {e_total_pme:12.6f} kJ/mol")
    print(f"    delta (MPID - DMFF PME)  = {e_mpid - e_total_pme:+.6f} kJ/mol")
    print(f"  DMFF (CutoffPeriodic):")
    print(f"    ADMPDispPmeForce         = {e_disp_cut:12.6f} kJ/mol")
    print(f"    SlaterDampingForce       = {e_damp_cut:12.6f} kJ/mol")
    print(f"    total                    = {e_total_cut:12.6f} kJ/mol")
    print(f"    delta (MPID - DMFF cut)  = {e_mpid - e_total_cut:+.6f} kJ/mol")

    # ── distance scan ──
    print("\n" + "=" * 72)
    print(" Distance scan (shift 2nd water along Y)")
    print("=" * 72)

    print(f"\n{'R_OO(A)':>8s}  {'E_MPID':>12s}  {'E_DMFF_PME':>12s}  {'E_DMFF_cut':>12s}  {'d(PME)':>10s}  {'d(cut)':>10s}")
    print("-" * 72)

    shifts = np.arange(-1.0, 8.1, 0.5)
    deltas_pme = []
    deltas_cut = []

    for dy in shifts:
        pos = shift_second_water(mpid.base_positions, dy)
        r_oo = np.linalg.norm(pos[3] - pos[0]) * 10.0

        e_m = mpid.energy(pos)
        e_dp = dmff_pme.energy(pos)
        e_dc = dmff_cut.energy(pos)
        d_pme = e_m - e_dp
        d_cut = e_m - e_dc

        deltas_pme.append(d_pme)
        deltas_cut.append(d_cut)

        print(f"{r_oo:8.3f}  {e_m:12.6f}  {e_dp:12.6f}  {e_dc:12.6f}  {d_pme:+10.6f}  {d_cut:+10.6f}")

    deltas_pme = np.array(deltas_pme)
    deltas_cut = np.array(deltas_cut)

    print("-" * 72)
    print(f"\nvs DMFF PME:              mean delta = {np.mean(deltas_pme):+.6f},  max |delta| = {np.max(np.abs(deltas_pme)):.6f} kJ/mol")
    print(f"vs DMFF CutoffPeriodic:   mean delta = {np.mean(deltas_cut):+.6f},  max |delta| = {np.max(np.abs(deltas_cut)):.6f} kJ/mol")

    # Check consistency: constant offset (self-energy) vs varying interaction delta
    mean_pme = np.mean(deltas_pme)
    dev_pme = np.max(np.abs(deltas_pme - mean_pme))
    mean_cut = np.mean(deltas_cut)
    dev_cut = np.max(np.abs(deltas_cut - mean_cut))

    print(f"\nInteraction consistency (deviation from mean delta):")
    print(f"  vs DMFF PME:            max dev = {dev_pme:.6f} kJ/mol")
    print(f"  vs DMFF CutoffPeriodic: max dev = {dev_cut:.6f} kJ/mol")

    best_dev = min(dev_pme, dev_cut)
    best_label = "PME" if dev_pme <= dev_cut else "CutoffPeriodic"
    if best_dev < 0.01:
        print(f"\nPASS: MPID and DMFF ({best_label}) dispersion agree within {best_dev:.4f} kJ/mol.")
    elif best_dev < 0.1:
        print(f"\nCLOSE: MPID and DMFF ({best_label}) dispersion agree within {best_dev:.4f} kJ/mol.")
    else:
        print(f"\nWARNING: max interaction discrepancy = {best_dev:.4f} kJ/mol.")


if __name__ == "__main__":
    main()
