#!/usr/bin/env python3
"""
Compare PhyNEO plugin vs DMFF ADMPPmeForce energy for a water dimer
using the local mpidwater.xml force field.

Usage:
    /opt/anaconda3/envs/mpid84/bin/python run_mpidwater_compare.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

PLUGIN_BUILD = REPO_ROOT / "build-reference-check" / "python"
if PLUGIN_BUILD.exists():
    sys.path.insert(0, str(PLUGIN_BUILD))

FF_XML   = str(SCRIPT_DIR / "mpidwater_lmax2.xml")
PDB_FILE = str(SCRIPT_DIR / "H2O_H2O_dimer.pdb")

# ── DMFF first (needs JAX x64) ─────────────────────────────────────────
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import numpy as np

import types as _types
if "jax.config" not in sys.modules:
    shim = _types.ModuleType("jax.config")
    shim.config = jax.config
    sys.modules["jax.config"] = shim

from dmff.api import Hamiltonian          # type: ignore
from dmff.common import nblist as _nblist # type: ignore

# ── OpenMM / plugin ────────────────────────────────────────────────────
import openmm as mm
import openmm.app as app
from openmm import unit

# load reference platform plugin
for build_dir in sorted(REPO_ROOT.glob("build*/")):
    ref_lib = build_dir / "platforms" / "reference" / "libOpenMMPhyNEOForceReference.dylib"
    if ref_lib.exists():
        mm.Platform.loadPluginLibrary(str(ref_lib))
        break

import phyneoforceplugin   # noqa: E402


# ======================================================================
#  Plugin calculator  (keeps only ADMPPmeForce)
# ======================================================================
class PluginCalc:
    def __init__(self, pdb_path, ff_xml, use_pme=True, thole_width=5.0):
        self.pdb = app.PDBFile(pdb_path)
        ff = app.ForceField(ff_xml)

        method = app.PME if use_pme else app.NoCutoff
        system = ff.createSystem(
            self.pdb.topology,
            nonbondedMethod=method,
            nonbondedCutoff=1.2 * unit.nanometer,
            polarization="extrapolated",
            defaultTholeWidth=thole_width,
        )

        # keep only ADMPPmeForce
        to_remove = []
        found = False
        for i in range(system.getNumForces()):
            xml = mm.XmlSerializer.serialize(system.getForce(i))
            if "ADMPPmeForce" in xml or "Multipole" in xml:
                found = True
            else:
                to_remove.append(i)
        assert found, "ADMPPmeForce not found in system"
        for i in sorted(to_remove, reverse=True):
            system.removeForce(i)

        platform = mm.Platform.getPlatformByName("Reference")
        integ = mm.VerletIntegrator(0.001 * unit.femtoseconds)
        self.sim = app.Simulation(self.pdb.topology, system, integ, platform)
        self.base_pos = np.array([[v.x, v.y, v.z] for v in self.pdb.positions])

    def energy(self, pos=None):
        if pos is None:
            pos = self.base_pos
        self.sim.context.setPositions(pos)
        state = self.sim.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


# ======================================================================
#  DMFF calculator
# ======================================================================
class DMFFCalc:
    def __init__(self, pdb_path, ff_xml, use_pme=True, box_nm=None, cutoff_nm=None):
        self.pdb = app.PDBFile(pdb_path)

        if use_pme:
            method = app.PME
            self.cutoff_nm = cutoff_nm or 1.2
        else:
            method = app.CutoffPeriodic
            self.cutoff_nm = cutoff_nm or 2.5

        if box_nm is not None:
            self.box_nm = jnp.eye(3) * box_nm
        else:
            bv = self.pdb.topology.getPeriodicBoxVectors()
            self.box_nm = jnp.array(bv._value)

        ham = Hamiltonian(ff_xml)
        pots = ham.createPotential(
            self.pdb.topology,
            nonbondedCutoff=self.cutoff_nm * unit.nanometer,
            nonbondedMethod=method,
            ethresh=1e-4,
            step_pol=None,
        )
        self.fn     = pots.dmff_potentials["ADMPPmeForce"]
        self.params = ham.getParameters()
        self.cov    = pots.meta["cov_map"]
        self.base_pos = jnp.array(self.pdb.positions._value)

    def energy(self, pos=None):
        if pos is None:
            pos = self.base_pos
        pos = jnp.array(pos)
        nbl = _nblist.NeighborListFreud(self.box_nm, self.cutoff_nm, self.cov, padding=False)
        nbl.allocate(pos, self.box_nm)
        return float(self.fn(pos, self.box_nm, nbl.pairs, self.params))


# ======================================================================
#  Utilities
# ======================================================================
def shift_mol2(base_pos_nm, dy_angstrom):
    """Shift second water (atoms 3-5) along Y."""
    pos = np.array(base_pos_nm, dtype=np.float64)
    pos[3:, 1] += dy_angstrom * 0.1
    return pos


def roo(pos_nm):
    return float(np.linalg.norm(pos_nm[3] - pos_nm[0]) * 10.0)  # Å


# ======================================================================
#  Main
# ======================================================================
def main():
    print("=" * 70)
    print("  Water dimer: PhyNEO plugin vs DMFF  (mpidwater.xml)")
    print("=" * 70)
    print(f"  PDB  : {PDB_FILE}")
    print(f"  FF   : {FF_XML}")
    print(f"  Plugin polarization : extrapolated  (defaultTholeWidth=5.0)")
    print(f"  DMFF   polarization : iterative (converged)")

    # ── single-point at equilibrium ──────────────────────────────────
    for label, use_pme in [("PME", True), ("NoCutoff", False)]:
        print(f"\n--- {label} ---")
        plug = PluginCalc(PDB_FILE, FF_XML, use_pme=use_pme)
        if use_pme:
            dmff = DMFFCalc(PDB_FILE, FF_XML, use_pme=True)
        else:
            dmff = DMFFCalc(PDB_FILE, FF_XML, use_pme=False, box_nm=6.0, cutoff_nm=2.5)

        e_plug = plug.energy()
        e_dmff = dmff.energy()
        delta  = e_plug - e_dmff
        r      = roo(plug.base_pos)
        print(f"  R_OO = {r:.3f} Å")
        print(f"  Plugin = {e_plug:12.6f}  kJ/mol")
        print(f"  DMFF   = {e_dmff:12.6f}  kJ/mol")
        print(f"  delta  = {delta:+.6f}  kJ/mol")

    # ── distance scan (PME) ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Distance scan — PME, shift 2nd water along Y")
    print("=" * 70)

    plug = PluginCalc(PDB_FILE, FF_XML, use_pme=True)
    dmff = DMFFCalc(PDB_FILE, FF_XML, use_pme=True)

    shifts = np.arange(-1.0, 8.6, 0.5)      # Å
    print(f"\n  {'R_OO(Å)':>8}  {'Plugin':>12}  {'DMFF':>12}  {'delta':>12}")
    print("  " + "-" * 50)

    deltas_phys = []
    for dy in shifts:
        pos  = shift_mol2(plug.base_pos, dy)
        r    = roo(pos)
        e_p  = plug.energy(pos)
        e_d  = dmff.energy(pos)
        dE   = e_p - e_d
        flag = "" if r >= 3.0 else "  ← short range"
        if r >= 3.0:
            deltas_phys.append(dE)
        print(f"  {r:8.3f}  {e_p:12.6f}  {e_d:12.6f}  {dE:+12.6f}{flag}")

    deltas_phys = np.array(deltas_phys)
    print("  " + "-" * 50)
    print(f"\n  R_OO ≥ 3.0 Å  ({len(deltas_phys)} points):")
    print(f"    mean delta    = {np.mean(deltas_phys):+.6f}  kJ/mol")
    print(f"    max |delta|   = {np.max(np.abs(deltas_phys)):.6f}  kJ/mol")
    print(f"    std  delta    = {np.std(deltas_phys):.6f}  kJ/mol")

    max_abs = np.max(np.abs(deltas_phys))
    if max_abs < 0.1:
        verdict = f"PASS  (max |Δ| = {max_abs:.4f} kJ/mol)"
    elif max_abs < 1.0:
        verdict = f"CLOSE (max |Δ| = {max_abs:.4f} kJ/mol)"
    else:
        verdict = f"WARN  (max |Δ| = {max_abs:.4f} kJ/mol)"
    print(f"\n  {verdict}")
    print("\n  Note: short-range discrepancies (R < 3 Å) are expected —")
    print("  plugin uses extrapolated polarization; DMFF uses iterative.")


if __name__ == "__main__":
    main()
