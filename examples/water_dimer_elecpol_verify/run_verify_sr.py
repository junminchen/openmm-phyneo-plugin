#!/usr/bin/env python3
"""
Verify that MPID and DMFF compute the same Slater short-range interaction
energies for a water–water dimer.

Six terms compared (all use CutoffPeriodic with matched cutoff):
  QqTtDampingForce   — Tang-Toennies damped charge-charge
  SlaterExForce      — Slater exchange + (s12/r)^12 hardcore
  SlaterSrEsForce    — Short-range electrostatic (charge penetration)
  SlaterSrPolForce   — Short-range polarization correction
  SlaterSrDispForce  — Short-range dispersion correction
  SlaterDhfForce     — Deformed Hartree-Fock correction

Usage:
    conda run -n mpid python run_verify_sr.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
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
from dmff_sr_custom_forces import (
    TERM_ORDER,
    infer_atom_types_and_bonds_from_topology,
    parse_dmff_sr_xml,
    add_dmff_short_range_forces,
    shortest_bond_separations,
    make_nonbonded_force,
    make_bond_force,
    particle_params,
    pair_params,
    scale_for_bond_separation,
)


# ======================================================================
#  1.  MPID: build one OpenMM system per SR term
# ======================================================================
class MPIDSrCalculator:
    """OpenMM system with a single Slater SR force (NonbondedForce + BondForce)."""

    def __init__(self, pdb_path: str, ff_xml: str, force_name: str,
                 cutoff_nm: float = 1.2, s12: float = 0.169):
        self.pdb = app.PDBFile(pdb_path)
        self.force_name = force_name

        atom_types, bonds = infer_atom_types_and_bonds_from_topology(
            self.pdb.topology, ff_xml
        )
        force_data = parse_dmff_sr_xml(ff_xml)
        section = force_data[force_name]

        system = mm.System()
        for atom in self.pdb.topology.atoms():
            system.addParticle(atom.element.mass)

        box_vecs = self.pdb.topology.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vecs)

        # Create CustomNonbondedForce
        nb_force = make_nonbonded_force(force_name, s12=s12)
        nb_force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        nb_force.setCutoffDistance(cutoff_nm)

        # Create CustomBondForce
        bond_force = make_bond_force(force_name, s12=s12)

        for atom_type in atom_types:
            nb_force.addParticle(particle_params(force_name, section, atom_type))

        # Exclusions for 1-2, 1-3, 1-4 pairs
        nb_pairs = shortest_bond_separations(len(atom_types), bonds, max_sep=3)
        all_intra = shortest_bond_separations(len(atom_types), bonds, max_sep=4)
        sep4_pairs = {k: v for k, v in all_intra.items() if k not in nb_pairs}

        for (i, j), separation in nb_pairs.items():
            nb_force.addExclusion(i, j)
            scale = scale_for_bond_separation(section["mscales"], separation)
            if abs(scale) > 1e-15:
                bond_force.addBond(
                    i, j,
                    pair_params(force_name, section, atom_types[i], atom_types[j], scale),
                )

        for (i, j), separation in sep4_pairs.items():
            mscale = scale_for_bond_separation(section["mscales"], separation)
            corr_scale = mscale - 1.0
            if abs(corr_scale) > 1e-15:
                bond_force.addBond(
                    i, j,
                    pair_params(force_name, section, atom_types[i], atom_types[j], corr_scale),
                )

        system.addForce(nb_force)
        system.addForce(bond_force)

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
#  2.  DMFF: per-term energy from Hamiltonian
# ======================================================================
class DMFFSrCalculator:
    """DMFF potentials for all SR terms."""

    def __init__(self, pdb_path: str, ff_xml: str, cutoff_nm: float = 1.2):
        self.pdb = app.PDBFile(pdb_path)
        box_vecs = self.pdb.topology.getPeriodicBoxVectors()
        self.box_nm = jnp.array(box_vecs._value)
        self.cutoff_nm = cutoff_nm

        hamiltonian = Hamiltonian(ff_xml)
        potentials = hamiltonian.createPotential(
            self.pdb.topology,
            nonbondedCutoff=cutoff_nm * unit.nanometer,
            nonbondedMethod=app.CutoffPeriodic,
            ethresh=1e-4,
        )

        self.pot_fns = {}
        for name in TERM_ORDER:
            fn = potentials.dmff_potentials.get(name)
            if fn is not None:
                self.pot_fns[name] = fn

        self.params = hamiltonian.getParameters()
        self.cov_map = potentials.meta["cov_map"]
        self.base_positions = jnp.array(self.pdb.positions._value)

    def energy(self, force_name: str, positions_nm=None):
        if force_name not in self.pot_fns:
            return None
        if positions_nm is None:
            positions_nm = self.base_positions
        pos = jnp.array(positions_nm)

        nbl = nblist.NeighborListFreud(
            self.box_nm, self.cutoff_nm, self.cov_map, padding=False
        )
        nbl.allocate(pos, self.box_nm)
        return float(self.pot_fns[force_name](pos, self.box_nm, nbl.pairs, self.params))


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
    cutoff_nm = 1.2
    s12 = 0.169  # nm, hardcore parameter

    print("=" * 78)
    print(" Water-water dimer: MPID vs DMFF Slater short-range forces")
    print("=" * 78)
    print(f"PDB    : {PDB_FILE}")
    print(f"FF     : {FF_XML}")
    print(f"cutoff : {cutoff_nm} nm")
    print(f"s12    : {s12} nm (SlaterExForce hardcore)")

    # Build calculators
    mpid_calcs = {}
    for name in TERM_ORDER:
        mpid_calcs[name] = MPIDSrCalculator(
            PDB_FILE, FF_XML, name, cutoff_nm=cutoff_nm, s12=s12
        )
    dmff_calc = DMFFSrCalculator(PDB_FILE, FF_XML, cutoff_nm=cutoff_nm)

    base_pos = mpid_calcs[TERM_ORDER[0]].base_positions

    # ── single-point comparison at equilibrium ──
    print(f"\n--- Single-point at equilibrium (R_OO = 3.5 A) ---\n")
    print(f"{'Force':>22s}  {'E_MPID':>12s}  {'E_DMFF':>12s}  {'delta':>12s}")
    print("-" * 64)

    total_mpid = 0.0
    total_dmff = 0.0
    for name in TERM_ORDER:
        e_m = mpid_calcs[name].energy()
        e_d = dmff_calc.energy(name)
        if e_d is None:
            print(f"{name:>22s}  {e_m:12.6f}  {'N/A':>12s}  {'N/A':>12s}")
            total_mpid += e_m
            continue
        delta = e_m - e_d
        total_mpid += e_m
        total_dmff += e_d
        print(f"{name:>22s}  {e_m:12.6f}  {e_d:12.6f}  {delta:+12.6f}")

    delta_total = total_mpid - total_dmff
    print("-" * 64)
    print(f"{'TOTAL':>22s}  {total_mpid:12.6f}  {total_dmff:12.6f}  {delta_total:+12.6f}")

    # ── distance scan ──
    print("\n" + "=" * 78)
    print(" Distance scan (shift 2nd water along Y)")
    print("=" * 78)

    shifts = np.arange(-1.0, 8.1, 0.5)

    # Print per-term scan
    for name in TERM_ORDER:
        if dmff_calc.energy(name) is None:
            continue

        print(f"\n  [{name}]")
        print(f"  {'R_OO(A)':>8s}  {'E_MPID':>12s}  {'E_DMFF':>12s}  {'delta':>12s}")
        print(f"  " + "-" * 50)

        max_delta = 0.0
        for dy in shifts:
            pos = shift_second_water(base_pos, dy)
            r_oo = np.linalg.norm(pos[3] - pos[0]) * 10.0
            e_m = mpid_calcs[name].energy(pos)
            e_d = dmff_calc.energy(name, pos)
            delta = e_m - e_d
            max_delta = max(max_delta, abs(delta))
            print(f"  {r_oo:8.3f}  {e_m:12.6f}  {e_d:12.6f}  {delta:+12.6f}")

        print(f"  " + "-" * 50)
        print(f"  Max |delta| = {max_delta:.6f} kJ/mol")

    # ── total SR scan ──
    print(f"\n  [TOTAL (all SR terms)]")
    print(f"  {'R_OO(A)':>8s}  {'E_MPID':>12s}  {'E_DMFF':>12s}  {'delta':>12s}")
    print(f"  " + "-" * 50)

    max_delta_total = 0.0
    for dy in shifts:
        pos = shift_second_water(base_pos, dy)
        r_oo = np.linalg.norm(pos[3] - pos[0]) * 10.0
        e_m_tot = sum(mpid_calcs[n].energy(pos) for n in TERM_ORDER)
        e_d_tot = sum(
            dmff_calc.energy(n, pos) for n in TERM_ORDER
            if dmff_calc.energy(n, pos) is not None
        )
        delta = e_m_tot - e_d_tot
        max_delta_total = max(max_delta_total, abs(delta))
        print(f"  {r_oo:8.3f}  {e_m_tot:12.6f}  {e_d_tot:12.6f}  {delta:+12.6f}")

    print(f"  " + "-" * 50)
    print(f"  Max |delta(total SR)| = {max_delta_total:.6f} kJ/mol")

    if max_delta_total < 0.001:
        print(f"\n  PASS: all SR forces agree within {max_delta_total:.6f} kJ/mol.")
    elif max_delta_total < 0.01:
        print(f"\n  CLOSE: all SR forces agree within {max_delta_total:.6f} kJ/mol.")
    else:
        print(f"\n  WARNING: max discrepancy = {max_delta_total:.6f} kJ/mol.")


if __name__ == "__main__":
    main()
