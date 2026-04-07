#!/opt/anaconda3/envs/mpid84/bin/python
"""
Dimer scan: compare E_es and E_pol across three methods.

Parameters (same for all three):
  O charge = -1.0614 e,  alpha_O = 0.00088 nm³ = 0.88 Å³
  H charge = +0.5307 e,  alpha_H = 0
  Permanent dipoles/quadrupoles = 0

Methods:
  Plugin  - PhyNEO OpenMM, NoCutoff, extrapolated polarization, thole_width=5.0
  DMFF    - JAX, PME (large box), iterative polarization, thole_width=5.0
  BFF     - numpy direct linear solver, BFF Thole damping a=0.39
"""
from __future__ import annotations
import os, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[1]

# Plugin build path (Reference platform library + python bindings)
PLUGIN_BUILD = REPO_ROOT / "build-reference-check" / "python"
if PLUGIN_BUILD.exists():
    sys.path.insert(0, str(PLUGIN_BUILD))
sys.path.insert(0, str(REPO_ROOT / "python"))

os.environ.setdefault("JAX_ENABLE_X64", "True")

import types as _types
import jax, jax.numpy as jnp, numpy as np

import types as _t
if "jax.config" not in sys.modules:
    shim = _t.ModuleType("jax.config")
    shim.config = jax.config
    sys.modules["jax.config"] = shim

from dmff.api import Hamiltonian
from dmff.common import nblist as _nblist

import openmm as mm
import openmm.app as app
from openmm import unit

# Load Reference platform plugin
for _bd in sorted(REPO_ROOT.glob("build*/")):
    _lib = _bd / "platforms" / "reference" / "libOpenMMPhyNEOForceReference.dylib"
    if _lib.exists():
        mm.Platform.loadPluginLibrary(str(_lib))
        break
import phyneoforceplugin  # noqa

XML_POL   = str(SCRIPT_DIR / "mpidwater_chargepol.xml")
XML_NOPOL = str(SCRIPT_DIR / "mpidwater_chargeonlynopol.xml")
PDB_FILE  = str(SCRIPT_DIR / "H2O_H2O_dimer.pdb")

# Water atom order in PDB (per residue): H1(381), H2(381), O(380)
Q_O = -1.0614; Q_H = 0.5307
A_O = 0.00088; A_H = 0.0     # nm³
CHARGES = np.array([Q_H, Q_H, Q_O, Q_H, Q_H, Q_O])
ALPHAS  = np.array([A_H, A_H, A_O, A_H, A_H, A_O])  # nm³
MOL     = np.array([0, 0, 0, 1, 1, 1])
# Topology: residue order is H1,H2,O per water → atom 2=O1, atom 5=O2
BONDS_12 = [(0,2),(1,2),(3,5),(4,5)]
BONDS_13 = [(0,1),(3,4)]
DIELECTRIC = 1389.35  # kJ·Å / (mol·e²)


# ─────────────────────────────────────────────
# Plugin calculator (keep only ADMPPmeForce, NoCutoff)
# ─────────────────────────────────────────────
class PluginCalc:
    def __init__(self, ff_xml, thole_width=5.0):
        pdb = app.PDBFile(PDB_FILE)
        ff  = app.ForceField(ff_xml)
        system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            polarization="extrapolated",
            defaultTholeWidth=thole_width,
        )
        to_rm = [i for i in range(system.getNumForces())
                 if "ADMPPmeForce" not in mm.XmlSerializer.serialize(system.getForce(i))
                 and "Multipole" not in mm.XmlSerializer.serialize(system.getForce(i))]
        for i in sorted(to_rm, reverse=True):
            system.removeForce(i)

        platform = mm.Platform.getPlatformByName("Reference")
        integ    = mm.VerletIntegrator(0.001 * unit.femtoseconds)
        self.sim = app.Simulation(pdb.topology, system, integ, platform)
        self.system = system

    def energy(self, pos_nm):
        self.sim.context.setPositions(pos_nm * unit.nanometer)
        s = self.sim.context.getState(getEnergy=True)
        return s.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    def induced_dipoles(self, pos_nm):
        """Return induced dipoles (e·nm) for all atoms."""
        self.sim.context.setPositions(pos_nm * unit.nanometer)
        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            if hasattr(f, 'getInducedDipoles'):
                return np.array(f.getInducedDipoles(self.sim.context))
        return None


# ─────────────────────────────────────────────
# DMFF calculator (PME, large box)
# ─────────────────────────────────────────────
class DMFFCalc:
    def __init__(self, ff_xml, box_nm=6.0, cutoff_nm=2.5):
        pdb = app.PDBFile(PDB_FILE)
        self.box_A    = jnp.eye(3) * box_nm * 10.0
        self.cutoff_nm = cutoff_nm

        ham  = Hamiltonian(ff_xml)
        pots = ham.createPotential(
            pdb.topology,
            nonbondedCutoff=cutoff_nm * unit.nanometer,
            nonbondedMethod=app.CutoffPeriodic,
            ethresh=1e-5,
            step_pol=None,
        )
        self.fn     = pots.dmff_potentials["ADMPPmeForce"]
        self.params = ham.getParameters()
        self.cov    = pots.meta["cov_map"]

    def energy(self, pos_nm):
        pos_A = jnp.array(pos_nm * 10.0)
        nbl   = _nblist.NeighborListFreud(self.box_A, self.cutoff_nm, self.cov, padding=False)
        nbl.allocate(pos_A, self.box_A)
        return float(self.fn(pos_A, self.box_A, nbl.pairs, self.params))


# ─────────────────────────────────────────────
# BFF numpy implementation
# ─────────────────────────────────────────────
def bff_epol(coords_nm, charges, alphas_nm3, a=0.39,
             excl_12=None, excl_13=None):
    """
    Polarization energy via direct linear solver + BFF Thole damping.

    Thole (BFF, a=0.39):
        u   = r / (ai*aj)^(1/6)
        D1  = 1 - exp(-a*u^3)           [field damping]
        D2  = 1 - (1+a*u^3)*exp(-a*u^3) [tensor damping]

    Equation (in Å, e, dimensionless k_e factor pulled out):
        [alpha_i^{-1} + T_ij] mu = E_perm
    where
        E_perm_i = sum_j D1_ij * q_j / r^2 * r_hat  [e/Å²]
        T_ij     = (3*r̂r̂*D2 - I*D1) / r^3           [Å^{-3}]

    E_pol [kJ/mol] = -0.5 * DIELECTRIC * mu · E_perm
    """
    coords  = coords_nm * 10.0
    alphas  = alphas_nm3 * 1000.0
    N       = len(charges)

    scale = np.ones((N, N)) - np.eye(N)
    for (i,j) in (excl_12 or []):
        scale[i,j] = scale[j,i] = 0.0
    for (i,j) in (excl_13 or []):
        scale[i,j] = scale[j,i] = 0.0

    pol_idx = [i for i in range(N) if alphas[i] > 1e-10]
    Np      = len(pol_idx)
    if Np == 0:
        return 0.0, None
    idx_map = {atom: ii for ii, atom in enumerate(pol_idx)}

    E_field = np.zeros((Np, 3))  # e/Å²
    T       = np.zeros((Np*3, Np*3))  # Å^{-3}

    for ii, i in enumerate(pol_idx):
        for j in range(N):
            if j == i:
                continue
            s = scale[i,j]
            if abs(s) < 1e-12:
                continue
            r_vec = coords[i] - coords[j]
            r     = np.linalg.norm(r_vec)
            r_hat = r_vec / r

            if alphas[j] > 1e-10:
                pol_int  = np.sqrt(alphas[i] * alphas[j])
                damp_val = max(-a * r**3 / pol_int, -1e6)
                exp_d    = np.exp(damp_val)
                D1       = 1.0 - exp_d
                D2       = max(1.0 - (1.0 - damp_val)*exp_d, 1e-6)
            else:
                D1, D2 = 1.0, None

            E_field[ii] += s * D1 * charges[j] * r_hat / r**2

            if j in idx_map:
                jj = idx_map[j]
                rr   = np.outer(r_hat, r_hat)
                T_ij = s * (3.0*rr*D2 - np.eye(3)*D1) / r**3
                T[ii*3:ii*3+3, jj*3:jj*3+3] += T_ij

    A = np.zeros((Np*3, Np*3))
    for ii, i in enumerate(pol_idx):
        A[ii*3:ii*3+3, ii*3:ii*3+3] = np.eye(3) / alphas[i]
    A += T

    E_flat  = E_field.flatten()
    mu_flat = np.linalg.solve(A, E_flat)  # e·Å

    e_pol = -0.5 * DIELECTRIC * np.dot(mu_flat, E_flat)
    return e_pol, mu_flat.reshape(Np, 3)


def coulomb_es(coords_nm, charges, mol_assign):
    """Intermolecular Coulomb [kJ/mol], analytical direct sum."""
    coords = coords_nm * 10.0
    N = len(charges)
    E = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if mol_assign[i] != mol_assign[j]:
                r  = np.linalg.norm(coords[i]-coords[j])
                E += DIELECTRIC * charges[i]*charges[j] / r
    return E


# ─────────────────────────────────────────────
# Geometry builder
# ─────────────────────────────────────────────
def water_dimer_pos(r_oo_ang):
    """Build O-O scan geometry; O along Y-axis; matches H2O_H2O_dimer.pdb atom order."""
    dOH  = 0.9572   # Å
    half = np.radians(104.52 / 2.0)
    dy   = dOH * np.sin(half)
    dz   = -dOH * np.cos(half)
    H1a  = np.array([0., dy,  dz])
    H1b  = np.array([0., -dy, dz])
    O1   = np.array([0., 0.,  0.])
    H2a  = np.array([0., r_oo_ang + dy,  dz])
    H2b  = np.array([0., r_oo_ang - dy,  dz])
    O2   = np.array([0., r_oo_ang,        0.])
    # Per-residue order: H1, H2, O
    return np.array([H1a, H1b, O1, H2a, H2b, O2]) / 10.0  # nm


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("Initialising calculators ...")
    plug_pol   = PluginCalc(XML_POL)
    plug_nopol = PluginCalc(XML_NOPOL)
    dmff_pol   = DMFFCalc(XML_POL)
    # Note: DMFF with alpha=0 crashes (zero-size array in JAX reduction).
    # We use plug_nopol.energy() as the E_es baseline for DMFF instead.
    # (Plugin NoCutoff ≈ direct Coulomb; DMFF PME with large box ≈ same)
    print("Done.\n")

    r_scan = [2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.5, 5.0, 6.0, 8.0]

    hdr = (f"{'R(Å)':>6} | {'E_es_ana':>10} | "
           f"{'E_pol_plugin':>13} {'E_pol_dmff':>11} {'E_pol_bff':>10} | "
           f"{'mu_O1_plugin(me·nm)':>20} {'mu_O1_bff(me·nm)':>17}")
    print(hdr)
    print("-" * len(hdr))

    for r_oo in r_scan:
        pos = water_dimer_pos(r_oo)

        # E_es (analytical Coulomb)
        e_es = coulomb_es(pos, CHARGES, MOL)

        # Plugin
        e_plug_pol   = plug_pol.energy(pos)
        e_plug_nopol = plug_nopol.energy(pos)
        e_pol_plugin = e_plug_pol - e_plug_nopol
        dipoles      = plug_pol.induced_dipoles(pos)
        mu_plug = np.linalg.norm(dipoles[2]) * 1000 if dipoles is not None else float('nan')  # me·nm (O1=atom2)

        # DMFF (use plug_nopol as E_es baseline)
        e_dmff_pol = dmff_pol.energy(pos)
        e_pol_dmff = e_dmff_pol - e_plug_nopol

        # BFF
        e_pol_bff, mu_bff = bff_epol(pos, CHARGES, ALPHAS, a=0.39,
                                      excl_12=BONDS_12, excl_13=BONDS_13)
        mu_bff_mag = np.linalg.norm(mu_bff[0]) / 10.0 * 1000  # e·Å→e·nm→me·nm (O1=idx0 in pol_idx)

        print(f"{r_oo:6.1f} | {e_es:+10.4f} | "
              f"{e_pol_plugin:+13.5f} {e_pol_dmff:+11.5f} {e_pol_bff:+10.5f} | "
              f"{mu_plug:20.5f} {mu_bff_mag:17.5f}")

    print()
    print("Notes:")
    print("  E_es_ana   : intermolecular Coulomb (analytic) — identical input for all")
    print("  E_pol_*    : E_total(with_pol) - E_total(no_pol)   [kJ/mol]")
    print("  Plugin     : extrapolated polarization, Thole_width=5.0 (linear)")
    print("  DMFF       : iterative converged,       Thole_width=5.0 (linear)")
    print("  DMFF note  : E_pol = E_dmff_total - E_plugin_nopol (DMFF zero-alpha fails)")
    print("  BFF        : direct linear solver,      a=0.39    (cubic u³)")
    print()

    # Thole damping comparison
    print("Thole D1(r) for O-O pair (alpha_O=0.88 Å³):")
    print(f"  {'r(Å)':>6}  {'u':>8}  {'BFF D1(a=0.39)':>15}  {'DMFF D1(w=5.0)':>15}  {'diff':>8}")
    alpha_O = 0.88
    for r in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        u     = r / (alpha_O**2) ** (1./6.)
        damp  = -0.39 * u**3
        D1b   = 1.0 - np.exp(damp)
        au    = 5.0 * u
        D1d   = 1.0 - np.exp(-au)*(1 + au + au**2/2.)
        print(f"  {r:6.1f}  {u:8.4f}  {D1b:15.6f}  {D1d:15.6f}  {D1b-D1d:+8.6f}")


if __name__ == "__main__":
    main()
