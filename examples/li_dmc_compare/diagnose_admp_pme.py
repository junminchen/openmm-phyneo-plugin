#!/usr/bin/env python3
"""Diagnose where ADMPPmeForce differs from the DMFF/check_lr reference."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import jax
import jax.numpy as jnp
import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

import types as _types

if "jax.config" not in sys.modules:
    shim = _types.ModuleType("jax.config")
    shim.config = jax.config
    sys.modules["jax.config"] = shim

from example_fixture import (  # noqa: E402
    BATCH,
    BOX_NM,
    CUTOFF_NM,
    DATA_FILE,
    DEFAULT_THOLE_WIDTH,
    FF_XML,
    OUTPUT_DIR,
    PAIR,
    add_local_python_paths,
    build_box,
    load_local_plugins,
    resolve_pair_paths,
)

add_local_python_paths(include_dmff=True)

import phyneoforceplugin  # noqa: E402
from dmff.api import Hamiltonian  # noqa: E402
from dmff.common import nblist  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default=PAIR)
    parser.add_argument("--batch", default=BATCH)
    parser.add_argument("--max-points", type=int, default=12)
    parser.add_argument("--output-prefix", default="li_dmc_admp_pme_diagnose_batch-000_n12")
    return parser.parse_args()


def patch_jax_pickle_compat() -> None:
    from jax._src import core as jax_core

    if getattr(jax_core.ShapedArray.update, "_phyneo_pickle_compat", False):
        return

    original_update = jax_core.ShapedArray.update

    def compat_update(self, shape=None, dtype=None, weak_type=None, **kwargs):
        kwargs.pop("named_shape", None)
        return original_update(self, shape=shape, dtype=dtype, weak_type=weak_type, **kwargs)

    compat_update._phyneo_pickle_compat = True  # type: ignore[attr-defined]
    jax_core.ShapedArray.update = compat_update


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "max_abs": float(np.max(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def energy_by_group(simulation: app.Simulation, group: int) -> float:
    state = simulation.context.getState(getEnergy=True, groups={group})
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def find_admp_force(system: mm.System):
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        if phyneoforceplugin.ADMPPmeForce.isinstance(force):
            return idx, phyneoforceplugin.ADMPPmeForce.cast(force)
    raise RuntimeError("ADMPPmeForce not found in system")


def build_openmm_bundle(pdb_path: Path, polarization: str):
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(str(FF_XML))
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=CUTOFF_NM * unit.nanometer,
        polarization=polarization,
        defaultTholeWidth=DEFAULT_THOLE_WIDTH,
    )
    system.setDefaultPeriodicBoxVectors(*build_box())
    force_idx, force = find_admp_force(system)
    force.setForceGroup(0)
    integrator = mm.VerletIntegrator(0.001)
    platform = mm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPeriodicBoxVectors(*build_box())
    return {"pdb": pdb, "system": system, "force_index": force_idx, "force": force, "simulation": simulation}


def get_openmm_force_summary(force) -> dict:
    mscales = list(force.getMScales())
    pscales = list(force.getPScales())
    dscales = list(force.getDScales())
    alpha, nx, ny, nz = force.getPMEParameters()
    multipoles = []
    for index in range(min(5, force.getNumMultipoles())):
        charge, dip, quad, octo, axis_type, atom_z, atom_x, atom_y, thole, alphas = force.getMultipoleParameters(index)
        multipoles.append(
            {
                "index": index,
                "charge": charge,
                "dipole": list(dip),
                "quadrupole": list(quad),
                "axis_type": axis_type,
                "z": atom_z,
                "x": atom_x,
                "y": atom_y,
                "thole": thole,
                "alphas": list(alphas),
            }
        )
    return {
        "nonbonded_method": int(force.getNonbondedMethod()),
        "polarization_type": int(force.getPolarizationType()),
        "cutoff_nm": float(force.getCutoffDistance()),
        "ewald_error_tolerance": float(force.getEwaldErrorTolerance()),
        "pme_parameters": {"alpha": float(alpha), "nx": int(nx), "ny": int(ny), "nz": int(nz)},
        "aewald": float(force.getAEwald()),
        "default_thole_width": float(force.getDefaultTholeWidth()),
        "mutual_max_iterations": int(force.getMutualInducedMaxIterations()),
        "mutual_target_epsilon": float(force.getMutualInducedTargetEpsilon()),
        "extrapolation_coefficients": list(force.getExtrapolationCoefficients()),
        "mScales": mscales,
        "pScales": pscales,
        "dScales": dscales,
        "first_multipoles": multipoles,
    }


def load_reference_scan(pair: str, batch: str):
    patch_jax_pickle_compat()
    with open(DATA_FILE, "rb") as handle:
        data = pickle.load(handle)
    return data[pair][batch]


def build_dmff_reference():
    dimer_pdb, pdb_a_path, pdb_b_path = resolve_pair_paths(PAIR)
    h_ab = Hamiltonian(str(FF_XML))
    h_a = Hamiltonian(str(FF_XML))
    h_b = Hamiltonian(str(FF_XML))
    pdb_ab = app.PDBFile(str(dimer_pdb))
    pdb_a = app.PDBFile(str(pdb_a_path))
    pdb_b = app.PDBFile(str(pdb_b_path))
    kwargs = dict(nonbondedCutoff=25 * unit.angstrom, nonbondedMethod=app.CutoffPeriodic, ethresh=1e-4, step_pol=20)
    pot_ab = h_ab.createPotential(pdb_ab.topology, **kwargs)
    pot_a = h_a.createPotential(pdb_a.topology, **kwargs)
    pot_b = h_b.createPotential(pdb_b.topology, **kwargs)
    params = h_ab.getParameters()

    box = jnp.eye(3) * BOX_NM

    def build_pairs(potential, positions):
        nbl = nblist.NeighborList(box, CUTOFF_NM, potential.meta["cov_map"])
        nbl.allocate(positions)
        pairs = nbl.pairs
        return pairs[pairs[:, 0] < pairs[:, 1]]

    pos_ab0 = jnp.array(pdb_ab.positions._value)
    pos_a0 = jnp.array(pdb_a.positions._value)
    pos_b0 = jnp.array(pdb_b.positions._value)

    pairs_ab = build_pairs(pot_ab, pos_ab0)
    pairs_a = build_pairs(pot_a, pos_a0)
    pairs_b = build_pairs(pot_b, pos_b0)

    pot_fn_ab = pot_ab.dmff_potentials["ADMPPmeForce"]
    pot_fn_a = pot_a.dmff_potentials["ADMPPmeForce"]
    pot_fn_b = pot_b.dmff_potentials["ADMPPmeForce"]

    generator = h_ab.getGenerators()[0]
    q_local = params["ADMPPmeForce"]["Q_local"]
    pol = params["ADMPPmeForce"]["pol"]
    thole = params["ADMPPmeForce"]["thole"]
    atom_type_map = pot_ab.meta["ADMPPmeForce_map_atomtype"]
    pol_type_map = pot_ab.meta["ADMPPmeForce_map_poltype"]

    def interaction(pos_a_ang, pos_b_ang) -> float:
        pos_a = jnp.array(pos_a_ang) * 0.1
        pos_b = jnp.array(pos_b_ang) * 0.1
        pos_ab = jnp.concatenate([pos_a, pos_b], axis=0)
        return float(
            pot_fn_ab(pos_ab, box, pairs_ab, params)
            - pot_fn_a(pos_a, box, pairs_a, params)
            - pot_fn_b(pos_b, box, pairs_b, params)
        )

    first_atoms = []
    for index in range(min(5, len(atom_type_map))):
        first_atoms.append(
            {
                "index": int(index),
                "atom_type_index": int(atom_type_map[index]),
                "pol_type_index": int(pol_type_map[index]),
                "charge_harmonic_q00": float(q_local[atom_type_map[index]][0]),
                "pol_from_paramset_nm3": float(pol[pol_type_map[index]] / 1000.0),
                "thole_from_paramset": float(thole[pol_type_map[index]]),
            }
        )

    return {
        "interaction": interaction,
        "generator_summary": {
            "mScales": [float(x) for x in generator.mScales],
            "pScales": [float(x) for x in generator.pScales],
            "dScales": [float(x) for x in generator.dScales],
            "ethresh": float(generator.ethresh),
            "lpol": bool(generator.lpol),
            "step_pol": int(generator.step_pol) if generator.step_pol is not None else None,
            "first_atoms": first_atoms,
        },
    }


def induced_norm(force, context) -> float:
    dipoles = np.array([[v.x, v.y, v.z] for v in force.getInducedDipoles(context)], dtype=float)
    return float(np.linalg.norm(dipoles, axis=1).sum())


def main() -> None:
    args = parse_args()
    load_local_plugins()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dimer_pdb, pdb_a_path, pdb_b_path = resolve_pair_paths(args.pair)

    ref = load_reference_scan(args.pair, args.batch)
    npts = min(args.max_points, len(ref["shift"]))
    shifts = np.asarray(ref["shift"][:npts], dtype=float)
    ref_pme = np.asarray(ref["lr_es"][:npts], dtype=float) + np.asarray(ref["lr_pol"][:npts], dtype=float)

    dmff_ref = build_dmff_reference()
    dmff_pme = []
    for i in range(npts):
        dmff_pme.append(dmff_ref["interaction"](ref["posA"][i], ref["posB"][i]))
    dmff_pme = np.asarray(dmff_pme, dtype=float)

    openmm_modes = {}
    for mode in ["direct", "mutual", "extrapolated"]:
        bundle_ab = build_openmm_bundle(dimer_pdb, mode)
        bundle_a = build_openmm_bundle(pdb_a_path, mode)
        bundle_b = build_openmm_bundle(pdb_b_path, mode)
        values = []
        induced_norms = []
        for i in range(npts):
            pos_a_nm = np.asarray(ref["posA"][i], dtype=np.float64) * 0.1
            pos_b_nm = np.asarray(ref["posB"][i], dtype=np.float64) * 0.1
            pos_ab_nm = np.concatenate([pos_a_nm, pos_b_nm], axis=0)

            bundle_ab["simulation"].context.setPositions(pos_ab_nm)
            bundle_a["simulation"].context.setPositions(pos_a_nm)
            bundle_b["simulation"].context.setPositions(pos_b_nm)

            energy = (
                energy_by_group(bundle_ab["simulation"], 0)
                - energy_by_group(bundle_a["simulation"], 0)
                - energy_by_group(bundle_b["simulation"], 0)
            )
            values.append(energy)
            induced_norms.append(induced_norm(bundle_ab["force"], bundle_ab["simulation"].context))

        openmm_modes[mode] = {
            "summary": get_openmm_force_summary(bundle_ab["force"]),
            "pme_energy": [float(x) for x in values],
            "induced_norm_sum": [float(x) for x in induced_norms],
            "delta_vs_check_lr": stats(np.asarray(values, dtype=float) - ref_pme),
            "delta_vs_dmff": stats(np.asarray(values, dtype=float) - dmff_pme),
        }

    print("ADMPPmeForce configuration")
    print(f"  pair={args.pair} batch={args.batch} points={npts}")
    print(f"  XML scales (raw) m/p/d: [0, 0, 0, 0, 0]")
    print(f"  DMFF generator mScales: {dmff_ref['generator_summary']['mScales']}")
    print(f"  DMFF generator pScales: {dmff_ref['generator_summary']['pScales']}")
    print(f"  DMFF generator dScales: {dmff_ref['generator_summary']['dScales']}")
    print(f"  OpenMM extrapolated mScales: {openmm_modes['extrapolated']['summary']['mScales']}")
    print(f"  OpenMM extrapolated pScales: {openmm_modes['extrapolated']['summary']['pScales']}")
    print(f"  OpenMM extrapolated dScales: {openmm_modes['extrapolated']['summary']['dScales']}")
    print(f"  OpenMM defaultTholeWidth: {openmm_modes['extrapolated']['summary']['default_thole_width']}")
    print(f"  OpenMM extrapolation coefficients: {openmm_modes['extrapolated']['summary']['extrapolation_coefficients']}")
    print(f"  OpenMM PME params: {openmm_modes['extrapolated']['summary']['pme_parameters']}")

    print("\nFirst-atom parameter spot check")
    for idx, (omm_atom, dmff_atom) in enumerate(zip(openmm_modes["extrapolated"]["summary"]["first_multipoles"], dmff_ref["generator_summary"]["first_atoms"])):
        print(
            f"  atom {idx}: q_openmm={omm_atom['charge']:+.8f} q_dmff={dmff_atom['charge_harmonic_q00']:+.8f} "
            f"alpha_openmm={omm_atom['alphas'][0]:.8e} alpha_dmff={dmff_atom['pol_from_paramset_nm3']:.8e} "
            f"thole_openmm={omm_atom['thole']:.6f} thole_dmff={dmff_atom['thole_from_paramset']:.6f}"
        )

    print("\nPME energy stats")
    print(f"  DMFF vs check_lr: {stats(dmff_pme - ref_pme)}")
    for mode in ["direct", "mutual", "extrapolated"]:
        print(f"  OpenMM {mode:12s} vs check_lr: {openmm_modes[mode]['delta_vs_check_lr']}")
        print(f"  OpenMM {mode:12s} vs DMFF    : {openmm_modes[mode]['delta_vs_dmff']}")

    print("\nPer-point PME energies")
    print("  point  shift      check_lr_pme    dmff_pme      direct       mutual    extrapolated")
    for i in range(npts):
        print(
            f"  {i:>4d}  {shifts[i]:>7.4f}  {ref_pme[i]:>12.6f}  {dmff_pme[i]:>12.6f}  "
            f"{openmm_modes['direct']['pme_energy'][i]:>12.6f}  {openmm_modes['mutual']['pme_energy'][i]:>12.6f}  "
            f"{openmm_modes['extrapolated']['pme_energy'][i]:>12.6f}"
        )

    print("\nInduced-dipole norm sums on dimer")
    for mode in ["direct", "mutual", "extrapolated"]:
        vals = openmm_modes[mode]["induced_norm_sum"]
        print(f"  {mode:12s}: first={vals[0]:.8f} last={vals[-1]:.8f} mean={np.mean(vals):.8f}")

    report = {
        "pair": args.pair,
        "batch": args.batch,
        "points": npts,
        "shifts": [float(x) for x in shifts],
        "check_lr_pme": [float(x) for x in ref_pme],
        "dmff": {
            "pme_energy": [float(x) for x in dmff_pme],
            "generator_summary": dmff_ref["generator_summary"],
            "delta_vs_check_lr": stats(dmff_pme - ref_pme),
        },
        "openmm": openmm_modes,
    }
    out_path = OUTPUT_DIR / f"{args.output_prefix}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()
