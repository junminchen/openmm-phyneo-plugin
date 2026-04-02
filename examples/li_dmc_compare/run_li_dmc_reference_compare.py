#!/usr/bin/env python3
"""Run a Reference-platform Li_DMC comparison between PhyNEOForce, DMFF, and SAPT-derived scan data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
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
    BOX_NM,
    CUTOFF_NM,
    DATA_FILE,
    DEFAULT_S12,
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
from dmff_sr_custom_forces import (  # noqa: E402
    TERM_ORDER,
    add_damped_dispersion_force,
    add_dmff_short_range_forces_from_xml,
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default=PAIR, help="Pair key in data_dimer.pickle.")
    parser.add_argument("--batch", action="append", help="Restrict to one or more batch ids, e.g. 000.")
    parser.add_argument("--max-points", type=int, default=None, help="Only evaluate the first N points from each selected batch.")
    parser.add_argument("--output-prefix", default="li_dmc_reference", help="Prefix for CSV/JSON outputs.")
    return parser.parse_args()


def interaction_delta_rows(rows, key_a, key_b, prefix):
    vals = np.array([row[key_a] - row[key_b] for row in rows], dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_max_abs": float(np.max(np.abs(vals))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(vals)))),
    }


def summarize_against_sapt(rows, model_key: str, sapt_key: str, prefix: str) -> dict[str, float]:
    vals = np.array([row[model_key] - row[sapt_key] for row in rows], dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_max_abs": float(np.max(np.abs(vals))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(vals)))),
    }


@dataclass
class OpenMMSystem:
    topology: app.Topology
    system: mm.System
    simulation: app.Simulation
    group_map: dict[str, int]


def _strip_motion_removers(system: mm.System) -> None:
    removable = [
        idx for idx in range(system.getNumForces())
        if type(system.getForce(idx)).__name__ == "CMMotionRemover"
    ]
    for idx in reversed(removable):
        system.removeForce(idx)


def build_openmm_system(pdb_path: Path, ff_xml: Path) -> OpenMMSystem:
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(str(ff_xml))
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=CUTOFF_NM * unit.nanometer,
        polarization="extrapolated",
        defaultTholeWidth=DEFAULT_THOLE_WIDTH,
    )
    _strip_motion_removers(system)
    system.setDefaultPeriodicBoxVectors(*build_box())

    group_map: dict[str, int] = {}
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        if phyneoforceplugin.ADMPPmeForce.isinstance(force):
            force.setForceGroup(0)
            group_map["ADMPPmeForce"] = 0

    start_group = 1
    add_dmff_short_range_forces_from_xml(system, pdb.topology, str(ff_xml), start_group=start_group, s12=DEFAULT_S12)
    for offset, force_name in enumerate(TERM_ORDER):
        group_map[force_name] = start_group + offset

    disp_group = start_group + len(TERM_ORDER)
    add_damped_dispersion_force(system, pdb.topology, str(ff_xml), cutoff_nm=CUTOFF_NM, force_group=disp_group)
    group_map["DampedDispersionForce"] = disp_group

    platform = mm.Platform.getPlatformByName("Reference")
    integrator = mm.VerletIntegrator(0.001)
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPeriodicBoxVectors(*build_box())
    return OpenMMSystem(pdb.topology, system, simulation, group_map)


def energy_by_group(simulation: app.Simulation, group: int) -> float:
    state = simulation.context.getState(getEnergy=True, groups={group})
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


class DMFFInteractionCalculator:
    def __init__(self, ff_xml: Path, pdb_ab: Path, pdb_a: Path, pdb_b: Path):
        pdb_ab_obj = app.PDBFile(str(pdb_ab))
        pdb_a_obj = app.PDBFile(str(pdb_a))
        pdb_b_obj = app.PDBFile(str(pdb_b))

        self.h_ab = Hamiltonian(str(ff_xml))
        self.h_a = Hamiltonian(str(ff_xml))
        self.h_b = Hamiltonian(str(ff_xml))

        kwargs = dict(nonbondedCutoff=25 * unit.angstrom, nonbondedMethod=app.CutoffPeriodic, ethresh=1e-4, step_pol=20)
        self.pot_ab = self.h_ab.createPotential(pdb_ab_obj.topology, **kwargs)
        self.pot_a = self.h_a.createPotential(pdb_a_obj.topology, **kwargs)
        self.pot_b = self.h_b.createPotential(pdb_b_obj.topology, **kwargs)
        self.params = self.h_ab.getParameters()
        self.box = jnp.eye(3) * BOX_NM

        self.pos_ab = jnp.array(pdb_ab_obj.positions._value)
        self.pos_a = jnp.array(pdb_a_obj.positions._value)
        self.pos_b = jnp.array(pdb_b_obj.positions._value)

        self.pairs_ab = self._pairs(self.pot_ab.meta["cov_map"], self.pos_ab)
        self.pairs_a = self._pairs(self.pot_a.meta["cov_map"], self.pos_a)
        self.pairs_b = self._pairs(self.pot_b.meta["cov_map"], self.pos_b)

        self.mapping = {
            "ADMPPmeForce": "ADMPPmeForce",
            "ADMPDispPmeForce": "ADMPDispPmeForce",
            "QqTtDampingForce": "QqTtDampingForce",
            "SlaterExForce": "SlaterExForce",
            "SlaterSrEsForce": "SlaterSrEsForce",
            "SlaterSrPolForce": "SlaterSrPolForce",
            "SlaterSrDispForce": "SlaterSrDispForce",
            "SlaterDhfForce": "SlaterDhfForce",
            "SlaterDampingForce": "SlaterDampingForce",
        }

    def _pairs(self, cov_map, pos):
        nbl = nblist.NeighborList(self.box, CUTOFF_NM, cov_map)
        nbl.allocate(pos)
        pairs = nbl.pairs
        return pairs[pairs[:, 0] < pairs[:, 1]]

    def _interaction(self, name: str, pos_a_ang, pos_b_ang) -> float:
        pos_a = jnp.array(pos_a_ang) * 0.1
        pos_b = jnp.array(pos_b_ang) * 0.1
        pos_ab = jnp.concatenate([pos_a, pos_b], axis=0)
        fn_ab = self.pot_ab.dmff_potentials[self.mapping[name]]
        fn_a = self.pot_a.dmff_potentials[self.mapping[name]]
        fn_b = self.pot_b.dmff_potentials[self.mapping[name]]
        return float(
            fn_ab(pos_ab, self.box, self.pairs_ab, self.params)
            - fn_a(pos_a, self.box, self.pairs_a, self.params)
            - fn_b(pos_b, self.box, self.pairs_b, self.params)
        )

    def compute(self, pos_a_ang, pos_b_ang) -> dict[str, float]:
        out = {name: self._interaction(name, pos_a_ang, pos_b_ang) for name in self.mapping}
        out["DampedDispersionForce"] = out["ADMPDispPmeForce"] + out["SlaterDampingForce"]
        out["ShortRangeTotal"] = sum(out[name] for name in TERM_ORDER)
        out["PhyNEOTotalComparable"] = out["ADMPPmeForce"] + out["DampedDispersionForce"] + out["ShortRangeTotal"]
        return out


class PhyNEOInteractionCalculator:
    def __init__(self, ff_xml: Path, pdb_ab: Path, pdb_a: Path, pdb_b: Path):
        self.ab = build_openmm_system(pdb_ab, ff_xml)
        self.a = build_openmm_system(pdb_a, ff_xml)
        self.b = build_openmm_system(pdb_b, ff_xml)
        self.group_map = dict(self.ab.group_map)

    def _set_positions(self, pos_a_ang, pos_b_ang) -> None:
        pos_a_nm = np.asarray(pos_a_ang, dtype=np.float64) * 0.1
        pos_b_nm = np.asarray(pos_b_ang, dtype=np.float64) * 0.1
        pos_ab_nm = np.concatenate([pos_a_nm, pos_b_nm], axis=0)

        self.ab.simulation.context.setPeriodicBoxVectors(*build_box())
        self.a.simulation.context.setPeriodicBoxVectors(*build_box())
        self.b.simulation.context.setPeriodicBoxVectors(*build_box())

        self.ab.simulation.context.setPositions(pos_ab_nm)
        self.a.simulation.context.setPositions(pos_a_nm)
        self.b.simulation.context.setPositions(pos_b_nm)

    def compute(self, pos_a_ang, pos_b_ang) -> dict[str, float]:
        self._set_positions(pos_a_ang, pos_b_ang)
        out = {}
        for name, group in self.group_map.items():
            out[name] = (
                energy_by_group(self.ab.simulation, group)
                - energy_by_group(self.a.simulation, group)
                - energy_by_group(self.b.simulation, group)
            )
        out["ShortRangeTotal"] = sum(out[name] for name in TERM_ORDER)
        out["PhyNEOTotalComparable"] = out["ADMPPmeForce"] + out["DampedDispersionForce"] + out["ShortRangeTotal"]
        return out


def main() -> None:
    args = parse_args()
    load_local_plugins()
    patch_jax_pickle_compat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_FILE, "rb") as handle:
        data = pickle.load(handle)
    scan = data[args.pair]
    selected_batches = sorted(scan)
    if args.batch:
        selected = set(args.batch)
        selected_batches = [batch for batch in selected_batches if batch in selected]
        missing = sorted(selected - set(selected_batches))
        if missing:
            raise KeyError(f"Unknown batch ids for {args.pair}: {missing}")

    dimer_pdb, pdb_a, pdb_b = resolve_pair_paths(args.pair)
    dmff_calc = DMFFInteractionCalculator(FF_XML, dimer_pdb, pdb_a, pdb_b)
    phyneo_calc = PhyNEOInteractionCalculator(FF_XML, dimer_pdb, pdb_a, pdb_b)

    rows: list[dict[str, float | int | str]] = []
    shared_components = [
        "ADMPPmeForce",
        "QqTtDampingForce",
        "SlaterExForce",
        "SlaterSrEsForce",
        "SlaterSrPolForce",
        "SlaterSrDispForce",
        "SlaterDhfForce",
        "DampedDispersionForce",
        "ShortRangeTotal",
        "PhyNEOTotalComparable",
    ]

    for batch in selected_batches:
        batch_data = scan[batch]
        shifts = batch_data["shift"]
        if args.max_points is not None:
            shifts = shifts[:args.max_points]
        print(f"Running batch {batch} with {len(shifts)} points")
        for idx, shift in enumerate(shifts):
            dmff_vals = dmff_calc.compute(batch_data["posA"][idx], batch_data["posB"][idx])
            phyneo_vals = phyneo_calc.compute(batch_data["posA"][idx], batch_data["posB"][idx])
            row: dict[str, float | int | str] = {"batch": batch, "point": idx, "shift": float(shift)}
            sapt_es = float(batch_data["es"][idx] + batch_data["lr_es"][idx])
            sapt_pol = float(batch_data["pol"][idx] + batch_data["lr_pol"][idx])
            sapt_disp = float(batch_data["disp"][idx] + batch_data["lr_disp"][idx])
            row["sapt_es"] = sapt_es
            row["sapt_pol"] = sapt_pol
            row["sapt_disp"] = sapt_disp
            row["sapt_espol"] = sapt_es + sapt_pol
            row["sapt_total"] = sapt_es + sapt_pol + sapt_disp
            for name in shared_components:
                row[f"dmff_{name}"] = dmff_vals[name]
                row[f"phyneo_{name}"] = phyneo_vals[name]
                row[f"delta_{name}"] = phyneo_vals[name] - dmff_vals[name]
            row["dmff_ADMPDispPmeForce"] = dmff_vals["ADMPDispPmeForce"]
            row["dmff_SlaterDampingForce"] = dmff_vals["SlaterDampingForce"]
            row["dmff_espol"] = (
                dmff_vals["ADMPPmeForce"]
                + dmff_vals["QqTtDampingForce"]
                + dmff_vals["SlaterSrEsForce"]
                + dmff_vals["SlaterSrPolForce"]
            )
            row["phyneo_espol"] = (
                phyneo_vals["ADMPPmeForce"]
                + phyneo_vals["QqTtDampingForce"]
                + phyneo_vals["SlaterSrEsForce"]
                + phyneo_vals["SlaterSrPolForce"]
            )
            row["delta_espol"] = row["phyneo_espol"] - row["dmff_espol"]
            row["dmff_disp_total"] = dmff_vals["DampedDispersionForce"] + dmff_vals["SlaterSrDispForce"]
            row["phyneo_disp_total"] = phyneo_vals["DampedDispersionForce"] + phyneo_vals["SlaterSrDispForce"]
            row["delta_disp_total"] = row["phyneo_disp_total"] - row["dmff_disp_total"]
            rows.append(row)

    if not rows:
        raise ValueError("No comparison points were selected.")

    suffix_parts = []
    if args.batch:
        suffix_parts.append("batch-" + "-".join(selected_batches))
    if args.max_points is not None:
        suffix_parts.append(f"n{args.max_points}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    csv_path = OUTPUT_DIR / f"{args.output_prefix}_compare{suffix}.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "pair": args.pair,
        "batches": selected_batches,
        "points": len(rows),
        "shared_components": shared_components,
        "notes": {
            "dmff_dispersion_split": "DMFF keeps ADMPDispPmeForce and SlaterDampingForce separate.",
            "phyneo_dispersion_split": "PhyNEO currently evaluates merged damped dispersion as DampedDispersionForce on the OpenMM side.",
        },
    }
    for name in shared_components:
        summary[name] = interaction_delta_rows(rows, f"phyneo_{name}", f"dmff_{name}", name)
    summary["SAPTComparisons"] = {
        "dmff_espol_vs_sapt": summarize_against_sapt(rows, "dmff_espol", "sapt_espol", "dmff_espol_vs_sapt"),
        "phyneo_espol_vs_sapt": summarize_against_sapt(rows, "phyneo_espol", "sapt_espol", "phyneo_espol_vs_sapt"),
        "dmff_disp_vs_sapt": summarize_against_sapt(rows, "dmff_disp_total", "sapt_disp", "dmff_disp_vs_sapt"),
        "phyneo_disp_vs_sapt": summarize_against_sapt(rows, "phyneo_disp_total", "sapt_disp", "phyneo_disp_vs_sapt"),
    }

    json_path = OUTPUT_DIR / f"{args.output_prefix}_summary{suffix}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    for name in ["ADMPPmeForce", "DampedDispersionForce", "ShortRangeTotal", "PhyNEOTotalComparable"]:
        stats = summary[name]
        print(
            f"{name:24s} mean={stats[f'{name}_mean']:+.6f} "
            f"max_abs={stats[f'{name}_max_abs']:.6f} "
            f"rmse={stats[f'{name}_rmse']:.6f}"
        )
    print("SAPT es+pol/disp comparisons")
    for name in ["dmff_espol_vs_sapt", "phyneo_espol_vs_sapt", "dmff_disp_vs_sapt", "phyneo_disp_vs_sapt"]:
        stats = summary["SAPTComparisons"][name]
        print(
            f"{name:24s} mean={stats[f'{name}_mean']:+.6f} "
            f"max_abs={stats[f'{name}_max_abs']:.6f} "
            f"rmse={stats[f'{name}_rmse']:.6f}"
        )


if __name__ == "__main__":
    main()
