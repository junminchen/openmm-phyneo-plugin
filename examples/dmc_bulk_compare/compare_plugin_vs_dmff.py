#!/usr/bin/env python3
"""Compare plugin and DMFF PME term energies on a packed DMC box."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types as _types
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_mpid84")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import openmm.app as app
import openmm.unit as unit

if "jax.config" not in sys.modules:
    shim = _types.ModuleType("jax.config")
    shim.config = jax.config
    sys.modules["jax.config"] = shim

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
INPUT_ROOT = HERE / "inputs"

BOX_PDB = HERE / "dmc_100mol_box.pdb"
FF_XML = INPUT_ROOT / "phyneo_ecl.xml"
CUTOFF_NM = 1.2

for candidate in (REPO_ROOT, REPO_ROOT / "python", WORKSPACE_ROOT / "DMFF"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dmff.api import Hamiltonian  # noqa: E402
from dmff.common import nblist  # noqa: E402
from dmff_sr_custom_forces import TERM_ORDER  # noqa: E402
from openmmtool import PhyNEOCalculator  # noqa: E402

DMC_MOLAR_MASS_G_PER_MOL = 90.07794
AVOGADRO = 6.02214076e23
SHORT_RANGE_TERM_ORDER = TERM_ORDER + ["SlaterDampingForce"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=BOX_PDB)
    parser.add_argument("--xml", type=Path, default=FF_XML)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--cutoff", type=float, default=CUTOFF_NM)
    parser.add_argument("--plugin-dispersion-mode", choices=("lrc", "native_pme"), default="lrc")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
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


def box_density_g_per_ml(pdb: app.PDBFile) -> float:
    residues = list(pdb.topology.residues())
    dimensions = pdb.topology.getUnitCellDimensions()
    if dimensions is None:
        raise ValueError("Periodic box vectors are required to compute density.")
    a_nm, b_nm, c_nm = dimensions.value_in_unit(unit.nanometer)
    volume_nm3 = float(a_nm * b_nm * c_nm)
    volume_cm3 = volume_nm3 * 1.0e-21
    mass_g = len(residues) * DMC_MOLAR_MASS_G_PER_MOL / AVOGADRO
    return float(mass_g / volume_cm3)


def build_box_matrix_nm(pdb: app.PDBFile) -> jnp.ndarray:
    box = pdb.topology.getPeriodicBoxVectors()
    if box is None:
        raise ValueError("Periodic box vectors are required for DMFF PME.")
    return jnp.array(
        [
            [box[0][0].value_in_unit(unit.nanometer), box[0][1].value_in_unit(unit.nanometer), box[0][2].value_in_unit(unit.nanometer)],
            [box[1][0].value_in_unit(unit.nanometer), box[1][1].value_in_unit(unit.nanometer), box[1][2].value_in_unit(unit.nanometer)],
            [box[2][0].value_in_unit(unit.nanometer), box[2][1].value_in_unit(unit.nanometer), box[2][2].value_in_unit(unit.nanometer)],
        ],
        dtype=jnp.float64,
    )


class DMFFBoxCalculator:
    def __init__(self, ff_xml: Path, pdb_path: Path, cutoff_nm: float):
        self.pdb = app.PDBFile(str(pdb_path))
        self.box = build_box_matrix_nm(self.pdb)
        self.positions = jnp.array(self.pdb.positions.value_in_unit(unit.nanometer))

        kwargs = dict(
            nonbondedCutoff=cutoff_nm * unit.nanometer,
            nonbondedMethod=app.PME,
            ethresh=1e-4,
            step_pol=20,
        )
        self.hamiltonian = Hamiltonian(str(ff_xml))
        self.potential = self.hamiltonian.createPotential(self.pdb.topology, **kwargs)
        self.params = self.hamiltonian.getParameters()
        cov_map = self.potential.meta["cov_map"]
        nbl = nblist.NeighborList(self.box, cutoff_nm, cov_map)
        nbl.allocate(self.positions)
        pairs = nbl.pairs
        self.pairs = pairs[pairs[:, 0] < pairs[:, 1]]

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

    def compute_raw(self) -> dict[str, float]:
        out = {}
        for label, dmff_name in self.mapping.items():
            fn = self.potential.dmff_potentials[dmff_name]
            out[label] = float(fn(self.positions, self.box, self.pairs, self.params))
        return out

    def compute_comparable(self, plugin_dispersion_mode: str) -> tuple[dict[str, float], dict[str, float]]:
        raw = self.compute_raw()
        comparable = {name: raw[name] for name in SHORT_RANGE_TERM_ORDER}
        if plugin_dispersion_mode == "native_pme":
            comparable["ADMPPmeForce"] = raw["ADMPPmeForce"] + raw["ADMPDispPmeForce"]
        else:
            comparable["ADMPPmeForce"] = raw["ADMPPmeForce"]
            comparable["DampedDispersionForce"] = raw["ADMPDispPmeForce"]
        comparable["ShortRangeTotal"] = sum(comparable[name] for name in SHORT_RANGE_TERM_ORDER)
        comparable["PhyNEOTotalComparable"] = comparable["ADMPPmeForce"] + comparable["ShortRangeTotal"]
        if plugin_dispersion_mode != "native_pme":
            comparable["PhyNEOTotalComparable"] += comparable["DampedDispersionForce"]
        return comparable, raw


class PluginBoxCalculator:
    def __init__(self, ff_xml: Path, pdb_path: Path, cutoff_nm: float, platform: str, dispersion_mode: str):
        self.calc = PhyNEOCalculator(
            pdb_path,
            ff_xml,
            platform_name=platform,
            separate_terms=True,
            use_periodic=True,
            cutoff=cutoff_nm,
            dispersion_mode=dispersion_mode,
            enable_intra=False,
        )

    def compute(self) -> tuple[dict[str, float], float]:
        terms = {}
        for _, (group, name) in self.calc.forcegroups.items():
            state = self.calc.simulation.context.getState(getEnergy=True, groups={group})
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            terms[name] = terms.get(name, 0.0) + float(energy)
        total = self.calc.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        comparable = {name: float(val) for name, val in terms.items()}
        comparable["ShortRangeTotal"] = sum(comparable.get(name, 0.0) for name in SHORT_RANGE_TERM_ORDER)
        comparable["PhyNEOTotalComparable"] = comparable.get("ADMPPmeForce", 0.0) + comparable["ShortRangeTotal"]
        if "DampedDispersionForce" in comparable:
            comparable["PhyNEOTotalComparable"] += comparable["DampedDispersionForce"]
        return comparable, float(total)


def write_csv(csv_path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = ["term", "plugin_kj_mol", "dmff_kj_mol", "delta_kj_mol", "delta_per_molecule_kj_mol"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_terms(plot_path: Path, rows: list[dict[str, float | str]]) -> None:
    terms = [str(row["term"]) for row in rows]
    plugin_vals = [float(row["plugin_kj_mol"]) for row in rows]
    dmff_vals = [float(row["dmff_kj_mol"]) for row in rows]
    deltas = [float(row["delta_kj_mol"]) for row in rows]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    x = range(len(terms))

    ax0.plot(x, plugin_vals, marker="o", label="Plugin")
    ax0.plot(x, dmff_vals, marker="s", label="DMFF")
    ax0.set_xticks(list(x))
    ax0.set_xticklabels(terms, rotation=45, ha="right")
    ax0.set_ylabel("Energy (kJ/mol)")
    ax0.set_title("100 DMC box: plugin vs DMFF")
    ax0.legend()

    ax1.bar(list(x), deltas)
    ax1.axhline(0.0, color="black", linewidth=1.0)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(terms, rotation=45, ha="right")
    ax1.set_ylabel("Plugin - DMFF (kJ/mol)")
    ax1.set_title("100 DMC box: per-term deltas")

    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.pdb.exists():
        raise FileNotFoundError(f"Missing input PDB: {args.pdb}. Run make_dmc_box.py first.")

    patch_jax_pickle_compat()
    pdb = app.PDBFile(str(args.pdb))
    density = box_density_g_per_ml(pdb)
    atom_count = sum(1 for _ in pdb.topology.atoms())
    molecule_count = sum(1 for _ in pdb.topology.residues())

    plugin_calc = PluginBoxCalculator(args.xml, args.pdb, args.cutoff, args.platform, args.plugin_dispersion_mode)
    dmff_calc = DMFFBoxCalculator(args.xml, args.pdb, args.cutoff)

    plugin_terms, plugin_total = plugin_calc.compute()
    dmff_terms, dmff_raw_terms = dmff_calc.compute_comparable(args.plugin_dispersion_mode)

    comparable_terms = sorted(set(plugin_terms) | set(dmff_terms) | {"TotalPotentialEnergy"})
    rows: list[dict[str, float | str]] = []
    for term in comparable_terms:
        plugin_val = plugin_total if term == "TotalPotentialEnergy" else plugin_terms.get(term, 0.0)
        dmff_val = plugin_total if False else (0.0)
        if term == "TotalPotentialEnergy":
            dmff_val = dmff_terms["PhyNEOTotalComparable"]
        else:
            dmff_val = dmff_terms.get(term, 0.0)
        delta = plugin_val - dmff_val
        rows.append(
            {
                "term": term,
                "plugin_kj_mol": plugin_val,
                "dmff_kj_mol": dmff_val,
                "delta_kj_mol": delta,
                "delta_per_molecule_kj_mol": delta / molecule_count,
            }
        )

    json_path = args.output_dir / "dmc_100mol_plugin_vs_dmff.json"
    csv_path = args.output_dir / "dmc_100mol_plugin_vs_dmff.csv"
    plot_path = args.output_dir / "dmc_100mol_plugin_vs_dmff.png"

    payload = {
        "pdb": str(args.pdb),
        "xml": str(args.xml),
        "platform": args.platform,
        "cutoff_nm": args.cutoff,
        "enable_intra": False,
        "atom_count": atom_count,
        "molecule_count": molecule_count,
        "box_density_g_ml": density,
        "dmff_raw_terms_kj_mol": dmff_raw_terms,
        "note": "Plugin energies are compared against DMFF comparable PME terms on the same periodic box geometry. Plugin intra terms are intentionally excluded.",
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    plot_terms(plot_path, rows)

    print(f"pdb: {args.pdb}")
    print(f"atom_count: {atom_count}")
    print(f"molecule_count: {molecule_count}")
    print(f"box_density_g_ml: {density:.6f}")
    print("\nPlugin vs DMFF term comparison (kJ/mol):")
    for row in rows:
        print(
            f"  {row['term']:28s} "
            f"Plugin={row['plugin_kj_mol']:16.8f} "
            f"DMFF={row['dmff_kj_mol']:16.8f} "
            f"Delta={row['delta_kj_mol']:16.8f}"
        )
    print("\nRaw DMFF terms (kJ/mol):")
    for name, value in sorted(dmff_raw_terms.items()):
        print(f"  {name:28s} {value:16.8f}")

    print(f"\nWrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
