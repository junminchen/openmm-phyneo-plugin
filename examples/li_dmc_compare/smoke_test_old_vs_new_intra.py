#!/usr/bin/env python3
"""Compare old ByteFF-pol monomer energy against the new PhyNEO builder using local Li/DMC fixtures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import openmm.app as app
import openmm.unit as unit

from example_fixture import (  # noqa: E402
    FF_XML,
    PARAMS_DIR,
    PDB_BANK,
    WORKSPACE_ROOT,
    add_local_python_paths,
)

add_local_python_paths(include_workspace=True)

from openmmtool import PhyNEOCalculator  # noqa: E402


POLFF_ROOT = WORKSPACE_ROOT / "polff"
if str(POLFF_ROOT) not in sys.path:
    sys.path.insert(0, str(POLFF_ROOT))


def load_old_openmmtool():
    if "bytemol.toolkit.asetool.basecalculator" not in sys.modules:
        shim = types.ModuleType("bytemol.toolkit.asetool.basecalculator")

        class BaseCalculator:
            implemented_properties: list[str] = []

            def __init__(self, *args, **kwargs):
                pass

        shim.BaseCalculator = BaseCalculator
        sys.modules["bytemol.toolkit.asetool.basecalculator"] = shim

    module_path = POLFF_ROOT / "byteff2" / "toolkit" / "openmmtool.py"
    spec = importlib.util.spec_from_file_location("byteff_old_openmmtool", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load old openmmtool from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_single_molecule_top(stem: str, params_dir: Path) -> Path:
    atp_path = params_dir / f"{stem}.atp"
    itp_path = params_dir / f"{stem}.itp"
    if not atp_path.exists() or not itp_path.exists():
        raise FileNotFoundError(f"Missing {stem}.atp or {stem}.itp in {params_dir}")

    lines = [
        "[ defaults ]",
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ",
        "1 1 yes 1.0 1.0",
        "",
        f'#include "{atp_path}"',
        f'#include "{itp_path}"',
        "",
        "[ system ]",
        f"{stem} monomer",
        "",
        "[ molecules ]",
        f"{stem:<16} 1",
    ]
    handle = tempfile.NamedTemporaryFile("w", suffix=".top", delete=False)
    with handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    return Path(handle.name)


def load_nonbonded_params(stem: str, params_dir: Path) -> dict:
    with open(params_dir / f"{stem}.json", "r", encoding="utf-8") as handle:
        params = json.load(handle)
    with open(params_dir / f"{stem}_nb_params.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)["metadata"]
    return {stem: params, "metadata": metadata}


def energy_from_context(simulation) -> float:
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)


def print_term_block(title: str, terms: dict[str, float]) -> None:
    print(title)
    if not terms:
        print("  (none)")
        return
    for name, energy in sorted(terms.items()):
        print(f"  {name:32s} {energy:16.8f} kcal/mol")


def sum_terms(terms: dict[str, float], keys: list[str]) -> float:
    return sum(terms.get(key, 0.0) for key in keys)


def print_comparison_rows(old_terms: dict[str, float], inter_terms: dict[str, float], intra_terms: dict[str, float]) -> None:
    new_short_range = sum_terms(
        inter_terms,
        [
            "DampedDispersionForce",
            "QqTtDampingForce",
            "SlaterDampingForce",
            "SlaterDhfForce",
            "SlaterExForce",
            "SlaterSrDispForce",
            "SlaterSrEsForce",
            "SlaterSrPolForce",
        ],
    )
    rows = [
        ("HarmonicBondForce", old_terms.get("HarmonicBondForce", 0.0), "IntraHarmonicBondForce", intra_terms.get("IntraHarmonicBondForce", 0.0)),
        ("HarmonicAngleForce", old_terms.get("HarmonicAngleForce", 0.0), "IntraHarmonicAngleForce", intra_terms.get("IntraHarmonicAngleForce", 0.0)),
        ("PeriodicTorsionForce", old_terms.get("PeriodicTorsionForce", 0.0), "IntraPeriodicTorsionForce", intra_terms.get("IntraPeriodicTorsionForce", 0.0)),
        ("AmoebaMultipoleForce", old_terms.get("AmoebaMultipoleForce", 0.0), "ADMPPmeForce", inter_terms.get("ADMPPmeForce", 0.0)),
        ("CustomNonbondedForce", old_terms.get("CustomNonbondedForce", 0.0), "NewShortRangeInterTotal", new_short_range),
        ("CustomBondForce", old_terms.get("CustomBondForce", 0.0), "IntraLennardJonesExceptions", intra_terms.get("IntraLennardJonesExceptions", 0.0)),
    ]

    print("candidate old/new term comparison:")
    for old_name, old_energy, new_name, new_energy in rows:
        delta = old_energy - new_energy
        print(
            f"  {old_name:24s} {old_energy:16.8f}  |  "
            f"{new_name:24s} {new_energy:16.8f}  |  delta {delta:16.8f} kcal/mol"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", nargs="+", default=["Li", "DMC"])
    parser.add_argument("--params-dir", type=Path, default=PARAMS_DIR)
    parser.add_argument("--xml", type=Path, default=FF_XML)
    parser.add_argument("--platform", default="Reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_module = load_old_openmmtool()

    for stem in args.molecules:
        pdb_path = PDB_BANK / f"{stem}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"Missing PDB for {stem}: {pdb_path}")

        pdb = app.PDBFile(str(pdb_path))
        coords_ang = np.asarray(pdb.positions.value_in_unit(unit.angstrom))

        top_path = write_single_molecule_top(stem, args.params_dir)
        try:
            nonbonded_params = load_nonbonded_params(stem, args.params_dir)
            old_calc = old_module.AmoebaCalculator(
                str(top_path),
                nonbonded_params,
                platform_name=args.platform,
                separate_terms=True,
                unit_cell=None,
                cutoff=1.0,
            )
            old_total, _ = old_calc._calculate_without_restraint(coords_ang)
            old_terms, _ = old_calc.get_separate_terms()

            new_inter = PhyNEOCalculator(
                pdb_path,
                args.xml,
                platform_name=args.platform,
                separate_terms=True,
                use_periodic=False,
                enable_intra=False,
            )
            new_inter_total = energy_from_context(new_inter.simulation)
            inter_terms, _ = new_inter.get_separate_terms()

            new_with_intra = PhyNEOCalculator(
                pdb_path,
                args.xml,
                platform_name=args.platform,
                separate_terms=True,
                use_periodic=False,
                enable_intra=True,
                intra_params_dir=args.params_dir,
            )
            new_total = energy_from_context(new_with_intra.simulation)
            new_terms, _ = new_with_intra.get_separate_terms()
        finally:
            top_path.unlink(missing_ok=True)

        intra_terms = {k: v for k, v in new_terms.items() if k.startswith("Intra")}
        print(f"\n=== {stem} ===")
        print(f"old_byteff_total            {old_total:16.8f} kcal/mol")
        print(f"new_phyneo_inter_only       {new_inter_total:16.8f} kcal/mol")
        print(f"new_phyneo_with_intra       {new_total:16.8f} kcal/mol")
        print(f"delta(old - new_with_intra) {old_total - new_total:16.8f} kcal/mol")
        print(f"delta(inter -> with_intra)  {new_total - new_inter_total:16.8f} kcal/mol")
        print_term_block("old builder terms:", old_terms)
        print_term_block("new builder intra terms:", intra_terms)
        print_term_block("new builder inter terms:", inter_terms)
        print_comparison_rows(old_terms, inter_terms, intra_terms)


if __name__ == "__main__":
    main()
