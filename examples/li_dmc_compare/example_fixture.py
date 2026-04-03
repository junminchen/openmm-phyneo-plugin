#!/usr/bin/env python3
"""Shared local paths and helpers for the self-contained Li-DMC validation example."""

from __future__ import annotations

import sys
from pathlib import Path

import openmm as mm

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
EXAMPLE_ROOT = Path(__file__).resolve().parent
FIXTURE_ROOT = EXAMPLE_ROOT / "fixtures"
OUTPUT_DIR = EXAMPLE_ROOT / "output"

DMFF_ROOT = WORKSPACE_ROOT / "DMFF"
FF_XML = FIXTURE_ROOT / "phyneo_ecl.xml"
DATA_FILE = FIXTURE_ROOT / "data_dimer_li_dmc_n12.pickle"
DIMER_BANK = FIXTURE_ROOT / "dimer_bank"
PDB_BANK = FIXTURE_ROOT / "pdb_bank"
PARAMS_DIR = FIXTURE_ROOT / "params_results"

PAIR = "conf_060_Li_DMC"
DEFAULT_BATCH = "000"
BATCH = DEFAULT_BATCH
BOX_NM = 6.0
CUTOFF_NM = 2.5
DEFAULT_THOLE_WIDTH = 5.0
DEFAULT_S12 = 0.169


def add_local_python_paths(include_dmff: bool = False, include_workspace: bool = False) -> None:
    if include_workspace:
        workspace_str = str(WORKSPACE_ROOT)
        if workspace_str not in sys.path:
            sys.path.insert(0, workspace_str)

    repo_str = str(REPO_ROOT)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)

    candidates = [REPO_ROOT / "python"]
    candidates.extend(
        build_dir / "python"
        for build_dir in sorted(REPO_ROOT.glob("build*/"))
        if (build_dir / "python" / "phyneoforceplugin.py").exists()
    )
    for candidate in reversed(candidates):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    if include_dmff:
        dmff_str = str(DMFF_ROOT)
        if dmff_str not in sys.path:
            sys.path.insert(0, dmff_str)


def load_local_plugins() -> None:
    for build_dir in sorted(REPO_ROOT.glob("build*/")):
        candidate = build_dir.resolve()
        reference_plugin = candidate / "platforms" / "reference" / "libOpenMMPhyNEOForceReference.dylib"
        if reference_plugin.exists():
            mm.Platform.loadPluginLibrary(str(reference_plugin))


def build_box() -> tuple[mm.Vec3, mm.Vec3, mm.Vec3]:
    return (
        mm.Vec3(BOX_NM, 0.0, 0.0),
        mm.Vec3(0.0, BOX_NM, 0.0),
        mm.Vec3(0.0, 0.0, BOX_NM),
    )


def resolve_pair_paths(pair: str) -> tuple[Path, Path, Path]:
    parts = pair.split("_")
    if len(parts) != 4 or parts[0] != "conf":
        raise ValueError(f"Unexpected pair key format: {pair}")
    pair_id = parts[1]
    atom_a, atom_b = parts[2], parts[3]
    dimer_pdb = DIMER_BANK / f"dimer_{pair_id}_{atom_a}_{atom_b}.pdb"
    pdb_a = PDB_BANK / f"{atom_a}.pdb"
    pdb_b = PDB_BANK / f"{atom_b}.pdb"
    missing = [str(path) for path in (dimer_pdb, pdb_a, pdb_b) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fixture geometry file(s) for {pair}: {missing}")
    return dimer_pdb, pdb_a, pdb_b
