#!/usr/bin/env python3
"""Generate a cubic PACKMOL box containing 100 DMC molecules."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE_ROOT = HERE.parent / "li_dmc_compare" / "fixtures"

DEFAULT_N_MOL = 100
DEFAULT_BOX_A = 28.0
DEFAULT_SEED = 12345
DEFAULT_TOLERANCE = 2.0
DEFAULT_BOX_SIDE_PADDING_A = 1.5

SINGLE_PDB = FIXTURE_ROOT / "pdb_bank" / "DMC.pdb"
OUT_PDB = HERE / "dmc_100mol_box.pdb"
INP_FILE = HERE / "_packmol_dmc.inp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmol", type=int, default=DEFAULT_N_MOL)
    parser.add_argument("--box-a", type=float, default=DEFAULT_BOX_A)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--add-box-sides", type=float, default=DEFAULT_BOX_SIDE_PADDING_A)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_packmol_input(nmol: int, box_a: float, seed: int, tolerance: float, add_box_sides: float) -> str:
    return f"""\
tolerance {tolerance}
filetype pdb
add_box_sides {add_box_sides}
seed {seed}
output {OUT_PDB}

structure {SINGLE_PDB}
  number {nmol}
  inside cube 0. 0. 0. {box_a}
end structure
"""


def run_packmol(packmol_input: str) -> None:
    INP_FILE.write_text(packmol_input, encoding="utf-8")
    with INP_FILE.open("r", encoding="utf-8") as handle:
        result = subprocess.run(
            ["packmol"],
            stdin=handle,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-1000:])
        sys.exit(f"PACKMOL failed with exit code {result.returncode}")


def cleanup() -> None:
    INP_FILE.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if OUT_PDB.exists() and not args.overwrite:
        print(f"Using existing box: {OUT_PDB}")
        return
    if not SINGLE_PDB.exists():
        raise FileNotFoundError(f"Missing monomer PDB: {SINGLE_PDB}")

    packmol_input = build_packmol_input(
        args.nmol, args.box_a, args.seed, args.tolerance, args.add_box_sides
    )
    run_packmol(packmol_input)
    cleanup()
    print(f"Box ready: {OUT_PDB} ({args.nmol} molecules, {args.box_a:.1f} A cubic)")


if __name__ == "__main__":
    main()
