#!/usr/bin/env python3
"""Generate a cubic box of 100 ethanol molecules using PACKMOL.

Writes ethanol_100mol.pdb ready for run_cuda_npt.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── box parameters ──────────────────────────────────────────────────────────
# 100 EtOH at ~0.77 g/mL (slightly below bulk density 0.789 g/mL so
# NPT can compress to the correct density)
N_MOL   = 100
BOX_A   = 21.5   # Angstroms
SEED    = 12345

SINGLE_PDB = HERE / "ethanol_single.pdb"
OUT_PDB    = HERE / "ethanol_100mol.pdb"
INP_FILE   = HERE / "_packmol.inp"

PACKMOL_INP = f"""\
tolerance 2.0
filetype pdb
seed {SEED}
output {OUT_PDB}

structure {SINGLE_PDB}
  number {N_MOL}
  inside box 0. 0. 0. {BOX_A} {BOX_A} {BOX_A}
end structure
"""


def run_packmol():
    INP_FILE.write_text(PACKMOL_INP)
    with open(INP_FILE) as inp:
        result = subprocess.run(
            ["packmol"],
            stdin=inp,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-1000:])
        sys.exit(f"PACKMOL failed (exit {result.returncode})")
    print(f"PACKMOL finished: {OUT_PDB}")


def prepend_cryst1():
    """PACKMOL does not write a CRYST1 record; add it so OpenMM can read PBC."""
    lines = OUT_PDB.read_text().splitlines()
    cryst = f"CRYST1{BOX_A:9.3f}{BOX_A:9.3f}{BOX_A:9.3f}  90.00  90.00  90.00 P 1           1"
    if not lines[0].startswith("CRYST1"):
        lines.insert(0, cryst)
        OUT_PDB.write_text("\n".join(lines) + "\n")
        print("Added CRYST1 record.")


def cleanup():
    INP_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    run_packmol()
    prepend_cryst1()
    cleanup()
    print(f"Box ready: {OUT_PDB}  ({N_MOL} molecules, {BOX_A} Å cubic)")
