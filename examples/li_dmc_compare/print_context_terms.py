#!/usr/bin/env python3
"""Print total and per-force-group context energies for the self-contained Li-DMC fixture."""

from __future__ import annotations

import argparse

import openmm.unit as unit

from example_fixture import (  # noqa: E402
    CUTOFF_NM,
    FF_XML,
    PAIR,
    PARAMS_DIR,
    add_local_python_paths,
    build_box,
    resolve_pair_paths,
)

add_local_python_paths(include_workspace=True)

from openmmtool import PhyNEOCalculator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", default=None, help="Defaults to the local Li-DMC dimer fixture.")
    parser.add_argument("--xml", default=str(FF_XML))
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--cutoff", type=float, default=CUTOFF_NM)
    parser.add_argument("--periodic", action="store_true", default=False)
    parser.add_argument("--enable-intra", action="store_true", default=False)
    parser.add_argument("--intra-params-dir", default=str(PARAMS_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdb_path = args.pdb or str(resolve_pair_paths(PAIR)[0])
    box_vectors = None
    if args.periodic:
        box_vectors = build_box()

    calc = PhyNEOCalculator(
        pdb_path,
        args.xml,
        platform_name=args.platform,
        separate_terms=True,
        use_periodic=args.periodic,
        box_vectors=box_vectors,
        cutoff=args.cutoff,
        enable_intra=args.enable_intra,
        intra_params_dir=args.intra_params_dir,
    )

    state = calc.simulation.context.getState(getEnergy=True)
    total = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    terms, _ = calc.get_separate_terms()

    print(f"pdb: {pdb_path}")
    print(f"xml: {args.xml}")
    print(f"platform: {args.platform}")
    print(f"periodic: {args.periodic}")
    print(f"enable_intra: {args.enable_intra}")
    print(f"total_potential_energy {total:16.8f} kcal/mol")
    print("context term energies:")
    for name, energy in sorted(terms.items()):
        print(f"  {name:32s} {energy:16.8f} kcal/mol")


if __name__ == "__main__":
    main()
