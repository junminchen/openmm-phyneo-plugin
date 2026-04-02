#!/usr/bin/env python3
"""Smoke test for the root openmmtool builder against DMFF component energies on the local Li-DMC fixture."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import types as _types

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_mpid84")

import jax
import numpy as np
import openmm.app as app
from openmm import unit

from example_fixture import (  # noqa: E402
    BOX_NM,
    CUTOFF_NM,
    DATA_FILE,
    FF_XML,
    PAIR,
    add_local_python_paths,
    resolve_pair_paths,
)

if "jax.config" not in sys.modules:
    shim = _types.ModuleType("jax.config")
    shim.config = jax.config
    sys.modules["jax.config"] = shim

add_local_python_paths(include_dmff=True, include_workspace=True)

from dmff.api import Hamiltonian  # noqa: E402
from dmff.common import nblist  # noqa: E402
from dmff_sr_custom_forces import TERM_ORDER  # noqa: E402
from openmmtool import PhyNEOCalculator  # noqa: E402


EXPECTED_COMPONENTS = [
    "ADMPPmeForce",
    "DampedDispersionForce",
    "QqTtDampingForce",
    "SlaterExForce",
    "SlaterSrEsForce",
    "SlaterSrPolForce",
    "SlaterSrDispForce",
    "SlaterDhfForce",
    "SlaterDampingForce",
]


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


def load_positions_from_scan(pair: str, batch: str, point: int):
    with open(DATA_FILE, "rb") as handle:
        data = pickle.load(handle)
    scan = data[pair]
    if batch not in scan:
        raise KeyError(f"Unknown batch {batch} for pair {pair}")
    batch_data = scan[batch]
    if point >= len(batch_data["shift"]):
        raise IndexError(f"Point {point} is out of range for batch {batch}; available {len(batch_data['shift'])}")
    return (
        np.asarray(batch_data["posA"][point], dtype=np.float64),
        np.asarray(batch_data["posB"][point], dtype=np.float64),
        float(batch_data["shift"][point]),
    )


class DMFFInteractionCalculator:
    def __init__(self, ff_xml, pdb_ab, pdb_a, pdb_b):
        pdb_ab_obj = app.PDBFile(str(pdb_ab))
        pdb_a_obj = app.PDBFile(str(pdb_a))
        pdb_b_obj = app.PDBFile(str(pdb_b))

        kwargs = dict(nonbondedCutoff=25 * unit.angstrom, nonbondedMethod=app.CutoffPeriodic, ethresh=1e-4, step_pol=20)

        self.h_ab = Hamiltonian(str(ff_xml))
        self.h_a = Hamiltonian(str(ff_xml))
        self.h_b = Hamiltonian(str(ff_xml))
        self.pot_ab = self.h_ab.createPotential(pdb_ab_obj.topology, **kwargs)
        self.pot_a = self.h_a.createPotential(pdb_a_obj.topology, **kwargs)
        self.pot_b = self.h_b.createPotential(pdb_b_obj.topology, **kwargs)
        self.params = self.h_ab.getParameters()
        self.box = jax.numpy.eye(3) * BOX_NM

        self.pos_ab = jax.numpy.array(pdb_ab_obj.positions._value)
        self.pos_a = jax.numpy.array(pdb_a_obj.positions._value)
        self.pos_b = jax.numpy.array(pdb_b_obj.positions._value)

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
        pos_a = jax.numpy.array(pos_a_ang) * 0.1
        pos_b = jax.numpy.array(pos_b_ang) * 0.1
        pos_ab = jax.numpy.concatenate([pos_a, pos_b], axis=0)
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


class OpenMMInteractionCalculator:
    def __init__(self, ff_xml, pdb_ab, pdb_a, pdb_b):
        box = [(BOX_NM, 0.0, 0.0), (0.0, BOX_NM, 0.0), (0.0, 0.0, BOX_NM)]
        self.ab = PhyNEOCalculator(pdb_ab, ff_xml, platform_name="Reference", separate_terms=True, use_periodic=True, box_vectors=box, cutoff=CUTOFF_NM)
        self.a = PhyNEOCalculator(pdb_a, ff_xml, platform_name="Reference", separate_terms=True, use_periodic=True, box_vectors=box, cutoff=CUTOFF_NM)
        self.b = PhyNEOCalculator(pdb_b, ff_xml, platform_name="Reference", separate_terms=True, use_periodic=True, box_vectors=box, cutoff=CUTOFF_NM)

    def _set_positions(self, pos_a_ang, pos_b_ang) -> None:
        pos_a_nm = np.asarray(pos_a_ang, dtype=np.float64) * 0.1
        pos_b_nm = np.asarray(pos_b_ang, dtype=np.float64) * 0.1
        pos_ab_nm = np.concatenate([pos_a_nm, pos_b_nm], axis=0)
        for calc, pos_nm in ((self.ab, pos_ab_nm), (self.a, pos_a_nm), (self.b, pos_b_nm)):
            calc.simulation.context.setPositions(pos_nm)

    def compute(self, pos_a_ang, pos_b_ang) -> dict[str, float]:
        self._set_positions(pos_a_ang, pos_b_ang)
        terms_ab, _ = self.ab.get_separate_terms()
        terms_a, _ = self.a.get_separate_terms()
        terms_b, _ = self.b.get_separate_terms()
        out = {}
        for name in set(terms_ab) | set(terms_a) | set(terms_b):
            out[name] = float(terms_ab.get(name, 0.0) - terms_a.get(name, 0.0) - terms_b.get(name, 0.0))
        out["ShortRangeTotal"] = sum(out.get(name, 0.0) for name in TERM_ORDER)
        out["PhyNEOTotalComparable"] = out.get("ADMPPmeForce", 0.0) + out.get("DampedDispersionForce", 0.0) + out["ShortRangeTotal"]
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default=PAIR)
    parser.add_argument("--batch", default="000")
    parser.add_argument("--point", type=int, default=0)
    return parser.parse_args()


def main():
    patch_jax_pickle_compat()
    args = parse_args()
    pdb_ab, pdb_a, pdb_b = resolve_pair_paths(args.pair)
    pos_a, pos_b, shift = load_positions_from_scan(args.pair, args.batch, args.point)

    dmff = DMFFInteractionCalculator(FF_XML, pdb_ab, pdb_a, pdb_b)
    openmm_calc = OpenMMInteractionCalculator(FF_XML, pdb_ab, pdb_a, pdb_b)

    dmff_terms = dmff.compute(pos_a, pos_b)
    openmm_terms = openmm_calc.compute(pos_a, pos_b)

    print(f"pair={args.pair} batch={args.batch} point={args.point} shift={shift:.6f}")
    print("\nOpenMM components:")
    for name in sorted(openmm_terms):
        print(f"  {name:28s} {openmm_terms[name]:16.8f} kJ/mol")

    print("\nDMFF components:")
    for name in sorted(dmff_terms):
        print(f"  {name:28s} {dmff_terms[name]:16.8f} kJ/mol")

    missing = [name for name in EXPECTED_COMPONENTS if name not in openmm_terms]
    if missing:
        raise RuntimeError(f"OpenMM builder is missing expected components: {missing}")

    print("\nOpenMM - DMFF deltas:")
    for name in sorted(set(openmm_terms) & set(dmff_terms)):
        delta = openmm_terms[name] - dmff_terms[name]
        print(f"  {name:28s} {delta:16.8f} kJ/mol")


if __name__ == "__main__":
    main()
