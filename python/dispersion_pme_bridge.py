#!/usr/bin/env python3
"""Dispersion PME bridge model for MPID/OpenMM workflows.

This module exposes DMFF's ADMPDispPmeForce term as a standalone evaluator for
an OpenMM topology and frame. It enables exact energy-term alignment with DMFF
for dispersion PME without changing MPID C++ kernels.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import openmm.app as app
from dmff import Hamiltonian
from dmff.common import nblist


class DispersionPMEBridgeModel:
    """Evaluate DMFF ADMPDispPmeForce on OpenMM geometry frames."""

    def __init__(
        self,
        topology,
        dmff_xml,
        cutoff_nm: float,
        box_nm,
        ethresh: float = 5e-4,
        pmax: int = 10,
        lpme: bool = True,
    ):
        del ethresh, pmax  # Controlled by the DMFF forcefield/generator defaults.

        self.box_nm = np.asarray(box_nm, dtype=np.float64)
        self.cutoff_nm = float(cutoff_nm)

        method = app.NoCutoff if lpme else app.CutoffPeriodic
        self.h = Hamiltonian(str(dmff_xml))
        # Keep createPotential defaults consistent with DMFFEnergyModel.
        self.pot = self.h.createPotential(topology, nonbondedMethod=method)
        self.params = self.h.getParameters()
        self.fn = self.pot.getPotentialFunc(["ADMPDispPmeForce"])

        self.nb = nblist.NeighborList(self.box_nm, self.cutoff_nm, self.pot.meta["cov_map"], padding=False)

    def eval(self, positions_nm) -> float:
        positions_nm = np.asarray(positions_nm, dtype=np.float64)
        pos = jnp.array(positions_nm)
        box = jnp.array(self.box_nm)
        self.nb.update(pos, box)
        return float(self.fn(pos, box, self.nb.pairs, self.params))
