# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lightweight OpenMM/PhyNEO builder driven by PDB topology + DMFF XML."""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import tempfile
import typing as T
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import openmm as omm
import openmm.app as app
import openmm.unit as openmm_unit
from openmm.app.gromacstopfile import GromacsTopFile

try:
    from bytemol.toolkit.asetool.basecalculator import BaseCalculator
except Exception:
    class BaseCalculator:  # type: ignore[no-redef]
        """Fallback base class when ByteMol is unavailable."""

        implemented_properties: list[str] = []

        def __init__(self) -> None:
            pass


logger = logging.getLogger(__name__)

FILE_ROOT = Path(__file__).resolve().parent
if (FILE_ROOT / "openmmapi").exists() and (FILE_ROOT / "python").exists():
    PLUGIN_ROOT = FILE_ROOT
    WORKSPACE_ROOT = FILE_ROOT.parent
else:
    WORKSPACE_ROOT = FILE_ROOT
    PLUGIN_ROOT = WORKSPACE_ROOT / "OpenMMPhyNEOPlugin"
POLFF_ROOT = WORKSPACE_ROOT / "polff"
DEFAULT_THOLE_WIDTH = 5.0
DEFAULT_POLARIZATION = "extrapolated"
DEFAULT_S12 = 0.169


def _add_local_python_paths() -> None:
    candidates = [POLFF_ROOT, PLUGIN_ROOT, PLUGIN_ROOT / "python"]
    candidates.extend(
        build_dir / "python"
        for build_dir in sorted(PLUGIN_ROOT.glob("build*/"))
        if (build_dir / "python" / "phyneoforceplugin.py").exists()
    )
    for candidate in reversed(candidates):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_add_local_python_paths()

try:
    import phyneoforceplugin  # type: ignore
except Exception as exc:  # pragma: no cover - import error path
    raise ImportError(
        "Failed to import phyneoforceplugin. Activate the mpid84 environment "
        "or add a local MPID build python directory to PYTHONPATH."
    ) from exc

from dmff_sr_custom_forces import (  # noqa: E402
    add_dmff_short_range_forces,
    add_dmff_short_range_term,
    configure_short_range_nonbonded_method,
    infer_atom_types_and_bonds_from_topology,
    parse_dmff_sr_xml,
    scale_for_bond_separation,
    shortest_bond_separations,
)


def load_local_plugin_libraries() -> None:
    """Load locally built plugin libraries when the package is not installed."""

    for build_dir in sorted(PLUGIN_ROOT.glob("build*/")):
        for lib_path in build_dir.glob("platforms/*/libOpenMMPhyNEOForce*.dylib"):
            try:
                omm.Platform.loadPluginLibrary(str(lib_path.resolve()))
            except Exception as exc:  # pragma: no cover - best effort loader
                msg = str(exc).lower()
                if "already" not in msg:
                    logger.debug("Skipping plugin library %s: %s", lib_path, exc)


def _coerce_box_vectors(
    box_vectors: T.Optional[T.Iterable[T.Union[omm.Vec3, T.Iterable[float]]]]
) -> T.Optional[tuple[omm.Vec3, omm.Vec3, omm.Vec3]]:
    if box_vectors is None:
        return None
    vectors = []
    for vec in box_vectors:
        if isinstance(vec, omm.Vec3):
            vectors.append(vec)
        else:
            values = list(vec)
            if len(values) != 3:
                raise ValueError("Each box vector must have exactly three components.")
            vectors.append(omm.Vec3(float(values[0]), float(values[1]), float(values[2])))
    if len(vectors) != 3:
        raise ValueError("box_vectors must contain exactly three vectors.")
    return vectors[0], vectors[1], vectors[2]


def _get_topology_box_vectors(pdb: app.PDBFile) -> T.Optional[tuple[omm.Vec3, omm.Vec3, omm.Vec3]]:
    box_vectors = pdb.topology.getPeriodicBoxVectors()
    if box_vectors is None:
        return None
    return box_vectors[0], box_vectors[1], box_vectors[2]


def _find_admp_pme_force(system: omm.System):
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if phyneoforceplugin.ADMPPmeForce.isinstance(force):
            return force
    raise RuntimeError("ADMPPmeForce not found after XML parsing.")


def normalize_identifier(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", name).upper()


def nx_covalent_map_and_pairs(natoms: int, pairs: T.Iterable[T.Iterable[int]]) -> tuple[dict[int, dict[str, list[int]]], dict[str, set[tuple[int, int]]]]:
    graph = nx.Graph()
    normalized_pairs = {tuple(sorted((int(pair[0]), int(pair[1])))) for pair in pairs}
    graph.add_nodes_from(range(natoms))
    graph.add_edges_from(normalized_pairs)

    covalent_map: dict[int, dict[str, list[int]]] = {}
    pair_recs: dict[str, set[tuple[int, int]]] = {label: set() for label in ("1-2", "1-3", "1-4", "1-5", "1-6")}
    for node in graph.nodes():
        neighbors = {label: [] for label in pair_recs}
        all_neighbors: set[int] = set()
        for target, distance in nx.single_source_shortest_path_length(graph, source=node, cutoff=4).items():
            all_neighbors.add(target)
            if distance == 1:
                neighbors["1-2"].append(target)
                pair_recs["1-2"].add(tuple(sorted((node, target))))
            elif distance == 2:
                neighbors["1-3"].append(target)
                pair_recs["1-3"].add(tuple(sorted((node, target))))
            elif distance == 3:
                neighbors["1-4"].append(target)
                pair_recs["1-4"].add(tuple(sorted((node, target))))
            elif distance == 4:
                neighbors["1-5"].append(target)
                pair_recs["1-5"].add(tuple(sorted((node, target))))
        neighbors["1-6"] = sorted(set(range(natoms)) - {node} - all_neighbors)
        pair_recs["1-6"] |= {tuple(sorted((node, neighbor))) for neighbor in neighbors["1-6"]}
        covalent_map[node] = neighbors
    return covalent_map, pair_recs


@dataclass(frozen=True)
class IntraParameterEntry:
    stem: str
    residue_names: tuple[str, ...]
    atom_count: int
    output_files: dict[str, str]
    smiles_source: str


class IntraGromacsForceBuilder:
    """Build bonded intra-molecular forces from generated params_results files."""

    _BONDED_FORCE_TYPES = (
        omm.HarmonicBondForce,
        omm.HarmonicAngleForce,
        omm.PeriodicTorsionForce,
        omm.RBTorsionForce,
        omm.CMAPTorsionForce,
        omm.CustomBondForce,
        omm.CustomAngleForce,
        omm.CustomTorsionForce,
        omm.CustomCompoundBondForce,
    )

    def __init__(self, params_dir: T.Union[str, Path]) -> None:
        self.params_dir = Path(params_dir)
        self.index_path = self.params_dir / "smiles_index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Missing {self.index_path}. Run the params generation script before enabling intra forces."
            )

        with open(self.index_path, "r", encoding="utf-8") as handle:
            raw_index = json.load(handle)

        self.entries: dict[str, IntraParameterEntry] = {}
        self.alias_map: dict[str, list[str]] = defaultdict(list)
        for stem, payload in raw_index.items():
            if payload.get("status", "ok") != "ok":
                continue
            residue_names = tuple(payload.get("residue_names", []))
            output_files = payload.get("output_files", {})
            entry = IntraParameterEntry(
                stem=stem,
                residue_names=residue_names,
                atom_count=int(payload["atom_count"]),
                output_files=output_files,
                smiles_source=str(payload.get("source", "unknown")),
            )
            self.entries[stem] = entry
            aliases = {
                normalize_identifier(stem),
                *(normalize_identifier(name) for name in residue_names),
            }
            aliases |= {
                alias[:-1]
                for alias in list(aliases)
                if alias.endswith("A") and len(alias) > 2
            }
            for alias in sorted(aliases):
                if stem not in self.alias_map[alias]:
                    self.alias_map[alias].append(stem)

    def _topology_components(
        self, topology: app.Topology
    ) -> list[dict[str, T.Any]]:
        graph = nx.Graph()
        atom_indices = {atom: atom.index for atom in topology.atoms()}
        graph.add_nodes_from(atom_indices.values())
        graph.add_edges_from((atom_indices[a], atom_indices[b]) for a, b in topology.bonds())

        components = []
        for component in nx.connected_components(graph):
            atom_ids = sorted(component)
            local_index = {atom_id: idx for idx, atom_id in enumerate(atom_ids)}
            residues = []
            residue_names = []
            residue_seen = set()
            for atom in topology.atoms():
                if atom.index not in component:
                    continue
                residue = atom.residue
                if residue.index not in residue_seen:
                    residue_seen.add(residue.index)
                    residues.append(residue)
                    residue_names.append(residue.name)
            local_pairs = []
            component_set = set(atom_ids)
            for atom_a, atom_b in topology.bonds():
                if atom_a.index in component_set and atom_b.index in component_set:
                    offset_a = local_index[atom_a.index]
                    offset_b = local_index[atom_b.index]
                    local_pairs.append((offset_a, offset_b))
            covalent_map, pair_recs = nx_covalent_map_and_pairs(len(atom_ids), local_pairs)
            components.append(
                {
                    "atom_indices": atom_ids,
                    "residue_names": residue_names,
                    "covalent_map": covalent_map,
                    "pair_recs": pair_recs,
                }
            )
        components.sort(key=lambda comp: comp["atom_indices"][0])
        return components

    def _match_entry(self, component: dict[str, T.Any]) -> IntraParameterEntry:
        residue_aliases = []
        for name in component["residue_names"]:
            normalized = normalize_identifier(name)
            residue_aliases.append(normalized)
            if normalized.endswith("A") and len(normalized) > 2:
                residue_aliases.append(normalized[:-1])

        candidates: list[str] = []
        for alias in residue_aliases:
            candidates.extend(self.alias_map.get(alias, []))
        candidates = [stem for stem in dict.fromkeys(candidates) if self.entries[stem].atom_count == len(component["atom_indices"])]
        if len(candidates) == 1:
            return self.entries[candidates[0]]

        atom_matches = [entry for entry in self.entries.values() if entry.atom_count == len(component["atom_indices"])]
        if len(atom_matches) == 1:
            return atom_matches[0]

        raise RuntimeError(
            "Unable to map topology component to params_results entry. "
            f"Residues={component['residue_names']} natoms={len(component['atom_indices'])} candidates={candidates or [entry.stem for entry in atom_matches]}"
        )

    def describe_components(self, topology: app.Topology) -> list[dict[str, T.Any]]:
        description = []
        for component in self._topology_components(topology):
            entry = self._match_entry(component)
            description.append(
                {
                    "stem": entry.stem,
                    "atom_indices": component["atom_indices"],
                    "residue_names": component["residue_names"],
                    "pair_recs": {key: sorted(value) for key, value in component["pair_recs"].items()},
                }
            )
        return description

    def _write_system_topology(self, stems_in_order: list[str]) -> Path:
        unique_stems = list(dict.fromkeys(stems_in_order))
        include_lines = []
        for stem in unique_stems:
            entry = self.entries[stem]
            atp_path = self.params_dir / Path(entry.output_files["atp"]).name
            itp_path = self.params_dir / Path(entry.output_files["itp"]).name
            include_lines.append(f'#include "{atp_path}"')
            include_lines.append(f'#include "{itp_path}"')

        lines = [
            "[ defaults ]",
            "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ",
            "1 1 yes 1.0 1.0",
            "",
            *include_lines,
            "",
            "[ system ]",
            "Generated Intra Force Topology",
            "",
            "[ molecules ]",
        ]
        lines.extend(f"{stem:<16} 1" for stem in stems_in_order)

        handle = tempfile.NamedTemporaryFile("w", suffix=".top", delete=False)
        with handle:
            handle.write("\n".join(lines))
            handle.write("\n")
        return Path(handle.name)

    @staticmethod
    def _clone_force(force: omm.Force) -> omm.Force:
        return omm.XmlSerializer.deserialize(omm.XmlSerializer.serialize(force))

    def build_system(
        self,
        topology: app.Topology,
        *,
        box_vectors: T.Optional[tuple[omm.Vec3, omm.Vec3, omm.Vec3]] = None,
    ) -> tuple[omm.System, list[dict[str, T.Any]]]:
        components = self._topology_components(topology)
        matched = []
        stems_in_order = []
        for component in components:
            entry = self._match_entry(component)
            matched.append(
                {
                    "entry": entry,
                    "atom_indices": component["atom_indices"],
                    "residue_names": component["residue_names"],
                    "pair_recs": component["pair_recs"],
                }
            )
            stems_in_order.append(entry.stem)

        top_path = self._write_system_topology(stems_in_order)
        try:
            top = GromacsTopFile(
                str(top_path),
                periodicBoxVectors=box_vectors if box_vectors is not None else None,
                includeDir=str(self.params_dir),
            )
            intra_system = top.createSystem(
                nonbondedMethod=app.NoCutoff,
                constraints=None,
                removeCMMotion=False,
            )
        finally:
            top_path.unlink(missing_ok=True)
        return intra_system, matched

    def add_to_system(
        self,
        system: omm.System,
        topology: app.Topology,
        *,
        box_vectors: T.Optional[tuple[omm.Vec3, omm.Vec3, omm.Vec3]] = None,
        start_group: T.Optional[int] = None,
    ) -> tuple[list[str], list[dict[str, T.Any]]]:
        intra_system, matched = self.build_system(topology, box_vectors=box_vectors)

        added_names: list[str] = []
        next_group = start_group
        for force_index in range(intra_system.getNumForces()):
            force = intra_system.getForce(force_index)
            if not isinstance(force, self._BONDED_FORCE_TYPES):
                continue
            cloned = self._clone_force(force)
            base_name = force.getName() or type(force).__name__
            cloned.setName(f"Intra{base_name}")
            if next_group is not None:
                cloned.setForceGroup(next_group)
                next_group += 1
            system.addForce(cloned)
            added_names.append(cloned.getName())
        return added_names, matched


def add_long_range_dispersion_force(
    system: omm.System,
    topology: app.Topology,
    xml_path: T.Union[str, Path],
    *,
    cutoff_nm: float = 1.0,
    force_group: T.Optional[int] = None,
    atom_types: T.Optional[list[str]] = None,
    bonds: T.Optional[list[tuple[int, int]]] = None,
) -> omm.CustomNonbondedForce | None:
    """Add undamped C6/C8/C10 dispersion with 1-4/1-5 corrections and LRC."""

    root = ET.parse(str(xml_path)).getroot()
    disp_node = root.find("ADMPDispPmeForce")
    if disp_node is None:
        return None

    disp_params = {}
    for atom in disp_node.findall("Atom"):
        atom_type = atom.attrib["type"]
        disp_params[atom_type] = {
            "C6": float(atom.get("C6", 0.0)),
            "C8": float(atom.get("C8", 0.0)),
            "C10": float(atom.get("C10", 0.0)),
        }

    if atom_types is None or bonds is None:
        atom_types, bonds = infer_atom_types_and_bonds_from_topology(topology, str(xml_path))

    mscales = [
        float(disp_node.attrib.get(f"mScale1{i}", default))
        for i, default in zip(range(2, 7), [0.0, 0.0, 0.0, 1.0, 1.0])
    ]
    nb_pairs = shortest_bond_separations(len(atom_types), bonds, max_sep=3)
    all_intra = shortest_bond_separations(len(atom_types), bonds, max_sep=5)
    corr_pairs = {k: v for k, v in all_intra.items() if k not in nb_pairs}

    expr = "-(c61*c62)/r^6-(c81*c82)/r^8-(c101*c102)/r^10"
    force = omm.CustomNonbondedForce(expr)
    force.setName("DampedDispersionForce")
    force.addPerParticleParameter("c6")
    force.addPerParticleParameter("c8")
    force.addPerParticleParameter("c10")
    configure_short_range_nonbonded_method(force, system)
    if force.getNonbondedMethod() == omm.CustomNonbondedForce.CutoffPeriodic:
        force.setCutoffDistance(cutoff_nm)
        force.setUseLongRangeCorrection(True)
    elif force.getNonbondedMethod() == omm.CustomNonbondedForce.CutoffNonPeriodic:
        force.setCutoffDistance(cutoff_nm)
        force.setUseLongRangeCorrection(False)

    bond_force = omm.CustomBondForce("scale*(-c6ij/r^6-c8ij/r^8-c10ij/r^10)")
    bond_force.setName("DampedDispersionForce")
    bond_force.addPerBondParameter("scale")
    bond_force.addPerBondParameter("c6ij")
    bond_force.addPerBondParameter("c8ij")
    bond_force.addPerBondParameter("c10ij")

    for atom_type in atom_types:
        params = disp_params.get(atom_type, {"C6": 0.0, "C8": 0.0, "C10": 0.0})
        force.addParticle([
            math.sqrt(abs(params["C6"])),
            math.sqrt(abs(params["C8"])),
            math.sqrt(abs(params["C10"])),
        ])

    def _pair_values(i: int, j: int, scale: float) -> list[float]:
        p_i = disp_params.get(atom_types[i], {"C6": 0.0, "C8": 0.0, "C10": 0.0})
        p_j = disp_params.get(atom_types[j], {"C6": 0.0, "C8": 0.0, "C10": 0.0})
        return [
            scale,
            math.sqrt(abs(p_i["C6"] * p_j["C6"])),
            math.sqrt(abs(p_i["C8"] * p_j["C8"])),
            math.sqrt(abs(p_i["C10"] * p_j["C10"])),
        ]

    for (i, j), separation in nb_pairs.items():
        force.addExclusion(i, j)
        scale = scale_for_bond_separation(mscales, separation)
        if abs(scale) > 1e-15:
            bond_force.addBond(i, j, _pair_values(i, j, scale))

    for (i, j), separation in corr_pairs.items():
        mscale = scale_for_bond_separation(mscales, separation)
        corr_scale = mscale - 1.0
        if abs(corr_scale) > 1e-15:
            bond_force.addBond(i, j, _pair_values(i, j, corr_scale))

    if force_group is not None:
        force.setForceGroup(force_group)
        bond_force.setForceGroup(force_group)

    system.addForce(force)
    system.addForce(bond_force)
    return force


def generate_openmm_system(
    pdb_path: T.Union[str, Path],
    xml_path: T.Union[str, Path],
    *,
    box_vectors: T.Optional[T.Iterable[T.Union[omm.Vec3, T.Iterable[float]]]] = None,
    cutoff: float = 1.0,
    use_periodic: T.Optional[bool] = None,
    polarization: str = DEFAULT_POLARIZATION,
    default_thole_width: float = DEFAULT_THOLE_WIDTH,
    s12: float = DEFAULT_S12,
    add_slater_damping: bool = True,
    enable_intra: bool = False,
    intra_params_dir: T.Optional[T.Union[str, Path]] = None,
) -> tuple[app.PDBFile, omm.System]:
    """Build a PDB + XML based OpenMM system using ADMPPmeForce + custom SR forces."""

    load_local_plugin_libraries()

    pdb = app.PDBFile(str(pdb_path))
    xml_path = str(xml_path)

    explicit_box = _coerce_box_vectors(box_vectors)
    topology_box = _get_topology_box_vectors(pdb)
    active_box = explicit_box if explicit_box is not None else topology_box

    if use_periodic is None:
        use_periodic = active_box is not None
    if use_periodic and active_box is None:
        raise ValueError("Periodic simulation requested but no periodic box vectors were provided in the PDB or arguments.")
    if explicit_box is not None:
        pdb.topology.setPeriodicBoxVectors(explicit_box)

    ff = app.ForceField(xml_path)
    nonbonded_method = app.PME if use_periodic else app.NoCutoff
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=cutoff * openmm_unit.nanometer,
        constraints=None,
        removeCMMotion=False,
        polarization=polarization,
        defaultTholeWidth=default_thole_width,
    )
    if use_periodic and active_box is not None:
        system.setDefaultPeriodicBoxVectors(*active_box)

    admp_force = _find_admp_pme_force(system)
    admp_force.setName("ADMPPmeForce")

    atom_types, bonds = infer_atom_types_and_bonds_from_topology(pdb.topology, xml_path)
    force_data = parse_dmff_sr_xml(xml_path, include_slater_damping=add_slater_damping)

    next_group = system.getNumForces()
    add_long_range_dispersion_force(
        system,
        pdb.topology,
        xml_path,
        cutoff_nm=cutoff,
        force_group=next_group,
        atom_types=atom_types,
        bonds=bonds,
    )
    next_group += 1

    add_dmff_short_range_forces(
        system,
        atom_types,
        bonds,
        force_data,
        start_group=next_group,
        s12=s12,
    )
    next_group += len(force_data)

    if add_slater_damping and "SlaterDampingForce" in force_data:
        add_dmff_short_range_term(
            system,
            atom_types,
            bonds,
            "SlaterDampingForce",
            force_data["SlaterDampingForce"],
            start_group=next_group,
            s12=s12,
        )

    if enable_intra:
        params_dir = Path(intra_params_dir) if intra_params_dir is not None else PROJECT_ROOT / "params_results"
        intra_builder = IntraGromacsForceBuilder(params_dir)
        intra_builder.add_to_system(
            system,
            pdb.topology,
            box_vectors=active_box,
            start_group=system.getNumForces(),
        )

    return pdb, system


class PhyNEOCalculator(BaseCalculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        pdb_path: T.Union[str, Path],
        xml_path: T.Union[str, Path],
        *,
        platform_name: str = "Reference",
        separate_terms: bool = False,
        apply_constraints: bool = False,
        cutoff: float = 1.0,
        use_periodic: T.Optional[bool] = None,
        box_vectors: T.Optional[T.Iterable[T.Union[omm.Vec3, T.Iterable[float]]]] = None,
        polarization: str = DEFAULT_POLARIZATION,
        default_thole_width: float = DEFAULT_THOLE_WIDTH,
        s12: float = DEFAULT_S12,
        add_slater_damping: bool = True,
        enable_intra: bool = False,
        intra_params_dir: T.Optional[T.Union[str, Path]] = None,
    ) -> None:
        super().__init__()

        if platform_name not in ["CPU", "Reference", "CUDA"]:
            raise ValueError(f"Unsupported platform: {platform_name}")

        self.pdb, self.system = generate_openmm_system(
            pdb_path,
            xml_path,
            box_vectors=box_vectors,
            cutoff=cutoff,
            use_periodic=use_periodic,
            polarization=polarization,
            default_thole_width=default_thole_width,
            s12=s12,
            add_slater_damping=add_slater_damping,
            enable_intra=enable_intra,
            intra_params_dir=intra_params_dir,
        )

        self.integrator = omm.VerletIntegrator(0.001 * openmm_unit.picoseconds)
        self.platform = omm.Platform.getPlatformByName(platform_name)
        if platform_name == "CUDA":
            self.platform.setPropertyDefaultValue("Precision", "mixed")

        if separate_terms:
            self.forcegroups = {}
            self.separate_forces = {}
            self.separate_energy = {}
            for i in range(self.system.getNumForces()):
                force = self.system.getForce(i)
                force.setForceGroup(i)
                self.forcegroups[force] = (i, force.getName() or type(force).__name__)
        else:
            self.forcegroups = None
            self.separate_forces = None
            self.separate_energy = None

        self.simulation = app.Simulation(self.pdb.topology, self.system, self.integrator, self.platform)
        if self.pdb.topology.getPeriodicBoxVectors() is not None:
            self.simulation.context.setPeriodicBoxVectors(*self.pdb.topology.getPeriodicBoxVectors())
        self.simulation.context.setPositions(self.pdb.positions)

        self._apply_constraints = apply_constraints
        self._last_state: T.Optional[omm.State] = None
        self._last_positions: T.Optional[np.ndarray] = None
        self._n_atoms = self.system.getNumParticles()

    def _calculate_without_restraint(self, coords: np.ndarray) -> tuple[float, np.ndarray]:
        assert coords.shape[0] == self._n_atoms
        self.simulation.context.setPositions(coords / 10.0)

        if self.apply_constraints:
            self.simulation.context.applyConstraints(1e-5)

        state = self.simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
        self._last_state = state
        self._last_positions = np.array(state.getPositions().value_in_unit(openmm_unit.angstrom))
        energy = state.getPotentialEnergy().value_in_unit(openmm_unit.kilocalorie_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            openmm_unit.kilocalories_per_mole / openmm_unit.angstroms
        )
        return energy, forces

    def calculate(self, coords: np.ndarray) -> tuple[float, np.ndarray]:
        return self._calculate_without_restraint(coords)

    def step(self, steps: int) -> None:
        self.simulation.step(steps)

    def get_induced_dipole(self) -> np.ndarray:
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if phyneoforceplugin.ADMPPmeForce.isinstance(force):
                admp_force = phyneoforceplugin.ADMPPmeForce.cast(force)
                dipoles = admp_force.getInducedDipoles(self.simulation.context)
                return np.array([[dipole.x, dipole.y, dipole.z] for dipole in dipoles]) * 10.0
        raise RuntimeError("ADMPPmeForce not found in system.")

    def get_separate_terms(self) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        if self.forcegroups is None:
            raise RuntimeError("separate_terms=False, no force groups were recorded.")

        self.separate_energy = {}
        self.separate_forces = {}
        for _, (i, name) in self.forcegroups.items():
            state = self.simulation.context.getState(getEnergy=True, getForces=True, groups=2**i)
            energy = state.getPotentialEnergy().value_in_unit(openmm_unit.kilocalorie_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(
                openmm_unit.kilocalories_per_mole / openmm_unit.angstroms
            )
            if name in self.separate_forces:
                self.separate_energy[name] += energy
                self.separate_forces[name] += forces
            else:
                self.separate_energy[name] = energy
                self.separate_forces[name] = forces
        return self.separate_energy, self.separate_forces

    def serialize(self, xml_path: T.Union[str, Path]) -> None:
        with open(xml_path, "w", encoding="utf-8") as output_file:
            output_file.write(omm.XmlSerializer.serialize(self.system))

    @property
    def apply_constraints(self) -> bool:
        return self._apply_constraints

    @property
    def last_openmm_positions(self) -> T.Optional[np.ndarray]:
        return self._last_positions

    @property
    def last_openmm_state(self) -> T.Optional[omm.State]:
        return self._last_state


# Backward-compatible alias for existing imports in this repository.
AmoebaCalculator = PhyNEOCalculator
