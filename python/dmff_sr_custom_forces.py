"""Build OpenMM custom-force versions of DMFF short-range terms."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import deque

import openmm as mm
import openmm.app as app
from openmm import unit

DIELECTRIC = 1389.35455846

TERM_ORDER = [
    "QqTtDampingForce",
    "SlaterExForce",
    "SlaterSrEsForce",
    "SlaterSrPolForce",
    "SlaterSrDispForce",
    "SlaterDhfForce",
    "SlaterDampingForce",
]


def parse_force_section(root, force_name):
    node = root.find(force_name)
    if node is None:
        raise ValueError(f"Missing <{force_name}> in XML")
    params = {}
    for atom in node.findall("Atom"):
        atom_type = atom.attrib["type"]
        params[atom_type] = {key: float(val) for key, val in atom.attrib.items() if key != "type"}
    mscales = [
        float(node.attrib.get(f"mScale1{i}", default))
        for i, default in zip(range(2, 7), [0.0, 0.0, 0.0, 1.0, 1.0])
    ]
    return {"params": params, "mscales": mscales}


def parse_dmff_sr_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    return {force_name: parse_force_section(root, force_name) for force_name in TERM_ORDER}


def parse_residue_templates(root):
    templates = {}
    residues = root.find("Residues")
    if residues is None:
        return templates
    for residue in residues.findall("Residue"):
        atoms = [(atom.attrib["name"], atom.attrib["type"]) for atom in residue.findall("Atom")]
        bonds = [(int(bond.attrib["from"]), int(bond.attrib["to"])) for bond in residue.findall("Bond")]
        templates[residue.attrib["name"]] = {"atoms": atoms, "bonds": bonds}
    return templates


def load_pdb_types_and_bonds(pdb_path, xml_path):
    pdb = app.PDBFile(pdb_path)
    atom_types, bonds = infer_atom_types_and_bonds_from_topology(pdb.topology, xml_path)
    positions_nm = pdb.positions.value_in_unit(unit.nanometer)
    return atom_types, bonds, positions_nm


def infer_atom_types_and_bonds_from_topology(topology, xml_path):
    root = ET.parse(xml_path).getroot()
    templates = parse_residue_templates(root)
    atom_types = []
    bonds = set()
    atom_index_map = {}
    index = 0
    for residue in topology.residues():
        if residue.name not in templates:
            raise ValueError(f"Residue {residue.name} not found in XML template section")
        template = templates[residue.name]
        by_name = {name: atom_type for name, atom_type in template["atoms"]}
        residue_atoms = list(residue.atoms())
        for atom in residue_atoms:
            if atom.name not in by_name:
                raise ValueError(f"Atom {atom.name} in residue {residue.name} not found in XML template")
            atom_types.append(by_name[atom.name])
            atom_index_map[(residue.index, atom.name)] = index
            index += 1
        for i, j in template["bonds"]:
            global_i = atom_index_map[(residue.index, residue_atoms[i].name)]
            global_j = atom_index_map[(residue.index, residue_atoms[j].name)]
            bonds.add(tuple(sorted((global_i, global_j))))
    for bond in topology.bonds():
        bonds.add(tuple(sorted((bond[0].index, bond[1].index))))
    return atom_types, sorted(bonds)


def shortest_bond_separations(num_atoms, bonds, max_sep=5):
    graph = [[] for _ in range(num_atoms)]
    for i, j in bonds:
        graph[i].append(j)
        graph[j].append(i)
    out = {}
    for start in range(num_atoms):
        visited = {start: 0}
        queue = deque([start])
        while queue:
            src = queue.popleft()
            dist = visited[src]
            if dist >= max_sep:
                continue
            for dst in graph[src]:
                if dst not in visited:
                    visited[dst] = dist + 1
                    queue.append(dst)
        for stop, dist in visited.items():
            if start < stop and 1 <= dist <= max_sep:
                out[(start, stop)] = dist
    return out


def scale_for_bond_separation(mscales, separation):
    if separation is None or separation > 5:
        return 1.0
    return mscales[separation - 1]


def term_energy_expression(force_name):
    if force_name == "QqTtDampingForce":
        return "-0.1*dielectric*q1*q2*exp(-br)*(1+br)/r; br=sqrt(b1*b2)*r"
    if force_name == "SlaterExForce":
        return "a1*a2*(1+br+br^2/3)*exp(-br) + a1*a2/(0.24*br)^14; br=sqrt(b1*b2)*r"
    if force_name in {"SlaterSrEsForce", "SlaterSrPolForce", "SlaterSrDispForce", "SlaterDhfForce"}:
        return "-a1*a2*(1+br+br^2/3)*exp(-br); br=sqrt(b1*b2)*r"
    if force_name == "SlaterDampingForce":
        return (
            "exp(-x)*("
            "(1+x+x^2/2+x^3/6+x^4/24+x^5/120+x^6/720)*(c61*c62)/r^6"
            "+(1+x+x^2/2+x^3/6+x^4/24+x^5/120+x^6/720+x^7/5040+x^8/40320)*(c81*c82)/r^8"
            "+(1+x+x^2/2+x^3/6+x^4/24+x^5/120+x^6/720+x^7/5040+x^8/40320+x^9/362880+x^10/3628800)*(c101*c102)/r^10"
            ");"
            "x=br-(2*br^2+3*br)/(br^2+3*br+3);"
            "br=sqrt(b1*b2)*r"
        )
    raise ValueError(force_name)


def make_nonbonded_force(force_name):
    force = mm.CustomNonbondedForce(term_energy_expression(force_name))
    if force_name == "QqTtDampingForce":
        force.addGlobalParameter("dielectric", DIELECTRIC)
        force.addPerParticleParameter("b")
        force.addPerParticleParameter("q")
    elif force_name == "SlaterDampingForce":
        force.addPerParticleParameter("b")
        force.addPerParticleParameter("c6")
        force.addPerParticleParameter("c8")
        force.addPerParticleParameter("c10")
    else:
        force.addPerParticleParameter("a")
        force.addPerParticleParameter("b")
    force.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
    return force


def configure_short_range_nonbonded_method(force, system):
    for parent_force in system.getForces():
        if isinstance(parent_force, mm.NonbondedForce):
            method = parent_force.getNonbondedMethod()
            if method in (mm.NonbondedForce.PME, mm.NonbondedForce.CutoffPeriodic, mm.NonbondedForce.LJPME, mm.NonbondedForce.Ewald):
                force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
                force.setCutoffDistance(parent_force.getCutoffDistance())
                if hasattr(parent_force, "getUseSwitchingFunction") and parent_force.getUseSwitchingFunction():
                    force.setUseSwitchingFunction(True)
                    force.setSwitchingDistance(parent_force.getSwitchingDistance())
            elif method == mm.NonbondedForce.CutoffNonPeriodic:
                force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
                force.setCutoffDistance(parent_force.getCutoffDistance())
            else:
                force.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
            return
        try:
            import mpidplugin  # type: ignore

            if mpidplugin.MPIDForce.isinstance(parent_force):
                mpid_force = mpidplugin.MPIDForce.cast(parent_force)
                method = mpid_force.getNonbondedMethod()
                if method == mpidplugin.MPIDForce.PME:
                    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
                    force.setCutoffDistance(mpid_force.getCutoffDistance())
                else:
                    force.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
                return
        except Exception:
            pass


def bond_energy_expression(force_name):
    if force_name == "QqTtDampingForce":
        return "scale*(-0.1*dielectric*qij*exp(-bij*r)*(1+bij*r)/r)"
    if force_name == "SlaterExForce":
        return "scale*(aij*(1+bij*r+(bij*r)^2/3)*exp(-bij*r) + aij/(0.24*bij*r)^14)"
    if force_name in {"SlaterSrEsForce", "SlaterSrPolForce", "SlaterSrDispForce", "SlaterDhfForce"}:
        return "scale*(-aij*(1+bij*r+(bij*r)^2/3)*exp(-bij*r))"
    if force_name == "SlaterDampingForce":
        x = "bij*r-(2*(bij*r)^2+3*(bij*r))/((bij*r)^2+3*(bij*r)+3)"
        return (
            "scale*exp(-("
            + x
            + "))*((1+("
            + x
            + ")+("
            + x
            + ")^2/2+("
            + x
            + ")^3/6+("
            + x
            + ")^4/24+("
            + x
            + ")^5/120+("
            + x
            + ")^6/720)*c6ij/r^6"
            "+(1+("
            + x
            + ")+("
            + x
            + ")^2/2+("
            + x
            + ")^3/6+("
            + x
            + ")^4/24+("
            + x
            + ")^5/120+("
            + x
            + ")^6/720+("
            + x
            + ")^7/5040+("
            + x
            + ")^8/40320)*c8ij/r^8"
            "+(1+("
            + x
            + ")+("
            + x
            + ")^2/2+("
            + x
            + ")^3/6+("
            + x
            + ")^4/24+("
            + x
            + ")^5/120+("
            + x
            + ")^6/720+("
            + x
            + ")^7/5040+("
            + x
            + ")^8/40320+("
            + x
            + ")^9/362880+("
            + x
            + ")^10/3628800)*c10ij/r^10)"
        )
    raise ValueError(force_name)


def make_bond_force(force_name):
    force = mm.CustomBondForce(bond_energy_expression(force_name))
    if force_name == "QqTtDampingForce":
        force.addGlobalParameter("dielectric", DIELECTRIC)
        for name in ("scale", "bij", "qij"):
            force.addPerBondParameter(name)
    elif force_name == "SlaterDampingForce":
        for name in ("scale", "bij", "c6ij", "c8ij", "c10ij"):
            force.addPerBondParameter(name)
    else:
        for name in ("scale", "aij", "bij"):
            force.addPerBondParameter(name)
    return force


def particle_params(force_name, force_section, atom_type):
    atom_params = force_section["params"][atom_type]
    if force_name == "QqTtDampingForce":
        return [atom_params["B"], atom_params["Q"]]
    if force_name == "SlaterDampingForce":
        return [
            atom_params["B"],
            math.sqrt(atom_params["C6"]),
            math.sqrt(atom_params["C8"]),
            math.sqrt(atom_params["C10"]),
        ]
    return [atom_params["A"], atom_params["B"]]


def pair_params(force_name, force_section, type_i, type_j, scale):
    p_i = force_section["params"][type_i]
    p_j = force_section["params"][type_j]
    bij = math.sqrt(p_i["B"] * p_j["B"])
    if force_name == "QqTtDampingForce":
        return [scale, bij, p_i["Q"] * p_j["Q"]]
    if force_name == "SlaterDampingForce":
        return [
            scale,
            bij,
            math.sqrt(p_i["C6"] * p_j["C6"]),
            math.sqrt(p_i["C8"] * p_j["C8"]),
            math.sqrt(p_i["C10"] * p_j["C10"]),
        ]
    return [scale, p_i["A"] * p_j["A"], bij]


def add_dmff_short_range_forces(system, atom_types, bonds, force_data, start_group=0):
    bonded_pairs = shortest_bond_separations(len(atom_types), bonds, max_sep=5)
    for group_offset, force_name in enumerate(TERM_ORDER):
        section = force_data[force_name]
        nb_force = make_nonbonded_force(force_name)
        configure_short_range_nonbonded_method(nb_force, system)
        bond_force = make_bond_force(force_name)
        for atom_type in atom_types:
            nb_force.addParticle(particle_params(force_name, section, atom_type))
        for (i, j), separation in bonded_pairs.items():
            nb_force.addExclusion(i, j)
            scale = scale_for_bond_separation(section["mscales"], separation)
            if abs(scale) > 1e-15:
                bond_force.addBond(i, j, pair_params(force_name, section, atom_types[i], atom_types[j], scale))
        nb_force.setForceGroup(start_group + group_offset)
        bond_force.setForceGroup(start_group + group_offset)
        system.addForce(nb_force)
        system.addForce(bond_force)
    return system


def add_dmff_short_range_forces_from_xml(system, topology, xml_path, start_group=0):
    atom_types, bonds = infer_atom_types_and_bonds_from_topology(topology, xml_path)
    force_data = parse_dmff_sr_xml(xml_path)
    return add_dmff_short_range_forces(system, atom_types, bonds, force_data, start_group=start_group)


def build_system(atom_types, bonds, force_data, particle_mass=39.9):
    system = mm.System()
    for _ in atom_types:
        system.addParticle(particle_mass)
    return add_dmff_short_range_forces(system, atom_types, bonds, force_data)


def default_types(topology_name):
    if topology_name == "dimer":
        return ["6", "8"]
    if topology_name == "chain7":
        return ["6", "7", "8", "9", "6", "7", "8"]
    raise ValueError(topology_name)


def default_bonds(topology_name):
    if topology_name == "dimer":
        return []
    if topology_name == "chain7":
        return [(i, i + 1) for i in range(6)]
    raise ValueError(topology_name)


def build_positions(topology_name, spacing_nm):
    if topology_name == "dimer":
        return [(0.0, 0.0, 0.0), (spacing_nm, 0.0, 0.0)]
    if topology_name == "chain7":
        return [(i * spacing_nm, 0.0, 0.0) for i in range(7)]
    raise ValueError(topology_name)


def energy_by_group(context, group):
    state = context.getState(getEnergy=True, groups={group})
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def dmff_reference_energies(force_data, atom_types, bonds, positions_nm):
    try:
        import jax.numpy as jnp
        from dmff.admp.pairwise import TT_damping_qq_kernel, slater_disp_damping_kernel, slater_sr_hc_kernel, slater_sr_kernel
    except Exception as exc:
        raise RuntimeError("DMFF import failed. Run this option in an environment that can import dmff.") from exc

    bonded_pairs = shortest_bond_separations(len(atom_types), bonds, max_sep=5)
    positions_nm = [tuple(p) for p in positions_nm]
    out = {}
    for force_name in TERM_ORDER:
        section = force_data[force_name]
        total = 0.0
        for i in range(len(atom_types)):
            for j in range(i + 1, len(atom_types)):
                separation = bonded_pairs.get((i, j))
                scale = scale_for_bond_separation(section["mscales"], separation)
                if abs(scale) < 1e-15:
                    continue
                dx = positions_nm[i][0] - positions_nm[j][0]
                dy = positions_nm[i][1] - positions_nm[j][1]
                dz = positions_nm[i][2] - positions_nm[j][2]
                dr_a = math.sqrt(dx * dx + dy * dy + dz * dz) * 10.0
                m = jnp.array([scale])
                pi = section["params"][atom_types[i]]
                pj = section["params"][atom_types[j]]
                bi = jnp.array([pi["B"] / 10.0])
                bj = jnp.array([pj["B"] / 10.0])
                if force_name == "QqTtDampingForce":
                    energy = TT_damping_qq_kernel(jnp.array([dr_a]), m, bi, bj, jnp.array([pi["Q"]]), jnp.array([pj["Q"]]))
                elif force_name == "SlaterExForce":
                    energy = slater_sr_hc_kernel(jnp.array([dr_a]), m, jnp.array([pi["A"]]), jnp.array([pj["A"]]), bi, bj)
                elif force_name in {"SlaterSrEsForce", "SlaterSrPolForce", "SlaterSrDispForce", "SlaterDhfForce"}:
                    energy = -slater_sr_kernel(jnp.array([dr_a]), m, jnp.array([pi["A"]]), jnp.array([pj["A"]]), bi, bj)
                elif force_name == "SlaterDampingForce":
                    energy = slater_disp_damping_kernel(
                        jnp.array([dr_a]), m, bi, bj,
                        jnp.array([math.sqrt(pi["C6"] * 1e6)]), jnp.array([math.sqrt(pj["C6"] * 1e6)]),
                        jnp.array([math.sqrt(pi["C8"] * 1e8)]), jnp.array([math.sqrt(pj["C8"] * 1e8)]),
                        jnp.array([math.sqrt(pi["C10"] * 1e10)]), jnp.array([math.sqrt(pj["C10"] * 1e10)]),
                    )
                else:
                    raise ValueError(force_name)
                total += float(energy[0])
        out[force_name] = total
    return out
