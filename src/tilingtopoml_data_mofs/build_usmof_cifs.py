"""Library: regenerate USMOF CIFs from the Ball et al. 2026 building blocks
and extract per-atom clustering data alongside each build.

The Ball 2026 deposit ships only feature/label CSVs and the xyz building blocks;
CIFs aren't deposited. The CSV `name` column encodes a PORMAKE construction
recipe (e.g. ``MOF_net-sxg_node1-N70_edge1-E15``). For every unique name we:

  1. Resolve the building-block keys from the deposit's three lookup CSVs.
  2. Run PORMAKE's ``Builder.build_by_type`` to materialize the framework.
  3. Walk the slot-by-slot atom layout to extract a generic per-atom
     "clustering" (cluster id + lattice offset per atom, vertex/edge flag per
     cluster, edge endpoint info per cluster). See ``push_mofs.py`` for the
     producer CLI that drives this and packs everything into a parquet.

This module is library-only — no Hydra entry point. ``push_mofs.py`` is the
single end-to-end build CLI.
"""

from __future__ import annotations

import logging
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pormake as pm
import pormake.log  # noqa: F401 — side-effect: configures pormake's "unique_logger"

# Suppress pormake's per-slot RMSD INFO chatter. disable_print() logs a
# WARNING on every call, so raising the threshold is quieter.
logging.getLogger("unique_logger").setLevel(logging.ERROR)

NAME_COL = "name"

# MOF_net-{net}_node1-{N#|orgN#}[_node2-{N#|orgN#}]_edge1-{E#|none}
_NAME_RE = re.compile(
    r"^MOF_net-(?P<net>[^_]+)"
    r"_node1-(?P<node1>(?:org)?N\d+)"
    r"(?:_node2-(?P<node2>(?:org)?N\d+))?"
    r"_edge1-(?P<edge>E\d+|none)$"
)


@dataclass(frozen=True)
class Recipe:
    name: str
    net: str
    node_bb_keys: tuple[str, ...]
    edge_bb_key: str | None  # None for edgeless MOFs (literal `edge1-none` in name)


def parse_mof_recipe(name: str) -> Recipe:
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"Could not parse USMOF name: {name!r}")
    nodes = (m["node1"],) if m["node2"] is None else (m["node1"], m["node2"])
    edge = None if m["edge"] == "none" else m["edge"]
    return Recipe(name=name, net=m["net"], node_bb_keys=nodes, edge_bb_key=edge)


def load_bb_key_map(construction_dir: Path) -> dict[str, str]:
    """Concatenate the three building-block key CSVs into a single ``bb_key -> xyz_basename`` map.

    The CSVs have columns ``name`` (xyz basename without .xyz) and ``alias``
    (the short building-block key used in MOF names, e.g. N42, orgN20, E1).
    Returns ``{bb_key: xyz_basename}``.
    """
    bb_key_dir = construction_dir / "USMOF_building_block_alias"
    parts = []
    for fname in ("inorganic_node.csv", "organic_node.csv", "organic_edge.csv"):
        df = pd.read_csv(bb_key_dir / fname)
        parts.append(df[["alias", "name"]])
    combined = pd.concat(parts, ignore_index=True)
    if combined["alias"].duplicated().any():
        dups = combined.loc[combined["alias"].duplicated(), "alias"].tolist()
        raise ValueError(f"Duplicate building-block keys across alias CSVs: {dups}")
    return dict(zip(combined["alias"], combined["name"]))


def load_inorganic_node_order(construction_dir: Path) -> dict[str, int]:
    """Row index of each inorganic-node building-block key in ``inorganic_node.csv``.

    The Ball 2026 deposit's build notebook assigns building blocks to topology
    node-types by *positional zip* of ``[inorganic bb_keys in inorganic_node.csv
    row order] + [organic bb_keys in organic_node.csv row order]`` onto
    ``topology.unique_node_types`` — it does NOT use the node1/node2 order from
    the MOF name. We only need this CSV row order to *detect* (and drop) the
    handful of MOFs where our name-order assignment would diverge from theirs;
    see the drop guard in :func:`_build_one`.
    """
    df = pd.read_csv(
        construction_dir / "USMOF_building_block_alias" / "inorganic_node.csv"
    )
    return {bb_key: i for i, bb_key in enumerate(df["alias"])}


def _neighbor_cell_image(
    neighbor, p_edge: np.ndarray, p_neighbor: np.ndarray, inv_cell: np.ndarray
) -> np.ndarray:
    """Integer cell image S of a topology neighbor bonded to the home-cell edge.

    ASE's ``distance_vector`` D satisfies ``D = r_neighbor_image - r_edge``, so
    ``S = (D + r_edge - r_neighbor_home) @ inv_cell``.
    """
    image_float = (neighbor.distance_vector + p_edge - p_neighbor) @ inv_cell
    image = np.rint(image_float).astype(np.int32)
    residual = float(np.max(np.abs(image_float - image)))
    if residual > 2e-3:
        raise ValueError(
            f"Non-integral cell image (residual {residual:.6g}); topology geometry may be corrupt"
        )
    return image


def _topology_edge_endpoints(
    edge_slot: int, topology
) -> tuple[int, int, np.ndarray]:
    """For an edge slot, return ``(src_slot, dst_slot, dst_offset)``.

    ``dst_offset = S_dst - S_src`` is the image of dst when src is in its home cell.
    """
    n_src, n_dst = topology.neighbor_list[edge_slot]
    p = topology.atoms.positions
    inv_cell = np.linalg.inv(topology.atoms.cell.array)
    p_e = p[edge_slot]
    S_src = _neighbor_cell_image(n_src, p_e, p[n_src.index], inv_cell)
    S_dst = _neighbor_cell_image(n_dst, p_e, p[n_dst.index], inv_cell)
    return int(n_src.index), int(n_dst.index), S_dst - S_src


def _atom_provenance(framework) -> tuple[np.ndarray, np.ndarray]:
    """Return per-atom slot ids and unwrapped Cartesian positions.

    PORMAKE builds ``framework.atoms`` by iterating ``located_bbs`` in slot
    order, skipping ``None`` slots, and dropping ``X`` (connection-point) atoms.
    Replaying that order assigns each surviving atom its originating slot.
    """
    located_bbs = framework.info["located_bbs"]
    slot_ids: list[int] = []
    positions: list[np.ndarray] = []
    for slot_id, bb in enumerate(located_bbs):
        if bb is None:
            continue
        for atom in bb.atoms:
            if atom.symbol == "X":
                continue
            slot_ids.append(slot_id)
            positions.append(np.asarray(atom.position, dtype=np.float64))
    return (
        np.asarray(slot_ids, dtype=np.int64),
        np.asarray(positions, dtype=np.float64),
    )



def _validate_topology_slots(framework) -> tuple[set[int], set[int], int]:
    """Validate topology slot classification and return (node_slot_set, edge_slot_set, n_clusters)."""
    topology = framework.info["topology"]
    n_clusters = topology.n_slots
    located_bbs = framework.info["located_bbs"]
    if len(located_bbs) != n_clusters:
        raise ValueError(
            "PORMAKE located_bbs/topology slot-count mismatch: "
            f"{len(located_bbs)} vs {n_clusters}"
        )
    node_slot_set = {int(s) for s in topology.node_indices}
    edge_slot_set = {int(s) for s in topology.edge_indices}
    if node_slot_set & edge_slot_set:
        raise ValueError("PORMAKE topology has slots classified as both node and edge")
    expected_slots = set(range(n_clusters))
    classified_slots = node_slot_set | edge_slot_set
    if classified_slots != expected_slots:
        raise ValueError(
            "PORMAKE topology slot classification is incomplete; "
            f"missing={sorted(expected_slots - classified_slots)}, "
            f"unexpected={sorted(classified_slots - expected_slots)}"
        )
    return node_slot_set, edge_slot_set, n_clusters


def extract_clustering(framework) -> dict:
    """Extract the generic per-atom clustering schema from a PORMAKE Framework.

    Returns a dict with the following parallel arrays:
      - ``atom_cluster`` (length n_atoms): per-atom ``{"cluster_id": int}``
      - ``atom_positions`` (length n_atoms): per-atom Cartesian Å
        ``{"x": float, "y": float, "z": float}``
      - ``unit_cell`` (length 3): lattice vectors a, b, c in Cartesian Å,
        one ``{"x": float, "y": float, "z": float}`` per row of the cell
      - ``is_cluster_vertex`` (length n_clusters): per-cluster bool
      - ``cluster_endpoints`` (length n_clusters): per-cluster
        ``{"src": int, "dst": int, "dst_offset": [int, int, int]}``
        (sentinel ``{-1, -1, [0,0,0]}`` for vertex clusters)

    cluster_id is the PORMAKE topology slot index (dense 0..n_slots-1 for
    successful builds; every slot is filled).
    """
    topology = framework.info["topology"]
    atoms = framework.atoms

    # Validate slot classification and get node/edge sets.
    node_slot_set, edge_slot_set, n_clusters = _validate_topology_slots(framework)

    # Assign each atom in the built framework to its originating topology slot.
    atom_slot, unwrapped_positions = _atom_provenance(framework)
    if not set(int(slot) for slot in atom_slot) <= set(range(n_clusters)):
        raise ValueError("Atom provenance references a non-topology slot")

    # Per-cluster metadata: vertex flag and edge endpoint info.
    # Use the topology (not framework bonds) as the source for endpoint images:
    # bonds cannot distinguish the two neighbor images of a periodic self-edge
    # (src == dst), and atom-less edge slots have no framework atoms at all.
    is_cluster_vertex = [c in node_slot_set for c in range(n_clusters)]
    cluster_endpoints: list[dict] = []
    for c in range(n_clusters):
        if c in edge_slot_set:
            src, dst, dst_off = _topology_edge_endpoints(c, topology)
            if src not in node_slot_set or dst not in node_slot_set:
                raise ValueError(
                    f"Topology edge slot {c} endpoints are not both node slots: "
                    f"src={src}, dst={dst}"
                )
            if src == dst and not np.any(dst_off):
                raise ValueError(
                    f"Topology edge slot {c} is a forbidden zero-offset self-edge"
                )
            cluster_endpoints.append({
                "src": int(src),
                "dst": int(dst),
                "dst_offset": [int(dst_off[0]), int(dst_off[1]), int(dst_off[2])],
            })
        else:
            # Sentinel for vertex clusters: no endpoint info.
            cluster_endpoints.append({"src": -1, "dst": -1, "dst_offset": [0, 0, 0]})

    # Per-atom arrays: which cluster each atom belongs to, and its position.
    atom_cluster = [
        {"cluster_id": int(slot)}
        for slot in atom_slot
    ]
    atom_positions = [
        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
        for p in unwrapped_positions
    ]

    # Unit cell: the three lattice vectors as Cartesian rows.
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    unit_cell = [
        {"x": float(cell[i, 0]), "y": float(cell[i, 1]), "z": float(cell[i, 2])}
        for i in range(3)
    ]

    return {
        "atom_cluster": atom_cluster,
        "atom_positions": atom_positions,
        "unit_cell": unit_cell,
        "is_cluster_vertex": is_cluster_vertex,
        "cluster_endpoints": cluster_endpoints,
    }


def _has_ambiguous_node_assignment(
    node_bb_keys: tuple[str, ...],
    topo_cns: list[int],
    inorganic_order: dict[str, int],
) -> bool:
    """Return True for the ~9 MOFs whose node→site assignment we cannot reproduce.

    We assign building blocks to topology node types by matching connection
    counts (CN). This is unambiguous when the two node types have different CNs.
    The one case CN cannot resolve is two *inorganic* nodes on two *same-CN*
    sites: the deposit's true tiebreak is inorganic_node.csv row order, but we
    only have the MOF name's node1/node2 order. When those orders disagree we
    would place the metal clusters on swapped sites, so we drop these rather
    than risk producing a wrong structure.
    """
    if len(node_bb_keys) != 2:
        return False
    if topo_cns[0] != topo_cns[1]:                              # different CNs: unambiguous
        return False
    if not all(k in inorganic_order for k in node_bb_keys):     # not both inorganic
        return False
    name_order = [inorganic_order[k] for k in node_bb_keys]
    return name_order != sorted(name_order)                      # name order ≠ CSV order


def _build_one(args: tuple) -> tuple[str, str, str | None, dict | None]:
    """Worker: build one MOF, return (name, status, cif_text, clustering).

    Status is one of: ``ok``, ``skipped:ambiguous_node_assignment``
    (the ~9 two-inorganic, same-CN MOFs we can't reproduce from the deposit —
    see the drop guard below), ``rmsd_warn:<value>``, ``failed:<reason>``.
    ``cif_text`` and ``clustering`` are ``None`` on skip / failure.
    """
    recipe, bb_key_map, inorganic_order, bb_dir_str, rmsd_warn = args
    bb_dir = Path(bb_dir_str)

    # Look up the abstract net (topology) by name from pormake's database.
    try:
        topo = pm.Database().get_topo(recipe.net)
    except Exception as exc:  # noqa: BLE001
        return recipe.name, f"failed:topology_lookup:{type(exc).__name__}:{exc}", None, None

    node_types = list(topo.unique_node_types)
    edge_types = [tuple(t) for t in topo.unique_edge_types]

    # Sanity check: the recipe must name exactly as many node BBs as the topology has node types.
    if len(recipe.node_bb_keys) != len(node_types):
        return recipe.name, (
            f"failed:node_count_mismatch:"
            f"recipe={len(recipe.node_bb_keys)},topo={len(node_types)}"
        ), None, None

    try:
        # Load each node building block from its xyz file.
        bbs: list = []
        for bb_key in recipe.node_bb_keys:
            xyz_name = bb_key_map.get(bb_key)
            if xyz_name is None:
                return recipe.name, f"failed:unknown_node_bb_key:{bb_key}", None, None
            xyz_path = bb_dir / f"{xyz_name}.xyz"
            if not xyz_path.exists():
                return recipe.name, f"failed:missing_xyz:{xyz_path.name}", None, None
            bbs.append(pm.BuildingBlock(str(xyz_path)))

        # Compare connection-point counts (CNs) of the loaded BBs against the topology's node types.
        topo_cns = [int(c) for c in topo.unique_cn]
        bb_cns = [int(b.n_connection_points) for b in bbs]

        if _has_ambiguous_node_assignment(recipe.node_bb_keys, topo_cns, inorganic_order):
            return recipe.name, "skipped:ambiguous_node_assignment", None, None

        # Match BBs to topology node types by CN; swap if needed.
        if bb_cns == topo_cns:
            ordered_bbs = bbs
        elif len(bbs) == 2 and bb_cns[::-1] == topo_cns:
            ordered_bbs = bbs[::-1]
        else:
            return recipe.name, (
                f"failed:connectivity_mismatch:"
                f"bb_cns={bb_cns},topo_cns={topo_cns}"
            ), None, None
        node_bbs: dict = {node_types[i]: ordered_bbs[i] for i in range(len(node_types))}

        # Load the edge building block if the recipe has one; edgeless MOFs use None.
        edge_bbs: dict | None
        if recipe.edge_bb_key is None:
            edge_bbs = None
        else:
            edge_xyz = bb_key_map.get(recipe.edge_bb_key)
            if edge_xyz is None:
                return recipe.name, f"failed:unknown_edge_bb_key:{recipe.edge_bb_key}", None, None
            edge_xyz_path = bb_dir / f"{edge_xyz}.xyz"
            if not edge_xyz_path.exists():
                return recipe.name, f"failed:missing_xyz:{edge_xyz_path.name}", None, None
            edge_bb = pm.BuildingBlock(str(edge_xyz_path))
            # The same linker BB is used for every edge type in the topology.
            edge_bbs = {e: edge_bb for e in edge_types}

        # Measure how well each node BB's connection geometry fits its topology slot.
        # Non-fatal: if the locator fails for any reason we record nan and proceed.
        max_rmsd = 0.0
        try:
            locator = pm.Locator()
            for i, t in enumerate(node_types):
                rmsd = float(locator.calculate_rmsd(
                    topo.unique_local_structures[i], node_bbs[t]
                ))
                if rmsd > max_rmsd:
                    max_rmsd = rmsd
        except Exception:  # noqa: BLE001
            max_rmsd = float("nan")

        # Build the framework and extract clustering and CIF text.
        builder_kwargs = {"topology": topo, "node_bbs": node_bbs}
        if edge_bbs is not None:
            builder_kwargs["edge_bbs"] = edge_bbs
        framework = pm.Builder().build_by_type(**builder_kwargs)
        clustering = extract_clustering(framework)
        with tempfile.TemporaryDirectory() as tmpdir:
            cif_path = Path(tmpdir) / "mof.cif"
            framework.write_cif(str(cif_path))
            cif_text = cif_path.read_text()
    except Exception as exc:  # noqa: BLE001
        return recipe.name, f"failed:builder_exception:{type(exc).__name__}:{exc}", None, None

    if math.isnan(max_rmsd) or max_rmsd > rmsd_warn:
        return recipe.name, f"rmsd_warn:{max_rmsd:.3f}", cif_text, clustering
    return recipe.name, "ok", cif_text, clustering
