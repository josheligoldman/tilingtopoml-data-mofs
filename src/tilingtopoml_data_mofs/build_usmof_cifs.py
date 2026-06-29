"""Library: regenerate USMOF CIFs from the Ball et al. 2026 building blocks
and extract per-atom clustering data alongside each build.

The Ball 2026 deposit ships only feature/label CSVs and the xyz building blocks;
CIFs aren't deposited. The CSV `name` column encodes a PORMAKE construction
recipe (e.g. ``MOF_net-sxg_node1-N70_edge1-E15``). For every unique name we:

  1. Resolve the building-block aliases from the deposit's three lookup CSVs.
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
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    node_aliases: tuple[str, ...]
    edge_alias: str | None  # None for edgeless MOFs (literal `edge1-none` in name)


def parse_name(name: str) -> Recipe:
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"Could not parse USMOF name: {name!r}")
    nodes = (m["node1"],) if m["node2"] is None else (m["node1"], m["node2"])
    edge = None if m["edge"] == "none" else m["edge"]
    return Recipe(name=name, net=m["net"], node_aliases=nodes, edge_alias=edge)


def load_alias_map(construction_dir: Path) -> dict[str, str]:
    """Concatenate the three alias CSVs into a single ``alias -> xyz_basename`` map.

    The CSVs have columns ``name`` (xyz basename without .xyz) and ``alias``
    (e.g. N42, orgN20, E1). Returns ``{alias: xyz_basename}``.
    """
    alias_dir = construction_dir / "USMOF_building_block_alias"
    parts = []
    for fname in ("inorganic_node.csv", "organic_node.csv", "organic_edge.csv"):
        df = pd.read_csv(alias_dir / fname)
        parts.append(df[["alias", "name"]])
    combined = pd.concat(parts, ignore_index=True)
    if combined["alias"].duplicated().any():
        dups = combined.loc[combined["alias"].duplicated(), "alias"].tolist()
        raise ValueError(f"Duplicate aliases across alias CSVs: {dups}")
    return dict(zip(combined["alias"], combined["name"]))


def load_inorganic_node_order(construction_dir: Path) -> dict[str, int]:
    """Row index of each inorganic-node alias in ``inorganic_node.csv``.

    The Ball 2026 deposit's build notebook assigns building blocks to topology
    node-types by *positional zip* of ``[inorganic aliases in inorganic_node.csv
    row order] + [organic aliases in organic_node.csv row order]`` onto
    ``topology.unique_node_types`` — it does NOT use the node1/node2 order from
    the MOF name. We only need this CSV row order to *detect* (and drop) the
    handful of MOFs where our name-order assignment would diverge from theirs;
    see the drop guard in :func:`_build_one`.
    """
    df = pd.read_csv(
        construction_dir / "USMOF_building_block_alias" / "inorganic_node.csv"
    )
    return {alias: i for i, alias in enumerate(df["alias"])}


def _topology_edge_endpoints(
    edge_slot: int, topology
) -> tuple[int, int, np.ndarray]:
    """For an edge slot, return ``(src_slot, dst_slot, dst_offset)``.

    Works for both atom-ful and atom-less edge slots — every slot has a
    placeholder atom in ``topology.atoms`` (loaded from the .cgd file
    when the topology is constructed), so ``topology.neighbor_list[c]``
    carries the two endpoint vertex slots regardless of whether a BB
    was later placed at ``c`` during framework assembly.

    ``Neighbor.distance_vector`` is ASE's ``D`` output: the Cartesian
    displacement ``r_j_image - r_i`` to the bonded image of the
    neighbor. With ``r_j_image = r_j_home + S @ cell``, inverting gives
    ``S = (distance_vector + r_i - r_j) @ inv(cell)`` — the integer
    image shift falls out as a single linear solve per endpoint.
    Translating the edge's frame so src lands at the origin then gives
    ``dst_offset = S_dst - S_src``.
    """
    nbrs = topology.neighbor_list[edge_slot]
    if len(nbrs) != 2:
        raise ValueError(
            f"Topology edge slot {edge_slot} has {len(nbrs)} neighbors; expected 2"
        )
    n_src, n_dst = nbrs[0], nbrs[1]
    src_slot = int(n_src.index)
    dst_slot = int(n_dst.index)

    cell = np.asarray(topology.atoms.cell.array, dtype=np.float64)  # rows = a,b,c
    inv_cell = np.linalg.inv(cell)
    p_edge = topology.atoms.positions[edge_slot]
    p_src  = topology.atoms.positions[src_slot]
    p_dst  = topology.atoms.positions[dst_slot]

    def exact_image(neighbor, vertex_position) -> np.ndarray:
        image_float = (
            neighbor.distance_vector + p_edge - vertex_position
        ) @ inv_cell
        image = np.rint(image_float).astype(np.int32)
        residual = float(np.max(np.abs(image_float - image), initial=0.0))
        # PORMAKE nudges fractional coordinates at cell boundaries by up to
        # 1e-4 during wrapping; allow that known displacement while remaining
        # far below the half-cell ambiguity threshold.
        if residual > 2e-3:
            raise ValueError(
                f"Topology edge slot {edge_slot} has a non-integral endpoint "
                f"image (maximum residual {residual:.6g})"
            )
        return image

    S_src = exact_image(n_src, p_src)
    S_dst = exact_image(n_dst, p_dst)
    return src_slot, dst_slot, (S_dst - S_src).astype(np.int32)


def _atom_provenance(
    framework,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return slot/local IDs, cell offsets, and canonical wrapped positions.

    PORMAKE constructs ``framework.atoms`` by concatenating ``located_bbs`` in
    topology-slot order, deleting connection-point ``X`` atoms without
    reordering the survivors, and finally wrapping a copy into the unit cell.
    Replaying only that documented assembly order gives each final atom its
    originating slot. The retained, unwrapped located-BB position defines a
    canonical wrapped position and integer lattice image directly; PORMAKE's
    final wrapped copy is used only to validate periodic equivalence.

    Validate every part of this contract so a future PORMAKE behavior change
    fails the row explicitly instead of silently corrupting its clustering.
    """
    atoms = framework.atoms
    located_bbs = framework.info["located_bbs"]

    slot_ids: list[int] = []
    local_atom_ids: list[int] = []
    symbols: list[str] = []
    unwrapped_positions: list[np.ndarray] = []
    for slot_id, bb in enumerate(located_bbs):
        if bb is None:
            continue
        for local_atom_id, atom in enumerate(bb.atoms):
            if atom.symbol == "X":
                continue
            slot_ids.append(slot_id)
            local_atom_ids.append(local_atom_id)
            symbols.append(atom.symbol)
            unwrapped_positions.append(np.asarray(atom.position, dtype=np.float64))

    n_atoms = len(atoms)
    if len(slot_ids) != n_atoms:
        raise ValueError(
            "PORMAKE provenance atom-count mismatch: "
            f"located_bbs produced {len(slot_ids)}, framework has {n_atoms}"
        )
    framework_symbols = list(atoms.get_chemical_symbols())
    if symbols != framework_symbols:
        mismatch = next(
            i for i, (expected, actual) in enumerate(zip(symbols, framework_symbols))
            if expected != actual
        )
        raise ValueError(
            "PORMAKE provenance atom-order mismatch at index "
            f"{mismatch}: located_bbs={symbols[mismatch]!r}, "
            f"framework={framework_symbols[mismatch]!r}"
        )

    cell = np.asarray(atoms.cell.array, dtype=np.float64)  # rows = a, b, c
    inv_cell = np.linalg.inv(cell)
    unwrapped = np.asarray(unwrapped_positions, dtype=np.float64)
    unwrapped_frac = unwrapped @ inv_cell
    nearest_integer = np.rint(unwrapped_frac)
    boundary = np.abs(unwrapped_frac - nearest_integer) < 1e-10
    unwrapped_frac = np.where(boundary, nearest_integer, unwrapped_frac)
    cell_offsets = np.floor(unwrapped_frac).astype(np.int32)
    wrapped_frac = unwrapped_frac - cell_offsets
    canonical_positions = wrapped_frac @ cell

    # PORMAKE's wrapped copy is validation-only; canonical output above does
    # not depend on Framework.wrap behavior.
    framework_positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    image_float = (unwrapped - framework_positions) @ inv_cell
    framework_images = np.rint(image_float).astype(np.int32)

    # Framework.wrap nudges coordinates close to cell boundaries by up to
    # 1e-4 fractional units, so permit that known numerical displacement while
    # rejecting any non-integer image ambiguity by a comfortable margin.
    residual = image_float - framework_images
    max_residual = float(np.max(np.abs(residual), initial=0.0))
    if max_residual > 2e-3:
        atom_idx, axis = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        raise ValueError(
            "PORMAKE provenance position mismatch: unwrapped and framework "
            "positions do not differ by an integer lattice image; "
            f"atom={atom_idx}, axis={axis}, residual={residual[atom_idx, axis]:.6g}"
        )

    return (
        np.asarray(slot_ids, dtype=np.int64),
        np.asarray(local_atom_ids, dtype=np.int64),
        cell_offsets,
        canonical_positions,
    )


def _validate_framework_connectivity(
    framework,
    atom_slots: np.ndarray,
    local_atom_ids: np.ndarray,
) -> None:
    """Require final construction bonds to realize topology slot incidences."""
    topology = framework.info["topology"]
    located_bbs = framework.info["located_bbs"]
    n_atoms = len(framework.atoms)
    if len(atom_slots) != n_atoms or len(local_atom_ids) != n_atoms:
        raise ValueError("Framework connectivity provenance length mismatch")

    provenance_to_final = {
        (int(slot), int(local)): atom_idx
        for atom_idx, (slot, local) in enumerate(zip(atom_slots, local_atom_ids))
    }
    if len(provenance_to_final) != n_atoms:
        raise ValueError("Duplicate (slot_id, local_atom_id) atom provenance")

    # Identify bonds already present inside the original building blocks.
    internal_bonds: Counter[tuple[int, int]] = Counter()
    for slot, bb in enumerate(located_bbs):
        if bb is None:
            continue
        symbols = list(bb.atoms.get_chemical_symbols())
        for local_i, local_j in bb.bonds:
            local_i, local_j = int(local_i), int(local_j)
            if symbols[local_i] == "X" or symbols[local_j] == "X":
                continue
            final_i = provenance_to_final[(slot, local_i)]
            final_j = provenance_to_final[(slot, local_j)]
            internal_bonds[tuple(sorted((final_i, final_j)))] += 1

    actual_bonds: Counter[tuple[int, int]] = Counter()
    for raw_i, raw_j in framework.bonds:
        i, j = int(raw_i), int(raw_j)
        if not (0 <= i < n_atoms and 0 <= j < n_atoms):
            raise ValueError(f"Framework bond index out of range: ({i}, {j})")
        actual_bonds[tuple(sorted((i, j)))] += 1

    missing_internal = internal_bonds - actual_bonds
    if missing_internal:
        raise ValueError(
            "Framework is missing retained intra-building-block bonds: "
            f"{dict(missing_internal)}"
        )
    external_bonds = actual_bonds - internal_bonds
    actual_slot_pairs: Counter[tuple[int, int]] = Counter()
    for (i, j), multiplicity in external_bonds.items():
        pair = tuple(sorted((int(atom_slots[i]), int(atom_slots[j]))))
        actual_slot_pairs[pair] += multiplicity

    occupied_slots = set(int(slot) for slot in atom_slots)
    expected_slot_pairs: Counter[tuple[int, int]] = Counter()
    for edge_raw in topology.edge_indices:
        edge = int(edge_raw)
        src, dst, _ = _topology_edge_endpoints(edge, topology)
        if edge in occupied_slots:
            expected_slot_pairs[tuple(sorted((edge, src)))] += 1
            expected_slot_pairs[tuple(sorted((edge, dst)))] += 1
        else:
            expected_slot_pairs[tuple(sorted((src, dst)))] += 1

    if actual_slot_pairs != expected_slot_pairs:
        missing = expected_slot_pairs - actual_slot_pairs
        unexpected = actual_slot_pairs - expected_slot_pairs
        raise ValueError(
            "Framework inter-slot bonds do not realize the topology; "
            f"missing={dict(missing)}, unexpected={dict(unexpected)}"
        )


def extract_clustering(framework) -> dict:
    """Extract the generic per-atom clustering schema from a PORMAKE Framework.

    Returns a dict with the following parallel arrays:
      - ``atom_cluster`` (length n_atoms): per-atom
        ``{"cluster_id": int, "cluster_offset": [int, int, int]}``
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
    n_atoms = len(atoms)

    node_slot_set = {int(s) for s in topology.node_indices}
    edge_slot_set = {int(s) for s in topology.edge_indices}
    n_clusters = topology.n_slots
    located_bbs = framework.info["located_bbs"]
    if len(located_bbs) != n_clusters:
        raise ValueError(
            "PORMAKE located_bbs/topology slot-count mismatch: "
            f"{len(located_bbs)} vs {n_clusters}"
        )
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

    atom_slot, local_atom_ids, per_atom_offsets, canonical_positions = (
        _atom_provenance(framework)
    )
    if not set(int(slot) for slot in atom_slot) <= expected_slots:
        raise ValueError("Atom provenance references a non-topology slot")
    _validate_framework_connectivity(framework, atom_slot, local_atom_ids)

    is_cluster_vertex = [c in node_slot_set for c in range(n_clusters)]
    cluster_endpoints: list[dict] = []
    for c in range(n_clusters):
        if c in edge_slot_set:
            # The topology is the authoritative source for endpoint images.
            # Inferring them from framework bonds cannot distinguish the two
            # neighbor images of a periodic self-edge (src == dst), and
            # atom-less edge slots have no framework atoms to inspect at all.
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
            cluster_endpoints.append({"src": -1, "dst": -1, "dst_offset": [0, 0, 0]})

    atom_cluster = [
        {
            "cluster_id": int(atom_slot[i]),
            "cluster_offset": [
                int(per_atom_offsets[i, 0]),
                int(per_atom_offsets[i, 1]),
                int(per_atom_offsets[i, 2]),
            ],
        }
        for i in range(n_atoms)
    ]

    atom_positions = [
        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
        for p in canonical_positions
    ]

    cell = np.asarray(atoms.cell.array, dtype=np.float64)  # rows are a, b, c
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


def _build_one(args: tuple) -> tuple[str, str, dict | None]:
    """Worker: build one MOF, return (name, status, clustering).

    Status is one of: ``ok``, ``skipped``, ``skipped:ambiguous_node_assignment``
    (the ~9 two-inorganic, same-CN MOFs we can't reproduce from the deposit —
    see the drop guard below), ``rmsd_warn:<value>``, ``failed:<reason>``.
    ``clustering`` is the dict from :func:`extract_clustering` on success
    (including ``rmsd_warn``), or ``None`` on failure / skip.
    """
    import pormake as pm  # imported per-worker to avoid fork-pickle weirdness
    import pormake.log  # noqa: F401 — importing configures the "unique_logger"
    # Silence pormake's console chatter (per-slot RMSD INFO lines) WITHOUT
    # calling pormake.log.disable_print(): that helper logs a ">>> Console logs
    # ... are disabled" WARNING on *every* call, and _build_one runs once per
    # MOF — i.e. thousands of spam lines. Raising the logger threshold drops the
    # noise silently, while still letting genuine ERROR-level messages through.
    logging.getLogger("unique_logger").setLevel(logging.ERROR)

    recipe, alias_map, inorganic_order, bb_dir_str, out_dir_str, skip_existing, rmsd_warn = args
    bb_dir = Path(bb_dir_str)
    out_path = Path(out_dir_str) / f"{recipe.name}.cif"
    if skip_existing and out_path.exists():
        return recipe.name, "skipped", None
    try:
        topo = pm.Database().get_topo(recipe.net)
    except Exception as exc:  # noqa: BLE001
        return recipe.name, f"failed:topology_lookup:{type(exc).__name__}:{exc}", None

    node_types = list(topo.unique_node_types)
    edge_types = [tuple(t) for t in topo.unique_edge_types]

    if len(recipe.node_aliases) != len(node_types):
        return recipe.name, (
            f"failed:node_count_mismatch:"
            f"recipe={len(recipe.node_aliases)},topo={len(node_types)}"
        ), None

    try:
        bbs: list = []
        for alias in recipe.node_aliases:
            xyz_name = alias_map.get(alias)
            if xyz_name is None:
                return recipe.name, f"failed:unknown_node_alias:{alias}", None
            xyz_path = bb_dir / f"{xyz_name}.xyz"
            if not xyz_path.exists():
                return recipe.name, f"failed:missing_xyz:{xyz_path.name}", None
            bbs.append(pm.BuildingBlock(str(xyz_path)))

        topo_cns = [int(c) for c in topo.unique_cn]
        bb_cns = [int(b.n_connection_points) for b in bbs]

        # Drop the ~9 MOFs whose node→site assignment we cannot reproduce from
        # the Ball 2026 deposit. We assign building blocks to topology sites by
        # matching connection counts (CN) — which is forced/unambiguous whenever
        # the two sites have *different* CNs, and is more robust than the
        # deposit's published notebook (its literal positional assignment raises
        # a shape error on those distinct-CN cases). The ONE case CN can't
        # disambiguate is two *inorganic* nodes on two same-CN sites: there the
        # deposit's true tiebreak is inorganic_node.csv row order, while we fall
        # back to the MOF name's node1/node2 order. When those two orders differ
        # we'd place the metal blocks on swapped sites vs. the dataset's ground
        # truth, and the deposited generation script isn't public to confirm
        # which is right — so we drop these rather than risk a wrong structure.
        # (When the orders agree, we already match the deposit, so we keep them.)
        if (
            len(recipe.node_aliases) == 2
            and topo_cns[0] == topo_cns[1]                          # same-CN sites: CN can't decide
            and all(a in inorganic_order for a in recipe.node_aliases)  # both inorganic nodes
            and [inorganic_order[a] for a in recipe.node_aliases]
                != sorted(inorganic_order[a] for a in recipe.node_aliases)  # name order ≠ CSV order
        ):
            return recipe.name, "skipped:ambiguous_node_assignment", None

        if bb_cns == topo_cns:
            ordered_bbs = bbs
        elif len(bbs) == 2 and bb_cns[::-1] == topo_cns:
            ordered_bbs = bbs[::-1]
        else:
            return recipe.name, (
                f"failed:connectivity_mismatch:"
                f"bb_cns={bb_cns},topo_cns={topo_cns}"
            ), None
        node_bbs: dict = {node_types[i]: ordered_bbs[i] for i in range(len(node_types))}

        edge_bbs: dict | None
        if recipe.edge_alias is None:
            edge_bbs = None
        else:
            edge_xyz = alias_map.get(recipe.edge_alias)
            if edge_xyz is None:
                return recipe.name, f"failed:unknown_edge_alias:{recipe.edge_alias}", None
            edge_xyz_path = bb_dir / f"{edge_xyz}.xyz"
            if not edge_xyz_path.exists():
                return recipe.name, f"failed:missing_xyz:{edge_xyz_path.name}", None
            edge_bb = pm.BuildingBlock(str(edge_xyz_path))
            edge_bbs = {e: edge_bb for e in edge_types}

        max_rmsd = 0.0
        try:
            locator = pm.Locator()
            for i, t in enumerate(node_types):
                rmsd = float(locator.calculate_rmsd(
                    topo.unique_local_structures[i], node_bbs[t]
                ))
                if rmsd > max_rmsd:
                    max_rmsd = rmsd
        except Exception:  # noqa: BLE001 — RMSD failure is non-fatal
            max_rmsd = float("nan")

        builder_kwargs = {"topology": topo, "node_bbs": node_bbs}
        if edge_bbs is not None:
            builder_kwargs["edge_bbs"] = edge_bbs
        framework = pm.Builder().build_by_type(**builder_kwargs)
        clustering = extract_clustering(framework)
        framework.write_cif(str(out_path))
    except Exception as exc:  # noqa: BLE001
        return recipe.name, f"failed:builder_exception:{type(exc).__name__}:{exc}", None

    if max_rmsd == max_rmsd and max_rmsd > rmsd_warn:  # max_rmsd != NaN
        return recipe.name, f"rmsd_warn:{max_rmsd:.3f}", clustering
    return recipe.name, "ok", clustering
