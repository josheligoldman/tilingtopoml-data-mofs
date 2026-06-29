"""Hydra CLI: end-to-end USMOF build → join → parquet → (optional) HF push.

For every unique ``name`` in the Ball et al. 2026 train+test feature CSVs:

  1. Build the MOF with PORMAKE (via ``build_usmof_cifs._build_one``) — this
     materializes ``<MOF_CIF_DIR>/<name>.cif`` AND captures a generic
     per-atom clustering in memory at the same time.
  2. Join the CIF text and the clustering columns onto the feature row.
  3. Write the result to ``MOF_CHEMFILE_DIR/train.parquet``.
  4. When ``MOF_HF_DATASET`` is set, upload the chemfile dir to that HF
     dataset repo under ``data/`` via ``HfApi.upload_large_folder``.

The parquet picks up five new columns alongside ``crystal_chemfile``:
``crystal_atom_cluster``, ``crystal_atom_positions``, ``crystal_unit_cell``,
``crystal_is_cluster_vertex``, ``crystal_cluster_endpoints``. Each is a JSON
string, not a nested list/struct column: Parquet2.jl (the Julia consumer)
cannot read nested parquet columns, so these are serialized the same way
``crystal_chemfile`` / ``tiling_periodic_complex`` already are. Rows whose
build failed get ``"[]"`` in all five (the sentinel for "no clustering,
consumer falls back to its default path").

Mirrors the staging-symlink + ``upload-large-folder`` pattern from
``~/AI4ChemS/CrystalTilingComplexes.jl/run.jl``.
"""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi
from omegaconf import DictConfig
from tqdm.auto import tqdm

from .build_usmof_cifs import (
    Recipe,
    _build_one,
    load_alias_map,
    load_inorganic_node_order,
    parse_name,
)

load_dotenv()

log = logging.getLogger(__name__)

SOURCE_NAME_COL = "name"            # column name in Ball et al. 2026 CSVs
NAME_COL = "crystal_name"           # renamed in the published dataset
CIF_COL = "crystal_chemfile"
SPLIT_COL = "split"
ATOM_CLUSTER_COL = "crystal_atom_cluster"
ATOM_POSITIONS_COL = "crystal_atom_positions"
UNIT_CELL_COL = "crystal_unit_cell"
IS_CLUSTER_VERTEX_COL = "crystal_is_cluster_vertex"
CLUSTER_ENDPOINTS_COL = "crystal_cluster_endpoints"
PARQUET_NAME = "train.parquet"   # `data/train.parquet` → HF infers a `train` split
STAGING_ROOT = Path("~/.cache/tilingtopoml-data/hf-upload-staging").expanduser()


def _stage_with_symlinks(chemfile_dir: Path, repo_id: str) -> Path:
    """Build a staging dir whose ``data/`` is populated with symlinks back to
    every file in ``chemfile_dir``. Mirrors ``cp -rs`` from the Julia example.
    """
    staging = STAGING_ROOT / repo_id.replace("/", "__")
    if staging.exists():
        shutil.rmtree(staging)
    data_dir = staging / "data"
    data_dir.mkdir(parents=True)
    for src in chemfile_dir.iterdir():
        if src.is_file():
            (data_dir / src.name).symlink_to(src.resolve())
    return staging


def _empty_clustering() -> dict:
    """Sentinel value for rows with no clustering — empty lists in all four
    fields. The downstream consumer detects this and falls back to its
    default clustering path (e.g. CrystalNets for `CrystalTilingComplexes.jl`).
    """
    return {
        ATOM_CLUSTER_COL: [],
        ATOM_POSITIONS_COL: [],
        UNIT_CELL_COL: [],
        IS_CLUSTER_VERTEX_COL: [],
        CLUSTER_ENDPOINTS_COL: [],
    }


def _build_all(
    names: list[str],
    construction_dir: Path,
    cifs_dir: Path,
    num_workers: int,
    skip_existing: bool,
    rmsd_warn_threshold: float,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Run ``_build_one`` for every name; return (status_by_name,
    clustering_by_name).

    ``status_by_name[name]`` is the per-build status string (``ok``,
    ``skipped``, ``rmsd_warn:...``, ``failed:...``). ``clustering_by_name``
    only contains successful builds (skipped builds re-read clustering from
    the existing CIF — wait, that's not possible; see note below).

    NOTE: when ``skip_existing=True`` and a CIF already exists, we skip the
    build entirely AND therefore have no clustering for that row. Those rows
    get empty-list clustering. Pass ``skip_existing=false`` to guarantee
    clustering for every successfully built MOF.
    """
    alias_map = load_alias_map(construction_dir)
    inorganic_order = load_inorganic_node_order(construction_dir)
    bb_dir = construction_dir / "USMOF_building_block"
    cifs_dir.mkdir(parents=True, exist_ok=True)

    recipes: list[Recipe] = []
    parse_failures: dict[str, str] = {}
    for name in names:
        try:
            recipes.append(parse_name(str(name)))
        except ValueError as exc:
            parse_failures[name] = f"failed:parse:{exc}"

    payloads = [
        (r, alias_map, inorganic_order, str(bb_dir), str(cifs_dir),
         skip_existing, rmsd_warn_threshold)
        for r in recipes
    ]
    log.info(
        "Building %d MOFs (workers=%d, skip_existing=%s)...",
        len(payloads), num_workers, skip_existing,
    )

    statuses: dict[str, str] = dict(parse_failures)
    clusterings: dict[str, dict] = {}
    if num_workers <= 1:
        for p in tqdm(payloads, desc="building USMOF CIFs", unit="MOF"):
            name, status, clustering = _build_one(p)
            statuses[name] = status
            if clustering is not None:
                clusterings[name] = clustering
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_build_one, p) for p in payloads]
            for fut in tqdm(
                as_completed(futures), total=len(futures),
                desc="building USMOF CIFs", unit="MOF",
            ):
                name, status, clustering = fut.result()
                statuses[name] = status
                if clustering is not None:
                    clusterings[name] = clustering

    tally: dict[str, int] = {}
    for s in statuses.values():
        bucket = s.split(":", 1)[0]
        tally[bucket] = tally.get(bucket, 0) + 1
    log.info("Build complete. Status counts: %s", tally)

    status_path = cifs_dir / "build_status.csv"
    pd.DataFrame(
        sorted(statuses.items()), columns=["name", "status"]
    ).to_csv(status_path, index=False)
    log.info("Wrote per-name build status to %s.", status_path)

    return statuses, clusterings


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    cifs_dir = Path(cfg.cifs_dir).expanduser()
    construction_dir = Path(cfg.construction_dir).expanduser()
    train_csv = Path(cfg.train_csv).expanduser()
    test_csv = Path(cfg.test_csv).expanduser()
    chemfile_dir = Path(cfg.chemfile_dir).expanduser()
    num_workers = int(cfg.get("num_workers", 8))
    skip_existing = bool(cfg.get("skip_existing", False))
    rmsd_warn_threshold = float(cfg.get("rmsd_warn_threshold", 0.3))

    train = pd.read_csv(train_csv).rename(columns={SOURCE_NAME_COL: NAME_COL})
    test = pd.read_csv(test_csv).rename(columns={SOURCE_NAME_COL: NAME_COL})
    if list(train.columns) != list(test.columns):
        raise SystemExit(
            "train/test CSV column lists differ; cannot concatenate. "
            f"train: {len(train.columns)} cols, test: {len(test.columns)} cols. "
            f"diff (train-only): {set(train.columns) - set(test.columns)} ; "
            f"diff (test-only): {set(test.columns) - set(train.columns)}"
        )
    train = train.assign(**{SPLIT_COL: "train"})
    test = test.assign(**{SPLIT_COL: "test"})
    df = pd.concat([train, test], ignore_index=True)
    log.info(
        "Loaded %d train + %d test rows (combined %d, %d columns).",
        len(train), len(test), len(df), len(df.columns),
    )

    unique_names = df[NAME_COL].astype(str).drop_duplicates().tolist()
    log.info("Building %d unique MOFs.", len(unique_names))

    statuses, clusterings = _build_all(
        unique_names, construction_dir, cifs_dir,
        num_workers=num_workers,
        skip_existing=skip_existing,
        rmsd_warn_threshold=rmsd_warn_threshold,
    )

    cifs_by_stem: dict[str, Path] = {p.stem: p for p in cifs_dir.glob("*.cif")}
    log.info("Found %d CIF files in %s after build.", len(cifs_by_stem), cifs_dir)

    matched_rows: list[int] = []
    cif_texts: list[str] = []
    atom_clusters: list[list] = []
    atom_positions: list[list] = []
    unit_cells: list[list] = []
    is_cluster_vertex: list[list] = []
    cluster_endpoints: list[list] = []
    n_clustering_present = 0
    for i, name in enumerate(df[NAME_COL].astype(str)):
        path = cifs_by_stem.get(name)
        if path is None:
            continue
        matched_rows.append(i)
        cif_texts.append(path.read_text())
        clustering = clusterings.get(name)
        if clustering is None:
            atom_clusters.append([])
            atom_positions.append([])
            unit_cells.append([])
            is_cluster_vertex.append([])
            cluster_endpoints.append([])
        else:
            atom_clusters.append(clustering["atom_cluster"])
            atom_positions.append(clustering["atom_positions"])
            unit_cells.append(clustering["unit_cell"])
            is_cluster_vertex.append(clustering["is_cluster_vertex"])
            cluster_endpoints.append(clustering["cluster_endpoints"])
            n_clustering_present += 1

    n_unmatched = len(df) - len(matched_rows)
    n_orphan = len(cifs_by_stem) - len({df[NAME_COL].astype(str).iloc[i] for i in matched_rows})
    log.info(
        "Matched %d / %d CSV rows; dropped %d unmatched rows and %d orphan CIFs. "
        "Clustering present on %d / %d kept rows.",
        len(matched_rows), len(df), n_unmatched, n_orphan,
        n_clustering_present, len(matched_rows),
    )

    kept_df = df.iloc[matched_rows].reset_index(drop=True).copy()
    kept_df[CIF_COL] = cif_texts
    # Serialize the clustering columns as JSON strings rather than nested
    # list/struct columns: Parquet2.jl (the Julia consumer) cannot read nested
    # parquet columns. Empty sentinel ([]) serializes to "[]".
    kept_df[ATOM_CLUSTER_COL] = [json.dumps(x) for x in atom_clusters]
    kept_df[ATOM_POSITIONS_COL] = [json.dumps(x) for x in atom_positions]
    kept_df[UNIT_CELL_COL] = [json.dumps(x) for x in unit_cells]
    kept_df[IS_CLUSTER_VERTEX_COL] = [json.dumps(x) for x in is_cluster_vertex]
    kept_df[CLUSTER_ENDPOINTS_COL] = [json.dumps(x) for x in cluster_endpoints]

    # Write the parquet directly via pandas/pyarrow rather than routing through
    # datasets.Dataset.from_pandas(...).to_parquet(). All columns are now simple
    # types (the clustering columns are JSON strings above), so the HF `datasets`
    # roundtrip buys nothing — and its multiprocessing parquet writer was
    # deadlocking after the build (runs hung for >1h with stray worker procs).
    # pandas.to_parquet is single-process, deterministic, and lower-memory.
    chemfile_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = chemfile_dir / PARQUET_NAME
    log.info("Writing parquet: %d rows, %d columns.", len(kept_df), len(kept_df.columns))
    kept_df.to_parquet(str(parquet_path), engine="pyarrow", index=False)
    log.info("Saved local parquet to %s (%.1f MB).",
             parquet_path, parquet_path.stat().st_size / 1e6)

    hf_dataset = cfg.get("hf_dataset")
    if not hf_dataset:
        log.info("hf_dataset / MOF_HF_DATASET not set; skipping upload.")
        return

    if bool(cfg.get("dry_run", False)):
        log.info("[dry-run] would upload %s/ -> %s data/ via upload_large_folder.",
                 chemfile_dir, hf_dataset)
        return

    staging = _stage_with_symlinks(chemfile_dir, hf_dataset)
    log.info(
        "Uploading %s -> %s (via HfApi.upload_large_folder)...",
        staging, hf_dataset,
    )
    HfApi().upload_large_folder(
        repo_id=hf_dataset,
        folder_path=str(staging),
        repo_type="dataset",
        num_workers=24,
    )
    shutil.rmtree(staging, ignore_errors=True)
    log.info("Uploaded.")
    log.info("URL: https://huggingface.co/datasets/%s", hf_dataset)


if __name__ == "__main__":
    main()
