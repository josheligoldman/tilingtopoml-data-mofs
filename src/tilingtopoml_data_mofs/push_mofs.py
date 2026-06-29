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
from typing import Iterator

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
IN_PROGRESS_PARQUET_NAME = PARQUET_NAME.removesuffix(".parquet") + "-in-progress.parquet"
WRITE_BUFFER_SIZE = 200          # rows buffered before flushing to disk
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


def _load_in_progress(path: Path) -> set[str]:
    """Return the set of MOF names already written to the in-progress parquet.

    Reads only the name column so we don't load CIF text or clustering data
    into memory just to find out what's done.  Returns an empty set when no
    in-progress file exists.
    """
    if not path.exists():
        return set()
    return set(
        pd.read_parquet(path, columns=[NAME_COL])[NAME_COL].astype(str)
    )


def _flush_to_disk(new_rows: list[dict], parquet_path: Path) -> int:
    """Atomically append ``new_rows`` to the in-progress parquet on disk.

    Reads the existing file (if any) fresh from disk each time rather than
    keeping an in-memory copy, so peak memory is one flush-buffer worth of
    new rows plus whatever pandas needs to read and write the file.  Writes to
    a sibling ``.tmp`` file then renames it over ``parquet_path`` — the rename
    is atomic on POSIX, so the on-disk file is always a valid complete parquet
    and a crash between flushes leaves the previous flush intact.  Returns the
    total number of rows now on disk.
    """
    new_df = pd.DataFrame(new_rows)
    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    tmp = parquet_path.with_suffix(".tmp.parquet")
    combined.to_parquet(str(tmp), engine="pyarrow", index=False)
    tmp.rename(parquet_path)
    return len(combined)


def _build_all(
    names: list[str],
    construction_dir: Path,
    cifs_dir: Path,
    num_workers: int,
    rmsd_warn_threshold: float,
) -> Iterator[tuple[str, str, dict | None]]:
    """Yield ``(name, status, clustering)`` as each MOF build completes.

    Parse failures are yielded first (before any worker is spawned) so the
    caller can record them immediately.  The generator does not tally statuses
    or write ``build_status.csv`` — the caller owns those.
    """
    alias_map = load_alias_map(construction_dir)
    inorganic_order = load_inorganic_node_order(construction_dir)
    bb_dir = construction_dir / "USMOF_building_block"
    cifs_dir.mkdir(parents=True, exist_ok=True)

    recipes: list[Recipe] = []
    for name in names:
        try:
            recipes.append(parse_name(str(name)))
        except ValueError as exc:
            yield name, f"failed:parse:{exc}", None

    payloads = [
        (r, alias_map, inorganic_order, str(bb_dir), str(cifs_dir),
         rmsd_warn_threshold)
        for r in recipes
    ]
    log.info("Building %d MOFs (workers=%d)...", len(payloads), num_workers)

    if num_workers <= 1:
        for p in tqdm(payloads, desc="building USMOF CIFs", unit="MOF"):
            yield _build_one(p)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_build_one, p) for p in payloads]
            for fut in tqdm(
                as_completed(futures), total=len(futures),
                desc="building USMOF CIFs", unit="MOF",
            ):
                yield fut.result()


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    cifs_dir = Path(cfg.cifs_dir).expanduser()
    construction_dir = Path(cfg.construction_dir).expanduser()
    train_csv = Path(cfg.train_csv).expanduser()
    test_csv = Path(cfg.test_csv).expanduser()
    chemfile_dir = Path(cfg.chemfile_dir).expanduser()
    num_workers = int(cfg.get("num_workers", 8))
    resume = bool(cfg.get("resume", False))
    rmsd_warn_threshold = float(cfg.get("rmsd_warn_threshold", 0.3))

    train = pd.read_csv(train_csv).rename(columns={SOURCE_NAME_COL: NAME_COL})
    test = pd.read_csv(test_csv).rename(columns={SOURCE_NAME_COL: NAME_COL})
    if set(train.columns) != set(test.columns):
        raise SystemExit(
            "train/test CSV column sets differ; cannot concatenate. "
            f"train-only: {set(train.columns) - set(test.columns)} ; "
            f"test-only: {set(test.columns) - set(train.columns)}"
        )
    train = train.assign(**{SPLIT_COL: "train"})
    test = test.assign(**{SPLIT_COL: "test"})
    df = pd.concat([train, test], ignore_index=True)
    log.info(
        "Loaded %d train + %d test rows (combined %d, %d columns).",
        len(train), len(test), len(df), len(df.columns),
    )

    unique_names = df[NAME_COL].astype(str).drop_duplicates().tolist()

    # Precompute name → CSV row indices for O(1) joins as builds complete.
    name_to_rows: dict[str, list[int]] = {}
    for i, name in enumerate(df[NAME_COL].astype(str)):
        name_to_rows.setdefault(name, []).append(i)

    chemfile_dir.mkdir(parents=True, exist_ok=True)
    in_progress_path = chemfile_dir / IN_PROGRESS_PARQUET_NAME

    # Load existing progress when resuming; ignore it when starting fresh.
    if resume:
        done_names = _load_in_progress(in_progress_path)
        if done_names:
            log.info(
                "Resuming: %d / %d names already in %s.",
                len(done_names), len(unique_names), in_progress_path.name,
            )
    else:
        done_names = set()
        if in_progress_path.exists():
            log.warning(
                "resume=false: ignoring existing %s and rebuilding from scratch.",
                in_progress_path.name,
            )

    names_to_build = [n for n in unique_names if n not in done_names]
    log.info(
        "Building %d unique MOFs (%d already done).",
        len(names_to_build), len(done_names),
    )

    statuses: dict[str, str] = {}
    buffer: list[dict] = []
    n_clustering_present = 0
    n_on_disk = len(done_names)  # rows already written in a previous run

    for name, status, clustering in _build_all(
        names_to_build, construction_dir, cifs_dir,
        num_workers=num_workers,
        rmsd_warn_threshold=rmsd_warn_threshold,
    ):
        statuses[name] = status
        cif_path = cifs_dir / f"{name}.cif"
        if not cif_path.exists():
            continue
        cif_text = cif_path.read_text()
        # Serialize clustering as JSON strings: Parquet2.jl (the Julia
        # consumer) cannot read nested parquet columns, so we match the
        # same wire format used by crystal_chemfile. Empty list = sentinel
        # for "no clustering; consumer falls back to its default path."
        if clustering is not None:
            cluster_cols = {
                ATOM_CLUSTER_COL:      json.dumps(clustering["atom_cluster"]),
                ATOM_POSITIONS_COL:    json.dumps(clustering["atom_positions"]),
                UNIT_CELL_COL:         json.dumps(clustering["unit_cell"]),
                IS_CLUSTER_VERTEX_COL: json.dumps(clustering["is_cluster_vertex"]),
                CLUSTER_ENDPOINTS_COL: json.dumps(clustering["cluster_endpoints"]),
            }
            n_clustering_present += 1
        else:
            cluster_cols = {k: "[]" for k in (
                ATOM_CLUSTER_COL, ATOM_POSITIONS_COL, UNIT_CELL_COL,
                IS_CLUSTER_VERTEX_COL, CLUSTER_ENDPOINTS_COL,
            )}
        for row_idx in name_to_rows.get(name, []):
            buffer.append({**df.iloc[row_idx].to_dict(), CIF_COL: cif_text, **cluster_cols})

        if len(buffer) >= WRITE_BUFFER_SIZE:
            n_on_disk = _flush_to_disk(buffer, in_progress_path)
            log.info(
                "Flushed %d rows to %s (total on disk: %d).",
                len(buffer), in_progress_path.name, n_on_disk,
            )
            buffer.clear()

    if buffer:
        n_on_disk = _flush_to_disk(buffer, in_progress_path)
        buffer.clear()

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

    if n_on_disk == 0:
        log.warning("No rows were written; skipping parquet rename.")
        return

    n_dropped = len(df) - n_on_disk
    log.info(
        "Kept %d / %d CSV rows (dropped %d with no built CIF). "
        "Clustering present on %d / %d new rows.",
        n_on_disk, len(df), n_dropped, n_clustering_present, len(names_to_build),
    )

    # Rename the in-progress file to the final output name now that the build
    # is complete. Using to_parquet directly (not datasets.Dataset.from_pandas)
    # avoids the multiprocessing parquet writer that was deadlocking after the
    # build (runs hung for >1h with stray worker procs).
    parquet_path = chemfile_dir / PARQUET_NAME
    in_progress_path.rename(parquet_path)
    log.info(
        "Renamed %s → %s (%.1f MB).",
        in_progress_path.name, parquet_path.name,
        parquet_path.stat().st_size / 1e6,
    )

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
