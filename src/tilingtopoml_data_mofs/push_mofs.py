"""Hydra CLI: end-to-end USMOF build → join → parquet.

For every unique ``name`` in the Ball et al. 2026 train+test feature CSVs:

  1. Build the MOF with PORMAKE (via ``build_usmof_cifs._build_one``) —
     captures a generic per-atom clustering in memory at the same time.
  2. Join the CIF text and the clustering columns onto the feature row.
  3. Write the result to ``MOF_CHEMFILE_DIR/train.parquet``.

The parquet picks up five new columns alongside ``crystal_chemfile``:
``crystal_atom_cluster``, ``crystal_atom_positions``, ``crystal_unit_cell``,
``crystal_is_cluster_vertex``, ``crystal_cluster_endpoints``. Each is a JSON
string, not a nested list/struct column: Parquet2.jl (the Julia consumer)
cannot read nested parquet columns, so these are serialized the same way
``crystal_chemfile`` / ``tiling_periodic_complex`` already are. Rows whose
build failed get ``"[]"`` in all five (the sentinel for "no clustering,
consumer falls back to its default path").
"""

from __future__ import annotations

import json
import logging
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm.auto import tqdm

from .build_usmof_cifs import (
    Recipe,
    _build_one,
    load_bb_key_map,
    load_inorganic_node_order,
    parse_mof_recipe,
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
PARQUET_NAME = "train.parquet"
IN_PROGRESS_DB_NAME = "train-in-progress.db"   # SQLite checkpoint
WRITE_BUFFER_SIZE = 200                         # rows per SQLite transaction


def _db_to_parquet(db_path: Path, parquet_path: Path) -> int:
    """Stream a SQLite checkpoint to parquet without loading it all into memory.

    Reads in chunks of WRITE_BUFFER_SIZE rows so peak memory is one chunk.
    Returns the total number of rows written.
    """
    conn = sqlite3.connect(db_path)
    n_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in pd.read_sql("SELECT * FROM mofs", conn, chunksize=WRITE_BUFFER_SIZE):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(parquet_path), table.schema)
            writer.write_table(table)
            n_rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
        conn.close()
    return n_rows


def _build_all(
    names: list[str],
    construction_dir: Path,
    num_workers: int,
    rmsd_warn_threshold: float,
) -> Iterator[tuple[str, str, str | None, dict | None]]:
    """Yield ``(name, status, cif_text, clustering)`` as each MOF build completes.

    Parse failures are yielded first (before any worker is spawned) so the
    caller can record them immediately.  The generator does not tally statuses
    or write ``build_status.csv`` — the caller owns those.
    """
    bb_key_map = load_bb_key_map(construction_dir)
    inorganic_order = load_inorganic_node_order(construction_dir)
    bb_dir = construction_dir / "USMOF_building_block"

    recipes: list[Recipe] = []
    for name in names:
        try:
            recipes.append(parse_mof_recipe(str(name)))
        except ValueError as exc:
            yield name, f"failed:parse:{exc}", None, None

    payloads = [
        (r, bb_key_map, inorganic_order, str(bb_dir), rmsd_warn_threshold)
        for r in recipes
    ]
    log.info("Building %d MOFs (workers=%d)...", len(payloads), num_workers)

    if num_workers <= 1:
        for p in tqdm(payloads, desc="building USMOF CIFs", unit="MOF"):
            yield _build_one(p)
    else:
        # Submit in bounded batches: at most batch_size futures live at once so
        # completed results (CIF text + clustering) don't pile up in the result
        # queue while the main process is busy writing to disk.
        batch_size = num_workers * 4
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            with tqdm(total=len(payloads), desc="building USMOF CIFs", unit="MOF") as pbar:
                for start in range(0, len(payloads), batch_size):
                    batch = payloads[start : start + batch_size]
                    futures = [ex.submit(_build_one, p) for p in batch]
                    for fut in as_completed(futures):
                        yield fut.result()
                        pbar.update(1)


@hydra.main(version_base=None, config_path="conf", config_name="mofs")
def main(cfg: DictConfig) -> None:
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
    train.insert(len(train.columns), SPLIT_COL, "train")
    test.insert(len(test.columns), SPLIT_COL, "test")
    df = pd.concat([train, test], ignore_index=True).copy()
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
    db_path = chemfile_dir / IN_PROGRESS_DB_NAME

    # Resume: read only the name column from SQLite — no CIF text loaded.
    # Fresh run: delete any leftover checkpoint from a prior run.
    if resume and db_path.exists():
        with sqlite3.connect(db_path) as conn:
            done_names = {
                row[0] for row in conn.execute(f'SELECT "{NAME_COL}" FROM mofs')
            }
        log.info(
            "Resuming: %d / %d names already in %s.",
            len(done_names), len(unique_names), db_path.name,
        )
    else:
        done_names = set()
        if db_path.exists():
            db_path.unlink()
            log.warning("resume=false: deleted existing %s.", db_path.name)

    names_to_build = [n for n in unique_names if n not in done_names]
    log.info(
        "Building %d unique MOFs (%d already done).",
        len(names_to_build), len(done_names),
    )

    statuses: dict[str, str] = {}
    buffer: list[dict] = []
    n_clustering_present = 0

    for name, status, cif_text, clustering in _build_all(
        names_to_build, construction_dir,
        num_workers=num_workers,
        rmsd_warn_threshold=rmsd_warn_threshold,
    ):
        statuses[name] = status
        if cif_text is None:
            continue

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
            with sqlite3.connect(db_path) as conn:
                pd.DataFrame(buffer).to_sql("mofs", conn, if_exists="append", index=False)
            log.info("Flushed %d rows to %s.", len(buffer), db_path.name)
            buffer.clear()

    # Flush any remaining rows.
    if buffer:
        with sqlite3.connect(db_path) as conn:
            pd.DataFrame(buffer).to_sql("mofs", conn, if_exists="append", index=False)
        buffer.clear()

    tally: dict[str, int] = {}
    for s in statuses.values():
        bucket = s.split(":", 1)[0]
        tally[bucket] = tally.get(bucket, 0) + 1
    log.info("Build complete. Status counts: %s", tally)

    status_path = chemfile_dir / "build_status.csv"
    pd.DataFrame(
        sorted(statuses.items()), columns=["name", "status"]
    ).to_csv(status_path, index=False)
    log.info("Wrote per-name build status to %s.", status_path)

    if not db_path.exists():
        log.warning("No rows were written; skipping parquet conversion.")
        return

    parquet_path = chemfile_dir / PARQUET_NAME
    n_total = _db_to_parquet(db_path, parquet_path)
    db_path.unlink()
    log.info(
        "Wrote %s (%d rows, %.1f MB). Clustering present on %d / %d new rows.",
        parquet_path.name, n_total, parquet_path.stat().st_size / 1e6,
        n_clustering_present, len(names_to_build),
    )


if __name__ == "__main__":
    main()
