"""
Offline Parquet Compaction Engine.
Solves the streaming small-file problem in the append-only Offline Store
by consolidating fragmented Parquet partitions into optimized, sorted,
and deduplicated Parquet files.
"""

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.core.config import settings
from app.core.logging import logger
from app.services.offline_store import OfflineFeatureStore


class CompactionResult:
    """Summary of compaction execution metrics."""
    def __init__(
        self,
        view_name: str,
        files_before: int,
        files_after: int,
        bytes_before: int,
        bytes_after: int,
        records_compacted: int,
        duration_ms: float,
    ) -> None:
        self.view_name = view_name
        self.files_before = files_before
        self.files_after = files_after
        self.bytes_before = bytes_before
        self.bytes_after = bytes_after
        self.records_compacted = records_compacted
        self.duration_ms = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        compression_ratio = (
            round((1.0 - (self.bytes_after / max(self.bytes_before, 1))) * 100, 2)
            if self.bytes_before > 0
            else 0.0
        )
        return {
            "view_name": self.view_name,
            "files_before": self.files_before,
            "files_after": self.files_after,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "compression_ratio_pct": compression_ratio,
            "records_compacted": self.records_compacted,
            "duration_ms": round(self.duration_ms, 2),
        }


class ParquetCompactor:
    """
    Consolidates streaming Parquet micro-batches into clean, sorted,
    chronologically ordered Parquet tables.
    """

    def __init__(self, offline_store: OfflineFeatureStore) -> None:
        self._offline_store = offline_store
        self._lock = threading.RLock()

    def compact_view(
        self,
        view_name: str,
        min_files_to_compact: int = 2,
    ) -> Optional[CompactionResult]:
        """
        Compacts all Parquet chunk files for a given view into a single optimized file.
        Deduplicates identical (entity_id, timestamp) rows and re-indexes chronologically.
        """
        start_time = time.perf_counter()
        view_dir = self._offline_store._get_view_dir(view_name)

        with self._lock:
            # Collect all current parquet files (excluding temporary files)
            parquet_files = [f for f in view_dir.glob("*.parquet") if not f.name.endswith(".tmp")]

            if len(parquet_files) < min_files_to_compact:
                logger.info(
                    "Compaction skipped for view '%s': %d files present (threshold: %d).",
                    view_name,
                    len(parquet_files),
                    min_files_to_compact,
                )
                return None

            bytes_before = sum(f.stat().st_size for f in parquet_files)

            # Read all existing records
            try:
                df = pd.read_parquet(view_dir)
            except Exception as e:
                logger.error("Failed to read Parquet files for compaction in view '%s': %s", view_name, e)
                raise

            if df.empty:
                return None

            records_count = len(df)
            df["entity_id"] = df["entity_id"].astype(str)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # Sort chronologically and deduplicate keeping latest
            df = df.sort_values(by=["timestamp", "entity_id"]).reset_index(drop=True)
            df = df.drop_duplicates(subset=["entity_id", "timestamp"], keep="last")

            # Write temporary compacted file
            ts_ms = int(time.time() * 1000)
            compact_id = uuid.uuid4().hex[:8]
            temp_file = view_dir / f"compacted-{ts_ms}-{compact_id}.parquet.tmp"
            final_file = view_dir / f"compacted-{ts_ms}-{compact_id}.parquet"

            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, temp_file, compression=self._offline_store._compression)

            # Atomically replace old files
            # Rename temp file to final
            temp_file.replace(final_file)

            # Remove previously compacted small files
            for old_file in parquet_files:
                if old_file != final_file and old_file.exists():
                    try:
                        old_file.unlink()
                    except OSError as e:
                        logger.warning("Could not delete legacy file %s: %s", old_file, e)

            bytes_after = final_file.stat().st_size
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            result = CompactionResult(
                view_name=view_name,
                files_before=len(parquet_files),
                files_after=1,
                bytes_before=bytes_before,
                bytes_after=bytes_after,
                records_compacted=records_count,
                duration_ms=elapsed_ms,
            )

            logger.info(
                "Compacted view '%s': %d files -> 1 file (%d bytes -> %d bytes) in %.2f ms",
                view_name,
                result.files_before,
                result.bytes_before,
                result.bytes_after,
                elapsed_ms,
            )

            return result

    def compact_all(self, min_files_to_compact: int = 2) -> Dict[str, Dict[str, Any]]:
        """Executes compaction across all active views in the offline store."""
        views = self._offline_store.list_views()
        results = {}
        for view in views:
            res = self.compact_view(view, min_files_to_compact=min_files_to_compact)
            if res:
                results[view] = res.to_dict()
        return results
