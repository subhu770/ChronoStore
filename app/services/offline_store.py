"""
Offline Feature Store Service.
Append-only immutable log of historical entity observations supporting Parquet temporal persistence.
Enables leak-free training dataset generation and point-in-time backtesting.
"""

import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.core.config import settings
from app.core.logging import logger
from app.models.entity import EntityRecord


class OfflineFeatureStore:
    """
    Thread-safe append-only Parquet storage engine for historical feature logs.
    Organizes datasets by view: {storage_dir}/{view_name}/{timestamp_uuid}.parquet.
    """

    def __init__(self, base_dir: Optional[Path] = None, compression: str = "snappy") -> None:
        self._base_dir = Path(base_dir or settings.get_offline_dir())
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._compression = compression
        self._lock = threading.RLock()

    def _get_view_dir(self, view_name: str) -> Path:
        """Returns the directory for a specific feature view, creating it if needed."""
        view_dir = self._base_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        return view_dir

    def append_records(self, view_name: str, records: Sequence[EntityRecord]) -> int:
        """
        Appends a batch of entity records into an immutable, timestamped Parquet file.
        Thread-safe and ACID within the append file boundary.
        """
        if not records:
            return 0

        # Transform records into tabular structure
        rows: List[Dict[str, Any]] = []
        for rec in records:
            # Flatten entity_id, timestamp, and all features into a single row
            row: Dict[str, Any] = {
                "entity_id": str(rec.entity_id),
                "timestamp": rec.timestamp,
            }
            row.update(rec.features)
            rows.append(row)

        df = pd.DataFrame(rows)
        # Ensure timestamp is standardized as datetime64[ns, UTC]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        view_dir = self._get_view_dir(view_name)
        timestamp_ms = int(time.time() * 1000)
        file_uuid = uuid.uuid4().hex[:8]
        file_name = f"part-{timestamp_ms}-{file_uuid}.parquet"
        file_path = view_dir / file_name

        with self._lock:
            # Write via PyArrow with compression
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, file_path, compression=self._compression)
            logger.debug(
                "Appended %d records for view '%s' to %s",
                len(records),
                view_name,
                file_name,
            )

        return len(records)

    def read_historical_data(
        self,
        view_name: str,
        entity_ids: Optional[Sequence[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Reads historical records for a feature view with optional entity, temporal,
        and column projection filters. Returns a DataFrame sorted by timestamp.
        """
        view_dir = self._get_view_dir(view_name)
        parquet_files = list(view_dir.glob("*.parquet"))

        if not parquet_files:
            # Return empty DataFrame with baseline schema
            cols = ["entity_id", "timestamp"]
            if feature_names:
                cols.extend(feature_names)
            empty_df = pd.DataFrame(columns=cols)
            empty_df["timestamp"] = pd.to_datetime(empty_df["timestamp"], utc=True)
            return empty_df

        with self._lock:
            try:
                # Read all parquet files in view directory
                df = pd.read_parquet(view_dir)
            except Exception as e:
                logger.error("Failed to read Parquet data for view '%s': %s", view_name, e)
                raise

        if df.empty:
            return df

        # Ensure timestamp is UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["entity_id"] = df["entity_id"].astype(str)

        # Entity filtering
        if entity_ids is not None:
            entity_set = set(str(e) for e in entity_ids)
            df = df[df["entity_id"].isin(entity_set)]

        # Temporal filtering
        if start_time is not None:
            st = pd.Timestamp(start_time)
            if st.tzinfo is None:
                st = st.tz_localize("UTC")
            else:
                st = st.tz_convert("UTC")
            df = df[df["timestamp"] >= st]

        if end_time is not None:
            et = pd.Timestamp(end_time)
            if et.tzinfo is None:
                et = et.tz_localize("UTC")
            else:
                et = et.tz_convert("UTC")
            df = df[df["timestamp"] <= et]

        # Column projection
        if feature_names is not None:
            requested_cols = ["entity_id", "timestamp"] + [
                col for col in feature_names if col in df.columns
            ]
            df = df[requested_cols]

        # Deterministic sorting: chronological order
        df = df.sort_values(by=["timestamp", "entity_id"]).reset_index(drop=True)
        return df

    def get_view_stats(self, view_name: str) -> Dict[str, Any]:
        """Calculates storage footprint and record count for a given feature view."""
        view_dir = self._get_view_dir(view_name)
        parquet_files = list(view_dir.glob("*.parquet"))
        total_bytes = sum(f.stat().st_size for f in parquet_files)

        total_records = 0
        if parquet_files:
            try:
                df = pd.read_parquet(view_dir, columns=["entity_id"])
                total_records = len(df)
            except Exception:
                pass

        return {
            "view_name": view_name,
            "parquet_file_count": len(parquet_files),
            "total_size_bytes": total_bytes,
            "total_records": total_records,
        }

    def list_views(self) -> List[str]:
        """Lists all feature views present in offline storage."""
        if not self._base_dir.exists():
            return []
        return [
            d.name
            for d in self._base_dir.iterdir()
            if d.is_dir() and list(d.glob("*.parquet"))
        ]

