"""
Unit Tests for OfflineFeatureStore.
Tests append-only Parquet persistence, chronological querying, entity filtering, and stats.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import pytest

from app.models.entity import EntityRecord
from app.services.offline_store import OfflineFeatureStore


class TestOfflineFeatureStore:
    """Suite testing Parquet persistence and temporal query operations."""

    def test_append_and_read_multiple_batches(self, temp_storage_dir: Path):
        """Verifies multi-batch append and unified reading across generated Parquet chunks."""
        store = OfflineFeatureStore(base_dir=temp_storage_dir)
        view_name = "transactions"
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Batch 1
        batch1 = [
            EntityRecord(
                entity_id="cust_1",
                timestamp=base_time,
                features={"amount": 50.0, "currency": "USD"},
            ),
            EntityRecord(
                entity_id="cust_2",
                timestamp=base_time + timedelta(minutes=5),
                features={"amount": 75.0, "currency": "EUR"},
            ),
        ]
        store.append_records(view_name, batch1)

        # Batch 2 (appended later)
        batch2 = [
            EntityRecord(
                entity_id="cust_1",
                timestamp=base_time + timedelta(minutes=10),
                features={"amount": 120.0, "currency": "USD"},
            ),
        ]
        store.append_records(view_name, batch2)

        # Read all historical data
        df = store.read_historical_data(view_name)
        assert len(df) == 3
        assert list(df["entity_id"]) == ["cust_1", "cust_2", "cust_1"]
        assert list(df["amount"]) == [50.0, 75.0, 120.0]

    def test_entity_and_time_filtering(self, temp_storage_dir: Path):
        """Verifies precise filtering on entity IDs and time boundaries."""
        store = OfflineFeatureStore(base_dir=temp_storage_dir)
        view_name = "metrics_view"
        t0 = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

        records = [
            EntityRecord(entity_id="srv_A", timestamp=t0, features={"cpu": 10.0}),
            EntityRecord(entity_id="srv_B", timestamp=t0 + timedelta(hours=1), features={"cpu": 20.0}),
            EntityRecord(entity_id="srv_A", timestamp=t0 + timedelta(hours=2), features={"cpu": 30.0}),
            EntityRecord(entity_id="srv_B", timestamp=t0 + timedelta(hours=3), features={"cpu": 40.0}),
        ]
        store.append_records(view_name, records)

        # Filter by entity
        df_a = store.read_historical_data(view_name, entity_ids=["srv_A"])
        assert len(df_a) == 2
        assert set(df_a["entity_id"]) == {"srv_A"}

        # Filter by time window (hours 1 to 2)
        df_window = store.read_historical_data(
            view_name,
            start_time=t0 + timedelta(hours=1),
            end_time=t0 + timedelta(hours=2),
        )
        assert len(df_window) == 2
        assert list(df_window["cpu"]) == [20.0, 30.0]

    def test_view_stats_and_listing(self, temp_storage_dir: Path):
        """Verifies storage metadata, file counts, and view discovery."""
        store = OfflineFeatureStore(base_dir=temp_storage_dir)
        t = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        store.append_records("view_one", [EntityRecord(entity_id="e1", timestamp=t, features={"x": 1})])
        store.append_records("view_two", [EntityRecord(entity_id="e2", timestamp=t, features={"y": 2})])

        views = store.list_views()
        assert "view_one" in views
        assert "view_two" in views

        stats_one = store.get_view_stats("view_one")
        assert stats_one["parquet_file_count"] >= 1
        assert stats_one["total_records"] == 1
        assert stats_one["total_size_bytes"] > 0

