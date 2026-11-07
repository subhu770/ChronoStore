"""
Additional Edge Case Tests for Online Store, Offline Store, ASOF Engine, and API.
"""

from datetime import datetime, timezone
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.models.entity import EntityRecord
from app.models.join import ObservationRecord
from app.services.asof_engine import AsofJoinEngine
from app.services.offline_store import OfflineFeatureStore
from app.services.online_store import OnlineFeatureStore
from app.services.registry import FeatureRegistry


def test_online_store_delete_and_clear():
    store = OnlineFeatureStore()
    now = datetime.now(timezone.utc)
    store.put("v1", "e1", now, {"a": 1})
    assert store.get("v1", "e1").found is True

    # Test delete existing
    assert store.delete("v1", "e1") is True
    assert store.get("v1", "e1").found is False
    # Test delete non-existent
    assert store.delete("v1", "e1") is False

    # Test clear
    store.put("v1", "e2", now, {"a": 2})
    store.clear()
    assert store.get("v1", "e2").found is False
    metrics = store.get_metrics()
    assert metrics["total_entries"] == 0


def test_offline_store_edge_cases(temp_storage_dir):
    store = OfflineFeatureStore(base_dir=temp_storage_dir)
    # Empty append
    assert store.append_records("empty_view", []) == 0

    # Read from non-existent view
    df = store.read_historical_data("non_existent_view")
    assert df.empty
    assert "entity_id" in df.columns
    assert "timestamp" in df.columns

    # Non-existent base dir list
    store_non_existent = OfflineFeatureStore(base_dir=temp_storage_dir / "does_not_exist")
    assert store_non_existent.list_views() == []


def test_asof_engine_edge_cases(temp_storage_dir):
    store = OfflineFeatureStore(base_dir=temp_storage_dir)
    engine = AsofJoinEngine(offline_store=store)

    # Empty observations DataFrame
    empty_res = engine.join_observations(pd.DataFrame(columns=["entity_id", "timestamp"]), ["view1"])
    assert empty_res.empty

    # Invalid observation types
    with pytest.raises(TypeError):
        engine.join_observations(12345, ["view1"])

    with pytest.raises(TypeError):
        engine.join_observations([12345], ["view1"])

    with pytest.raises(ValueError, match="must contain both 'entity_id' and 'timestamp'"):
        engine.join_observations([{"not_entity": "foo"}], ["view1"])


def test_api_root_and_edge_cases(client: TestClient):
    # Test root landing page
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["service"] == "ChronoStore"

