"""
Unit Tests for ParquetCompactor and DriftDetector services.
"""

from datetime import datetime, timezone
import numpy as np
import pytest

from app.models.entity import EntityRecord
from app.services.compactor import ParquetCompactor
from app.services.drift_detector import DriftDetector, calculate_categorical_psi, calculate_psi
from app.services.offline_store import OfflineFeatureStore


def test_parquet_compactor_flow(temp_storage_dir):
    offline_store = OfflineFeatureStore(base_dir=temp_storage_dir)
    compactor = ParquetCompactor(offline_store=offline_store)
    view_name = "compaction_test"
    base_t = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Write 3 separate batches producing 3 distinct parquet chunk files
    offline_store.append_records(view_name, [EntityRecord(entity_id="u1", timestamp=base_t, features={"x": 1.0})])
    offline_store.append_records(view_name, [EntityRecord(entity_id="u2", timestamp=base_t, features={"x": 2.0})])
    offline_store.append_records(view_name, [EntityRecord(entity_id="u1", timestamp=base_t, features={"x": 1.5})])  # duplicate update

    stats_before = offline_store.get_view_stats(view_name)
    assert stats_before["parquet_file_count"] == 3

    # Run compaction
    result = compactor.compact_view(view_name, min_files_to_compact=2)
    assert result is not None
    assert result.files_before == 3
    assert result.files_after == 1
    assert result.records_compacted >= 2

    # Verify that offline store reads compacted data seamlessly
    df = offline_store.read_historical_data(view_name)
    assert len(df) == 2  # u1 deduplicated to latest 1.5 and u2 is 2.0
    u1_row = df[df["entity_id"] == "u1"].iloc[0]
    assert u1_row["x"] == 1.5

    # Run compact_all
    summary = compactor.compact_all(min_files_to_compact=1)
    assert view_name in summary


def test_drift_detector_numeric_and_categorical():
    detector = DriftDetector()

    # Stable numeric distribution
    ref_nums = np.random.normal(loc=50, scale=10, size=500)
    curr_nums = np.random.normal(loc=50.2, scale=9.8, size=500)
    res_stable = detector.analyze_feature("score", ref_nums, curr_nums)
    assert res_stable["drift_detected"] is False
    assert res_stable["severity"] == "NONE"

    # Drifted numeric distribution (mean shifted from 50 to 80)
    drifted_nums = np.random.normal(loc=80, scale=10, size=500)
    res_drifted = detector.analyze_feature("score", ref_nums, drifted_nums)
    assert res_drifted["drift_detected"] is True
    assert res_drifted["severity"] in {"MODERATE", "CRITICAL"}
    assert res_drifted["psi_score"] > 0.1

    # Categorical drift
    ref_cats = ["A"] * 400 + ["B"] * 100
    curr_cats = ["A"] * 100 + ["B"] * 400
    res_cat = detector.analyze_feature("tier", ref_cats, curr_cats)
    assert res_cat["drift_detected"] is True
    assert res_cat["data_type"] == "categorical"

    # Dataset-level drift
    import pandas as pd
    ref_df = pd.DataFrame({"score": ref_nums, "tier": ref_cats})
    curr_df = pd.DataFrame({"score": drifted_nums, "tier": curr_cats})
    dataset_res = detector.analyze_dataset(ref_df, curr_df)
    assert dataset_res["dataset_drift_detected"] is True
    assert dataset_res["features_drifted"] >= 1
