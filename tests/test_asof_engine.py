"""
Unit and Integration Tests for AsofJoinEngine.
Tests exact temporal boundaries, 1-second lookahead prevention, multi-entity isolation,
lookback staleness, and strict leakage assertion.
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from app.models.entity import EntityRecord
from app.models.join import ObservationRecord
from app.services.asof_engine import AsofJoinEngine, DataLeakageDetectedError
from app.services.offline_store import OfflineFeatureStore


class TestAsofJoinEngine:
    """Suite testing point-in-time correctness and leakage prevention."""

    def test_exact_temporal_boundary_and_no_lookahead(self, temp_storage_dir):
        """
        Explicitly verify that if an entity feature updated at t=10:05,
        an inference observation at t=10:04 retrieves the t<10:04 value, never the t=10:05 value.
        Also verifies the exact 1-second boundary at t=10:04:59.
        """
        offline_store = OfflineFeatureStore(base_dir=temp_storage_dir)
        engine = AsofJoinEngine(offline_store=offline_store)

        view_name = "account_risk"
        t_10_00 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        t_10_05 = datetime(2026, 1, 1, 10, 5, 0, tzinfo=timezone.utc)

        # Ingest historical updates for entity 'user_alpha'
        events = [
            EntityRecord(
                entity_id="user_alpha",
                timestamp=t_10_00,
                features={"risk_score": 0.12, "tier": "SILVER"},
            ),
            EntityRecord(
                entity_id="user_alpha",
                timestamp=t_10_05,
                features={"risk_score": 0.85, "tier": "GOLD"},
            ),
        ]
        offline_store.append_records(view_name, events)

        # Test Observation 1: t=10:04:00 (1 minute before update)
        t_10_04 = datetime(2026, 1, 1, 10, 4, 0, tzinfo=timezone.utc)
        engine.verify_temporal_boundary(
            view_name=view_name,
            entity_id="user_alpha",
            t_past=t_10_00,
            t_future=t_10_05,
            t_observation=t_10_04,
            past_value=0.12,
            future_value=0.85,
            feature_name="risk_score",
        )

        # Test Observation 2: t=10:04:59 (1-second boundary before update)
        t_10_04_59 = datetime(2026, 1, 1, 10, 4, 59, tzinfo=timezone.utc)
        obs_1s_before = [ObservationRecord(entity_id="user_alpha", timestamp=t_10_04_59)]
        res_1s_before = engine.join_observations(obs_1s_before, [view_name])
        assert res_1s_before.iloc[0]["risk_score"] == 0.12
        assert res_1s_before.iloc[0]["tier"] == "SILVER"

        # Test Observation 3: t=10:05:00 exact match (allow_exact_matches=True)
        obs_exact = [ObservationRecord(entity_id="user_alpha", timestamp=t_10_05)]
        res_exact = engine.join_observations(obs_exact, [view_name], allow_exact_matches=True)
        assert res_exact.iloc[0]["risk_score"] == 0.85
        assert res_exact.iloc[0]["tier"] == "GOLD"

        # Test Observation 4: t=10:05:00 with allow_exact_matches=False (strictly before)
        res_strict_before = engine.join_observations(obs_exact, [view_name], allow_exact_matches=False)
        assert res_strict_before.iloc[0]["risk_score"] == 0.12
        assert res_strict_before.iloc[0]["tier"] == "SILVER"

        # Test Observation 5: t=10:05:01 (1 second after update)
        t_10_05_01 = datetime(2026, 1, 1, 10, 5, 1, tzinfo=timezone.utc)
        obs_after = [ObservationRecord(entity_id="user_alpha", timestamp=t_10_05_01)]
        res_after = engine.join_observations(obs_after, [view_name])
        assert res_after.iloc[0]["risk_score"] == 0.85
        assert res_after.iloc[0]["tier"] == "GOLD"

    def test_lookback_staleness_window(self, temp_storage_dir):
        """Verifies that features older than lookback_seconds evaluate to NaN."""
        offline_store = OfflineFeatureStore(base_dir=temp_storage_dir)
        engine = AsofJoinEngine(offline_store=offline_store)

        view_name = "sensor_telemetry"
        t_09_00 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        offline_store.append_records(
            view_name,
            [
                EntityRecord(
                    entity_id="sensor_1",
                    timestamp=t_09_00,
                    features={"temperature": 72.5},
                )
            ],
        )

        # Observation at 09:30:00 (30 minutes = 1800s later)
        t_09_30 = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        obs = [ObservationRecord(entity_id="sensor_1", timestamp=t_09_30)]

        # Join with 10-minute (600s) tolerance -> should evaluate to NaN (too stale)
        res_stale = engine.join_observations(obs, [view_name], lookback_seconds=600)
        assert pd.isna(res_stale.iloc[0]["temperature"])

        # Join with 45-minute (2700s) tolerance -> should successfully match
        res_fresh = engine.join_observations(obs, [view_name], lookback_seconds=2700)
        assert res_fresh.iloc[0]["temperature"] == 72.5

    def test_multi_entity_isolation(self, temp_storage_dir):
        """Ensures entity timelines are completely isolated with zero cross-entity contamination."""
        offline_store = OfflineFeatureStore(base_dir=temp_storage_dir)
        engine = AsofJoinEngine(offline_store=offline_store)

        view_name = "customer_balance"
        t1 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 10, 10, 0, tzinfo=timezone.utc)

        events = [
            EntityRecord(entity_id="user_X", timestamp=t1, features={"balance": 100.0}),
            EntityRecord(entity_id="user_Y", timestamp=t1, features={"balance": 500.0}),
            EntityRecord(entity_id="user_X", timestamp=t2, features={"balance": 250.0}),
        ]
        offline_store.append_records(view_name, events)

        # Observation at 10:05:00 for both users
        t_obs = datetime(2026, 1, 1, 10, 5, 0, tzinfo=timezone.utc)
        observations = [
            ObservationRecord(entity_id="user_X", timestamp=t_obs),
            ObservationRecord(entity_id="user_Y", timestamp=t_obs),
        ]

        result = engine.join_observations(observations, [view_name])
        x_row = result[result["entity_id"] == "user_X"].iloc[0]
        y_row = result[result["entity_id"] == "user_Y"].iloc[0]

        assert x_row["balance"] == 100.0
        assert y_row["balance"] == 500.0

    def test_multi_view_join(self, temp_storage_dir):
        """Verifies simultaneous leak-free joining across multiple independent feature views."""
        offline_store = OfflineFeatureStore(base_dir=temp_storage_dir)
        engine = AsofJoinEngine(offline_store=offline_store)

        view_profile = "user_profile"
        view_activity = "user_activity"

        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 10, 5, 0, tzinfo=timezone.utc)

        offline_store.append_records(
            view_profile,
            [EntityRecord(entity_id="user_1", timestamp=t0, features={"age": 30, "segment": "VIP"})],
        )
        offline_store.append_records(
            view_activity,
            [EntityRecord(entity_id="user_1", timestamp=t1, features={"clicks_last_1h": 14})],
        )

        # Observation at 10:06:00
        t_obs = datetime(2026, 1, 1, 10, 6, 0, tzinfo=timezone.utc)
        obs = [ObservationRecord(entity_id="user_1", timestamp=t_obs, label=1)]

        result = engine.join_observations(
            observations=obs,
            feature_views=[view_profile, view_activity],
        )

        row = result.iloc[0]
        assert row["age"] == 30
        assert row["segment"] == "VIP"
        assert row["clicks_last_1h"] == 14
        assert row["label"] == 1

    def test_preserves_input_observation_order(self, temp_storage_dir):
        """Ensures observations passed in arbitrary non-chronological order maintain original row order."""
        offline_store = OfflineFeatureStore(base_dir=temp_storage_dir)
        engine = AsofJoinEngine(offline_store=offline_store)

        view_name = "test_view"
        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        offline_store.append_records(
            view_name,
            [EntityRecord(entity_id="u1", timestamp=t0, features={"val": 10})],
        )

        t_late = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t_early = datetime(2026, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
        t_mid = datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

        # Observations passed in non-chronological order
        observations = [
            ObservationRecord(entity_id="u1", timestamp=t_late, metadata={"tag": "third"}),
            ObservationRecord(entity_id="u1", timestamp=t_early, metadata={"tag": "first"}),
            ObservationRecord(entity_id="u1", timestamp=t_mid, metadata={"tag": "second"}),
        ]

        res = engine.join_observations(observations, [view_name])
        assert res.iloc[0]["tag"] == "third"
        assert res.iloc[1]["tag"] == "first"
        assert res.iloc[2]["tag"] == "second"

    def test_observation_prior_to_any_features_returns_nan(self, temp_storage_dir):
        """Verifies that an observation prior to the first feature update returns NaN without error."""
        offline_store = OfflineFeatureStore(base_dir=temp_storage_dir)
        engine = AsofJoinEngine(offline_store=offline_store)

        view_name = "test_view"
        t_first = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        offline_store.append_records(
            view_name,
            [EntityRecord(entity_id="u1", timestamp=t_first, features={"val": 42})],
        )

        t_before = datetime(2026, 1, 1, 9, 59, 59, tzinfo=timezone.utc)
        obs = [ObservationRecord(entity_id="u1", timestamp=t_before)]

        res = engine.join_observations(obs, [view_name])
        assert pd.isna(res.iloc[0]["val"])

    def test_strict_leakage_assertion_raises_on_synthetic_leak(self, temp_storage_dir):
        """Tests that strict_leakage_check catches any artificial future timestamp leakage."""
        offline_store = OfflineFeatureStore(base_dir=temp_storage_dir)
        engine = AsofJoinEngine(offline_store=offline_store)

        # Create a DataFrame simulating an erroneous merge where feat_ts > obs_ts
        corrupt_df = pd.DataFrame({
            "entity_id": ["user_leak"],
            "timestamp": pd.to_datetime(["2026-01-01 10:00:00"], utc=True),
            "_audit_ts_mock": pd.to_datetime(["2026-01-01 10:05:00"], utc=True),  # 5 minutes ahead!
            "val": [999],
        })

        with pytest.raises(DataLeakageDetectedError, match="Critical Data Leakage Detected"):
            engine._verify_leakage(
                merged_df=corrupt_df,
                obs_ts_col="timestamp",
                feat_ts_col="_audit_ts_mock",
                view_name="mock",
                allow_exact=True,
            )

