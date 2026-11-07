"""
Integration Tests for Training-Serving Parity and Skew Prevention.
Verifies that features logged online match the historical offline ASOF reconstruction
with 100% fidelity, eliminating training-serving skew.
"""

from datetime import datetime, timezone
from app.models.entity import EntityRecord
from app.models.join import ObservationRecord
from app.services.registry import FeatureRegistry


class TestSkewAndParity:
    """Verifies dual-layer consistency between Online and Offline stores."""

    def test_training_serving_exact_parity(self, clean_registry: FeatureRegistry):
        """
        Validates that for any entity at timestamp T_latest:
        OnlineStore.get(entity) == AsofJoinEngine.join_observations(entity, T_latest).
        """
        view_name = "user_risk_profile"
        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 12, 30, 0, tzinfo=timezone.utc)
        t_latest = datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        # Sequence of updates over time
        history = [
            EntityRecord(
                entity_id="entity_parity_1",
                timestamp=t0,
                features={"score": 500, "velocity": 1.2, "status": "TIER_1"},
            ),
            EntityRecord(
                entity_id="entity_parity_1",
                timestamp=t1,
                features={"score": 550, "velocity": 2.5, "status": "TIER_2"},
            ),
            EntityRecord(
                entity_id="entity_parity_1",
                timestamp=t_latest,
                features={"score": 720, "velocity": 5.8, "status": "TIER_PREMIUM"},
            ),
        ]

        # Ingest updates into both stores
        clean_registry.online_store.batch_put(view_name, history)
        clean_registry.offline_store.append_records(view_name, history)

        # 1. Fetch from Online Store (representing real-time model inference at t_latest)
        online_result = clean_registry.online_store.get(view_name, "entity_parity_1")
        assert online_result.found is True
        online_features = online_result.features

        # 2. Reconstruct state using Offline ASOF Join (representing training dataset generation)
        offline_enriched = clean_registry.asof_engine.join_observations(
            observations=[ObservationRecord(entity_id="entity_parity_1", timestamp=t_latest)],
            feature_views=[view_name],
            allow_exact_matches=True,
            strict_leakage_check=True,
        )

        offline_row = offline_enriched.iloc[0]

        # 3. Assert Exact Parity across all feature columns
        for feat_name, online_val in online_features.items():
            offline_val = offline_row[feat_name]
            assert online_val == offline_val, (
                f"Training-serving skew detected for feature '{feat_name}'! "
                f"Online: {online_val} != Offline: {offline_val}"
            )

        assert offline_row["score"] == 720
        assert offline_row["velocity"] == 5.8
        assert offline_row["status"] == "TIER_PREMIUM"

