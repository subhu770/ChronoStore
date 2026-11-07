"""
FastAPI REST API Contract Tests for ChronoStore.
Tests all endpoints including registration, online ingestion/retrieval,
historical joins, and system telemetry.
"""

from datetime import datetime, timezone
from fastapi.testclient import TestClient


class TestApiRoutes:
    """End-to-end API test suite using FastAPI TestClient."""

    def test_health_and_metrics_endpoints(self, client: TestClient):
        """Verifies liveness, health check, and metrics endpoints."""
        res_health = client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data["status"] == "healthy"
        assert health_data["service"] == "ChronoStore"
        assert "subsystems" in health_data

        res_metrics = client.get("/metrics")
        assert res_metrics.status_code == 200
        metrics_data = res_metrics.json()
        assert "online_store" in metrics_data
        assert "offline_store" in metrics_data

    def test_feature_view_lifecycle(self, client: TestClient):
        """Tests registering, listing, and retrieving feature view metadata."""
        payload = {
            "name": "merchant_risk_profile",
            "entity_id_key": "merchant_id",
            "description": "Risk profile and dispute rate features",
            "ttl_seconds": 7200,
            "features": [
                {"name": "dispute_rate", "data_type": "float", "description": "Chargeback ratio"},
                {"name": "is_high_risk", "data_type": "bool", "description": "Flag for enhanced review"},
            ],
        }

        # Create
        res_post = client.post("/api/v1/features/views", json=payload)
        assert res_post.status_code == 201
        data = res_post.json()
        assert data["name"] == "merchant_risk_profile"
        assert set(data["feature_names"]) == {"dispute_rate", "is_high_risk"}

        # Get by name
        res_get = client.get("/api/v1/features/views/merchant_risk_profile")
        assert res_get.status_code == 200
        assert res_get.json()["ttl_seconds"] == 7200

        # List views
        res_list = client.get("/api/v1/features/views")
        assert res_list.status_code == 200
        names = [v["name"] for v in res_list.json()]
        assert "merchant_risk_profile" in names

        # 404 for missing view
        res_missing = client.get("/api/v1/features/views/unknown_view_123")
        assert res_missing.status_code == 404

    def test_online_ingestion_and_retrieval(self, client: TestClient):
        """Tests POST /api/v1/features/online and POST /api/v1/features/retrieve."""
        view_name = "user_kyc"
        # Register view
        client.post(
            "/api/v1/features/views",
            json={
                "name": view_name,
                "features": [
                    {"name": "is_verified", "data_type": "bool"},
                    {"name": "risk_tier", "data_type": "string"},
                ],
            },
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        ingest_payload = {
            "view_name": view_name,
            "sync_to_offline": True,
            "records": [
                {
                    "entity_id": "usr_99",
                    "timestamp": now_iso,
                    "features": {"is_verified": True, "risk_tier": "LOW"},
                },
                {
                    "entity_id": "usr_100",
                    "timestamp": now_iso,
                    "features": {"is_verified": False, "risk_tier": "HIGH"},
                },
            ],
        }

        # Ingest
        res_ingest = client.post("/api/v1/features/online", json=ingest_payload)
        assert res_ingest.status_code == 200
        ingest_data = res_ingest.json()
        assert ingest_data["ingested_count"] == 2
        assert ingest_data["offline_persisted"] is True
        assert ingest_data["duration_ms"] > 0

        # Retrieve
        retrieval_payload = {
            "view_name": view_name,
            "entity_ids": ["usr_99", "usr_100", "usr_absent"],
            "feature_names": ["is_verified", "risk_tier"],
        }
        res_ret = client.post("/api/v1/features/retrieve", json=retrieval_payload)
        assert res_ret.status_code == 200
        ret_data = res_ret.json()
        assert ret_data["hit_count"] == 2
        assert ret_data["miss_count"] == 1
        assert ret_data["results"]["usr_99"]["features"]["is_verified"] is True
        assert ret_data["results"]["usr_100"]["features"]["risk_tier"] == "HIGH"
        assert ret_data["results"]["usr_absent"]["found"] is False

    def test_historical_join_endpoint(self, client: TestClient):
        """Tests POST /api/v1/features/historical-join API contract."""
        view_name = "order_metrics"
        # Ingest offline records
        client.post(
            "/api/v1/features/views",
            json={
                "name": view_name,
                "features": [{"name": "total_spend", "data_type": "float"}],
            },
        )

        client.post(
            "/api/v1/features/online",
            json={
                "view_name": view_name,
                "sync_to_offline": True,
                "records": [
                    {
                        "entity_id": "cust_A",
                        "timestamp": "2026-01-01T10:00:00Z",
                        "features": {"total_spend": 25.0},
                    },
                    {
                        "entity_id": "cust_A",
                        "timestamp": "2026-01-01T10:10:00Z",
                        "features": {"total_spend": 75.0},
                    },
                ],
            },
        )

        # Historical join at 10:05:00
        join_payload = {
            "observations": [
                {
                    "entity_id": "cust_A",
                    "timestamp": "2026-01-01T10:05:00Z",
                    "label": 0,
                    "metadata": {"order_id": "ord_1"},
                }
            ],
            "feature_views": [view_name],
            "strict_leakage_check": True,
        }

        res_join = client.post("/api/v1/features/historical-join", json=join_payload)
        assert res_join.status_code == 200
        data = res_join.json()
        assert data["row_count"] == 1
        row = data["joined_features"][0]
        assert row["entity_id"] == "cust_A"
        assert row["total_spend"] == 25.0  # Must be 25.0, not future 75.0!
        assert row["label"] == 0
        assert row["order_id"] == "ord_1"
        assert data["statistics"]["leakage_check_passed"] is True

