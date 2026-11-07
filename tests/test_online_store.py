"""
Unit Tests for OnlineFeatureStore.
Tests low-latency retrieval (<5ms), thread-safety under concurrent load,
TTL expiration, and batch entity operations.
"""

import concurrent.futures
import time
from datetime import datetime, timezone
import pytest

from app.models.entity import EntityRecord
from app.services.online_store import OnlineFeatureStore


class TestOnlineFeatureStore:
    """Suite testing performance, concurrency, and caching semantics of the Online Store."""

    def test_put_and_get_single_entity(self):
        """Verifies simple put and O(1) get operations."""
        store = OnlineFeatureStore()
        view_name = "user_features"
        now = datetime.now(timezone.utc)

        store.put(
            view_name=view_name,
            entity_id="user_1",
            timestamp=now,
            features={"score": 95.5, "country": "US"},
        )

        res = store.get(view_name, "user_1")
        assert res.found is True
        assert res.is_stale is False
        assert res.entity_id == "user_1"
        assert res.features["score"] == 95.5
        assert res.features["country"] == "US"

        # Missing entity lookup
        res_missing = store.get(view_name, "user_nonexistent")
        assert res_missing.found is False
        assert res_missing.features == {}

    def test_batch_put_and_batch_get_with_projection(self):
        """Verifies batch insertion and retrieval with column projections."""
        store = OnlineFeatureStore()
        view_name = "fraud_view"
        now = datetime.now(timezone.utc)

        records = [
            EntityRecord(
                entity_id=f"user_{i}",
                timestamp=now,
                features={"f1": i * 1.5, "f2": i * 10, "f3": f"tier_{i}"},
            )
            for i in range(100)
        ]
        store.batch_put(view_name, records)

        # Batch get with projection (f1 and f3 only)
        eids = [f"user_{i}" for i in range(10)] + ["user_ghost"]
        batch_res = store.batch_get(view_name, eids, feature_names=["f1", "f3"])

        assert len(batch_res) == 11
        assert batch_res["user_0"].found is True
        assert batch_res["user_0"].features == {"f1": 0.0, "f3": "tier_0"}
        assert "f2" not in batch_res["user_0"].features
        assert batch_res["user_ghost"].found is False

    def test_latency_benchmark_under_5ms(self):
        """
        Benchmarking test ensuring multi-entity retrieval executes in <5ms.
        In-memory target is typically <0.1ms.
        """
        store = OnlineFeatureStore()
        view_name = "benchmark_view"
        now = datetime.now(timezone.utc)

        # Pre-populate 1,000 entities
        records = [
            EntityRecord(
                entity_id=f"entity_{i}",
                timestamp=now,
                features={"weight": float(i), "active": True},
            )
            for i in range(1000)
        ]
        store.batch_put(view_name, records)

        query_ids = [f"entity_{i * 2}" for i in range(50)]  # 50 entities

        # Warm-up
        store.batch_get(view_name, query_ids)

        # Measure 100 retrieval iterations
        latencies_ms = []
        for _ in range(100):
            t0 = time.perf_counter()
            res = store.batch_get(view_name, query_ids)
            latency = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency)
            assert len(res) == 50

        avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
        p99_latency_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]

        print(f"\n[Online Store Latency Benchmark] 50-entity Batch: Avg={avg_latency_ms:.4f}ms, P99={p99_latency_ms:.4f}ms")
        assert avg_latency_ms < 5.0, f"Average latency {avg_latency_ms}ms exceeded 5ms limit!"
        assert p99_latency_ms < 5.0, f"P99 latency {p99_latency_ms}ms exceeded 5ms limit!"

    def test_ttl_expiration(self):
        """Tests entry expiration when TTL is exceeded."""
        store = OnlineFeatureStore()
        view_name = "ttl_view"
        now = datetime.now(timezone.utc)

        # Put with 1 second TTL
        store.put(
            view_name=view_name,
            entity_id="ephemeral_user",
            timestamp=now,
            features={"temp_token": "xyz123"},
            ttl_seconds=1,
        )

        # Immediate retrieval -> valid
        res_fresh = store.get(view_name, "ephemeral_user")
        assert res_fresh.found is True

        # Wait for TTL to expire
        time.sleep(1.1)

        # Expired retrieval -> stale / not found
        res_expired = store.get(view_name, "ephemeral_user")
        assert res_expired.found is False
        assert res_expired.is_stale is True

    def test_concurrent_multithreaded_read_write(self):
        """Verifies thread-safety under heavy concurrent multithreaded read/write traffic."""
        store = OnlineFeatureStore()
        view_name = "concurrency_view"
        now = datetime.now(timezone.utc)
        thread_count = 10
        ops_per_thread = 200

        def writer_task(thread_id: int):
            for i in range(ops_per_thread):
                eid = f"concurrent_entity_{i % 50}"
                store.put(
                    view_name=view_name,
                    entity_id=eid,
                    timestamp=now,
                    features={"thread": thread_id, "counter": i},
                )

        def reader_task(thread_id: int):
            for i in range(ops_per_thread):
                eids = [f"concurrent_entity_{j}" for j in range(10)]
                store.batch_get(view_name, eids)

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = []
            for t in range(thread_count // 2):
                futures.append(executor.submit(writer_task, t))
            for t in range(thread_count // 2, thread_count):
                futures.append(executor.submit(reader_task, t))

            for f in concurrent.futures.as_completed(futures):
                f.result()  # Should not raise any thread-safety exceptions

        metrics = store.get_metrics()
        assert metrics["total_writes"] == (thread_count // 2) * ops_per_thread
        assert metrics["total_reads"] > 0

