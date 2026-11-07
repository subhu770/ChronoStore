"""
Online Feature Store Service.
High-throughput, ultra-low latency (<5ms) in-memory key-value store with thread safety,
TTL expiration, and O(1) batch entity retrieval.
"""

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from app.core.logging import logger
from app.models.entity import EntityRecord, OnlineEntityFeatureValue


@dataclass(slots=True)
class OnlineStoreEntry:
    """Internal representation of a cached feature vector for an entity."""
    entity_id: str
    timestamp: datetime
    features: Dict[str, Any]
    ingested_at: float  # time.time() timestamp
    ttl_seconds: Optional[int] = None

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Determines if the entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        now = current_time if current_time is not None else time.time()
        return (now - self.ingested_at) > self.ttl_seconds


class OnlineFeatureStore:
    """
    Thread-safe in-memory key-value store for serving real-time feature vectors.
    Keyed by (view_name, entity_id).
    """

    def __init__(self, default_ttl_seconds: Optional[int] = None) -> None:
        self._default_ttl_seconds = default_ttl_seconds
        # Structure: {view_name: {entity_id: OnlineStoreEntry}}
        self._store: Dict[str, Dict[str, OnlineStoreEntry]] = {}
        self._lock = threading.RLock()

        # Operational Metrics
        self._total_reads: int = 0
        self._total_writes: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._total_read_time_ns: int = 0

    def put(
        self,
        view_name: str,
        entity_id: str,
        timestamp: datetime,
        features: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Inserts or updates the latest feature vector for an entity."""
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        entry = OnlineStoreEntry(
            entity_id=entity_id,
            timestamp=timestamp,
            features=features,
            ingested_at=time.time(),
            ttl_seconds=ttl,
        )

        with self._lock:
            if view_name not in self._store:
                self._store[view_name] = {}
            self._store[view_name][entity_id] = entry
            self._total_writes += 1

    def batch_put(
        self,
        view_name: str,
        records: Sequence[EntityRecord],
        ttl_seconds: Optional[int] = None,
    ) -> int:
        """
        Atomically ingests a batch of entity records into the online store.
        Returns the number of records ingested.
        """
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds

        with self._lock:
            if view_name not in self._store:
                self._store[view_name] = {}
            target_view = self._store[view_name]

            for record in records:
                target_view[record.entity_id] = OnlineStoreEntry(
                    entity_id=record.entity_id,
                    timestamp=record.timestamp,
                    features=record.features,
                    ingested_at=now,
                    ttl_seconds=ttl,
                )
            self._total_writes += len(records)

        return len(records)

    def get(
        self,
        view_name: str,
        entity_id: str,
        feature_names: Optional[Sequence[str]] = None,
    ) -> OnlineEntityFeatureValue:
        """
        Retrieves feature vector for a single entity in O(1) time.
        """
        start_ns = time.perf_counter_ns()
        now = time.time()

        with self._lock:
            self._total_reads += 1
            view_dict = self._store.get(view_name)
            entry = view_dict.get(entity_id) if view_dict else None

            if entry is None:
                self._misses += 1
                self._total_read_time_ns += time.perf_counter_ns() - start_ns
                return OnlineEntityFeatureValue(
                    entity_id=entity_id,
                    timestamp=None,
                    features={},
                    found=False,
                )

            # Check TTL expiration
            if entry.is_expired(now):
                self._misses += 1
                self._total_read_time_ns += time.perf_counter_ns() - start_ns
                return OnlineEntityFeatureValue(
                    entity_id=entity_id,
                    timestamp=entry.timestamp,
                    features={},
                    is_stale=True,
                    found=False,
                )

            self._hits += 1
            features = (
                {k: entry.features[k] for k in feature_names if k in entry.features}
                if feature_names is not None
                else dict(entry.features)
            )
            result = OnlineEntityFeatureValue(
                entity_id=entity_id,
                timestamp=entry.timestamp,
                features=features,
                found=True,
                is_stale=False,
            )

        self._total_read_time_ns += time.perf_counter_ns() - start_ns
        return result

    def batch_get(
        self,
        view_name: str,
        entity_ids: Sequence[str],
        feature_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, OnlineEntityFeatureValue]:
        """
        O(1) multi-entity batch retrieval with field projection and TTL checks.
        Optimized to release lock quickly.
        """
        start_ns = time.perf_counter_ns()
        now = time.time()
        results: Dict[str, OnlineEntityFeatureValue] = {}
        hits = 0
        misses = 0

        # Snapshot entries under lock
        with self._lock:
            self._total_reads += len(entity_ids)
            view_dict = self._store.get(view_name, {})
            entries = [view_dict.get(eid) for eid in entity_ids]

        # Field projection & packaging outside lock to minimize contention
        names_set = set(feature_names) if feature_names is not None else None

        for eid, entry in zip(entity_ids, entries):
            if entry is None:
                misses += 1
                results[eid] = OnlineEntityFeatureValue(
                    entity_id=eid,
                    timestamp=None,
                    features={},
                    found=False,
                )
            elif entry.is_expired(now):
                misses += 1
                results[eid] = OnlineEntityFeatureValue(
                    entity_id=eid,
                    timestamp=entry.timestamp,
                    features={},
                    is_stale=True,
                    found=False,
                )
            else:
                hits += 1
                if names_set is not None:
                    feat_subset = {k: v for k, v in entry.features.items() if k in names_set}
                else:
                    feat_subset = dict(entry.features)

                results[eid] = OnlineEntityFeatureValue(
                    entity_id=eid,
                    timestamp=entry.timestamp,
                    features=feat_subset,
                    found=True,
                    is_stale=False,
                )

        with self._lock:
            self._hits += hits
            self._misses += misses
            self._total_read_time_ns += time.perf_counter_ns() - start_ns

        return results

    def delete(self, view_name: str, entity_id: str) -> bool:
        """Removes an entity from the specified feature view."""
        with self._lock:
            if view_name in self._store and entity_id in self._store[view_name]:
                del self._store[view_name][entity_id]
                return True
            return False

    def clear(self) -> None:
        """Flushes all cached entries in the online store."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._total_reads = 0
            self._total_writes = 0
            self._total_read_time_ns = 0

    def get_metrics(self) -> Dict[str, Any]:
        """Returns real-time performance and operational metrics."""
        with self._lock:
            total_entries = sum(len(entities) for entities in self._store.values())
            avg_read_latency_us = (
                (self._total_read_time_ns / (self._total_reads * 1000.0))
                if self._total_reads > 0
                else 0.0
            )
            hit_ratio = (
                (self._hits / self._total_reads)
                if self._total_reads > 0
                else 0.0
            )

            views_summary = {v: len(entities) for v, entities in self._store.items()}

            return {
                "total_entries": total_entries,
                "views": views_summary,
                "total_reads": self._total_reads,
                "total_writes": self._total_writes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(hit_ratio, 4),
                "avg_read_latency_us": round(avg_read_latency_us, 2),
            }

