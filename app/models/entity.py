"""
Pydantic v2 schemas for Entity Records, Online Feature Ingestion, and Retrieval.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


def ensure_utc(dt: datetime) -> datetime:
    """Normalizes any datetime to a timezone-aware UTC datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class EntityRecord(BaseModel):
    """
    Represents an atomic feature observation for an entity at an exact point in time.
    """
    entity_id: str = Field(..., description="Unique entity identifier (e.g. user_123, merchant_456)")
    timestamp: datetime = Field(
        ...,
        description="Event timestamp when feature values became effective (ISO 8601 UTC)"
    )
    features: Dict[str, Any] = Field(..., description="Key-value mapping of feature names to values")
    view_name: Optional[str] = Field(default=None, description="Optional associated feature view")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_and_validate_timestamp(cls, v: Any) -> datetime:
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return ensure_utc(dt)
        elif isinstance(v, datetime):
            return ensure_utc(v)
        raise ValueError(f"Invalid timestamp value: {v}")

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("entity_id cannot be blank")
        return s


class OnlineIngestRequest(BaseModel):
    """Payload for real-time feature updates into the Online Store."""
    view_name: str = Field(..., description="Feature view to ingest into")
    records: List[EntityRecord] = Field(..., min_length=1, description="List of entity feature updates")
    sync_to_offline: bool = Field(
        default=True,
        description="Dual-write flag: also append updates to the offline historical log"
    )


class OnlineIngestResponse(BaseModel):
    """Confirmation response following feature ingestion."""
    view_name: str
    ingested_count: int
    offline_persisted: bool
    duration_ms: float


class OnlineRetrievalRequest(BaseModel):
    """Payload for ultra-low latency real-time feature vector retrieval."""
    view_name: str = Field(..., description="Target feature view")
    entity_ids: List[str] = Field(..., min_length=1, description="List of entity keys to fetch")
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Optional projection list. If omitted, all features are returned."
    )


class OnlineEntityFeatureValue(BaseModel):
    """Feature vector response for a single entity."""
    entity_id: str
    timestamp: Optional[datetime] = None
    features: Dict[str, Any] = Field(default_factory=dict)
    is_stale: bool = False
    found: bool = True


class OnlineRetrievalResponse(BaseModel):
    """Batch online retrieval result with operational latency metrics."""
    view_name: str
    results: Dict[str, OnlineEntityFeatureValue] = Field(
        ..., description="Map of entity_id to its latest feature vector"
    )
    retrieval_latency_ms: float
    hit_count: int
    miss_count: int

