"""
Pydantic v2 schemas for Historical Point-in-Time (ASOF) Joins.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from app.models.entity import ensure_utc


class ObservationRecord(BaseModel):
    """
    Represents an entity observation/event at a specific historical point in time.
    For example: a transaction, loan application, or click event.
    """
    entity_id: str = Field(..., description="Unique entity identifier")
    timestamp: datetime = Field(
        ...,
        description="Observation timestamp at which feature values must be reconstructed without leakage"
    )
    label: Optional[Any] = Field(
        default=None,
        description="Optional target label for ML training (e.g., is_fraud=1, default=0)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Arbitrary context metadata to pass through the join intact"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_and_validate_timestamp(cls, v: Any) -> datetime:
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return ensure_utc(dt)
        elif isinstance(v, datetime):
            return ensure_utc(v)
        raise ValueError(f"Invalid timestamp value: {v}")


class PointInTimeJoinRequest(BaseModel):
    """
    Request to perform a point-in-time leak-free join between
    an observation sequence and one or more feature views.
    """
    observations: List[ObservationRecord] = Field(
        ..., min_length=1, description="Historical observations requiring feature enrichment"
    )
    feature_views: List[str] = Field(
        ..., min_length=1, description="List of feature views to join against"
    )
    features: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of feature names to extract across views"
    )
    lookback_seconds: Optional[float] = Field(
        default=None,
        description="Maximum allowed staleness window (seconds). Features older than this relative to observation time become null."
    )
    allow_exact_matches: bool = Field(
        default=True,
        description="If True, features updated at exactly t_obs are valid. If False, strictly t_feat < t_obs."
    )
    strict_leakage_check: bool = Field(
        default=True,
        description="Assert zero future-feature timestamp leaks post-merge (raises error if violation found)."
    )


class PointInTimeJoinResponse(BaseModel):
    """Result of historical Point-in-Time feature join."""
    row_count: int
    joined_features: List[Dict[str, Any]] = Field(
        ..., description="Enriched observation records with point-in-time feature values"
    )
    execution_time_ms: float
    statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics (match counts, missing value rates, leakage verification status)"
    )

