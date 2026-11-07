"""
Pydantic v2 schemas for Feature definitions and Feature Views.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class FeatureDataType(str, Enum):
    """Supported data types for features within ChronoStore."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    JSON = "json"
    VECTOR = "vector"


class FeatureDefinition(BaseModel):
    """Specification of an individual feature field."""
    name: str = Field(..., description="Unique name of the feature within the view")
    data_type: FeatureDataType = Field(default=FeatureDataType.FLOAT, description="Feature data type")
    description: Optional[str] = Field(default=None, description="Human-readable feature description")
    default_value: Optional[Any] = Field(default=None, description="Default fallback value if missing")
    tags: Dict[str, str] = Field(default_factory=dict, description="Metadata tags (e.g. tier, owner, team)")


class FeatureView(BaseModel):
    """
    Logical grouping of features that belong to a single entity key,
    sharing common ingestion pipelines and TTL policies.
    """
    name: str = Field(..., description="Unique name of the feature view (e.g., 'user_transaction_stats')")
    entity_id_key: str = Field(default="entity_id", description="Primary entity identifier column name")
    features: Dict[str, FeatureDefinition] = Field(
        ..., description="Map of feature name to its specification"
    )
    ttl_seconds: Optional[int] = Field(
        default=None, description="TTL in seconds for online caching; None implies no expiration"
    )
    description: Optional[str] = Field(default=None, description="Purpose and details of this feature view")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the feature view was registered"
    )

    @field_validator("name")
    @classmethod
    def validate_view_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Feature view name cannot be empty.")
        if not v.replace("_", "").isalnum():
            raise ValueError("Feature view name must be alphanumeric with optional underscores.")
        return v


class FeatureViewRegisterRequest(BaseModel):
    """Request payload to register a new feature view."""
    name: str = Field(..., description="Name of the feature view")
    entity_id_key: str = Field(default="entity_id", description="Entity key column name")
    features: List[FeatureDefinition] = Field(..., description="List of feature definitions")
    ttl_seconds: Optional[int] = Field(default=None, description="Cache TTL in seconds")
    description: Optional[str] = Field(default=None, description="View description")


class FeatureViewResponse(BaseModel):
    """Response payload representing registered feature view metadata."""
    name: str
    entity_id_key: str
    feature_names: List[str]
    features: Dict[str, FeatureDefinition]
    ttl_seconds: Optional[int]
    description: Optional[str]
    created_at: datetime

