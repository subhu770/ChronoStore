"""
Feature Store API Routes.
Exposes endpoints for Feature View registration, Real-Time Online ingestion & retrieval,
and Historical Point-in-Time (ASOF) joins.
"""

import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_registry
from app.core.logging import logger
from app.models.entity import (
    OnlineIngestRequest,
    OnlineIngestResponse,
    OnlineRetrievalRequest,
    OnlineRetrievalResponse,
)
from app.models.feature import (
    FeatureViewRegisterRequest,
    FeatureViewResponse,
)
from app.models.join import (
    PointInTimeJoinRequest,
    PointInTimeJoinResponse,
)
from app.services.asof_engine import DataLeakageDetectedError
from app.services.registry import FeatureRegistry

router = APIRouter(prefix="/features", tags=["Features"])


@router.post(
    "/views",
    response_model=FeatureViewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new Feature View",
)
def register_feature_view(
    request: FeatureViewRegisterRequest,
    reg: FeatureRegistry = Depends(get_registry),
) -> FeatureViewResponse:
    """Registers schema metadata and configuration for a logical group of features."""
    try:
        return reg.register_view(request)
    except Exception as e:
        logger.error("Failed to register feature view: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not register feature view: {str(e)}",
        )


@router.get(
    "/views",
    response_model=List[FeatureViewResponse],
    summary="List all registered Feature Views",
)
def list_feature_views(
    reg: FeatureRegistry = Depends(get_registry),
) -> List[FeatureViewResponse]:
    """Returns a list of all active feature views with their schemas."""
    return reg.list_views()


@router.get(
    "/views/{view_name}",
    response_model=FeatureViewResponse,
    summary="Get Feature View metadata",
)
def get_feature_view(
    view_name: str,
    reg: FeatureRegistry = Depends(get_registry),
) -> FeatureViewResponse:
    """Retrieves schema definition and metadata for a specific feature view."""
    view = reg.get_view(view_name)
    if not view:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature view '{view_name}' is not registered.",
        )
    return FeatureViewResponse(
        name=view.name,
        entity_id_key=view.entity_id_key,
        feature_names=list(view.features.keys()),
        features=view.features,
        ttl_seconds=view.ttl_seconds,
        description=view.description,
        created_at=view.created_at,
    )


@router.post(
    "/online",
    response_model=OnlineIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest real-time feature updates into Online Store (and sync to Offline)",
)
def ingest_online_features(
    request: OnlineIngestRequest,
    reg: FeatureRegistry = Depends(get_registry),
) -> OnlineIngestResponse:
    """
    Ingests streaming or transactional feature updates into the ultra-low latency
    in-memory Online Store. When sync_to_offline is True, synchronously appends to
    the immutable Parquet historical log to ensure training-serving parity.
    """
    start_time = time.perf_counter()

    # Ingest into Online Store
    view = reg.get_view(request.view_name)
    ttl = view.ttl_seconds if view else None
    ingested_count = reg.online_store.batch_put(
        view_name=request.view_name,
        records=request.records,
        ttl_seconds=ttl,
    )

    # Dual-write / sync to offline store
    offline_persisted = False
    if request.sync_to_offline:
        reg.offline_store.append_records(
            view_name=request.view_name,
            records=request.records,
        )
        offline_persisted = True

    duration_ms = (time.perf_counter() - start_time) * 1000.0

    return OnlineIngestResponse(
        view_name=request.view_name,
        ingested_count=ingested_count,
        offline_persisted=offline_persisted,
        duration_ms=round(duration_ms, 3),
    )


@router.post(
    "/retrieve",
    response_model=OnlineRetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Real-time low-latency retrieval of latest feature vectors",
)
def retrieve_online_features(
    request: OnlineRetrievalRequest,
    reg: FeatureRegistry = Depends(get_registry),
) -> OnlineRetrievalResponse:
    """
    Ultra-low latency (<5ms) batch retrieval of latest feature vectors by entity keys.
    Supports optional feature column projection.
    """
    start_time = time.perf_counter()

    results = reg.online_store.batch_get(
        view_name=request.view_name,
        entity_ids=request.entity_ids,
        feature_names=request.feature_names,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    hit_count = sum(1 for v in results.values() if v.found)
    miss_count = len(results) - hit_count

    return OnlineRetrievalResponse(
        view_name=request.view_name,
        results=results,
        retrieval_latency_ms=round(duration_ms, 3),
        hit_count=hit_count,
        miss_count=miss_count,
    )


@router.post(
    "/historical-join",
    response_model=PointInTimeJoinResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute leak-free historical Point-in-Time (ASOF) feature join",
)
def historical_point_in_time_join(
    request: PointInTimeJoinRequest,
    reg: FeatureRegistry = Depends(get_registry),
) -> PointInTimeJoinResponse:
    """
    Accepts observation entities with historical timestamps and computes leak-free
    point-in-time feature vectors using backward ASOF alignment. Strictly prevents
    future lookahead leakage for model backtesting and training dataset creation.
    """
    start_time = time.perf_counter()

    try:
        enriched_df = reg.asof_engine.join_observations(
            observations=request.observations,
            feature_views=request.feature_views,
            features=request.features,
            lookback_seconds=request.lookback_seconds,
            allow_exact_matches=request.allow_exact_matches,
            strict_leakage_check=request.strict_leakage_check,
        )
    except DataLeakageDetectedError as e:
        logger.error("Data leakage violation during join: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Temporal data leakage detected: {str(e)}",
        )
    except Exception as e:
        logger.error("Error during historical join: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal join processing failure: {str(e)}",
        )

    # Standardize output for clean JSON serialization
    clean_df = enriched_df.astype(object).where(pd.notnull(enriched_df), None)
    for col in clean_df.columns:
        if pd.api.types.is_datetime64_any_dtype(enriched_df[col]):
            clean_df[col] = enriched_df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    joined_records: List[Dict[str, Any]] = clean_df.to_dict(orient="records")
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Compute statistics
    total_cells = len(joined_records) * (len(clean_df.columns) if not clean_df.empty else 1)
    non_null_cells = sum(
        1 for r in joined_records for v in r.values() if v is not None
    )
    fill_rate = round(non_null_cells / max(total_cells, 1), 4)

    stats = {
        "views_joined": request.feature_views,
        "total_records": len(joined_records),
        "fill_rate": fill_rate,
        "leakage_check_passed": request.strict_leakage_check,
    }

    return PointInTimeJoinResponse(
        row_count=len(joined_records),
        joined_features=joined_records,
        execution_time_ms=round(elapsed_ms, 3),
        statistics=stats,
    )

