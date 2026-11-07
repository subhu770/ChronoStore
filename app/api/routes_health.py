"""
Health and Operational Metrics API Routes.
Exposes endpoints for system monitoring, liveness checks, and cache telemetry.
"""

import os
import time
from typing import Any, Dict
from fastapi import APIRouter, Depends, status

from app.api.deps import get_registry
from app.core.config import settings
from app.services.registry import FeatureRegistry

router = APIRouter(tags=["System"])

# Service startup epoch timestamp
START_TIME = time.time()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
)
def get_health(reg: FeatureRegistry = Depends(get_registry)) -> Dict[str, Any]:
    """Returns the operational status and subsystem availability of ChronoStore."""
    uptime_seconds = round(time.time() - START_TIME, 2)
    offline_dir = settings.get_offline_dir()
    offline_writable = os.access(offline_dir, os.W_OK)

    online_metrics = reg.online_store.get_metrics()
    offline_views = reg.offline_store.list_views()

    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": settings.ENV,
        "uptime_seconds": uptime_seconds,
        "subsystems": {
            "online_store": {
                "status": "online",
                "cached_entries": online_metrics["total_entries"],
            },
            "offline_store": {
                "status": "available" if offline_writable else "degraded",
                "storage_directory": str(offline_dir),
                "persisted_views_count": len(offline_views),
            },
            "registry": {
                "registered_views_count": len(reg.list_views()),
            },
        },
    }


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Operational and Performance Metrics",
)
def get_metrics(reg: FeatureRegistry = Depends(get_registry)) -> Dict[str, Any]:
    """Returns granular operational metrics for latency monitoring and cache hit analysis."""
    online_metrics = reg.online_store.get_metrics()

    offline_views = reg.offline_store.list_views()
    offline_stats = {
        view: reg.offline_store.get_view_stats(view) for view in offline_views
    }

    return {
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "online_store": online_metrics,
        "offline_store": {
            "views": offline_stats,
            "total_views": len(offline_views),
        },
    }

