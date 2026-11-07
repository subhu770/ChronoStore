"""
API Router aggregation module.
Combines feature store and system health routers under standard versioning.
"""

from fastapi import APIRouter
from app.api.routes_features import router as features_router
from app.api.routes_health import router as health_router

api_router = APIRouter()

# Mount feature endpoints under /features
api_router.include_router(features_router)

# Mount system health endpoints under root and versioned API
api_router.include_router(health_router)

