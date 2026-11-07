"""
Compatibility and convenience exports for API routes.
"""

from app.api.router import api_router
from app.api.routes_features import router as features_router
from app.api.routes_health import router as health_router

__all__ = ["api_router", "features_router", "health_router"]

