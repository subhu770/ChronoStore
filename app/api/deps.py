"""
FastAPI dependency injection utilities for ChronoStore.
"""

from typing import Generator
from app.services.registry import FeatureRegistry, registry


def get_registry() -> Generator[FeatureRegistry, None, None]:
    """Provides the active FeatureRegistry singleton."""
    yield registry

