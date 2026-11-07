"""
Pytest configuration and shared test fixtures for ChronoStore.
"""

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_registry
from app.core.config import settings
from app.main import app
from app.models.entity import EntityRecord
from app.models.feature import FeatureDataType, FeatureDefinition, FeatureViewRegisterRequest
from app.services.asof_engine import AsofJoinEngine
from app.services.offline_store import OfflineFeatureStore
from app.services.online_store import OnlineFeatureStore
from app.services.registry import FeatureRegistry


@pytest.fixture
def temp_storage_dir() -> Generator[Path, None, None]:
    """Provides an isolated temporary directory for offline Parquet storage."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_chronostore_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def clean_registry(temp_storage_dir: Path) -> FeatureRegistry:
    """Provides a fresh isolated FeatureRegistry with dedicated storage."""
    online_store = OnlineFeatureStore(default_ttl_seconds=3600)
    offline_store = OfflineFeatureStore(base_dir=temp_storage_dir)
    asof_engine = AsofJoinEngine(offline_store=offline_store)
    reg = FeatureRegistry(
        online_store=online_store,
        offline_store=offline_store,
        asof_engine=asof_engine,
    )
    return reg


@pytest.fixture
def sample_feature_view_req() -> FeatureViewRegisterRequest:
    """Sample Feature View definition request."""
    return FeatureViewRegisterRequest(
        name="user_credit_features",
        entity_id_key="entity_id",
        description="User financial and risk attributes",
        ttl_seconds=3600,
        features=[
            FeatureDefinition(name="credit_score", data_type=FeatureDataType.INT),
            FeatureDefinition(name="debt_to_income", data_type=FeatureDataType.FLOAT),
            FeatureDefinition(name="account_tier", data_type=FeatureDataType.STRING),
        ],
    )


@pytest.fixture
def client(clean_registry: FeatureRegistry) -> Generator[TestClient, None, None]:
    """TestClient configured with dependency overrides for isolated testing."""
    app.dependency_overrides[get_registry] = lambda: clean_registry
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

