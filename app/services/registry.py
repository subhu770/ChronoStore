"""
Feature Registry Service.
Central catalog for Feature Views, Schemas, and Orchestration of Online/Offline stores.
"""

import threading
from typing import Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.feature import FeatureView, FeatureViewRegisterRequest, FeatureViewResponse
from app.services.asof_engine import AsofJoinEngine
from app.services.offline_store import OfflineFeatureStore
from app.services.online_store import OnlineFeatureStore


class FeatureRegistry:
    """
    Thread-safe registry for metadata, schema definitions, and shared engine references.
    """

    def __init__(
        self,
        online_store: Optional[OnlineFeatureStore] = None,
        offline_store: Optional[OfflineFeatureStore] = None,
        asof_engine: Optional[AsofJoinEngine] = None,
    ) -> None:
        self._views: Dict[str, FeatureView] = {}
        self._lock = threading.RLock()

        self.online_store = online_store or OnlineFeatureStore(
            default_ttl_seconds=settings.ONLINE_STORE_DEFAULT_TTL_SECONDS
        )
        self.offline_store = offline_store or OfflineFeatureStore(
            base_dir=settings.get_offline_dir(),
            compression=settings.OFFLINE_PARQUET_COMPRESSION,
        )
        self.asof_engine = asof_engine or AsofJoinEngine(offline_store=self.offline_store)

    def register_view(self, req: FeatureViewRegisterRequest) -> FeatureViewResponse:
        """Registers a new feature view or updates an existing one."""
        with self._lock:
            feature_dict = {f.name: f for f in req.features}
            view = FeatureView(
                name=req.name,
                entity_id_key=req.entity_id_key,
                features=feature_dict,
                ttl_seconds=req.ttl_seconds,
                description=req.description,
            )
            self._views[view.name] = view
            logger.info("Registered feature view '%s' with %d features.", view.name, len(view.features))

            return FeatureViewResponse(
                name=view.name,
                entity_id_key=view.entity_id_key,
                feature_names=list(view.features.keys()),
                features=view.features,
                ttl_seconds=view.ttl_seconds,
                description=view.description,
                created_at=view.created_at,
            )

    def get_view(self, view_name: str) -> Optional[FeatureView]:
        """Retrieves a feature view definition by name."""
        with self._lock:
            return self._views.get(view_name)

    def list_views(self) -> List[FeatureViewResponse]:
        """Lists all registered feature views."""
        with self._lock:
            responses = []
            for view in self._views.values():
                responses.append(
                    FeatureViewResponse(
                        name=view.name,
                        entity_id_key=view.entity_id_key,
                        feature_names=list(view.features.keys()),
                        features=view.features,
                        ttl_seconds=view.ttl_seconds,
                        description=view.description,
                        created_at=view.created_at,
                    )
                )
            return responses

    def view_exists(self, view_name: str) -> bool:
        """Checks if a feature view is registered."""
        with self._lock:
            return view_name in self._views


# Global singleton registry instance
registry = FeatureRegistry()

