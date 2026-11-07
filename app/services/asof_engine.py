"""
Point-in-Time (ASOF) Join Engine.
Implements mathematically rigorous, leak-free temporal joins using pandas.merge_asof.
Ensures training datasets reflect the exact state of feature values known at each historical
observation instant, strictly preventing future lookahead data leakage.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

from app.core.logging import logger
from app.models.join import ObservationRecord
from app.services.offline_store import OfflineFeatureStore


class DataLeakageDetectedError(RuntimeError):
    """Raised when an ASOF join matches an event timestamp strictly ahead of observation time."""
    pass


class AsofJoinEngine:
    """
    Core temporal join engine for ChronoStore.
    Executes vectorized, backward-aligned point-in-time joins across historical observation logs.
    """

    def __init__(self, offline_store: OfflineFeatureStore) -> None:
        self._offline_store = offline_store

    def join_observations(
        self,
        observations: Union[pd.DataFrame, Sequence[ObservationRecord], Sequence[Dict[str, Any]]],
        feature_views: Sequence[str],
        features: Optional[Sequence[str]] = None,
        lookback_seconds: Optional[float] = None,
        allow_exact_matches: bool = True,
        strict_leakage_check: bool = True,
    ) -> pd.DataFrame:
        """
        Executes a leak-free point-in-time backward join of observations against historical feature logs.

        Parameters:
            observations: Input observation records containing 'entity_id' and 'timestamp'.
            feature_views: List of feature view names to join against.
            features: Optional subset of feature column names to project.
            lookback_seconds: Max allowed feature age in seconds. Older features evaluate to NaN.
            allow_exact_matches: If True, features at t_feat == t_obs are valid. If False, strictly t_feat < t_obs.
            strict_leakage_check: Asserts that no feature timestamp exceeds observation timestamp.

        Returns:
            pd.DataFrame: Enriched dataset preserving original input row ordering.
        """
        start_time = time.perf_counter()

        # Step 1: Normalize observations to DataFrame
        obs_df = self._normalize_observations(observations)
        if obs_df.empty:
            logger.warning("Empty observations received in AsofJoinEngine.")
            return obs_df

        # Preserve original order
        obs_df["_chronostore_orig_index"] = np.arange(len(obs_df))
        
        # Sort observations by timestamp as required by merge_asof
        result_df = obs_df.sort_values(by="timestamp").copy()

        unique_entities = result_df["entity_id"].unique().tolist()
        min_obs_time = result_df["timestamp"].min()
        max_obs_time = result_df["timestamp"].max()

        # Configure staleness tolerance if specified
        tolerance = pd.Timedelta(seconds=lookback_seconds) if lookback_seconds is not None else None

        # Step 2: Sequentially join against each requested feature view
        for view_name in feature_views:
            # Query historical records from offline store for the observed entities up to max_obs_time
            hist_df = self._offline_store.read_historical_data(
                view_name=view_name,
                entity_ids=unique_entities,
                start_time=None,  # Need historical events prior to observations
                end_time=max_obs_time,
                feature_names=features,
            )

            # Determine feature columns in this view
            reserved_cols = {"entity_id", "timestamp"}
            view_feature_cols = [c for c in hist_df.columns if c not in reserved_cols]

            if hist_df.empty or not view_feature_cols:
                logger.info(
                    "No historical records found for view '%s' within requested entities/timestamps. Filling with NaN.",
                    view_name,
                )
                target_cols = features if features else view_feature_cols
                for col in target_cols:
                    if col not in result_df.columns:
                        result_df[col] = np.nan
                continue

            # Ensure clean types and formatting for ASOF merge
            hist_df["entity_id"] = hist_df["entity_id"].astype(str)
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], utc=True)
            
            # Deterministic deduplication: if multiple writes occur at exact same timestamp, retain the latest write
            hist_df = hist_df.drop_duplicates(subset=["entity_id", "timestamp"], keep="last")
            hist_df = hist_df.sort_values(by="timestamp")

            # Create an audit timestamp column for leakage verification
            audit_ts_col = f"_audit_ts_{view_name}"
            hist_df[audit_ts_col] = hist_df["timestamp"]

            # Step 3: Perform vectorized backward merge_asof
            result_df = pd.merge_asof(
                left=result_df,
                right=hist_df[["entity_id", "timestamp", audit_ts_col] + view_feature_cols],
                on="timestamp",
                by="entity_id",
                direction="backward",
                allow_exact_matches=allow_exact_matches,
                tolerance=tolerance,
            )

            # Step 4: Strict Leakage Verification
            if strict_leakage_check:
                self._verify_leakage(
                    merged_df=result_df,
                    obs_ts_col="timestamp",
                    feat_ts_col=audit_ts_col,
                    view_name=view_name,
                    allow_exact=allow_exact_matches,
                )

            # Drop the internal audit timestamp column
            result_df.drop(columns=[audit_ts_col], inplace=True)

        # Step 5: Restore original input order and drop tracking index
        result_df = result_df.sort_values(by="_chronostore_orig_index").reset_index(drop=True)
        result_df.drop(columns=["_chronostore_orig_index"], inplace=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug(
            "ASOF Point-in-Time join completed: %d rows enriched in %.2f ms across views: %s",
            len(result_df),
            elapsed_ms,
            feature_views,
        )

        return result_df

    def _normalize_observations(
        self,
        observations: Union[pd.DataFrame, Sequence[ObservationRecord], Sequence[Dict[str, Any]]],
    ) -> pd.DataFrame:
        """Standardizes input observations into a typed, UTC-normalized DataFrame."""
        if isinstance(observations, pd.DataFrame):
            df = observations.copy()
        elif isinstance(observations, Sequence):
            rows = []
            for item in observations:
                if isinstance(item, ObservationRecord):
                    row = {
                        "entity_id": item.entity_id,
                        "timestamp": item.timestamp,
                    }
                    if item.label is not None:
                        row["label"] = item.label
                    if item.metadata:
                        row.update(item.metadata)
                    rows.append(row)
                elif isinstance(item, dict):
                    rows.append(dict(item))
                else:
                    raise TypeError(f"Unsupported observation item type: {type(item)}")
            df = pd.DataFrame(rows)
        else:
            raise TypeError(f"Unsupported observations container: {type(observations)}")

        if df.empty:
            return df

        if "entity_id" not in df.columns or "timestamp" not in df.columns:
            raise ValueError("Observations must contain both 'entity_id' and 'timestamp' columns.")

        df["entity_id"] = df["entity_id"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def _verify_leakage(
        self,
        merged_df: pd.DataFrame,
        obs_ts_col: str,
        feat_ts_col: str,
        view_name: str,
        allow_exact: bool,
    ) -> None:
        """
        Rigorous audit asserting that no matched feature timestamp is in the future
        relative to the observation timestamp.
        """
        matched = merged_df[merged_df[feat_ts_col].notna()]
        if matched.empty:
            return

        if allow_exact:
            leaks = matched[matched[feat_ts_col] > matched[obs_ts_col]]
        else:
            leaks = matched[matched[feat_ts_col] >= matched[obs_ts_col]]

        if not leaks.empty:
            first_leak = leaks.iloc[0]
            obs_t = first_leak[obs_ts_col]
            feat_t = first_leak[feat_ts_col]
            eid = first_leak["entity_id"]
            diff_sec = (feat_t - obs_t).total_seconds()
            raise DataLeakageDetectedError(
                f"Critical Data Leakage Detected in view '{view_name}'! "
                f"Entity '{eid}' observation at {obs_t.isoformat()} matched feature timestamp "
                f"{feat_t.isoformat()} which is {diff_sec:.3f} seconds in the future."
            )

    def verify_temporal_boundary(
        self,
        view_name: str,
        entity_id: str,
        t_past: datetime,
        t_future: datetime,
        t_observation: datetime,
        past_value: Any,
        future_value: Any,
        feature_name: str,
    ) -> bool:
        """
        Explicit verification method testing that:
        Given an entity feature update at t=t_past with value v_past,
        and another update at t=t_future with value v_future (t_past <= t_observation < t_future),
        an inference observation at t=t_observation retrieves v_past and NEVER v_future.
        """
        obs = [
            ObservationRecord(
                entity_id=entity_id,
                timestamp=t_observation,
            )
        ]
        enriched = self.join_observations(
            observations=obs,
            feature_views=[view_name],
            features=[feature_name],
            strict_leakage_check=True,
        )

        actual_val = enriched.iloc[0][feature_name]

        # Verify against boundary expectation
        if actual_val != past_value:
            raise AssertionError(
                f"Temporal boundary violation for entity {entity_id} at {t_observation}: "
                f"Expected past value {past_value}, but got {actual_val} (future value was {future_value})."
            )
        return True

