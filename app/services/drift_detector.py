"""
Feature Drift & Training-Serving Skew Detector.
Calculates Population Stability Index (PSI), Kolmogorov-Smirnov (KS) tests,
and categorical divergence to detect data distribution shifts between
historical training baselines and real-time inference distributions.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from scipy import stats

from app.core.logging import logger


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes the Population Stability Index (PSI) between two numeric distributions.
    Interpretation:
        PSI < 0.1: No significant change / stable.
        0.1 <= PSI < 0.2: Moderate shift / warning.
        PSI >= 0.2: Significant shift / model retraining required.
    """
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0

    # Determine quantile bin edges from reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges = np.unique(bin_edges)  # Prevent identical bin edges

    if len(bin_edges) <= 1:
        return 0.0

    # Expand edge boundaries slightly to encompass extremes
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    # Calculate frequency counts per bin
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    # Convert to proportions with smoothing epsilon to prevent division by zero
    ref_prop = (ref_counts / len(ref_clean)) + epsilon
    curr_prop = (curr_counts / len(curr_clean)) + epsilon

    # Re-normalize proportions
    ref_prop = ref_prop / np.sum(ref_prop)
    curr_prop = curr_prop / np.sum(curr_prop)

    # PSI calculation formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_value = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
    return float(np.round(psi_value, 4))


def calculate_categorical_psi(
    reference: Sequence[Any],
    current: Sequence[Any],
    epsilon: float = 1e-4,
) -> float:
    """Computes PSI for categorical string features across unique categories."""
    ref_series = pd.Series([str(x) for x in reference if pd.notna(x)])
    curr_series = pd.Series([str(x) for x in current if pd.notna(x)])

    if ref_series.empty or curr_series.empty:
        return 0.0

    all_categories = sorted(list(set(ref_series.unique()).union(set(curr_series.unique()))))
    ref_counts = ref_series.value_counts()
    curr_counts = curr_series.value_counts()

    ref_prop = np.array([ref_counts.get(cat, 0) / len(ref_series) + epsilon for cat in all_categories])
    curr_prop = np.array([curr_counts.get(cat, 0) / len(curr_series) + epsilon for cat in all_categories])

    ref_prop = ref_prop / np.sum(ref_prop)
    curr_prop = curr_prop / np.sum(curr_prop)

    psi_value = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
    return float(np.round(psi_value, 4))


class DriftDetector:
    """
    Automated feature distribution monitoring engine.
    Compares reference baseline data against serving inference observations.
    """

    def __init__(
        self,
        psi_threshold_warning: float = 0.1,
        psi_threshold_critical: float = 0.2,
        ks_alpha: float = 0.05,
    ) -> None:
        self.psi_threshold_warning = psi_threshold_warning
        self.psi_threshold_critical = psi_threshold_critical
        self.ks_alpha = ks_alpha

    def analyze_feature(
        self,
        feature_name: str,
        reference_values: Union[Sequence[Any], np.ndarray, pd.Series],
        current_values: Union[Sequence[Any], np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Analyzes a single feature column for distribution drift."""
        ref_s = pd.Series(reference_values).dropna()
        curr_s = pd.Series(current_values).dropna()

        if ref_s.empty or curr_s.empty:
            return {
                "feature_name": feature_name,
                "data_type": "unknown",
                "drift_detected": False,
                "severity": "INSUFFICIENT_DATA",
                "psi_score": 0.0,
            }

        is_numeric = pd.api.types.is_numeric_dtype(ref_s) and pd.api.types.is_numeric_dtype(curr_s)

        if is_numeric:
            ref_arr = ref_s.to_numpy(dtype=float)
            curr_arr = curr_s.to_numpy(dtype=float)

            psi = calculate_psi(ref_arr, curr_arr)
            ks_res = stats.ks_2samp(ref_arr, curr_arr)
            ks_stat = round(float(ks_res.statistic), 4)
            ks_pval = round(float(ks_res.pvalue), 4)

            drift_detected = bool(psi >= self.psi_threshold_warning or ks_pval < self.ks_alpha)

            if psi >= self.psi_threshold_critical:
                severity = "CRITICAL"
            elif psi >= self.psi_threshold_warning or ks_pval < self.ks_alpha:
                severity = "MODERATE"
            else:
                severity = "NONE"

            stats_dict = {
                "reference": {
                    "count": int(len(ref_arr)),
                    "mean": round(float(np.mean(ref_arr)), 3),
                    "std": round(float(np.std(ref_arr)), 3),
                    "median": round(float(np.median(ref_arr)), 3),
                },
                "current": {
                    "count": int(len(curr_arr)),
                    "mean": round(float(np.mean(curr_arr)), 3),
                    "std": round(float(np.std(curr_arr)), 3),
                    "median": round(float(np.median(curr_arr)), 3),
                },
            }

            return {
                "feature_name": feature_name,
                "data_type": "numeric",
                "drift_detected": drift_detected,
                "severity": severity,
                "psi_score": psi,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_pval,
                "metrics": stats_dict,
            }
        else:
            # Categorical drift evaluation
            psi = calculate_categorical_psi(ref_s.tolist(), curr_s.tolist())
            drift_detected = bool(psi >= self.psi_threshold_warning)

            if psi >= self.psi_threshold_critical:
                severity = "CRITICAL"
            elif psi >= self.psi_threshold_warning:
                severity = "MODERATE"
            else:
                severity = "NONE"

            return {
                "feature_name": feature_name,
                "data_type": "categorical",
                "drift_detected": drift_detected,
                "severity": severity,
                "psi_score": psi,
                "metrics": {
                    "reference_unique_count": int(ref_s.nunique()),
                    "current_unique_count": int(curr_s.nunique()),
                },
            }

    def analyze_dataset(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Runs dataset-wide drift detection across common feature columns."""
        common_cols = [
            col
            for col in reference_df.columns
            if col in current_df.columns and col not in {"entity_id", "timestamp"}
        ]
        target_features = [f for f in (feature_names or common_cols) if f in common_cols]

        feature_reports = {}
        drifted_count = 0
        critical_count = 0

        for feat in target_features:
            report = self.analyze_feature(
                feature_name=feat,
                reference_values=reference_df[feat],
                current_values=current_df[feat],
            )
            feature_reports[feat] = report
            if report["drift_detected"]:
                drifted_count += 1
            if report.get("severity") == "CRITICAL":
                critical_count += 1

        drift_ratio = round(drifted_count / max(len(target_features), 1), 3)

        return {
            "features_analyzed": len(target_features),
            "features_drifted": drifted_count,
            "critical_features": critical_count,
            "dataset_drift_ratio": drift_ratio,
            "dataset_drift_detected": bool(drift_ratio > 0.3 or critical_count > 0),
            "feature_reports": feature_reports,
        }
