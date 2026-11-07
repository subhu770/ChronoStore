"""
ChronoStore Bootstrap & Streaming Simulation Demo.
Simulates real-time streaming transaction risk features and demonstrates
Point-in-Time (ASOF) leak-free historical training dataset generation.
"""

import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path for direct script invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.models.entity import EntityRecord, OnlineRetrievalRequest
from app.models.feature import FeatureDataType, FeatureDefinition, FeatureViewRegisterRequest
from app.models.join import ObservationRecord
from app.services.registry import FeatureRegistry


def run_simulation() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="chronostore_demo_"))
    print("=" * 80)
    print("      CHRONOSTORE: PRODUCTION DUAL-LAYER FEATURE STORE DEMO")
    print("=" * 80)
    print(f"Temporary Demo Storage: {temp_dir}\n")

    try:
        # Initialize Feature Registry with dedicated demo storage
        registry = FeatureRegistry()
        registry.offline_store._base_dir = temp_dir
        registry.asof_engine._offline_store = registry.offline_store

        # ---------------------------------------------------------
        # STEP 1: Register Feature View Schema
        # ---------------------------------------------------------
        view_name = "user_fraud_features"
        print(f"[*] Step 1: Registering Feature View '{view_name}'...")
        registry.register_view(
            FeatureViewRegisterRequest(
                name=view_name,
                entity_id_key="entity_id",
                description="Real-time rolling fraud indicators and transaction risk velocity",
                ttl_seconds=86400,
                features=[
                    FeatureDefinition(name="rolling_tx_count_1h", data_type=FeatureDataType.INT),
                    FeatureDefinition(name="rolling_spend_amount_24h", data_type=FeatureDataType.FLOAT),
                    FeatureDefinition(name="failed_pin_attempts", data_type=FeatureDataType.INT),
                    FeatureDefinition(name="risk_score", data_type=FeatureDataType.FLOAT),
                    FeatureDefinition(name="account_state", data_type=FeatureDataType.STRING),
                ],
            )
        )
        print("    [✓] Schema registered successfully.\n")

        # ---------------------------------------------------------
        # STEP 2: Stream Feature Updates Over Time
        # ---------------------------------------------------------
        print("[*] Step 2: Streaming temporal feature updates into Online & Offline stores...")

        # Temporal timeline
        t0 = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 15, 10, 5, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 15, 10, 10, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 15, 10, 15, 0, tzinfo=timezone.utc)

        stream_events = [
            # t0 = 10:00:00 Baseline
            EntityRecord(
                entity_id="user_101",
                timestamp=t0,
                features={
                    "rolling_tx_count_1h": 1,
                    "rolling_spend_amount_24h": 45.0,
                    "failed_pin_attempts": 0,
                    "risk_score": 0.05,
                    "account_state": "ACTIVE",
                },
            ),
            EntityRecord(
                entity_id="user_202",
                timestamp=t0,
                features={
                    "rolling_tx_count_1h": 2,
                    "rolling_spend_amount_24h": 120.0,
                    "failed_pin_attempts": 0,
                    "risk_score": 0.10,
                    "account_state": "ACTIVE",
                },
            ),
            # t1 = 10:05:00 Normal activity
            EntityRecord(
                entity_id="user_101",
                timestamp=t1,
                features={
                    "rolling_tx_count_1h": 2,
                    "rolling_spend_amount_24h": 90.0,
                    "failed_pin_attempts": 0,
                    "risk_score": 0.08,
                    "account_state": "ACTIVE",
                },
            ),
            EntityRecord(
                entity_id="user_202",
                timestamp=t1,
                features={
                    "rolling_tx_count_1h": 3,
                    "rolling_spend_amount_24h": 150.0,
                    "failed_pin_attempts": 0,
                    "risk_score": 0.15,
                    "account_state": "ACTIVE",
                },
            ),
            # t2 = 10:10:00 Fraud burst for user_202!
            EntityRecord(
                entity_id="user_202",
                timestamp=t2,
                features={
                    "rolling_tx_count_1h": 15,
                    "rolling_spend_amount_24h": 4500.0,
                    "failed_pin_attempts": 3,
                    "risk_score": 0.89,
                    "account_state": "FLAGGED",
                },
            ),
            # t3 = 10:15:00 user_202 account locked; user_303 onboarded
            EntityRecord(
                entity_id="user_202",
                timestamp=t3,
                features={
                    "rolling_tx_count_1h": 16,
                    "rolling_spend_amount_24h": 4900.0,
                    "failed_pin_attempts": 5,
                    "risk_score": 0.99,
                    "account_state": "LOCKED",
                },
            ),
            EntityRecord(
                entity_id="user_303",
                timestamp=t3,
                features={
                    "rolling_tx_count_1h": 1,
                    "rolling_spend_amount_24h": 10.0,
                    "failed_pin_attempts": 0,
                    "risk_score": 0.02,
                    "account_state": "ACTIVE",
                },
            ),
        ]

        # Ingest batch
        registry.online_store.batch_put(view_name, stream_events)
        registry.offline_store.append_records(view_name, stream_events)
        print(f"    [✓] Ingested {len(stream_events)} temporal events across 4 historical timestamps.\n")

        # ---------------------------------------------------------
        # STEP 3: Online Store Low-Latency Retrieval Benchmark
        # ---------------------------------------------------------
        print("[*] Step 3: Benchmarking Online Store real-time retrieval latency (<5ms target)...")
        queries = ["user_101", "user_202", "user_303"]
        start_bench = time.perf_counter()
        iterations = 500
        for _ in range(iterations):
            registry.online_store.batch_get(view_name, queries)
        elapsed_bench_ms = ((time.perf_counter() - start_bench) * 1000.0) / iterations
        print(f"    [✓] Average batch retrieval latency: {elapsed_bench_ms:.4f} ms per batch (Target: <5.0 ms)")

        online_sample = registry.online_store.batch_get(view_name, ["user_202"])
        print(f"    [✓] Online latest state for user_202: state={online_sample['user_202'].features['account_state']}, "
              f"risk={online_sample['user_202'].features['risk_score']}\n")

        # ---------------------------------------------------------
        # STEP 4: Historical Observations for ML Training Dataset
        # ---------------------------------------------------------
        print("[*] Step 4: Generating training dataset with Historical Point-in-Time (ASOF) join...")
        observations = [
            # Obs 1: user_202 checkout attempted at 10:07:30 (BEFORE 10:10 fraud burst)
            ObservationRecord(
                entity_id="user_202",
                timestamp=datetime(2026, 1, 15, 10, 7, 30, tzinfo=timezone.utc),
                label=0,  # Legitimate checkout
                metadata={"event_name": "checkout_10_07_30"},
            ),
            # Obs 2: user_202 checkout attempted at 10:12:00 (AFTER 10:10 burst, BEFORE 10:15 lock)
            ObservationRecord(
                entity_id="user_202",
                timestamp=datetime(2026, 1, 15, 10, 12, 0, tzinfo=timezone.utc),
                label=1,  # Fraudulent checkout
                metadata={"event_name": "checkout_10_12_00"},
            ),
            # Obs 3: user_101 checkout at 10:04:59 (1 second BEFORE 10:05 update!)
            ObservationRecord(
                entity_id="user_101",
                timestamp=datetime(2026, 1, 15, 10, 4, 59, tzinfo=timezone.utc),
                label=0,
                metadata={"event_name": "checkout_10_04_59"},
            ),
            # Obs 4: user_303 checkout at 10:08:00 (BEFORE user_303 onboarded at 10:15)
            ObservationRecord(
                entity_id="user_303",
                timestamp=datetime(2026, 1, 15, 10, 8, 0, tzinfo=timezone.utc),
                label=0,
                metadata={"event_name": "checkout_10_08_00"},
            ),
        ]

        # Perform Leak-Free Point-in-Time Join
        enriched_df = registry.asof_engine.join_observations(
            observations=observations,
            feature_views=[view_name],
            strict_leakage_check=True,
        )

        print("    [✓] Historical ASOF Join completed successfully!\n")

        # ---------------------------------------------------------
        # STEP 5: Verification & Anti-Leakage Comparative Analysis
        # ---------------------------------------------------------
        print("=" * 80)
        print("              LEAK-FREE ASOF JOIN vs NAIVE LATEST-STATE JOIN")
        print("=" * 80)

        comparison_rows = []
        for _, row in enriched_df.iterrows():
            eid = row["entity_id"]
            obs_t = row["timestamp"].strftime("%H:%M:%S")
            event_name = row["event_name"]
            asof_spend = row["rolling_spend_amount_24h"]
            asof_risk = row["risk_score"]
            asof_state = row["account_state"]

            # Naive latest state (simulates standard data warehouse join without point-in-time)
            latest = registry.online_store.get(view_name, eid).features
            naive_spend = latest.get("rolling_spend_amount_24h")
            naive_risk = latest.get("risk_score")
            naive_state = latest.get("account_state")

            is_leaked = (asof_risk != naive_risk)

            comparison_rows.append({
                "Entity": eid,
                "Obs Time": obs_t,
                "Event": event_name,
                "ChronoStore (ASOF) State": f"{asof_state} (risk={asof_risk}, spend={asof_spend})",
                "Naive (Latest) State": f"{naive_state} (risk={naive_risk}, spend={naive_spend})",
                "Leak Prevented?": "YES (LEAKAGE BLOCKED)" if is_leaked else "MATCH (NO CHANGE)",
            })

        comp_df = pd.DataFrame(comparison_rows)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(comp_df.to_string(index=False))
        print("\n" + "=" * 80)

        # Explicit Boundary Verification check:
        # Obs at 10:04:59 must have rolling_spend=45.0 (t0), NEVER 90.0 (t1 at 10:05:00)
        obs_101 = enriched_df[enriched_df["entity_id"] == "user_101"].iloc[0]
        assert obs_101["rolling_spend_amount_24h"] == 45.0, (
            f"Leakage detected: expected 45.0, got {obs_101['rolling_spend_amount_24h']}"
        )
        print(">>> CRITICAL TEST PASSED: At t=10:04:59, retrieved t=10:00:00 spend (45.0), NOT t=10:05:00 spend (90.0)!")

        # Obs at 10:07:30 must have rolling_spend=150.0 (t1), NEVER 4500.0 (t2 at 10:10:00)
        obs_202_early = enriched_df[enriched_df["event_name"] == "checkout_10_07_30"].iloc[0]
        assert obs_202_early["rolling_spend_amount_24h"] == 150.0, (
            f"Leakage detected: expected 150.0, got {obs_202_early['rolling_spend_amount_24h']}"
        )
        assert obs_202_early["account_state"] == "ACTIVE"
        print(">>> CRITICAL TEST PASSED: At t=10:07:30, user_202 state was ACTIVE (spend 150.0), NOT FLAGGED (spend 4500.0)!")

        # Obs at 10:08:00 for user_303 must have null features (onboarded later at 10:15)
        obs_303 = enriched_df[enriched_df["entity_id"] == "user_303"].iloc[0]
        assert pd.isna(obs_303["risk_score"]), (
            f"Leakage detected: expected NaN for un-onboarded user_303, got {obs_303['risk_score']}"
        )
        print(">>> CRITICAL TEST PASSED: At t=10:08:00, user_303 features were NaN (user onboarded later at 10:15:00)!")
        print("\nSimulation completed successfully. ChronoStore guarantees zero future-data leakage.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_simulation()
