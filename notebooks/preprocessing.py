"""
preprocessing.py — Module 1
============================
Data loading, cleaning, and RFM (Recency-Frequency-Monetary) feature engineering.

RFM is a classic behavioural segmentation framework:
  • Recency   — days since the user's most recent purchase  (lower  = more engaged)
  • Frequency — total number of purchase transactions        (higher = more engaged)
  • Monetary  — total amount spent across all purchases      (higher = more valuable)

Only 'purchase' events are used for RFM, because we want to measure
confirmed buying behaviour, not browsing intent.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


# ─── Public API ───────────────────────────────────────────────────────────────

def load_data(filepath: str = "E:/Harshi/6thSEM/ML_Lab/ecommerce-behaviour-analysis/data/sample_ecommerce.csv") -> pd.DataFrame:
    """
    Load the raw ecommerce CSV and apply minimal, safe cleaning:
      - Parse event_time as timezone-aware datetime then strip tz (keeps
        arithmetic simple while preserving correct values).
      - Coerce price to float; replace non-positive prices with NaN, then 0.
      - Drop any row missing the three columns every module depends on.

    Returns a clean DataFrame with a UTC-naive 'event_time' column.
    """
    df = pd.read_csv(filepath)

    # ── datetime ──────────────────────────────────────────────────────────────
    # The raw column looks like "2019-11-01 00:00:00+00:00".
    # utc=True parses mixed-offset strings safely; .dt.tz_localize(None)
    # strips the timezone so plain timedelta arithmetic works downstream.
    df["event_time"] = (
        pd.to_datetime(df["event_time"], utc=True)
          .dt.tz_localize(None)
    )

    # ── price ─────────────────────────────────────────────────────────────────
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.loc[df["price"] <= 0, "price"] = np.nan
    df["price"] = df["price"].fillna(0.0)

    # ── drop critical nulls ───────────────────────────────────────────────────
    df.dropna(subset=["user_id", "product_id", "event_type"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[✓] Loaded  : {len(df):,} rows × {df.shape[1]} columns")
    print(f"    Dates   : {df['event_time'].min().date()} → {df['event_time'].max().date()}")
    print(f"    Events  : {df['event_type'].value_counts().to_dict()}")
    print(f"    Users   : {df['user_id'].nunique():,}  |  Products: {df['product_id'].nunique():,}")
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user RFM metrics from purchase events only.

    Algorithm
    ---------
    1. Filter rows where event_type == 'purchase'.
    2. Choose a snapshot date = last purchase date + 1 day.
       (Adding 1 day ensures the most-recent buyer gets recency = 1, not 0.)
    3. Group by user_id and aggregate:
         last_purchase  → max(event_time)
         frequency      → count of purchase rows
         monetary       → sum of price
    4. Recency = (snapshot_date − last_purchase).days

    Returns a DataFrame indexed by user_id with columns:
      recency, frequency, monetary
    """
    purchases = df[df["event_type"] == "purchase"].copy()

    if purchases.empty:
        raise ValueError("No purchase rows found — cannot compute RFM.")

    snapshot = purchases["event_time"].max() + timedelta(days=1)

    rfm = (
        purchases
        .groupby("user_id", as_index=False)
        .agg(
            last_purchase=("event_time", "max"),
            frequency=("event_time", "count"),
            monetary=("price", "sum"),
        )
    )

    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days
    rfm.drop(columns=["last_purchase"], inplace=True)

    # Round monetary to 2 dp for readability
    rfm["monetary"] = rfm["monetary"].round(2)

    print(f"\n[✓] RFM     : {len(rfm):,} users with purchase history")
    print(rfm[["recency", "frequency", "monetary"]].describe().round(2).to_string())
    return rfm


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    df  = load_data("sample_ecommerce.csv")
    rfm = compute_rfm(df)
    print("\nSample RFM rows:")
    print(rfm.head(8).to_string(index=False))
