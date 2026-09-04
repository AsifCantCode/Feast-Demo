import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("feature_repo/data")
OUT.mkdir(parents=True, exist_ok=True)
RAW_DATA = Path("data/diamonds.csv")
if not RAW_DATA.exists():
    # Fallback to relative class-demo path if present
    RAW_DATA = Path("../class-demo/data/diamonds.csv")

df = pd.read_csv(RAW_DATA).reset_index(drop=True)
df["diamond_id"] = df.index.astype("int64")

cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = ["J", "I", "H", "G", "F", "E", "D"]
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
df["cut"] = df["cut"].map(
    {v: i for i, v in enumerate(cut_order)}).astype("int64")
df["color"] = df["color"].map(
    {v: i for i, v in enumerate(color_order)}).astype("int64")
df["clarity"] = df["clarity"].map(
    {v: i for i, v in enumerate(clarity_order)}).astype("int64")

rng = np.random.default_rng(42)
base = pd.Timestamp("2025-05-01")
df["event_timestamp"] = base - pd.to_timedelta(
    rng.integers(0, 365, size=len(df)), unit="D"
)
df["created"] = pd.Timestamp("2026-05-01")

# aling

# team 1: physical db
physical = df[["diamond_id", "event_timestamp", "created",
               "carat", "depth", "table", "x", "y", "z"]].copy()

# team 2: quality db
quality = df[["diamond_id", "event_timestamp", "created",
              "cut", "color", "clarity"]].copy()

# team 3: labels
labels = df[["diamond_id", "event_timestamp", "price"]].copy()

for c in ["carat", "depth", "table", "x", "y", "z"]:
    physical[c] = physical[c].astype("float32")

physical.to_parquet(OUT / "physical_features.parquet", index=False)
quality.to_parquet(OUT / "quality_features.parquet",   index=False)
labels.to_parquet(OUT / "labels.parquet", index=False)

print(f"Wrote {len(df)} diamonds across 3 parquet files -> {OUT}")
