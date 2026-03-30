# ══════════════════════════════════════════════════════════════════════════════
# feature_engineering.py — Script 05 (FIXED)
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR     = ARTIFACTS_DIR / "plots"
SCALERS_DIR   = ARTIFACTS_DIR / "scalers"

for d in [PLOTS_DIR, SCALERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SPLIT_DATE = pd.Timestamp("2023-01-01")


def load_data():
    df = pd.read_csv(DATA_DIR / "processed_data.csv", parse_dates=["date"])
    eoq_df = None
    eoq_path = ARTIFACTS_DIR / "eoq_analysis.csv"
    if eoq_path.exists():
        eoq_df = pd.read_csv(eoq_path)
    print(f"📂 Loaded {len(df):,} rows")
    return df, eoq_df


# ══════════════════════════════════════════════════════════════════════════════
# 1. Lag & Rolling Features
# ══════════════════════════════════════════════════════════════════════════════
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n⚙️  Adding lag & rolling features...")
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    grp = ["store_id", "item_id"]

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in [1, 7, 14, 28, 56]:
        df[f"lag_{lag}d"] = df.groupby(grp)["sales"].shift(lag)
        print(f"     lag_{lag}d ✓")

    # ── Rolling features (on shift-1 sales to avoid leakage) ────────────────
    # Create a helper column: yesterday's sales
    df["_shifted"] = df.groupby(grp)["sales"].shift(1)

    for window in [7, 14, 28]:
        df[f"roll_mean_{window}d"] = (
            df.groupby(grp)["_shifted"]
              .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"roll_std_{window}d"] = (
            df.groupby(grp)["_shifted"]
              .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )
        df[f"roll_max_{window}d"] = (
            df.groupby(grp)["_shifted"]
              .transform(lambda x: x.rolling(window, min_periods=1).max())
        )
        df[f"roll_min_{window}d"] = (
            df.groupby(grp)["_shifted"]
              .transform(lambda x: x.rolling(window, min_periods=1).min())
        )
        print(f"     roll_{window}d (mean/std/max/min) ✓")

    # Drop helper
    df.drop(columns=["_shifted"], inplace=True)

    # ── Trend & velocity ─────────────────────────────────────────────────────
    df["trend_short"] = df["lag_1d"]  - df["lag_7d"]
    df["trend_long"]  = df["lag_7d"]  - df["lag_28d"]
    df["velocity"]    = (
        (df["roll_mean_7d"] - df["roll_mean_28d"]) /
        (df["roll_mean_28d"] + 1e-9)
    )

    print(f"   ✅ Lag/rolling features done  (cols: {df.shape[1]})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. EOQ-Derived Features
# ══════════════════════════════════════════════════════════════════════════════
def add_eoq_features(df: pd.DataFrame, eoq_df) -> pd.DataFrame:
    if eoq_df is None:
        print("   ⚠️  EOQ data not found, skipping")
        return df

    print("\n📦 Adding EOQ-derived features...")
    eoq_feats = eoq_df[[
        "item_id", "EOQ", "Reorder_Point", "Safety_Stock",
        "avg_unit_price", "Annual_Holding_Cost",
        "Annual_Ordering_Cost", "Total_Annual_Cost"
    ]].copy()
    eoq_feats.columns = [
        "item_id", "eoq", "reorder_point", "safety_stock",
        "avg_item_price", "annual_holding_cost",
        "annual_ordering_cost", "total_annual_cost"
    ]
    df = df.merge(eoq_feats, on="item_id", how="left")

    if "roll_mean_7d" in df.columns:
        df["below_rop"]    = (df["roll_mean_7d"] < df["reorder_point"]).astype(int)
        df["above_safety"] = (df["roll_mean_7d"] > df["safety_stock"]).astype(int)

    print(f"   ✅ EOQ features merged  (cols: {df.shape[1]})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Price & Promo Features
# ══════════════════════════════════════════════════════════════════════════════
def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n💲 Adding price & promo features...")
    item_avg = df.groupby("item_id")["price"].transform("mean")
    df["price_deviation"]  = df["price"] - item_avg
    df["price_pct_change"] = df["price_deviation"] / (item_avg + 1e-9)
    is_wknd = df["is_weekend"] if "is_weekend" in df.columns else 0
    df["promo_x_weekend"]  = df["promo"] * is_wknd
    df["promo_x_month"]    = df["promo"] * df["month"]
    df["price_x_promo"]    = df["price"] * df["promo"]
    print(f"   ✅ Price features done  (cols: {df.shape[1]})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Drop rows with NaN in lag columns
# ══════════════════════════════════════════════════════════════════════════════
def clean_engineered(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_")]
    before   = len(df)
    df = df.dropna(subset=lag_cols)
    print(f"\n🧹 Dropped {before - len(df):,} NaN rows → {len(df):,} remaining")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. Feature Selection — VarianceThreshold only
# ══════════════════════════════════════════════════════════════════════════════
EXCLUDE_COLS = [
    "date", "store_id", "item_id", "sales", "sales_scaled",
    "promo_label", "avg_item_price",
]


def feature_selection(df: pd.DataFrame) -> list:
    print("\n🔍 Feature selection (VarianceThreshold)...")

    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    print(f"   Candidates: {len(feature_cols)}")

    # Fit on TRAINING data only
    train_mask = df["date"] < SPLIT_DATE
    X_train = df.loc[train_mask, feature_cols].fillna(0)

    vt = VarianceThreshold(threshold=0.001)
    vt.fit(X_train)

    selected = [c for c, keep in zip(feature_cols, vt.get_support()) if keep]
    dropped  = [c for c, keep in zip(feature_cols, vt.get_support()) if not keep]

    print(f"   Kept: {len(selected)}  |  Dropped: {len(dropped)}")
    if dropped:
        print(f"   Dropped: {dropped}")

    with open(ARTIFACTS_DIR / "selected_features.json", "w") as f:
        json.dump(selected, f, indent=2)
    print(f"   ✅ Saved: selected_features.json")

    # Scaler fitted on training only
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, selected].fillna(0))
    joblib.dump(scaler, SCALERS_DIR / "feature_scaler.pkl")
    print(f"   ✅ Feature scaler saved")

    return selected


def save_engineered(df: pd.DataFrame):
    out = DATA_DIR / "engineered_data.csv"
    df.to_csv(out, index=False)
    print(f"\n💾 Saved → {out}  |  {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        df, eoq_df = load_data()

        df = add_lag_features(df)
        df = add_eoq_features(df, eoq_df)
        df = add_price_features(df)
        df = clean_engineered(df)
        selected = feature_selection(df)
        save_engineered(df)

        n_train = (df["date"] < SPLIT_DATE).sum()
        n_test  = (df["date"] >= SPLIT_DATE).sum()
        print(f"\n   📊 Split: Train={n_train:,} | Test={n_test:,}")
        print(f"   📋 Features: {len(selected)}")
        print(f"\n✅ Script 05 complete.")

    except Exception as e:
        print(f"\n❌ Script 05 FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise