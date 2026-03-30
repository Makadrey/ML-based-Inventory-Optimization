# ══════════════════════════════════════════════════════════════════════════════
# preprocessing.py — Script 02
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
# from load_data import load_data

BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SCALERS_DIR   = ARTIFACTS_DIR / "scalers"
ENCODERS_DIR  = ARTIFACTS_DIR / "encoders"

for d in [DATA_DIR, ARTIFACTS_DIR, SCALERS_DIR, ENCODERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_raw(path=None):
    if path is None:
        for name in ["retail_sales.csv", "retail_demand.csv", "train.csv"]:
            candidate = DATA_DIR / name
            if candidate.exists():
                path = candidate
                break
    if path is None:
        raise FileNotFoundError("No dataset found in data/. Run script 01 first.")
    df = pd.read_csv(path)
    print(f"📂 Loaded {len(df):,} rows from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, types, duplicates, and outliers."""
    print("\n🧹 Cleaning data...")
    initial_rows = len(df)

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates()
    print(f"   Removed {initial_rows - len(df)} duplicate rows")

    # ── Missing values ──────────────────────────────────────────────────────
    if df.isnull().sum().sum() > 0:
        print(f"   Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        for col in ["sales", "price"]:
            if col in df.columns:
                df[col] = df.groupby("item_id")[col].transform(
                    lambda x: x.fillna(x.median())
                )
        df["promo"]   = df["promo"].fillna(0).astype(int)
        df["weekday"] = df["weekday"].fillna(df["date"].dt.dayofweek)
        df["month"]   = df["month"].fillna(df["date"].dt.month)
    else:
        print("   ✅ No missing values found")

    # ── Enforce types ────────────────────────────────────────────────────────
    df["sales"]   = df["sales"].clip(lower=0).astype(int)
    df["price"]   = df["price"].astype(float).clip(lower=0)
    df["promo"]   = df["promo"].astype(int)
    df["weekday"] = df["weekday"].astype(int)
    df["month"]   = df["month"].astype(int)

    # ── Outlier capping (IQR, per item) ─────────────────────────────────────
    print("   Handling sales outliers (IQR × 3 capping per item)...")
    def cap_outliers(grp):
        q1, q3  = grp["sales"].quantile([0.25, 0.75])
        iqr     = q3 - q1
        upper   = q3 + 3 * iqr
        grp["sales"] = grp["sales"].clip(upper=upper).astype(int)
        return grp

    df = df.groupby("item_id", group_keys=False).apply(cap_outliers)
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    print(f"   ✅ Clean shape: {df.shape}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode store_id and item_id and save encoders."""
    print("\n🔤 Encoding categoricals...")

    for col in ["store_id", "item_id"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])
        joblib.dump(le, ENCODERS_DIR / f"le_{col}.pkl")
        print(f"   {col}: {len(le.classes_)} classes → encoder saved")

    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract rich temporal features from date."""
    print("\n📅 Extracting datetime features...")

    df["year"]           = df["date"].dt.year
    df["quarter"]        = df["date"].dt.quarter
    df["week"]           = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_year"]    = df["date"].dt.dayofyear
    df["is_weekend"]     = (df["weekday"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

    # Cyclical encoding
    df["month_sin"]   = np.sin(2 * np.pi * df["month"]   / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"]   / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["week_sin"]    = np.sin(2 * np.pi * df["week"]    / 52)
    df["week_cos"]    = np.cos(2 * np.pi * df["week"]    / 52)

    print(f"   ✅ Datetime features added  (total cols: {df.shape[1]})")
    return df


def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Scale price; leave sales unscaled (target stays interpretable)."""
    print("\n⚖️  Scaling numeric features...")

    price_scaler = StandardScaler()
    df["price_scaled"] = price_scaler.fit_transform(df[["price"]])
    joblib.dump(price_scaler, SCALERS_DIR / "price_scaler.pkl")

    sales_scaler = MinMaxScaler(feature_range=(0, 1))
    sales_scaler.fit(df[["sales"]])
    joblib.dump(sales_scaler, SCALERS_DIR / "sales_scaler.pkl")

    print("   ✅ Scalers saved to artifacts/scalers/")
    return df


def save_processed(df: pd.DataFrame) -> Path:
    out_path = DATA_DIR / "processed_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\n💾 Processed data saved → {out_path}  |  Shape: {df.shape}")
    return out_path


if __name__ == "__main__":
    df = load_raw()
    # df = load_data()
    df = clean_data(df)
    df = encode_categoricals(df)
    df = add_datetime_features(df)
    df = scale_numeric(df)
    save_processed(df)
    print("\n✅ Script 02 complete.")