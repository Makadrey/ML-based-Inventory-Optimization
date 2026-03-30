# ══════════════════════════════════════════════════════════════════════════════
# load_data.py — Script 01
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

print(f"📁 Base Directory     : {BASE_DIR}")
print(f"📂 Data Directory     : {DATA_DIR}")
print(f"🧱 Artifacts Directory: {ARTIFACTS_DIR}")


def load_data(filepath=None):
    """Load retail demand dataset. Falls back to synthetic data if not found."""
    if filepath is None:
        for name in ["retail_sales.csv", "train.csv", "data.csv", "sales.csv", "retail_demand.csv"]:
            candidate = DATA_DIR / name
            if candidate.exists():
                filepath = candidate
                break

    if filepath is None or not Path(filepath).exists():
        print("⚠️  Dataset not found — generating synthetic data for testing...")
        df = _generate_synthetic_data()
        save_path = DATA_DIR / "retail_demand.csv"
        df.to_csv(save_path, index=False)
        print(f"✅ Synthetic data saved → {save_path}")
        return df

    print(f"📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} rows")
    return df


def _generate_synthetic_data():
    """
    Generate realistic synthetic retail demand data.
    Schema: date, store_id, item_id, sales, price, promo, weekday, month
    10 stores × 20 items × 5 years (2019-2023)
    """
    np.random.seed(42)
    dates  = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    stores = [f"store_{i}" for i in range(1, 11)]
    items  = [f"item_{i}"  for i in range(1, 21)]

    item_base_prices = {item: round(np.random.uniform(8, 80), 2) for item in items}

    records = []
    for date in dates:
        weekday = date.dayofweek
        month   = date.month
        for store in stores:
            for item in items:
                base_price = item_base_prices[item]
                promo      = int(np.random.random() < 0.10)
                price      = round(base_price * (0.80 if promo else 1.0), 2)

                base        = np.random.uniform(20, 50)
                seasonal    = 8 * np.sin(2 * np.pi * month / 12)
                weekend     = 12 if weekday >= 5 else 0
                promo_lift  = 18 if promo else 0
                trend       = (date.year - 2019) * 0.3
                noise       = np.random.normal(0, 5)
                sales       = max(0, int(base + seasonal + weekend + promo_lift + trend + noise))

                records.append({
                    "date":     date.strftime("%Y-%m-%d"),
                    "store_id": store,
                    "item_id":  item,
                    "sales":    sales,
                    "price":    price,
                    "promo":    promo,
                    "weekday":  weekday,
                    "month":    month,
                })

    return pd.DataFrame(records)


def initial_inspection(df: pd.DataFrame) -> pd.DataFrame:
    """Print a comprehensive data report and save summary JSON."""
    print("\n" + "=" * 60)
    print("   DATA LOADING REPORT")
    print("=" * 60)

    df["date"] = pd.to_datetime(df["date"])

    print(f"\n📊 Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"📋 Columns        : {list(df.columns)}")
    print(f"\n🔍 Data Types:\n{df.dtypes}")
    print(f"\n🔍 Missing Values:\n{df.isnull().sum()}")
    print(f"\n📈 Numeric Summary:\n{df.describe().round(3)}")

    n_stores   = df["store_id"].nunique() if "store_id" in df.columns else "—"
    n_items    = df["item_id"].nunique()  if "item_id"  in df.columns else "—"
    date_range = (df["date"].max() - df["date"].min()).days

    print(f"\n🏪 Unique Stores  : {n_stores}")
    print(f"📦 Unique Items   : {n_items}")
    print(f"📅 Date Range     : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"⏱️  Total Days     : {date_range}")

    if "sales" in df.columns:
        print(f"\n💰 Sales Stats:")
        print(f"   Mean  : {df['sales'].mean():.2f}")
        print(f"   Std   : {df['sales'].std():.2f}")
        print(f"   Min   : {df['sales'].min()}")
        print(f"   Max   : {df['sales'].max()}")

    if "promo" in df.columns:
        print(f"\n🎯 Promotion Rate : {df['promo'].mean()*100:.1f}%")

    summary = {
        "rows":          int(df.shape[0]),
        "columns":       int(df.shape[1]),
        "column_names":  list(df.columns),
        "unique_stores": int(n_stores) if isinstance(n_stores, (int, np.integer)) else None,
        "unique_items":  int(n_items)  if isinstance(n_items,  (int, np.integer)) else None,
        "date_min":      str(df["date"].min().date()),
        "date_max":      str(df["date"].max().date()),
        "sales_mean":    float(df["sales"].mean()) if "sales" in df.columns else None,
        "sales_std":     float(df["sales"].std())  if "sales" in df.columns else None,
        "missing":       {k: int(v) for k, v in df.isnull().sum().items()},
    }
    out_path = ARTIFACTS_DIR / "data_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary saved → {out_path}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_data()
    df = initial_inspection(df)
    print("\n✅ Script 01 complete. Data ready for preprocessing.")
