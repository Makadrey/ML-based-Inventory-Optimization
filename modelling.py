# ══════════════════════════════════════════════════════════════════════════════
# modelling.py — Script 06
# HistGBM / XGBoost / LightGBM — Train 2019-2022, Predict 2023
# Saves row-level predictions for per-store-item analysis
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
PLOTS_DIR     = ARTIFACTS_DIR / "plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DATE = pd.Timestamp("2023-01-01")

HOLDING_COST_RATE  = 0.25
ORDERING_COST      = 50.0

ML_MODELS = ["HistGBM", "XGBoost", "LightGBM"]

PALETTE_ML = {
    "HistGBM":  "#2E86AB",
    "XGBoost":  "#1B4F72",
    "LightGBM": "#44BBA4",
}


# ══════════════════════════════════════════════════════════════════════════════
# Load
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_csv(DATA_DIR / "engineered_data.csv", parse_dates=["date"])
    with open(ARTIFACTS_DIR / "selected_features.json") as f:
        features = json.load(f)
    eoq_df = None
    if (ARTIFACTS_DIR / "eoq_analysis.csv").exists():
        eoq_df = pd.read_csv(ARTIFACTS_DIR / "eoq_analysis.csv")
    print(f"📂 Loaded {len(df):,} rows  |  {len(features)} features")
    return df, features, eoq_df


# ══════════════════════════════════════════════════════════════════════════════
# Temporal Split
# ══════════════════════════════════════════════════════════════════════════════
def time_split(df: pd.DataFrame, features: list):
    df = df.sort_values("date").copy()
    train = df[df["date"] <  SPLIT_DATE].copy()
    test  = df[df["date"] >= SPLIT_DATE].copy()

    avail = [f for f in features if f in df.columns]

    X_train = train[avail].fillna(0)
    X_test  = test[avail].fillna(0)
    y_train = train["sales"]
    y_test  = test["sales"]

    print(f"\n  🔀 Temporal Split:")
    print(f"     Train: {train['date'].min().date()} → "
          f"{train['date'].max().date()}  ({len(train):,} rows)")
    print(f"     Test:  {test['date'].min().date()} → "
          f"{test['date'].max().date()}  ({len(test):,} rows)")
    print(f"     Features: {len(avail)}")

    return X_train, X_test, y_train, y_test, train, test, avail


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation — overall MAE / RMSE / R² on FULL test set (all stores × items)
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(y_true, y_pred, model_name=""):
    y_true = np.array(y_true)
    y_pred = np.maximum(np.array(y_pred), 0)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"   {model_name:<15} │ MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}"
          f"  (computed on {len(y_true):,} observations)")
    return {"model": model_name,
            "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════
def train_histgbm(X_train, y_train, X_test, y_test):
    print("\n  📊 Training HistGradientBoosting...")
    model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )
    model.fit(X_train, y_train)
    pred    = np.maximum(model.predict(X_test), 0)
    metrics = evaluate(y_test, pred, "HistGBM")
    joblib.dump(model, MODELS_DIR / "histgbm.pkl")
    return model, pred, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n  🚀 Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
    pred    = np.maximum(model.predict(X_test), 0)
    metrics = evaluate(y_test, pred, "XGBoost")
    model.save_model(str(MODELS_DIR / "xgboost.json"))
    return model, pred, metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    print("\n  💡 Training LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
    pred    = np.maximum(model.predict(X_test), 0)
    metrics = evaluate(y_test, pred, "LightGBM")
    joblib.dump(model, MODELS_DIR / "lightgbm.pkl")
    return model, pred, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Save Predictions (row-level AND aggregated daily)
# ══════════════════════════════════════════════════════════════════════════════
def save_predictions(test_df, all_preds):
    # ── Row-level (for per-store-item analysis in the app) ────────────────
    result = test_df[["date", "store_id", "item_id", "sales"]].copy()
    result = result.rename(columns={"sales": "actual"})
    for name, pred in all_preds.items():
        result[name] = np.round(pred, 2)
    result.to_csv(ARTIFACTS_DIR / "ml_predictions_2023_full.csv", index=False)
    print(f"\n   ✅ ml_predictions_2023_full.csv  ({len(result):,} rows)")

    # ── Aggregated daily (for overall plots & comparison with trad) ───────
    model_cols = list(all_preds.keys())
    daily = result.groupby("date")[["actual"] + model_cols].sum().reset_index()
    daily.to_csv(ARTIFACTS_DIR / "ml_daily_predictions_2023.csv", index=False)
    print(f"   ✅ ml_daily_predictions_2023.csv  ({len(daily)} days)")


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════
def plot_ml_predictions_2023(test_df, all_preds):
    daily_actual = test_df.groupby("date")["sales"].sum()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(daily_actual.index, daily_actual.values,
            label="Actual", color="black", lw=2, alpha=0.9)

    for name, pred in all_preds.items():
        tmp = test_df[["date"]].copy()
        tmp["pred"] = pred
        dp = tmp.groupby("date")["pred"].sum()
        ax.plot(dp.index, dp.values, label=name,
                color=PALETTE_ML.get(name, "grey"), lw=1.5, ls="--", alpha=0.8)

    ax.set_title("ML Models vs Actual — 2023 (Aggregated Daily Sales)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Daily Sales")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_ml_vs_actual_2023.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 13_ml_vs_actual_2023.png")


def plot_model_comparison_bars(all_metrics):
    mdf = pd.DataFrame(all_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [PALETTE_ML.get(m, "#999") for m in mdf["model"]]

    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        bars = ax.bar(mdf["model"], mdf[metric],
                      color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, mdf[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)

    plt.suptitle("ML Model Comparison — 2023 (overall, all stores × items)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "14_ml_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 14_ml_model_comparison.png")


def plot_residuals(y_test, best_pred, best_name, test_df):
    residuals = np.array(y_test) - np.array(best_pred[:len(y_test)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(best_pred[:len(y_test)], residuals,
                    alpha=0.05, color="#2E86AB", s=5)
    axes[0].axhline(0, color="red", lw=1.5)
    axes[0].set_title(f"{best_name} — Residual vs Predicted",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=60, color="#2E86AB", edgecolor="white", alpha=0.8)
    axes[1].set_title(f"{best_name} — Residual Distribution",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")

    tmp = test_df[["date"]].copy()
    tmp["residual"] = residuals
    daily_res = tmp.groupby("date")["residual"].mean()
    axes[2].plot(daily_res.index, daily_res.values, color="#C73E1D", lw=1.2)
    axes[2].axhline(0, color="black", ls="--", lw=1)
    axes[2].set_title(f"{best_name} — Mean Residual Over Time",
                      fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "15_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 15_residuals.png")


def plot_sample_store_items(test_df, all_preds, best_name):
    """2×3 grid of the 6 highest-volume store-item combos — best model only."""
    best_pred = all_preds[best_name]
    tmp = test_df[["date", "store_id", "item_id", "sales"]].copy()
    tmp["pred"] = best_pred

    totals = tmp.groupby(["store_id", "item_id"])["sales"].sum()
    top6   = totals.nlargest(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for idx, (store, item) in enumerate(top6):
        ax  = axes.flatten()[idx]
        sub = tmp[(tmp["store_id"] == store) &
                  (tmp["item_id"] == item)].sort_values("date")
        ax.plot(sub["date"], sub["sales"], label="Actual",
                color="black", lw=1.5)
        ax.plot(sub["date"], sub["pred"],  label=best_name,
                color=PALETTE_ML[best_name], lw=1.2, ls="--", alpha=0.8)
        mae = mean_absolute_error(sub["sales"], sub["pred"])
        ax.set_title(f"{item} @ {store}  (MAE={mae:.1f})",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    plt.suptitle(f"Sample Store-Item Forecasts ({best_name}) — 2023",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "16_sample_store_item.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 16_sample_store_item.png")


# ══════════════════════════════════════════════════════════════════════════════
# Inventory Optimization
# ══════════════════════════════════════════════════════════════════════════════
def ml_inventory_optimization(test_df, predictions, eoq_df, model_name):
    if eoq_df is None:
        return None

    print(f"\n📦 Inventory Optimization  ({model_name})...")
    test_df = test_df.copy()
    test_df["ml_pred"] = predictions[:len(test_df)]

    results = []
    for item_id, grp in test_df.groupby("item_id"):
        row = eoq_df[eoq_df["item_id"] == item_id]
        if row.empty:
            continue
        row = row.iloc[0]

        avg_price    = row["avg_unit_price"]
        ml_avg_daily = grp["ml_pred"].mean()
        ml_std_daily = grp["ml_pred"].std()

        annual_demand = ml_avg_daily * 365
        H          = avg_price * HOLDING_COST_RATE
        ml_eoq     = np.sqrt(2 * annual_demand * ORDERING_COST / (H + 1e-9))
        ml_safety  = 1.645 * ml_std_daily * np.sqrt(7)
        ml_rop     = ml_avg_daily * 7 + ml_safety
        ml_cost    = ((annual_demand / (ml_eoq + 1e-9)) * ORDERING_COST
                      + (ml_eoq / 2) * H)

        trad_cost = row["Total_Annual_Cost"]
        saving    = trad_cost - ml_cost

        results.append({
            "item_id":        item_id,
            "trad_EOQ":       row["EOQ"],
            "ml_EOQ":         round(ml_eoq, 2),
            "trad_ROP":       row["Reorder_Point"],
            "ml_ROP":         round(ml_rop, 2),
            "trad_Safety":    row["Safety_Stock"],
            "ml_Safety":      round(ml_safety, 2),
            "trad_TotalCost": round(trad_cost, 2),
            "ml_TotalCost":   round(ml_cost, 2),
            "cost_saving":    round(saving, 2),
            "pct_saving":     round(saving / (trad_cost + 1e-9) * 100, 2),
        })

    opt_df = pd.DataFrame(results)
    opt_df.to_csv(ARTIFACTS_DIR / f"ml_optimization_{model_name}.csv", index=False)
    total_save = opt_df["cost_saving"].sum()
    total_trad = opt_df["trad_TotalCost"].sum()
    print(f"   Traditional : £{total_trad:,.2f}")
    print(f"   ML-optimised: £{opt_df['ml_TotalCost'].sum():,.2f}")
    print(f"   Savings     : £{total_save:,.2f} "
          f"({total_save / total_trad * 100:.1f}%)")
    return opt_df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df, features, eoq_df = load_data()

    print("\n" + "=" * 60)
    print("   MODEL TRAINING — Train: 2019-2022 | Test: 2023")
    print("=" * 60)

    X_train, X_test, y_train, y_test, train_df, test_df, used_feats = \
        time_split(df, features)

    all_preds   = {}
    all_metrics = []

    hgb, hgb_pred, hgb_m = train_histgbm(X_train, y_train, X_test, y_test)
    all_preds["HistGBM"] = hgb_pred;  all_metrics.append(hgb_m)

    xm, xp, xmet = train_xgboost(X_train, y_train, X_test, y_test)
    all_preds["XGBoost"] = xp;  all_metrics.append(xmet)

    lm, lp, lmet = train_lightgbm(X_train, y_train, X_test, y_test)
    all_preds["LightGBM"] = lp;  all_metrics.append(lmet)

    # ── Best model ────────────────────────────────────────────────────────
    best_m    = min(all_metrics, key=lambda x: x["MAE"])
    best_name = best_m["model"]
    best_pred = all_preds[best_name]
    print(f"\n  🏆 Best: {best_name}  MAE={best_m['MAE']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    pd.DataFrame(all_metrics).to_csv(
        ARTIFACTS_DIR / "model_metrics.csv", index=False)
    with open(ARTIFACTS_DIR / "best_model.json", "w") as f:
        json.dump({"best_model": best_name, "metrics": best_m}, f, indent=2)

    save_predictions(test_df, all_preds)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_ml_predictions_2023(test_df, all_preds)
    plot_model_comparison_bars(all_metrics)
    plot_residuals(y_test, best_pred, best_name, test_df)
    plot_sample_store_items(test_df, all_preds, best_name)

    # ── Inventory optimisation ────────────────────────────────────────────
    ml_inventory_optimization(test_df, best_pred, eoq_df, best_name)

    print(f"\n✅ Script 06 complete.")