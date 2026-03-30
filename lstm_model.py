# ══════════════════════════════════════════════════════════════════════════════
# lstm_model.py — Script 06b
# LSTM Demand Forecasting — Train 2019-2022, Predict 2023
# Dual-input: sequential branch (past 14 days) + static branch (target-day)
# Saves predictions in same format as ML models for seamless comparison
# ══════════════════════════════════════════════════════════════════════════════
# REQUIRES: pip install tensorflow
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

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
SCALERS_DIR   = ARTIFACTS_DIR / "scalers"
PLOTS_DIR     = ARTIFACTS_DIR / "plots"

for d in [MODELS_DIR, SCALERS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SPLIT_DATE         = pd.Timestamp("2023-01-01")
SEQ_LEN            = 14
BATCH_SIZE         = 512
EPOCHS             = 20
PATIENCE           = 3
MAX_TRAIN_SAMPLES  = 300_000
HOLDING_COST_RATE  = 0.25
ORDERING_COST      = 50.0

# Features used INSIDE each timestep of the sequence (past 14 days).
SEQ_FEATURES = [
    "sales_scaled",
    "price_scaled",
    "promo",
    "weekday_sin", "weekday_cos",
    "month_sin",   "month_cos",
    "is_weekend",
]

# Features for the TARGET day (known at prediction time).
STATIC_FEATURES = [
    "price_scaled",
    "promo",
    "weekday_sin", "weekday_cos",
    "month_sin",   "month_cos",
    "is_weekend",
    "is_month_start", "is_month_end",
    "store_id_enc",   "item_id_enc",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    path = DATA_DIR / "processed_data.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"📂 Loaded {len(df):,} rows  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Prepare Sequences
# ══════════════════════════════════════════════════════════════════════════════
def prepare_data(df: pd.DataFrame):
    """
    For every store-item pair create sliding windows of SEQ_LEN days.
    Subsamples training data to MAX_TRAIN_SAMPLES for speed.
    """
    print("\n⚙️  Preparing LSTM sequences...")
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # ── Scale sales (fit on training period ONLY) ─────────────────────────
    train_mask   = df["date"] < SPLIT_DATE
    sales_scaler = MinMaxScaler(feature_range=(0, 1))
    sales_scaler.fit(df.loc[train_mask, ["sales"]])
    df["sales_scaled"] = sales_scaler.transform(df[["sales"]])
    joblib.dump(sales_scaler, SCALERS_DIR / "lstm_sales_scaler.pkl")

    # Verify columns exist
    needed = set(SEQ_FEATURES + STATIC_FEATURES)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in processed_data.csv: {missing}")

    # ── Build arrays ──────────────────────────────────────────────────────
    X_seq_train, X_static_train, y_train = [], [], []
    X_seq_test,  X_static_test,  y_test  = [], [], []
    test_meta = []

    pairs = list(df.groupby(["store_id", "item_id"]))
    print(f"   Processing {len(pairs)} store-item pairs "
          f"(SEQ_LEN={SEQ_LEN})...")

    for (store_id, item_id), grp in pairs:
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < SEQ_LEN + 1:
            continue

        seq_vals    = grp[SEQ_FEATURES].values.astype(np.float32)
        static_vals = grp[STATIC_FEATURES].values.astype(np.float32)
        sales_raw   = grp["sales"].values
        sales_sc    = grp["sales_scaled"].values.astype(np.float32)
        dates       = grp["date"].values

        for i in range(SEQ_LEN, len(grp)):
            x_seq    = seq_vals[i - SEQ_LEN : i]
            x_static = static_vals[i]
            y_val    = sales_sc[i]
            t_date   = dates[i]

            if pd.Timestamp(t_date) < SPLIT_DATE:
                X_seq_train.append(x_seq)
                X_static_train.append(x_static)
                y_train.append(y_val)
            else:
                X_seq_test.append(x_seq)
                X_static_test.append(x_static)
                y_test.append(y_val)
                test_meta.append({
                    "date":     t_date,
                    "store_id": store_id,
                    "item_id":  item_id,
                    "actual":   int(sales_raw[i]),
                })

    X_seq_train    = np.array(X_seq_train)
    X_static_train = np.array(X_static_train)
    y_train        = np.array(y_train)
    X_seq_test     = np.array(X_seq_test)
    X_static_test  = np.array(X_static_test)
    y_test         = np.array(y_test)

    # ── Subsample training data for speed ─────────────────────────────────
    if len(X_seq_train) > MAX_TRAIN_SAMPLES:
        print(f"\n   ⚡ Subsampling training: {len(X_seq_train):,} → "
              f"{MAX_TRAIN_SAMPLES:,}")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_seq_train), MAX_TRAIN_SAMPLES, replace=False)
        idx.sort()
        X_seq_train    = X_seq_train[idx]
        X_static_train = X_static_train[idx]
        y_train        = y_train[idx]

    print(f"\n   Train : {X_seq_train.shape[0]:,} samples   "
          f"seq {X_seq_train.shape}  static {X_static_train.shape}")
    print(f"   Test  : {X_seq_test.shape[0]:,} samples   "
          f"seq {X_seq_test.shape}  static {X_static_test.shape}")

    return (X_seq_train, X_static_train, y_train,
            X_seq_test,  X_static_test,  y_test,
            test_meta, sales_scaler)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model Architecture
# ══════════════════════════════════════════════════════════════════════════════
def build_model(seq_shape: tuple, static_dim: int) -> Model:
    """
    Lighter two-input architecture for faster training:
        Sequence  →  LSTM(64)  ─┐
                                ├─ Dense(32) → 1
        Static    →  Dense(16) ─┘
    """
    # ── Sequence branch (single LSTM layer) ───────────────────────────────
    seq_in = Input(shape=seq_shape, name="seq_input")
    x = LSTM(64, return_sequences=False)(seq_in)
    x = Dropout(0.2)(x)

    # ── Static branch ────────────────────────────────────────────────────
    static_in = Input(shape=(static_dim,), name="static_input")
    s = Dense(16, activation="relu")(static_in)
    s = BatchNormalization()(s)

    # ── Merge + head ──────────────────────────────────────────────────────
    merged = Concatenate()([x, s])
    merged = Dense(32, activation="relu")(merged)
    merged = Dropout(0.15)(merged)
    out    = Dense(1, name="output")(merged)

    model = Model(inputs=[seq_in, static_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. Train
# ══════════════════════════════════════════════════════════════════════════════
def train_lstm(X_seq_tr, X_stat_tr, y_tr,
               X_seq_te, X_stat_te, y_te):
    print("\n🧠 Building & training LSTM...")

    model = build_model(
        seq_shape  = (X_seq_tr.shape[1], X_seq_tr.shape[2]),
        static_dim = X_stat_tr.shape[1],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=2, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        [X_seq_tr, X_stat_tr], y_tr,
        validation_data=([X_seq_te, X_stat_te], y_te),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(MODELS_DIR / "lstm_model.keras"))
    print("   ✅ Model saved → lstm_model.keras")
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 5. Evaluate, Save & Plot
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_and_save(model, X_seq_te, X_stat_te, y_te,
                      test_meta, sales_scaler, history):
    print("\n📊 Evaluating LSTM on 2023 holdout...")

    # ── Predict & inverse-scale ───────────────────────────────────────────
    y_pred_sc = model.predict([X_seq_te, X_stat_te],
                              batch_size=BATCH_SIZE).flatten()
    y_pred = sales_scaler.inverse_transform(
        y_pred_sc.reshape(-1, 1)
    ).flatten()
    y_pred = np.maximum(y_pred, 0).round(2)

    actual = np.array([m["actual"] for m in test_meta], dtype=float)

    mae  = mean_absolute_error(actual, y_pred)
    rmse = np.sqrt(mean_squared_error(actual, y_pred))
    r2   = r2_score(actual, y_pred)

    print(f"\n   {'LSTM':<15} │ MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}"
          f"  ({len(actual):,} observations)")

    metrics = {"model": "LSTM",
               "MAE": round(mae, 4), "RMSE": round(rmse, 4),
               "R2": round(r2, 4)}

    # ── Build result DataFrame ────────────────────────────────────────────
    result = pd.DataFrame(test_meta)
    result["LSTM"] = y_pred
    result["date"] = pd.to_datetime(result["date"])
    result.to_csv(ARTIFACTS_DIR / "lstm_predictions_2023.csv", index=False)
    print(f"   ✅ lstm_predictions_2023.csv  ({len(result):,} rows)")

    # ── Update model_metrics.csv ──────────────────────────────────────────
    _update_metrics_csv(metrics)

    # ── Merge LSTM column into the ML prediction files ────────────────────
    _merge_into_full(result)
    _merge_into_daily(result)

    # ── Update best_model.json if LSTM wins ───────────────────────────────
    _update_best_model(metrics)

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_training_history(history)
    _plot_lstm_vs_actual(result)
    _plot_lstm_residuals(result)

    return metrics, result


# ── helper: update model_metrics.csv ──────────────────────────────────────────
def _update_metrics_csv(metrics):
    path = ARTIFACTS_DIR / "model_metrics.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["model"] != "LSTM"]
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    print("   ✅ model_metrics.csv updated")


# ── helper: merge LSTM into ml_predictions_2023_full.csv ──────────────────────
def _merge_into_full(result):
    path = ARTIFACTS_DIR / "ml_predictions_2023_full.csv"
    if not path.exists():
        print("   ⚠️  ml_predictions_2023_full.csv not found — "
              "run Script 06 first to create it.  Skipping merge.")
        return
    full = pd.read_csv(path, parse_dates=["date"])
    if "LSTM" in full.columns:
        full.drop(columns=["LSTM"], inplace=True)
    merged = full.merge(
        result[["date", "store_id", "item_id", "LSTM"]],
        on=["date", "store_id", "item_id"],
        how="left",
    )
    merged.to_csv(path, index=False)
    print("   ✅ ml_predictions_2023_full.csv updated with LSTM")


# ── helper: merge LSTM into ml_daily_predictions_2023.csv ─────────────────────
def _merge_into_daily(result):
    daily_lstm = (result.groupby("date")
                  .agg(actual=("actual", "sum"), LSTM=("LSTM", "sum"))
                  .reset_index())
    path = ARTIFACTS_DIR / "ml_daily_predictions_2023.csv"
    if not path.exists():
        daily_lstm.to_csv(path, index=False)
        print("   ✅ ml_daily_predictions_2023.csv created")
        return
    daily = pd.read_csv(path, parse_dates=["date"])
    if "LSTM" in daily.columns:
        daily.drop(columns=["LSTM"], inplace=True)
    merged = daily.merge(daily_lstm[["date", "LSTM"]], on="date", how="left")
    merged.to_csv(path, index=False)
    print("   ✅ ml_daily_predictions_2023.csv updated with LSTM")


# ── helper: update best_model.json ────────────────────────────────────────────
def _update_best_model(metrics):
    path = ARTIFACTS_DIR / "best_model.json"
    if path.exists():
        with open(path) as f:
            current = json.load(f)
        if metrics["MAE"] < current["metrics"]["MAE"]:
            with open(path, "w") as f:
                json.dump({"best_model": "LSTM", "metrics": metrics}, f, indent=2)
            print("   🏆 LSTM is now the best model!")
        else:
            print(f"   Current best: {current['best_model']} "
                  f"(MAE={current['metrics']['MAE']:.4f})")
    else:
        with open(path, "w") as f:
            json.dump({"best_model": "LSTM", "metrics": metrics}, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Inventory Optimization (same logic as Script 06, using LSTM predictions)
# ══════════════════════════════════════════════════════════════════════════════
def lstm_inventory_optimization(result_df):
    eoq_path = ARTIFACTS_DIR / "eoq_analysis.csv"
    if not eoq_path.exists():
        print("   ⚠️  eoq_analysis.csv not found — skipping optimization")
        return None

    eoq_df = pd.read_csv(eoq_path)
    print(f"\n📦 Inventory Optimization (LSTM)...")

    results = []
    for item_id, grp in result_df.groupby("item_id"):
        row = eoq_df[eoq_df["item_id"] == item_id]
        if row.empty:
            continue
        row = row.iloc[0]

        avg_price      = row["avg_unit_price"]
        lstm_avg_daily = grp["LSTM"].mean()
        lstm_std_daily = grp["LSTM"].std()

        annual_demand = lstm_avg_daily * 365
        H             = avg_price * HOLDING_COST_RATE
        lstm_eoq      = np.sqrt(2 * annual_demand * ORDERING_COST / (H + 1e-9))
        lstm_safety   = 1.645 * lstm_std_daily * np.sqrt(7)
        lstm_rop      = lstm_avg_daily * 7 + lstm_safety
        lstm_cost     = ((annual_demand / (lstm_eoq + 1e-9)) * ORDERING_COST
                         + (lstm_eoq / 2) * H)

        trad_cost = row["Total_Annual_Cost"]
        saving    = trad_cost - lstm_cost

        results.append({
            "item_id":        item_id,
            "trad_EOQ":       row["EOQ"],
            "lstm_EOQ":       round(lstm_eoq, 2),
            "trad_ROP":       row["Reorder_Point"],
            "lstm_ROP":       round(lstm_rop, 2),
            "trad_Safety":    row["Safety_Stock"],
            "lstm_Safety":    round(lstm_safety, 2),
            "trad_TotalCost": round(trad_cost, 2),
            "lstm_TotalCost": round(lstm_cost, 2),
            "cost_saving":    round(saving, 2),
            "pct_saving":     round(saving / (trad_cost + 1e-9) * 100, 2),
        })

    opt_df = pd.DataFrame(results)
    opt_df.to_csv(ARTIFACTS_DIR / "ml_optimization_LSTM.csv", index=False)

    total_trad = opt_df["trad_TotalCost"].sum()
    total_lstm = opt_df["lstm_TotalCost"].sum()
    total_save = opt_df["cost_saving"].sum()
    print(f"   Traditional   : £{total_trad:,.2f}")
    print(f"   LSTM-optimised: £{total_lstm:,.2f}")
    print(f"   Savings       : £{total_save:,.2f} "
          f"({total_save / (total_trad + 1e-9) * 100:.1f}%)")
    return opt_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. Plots
# ══════════════════════════════════════════════════════════════════════════════
def _plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"],     label="Train", color="#2E86AB", lw=2)
    axes[0].plot(history.history["val_loss"], label="Val",   color="#C73E1D", lw=2)
    axes[0].set_title("LSTM — Loss (MSE)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE"); axes[0].legend()

    axes[1].plot(history.history["mae"],     label="Train", color="#2E86AB", lw=2)
    axes[1].plot(history.history["val_mae"], label="Val",   color="#C73E1D", lw=2)
    axes[1].set_title("LSTM — MAE (scaled)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "22_lstm_training_history.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 22_lstm_training_history.png")


def _plot_lstm_vs_actual(result):
    daily = result.groupby("date")[["actual", "LSTM"]].sum().reset_index()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(daily["date"], daily["actual"],
            label="Actual", color="black", lw=2, alpha=0.9)
    ax.plot(daily["date"], daily["LSTM"],
            label="LSTM",   color="#9b59b6", lw=1.5, ls="--", alpha=0.85)
    ax.set_title("LSTM vs Actual — 2023 (Aggregated Daily Sales)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Daily Sales")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "23_lstm_vs_actual_2023.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 23_lstm_vs_actual_2023.png")


def _plot_lstm_residuals(result):
    daily = result.groupby("date")[["actual", "LSTM"]].sum().reset_index()
    daily["residual"] = daily["actual"] - daily["LSTM"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(daily["LSTM"], daily["residual"],
                    alpha=0.4, color="#9b59b6", s=15)
    axes[0].axhline(0, color="red", lw=1.5)
    axes[0].set_title("Residual vs Predicted", fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")

    axes[1].hist(daily["residual"], bins=40, color="#9b59b6",
                 edgecolor="white", alpha=0.8)
    axes[1].set_title("Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")

    axes[2].plot(daily["date"], daily["residual"], color="#9b59b6", lw=1.2)
    axes[2].axhline(0, color="black", ls="--", lw=1)
    axes[2].set_title("Residual Over Time", fontweight="bold")
    axes[2].set_xlabel("Date")

    plt.suptitle("LSTM — Residual Analysis (Aggregated Daily)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "24_lstm_residuals.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 24_lstm_residuals.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("   LSTM MODEL — Train: 2019-2022 | Test: 2023")
    print("=" * 60)

    df = load_data()

    (X_seq_tr, X_stat_tr, y_tr,
     X_seq_te, X_stat_te, y_te,
     test_meta, sales_scaler) = prepare_data(df)

    model, history = train_lstm(
        X_seq_tr, X_stat_tr, y_tr,
        X_seq_te, X_stat_te, y_te,
    )

    metrics, result = evaluate_and_save(
        model, X_seq_te, X_stat_te, y_te,
        test_meta, sales_scaler, history,
    )

    lstm_inventory_optimization(result)

    print(f"\n✅ LSTM Script complete.")
    print(f"   MAE  = {metrics['MAE']:.4f}")
    print(f"   RMSE = {metrics['RMSE']:.4f}")
    print(f"   R²   = {metrics['R2']:.4f}")