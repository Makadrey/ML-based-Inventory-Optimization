# ══════════════════════════════════════════════════════════════════════════════
# analysis.py — Script 04  (EOQ + Traditional Baselines on 2023 holdout)
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
PLOTS_DIR     = BASE_DIR / "artifacts" / "plots"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# ─── Constants ────────────────────────────────────────────────────────────────
HOLDING_COST_RATE  = 0.25
ORDERING_COST      = 50.0
STOCKOUT_COST_RATE = 1.5
LEAD_TIME_DAYS     = 7
SERVICE_LEVEL      = 0.95
Z_SCORE            = 1.645

SPLIT_DATE = pd.Timestamp("2023-01-01")


def load_data():
    df = pd.read_csv(DATA_DIR / "processed_data.csv", parse_dates=["date"])
    print(f"📂 Loaded {len(df):,} rows  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. Descriptive Stats
# ══════════════════════════════════════════════════════════════════════════════
def descriptive_stats(df):
    print("\n" + "="*60)
    print("   DESCRIPTIVE STATISTICS")
    print("="*60)
    print(df[["sales", "price"]].describe().round(3))

    item_stats = df.groupby("item_id")["sales"].agg(["mean","std","min","max"]).round(2)
    item_stats["cv"] = (item_stats["std"] / item_stats["mean"]).round(3)
    print(f"\nDemand variability (CV) — first 10 items:\n{item_stats.head(10)}")

    item_stats.to_csv(ARTIFACTS_DIR / "item_demand_stats.csv")
    print(f"\n✅ Saved: item_demand_stats.csv")
    return item_stats


# ══════════════════════════════════════════════════════════════════════════════
# 2. Seasonal Decomposition (on training data only)
# ══════════════════════════════════════════════════════════════════════════════
def seasonal_decomposition_plot(df):
    print("\n📉 Seasonal decomposition (training period only)...")
    train = df[df["date"] < SPLIT_DATE]
    daily = train.groupby("date")["sales"].sum().asfreq("D").ffill()

    try:
        decomp = seasonal_decompose(daily, model="additive", period=7)
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        for ax, (label, data) in zip(axes, [
            ("Observed",  decomp.observed),
            ("Trend",     decomp.trend),
            ("Seasonal",  decomp.seasonal),
            ("Residual",  decomp.resid),
        ]):
            ax.plot(data, lw=1.2, color="#2E86AB")
            ax.set_ylabel(label)
            ax.set_title(f"Seasonal Decomposition – {label}", fontsize=11)
        plt.suptitle("Additive Seasonal Decomposition (Training Data: 2019–2022)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "08_seasonal_decomposition.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print("   ✅ Saved: 08_seasonal_decomposition.png")
    except Exception as e:
        print(f"   ⚠️  Decomposition skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. EOQ Model (based on training data)
# ══════════════════════════════════════════════════════════════════════════════
def calculate_eoq(annual_demand, unit_price,
                  ordering_cost=ORDERING_COST, holding_rate=HOLDING_COST_RATE):
    H = unit_price * holding_rate
    if H <= 0 or annual_demand <= 0:
        return 0
    return round(np.sqrt((2 * annual_demand * ordering_cost) / H), 2)


def calculate_rop(avg_daily_demand, lead_time=LEAD_TIME_DAYS,
                  std_daily_demand=0, z=Z_SCORE):
    safety_stock = z * std_daily_demand * np.sqrt(lead_time)
    rop = avg_daily_demand * lead_time + safety_stock
    return round(rop, 2), round(safety_stock, 2)


def calculate_total_inventory_cost(annual_demand, eoq, unit_price,
                                    ordering_cost=ORDERING_COST,
                                    holding_rate=HOLDING_COST_RATE):
    if eoq <= 0:
        return 0, 0, 0
    H = unit_price * holding_rate
    annual_ordering = (annual_demand / eoq) * ordering_cost
    annual_holding  = (eoq / 2) * H
    total = annual_ordering + annual_holding
    return round(total, 2), round(annual_ordering, 2), round(annual_holding, 2)


def run_eoq_analysis(df):
    print("\n" + "="*60)
    print("   EOQ / REORDER POINT / SAFETY STOCK ANALYSIS")
    print("   (Computed from TRAINING data: 2019-2022 only)")
    print("="*60)

    train = df[df["date"] < SPLIT_DATE]

    results = []
    for item_id, grp in train.groupby("item_id"):
        avg_price    = grp["price"].mean()
        daily_demand = grp.groupby("date")["sales"].sum()
        avg_daily    = daily_demand.mean()
        std_daily    = daily_demand.std()
        annual_demand = avg_daily * 365

        eoq = calculate_eoq(annual_demand, avg_price)
        rop, safety_stock = calculate_rop(avg_daily, std_daily_demand=std_daily)
        total_cost, ord_cost, hold_cost = calculate_total_inventory_cost(
            annual_demand, eoq, avg_price)
        orders_per_year = round(annual_demand / eoq, 2) if eoq > 0 else 0

        results.append({
            "item_id":              item_id,
            "avg_daily_demand":     round(avg_daily, 2),
            "std_daily_demand":     round(std_daily, 2),
            "annual_demand":        round(annual_demand, 2),
            "avg_unit_price":       round(avg_price, 2),
            "EOQ":                  eoq,
            "Reorder_Point":        rop,
            "Safety_Stock":         safety_stock,
            "Orders_Per_Year":      orders_per_year,
            "Annual_Ordering_Cost": ord_cost,
            "Annual_Holding_Cost":  hold_cost,
            "Total_Annual_Cost":    total_cost,
            "Lead_Time_Days":       LEAD_TIME_DAYS,
            "Service_Level":        SERVICE_LEVEL,
        })

    eoq_df = pd.DataFrame(results)
    eoq_df.to_csv(ARTIFACTS_DIR / "eoq_analysis.csv", index=False)

    print(f"\n  Sample EOQ Results:")
    print(eoq_df[["item_id","EOQ","Reorder_Point","Safety_Stock","Total_Annual_Cost"]]
          .head(10).to_string(index=False))
    print(f"\n  Total Annual Inventory Cost: £{eoq_df['Total_Annual_Cost'].sum():,.2f}")
    print(f"  Avg EOQ per item           : {eoq_df['EOQ'].mean():.1f} units")
    print(f"  Avg Safety Stock           : {eoq_df['Safety_Stock'].mean():.1f} units")
    print(f"\n  ✅ Saved: eoq_analysis.csv")

    _plot_eoq_cost_curve(eoq_df.iloc[0])
    return eoq_df


def _plot_eoq_cost_curve(row):
    D = row["annual_demand"]; S = ORDERING_COST
    H = row["avg_unit_price"] * HOLDING_COST_RATE
    if H <= 0 or D <= 0:
        return

    q_range  = np.linspace(1, row["EOQ"] * 3, 300)
    ordering = (D / q_range) * S
    holding  = (q_range / 2) * H
    total    = ordering + holding

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(q_range, ordering, label="Ordering Cost", color="#F18F01", lw=2)
    ax.plot(q_range, holding,  label="Holding Cost",  color="#2E86AB", lw=2)
    ax.plot(q_range, total,    label="Total Cost",    color="#C73E1D", lw=2.5)
    ax.axvline(row["EOQ"], color="#44BBA4", ls="--", lw=2,
               label=f"EOQ = {row['EOQ']:.0f}")
    ax.set_title(f"EOQ Cost Curve — {row['item_id']}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Order Quantity (units)"); ax.set_ylabel("Annual Cost (£)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_eoq_cost_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: 09_eoq_cost_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Traditional Forecasting Baselines
#    Train on 2019-2022, predict EVERY DAY in 2023, compare with actual
# ══════════════════════════════════════════════════════════════════════════════
TRAD_MODELS = ["Naive", "MovingAvg_28", "HoltWinters"]

PALETTE_TRAD = {
    "Naive":        "#F18F01",
    "MovingAvg_28": "#C73E1D",
    "HoltWinters":  "#44BBA4",
    "Actual":       "black",
}


def _forecast_traditional_2023(series_full: pd.Series) -> dict:
    """
    Given a FULL daily series (2019–2023), train on pre-2023 and produce
    day-by-day predictions for 2023.

    Returns dict of {model_name: pd.Series indexed by date}.
    """
    train = series_full[series_full.index < SPLIT_DATE]
    test  = series_full[series_full.index >= SPLIT_DATE]

    if len(train) < 60 or len(test) == 0:
        return {}, test

    n_test = len(test)
    preds  = {}

    # 1) Naive: last training value repeated
    preds["Naive"] = pd.Series(
        np.full(n_test, train.iloc[-1]),
        index=test.index
    )

    # 2) Moving Average (28-day from end of training)
    ma_val = train.rolling(28).mean().iloc[-1]
    preds["MovingAvg_28"] = pd.Series(
        np.full(n_test, ma_val),
        index=test.index
    )

    # 3) Holt-Winters: fit on train, forecast n_test steps
    try:
        hw = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=7
        ).fit(optimized=True, disp=False)
        hw_pred = hw.forecast(n_test)
        hw_pred.index = test.index
        preds["HoltWinters"] = hw_pred.clip(lower=0)
    except Exception:
        preds["HoltWinters"] = pd.Series(
            np.full(n_test, train.mean()),
            index=test.index
        )

    return preds, test


def traditional_forecasting(df):
    """
    Evaluate traditional baselines: train on 2019-2022, predict 2023.
    Aggregate across ALL store-item pairs.
    """
    print("\n" + "="*60)
    print("   TRADITIONAL FORECASTING BASELINES")
    print("   Train: 2019-2022  →  Test: 2023 (every day)")
    print("="*60)

    # ── Per-series evaluation ──────────────────────────────────────────────
    all_rows = []
    # Also accumulate aggregated daily predictions for plotting
    daily_preds = {m: [] for m in TRAD_MODELS}
    daily_actual = []

    pairs = list(df.groupby(["store_id", "item_id"]))
    print(f"   Evaluating {len(pairs)} store-item series...")

    for (store_id, item_id), grp in pairs:
        series = grp.sort_values("date").set_index("date")["sales"]
        series = series.asfreq("D")
        series = series.ffill().fillna(0)

        result = _forecast_traditional_2023(series)
        if not result:
            continue
        preds_dict, actual = result

        # Accumulate for aggregate daily plot
        daily_actual.append(actual)
        for model_name in TRAD_MODELS:
            if model_name in preds_dict:
                daily_preds[model_name].append(preds_dict[model_name])

        # Per-series metrics
        for model_name, pred in preds_dict.items():
            actual_arr = actual.values.astype(float)
            pred_arr   = pred.values.astype(float)
            mae  = np.mean(np.abs(actual_arr - pred_arr))
            rmse = np.sqrt(np.mean((actual_arr - pred_arr) ** 2))
            mape = np.mean(np.abs((actual_arr - pred_arr) /
                                   (actual_arr + 1e-5))) * 100
            all_rows.append({
                "store_id": store_id,
                "item_id":  item_id,
                "model":    model_name,
                "MAE":      mae,
                "RMSE":     rmse,
                "MAPE":     mape,
            })

    raw_df = pd.DataFrame(all_rows)

    # ── Aggregate: mean across all series per model ─────────────────────────
    agg = (
        raw_df.groupby("model")[["MAE", "RMSE", "MAPE"]]
        .mean()
        .round(4)
        .reset_index()
    )
    agg.to_csv(ARTIFACTS_DIR / "traditional_forecast_errors.csv", index=False)
    raw_df.to_csv(ARTIFACTS_DIR / "traditional_forecast_errors_raw.csv", index=False)

    print(f"\n{'Model':<20} {'MAE':>8}  {'RMSE':>8}  {'MAPE':>8}")
    print("-" * 50)
    for _, row in agg.iterrows():
        print(f"  {row['model']:<18} "
              f"{row['MAE']:>8.3f}  {row['RMSE']:>8.3f}  {row['MAPE']:>7.2f}%")

    print(f"\n  ✅ Saved: traditional_forecast_errors.csv  "
          f"({len(raw_df):,} series-model evaluations)")

    # ── Aggregate daily plot: sum across all series ─────────────────────────
    agg_actual = pd.concat(daily_actual, axis=1).sum(axis=1)
    agg_preds  = {}
    for model_name in TRAD_MODELS:
        if daily_preds[model_name]:
            agg_preds[model_name] = pd.concat(
                daily_preds[model_name], axis=1
            ).sum(axis=1)

    _plot_traditional_2023(agg_actual, agg_preds)

    # Save aggregate predictions for later comparison with ML
    agg_trad_df = pd.DataFrame({"date": agg_actual.index, "actual": agg_actual.values})
    for m, s in agg_preds.items():
        agg_trad_df[m] = s.values
    agg_trad_df.to_csv(ARTIFACTS_DIR / "traditional_daily_predictions_2023.csv",
                       index=False)
    print("   ✅ Saved: traditional_daily_predictions_2023.csv")

    return agg


def _plot_traditional_2023(actual_series, preds_dict):
    """Plot aggregated actual vs traditional predictions for 2023."""
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(actual_series.index, actual_series.values,
            label="Actual 2023", color="black", lw=2, alpha=0.9)

    for model_name, pred_series in preds_dict.items():
        ax.plot(pred_series.index, pred_series.values,
                label=model_name,
                color=PALETTE_TRAD.get(model_name, "grey"),
                lw=1.5, ls="--", alpha=0.8)

    ax.set_title("Traditional Baselines vs Actual — 2023 (Aggregated Daily Sales)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Daily Sales (all stores × items)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_traditional_vs_actual_2023.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: 10_traditional_vs_actual_2023.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Stockout & Overstocking Simulation
# ══════════════════════════════════════════════════════════════════════════════
def stockout_analysis(df, eoq_df):
    """Simulate inventory with proper lead-time and ROP logic."""
    print("\n" + "=" * 60)
    print("   STOCKOUT & OVERSTOCKING SIMULATION (corrected)")
    print("=" * 60)

    sim_results = []
    for _, row in eoq_df.iterrows():
        item_id   = row["item_id"]
        grp       = df[df["item_id"] == item_id].groupby("date")["sales"].sum()
        demands   = grp.values

        rop       = row["Reorder_Point"]
        eoq       = row["EOQ"]
        avg_price = row["avg_unit_price"]

        inventory       = eoq           # start with one full order
        stockout_units  = 0
        overstock_units = 0
        orders_placed   = 0

        # Pipeline of incoming orders: list of (arrival_day, qty)
        pipeline        = []
        order_placed    = False

        for day, demand in enumerate(demands):
            # ── Receive any arriving orders ───────────────────────────
            arrived = [qty for (arr, qty) in pipeline if arr <= day]
            pipeline = [(arr, qty) for (arr, qty) in pipeline if arr > day]
            inventory += sum(arrived)

            # ── Fulfil demand ─────────────────────────────────────────
            inventory -= demand
            if inventory < 0:
                stockout_units += abs(inventory)
                inventory = 0

            # ── Check ROP → place order if not already in transit ─────
            in_transit = sum(qty for (_, qty) in pipeline)
            if (inventory + in_transit) <= rop and not order_placed:
                pipeline.append((day + LEAD_TIME_DAYS, eoq))
                orders_placed += 1
                order_placed = True
            elif (inventory + in_transit) > rop:
                order_placed = False      # reset flag once above ROP

            # ── Track overstock ───────────────────────────────────────
            overstock_units += max(0, inventory - eoq * 1.5)

        stockout_cost  = stockout_units  * avg_price * STOCKOUT_COST_RATE
        overstock_cost = overstock_units * avg_price * HOLDING_COST_RATE / 365

        sim_results.append({
            "item_id":         item_id,
            "stockout_units":  int(stockout_units),
            "overstock_units": int(overstock_units),
            "stockout_cost":   round(stockout_cost, 2),
            "overstock_cost":  round(overstock_cost, 2),
            "total_sim_cost":  round(stockout_cost + overstock_cost, 2),
            "orders_placed":   orders_placed,
        })

    sim_df = pd.DataFrame(sim_results)
    sim_df.to_csv(ARTIFACTS_DIR / "stockout_analysis.csv", index=False)
    print(f"  Total Stockout Cost  : £{sim_df['stockout_cost'].sum():,.2f}")
    print(f"  Total Overstock Cost : £{sim_df['overstock_cost'].sum():,.2f}")
    print(f"  ✅ Saved: stockout_analysis.csv")
    return sim_df

if __name__ == "__main__":
    df         = load_data()
    item_stats = descriptive_stats(df)
    seasonal_decomposition_plot(df)
    eoq_df     = run_eoq_analysis(df)
    trad_agg   = traditional_forecasting(df)
    sim_df     = stockout_analysis(df, eoq_df)
    print("\n✅ Script 04 complete.")