# ══════════════════════════════════════════════════════════════════════════════
# model_comparison.py — Script 07  (updated: HistGBM instead of RandomForest)
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR      = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR     = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ML_MODELS   = ["HistGBM", "XGBoost", "LightGBM", "LSTM"] 
TRAD_MODELS = ["Naive", "MovingAvg_28", "HoltWinters"]
ALL_MODELS  = TRAD_MODELS + ML_MODELS

PALETTE = {
    "Naive":        "#F18F01",
    "MovingAvg_28": "#C73E1D",
    "HoltWinters":  "#E0A458",
    "HistGBM":      "#2E86AB",
    "XGBoost":      "#1B4F72",
    "LightGBM":     "#44BBA4",
    "LSTM":         "#9b59b6",                                      # ← added LSTM
}


# ════════════════════════════════════════════════════════════════════════════
# 1. Load & merge daily predictions
# ════════════════════════════════════════════════════════════════════════════
def load_predictions():
    trad_path = ARTIFACTS_DIR / "traditional_daily_predictions_2023.csv"
    ml_path   = ARTIFACTS_DIR / "ml_daily_predictions_2023.csv"

    for p in [trad_path, ml_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    trad = pd.read_csv(trad_path, parse_dates=["date"])
    ml   = pd.read_csv(ml_path,   parse_dates=["date"])

    merged = trad.merge(ml.drop(columns=["actual"]), on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    print(f"✅ Loaded {len(merged)} days of predictions")
    return merged


# ════════════════════════════════════════════════════════════════════════════
# 2. Metrics (on aggregated daily level)
# ════════════════════════════════════════════════════════════════════════════
def compute_metrics(merged):
    actual = merged["actual"].values
    rows = []
    for model in ALL_MODELS:
        if model not in merged.columns:
            continue
        pred = merged[model].values
        mae  = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2   = r2_score(actual, pred)
        mtype = "Traditional" if model in TRAD_MODELS else "ML"
        rows.append({"model": model, "type": mtype,
                     "MAE": round(mae, 4), "RMSE": round(rmse, 4),
                     "R2": round(r2, 4)})

    combined = pd.DataFrame(rows)
    combined.to_csv(ARTIFACTS_DIR / "full_model_comparison.csv", index=False)
    print(f"✅ full_model_comparison.csv saved")
    return combined


# ════════════════════════════════════════════════════════════════════════════
# 3. THE KEY PLOT — all models vs actual over 2023
# ════════════════════════════════════════════════════════════════════════════
def plot_all_vs_actual(merged):
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(merged["date"], merged["actual"],
            label="Actual", color="black", lw=2.5, alpha=0.9)

    for model in TRAD_MODELS:
        if model in merged.columns:
            ax.plot(merged["date"], merged[model], label=model,
                    color=PALETTE.get(model), lw=1.5, ls="--", alpha=0.7)

    for model in ML_MODELS:
        if model in merged.columns:
            ax.plot(merged["date"], merged[model], label=model,
                    color=PALETTE.get(model), lw=2, alpha=0.85)

    ax.set_title("All Models vs Actual — 2023 Daily Sales\n"
                 "(Dashed = Traditional, Solid = ML)",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Daily Sales")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "17_all_models_vs_actual_2023.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 17_all_models_vs_actual_2023.png")


def plot_all_vs_actual_monthly(merged):
    m2 = merged.copy()
    m2["month"] = m2["date"].dt.to_period("M")
    model_cols = [c for c in merged.columns if c in ALL_MODELS]
    monthly = m2.groupby("month")[["actual"] + model_cols].sum()
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly.index, monthly["actual"],
            label="Actual", color="black", lw=2.5, marker="o", markersize=6)

    for model in TRAD_MODELS:
        if model in monthly.columns:
            ax.plot(monthly.index, monthly[model], label=model,
                    color=PALETTE.get(model), lw=1.5, ls="--",
                    marker="s", markersize=5, alpha=0.7)
    for model in ML_MODELS:
        if model in monthly.columns:
            ax.plot(monthly.index, monthly[model], label=model,
                    color=PALETTE.get(model), lw=2,
                    marker="^", markersize=6, alpha=0.85)

    ax.set_title("Monthly — All Models vs Actual (2023)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Total Monthly Sales")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "18_monthly_all_models_2023.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 18_monthly_all_models_2023.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. Bar chart
# ════════════════════════════════════════════════════════════════════════════
def plot_metric_bars(combined):
    combined_s = combined.sort_values("MAE")
    colors = [PALETTE.get(m, "#999") for m in combined_s["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel in zip(axes,
                                  ["MAE", "RMSE"],
                                  ["MAE (units)", "RMSE (units)"]):
        bars = ax.bar(combined_s["model"], combined_s[metric],
                      color=colors, edgecolor="white", width=0.6)
        for bar, val in zip(bars, combined_s[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
        n_trad = sum(1 for m in combined_s["model"] if m in TRAD_MODELS)
        ax.axvline(x=n_trad - 0.5, color="grey", ls="--", lw=1.2, alpha=0.7)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Model Comparison on 2023 Holdout",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "19_comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 19_comparison_bars.png")


# ════════════════════════════════════════════════════════════════════════════
# 5. Heatmap
# ════════════════════════════════════════════════════════════════════════════
def plot_heatmap(combined):
    pivot = combined.set_index("model")[["MAE", "RMSE", "R2"]]
    pivot = pivot.reindex([m for m in ALL_MODELS if m in pivot.index])

    fig, ax = plt.subplots(figsize=(7, 5))
    data = pivot.values.astype(float)
    im   = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    vmax = np.nanmax(np.abs(data))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt_c = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_c)

    n_trad = sum(1 for m in pivot.index if m in TRAD_MODELS)
    ax.axhline(n_trad - 0.5, color="white", lw=2.5)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Performance Heatmap — 2023", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "20_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 20_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
# 6. Improvement %
# ════════════════════════════════════════════════════════════════════════════
def plot_improvement(combined):
    trad = combined[combined["type"] == "Traditional"]
    ml   = combined[combined["type"] == "ML"]
    if trad.empty:
        return

    best_trad_mae  = trad["MAE"].min()
    best_trad_rmse = trad["RMSE"].min()

    imp_data = []
    for _, row in ml.iterrows():
        imp_data.append({
            "model":   row["model"],
            "MAE ↓%":  (best_trad_mae  - row["MAE"])  / (best_trad_mae  + 1e-9) * 100,
            "RMSE ↓%": (best_trad_rmse - row["RMSE"]) / (best_trad_rmse + 1e-9) * 100,
        })
    if not imp_data:
        return

    imp_df = pd.DataFrame(imp_data)
    imp_df.to_csv(ARTIFACTS_DIR / "ml_improvement_over_traditional.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(imp_df))
    w = 0.3
    for i, metric in enumerate(["MAE ↓%", "RMSE ↓%"]):
        bars = ax.bar(x + i * w, imp_df[metric], w, label=metric,
                      color=["#2E86AB", "#44BBA4"][i])
        for bar, val in zip(bars, imp_df[metric]):
            c = "green" if val > 0 else "red"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontsize=9,
                    fontweight="bold", color=c)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(imp_df["model"])
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_ylabel("Improvement (%)")
    ax.set_title("ML Improvement over Best Traditional Baseline (2023)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "21_improvement_pct.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 21_improvement_pct.png")


# ════════════════════════════════════════════════════════════════════════════
# 7. Summary
# ════════════════════════════════════════════════════════════════════════════
def print_summary(combined):
    print("\n" + "=" * 68)
    print("   FULL MODEL COMPARISON — 2023 HOLDOUT")
    print("=" * 68)
    print(f"\n  {'Model':<20} {'Type':>12}  {'MAE':>10}  {'RMSE':>10}  {'R²':>10}")
    print(f"  {'-'*20} {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for _, row in combined.sort_values("MAE").iterrows():
        tag = " ⭐" if row["MAE"] == combined["MAE"].min() else ""
        print(f"  {row['model']:<20} {row['type']:>12}  "
              f"{row['MAE']:>10.3f}  {row['RMSE']:>10.3f}  "
              f"{row['R2']:>10.4f}{tag}")
    print("=" * 68)


if __name__ == "__main__":
    print("📊 Loading predictions...")
    merged   = load_predictions()
    combined = compute_metrics(merged)

    print("\n📊 Generating plots...")
    plot_all_vs_actual(merged)
    plot_all_vs_actual_monthly(merged)
    plot_metric_bars(combined)
    plot_heatmap(combined)
    plot_improvement(combined)
    print_summary(combined)

    print(f"\n✅ Script 07 complete.")