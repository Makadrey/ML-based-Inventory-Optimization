# ══════════════════════════════════════════════════════════════════════════════
# visualization.py — Script 03
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "artifacts" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]
sns.set_palette(PALETTE)


def load_data():
    df = pd.read_csv(DATA_DIR / "processed_data.csv", parse_dates=["date"])
    print(f"📂 Loaded {len(df):,} rows")
    return df


def plot_sales_over_time(df):
    daily = df.groupby("date")["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["date"], daily["sales"], color=PALETTE[0], lw=1.2, alpha=0.8)
    ax.fill_between(daily["date"], daily["sales"], alpha=0.15, color=PALETTE[0])
    daily["rolling_30"] = daily["sales"].rolling(30).mean()
    ax.plot(daily["date"], daily["rolling_30"], color=PALETTE[2], lw=2,
            label="30-day avg", ls="--")
    # Mark train/test split
    ax.axvline(pd.Timestamp("2023-01-01"), color="red", ls="--", lw=2,
               label="Train / Test split (2023)")
    ax.set_title("Total Daily Sales Over Time", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Units Sold")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_sales_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: 01_sales_over_time.png")


def plot_seasonality(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    monthly = df.groupby("month")["sales"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    axes[0].bar(monthly["month"], monthly["sales"], color=PALETTE[:12], edgecolor="white")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(month_names, rotation=45)
    axes[0].set_title("Average Sales by Month", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Avg Units Sold")

    weekday = df.groupby("weekday")["sales"].mean().reset_index()
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    axes[1].bar(weekday["weekday"], weekday["sales"], color=PALETTE, edgecolor="white")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names)
    axes[1].set_title("Average Sales by Day of Week", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Avg Units Sold")

    plt.suptitle("Seasonality Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_seasonality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: 02_seasonality.png")


def plot_promo_effect(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = df.copy()
    df["promo_label"] = df["promo"].map({0: "No Promo", 1: "Promo"})
    sns.boxplot(data=df, x="promo_label", y="sales", ax=axes[0],
                palette=["#2E86AB", "#F18F01"])
    axes[0].set_title("Sales Distribution: Promo vs No Promo", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(""); axes[0].set_ylabel("Units Sold")

    promo_avg = df.groupby("promo")["sales"].mean()
    bars = axes[1].bar(["No Promo", "Promo"], promo_avg.values,
                       color=["#2E86AB", "#F18F01"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, promo_avg.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", fontweight="bold")
    axes[1].set_title("Average Sales: Promo Impact", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Avg Units Sold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_promo_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: 03_promo_effect.png")


def plot_top_items_stores(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    top_stores = df.groupby("store_id")["sales"].sum().nlargest(10).reset_index()
    axes[0].barh(top_stores["store_id"], top_stores["sales"], color=PALETTE[0])
    axes[0].set_title("Top 10 Stores by Total Sales", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Total Units Sold")

    top_items = df.groupby("item_id")["sales"].sum().nlargest(10).reset_index()
    axes[1].barh(top_items["item_id"], top_items["sales"], color=PALETTE[1])
    axes[1].set_title("Top 10 Items by Total Sales", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Total Units Sold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_top_items_stores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: 04_top_items_stores.png")


def plot_price_vs_sales(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sample = df.sample(min(5000, len(df)), random_state=42)
    axes[0].scatter(sample["price"], sample["sales"],
                    alpha=0.2, color=PALETTE[0], s=10)
    axes[0].set_xlabel("Price"); axes[0].set_ylabel("Units Sold")
    axes[0].set_title("Price vs Sales", fontsize=12, fontweight="bold")

    num_cols = [c for c in ["sales", "price", "promo", "weekday", "month", "is_weekend"]
                if c in df.columns]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=axes[1], linewidths=0.5, annot_kws={"size": 10})
    axes[1].set_title("Feature Correlation Heatmap", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_price_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: 05_price_correlation.png")


def plot_sales_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["sales"], bins=50, color=PALETTE[0], edgecolor="white", alpha=0.8)
    axes[0].set_title("Sales Distribution (Histogram)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Units Sold"); axes[0].set_ylabel("Frequency")

    sns.kdeplot(df["sales"], ax=axes[1], fill=True, color=PALETTE[0], alpha=0.4)
    axes[1].set_title("Sales Distribution (KDE)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Units Sold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_sales_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: 06_sales_distribution.png")


def plot_yearly_trends(df):
    yearly_monthly = df.groupby(["year", "month"])["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    for year, grp in yearly_monthly.groupby("year"):
        ax.plot(grp["month"], grp["sales"], marker="o", label=str(year), lw=2)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_title("Year-over-Year Monthly Sales Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Total Sales")
    ax.legend(title="Year")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_yearly_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: 07_yearly_trends.png")


def plot_train_test_split(df):
    """Visualise the train (2019-2022) vs test (2023) split."""
    daily = df.groupby("date")["sales"].sum().reset_index()
    split = pd.Timestamp("2023-01-01")

    fig, ax = plt.subplots(figsize=(14, 5))
    train_d = daily[daily["date"] < split]
    test_d  = daily[daily["date"] >= split]

    ax.plot(train_d["date"], train_d["sales"], color="#2E86AB", lw=1.2,
            label="Train (2019–2022)", alpha=0.8)
    ax.plot(test_d["date"], test_d["sales"], color="#C73E1D", lw=1.2,
            label="Test (2023)", alpha=0.8)
    ax.axvline(split, color="black", ls="--", lw=2, label="Split point")
    ax.fill_between(train_d["date"], train_d["sales"], alpha=0.1, color="#2E86AB")
    ax.fill_between(test_d["date"], test_d["sales"], alpha=0.1, color="#C73E1D")
    ax.set_title("Train / Test Split: 2019–2022 vs 2023", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Daily Sales")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "00_train_test_split.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: 00_train_test_split.png")


if __name__ == "__main__":
    df = load_data()
    print("\n📊 Generating visualizations...")
    plot_train_test_split(df)
    plot_sales_over_time(df)
    plot_seasonality(df)
    plot_promo_effect(df)
    plot_top_items_stores(df)
    plot_price_vs_sales(df)
    plot_sales_distribution(df)
    plot_yearly_trends(df)
    print("\n✅ Script 03 complete.")