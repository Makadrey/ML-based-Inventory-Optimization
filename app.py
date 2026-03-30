# ══════════════════════════════════════════════════════════════════════════════
# app.py — Streamlit Dashboard
# HistGBM / XGBoost / LightGBM  ·  No MAPE  ·  Per-store-item drill-down
# ══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

APP_DIR   = Path(__file__).resolve().parent
DATA_DIR  = APP_DIR / "data"
ART_DIR   = APP_DIR / "artifacts"
MOD_DIR   = ART_DIR / "models"
PLOTS_DIR = ART_DIR / "plots"

ML_MODELS   = ["HistGBM", "XGBoost", "LightGBM", "LSTM"]        
TRAD_MODELS = ["Naive", "MovingAvg_28", "HoltWinters"]
ALL_MODELS  = TRAD_MODELS + ML_MODELS

PALETTE = {
    "Naive": "#F18F01", "MovingAvg_28": "#C73E1D", "HoltWinters": "#E0A458",
    "HistGBM": "#2E86AB", "XGBoost": "#1B4F72", "LightGBM": "#44BBA4",
    "LSTM": "#9b59b6",                                     
}
HOLDING_COST_RATE = 0.25
ORDERING_COST_DEF = 50.0
Z_95 = 1.645

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Inventory Optimization | MSc AI & DS",
                   page_icon="📦", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #f8fafc; }
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 60%, #1a3a5c 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #3b82f6;
    margin-bottom: 1rem;
}
.metric-card.green  { border-left-color: #10b981; }
.metric-card.orange { border-left-color: #f59e0b; }
.metric-card.red    { border-left-color: #ef4444; }
.metric-label { font-size:.78rem; color:#64748b; font-weight:500;
    text-transform:uppercase; letter-spacing:.05em; }
.metric-value { font-size:1.9rem; font-weight:700; color:#0f172a; line-height:1.1; }
.metric-sub   { font-size:.8rem; color:#94a3b8; margin-top:2px; }
.section-header {
    background: linear-gradient(90deg,#1e40af,#3b82f6); color:white;
    padding:.6rem 1.2rem; border-radius:8px; font-size:1rem;
    font-weight:600; margin-bottom:1rem; margin-top:.5rem;
}
.hero-banner {
    background: linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0e4f8a 100%);
    color:white; padding:2rem 2.5rem; border-radius:16px; margin-bottom:2rem;
}
.hero-banner h1 { font-size:1.8rem; font-weight:700; margin:0; }
.hero-banner p  { color:#94a3b8; margin:.3rem 0 0; font-size:.95rem; }
</style>
""", unsafe_allow_html=True)


# ─── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_datasets():
    data = {}
    for key, path in [
        ("processed",   DATA_DIR / "processed_data.csv"),
        ("eoq",         ART_DIR  / "eoq_analysis.csv"),
        ("stockout",    ART_DIR  / "stockout_analysis.csv"),
        ("metrics",     ART_DIR  / "model_metrics.csv"),
        ("comparison",  ART_DIR  / "full_model_comparison.csv"),
        ("trad_daily",  ART_DIR  / "traditional_daily_predictions_2023.csv"),
        ("ml_daily",    ART_DIR  / "ml_daily_predictions_2023.csv"),
        ("preds_full",  ART_DIR  / "ml_predictions_2023_full.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            data[key] = df
    return data


# ─── EOQ helpers ──────────────────────────────────────────────────────────────
def calc_eoq(D, price, S=ORDERING_COST_DEF, h=HOLDING_COST_RATE):
    H = price * h
    return round(np.sqrt(2 * D * S / H), 1) if H > 0 and D > 0 else 0.0

def calc_rop(avg_d, std_d, lt=7, z=Z_95):
    ss = z * std_d * np.sqrt(lt)
    return round(avg_d * lt + ss, 1), round(ss, 1)

def calc_total_cost(D, Q, price, S=ORDERING_COST_DEF, h=HOLDING_COST_RATE):
    H = price * h
    if Q <= 0: return 0., 0., 0.
    oc = (D / Q) * S;  hc = (Q / 2) * H
    return round(oc + hc, 2), round(oc, 2), round(hc, 2)


# ─── Metric helper (no MAPE) ─────────────────────────────────────────────────
def _metrics(actual, pred):
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2   = r2_score(actual, pred)
    return mae, rmse, r2


# ════════════════════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════════════════════
def main():
    data = load_datasets()

    with st.sidebar:
        st.markdown("### 📦 Inventory Optimizer")
        st.markdown("---")
        st.markdown("**MSc AI & Data Science**")
        st.markdown("Oluwadamilare Adubi · 2414566")
        st.markdown("---")
        page = st.selectbox("📍 Navigate", [
            "🏠 Dashboard",
            "🔮 Demand Forecast",
            "📦 EOQ Calculator",
            "📊 Model Evaluation",
            "📈 EDA & Insights",
            "ℹ️  About",
        ])

    st.markdown("""
    <div class='hero-banner'>
        <h1>📦 Inventory Optimization System</h1>
        <p>ML-based demand forecasting & cost optimization ·
           Train 2019–2022 | Test 2023</p>
    </div>""", unsafe_allow_html=True)

    {"🏠 Dashboard":       dashboard,
     "🔮 Demand Forecast": forecast_page,
     "📦 EOQ Calculator":  eoq_calculator,
     "📊 Model Evaluation": evaluation_page,
     "📈 EDA & Insights":  eda_page,
     "ℹ️  About":          about_page,
    }[page](data)


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD  — overall metrics for all stores combined
# ════════════════════════════════════════════════════════════════════════════
def dashboard(data):
    st.markdown("<div class='section-header'>📊 Project Overview</div>",
                unsafe_allow_html=True)

    df   = data.get("processed")
    eoq  = data.get("eoq")
    comp = data.get("comparison")
    ml_m = data.get("metrics")          # row-level overall metrics

    if df is None:
        st.warning("Run the pipeline first."); return

    # ── KPI cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total_sales = int(df["sales"].sum())
    n_stores    = df["store_id"].nunique()
    n_items     = df["item_id"].nunique()
    train_pct   = round(len(df[df["date"] < "2023-01-01"]) / len(df) * 100, 1)

    c1.markdown(f"<div class='metric-card'><div class='metric-label'>Total Units Sold"
                f"</div><div class='metric-value'>{total_sales:,}</div></div>",
                unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card green'><div class='metric-label'>Stores"
                f"</div><div class='metric-value'>{n_stores}</div></div>",
                unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card orange'><div class='metric-label'>Products"
                f"</div><div class='metric-value'>{n_items}</div></div>",
                unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card red'><div class='metric-label'>Train / Test"
                f"</div><div class='metric-value'>{train_pct}% / "
                f"{100-train_pct:.1f}%</div>"
                f"<div class='metric-sub'>2019–2022 / 2023</div></div>",
                unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    # ── Inventory cost ────────────────────────────────────────────────────
    with c1:
        st.markdown("<div class='section-header'>💰 Inventory Cost Summary</div>",
                    unsafe_allow_html=True)
        if eoq is not None:
            cc1, cc2 = st.columns(2)
            cc1.metric("Ordering Cost/yr",
                       f"£{eoq['Annual_Ordering_Cost'].sum():,.0f}")
            cc2.metric("Holding Cost/yr",
                       f"£{eoq['Annual_Holding_Cost'].sum():,.0f}")
            cc1.metric("Total Inventory Cost",
                       f"£{eoq['Total_Annual_Cost'].sum():,.0f}")
            cc2.metric("Avg EOQ", f"{eoq['EOQ'].mean():.0f} units")

    # ── Best ML model (row-level, all stores combined) ────────────────────
    with c2:
        st.markdown("<div class='section-header'>🏆 Best ML Model "
                    "(overall)</div>", unsafe_allow_html=True)
        if ml_m is not None and not ml_m.empty:
            best = ml_m.loc[ml_m["MAE"].idxmin()]
            st.success(f"🥇 **{best['model']}** — lowest overall MAE")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE",  f"{best['MAE']:.3f}")
            mc2.metric("RMSE", f"{best['RMSE']:.3f}")
            mc3.metric("R²",   f"{best['R2']:.4f}")

            # Improvement over trad
            if comp is not None:
                trad = comp[comp["type"] == "Traditional"]
                if not trad.empty:
                    imp = ((trad["MAE"].min() - best["MAE"])
                           / (trad["MAE"].min() + 1e-9) * 100)
                    if imp > 0:
                        st.info(f"📉 {imp:.1f}% lower MAE than best "
                                f"traditional baseline (aggregated daily)")

    # ── Trend chart ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>📈 Daily Sales Trend</div>",
                unsafe_allow_html=True)
    daily = df.groupby("date")["sales"].sum().reset_index()
    daily["r30"] = daily["sales"].rolling(30).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["sales"],
                             name="Daily", line=dict(color="#bfdbfe", width=1),
                             opacity=0.6))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["r30"],
                             name="30-day Avg",
                             line=dict(color="#2563eb", width=2.5)))
    # WITH THIS:
    fig.add_shape(type="line", x0=pd.Timestamp("2023-01-01"),
                x1=pd.Timestamp("2023-01-01"), y0=0, y1=1,
                yref="paper", line=dict(color="red", width=2, dash="dash"))
    fig.add_annotation(x=pd.Timestamp("2023-01-01"), y=1, yref="paper",
                    text="Train / Test", showarrow=False,
                    font=dict(color="red", size=11),
                    xanchor="left", yanchor="bottom")

    fig.update_layout(template="plotly_white", height=350,
                      legend=dict(orientation="h", yanchor="bottom",
                                  y=1.02, xanchor="right", x=1),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# DEMAND FORECAST PAGE — overall, by store, by store-item
# ════════════════════════════════════════════════════════════════════════════
def forecast_page(data):
    st.markdown("<div class='section-header'>🔮 2023 Forecast — "
                "Actual vs Predicted</div>", unsafe_allow_html=True)

    preds_full = data.get("preds_full")    # row-level
    trad_daily = data.get("trad_daily")    # aggregated daily
    ml_daily   = data.get("ml_daily")      # aggregated daily

    if preds_full is None:
        st.warning("Run the pipeline first (scripts 04 + 06)."); return

    model_cols = [c for c in preds_full.columns if c in ML_MODELS]

    tab_all, tab_store, tab_store_item = st.tabs([
        "📊 Overall (all stores)", "🏪 By Store", "📦 By Store & Item"
    ])

    # ── TAB 1: Overall ────────────────────────────────────────────────────
    with tab_all:
        st.markdown("> Aggregated across **all stores × items** — "
                    "traditional (dashed) and ML (solid) vs actual.")

        if trad_daily is not None and ml_daily is not None:
            merged = trad_daily.merge(
                ml_daily.drop(columns=["actual"]), on="date", how="inner"
            ).sort_values("date")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged["date"], y=merged["actual"],
                name="Actual", line=dict(color="black", width=2.5)))
            for m in TRAD_MODELS:
                if m in merged.columns:
                    fig.add_trace(go.Scatter(
                        x=merged["date"], y=merged[m], name=m,
                        line=dict(color=PALETTE[m], width=1.5, dash="dash"),
                        opacity=0.7))
            for m in ML_MODELS:
                if m in merged.columns:
                    fig.add_trace(go.Scatter(
                        x=merged["date"], y=merged[m], name=m,
                        line=dict(color=PALETTE[m], width=2), opacity=0.85))
            fig.update_layout(template="plotly_white", height=500,
                              title="All Models vs Actual — 2023",
                              xaxis_title="Date",
                              yaxis_title="Total Daily Sales",
                              legend=dict(orientation="h", yanchor="bottom",
                                          y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        # Overall row-level metrics (all stores at once)
        st.markdown("**Overall ML Metrics (computed on all store-item "
                    "observations):**")
        cols = st.columns(len(model_cols))
        for col, m in zip(cols, model_cols):
            mae, rmse, r2 = _metrics(preds_full["actual"], preds_full[m])
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{m}</div>"
                f"<div class='metric-value'>MAE {mae:.2f}</div>"
                f"<div class='metric-sub'>RMSE {rmse:.2f} · R² {r2:.4f}"
                f"</div></div>", unsafe_allow_html=True)

    # ── TAB 2: By Store ──────────────────────────────────────────────────
    with tab_store:
        stores = sorted(preds_full["store_id"].unique())
        sel_store = st.selectbox("🏪 Select Store", stores, key="fs")

        sd = preds_full[preds_full["store_id"] == sel_store]
        daily_s = sd.groupby("date")[["actual"] + model_cols].sum().reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=daily_s["date"], y=daily_s["actual"],
            name="Actual", line=dict(color="black", width=2)))
        for m in model_cols:
            fig2.add_trace(go.Scatter(
                x=daily_s["date"], y=daily_s[m], name=m,
                line=dict(color=PALETTE.get(m), width=1.8, dash="dash")))
        fig2.update_layout(template="plotly_white", height=420,
                           title=f"Store: {sel_store} — Daily Sales 2023",
                           xaxis_title="Date",
                           yaxis_title="Daily Sales (all items)")
        st.plotly_chart(fig2, use_container_width=True)

        # Per-store metrics
        st.markdown(f"**Metrics for {sel_store}  (all items combined):**")
        cols2 = st.columns(len(model_cols))
        for col, m in zip(cols2, model_cols):
            mae, rmse, r2 = _metrics(sd["actual"], sd[m])
            col.metric(f"{m} MAE", f"{mae:.2f}")
            col.metric(f"{m} RMSE", f"{rmse:.2f}")

    # ── TAB 3: By Store & Item ───────────────────────────────────────────
    with tab_store_item:
        c1, c2 = st.columns(2)
        stores2 = sorted(preds_full["store_id"].unique())
        sel_s2  = c1.selectbox("🏪 Store", stores2, key="fsi_s")
        items2  = sorted(
            preds_full[preds_full["store_id"] == sel_s2]["item_id"].unique())
        sel_i2  = c2.selectbox("📦 Item", items2, key="fsi_i")

        sub = (preds_full[
            (preds_full["store_id"] == sel_s2) &
            (preds_full["item_id"]  == sel_i2)
        ].sort_values("date"))

        if sub.empty:
            st.warning("No data for this combination."); return

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=sub["date"], y=sub["actual"],
            name="Actual", line=dict(color="black", width=2)))
        for m in model_cols:
            fig3.add_trace(go.Scatter(
                x=sub["date"], y=sub[m], name=m,
                line=dict(color=PALETTE.get(m), width=1.8, dash="dash"),
                opacity=0.85))
        fig3.update_layout(
            template="plotly_white", height=420,
            title=f"{sel_i2} @ {sel_s2} — 2023",
            xaxis_title="Date", yaxis_title="Units Sold")
        st.plotly_chart(fig3, use_container_width=True)

        # Per-item metrics
        st.markdown(f"**Metrics for {sel_i2} @ {sel_s2}:**")
        cols3 = st.columns(len(model_cols))
        for col, m in zip(cols3, model_cols):
            mae, rmse, r2 = _metrics(sub["actual"], sub[m])
            col.metric(f"{m} MAE", f"{mae:.2f}")
            col.metric(f"{m} RMSE", f"{rmse:.2f}")
            col.metric(f"{m} R²", f"{r2:.4f}")

        # Recent rows
        st.markdown("---")
        st.markdown("**📋 Recent Predictions:**")
        show = sub.tail(21)[["date", "actual"] + model_cols].copy()
        for m in model_cols:
            show[f"{m}_err"] = (show["actual"] - show[m]).round(1)
        st.dataframe(show.set_index("date").round(1),
                     use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION PAGE
# ════════════════════════════════════════════════════════════════════════════
def evaluation_page(data):
    st.markdown("<div class='section-header'>📊 Model Evaluation — "
                "2023 Holdout</div>", unsafe_allow_html=True)

    comp = data.get("comparison")        # aggregated daily comparison
    ml_m = data.get("metrics")           # row-level ML metrics

    if comp is None and ml_m is None:
        st.warning("Run the pipeline first."); return

    tab_comp, tab_plots = st.tabs(["📊 Comparison", "🖼️ Plots"])

    with tab_comp:
        # ── Combined table ────────────────────────────────────────────────
        if comp is not None:
            st.markdown("**All models — aggregated daily metrics on 2023:**")
            display_cols = [c for c in ["model","type","MAE","RMSE","R2"]
                           if c in comp.columns]
            st.dataframe(
                comp[display_cols].sort_values("MAE")
                .style
                .highlight_min(subset=["MAE","RMSE"], color="#059669")
                .highlight_max(subset=["R2"], color="#059669")
                .format({"MAE":"{:.4f}","RMSE":"{:.4f}","R2":"{:.4f}"}),
                use_container_width=True)

            best = comp.loc[comp["MAE"].idxmin()]
            st.success(f"🏆 **{best['model']}** — lowest aggregated MAE "
                       f"({best['MAE']:.4f})")

        # ── Row-level ML metrics ──────────────────────────────────────────
        if ml_m is not None:
            st.markdown("---")
            st.markdown("**ML models — row-level metrics "
                        "(all store-item observations):**")
            display_cols2 = [c for c in ["model","MAE","RMSE","R2"]
                            if c in ml_m.columns]
            st.dataframe(
                ml_m[display_cols2].sort_values("MAE")
                .style
                .highlight_min(subset=["MAE","RMSE"], color="#059669")
                .format({"MAE":"{:.4f}","RMSE":"{:.4f}","R2":"{:.4f}"}),
                use_container_width=True)

        # ── Bar chart ─────────────────────────────────────────────────────
        if comp is not None:
            st.markdown("---")
            sorted_c = comp.sort_values("MAE")
            colors = [PALETTE.get(m, "#999") for m in sorted_c["model"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="MAE", x=sorted_c["model"], y=sorted_c["MAE"],
                marker_color=colors, opacity=0.9,
                text=[f"{v:.3f}" for v in sorted_c["MAE"]],
                textposition="outside"))
            fig.add_trace(go.Bar(
                name="RMSE", x=sorted_c["model"], y=sorted_c["RMSE"],
                marker_color=colors, opacity=0.5,
                text=[f"{v:.3f}" for v in sorted_c["RMSE"]],
                textposition="outside"))
            fig.update_layout(barmode="group", template="plotly_white",
                              height=400,
                              title="MAE & RMSE — All Models (2023)")
            st.plotly_chart(fig, use_container_width=True)

    with tab_plots:
        st.markdown("**🖼️ Saved Plots:**")
        plots = sorted(PLOTS_DIR.glob("*.png")) if PLOTS_DIR.exists() else []
        if plots:
            sel = st.selectbox("Select plot:", [p.name for p in plots])
            st.image(str(PLOTS_DIR / sel), use_container_width=True)
        else:
            st.info("No plots found.")


# ════════════════════════════════════════════════════════════════════════════
# EOQ CALCULATOR
# ════════════════════════════════════════════════════════════════════════════
def eoq_calculator(data):
    st.markdown("<div class='section-header'>📦 EOQ & Inventory Policy</div>",
                unsafe_allow_html=True)
    # tab1 = st.tabs(["Calculator"])

    # with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        ad = st.number_input("Annual Demand", value=5000, step=100)
    with c2:
        up = st.number_input("Unit Price (£)", value=25.0, step=1.0)
    with c3:
        oc = st.number_input("Ordering Cost (£)", value=50.0, step=5.0)

    if st.button("⚡ Calculate", type="primary"):
        eoq_v = calc_eoq(ad, up, oc, HOLDING_COST_RATE)
        tc, o_c, h_c = calc_total_cost(ad, eoq_v, up, oc, HOLDING_COST_RATE)
        st.success("✅ Computed!")
        r1, r2, r3 = st.columns(3)
        r1.metric("EOQ",         f"{eoq_v:,.0f} units")
        r2.metric("Ordering/yr", f"£{o_c:,.2f}")
        r3.metric("Holding/yr",  f"£{h_c:,.2f}")
        r1.metric("Total/yr",    f"£{tc:,.2f}")
        r2.metric("Orders/yr",   f"{ad / eoq_v:.1f}")
        r3.metric("Holding Rate","20% (fixed)")

        q = np.linspace(max(1, eoq_v * 0.1), eoq_v * 3.5, 300)
        H = up * HOLDING_COST_RATE
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q, y=(ad/q)*oc,
                                    name="Ordering", line=dict(color="#f59e0b")))
        fig.add_trace(go.Scatter(x=q, y=(q/2)*H,
                                    name="Holding", line=dict(color="#3b82f6")))
        fig.add_trace(go.Scatter(x=q, y=(ad/q)*oc+(q/2)*H,
                                    name="Total",
                                    line=dict(color="#ef4444", width=2.5)))
        fig.add_vline(x=eoq_v, line_dash="dash", line_color="#10b981",
                        annotation_text=f"EOQ={eoq_v:.0f}")
        fig.update_layout(template="plotly_white", height=350,
                            xaxis_title="Order Qty",
                            yaxis_title="Annual Cost (£)")
        st.plotly_chart(fig, use_container_width=True)

   
# ════════════════════════════════════════════════════════════════════════════
# EDA
# ════════════════════════════════════════════════════════════════════════════
def eda_page(data):
    st.markdown("<div class='section-header'>📈 Exploratory Data Analysis</div>",
                unsafe_allow_html=True)
    df = data.get("processed")
    if df is None:
        st.warning("Run the pipeline first."); return

    tab1, tab2, tab3 = st.tabs(["📅 Seasonality", "🏪 Store/Item", "🔍 Data"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            monthly = df.groupby("month")["sales"].mean().reset_index()
            month_map = {i: m for i, m in enumerate(
                ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"], 1)}
            monthly["mn"] = monthly["month"].map(month_map)
            fig = px.bar(monthly, x="mn", y="sales",
                         title="Avg Sales by Month", color="sales",
                         color_continuous_scale="Blues")
            fig.update_layout(template="plotly_white", height=320,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            wd = df.groupby("weekday")["sales"].mean().reset_index()
            day_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",
                       4:"Fri",5:"Sat",6:"Sun"}
            wd["dn"] = wd["weekday"].map(day_map)
            fig2 = px.bar(wd, x="dn", y="sales",
                          title="Avg Sales by Day", color="sales",
                          color_continuous_scale="Oranges")
            fig2.update_layout(template="plotly_white", height=320,
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            ts = df.groupby("store_id")["sales"].sum().nlargest(10).reset_index()
            fig = px.bar(ts.sort_values("sales"), x="sales", y="store_id",
                         orientation="h", title="Top 10 Stores",
                         color="sales", color_continuous_scale="Blues")
            fig.update_layout(template="plotly_white", height=350,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ti = df.groupby("item_id")["sales"].sum().nlargest(10).reset_index()
            fig = px.bar(ti.sort_values("sales"), x="sales", y="item_id",
                         orientation="h", title="Top 10 Items",
                         color="sales", color_continuous_scale="Oranges")
            fig.update_layout(template="plotly_white", height=350,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.dataframe(df.sample(min(500, len(df))).sort_values("date"),
                     use_container_width=True, height=400)
        st.caption(f"Showing 500 of {len(df):,} rows")


# ════════════════════════════════════════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════════════════════════════════════════
def about_page(data):
    st.markdown("<div class='section-header'>ℹ️ About This Project</div>",
                unsafe_allow_html=True)
    st.markdown("""
### Inventory Levels Optimization using ML-based Forecasting

| Field | Details |
|-------|---------|
| **Student** | Oluwadamilare Adubi |
| **Student Number** | 2414566 |
| **Programme** | MSc Artificial Intelligence & Data Science |
| **Supervisor** | Aliyu Aliyu |

### Approach
- **Train**: 2019–2022 data (models never see 2023)
- **Test**: All of 2023 — predict daily sales for every store-item pair
- **Traditional Baselines**: Naive, 28-day Moving Average, Holt-Winters
- **ML Models**: HistGradientBoosting, XGBoost, LightGBM
- **Metrics**: MAE, RMSE, R² — computed on 2023 holdout only

""")


if __name__ == "__main__":
    main()