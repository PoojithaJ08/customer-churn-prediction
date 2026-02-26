"""
dashboard/app.py
─────────────────
Churn Prediction Analytics Dashboard
Connects to trained model + customer data to show:
  - Model performance metrics
  - At-risk customer breakdown
  - Feature importance
  - Risk tier distribution
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import get_feature_matrix, engineer_features, encode_categoricals, DROP_COLS

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

BG      = "#080c14"
CARD    = "#111827"
BORDER  = "#1f2937"
BORDER2 = "#374151"
CYAN    = "#22d3ee"
RED     = "#f87171"
AMBER   = "#fbbf24"
GREEN   = "#4ade80"
BLUE    = "#60a5fa"
TEXT    = "#f9fafb"
SUBTEXT = "#9ca3af"
MUTED   = "#4b5563"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
*, html, body {{ font-family: 'DM Sans', sans-serif !important; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding: 2.5rem 3rem !important; max-width: 1440px !important; }}
#MainMenu, footer, header, .stDeployButton, div[data-testid="stToolbar"] {{ visibility: hidden; display: none; }}
.kpi-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1.25rem; margin-bottom:2.5rem; }}
.kpi {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:1.5rem 1.75rem; position:relative; }}
.kpi-accent {{ position:absolute; top:0; left:1.75rem; right:1.75rem; height:2px; border-radius:0 0 4px 4px; }}
.kpi-val {{ font-size:2.4rem; font-weight:700; color:{TEXT}; line-height:1; letter-spacing:-1px; margin:0.8rem 0 0.5rem; }}
.kpi-lbl {{ font-size:0.72rem; color:{SUBTEXT}; font-weight:500; text-transform:uppercase; letter-spacing:0.6px; }}
.kpi-tag {{ display:inline-block; margin-top:0.5rem; font-size:0.68rem; font-family:'DM Mono',monospace; color:{MUTED}; background:#0d1420; padding:2px 8px; border-radius:4px; border:1px solid {BORDER}; }}
.db-header {{ display:flex; align-items:flex-end; justify-content:space-between; margin-bottom:2.5rem; padding-bottom:1.5rem; border-bottom:1px solid {BORDER}; }}
.db-title {{ font-size:1.5rem; font-weight:700; color:{TEXT}; letter-spacing:-0.4px; }}
.db-title span {{ color:{CYAN}; }}
.db-meta {{ font-size:0.75rem; color:{MUTED}; font-family:'DM Mono',monospace; }}
.sl {{ font-size:0.68rem; font-weight:600; color:{MUTED}; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.6rem; font-family:'DM Mono',monospace; }}
.cc {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:1.5rem; }}
.ct {{ font-size:0.9rem; font-weight:600; color:{TEXT}; margin-bottom:0.2rem; }}
.cd {{ font-size:0.73rem; color:{MUTED}; margin-bottom:1rem; line-height:1.4; }}
.div {{ height:1px; background:{BORDER}; margin:1.75rem 0; }}
.risk-row {{ display:flex; align-items:center; justify-content:space-between; padding:0.65rem 0; border-bottom:1px solid {BORDER}; }}
.risk-row:last-child {{ border-bottom:none; }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df      = pd.read_csv("data/customers.csv")
    metrics = json.loads(Path("models/metrics.json").read_text())
    model   = joblib.load("models/churn_model.pkl")
    fnames  = joblib.load("models/feature_names.pkl")
    return df, metrics, model, fnames

@st.cache_data
def score_customers(df):
    model, fnames = joblib.load("models/churn_model.pkl"), joblib.load("models/feature_names.pkl")
    df_feat = engineer_features(df)
    df_feat = encode_categoricals(df_feat)
    drop    = [c for c in DROP_COLS if c in df_feat.columns]
    X       = df_feat.drop(columns=drop)
    for col in fnames:
        if col not in X.columns:
            X[col] = 0
    X     = X[fnames]
    proba = model.predict_proba(X)[:, 1]
    df    = df.copy()
    df["churn_probability"] = proba.round(4)
    df["risk_tier"] = pd.cut(proba, bins=[0,0.30,0.60,0.80,1.0],
        labels=["Low","Medium","High","Critical"], include_lowest=True)
    return df

try:
    df, metrics, model, fnames = load_data()
    scored = score_customers(df)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.code("python data/generate_data.py && python src/train.py")
    st.stop()

at_risk    = scored[scored["churn_probability"] >= 0.60]
critical   = scored[scored["risk_tier"] == "Critical"]
savings_est = len(at_risk) * 10  # $10 avg retention value per customer saved

# ── Header ──────────────────────────────
st.markdown(f"""
<div class="db-header">
  <div>
    <div class="db-title">Customer <span>Churn</span> Prediction</div>
    <div style="font-size:0.78rem;color:{MUTED};margin-top:0.3rem;">
      Random Forest · 50K customers · identify at-risk accounts 60 days before churn
    </div>
  </div>
  <div class="db-meta">churn-prediction-model · sklearn {__import__('sklearn').__version__}</div>
</div>""", unsafe_allow_html=True)

# ── KPIs ────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><div class="kpi-accent" style="background:{CYAN}"></div>
    <div class="kpi-lbl">Model Accuracy</div>
    <div class="kpi-val">{metrics['accuracy']:.1%}</div>
    <div class="kpi-tag">ROC-AUC {metrics['roc_auc']:.3f}</div></div>
  <div class="kpi"><div class="kpi-accent" style="background:{RED}"></div>
    <div class="kpi-lbl">At-Risk Customers</div>
    <div class="kpi-val">{len(at_risk):,}</div>
    <div class="kpi-tag">probability ≥ 60%</div></div>
  <div class="kpi"><div class="kpi-accent" style="background:{AMBER}"></div>
    <div class="kpi-lbl">Critical Risk</div>
    <div class="kpi-val">{len(critical):,}</div>
    <div class="kpi-tag">act within 7 days</div></div>
  <div class="kpi"><div class="kpi-accent" style="background:{GREEN}"></div>
    <div class="kpi-lbl">Dataset Churn Rate</div>
    <div class="kpi-val">{metrics['churn_rate']:.1%}</div>
    <div class="kpi-tag">{int(metrics['churn_rate']*50000):,} of 50K churned</div></div>
</div>""", unsafe_allow_html=True)

# ── Row 1: Risk distribution + Feature importance ──
c1, c2 = st.columns([1.2, 2], gap="large")

with c1:
    st.markdown('<div class="sl">Risk Breakdown</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cc"><div class="ct">Customers by Risk Tier</div><div class="cd">Prioritize Critical + High for retention campaigns</div>', unsafe_allow_html=True)
    tier_counts = scored["risk_tier"].value_counts()
    tier_order  = ["Critical","High","Medium","Low"]
    tier_colors = {"Critical": RED, "High": AMBER, "Medium": CYAN, "Low": GREEN}
    fig1 = go.Figure(go.Pie(
        labels=[t for t in tier_order if t in tier_counts],
        values=[tier_counts.get(t, 0) for t in tier_order if t in tier_counts],
        hole=0.60,
        marker=dict(colors=[tier_colors[t] for t in tier_order if t in tier_counts],
                    line=dict(color=CARD, width=3)),
        textinfo="none",
        hovertemplate="%{label}: %{value:,} customers (%{percent})<extra></extra>",
    ))
    total_at_risk = tier_counts.get("Critical",0) + tier_counts.get("High",0)
    fig1.add_annotation(text=f"<b>{total_at_risk:,}</b><br><span style='font-size:10px'>at-risk</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT, size=14))
    fig1.update_layout(plot_bgcolor=CARD, paper_bgcolor=CARD,
        margin=dict(t=10,b=40,l=0,r=0), height=280,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10,color=SUBTEXT), orientation="h", y=-0.05, x=0.05),
        hoverlabel=dict(bgcolor=BORDER, font_size=12))
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="sl">Model Insights</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cc"><div class="ct">Top 10 Churn Drivers</div><div class="cd">Features with highest predictive power — what to fix to reduce churn</div>', unsafe_allow_html=True)
    top_features = pd.DataFrame(metrics["top_features"]).head(10)
    fig2 = go.Figure(go.Bar(
        x=top_features["importance"][::-1],
        y=top_features["feature"][::-1],
        orientation="h",
        marker=dict(color=CYAN, opacity=0.8, line=dict(width=0)),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig2.update_layout(plot_bgcolor=CARD, paper_bgcolor=CARD,
        margin=dict(t=10,b=10,l=0,r=10), height=280,
        xaxis=dict(gridcolor=BORDER, tickfont=dict(size=10,color=SUBTEXT), gridwidth=0.5, zeroline=False, linecolor=BORDER, tickformat=".3f"),
        yaxis=dict(tickfont=dict(size=10,color=SUBTEXT), linecolor=BORDER, showgrid=False),
        hoverlabel=dict(bgcolor=BORDER, font_size=12))
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── Row 2: Churn by plan + Score distribution ──
c3, c4 = st.columns([1, 1], gap="large")

with c3:
    st.markdown('<div class="sl">Segment Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cc"><div class="ct">Churn Rate by Plan Type</div><div class="cd">Starter tier highest risk · Enterprise most stable</div>', unsafe_allow_html=True)
    plan_churn = scored.groupby("plan_type")["churned"].mean().reset_index()
    plan_churn.columns = ["plan_type","churn_rate"]
    plan_colors_map = {"Starter": AMBER, "Growth": CYAN, "Enterprise": BLUE}
    fig3 = go.Figure(go.Bar(
        x=plan_churn["plan_type"],
        y=plan_churn["churn_rate"],
        marker=dict(color=[plan_colors_map.get(p, CYAN) for p in plan_churn["plan_type"]], opacity=0.85, line=dict(width=0)),
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    ))
    fig3.update_layout(plot_bgcolor=CARD, paper_bgcolor=CARD,
        margin=dict(t=10,b=10,l=0,r=0), height=250,
        xaxis=dict(gridcolor=BORDER, tickfont=dict(size=11,color=SUBTEXT), showgrid=False, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10,color=SUBTEXT), tickformat=".0%", gridwidth=0.5, zeroline=False, linecolor=BORDER),
        showlegend=False, hoverlabel=dict(bgcolor=BORDER, font_size=12), bargap=0.4)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown('<div class="sl">Score Distribution</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cc"><div class="ct">Churn Probability Distribution</div><div class="cd">Most customers low risk · long tail of high-risk accounts</div>', unsafe_allow_html=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(
        x=scored[scored["churned"]==0]["churn_probability"],
        nbinsx=40, name="Did not churn",
        marker=dict(color=BLUE, opacity=0.6, line=dict(width=0)),
        hovertemplate="Prob %{x:.2f}: %{y:,} customers<extra>No churn</extra>",
    ))
    fig4.add_trace(go.Histogram(
        x=scored[scored["churned"]==1]["churn_probability"],
        nbinsx=40, name="Churned",
        marker=dict(color=RED, opacity=0.7, line=dict(width=0)),
        hovertemplate="Prob %{x:.2f}: %{y:,} customers<extra>Churned</extra>",
    ))
    fig4.update_layout(barmode="overlay", plot_bgcolor=CARD, paper_bgcolor=CARD,
        margin=dict(t=10,b=10,l=0,r=0), height=250,
        xaxis=dict(gridcolor=BORDER, tickfont=dict(size=10,color=SUBTEXT), showgrid=False, linecolor=BORDER, title=dict(text="Churn Probability", font=dict(size=10, color=SUBTEXT))),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10,color=SUBTEXT), gridwidth=0.5, zeroline=False, linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10,color=SUBTEXT), orientation="h", y=1.1, x=0),
        hoverlabel=dict(bgcolor=BORDER, font_size=12))
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── At-risk customer table ──
st.markdown('<div class="sl">At-Risk Customers</div>', unsafe_allow_html=True)
st.markdown(f'<div class="cc"><div class="ct">Top 20 Highest Risk Customers — Prioritize for Immediate Outreach</div><div class="cd">Sorted by churn probability · Critical tier = act within 7 days</div>', unsafe_allow_html=True)

display_cols = ["customer_id","plan_type","contract_length","monthly_charges","tenure_months","days_since_login","support_tickets_90d","churn_probability","risk_tier"]
top20 = scored.nlargest(20, "churn_probability")[display_cols].copy()
top20["churn_probability"] = top20["churn_probability"].apply(lambda x: f"{x:.1%}")
top20["monthly_charges"]   = top20["monthly_charges"].apply(lambda x: f"${x:.0f}")

st.dataframe(top20.reset_index(drop=True), use_container_width=True,
    column_config={
        "customer_id":        st.column_config.TextColumn("Customer"),
        "plan_type":          st.column_config.TextColumn("Plan"),
        "contract_length":    st.column_config.TextColumn("Contract"),
        "monthly_charges":    st.column_config.TextColumn("MRR"),
        "tenure_months":      st.column_config.NumberColumn("Tenure (mo)"),
        "days_since_login":   st.column_config.NumberColumn("Days Inactive"),
        "support_tickets_90d":st.column_config.NumberColumn("Support Tickets"),
        "churn_probability":  st.column_config.TextColumn("Churn Prob"),
        "risk_tier":          st.column_config.TextColumn("Risk Tier"),
    })
st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ──
st.markdown(f"""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid {BORDER};
     display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:0.7rem;color:{MUTED};font-family:'DM Mono',monospace;">
    Random Forest · 5-fold CV · 50K synthetic records · sklearn
  </span>
  <a href="https://github.com/PoojithaJ08/customer-churn-prediction"
     style="font-size:0.7rem;color:{MUTED};text-decoration:none;font-family:'DM Mono',monospace;">
    github ↗
  </a>
</div>""", unsafe_allow_html=True)
