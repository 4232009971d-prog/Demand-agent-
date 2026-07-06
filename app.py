"""
Forecast Challenge Agent
Questions every forecast and validates assumptions before they become expensive mistakes.

Checks:
  1. Historical trend alignment
  2. Promotional uplift check
  3. Normal variation analysis
  4. Assumption validation (AI reasoning via Claude)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import re
import anthropic

st.set_page_config(page_title="Forecast Challenge Agent", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "col_map" not in st.session_state:
    st.session_state.col_map = {}
if "flags" not in st.session_state:
    st.session_state.flags = None
if "ai_notes" not in st.session_state:
    st.session_state.ai_notes = {}

# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------
CANDIDATES = {
    "date": ["date", "period", "month", "week", "ds"],
    "product": ["sku", "product", "item", "material", "product_code", "item_code"],
    "forecast": ["forecast", "forecasted", "forecast_qty", "fcst", "planned", "plan_qty", "yhat"],
    "actual": ["actual", "actual_qty", "sales", "sold", "history", "historical", "demand", "y"],
    "promo": ["promo", "promotion", "promo_flag", "on_promo", "is_promo", "campaign"],
}

def auto_detect_columns(columns):
    detected = {}
    lower_cols = {c: c.lower().strip().replace(" ", "_") for c in columns}
    for field, keywords in CANDIDATES.items():
        match = None
        for orig, low in lower_cols.items():
            if any(k == low for k in keywords):
                match = orig
                break
        if not match:
            for orig, low in lower_cols.items():
                if any(k in low for k in keywords):
                    match = orig
                    break
        detected[field] = match
    return detected

# ---------------------------------------------------------------------------
# Statistical checks
# ---------------------------------------------------------------------------
def run_trend_check(group, forecast_col, actual_col, date_col):
    """Compare forecast vs linear trend extrapolated from history."""
    hist = group.dropna(subset=[actual_col]).sort_values(date_col)
    if len(hist) < 3:
        return None, "Not enough history for trend analysis"
    x = np.arange(len(hist))
    y = hist[actual_col].values
    slope, intercept = np.polyfit(x, y, 1)
    trend_next = slope * len(hist) + intercept

    fcst_rows = group[group[forecast_col].notna()].sort_values(date_col)
    if fcst_rows.empty:
        return None, None
    fcst_val = fcst_rows[forecast_col].iloc[-1]

    if trend_next <= 0:
        pct_dev = 0
    else:
        pct_dev = (fcst_val - trend_next) / trend_next * 100

    flagged = abs(pct_dev) > 25
    detail = f"Trend-implied value ≈ {trend_next:,.0f}, forecast = {fcst_val:,.0f} ({pct_dev:+.1f}% vs trend)"
    return flagged, detail


def run_promo_check(group, forecast_col, actual_col, promo_col):
    """Flag forecasts with uplift not matched to a promo flag, or promo with no uplift."""
    if promo_col not in group.columns:
        return None, None
    hist = group.dropna(subset=[actual_col])
    if hist.empty or promo_col not in hist.columns:
        return None, None

    non_promo_avg = hist[hist[promo_col].isin([0, False, "N", "No", "n", "no"])][actual_col].mean()
    promo_avg = hist[hist[promo_col].isin([1, True, "Y", "Yes", "y", "yes"])][actual_col].mean()

    fcst_rows = group[group[forecast_col].notna()]
    if fcst_rows.empty:
        return None, None
    latest = fcst_rows.iloc[-1]
    fcst_val = latest[forecast_col]
    is_promo = latest.get(promo_col) in [1, True, "Y", "Yes", "y", "yes"]

    if pd.isna(non_promo_avg):
        return None, None

    uplift_pct = (fcst_val - non_promo_avg) / non_promo_avg * 100 if non_promo_avg else 0

    if is_promo and uplift_pct < 10:
        return True, f"Marked as promo period but forecast shows only {uplift_pct:+.1f}% vs non-promo baseline ({non_promo_avg:,.0f})"
    if not is_promo and uplift_pct > 25:
        return True, f"No promo flagged, but forecast is {uplift_pct:+.1f}% above non-promo baseline ({non_promo_avg:,.0f}) — unexplained uplift"
    return False, f"Uplift vs baseline: {uplift_pct:+.1f}%"


def run_variation_check(group, forecast_col, actual_col, date_col):
    """Flag forecasts outside historical mean ± 2 std dev."""
    hist = group.dropna(subset=[actual_col])
    if len(hist) < 4:
        return None, "Not enough history for variation analysis"
    mean = hist[actual_col].mean()
    std = hist[actual_col].std()
    fcst_rows = group[group[forecast_col].notna()].sort_values(date_col)
    if fcst_rows.empty:
        return None, None
    fcst_val = fcst_rows[forecast_col].iloc[-1]

    if std == 0:
        z = 0
    else:
        z = (fcst_val - mean) / std

    flagged = abs(z) > 2
    detail = f"Historical mean {mean:,.0f} ± {std:,.0f} (1σ). Forecast {fcst_val:,.0f} is {z:+.1f}σ from mean"
    return flagged, detail


def run_all_checks(df, col_map):
    date_col = col_map["date"]
    product_col = col_map["product"]
    forecast_col = col_map["forecast"]
    actual_col = col_map["actual"]
    promo_col = col_map.get("promo")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    results = []
    groups = df.groupby(product_col) if product_col else [("All", df)]

    for name, group in groups:
        trend_flag, trend_detail = run_trend_check(group, forecast_col, actual_col, date_col)
        var_flag, var_detail = run_variation_check(group, forecast_col, actual_col, date_col)
        promo_flag, promo_detail = (None, None)
        if promo_col:
            promo_flag, promo_detail = run_promo_check(group, forecast_col, actual_col, promo_col)

        checks = []
        if trend_flag is not None:
            checks.append(("Historical Trend Alignment", trend_flag, trend_detail))
        if var_flag is not None:
            checks.append(("Normal Variation Analysis", var_flag, var_detail))
        if promo_flag is not None:
            checks.append(("Promotional Uplift Check", promo_flag, promo_detail))

        risk_score = sum(1 for _, f, _ in checks if f)

        results.append({
            "product": name,
            "checks": checks,
            "risk_score": risk_score,
            "n_checks_flagged": risk_score,
        })

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# AI assumption validation
# ---------------------------------------------------------------------------
def get_ai_reasoning(product, checks, api_key=None):
    """Ask Claude to reason about why flagged items look risky."""
    flagged_checks = [(name, detail) for name, flag, detail in checks if flag]
    if not flagged_checks:
        return "No red flags — forecast looks statistically consistent with history."

    check_summary = "\n".join(f"- {name}: {detail}" for name, detail in flagged_checks)

    prompt = f"""You are a supply chain forecast analyst. A forecast for "{product}" triggered the following statistical red flags:

{check_summary}

In 2-3 concise sentences, explain in plain English why this forecast looks risky and what assumption the planner should double-check before finalizing it. Be specific and actionable, not generic."""

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"(AI reasoning unavailable: {e})"

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📊 Forecast Challenge Agent")
st.caption("Questions every forecast and validates assumptions before they become expensive mistakes.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Claude API Key", type="password", help="Needed for AI assumption validation. Get one at console.anthropic.com")
    st.markdown("---")
    st.markdown("**Checks performed:**")
    st.markdown("- Historical trend alignment\n- Promotional uplift check\n- Normal variation analysis\n- AI assumption validation")

uploaded_file = st.file_uploader("Upload forecast file (Excel or CSV)", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

if st.session_state.df is not None:
    df = st.session_state.df
    st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    detected = auto_detect_columns(df.columns.tolist())

    st.subheader("Column Mapping")
    st.caption("Auto-detected — override any that look wrong")

    cols = st.columns(5)
    col_options = ["(none)"] + df.columns.tolist()

    field_labels = {
        "date": "Date / Period",
        "product": "Product / SKU",
        "forecast": "Forecast Qty",
        "actual": "Actual / Historical Qty",
        "promo": "Promo Flag (optional)",
    }

    col_map = {}
    for i, (field, label) in enumerate(field_labels.items()):
        default = detected.get(field)
        default_idx = col_options.index(default) if default in col_options else 0
        with cols[i]:
            selected = st.selectbox(label, col_options, index=default_idx, key=f"map_{field}")
        col_map[field] = None if selected == "(none)" else selected

    required_ok = all(col_map[f] for f in ["date", "forecast", "actual"])

    if not required_ok:
        st.warning("Please map Date, Forecast Qty, and Actual Qty columns to continue.")
    else:
        if st.button("🔍 Run Forecast Challenge", type="primary"):
            with st.spinner("Running statistical checks..."):
                results = run_all_checks(df, col_map)
                st.session_state.flags = results
                st.session_state.col_map = col_map

            if api_key:
                with st.spinner("Getting AI reasoning on flagged items..."):
                    for _, row in results.iterrows():
                        if row["risk_score"] > 0:
                            note = get_ai_reasoning(row["product"], row["checks"], api_key)
                            st.session_state.ai_notes[row["product"]] = note

if st.session_state.flags is not None:
    results = st.session_state.flags
    col_map = st.session_state.col_map

    st.markdown("---")
    st.subheader("Results")

    total = len(results)
    flagged_count = (results["risk_score"] > 0).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Products Analyzed", total)
    c2.metric("Flagged for Review", flagged_count)
    c3.metric("Clean Forecasts", total - flagged_count)

    sorted_results = results.sort_values("risk_score", ascending=False)

    for _, row in sorted_results.iterrows():
        risk = row["risk_score"]
        icon = "🔴" if risk >= 2 else ("🟡" if risk == 1 else "🟢")

        with st.expander(f"{icon} {row['product']} — {risk} flag(s)", expanded=(risk > 0)):
            for name, flag, detail in row["checks"]:
                if flag is None:
                    continue
                mark = "⚠️" if flag else "✅"
                st.markdown(f"{mark} **{name}**: {detail}")

            if risk > 0:
                note = st.session_state.ai_notes.get(row["product"])
                if note:
                    st.info(f"**AI Assessment:** {note}")
                elif not api_key:
                    st.caption("Add a Claude API key in the sidebar to get AI reasoning on this flag.")

            # Chart
            product_col = col_map.get("product")
            if product_col:
                group = df[df[product_col] == row["product"]].copy()
            else:
                group = df.copy()
            group[col_map["date"]] = pd.to_datetime(group[col_map["date"]], errors="coerce")
            group = group.sort_values(col_map["date"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=group[col_map["date"]], y=group[col_map["actual"]],
                                      mode="lines+markers", name="Actual", line=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=group[col_map["date"]], y=group[col_map["forecast"]],
                                      mode="lines+markers", name="Forecast", line=dict(color="#ff7f0e", dash="dash")))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown("---")
    export_rows = []
    for _, row in sorted_results.iterrows():
        for name, flag, detail in row["checks"]:
            if flag:
                export_rows.append({
                    "Product": row["product"],
                    "Check": name,
                    "Detail": detail,
                    "AI Assessment": st.session_state.ai_notes.get(row["product"], ""),
                })
    if export_rows:
        export_df = pd.DataFrame(export_rows)
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Flag Report (CSV)", csv, "forecast_flags.csv", "text/csv")
    else:
        st.success("No flags to export — all forecasts passed the checks.")
else:
    if st.session_state.df is None:
        st.info("👆 Upload a forecast file to get started")
