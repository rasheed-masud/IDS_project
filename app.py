"""
Fitbit Calories Predictor 
Uses: Streamlit, Plotly, RandomForest (trained in model.py)
Theme: Fitbit (Teal + Purple)
Author: IDS Project (Final Version)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
META_PATH = "meta.json"
DATA_PATH = "dailyActivity_merged.csv"

# ----------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------
st.set_page_config(
    page_title="Fitbit Calories Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------
# CSS — Fitbit Teal/Purple Theme
# ----------------------------------------------------
st.markdown(
    """
    <style>
    :root{
      --fitbit-teal: #1DB3B1;
      --fitbit-dark: #071428;
      --fitbit-accent: #6A5AE0;
      --card-bg: rgba(255,255,255,0.03);
    }
    .app-header {
        border-radius: 12px;
        padding: 18px;
        background: linear-gradient(90deg, rgba(29,179,177,0.08), rgba(106,90,224,0.06));
        margin-bottom: 16px;
    }
    .big-title {
        font-size:28px;
        font-weight:700;
        color:#0b2b2b;
    }
    .subtitle {
        color: #243b3b;
        margin-top:6px;
    }
    .card {
        border-radius:12px;
        padding:14px;
        box-shadow: 0 6px 18px rgba(12,40,60,0.06);
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(250,250,255,0.96));
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "ActivityDate" in df.columns:
        df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    return df

@st.cache_resource
def load_model_meta():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, scaler, meta

def predict_one(model, scaler, meta, input_values):
    feats = meta["features"]
    X = np.array([input_values[f] for f in feats]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0])

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ----------------------------------------------------
# LOAD DATA & MODEL
# ----------------------------------------------------
df = load_data()
model, scaler, meta = load_model_meta()
features = meta["features"]
metrics = meta.get("metrics", {})

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='card'><h3 style='margin:0;color:var(--fitbit-dark)'>Fitbit Calories Predictor</h3>"
        "<div style='color:#63738a;font-size:13px;'>IDS f24 — Final Project</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

    if st.button("Reload all"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Theme: Fitbit (teal + purple)")

# ----------------------------------------------------
# HEADER — NO LOGO
# ----------------------------------------------------
st.markdown(
    """
    <div class='app-header'>
        <div style='display:flex;justify-content:space-between;align-items:center'>
            <div>
                <div class='big-title'>Fitbit Calories Predictor</div>
                <div class='subtitle'>Interactive ML app — Predict daily calories burned</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab_overview, tab_eda, tab_model, tab_predict, tab_about = st.tabs(
    ["Overview", "EDA", "Model", "Predict", "About"]
)

# ====================================================
# TAB 1 — OVERVIEW
# ====================================================
with tab_overview:
    st.markdown("### Dataset Overview")

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])
        if "ActivityDate" in df.columns:
            st.write("Date range:", f"{df['ActivityDate'].min().date()} → {df['ActivityDate'].max().date()}")
        st.dataframe(df.head(6), use_container_width=True)

    with c2:
        st.markdown("**Feature Means (Defaults)**")
        st.json(meta.get("feature_means", {}))

    with c3:
        st.markdown("**Model Metrics**")
        chosen = metrics.get("rf") or metrics.get("linear") or {}
        st.metric("MAE", round(chosen.get("mae", 0), 2))
        st.metric("RMSE", round(chosen.get("rmse", 0), 2))
        st.metric("R²", round(chosen.get("r2", 0), 3))

    st.markdown("---")
    st.markdown("### Quick Stats")

    avg_cal = df["Calories"].mean()
    top_steps = df["TotalSteps"].max()

    colA, colB, colC = st.columns(3)
    colA.info(f"Avg Calories: {avg_cal:.1f} kcal")
    colB.success(f"Max Steps (Day): {int(top_steps):,}")
    if "SedentaryMinutes" in df.columns:
        colC.warning(f"Avg Sedentary: {df['SedentaryMinutes'].mean():.1f} min")

# ====================================================
# TAB 2 — EDA
# ====================================================
with tab_eda:
    st.markdown("## Exploratory Data Analysis")

    # date filter
    if "ActivityDate" in df.columns:
        min_d = df["ActivityDate"].min().date()
        max_d = df["ActivityDate"].max().date()
        dr = st.date_input("Filter Date Range", [min_d, max_d])
        df_eda = df[(df["ActivityDate"] >= pd.to_datetime(dr[0])) &
                    (df["ActivityDate"] <= pd.to_datetime(dr[1]))]
    else:
        df_eda = df

    # Histograms
    st.markdown("### Distribution of Key Features")
    hist_cols = ["TotalSteps", "TotalDistance", "Calories"]
    fig = go.Figure()
    for c in hist_cols:
        if c in df_eda.columns:
            fig.add_trace(go.Histogram(x=df_eda[c], name=c, opacity=0.7))
    fig.update_layout(barmode="overlay", height=380)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter Plot
    if "TotalSteps" in df_eda.columns:
        st.markdown("### Calories vs TotalSteps")
        fig2 = px.scatter(
            df_eda,
            x="TotalSteps",
            y="Calories",
            hover_data=["ActivityDate"] if "ActivityDate" in df_eda else None,
            trendline="ols",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation Heatmap
    st.markdown("### Correlation Heatmap")
    corr = df_eda[[c for c in features + ["Calories"] if c in df_eda]].corr()
    fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                     color_continuous_scale=px.colors.diverging.RdBu)
    st.plotly_chart(fig3, use_container_width=True)

    # Trendline
    if "ActivityDate" in df_eda.columns:
        st.markdown("### Average Calories Over Time")
        ts = df_eda.groupby("ActivityDate")["Calories"].mean().reset_index()
        fig4 = px.line(ts, x="ActivityDate", y="Calories", markers=True)
        st.plotly_chart(fig4, use_container_width=True)

# ====================================================
# TAB 3 — MODEL
# ====================================================
with tab_model:
    st.markdown("## Model Details")
    st.write("Loaded Model:", meta.get("model_type", "Unknown"))
    st.json(metrics)

    if meta.get("model_type", "").lower().startswith("randomforest"):
        try:
            imp = model.feature_importances_
            fi = pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance")
            fig_fi = px.bar(fi, x="importance", y="feature", orientation="h")
            st.plotly_chart(fig_fi, use_container_width=True)
        except:
            st.error("Could not load feature importances.")

    with st.expander("Download Model Files"):
        try:
            st.download_button(
                "Download model.pkl",
                data=open(MODEL_PATH, "rb").read(),
                file_name="model.pkl",
            )
            st.download_button(
                "Download meta.json",
                data=open(META_PATH, "r").read(),
                file_name="meta.json",
            )
        except:
            st.error("One or more files missing.")

# ====================================================
# TAB 4 — PREDICTION
# ====================================================
with tab_predict:
    st.markdown("## Live Prediction")

    defaults = meta.get("feature_means", {})
    cols = st.columns(2)
    user_inputs = {}

    for i, feat in enumerate(features):
        default = float(defaults.get(feat, 0))
        with cols[i % 2]:
            user_inputs[feat] = st.number_input(feat, value=default, format="%.2f")

    if st.button("Predict Calories"):
        try:
            pred = predict_one(model, scaler, meta, user_inputs)
            avg = df["Calories"].mean()
            pct = min(100, max(0, pred / avg * 100))

            st.success(f"Predicted Calories: **{pred:.1f} kcal**")
            st.progress(int(pct))
            st.write(f"{pct:.0f}% of dataset average ({avg:.1f} kcal).")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.markdown("### Batch Prediction (CSV Upload)")

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.dataframe(df_up.head())

            if all(f in df_up.columns for f in features):
                if st.button("Run Batch Prediction"):
                    X = df_up[features].fillna(0).values
                    Xs = scaler.transform(X)
                    preds = model.predict(Xs)
                    df_up["PredictedCalories"] = preds

                    st.success("Batch prediction complete.")
                    st.dataframe(df_up.head())

                    st.download_button(
                        "Download predictions.csv",
                        data=df_to_csv_bytes(df_up),
                        file_name="predictions.csv",
                    )
            else:
                st.warning("Uploaded CSV missing required features.")
        except Exception as e:
            st.error(f"Batch prediction error: {e}")
    else:
        st.info("Upload a CSV from the sidebar to enable batch prediction.")

# ====================================================
# TAB 5 — ABOUT
# ====================================================
with tab_about:
    st.markdown("## About this Project")
    st.write(
        """
        **Course:** IDS f24  
        **Task:** Predict daily calories burned using Fitbit activity data  
        **Includes:** EDA, Model Training, RandomForest, Streamlit App
        """
    )
     
