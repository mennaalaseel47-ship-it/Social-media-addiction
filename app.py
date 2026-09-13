import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Social Media Addiction Predictor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
PIPELINE_PATH = APP_DIR / "rf_pipeline.pkl"
METADATA_PATH = APP_DIR / "metadata.json"
EDA_SUMMARY_PATH = APP_DIR / "eda_summary.json"


MODEL_COMPARISON = pd.DataFrame(
    [
        ["Random Forest", 0.0889, 0.0542, 0.2329, 0.9796],
        ["KNN", 0.1234, 0.0757, 0.2752, 0.9715],
        ["Decision Tree (Scaled)", 0.0851, 0.0851, 0.2917, 0.9680],
        ["Decision Tree (Raw)", 0.0851, 0.0851, 0.2917, 0.9680],
        ["RBF SVR (Scaled)", 0.1487, 0.0895, 0.2992, 0.9663],
        ["Linear SVR (Scaled)", 0.1974, 0.1095, 0.3309, 0.9588],
        ["Linear SVR (Raw)", 0.1972, 0.1096, 0.3310, 0.9588],
        ["Decision Tree (Pre-Pruned)", 0.1847, 0.1133, 0.3366, 0.9574],
        ["RBF SVR (Raw)", 0.1589, 0.1140, 0.3376, 0.9571],
    ],
    columns=["Model", "MAE", "MSE", "RMSE", "R2"],
)


# --------------------------------------------------------------------------
# Design system — a small "digital wellbeing" palette: dusk-toned ink and
# paper, a teal accent for calm/healthy states, amber and clay for rising
# and high risk. Fraunces carries headlines, Inter carries everything else.
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink: #20232B;
    --paper: #F6F5F1;
    --panel: #FFFFFF;
    --line: #E4E1D8;
    --teal: #1F6F6B;
    --teal-soft: #E4F0EE;
    --amber: #C98A2B;
    --amber-soft: #FBEEDA;
    --clay: #B0502F;
    --clay-soft: #F7E7DF;
    --muted: #6B6E76;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.stApp {
    background-color: var(--paper);
}

h1, h2, h3 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    letter-spacing: -0.01em;
}

section[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px solid var(--line);
}

.hero {
    padding: 0.25rem 0 1.25rem 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1.75rem;
}
.hero h1 {
    font-size: 2.3rem;
    margin-bottom: 0.35rem;
}
.hero p {
    color: var(--muted);
    font-size: 1.02rem;
    max-width: 620px;
    line-height: 1.5;
}

.stat-row { display: flex; gap: 0.9rem; margin: 1rem 0 1.6rem 0; flex-wrap: wrap; }
.stat-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    flex: 1;
    min-width: 140px;
}
.stat-card .num {
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--teal);
    line-height: 1.1;
}
.stat-card .label {
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 0.2rem;
}

.section-block {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.3rem;
}
.section-block h3 { margin-top: 0; font-size: 1.15rem; }
.section-block ul { margin-bottom: 0; padding-left: 1.2rem; }
.section-block li { margin-bottom: 0.45rem; line-height: 1.5; color: #383B42; }

.feature-pill {
    display: inline-block;
    background: var(--teal-soft);
    color: var(--teal);
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    margin: 0.15rem 0.3rem 0.15rem 0;
    font-size: 0.85rem;
    font-weight: 500;
}

.gauge-wrap {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.4rem 1.6rem 1.1rem 1.6rem;
    margin-top: 0.6rem;
}
.gauge-title {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    color: var(--muted);
    margin-bottom: 0.2rem;
}
.gauge-score {
    font-family: 'Fraunces', serif;
    font-size: 2.6rem;
    font-weight: 600;
}
.gauge-track {
    position: relative;
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--teal) 0%, var(--teal) 33%, var(--amber) 33%, var(--amber) 66%, var(--clay) 66%, var(--clay) 100%);
    margin: 0.9rem 0 0.5rem 0;
}
.gauge-marker {
    position: absolute;
    top: -6px;
    width: 3px;
    height: 22px;
    background: var(--ink);
    border-radius: 2px;
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    color: var(--muted);
    font-size: 0.78rem;
}
.risk-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-weight: 600;
    font-size: 0.9rem;
    margin-top: 0.6rem;
}
.risk-low { background: var(--teal-soft); color: var(--teal); }
.risk-mid { background: var(--amber-soft); color: var(--amber); }
.risk-high { background: var(--clay-soft); color: var(--clay); }

.caption-muted { color: var(--muted); font-size: 0.85rem; margin-top: 0.6rem; }

div[data-testid="stForm"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.3rem 1.5rem 0.6rem 1.5rem;
}

.stButton > button, div[data-testid="stFormSubmitButton"] button {
    background: var(--ink);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    background: var(--teal);
    color: white;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    if not PIPELINE_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    pipeline = joblib.load(PIPELINE_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return pipeline, metadata


@st.cache_data
def load_eda_summary():
    if not EDA_SUMMARY_PATH.exists():
        return None
    with open(EDA_SUMMARY_PATH) as f:
        return json.load(f)



def get_feature_importance(pipeline):
    """Map Random Forest feature importances back to readable names."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        names = preprocessor.get_feature_names_out()
        imp = pd.DataFrame({"Feature": names, "Importance": model.feature_importances_})
        imp["Feature"] = (imp["Feature"].str.replace("ordinal__", "", regex=False)
                           .str.replace("onehot__", "", regex=False)
                           .str.replace("numeric__", "", regex=False)
                           .str.replace("_", " ", regex=False))
        return imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    except Exception:
        return None

def render_gauge(score: float, lo: float = 1.0, hi: float = 9.0) -> str:
    """Build an inline HTML gauge for the 1-9 addiction score scale."""
    pct = max(0.0, min(1.0, (score - lo) / (hi - lo))) * 100

    if score < 4:
        risk_label, risk_class = "Low predicted risk", "risk-low"
    elif score < 7:
        risk_label, risk_class = "Moderate predicted risk", "risk-mid"
    else:
        risk_label, risk_class = "High predicted risk", "risk-high"

    return f"""
    <div class="gauge-wrap">
        <div class="gauge-title">Predicted addiction score</div>
        <div class="gauge-score">{score:.1f} <span style="font-size:1.1rem;color:var(--muted);font-family:'Inter',sans-serif;">/ 9</span></div>
        <div class="gauge-track">
            <div class="gauge-marker" style="left: calc({pct}% - 2px);"></div>
        </div>
        <div class="gauge-labels"><span>1 · Low</span><span>5 · Moderate</span><span>9 · High</span></div>
        <span class="risk-badge {risk_class}">{risk_label}</span>
    </div>
    """


pipeline, metadata = load_artifacts()
eda = load_eda_summary()

# ------------------------------------------------------------------ Sidebar --
with st.sidebar:
    st.markdown("### 🌙 Digital Wellbeing")
    st.caption("Social media addiction predictor")
    st.markdown("---")
    st.markdown(
        "A regression model trained on the **Student Social Media Addiction** "
        "dataset (Kaggle), estimating a self-reported addiction score from "
        "usage, sleep, and academic/lifestyle context."
    )
    if metadata:
        m = metadata["test_metrics"]
        st.markdown("**Deployed model**")
        st.markdown(f"Random Forest · R² {m['R2']} · RMSE {m['RMSE']}")
    st.markdown("---")
    st.caption(
        "Educational project — not a clinical or diagnostic tool. "
        "Predictions reflect patterns in survey data, not individual assessment."
    )

# --------------------------------------------------------------------- Hero --
st.markdown(
    """
    <div class="hero">
        <h1>Student Social Media Addiction Predictor</h1>
        <p>Estimate a student's social media addiction score from demographic,
        academic, and usage habits — and see which patterns the underlying
        data actually supports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_predict = st.tabs(["Project Overview", "Predict My Score"])

# ---------------------------------------------------------------- Overview --
with tab_overview:
    n_students = eda["n_students"] if eda else 705
    metrics = metadata["test_metrics"] if metadata else {"R2": 0.9796, "RMSE": 0.2329, "MAE": 0.0889}

    st.markdown("### Dataset & model at a glance")
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card"><div class="num">{n_students}</div><div class="label">Students after cleaning</div></div>
      <div class="stat-card"><div class="num">8</div><div class="label">Model features</div></div>
      <div class="stat-card"><div class="num">{metrics['R2']:.3f}</div><div class="label">Random Forest R²</div></div>
      <div class="stat-card"><div class="num">{metrics['RMSE']:.3f}</div><div class="label">Random Forest RMSE</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-block">
      <h3>EDA dashboard</h3>
      <p style="color:#383B42;line-height:1.55;margin-bottom:0;">
      Explore the target distribution, strongest dataset-level relationships,
      platform differences, correlations, model performance, and the features
      the deployed Random Forest relies on most.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if eda is None:
        st.warning("No `eda_summary.json` found. Run `train_model.py` first to generate the EDA statistics.")
    else:
        st.markdown("### 1. Target distribution")
        score_df = pd.DataFrame({"Students": eda["addiction_score_hist"]["counts"]},
                                index=eda["addiction_score_hist"]["labels"])
        st.bar_chart(score_df, height=250)
        st.caption("Distribution of the target `Addicted_Score` on the 1–9 scale.")

        st.markdown("### 2. Key relationships")
        c1, c2 = st.columns(2)
        with c1:
            usage_df = pd.DataFrame({"Average addiction score": eda["usage_vs_addiction"]["avg_addiction"]},
                                    index=eda["usage_vs_addiction"]["labels"])
            st.bar_chart(usage_df, height=280)
            st.caption("Average addiction score by daily usage bucket.")
        with c2:
            age_df = pd.DataFrame({"High-addiction rate (%)": eda["age_vs_addiction"]["addiction_rate_pct"]},
                                  index=[str(x) for x in eda["age_vs_addiction"]["ages"]])
            st.line_chart(age_df, height=280)
            st.caption("Share of students with `Addicted_Score ≥ 7` by age.")

        st.markdown("### 3. Platform differences")
        platform_df = pd.DataFrame({
            "Average usage hours": eda["platform_stats"]["avg_usage_hours"],
            "Average addiction score": eda["platform_stats"]["avg_addiction_score"],
        }, index=eda["platform_stats"]["platforms"])
        pc1, pc2 = st.columns(2)
        with pc1:
            st.bar_chart(platform_df[["Average addiction score"]].sort_values("Average addiction score", ascending=False), height=300)
        with pc2:
            st.bar_chart(platform_df[["Average usage hours"]].sort_values("Average usage hours", ascending=False), height=300)
        st.caption("Usage intensity and addiction score are shown separately because they use different scales.")

        st.markdown("### 4. Numeric correlations")
        corr_df = pd.DataFrame(eda["correlation"]["matrix"], columns=eda["correlation"]["columns"], index=eda["correlation"]["columns"])
        cc1, cc2 = st.columns([1.35, 1])
        with cc1:
            try:
                st.dataframe(corr_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}"), use_container_width=True, height=285)
            except Exception:
                st.dataframe(corr_df, use_container_width=True, height=285)
        with cc2:
            target_corr = corr_df["Addicted_Score"].drop("Addicted_Score").sort_values(key=lambda s: s.abs(), ascending=False).to_frame("Correlation with addiction score")
            st.dataframe(target_corr.style.format("{:.3f}"), use_container_width=True, height=285)
            st.caption("Ranked by absolute correlation with the target. Correlation is not causation.")

        st.markdown("### 5. Model comparison")
        st.caption("Same held-out test set: lower MAE/RMSE is better; higher R² is better.")
        ranked = MODEL_COMPARISON.sort_values("RMSE").reset_index(drop=True)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.bar_chart(ranked.set_index("Model")[["RMSE"]], height=350)
            st.caption("RMSE — lower is better.")
        with mc2:
            st.bar_chart(ranked.set_index("Model")[["R2"]], height=350)
            st.caption("R² — higher is better.")

        best = ranked.iloc[0]
        st.markdown(f"""
        <div class="section-block">
          <h3>Why Random Forest?</h3>
          <p style="color:#383B42;line-height:1.55;margin-bottom:0;">
          <strong>{best['Model']}</strong> achieved the lowest RMSE
          (<strong>{best['RMSE']:.4f}</strong>) and highest R²
          (<strong>{best['R2']:.4f}</strong>) in the supplied comparison, so it is the deployed model.
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(ranked.style.format({"MAE":"{:.4f}","MSE":"{:.4f}","RMSE":"{:.4f}","R2":"{:.4f}"}).highlight_min(subset=["MAE","MSE","RMSE"], color="#E4F0EE").highlight_max(subset=["R2"], color="#E4F0EE"), use_container_width=True, hide_index=True)

        importance = get_feature_importance(pipeline) if pipeline is not None else None
        if importance is not None:
            st.markdown("### 6. Random Forest feature importance")
            st.caption("Relative importance inside the trained forest; this is not a causal effect.")
            st.bar_chart(importance.head(10).set_index("Feature")[["Importance"]], height=330)
            with st.expander("See full feature-importance table"):
                st.dataframe(importance.style.format({"Importance":"{:.4f}"}), use_container_width=True, hide_index=True)

        st.markdown("### 7. Modeling decisions")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("""<div class="section-block"><h3>Features kept</h3><p style="color:#383B42;line-height:1.55;">Age · Gender · Academic Level · Most Used Platform · Affects Academic Performance · Mental Health Score · Relationship Status · Usage/Sleep Ratio</p></div>""", unsafe_allow_html=True)
        with d2:
            st.markdown("""<div class="section-block"><h3>Feature engineering</h3><ul><li><code>Country</code> was removed after EDA showed no meaningful relationship with the target.</li><li><code>Conflicts_Over_Social_Media</code> was removed because of redundancy with <code>Mental_Health_Score</code>.</li><li><code>Usage_Sleep_Ratio</code> combines daily usage and sleep.</li></ul></div>""", unsafe_allow_html=True)

        st.caption("Charts use aggregated EDA statistics generated by `train_model.py`; individual student rows are not shipped in `eda_summary.json`.")

# ------------------------------------------------------------------ Predict --
with tab_predict:
    if pipeline is None:
        st.error(
            "Model files not found. Run `train_model.py` (with `dataset.csv` "
            "in this folder) first — it produces `rf_pipeline.pkl` and "
            "`metadata.json` that this app needs."
        )
    else:
        st.markdown("#### Tell us about yourself")
        opts = metadata["dropdown_options"]
        ranges = metadata["raw_ranges"]

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider(
                    "Age",
                    min_value=int(ranges["Age"][0]),
                    max_value=int(ranges["Age"][1]),
                    value=int((ranges["Age"][0] + ranges["Age"][1]) // 2),
                )
                gender = st.selectbox("Gender", opts["Gender"])
                academic_level = st.selectbox("Academic level", opts["Academic_Level"])
                platform = st.selectbox("Most used platform", opts["Most_Used_Platform"])

            with col2:
                affects_academics = st.selectbox(
                    "Does social media affect your academic performance?",
                    opts["Affects_Academic_Performance"],
                )
                relationship_status = st.selectbox(
                    "Relationship status", opts["Relationship_Status"]
                )
                mental_health = st.slider(
                    "Mental health score (1 = poor, 10 = excellent)",
                    min_value=1,
                    max_value=10,
                    value=int((ranges["Mental_Health_Score"][0] + ranges["Mental_Health_Score"][1]) // 2),
                )

            st.markdown("**Usage & sleep**")
            col3, col4 = st.columns(2)
            with col3:
                usage_hours = st.slider(
                    "Average daily social media usage (hours)",
                    min_value=float(ranges["Avg_Daily_Usage_Hours"][0]),
                    max_value=float(ranges["Avg_Daily_Usage_Hours"][1]),
                    value=round(
                        (ranges["Avg_Daily_Usage_Hours"][0] + ranges["Avg_Daily_Usage_Hours"][1]) / 2, 1
                    ),
                    step=0.1,
                )
            with col4:
                sleep_hours = st.slider(
                    "Sleep per night (hours)",
                    min_value=float(ranges["Sleep_Hours_Per_Night"][0]),
                    max_value=float(ranges["Sleep_Hours_Per_Night"][1]),
                    value=round(
                        (ranges["Sleep_Hours_Per_Night"][0] + ranges["Sleep_Hours_Per_Night"][1]) / 2, 1
                    ),
                    step=0.1,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Predict my addiction score", use_container_width=True)

        if submitted:
            usage_sleep_ratio = usage_hours / sleep_hours

            input_row = pd.DataFrame(
                [
                    {
                        "Age": age,
                        "Gender": gender,
                        "Academic_Level": academic_level,
                        "Most_Used_Platform": platform,
                        "Affects_Academic_Performance": affects_academics,
                        "Mental_Health_Score": mental_health,
                        "Relationship_Status": relationship_status,
                        "Usage_Sleep_Ratio": usage_sleep_ratio,
                    }
                ]
            )

            prediction = float(pipeline.predict(input_row)[0])
            prediction = max(ranges["Addicted_Score"][0], min(ranges["Addicted_Score"][1], prediction))

            st.markdown(
                render_gauge(prediction, ranges["Addicted_Score"][0], ranges["Addicted_Score"][1]),
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="caption-muted">This is an estimate from a model '
                'trained on survey data — not a clinical assessment.</div>',
                unsafe_allow_html=True,
            )
