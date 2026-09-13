# 🌙 Student Social Media Addiction Predictor

A small end-to-end ML project: a notebook explores the **Student Social
Media Addiction Analysis** dataset (Kaggle) and compares regressors, a
training script turns that work into a deployable pipeline, and a
Streamlit app serves predictions through a simple form.

## Contents

- [How it works](#how-it-works)
- [What changed from the original notebook](#what-changed-from-the-original-notebook)
- [Setup](#setup)
- [Files](#files)
- [Notes / assumptions](#notes--assumptions)
- [Troubleshooting](#troubleshooting)

## How it works

```
dataset.csv ──▶ train_model.py ──▶ rf_pipeline.pkl + metadata.json ──▶ app.py
               (clean, engineer,        (encoders + model            (Streamlit
                encode, fit RF)          bundled together)             UI)
```

1. **`train_model.py`** loads `dataset.csv`, applies the same cleaning and
   feature engineering used in the notebook, fits a `RandomForestRegressor`
   inside a single `sklearn.Pipeline`, and saves three artifacts:
   - `rf_pipeline.pkl` — preprocessing (ordinal + one-hot encoding) and the
     trained model, bundled together.
   - `metadata.json` — dropdown options, slider ranges, and test-set metrics,
     generated from your actual data.
   - `eda_summary.json` — aggregated EDA stats (correlations, group averages,
     binned trends) for the app's Overview tab. It's aggregated on purpose —
     correlations and averages, never individual student rows — so it's safe
     to deploy even when `dataset.csv` itself isn't.
2. **`app.py`** loads those artifacts and serves a two-tab Streamlit UI: a
   project overview built from real EDA charts (usage vs. addiction score,
   addiction rate by age, platform breakdown, a correlation heatmap) with the
   model comparison tucked into an expander, and a prediction form that turns
   raw answers into a score from 1–9.

## What changed from the original notebook

The notebook trained several models and saved only the fitted
`RandomForestRegressor` (`rf_default`) with `joblib.dump(...)`. That's not
enough to deploy: the model expects an already-encoded row (ordinal-encoded
`Academic_Level`, one-hot-encoded `Gender` / `Affects_Academic_Performance` /
`Relationship_Status` / `Most_Used_Platform`, plus the engineered
`Usage_Sleep_Ratio`), and the encoders themselves were never saved.

`train_model.py` fixes that by wrapping the *same* cleaning, feature
engineering, and encoding steps from the notebook, plus the Random Forest,
into one `sklearn.Pipeline`. Saving that single object means the Streamlit
app can just hand it a raw row (`Age=20, Gender="Female", ...`) and get a
prediction back — no manual encoding logic duplicated in the app.

## Setup

1. Put `dataset.csv` (the Kaggle "Student Social Media Addiction" dataset) in
   this folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train and export the pipeline:
   ```bash
   python train_model.py
   ```
   This creates `rf_pipeline.pkl`, `metadata.json`, and `eda_summary.json`,
   and prints the held-out test metrics (MAE / MSE / RMSE / R²).
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. Open the URL Streamlit prints (usually `http://localhost:8501`), pick
   **Project Overview** to see the dataset and model comparison, or
   **Predict My Score** to fill in the form and get a prediction.

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Rebuilds the pipeline (cleaning + encoding + Random Forest) from `dataset.csv` and saves `rf_pipeline.pkl` + `metadata.json` + `eda_summary.json`. |
| `app.py` | Two-tab Streamlit app: an EDA-driven project overview and a prediction form with a visual score gauge. |
| `requirements.txt` | Dependencies. |
| `eda_models.ipynb` | Original exploration notebook — EDA, feature engineering, and the model comparison the app's Overview tab summarizes. |
| `rf_pipeline.pkl` *(generated)* | Fitted pipeline (encoders + model), produced by `train_model.py`. Not committed — regenerate it locally. |
| `metadata.json` *(generated)* | Dropdown options, slider ranges, and test metrics, produced by `train_model.py`. Not committed — regenerate it locally. |
| `eda_summary.json` *(generated)* | Aggregated correlations and group trends, produced by `train_model.py`. Safe to commit and deploy — contains no individual student rows. |

## Notes / assumptions

- Dropdown options (platforms, gender values, etc.) and slider ranges are
  read from `metadata.json`, which is generated from your actual
  `dataset.csv` — so they'll always match what the model was trained on,
  even if your data differs slightly from the original Kaggle version.
- The prediction form asks for daily usage hours and sleep hours separately
  (more intuitive than asking for a ratio) and computes `Usage_Sleep_Ratio`
  internally, exactly as the notebook did.
- The model comparison table on the Overview tab is the static result from
  the notebook's evaluation; it isn't recomputed by the app.
- This is an educational project, not a clinical or diagnostic tool — the
  app says so explicitly next to every prediction.
[README.md](https://github.com/user-attachments/files/32158424/README.md)
# Social-media-addiction
