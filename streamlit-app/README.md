# AI for Safer Roads — Streamlit Dashboard (local)

A local Streamlit version of the ADB "AI for Safer Roads" 2026 speed-limit misalignment dashboard.
Same six views as the web app, but the peer models train with **real scikit-learn** (exact notebook
parity — RF test MAE ≈ 4.90 / 5.64 vs the notebook's 4.89 / 5.63).

## Run it

Double-click **`run.bat`**, or from a terminal:

```bash
cd "streamlit-app"
python -m streamlit run app.py
```

It opens at http://localhost:8501. First load auto-trains the default Random Forest peer model
(~10–15 s), then the dashboard is interactive.

## First-time setup (already done on this machine)

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. The prepared data (`data/segments.json`, `data/model-defaults.json`) is copied
in from the `dashboard/` build — no GeoJSON parsing or geopandas needed.

## Views

- **Overview** — KPIs, priority bands, score histogram, speed-vs-85th scatter, hotspot patterns.
- **Map Explorer** — pydeck map of all segments colored by band; top-300 toggle; hover for details.
- **Modeling Lab** — pick the peer model (Random Forest / Gradient Boosting / KNN / Ridge / Decision
  Tree). Every hyperparameter is editable; **leave a field blank to use the tuned default** (shown per
  country in the placeholder). Metrics (MAE/RMSE/R² train+test), predicted-vs-actual, residuals,
  feature importance, and a session comparison table. "Apply to scoring" pushes the peer-gap into the
  dashboard-wide priority score.
- **Scoring Studio** — weight sliders, caps, band thresholds, an extra user-defined component, presets,
  and a live sensitivity check (Spearman + top-300 overlap vs baseline).
- **What-if** — describe a hypothetical segment; the applied model predicts its typical limit and the
  engine shows the full priority breakdown.
- **Segments** — sortable table of all scored segments with CSV export.

## Difference from the web (Vercel/Hostinger) version

| | Web dashboard | This Streamlit app |
|---|---|---|
| ML engine | custom TypeScript (browser) | scikit-learn (Python) |
| Notebook parity | close (~0.3 MAE) | exact |
| Runs | static site, any host | local Python only |
