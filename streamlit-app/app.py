"""AI for Safer Roads — Speed Limit Misalignment Dashboard (Streamlit, local).

Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

import core
from core import (
    BAND_COLORS, BAND_RGB, BANDS, BASELINE_CONFIG, COUNTRIES, EXTRA_SIGNALS,
    HYPERPARAM_SPECS, MODEL_LABELS, NOTEBOOK_BASELINE, PRESETS,
    RECOMMENDATIONS, REFERENCE_SPEEDS,
)

st.set_page_config(page_title="AI for Safer Roads — Speed Limit Dashboard", layout="wide", page_icon="🛣️")


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading scored road segments…")
def get_data():
    df, geometries, meta = core.load_segments()
    return df, geometries, meta


@st.cache_data
def get_model_defaults():
    return core.load_model_defaults()


@st.cache_data
def get_paths(_geometries_key: int):
    """Flatten segment geometries into one path row per polyline for pydeck."""
    _, geometries, _ = core.load_segments()
    rows = []
    for i, lines in enumerate(geometries):
        for line in lines:
            rows.append({"i": i, "path": line})
    return pd.DataFrame(rows)


df, geometries, meta = get_data()
model_defaults = get_model_defaults()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "scoring" not in st.session_state:
    st.session_state.scoring = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASELINE_CONFIG.items()}
if "runs" not in st.session_state:
    st.session_state.runs = []
if "applied_run" not in st.session_state:
    st.session_state.applied_run = None
if "run_counter" not in st.session_state:
    st.session_state.run_counter = 0


def add_run(kind: str, overrides: dict) -> int:
    progress_bar = st.progress(0.0, text="Starting…")
    result = core.train_per_country(
        df, kind, overrides, model_defaults,
        progress=lambda frac, text: progress_bar.progress(frac, text=text),
    )
    progress_bar.empty()
    st.session_state.run_counter += 1
    result["label"] = f"{MODEL_LABELS[kind]} #{st.session_state.run_counter}"
    st.session_state.runs.append(result)
    return len(st.session_state.runs) - 1


# First-load experience: train the default Random Forest automatically.
if not st.session_state.runs:
    with st.spinner("Training the default Random Forest peer model (first load only)…"):
        idx = add_run("rf", {})
        st.session_state.applied_run = idx

applied = st.session_state.runs[st.session_state.applied_run] if st.session_state.applied_run is not None else None
peer_pred = applied["predictions"] if applied is not None else None
scores = core.compute_scores(df, peer_pred, st.session_state.scoring)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")
f_countries = st.sidebar.multiselect("Country", COUNTRIES, default=COUNTRIES)
f_classes = st.sidebar.multiselect("Road class", ["motorway", "trunk", "primary", "secondary"],
                                   default=["motorway", "trunk", "primary", "secondary"])
f_landuse = st.sidebar.multiselect("Land use", ["URBAN", "RURAL"], default=["URBAN", "RURAL"])
f_bands = st.sidebar.multiselect("Priority band", BANDS, default=BANDS)
f_score = st.sidebar.slider("Priority score range", 0, 100, (0, 100))
f_search = st.sidebar.text_input("Search road name / ID")

mask = (
    df["country"].isin(f_countries)
    & df["road_class"].isin(f_classes)
    & df["land_use"].isin(f_landuse)
    & scores["band"].isin(f_bands)
    & scores["score"].between(f_score[0], f_score[1])
)
if f_search.strip():
    q = f_search.strip().lower()
    mask &= (df["name"].fillna("").str.lower().str.contains(q, regex=False)
             | df["id"].str.lower().str.contains(q, regex=False))

fdf = df[mask]
fscores = scores[mask]

st.sidebar.markdown("---")
if applied is not None:
    maes = " · ".join(f"{c['country'][0]}: {c['test_mae']:.2f}" for c in applied["countries"])
    st.sidebar.caption(f"**Peer model:** {applied['label']}\n\nTest MAE — {maes}")

# ---------------------------------------------------------------------------
# Header + tabs
# ---------------------------------------------------------------------------
st.title("🛣️ AI for Safer Roads — Speed Limit Misalignment Dashboard")
st.caption("ADB Innovation Challenge 2026 · Maharashtra + Thailand · running fully local")

# Persistent navigation (st.tabs loses the selected tab on widget reruns).
NAV_PAGES = ["📊 Overview", "🗺️ Map Explorer", "🧠 Modeling Lab", "⚖️ Scoring Studio", "🔮 What-if", "📋 Segments"]
nav = st.radio("View", NAV_PAGES, horizontal=True, label_visibility="collapsed", key="nav")
st.markdown("---")

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if nav == "📊 Overview":
    total_km = fdf["length_m"].fillna(0).sum() / 1000
    crit_high = fscores["band"].isin(["Critical review", "High priority"])
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Road segments (filtered)", f"{len(fdf):,}", f"of {len(df):,} scored", delta_color="off")
    k2.metric("Network length", f"{total_km:,.0f} km")
    k3.metric("Critical + High segments", f"{int(crit_high.sum()):,}",
              f"{fdf.loc[crit_high, 'length_m'].fillna(0).sum() / 1000:,.0f} km", delta_color="off")
    for col, country in zip((k4, k5), COUNTRIES):
        sub = fscores[fdf["country"] == country]
        col.metric(f"Median score · {country}", f"{sub['score'].median():.1f}" if len(sub) else "—",
                   f"{(fdf['country'] == country).sum():,} segments", delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        band_df = (
            pd.DataFrame({"country": fdf["country"], "band": fscores["band"]})
            .value_counts().reset_index(name="segments")
        )
        fig = px.bar(band_df, x="country", y="segments", color="band",
                     color_discrete_map=BAND_COLORS, category_orders={"band": BANDS},
                     title="Priority bands by country")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(fscores, x="score", nbins=40, title="Priority score distribution")
        fig.update_traces(marker_color="#1f5eff")
        st.plotly_chart(fig, use_container_width=True)

    sample = fdf.join(fscores["score"]).sample(min(4000, len(fdf)), random_state=42) if len(fdf) else fdf
    fig = px.scatter(sample, x="speed_limit", y="f85", color="country",
                     opacity=0.4, title="Posted speed limit vs 85th percentile speed",
                     labels={"speed_limit": "Posted limit (km/h)", "f85": "85th percentile speed (km/h)"})
    fig.add_shape(type="line", x0=0, y0=0, x1=130, y1=130, line=dict(dash="dash", color="#17233b"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strongest patterns (land use × road class)")
    pattern = (
        pd.DataFrame({
            "country": fdf["country"], "land_use": fdf["land_use"], "road_class": fdf["road_class"],
            "score": fscores["score"], "km": fdf["length_m"].fillna(0) / 1000,
        })
        .groupby(["country", "land_use", "road_class"])
        .agg(segments=("score", "size"), length_km=("km", "sum"), avg_priority=("score", "mean"))
        .round({"length_km": 0, "avg_priority": 1})
        .sort_values("avg_priority", ascending=False)
        .head(12)
        .reset_index()
    )
    st.dataframe(pattern, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Map Explorer
# ---------------------------------------------------------------------------
if nav == "🗺️ Map Explorer":
    col_a, col_b = st.columns([1, 3])
    top300 = col_a.toggle("Top 300 priority segments only")
    col_b.caption("Colors follow the priority band. Hover a road for details.")

    map_idx = fscores.index
    if top300:
        map_idx = fscores["score"].nlargest(300).index

    paths = get_paths(0)
    sel = paths[paths["i"].isin(set(map_idx))].copy()
    meta_cols = pd.DataFrame({
        "name": df["name"].fillna(df["id"]),
        "band": scores["band"],
        "score": scores["score"],
        "posted": df["speed_limit"],
        "ref": df["safe_ref"],
    })
    sel = sel.merge(meta_cols, left_on="i", right_index=True)
    sel["color"] = sel["band"].map(BAND_RGB)

    layer = pdk.Layer(
        "PathLayer", data=sel, get_path="path", get_color="color",
        width_min_pixels=2, pickable=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=17, longitude=88, zoom=4),
        map_provider="carto", map_style="light",
        tooltip={"html": "<b>{name}</b><br/>{band} — score {score}<br/>Posted {posted} km/h · Safe ref {ref} km/h"},
    )
    st.pydeck_chart(deck, height=620)

    counts = fscores.loc[map_idx, "band"].value_counts()
    st.caption("   ".join(f"● {b}: {counts.get(b, 0):,}" for b in BANDS))

# ---------------------------------------------------------------------------
# Modeling Lab
# ---------------------------------------------------------------------------
if nav == "🧠 Modeling Lab":
    st.subheader("1 · Choose a peer model")
    st.caption("The peer model estimates the speed limit typical for comparable segments; "
               "its gap vs the posted limit feeds the PeerGap score component.")
    kind = st.radio("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get, horizontal=True,
                    label_visibility="collapsed")

    st.subheader("2 · Hyperparameters")
    st.caption("Leave a field **empty** to use the tuned default (shown per country in the placeholder). "
               "Values you enter apply to both countries.")
    overrides: dict = {}
    spec = HYPERPARAM_SPECS[kind]
    cols = st.columns(min(3, len(spec)))
    for pos, (key, label, ptype, help_text) in enumerate(spec):
        tuned = {c[0]: core.tuned_defaults(kind, c, model_defaults).get(key) for c in COUNTRIES}
        placeholder = " · ".join(f"{c}: {v}" for c, v in tuned.items())
        with cols[pos % len(cols)]:
            if isinstance(ptype, list):
                choice = st.selectbox(label, [f"Tuned default ({placeholder})"] + ptype, help=help_text,
                                      key=f"hp_{kind}_{key}")
                if not choice.startswith("Tuned default"):
                    overrides[key] = choice
            else:
                raw = st.text_input(label, placeholder=f"tuned: {placeholder}", help=help_text,
                                    key=f"hp_{kind}_{key}")
                if raw.strip():
                    try:
                        overrides[key] = int(raw) if ptype == "int" else float(raw)
                    except ValueError:
                        st.error(f"'{raw}' is not a valid number for {label}")

    if st.button(f"🚂 Train {MODEL_LABELS[kind]}", type="primary"):
        idx = add_run(kind, overrides)
        if st.session_state.applied_run is None:
            st.session_state.applied_run = idx
        st.rerun()

    if st.session_state.runs:
        st.subheader("3 · Results")
        labels = [r["label"] for r in st.session_state.runs]
        shown_i = st.selectbox("Inspect run", range(len(labels)),
                               index=len(labels) - 1, format_func=lambda i: labels[i])
        run = st.session_state.runs[shown_i]

        metrics_df = pd.DataFrame([{
            "Country": c["country"],
            "Rows (train/test)": f"{c['n_train']:,} / {c['n_test']:,}",
            "MAE train": round(c["train_mae"], 2), "MAE test": round(c["test_mae"], 2),
            "RMSE train": round(c["train_rmse"], 2), "RMSE test": round(c["test_rmse"], 2),
            "R² train": round(c["train_r2"], 2), "R² test": round(c["test_r2"], 2),
            "Notebook RF test MAE": NOTEBOOK_BASELINE.get(c["country"]),
        } for c in run["countries"]])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        is_applied = st.session_state.applied_run == shown_i
        if st.button("✓ Applied to scoring" if is_applied else "Apply to scoring",
                     disabled=is_applied, type="secondary"):
            st.session_state.applied_run = shown_i
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            scat = pd.concat([c["scatter"].sample(min(800, len(c["scatter"])), random_state=42)
                              for c in run["countries"]])
            fig = px.scatter(scat, x="actual", y="predicted", color="country", opacity=0.4,
                             title="Predicted vs actual posted limit (test split)")
            fig.add_shape(type="line", x0=0, y0=0, x1=130, y1=130, line=dict(dash="dash", color="#17233b"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            resid = pd.concat([pd.DataFrame({"residual": c["residuals"], "country": c["country"]})
                               for c in run["countries"]])
            fig = px.histogram(resid, x="residual", color="country", nbins=40, barmode="overlay",
                               title="Test residuals (actual − predicted, km/h)")
            st.plotly_chart(fig, use_container_width=True)

        importances = [c["importances"] for c in run["countries"] if c["importances"] is not None]
        if importances:
            avg_imp = (pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=True).tail(10) * 100)
            fig = go.Figure(go.Bar(x=avg_imp.values, y=avg_imp.index, orientation="h", marker_color="#1f5eff"))
            fig.update_layout(title="Feature importance (%, averaged across countries)", height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("KNN has no feature importances.")

        st.subheader("Model comparison — all runs this session")
        comp = pd.DataFrame([{
            "Run": r["label"] + (" ✓" if st.session_state.applied_run == i else ""),
            "Model": MODEL_LABELS[r["kind"]],
            "Overrides": ", ".join(f"{k}={v}" for k, v in r["overrides"].items()) or "tuned defaults",
            **{f"Test MAE ({c['country'][0]})": round(c["test_mae"], 2) for c in r["countries"]},
            **{f"Test R² ({c['country'][0]})": round(c["test_r2"], 2) for c in r["countries"]},
        } for i, r in enumerate(st.session_state.runs)])
        st.dataframe(comp, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Scoring Studio
# ---------------------------------------------------------------------------
if nav == "⚖️ Scoring Studio":
    cfg = st.session_state.scoring
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Component weights")
        st.caption("Weights are rescaled so the effective total is 100 points. Notebook baseline: 45/20/15/15/5.")
        weight_labels = {
            "posted": "Posted-limit excess", "operating": "Operating-speed excess",
            "vru": "VRU exposure", "traffic": "Traffic exposure", "peer": "Peer-model gap",
        }
        new_weights = {}
        for key, label in weight_labels.items():
            new_weights[key] = st.slider(label, 0, 60, cfg["weights"][key], key=f"w_{key}")

        st.subheader("Extra scoring component")
        signal = st.selectbox("Signal", list(EXTRA_SIGNALS), format_func=EXTRA_SIGNALS.get,
                              index=list(EXTRA_SIGNALS).index(cfg["extra"]["signal"]))
        extra_weight = st.slider("Extra weight", 0, 30, cfg["extra"]["weight"]) if signal != "none" else 0

        preset_cols = st.columns(len(PRESETS) + 1)
        for col, (name, weights) in zip(preset_cols, PRESETS.items()):
            if col.button(name):
                for k, v in weights.items():
                    st.session_state[f"w_{k}"] = v
                st.rerun()
        if preset_cols[-1].button("Reset all"):
            for k in ("w_posted", "w_operating", "w_vru", "w_traffic", "w_peer"):
                st.session_state.pop(k, None)
            st.session_state.scoring = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASELINE_CONFIG.items()}
            st.rerun()

    with c2:
        st.subheader("Normalization caps (km/h)")
        cap_cols = st.columns(3)
        caps = {
            "posted": cap_cols[0].number_input("Posted excess cap", 5, 60, cfg["caps"]["posted"]),
            "operating": cap_cols[1].number_input("Operating excess cap", 5, 60, cfg["caps"]["operating"]),
            "peer": cap_cols[2].number_input("Peer gap cap", 5, 60, cfg["caps"]["peer"]),
        }
        st.subheader("Priority band thresholds")
        thr_cols = st.columns(3)
        thresholds = {
            "critical": thr_cols[0].number_input("Critical review ≥", 0, 100, cfg["thresholds"]["critical"]),
            "high": thr_cols[1].number_input("High priority ≥", 0, 100, cfg["thresholds"]["high"]),
            "moderate": thr_cols[2].number_input("Moderate priority ≥", 0, 100, cfg["thresholds"]["moderate"]),
        }

        st.session_state.scoring = {
            "weights": new_weights, "caps": caps, "thresholds": thresholds,
            "extra": {"signal": signal, "weight": extra_weight},
        }

        st.subheader("Sensitivity vs baseline")
        sens = core.sensitivity_vs_baseline(df, peer_pred, st.session_state.scoring)
        s1, s2 = st.columns(2)
        s1.metric("Spearman rank correlation", f"{sens['spearman']:.3f}")
        s2.metric("Top-300 overlap", f"{sens['overlap']} / 300", f"{sens['overlap_pct']}%", delta_color="off")
        st.caption("High correlation and overlap mean your settings still flag the same roads as the baseline — "
                   "the ranking is robust.")

# ---------------------------------------------------------------------------
# What-if
# ---------------------------------------------------------------------------
if nav == "🔮 What-if":
    st.subheader("Describe a hypothetical road segment")
    model_name = MODEL_LABELS[applied["kind"]] if applied else "—"
    st.caption(f"The applied peer model (**{model_name}**) predicts the typical speed limit; "
               "the scoring engine produces the full priority breakdown.")

    with st.form("whatif"):
        c1, c2, c3 = st.columns(3)
        wi_country = c1.selectbox("Country model", COUNTRIES)
        wi_class = c2.selectbox("Road class", ["secondary", "primary", "trunk", "motorway"])
        wi_landuse = c3.selectbox("Land use", ["URBAN", "RURAL"])
        c4, c5, c6 = st.columns(3)
        wi_limit = c4.number_input("Posted speed limit (km/h)", 10, 140, 60)
        wi_median = c5.number_input("Median speed (km/h)", 0, 160, 42)
        wi_f85 = c6.number_input("85th percentile speed (km/h)", 0, 180, 55)
        c7, c8, c9, c10 = st.columns(4)
        wi_ws = c7.number_input("Weighted sample", 0, 10_000_000, 5000)
        wi_ss = c8.number_input("Sample size", 0, 10_000_000, 8000)
        wi_rp = c9.number_input("Traffic exposure percentile", 0, 100, 75)
        wi_len = c10.number_input("Segment length (km)", 0.1, 500.0, 4.0)
        submitted = st.form_submit_button("🔮 Predict & score this segment", type="primary")

    if submitted and applied is not None:
        pipe = applied["fitted"][wi_country]
        X = pd.DataFrame([{
            "road_class": wi_class, "land_use": wi_landuse, "median_speed": wi_median,
            "f85": wi_f85, "weighted_sample": wi_ws, "sample_size": wi_ss,
            "ranked_pct": wi_rp, "length_m": wi_len * 1000,
        }])
        predicted = float(pipe.predict(X)[0])
        ref = REFERENCE_SPEEDS[(wi_landuse, wi_class)]

        max_log_ss = np.log1p(df["sample_size"].fillna(0)).max() or 1
        row = pd.DataFrame([{
            "speed_limit": wi_limit,
            "posted_excess": max(0, wi_limit - ref),
            "operating_excess": max(0, min(wi_f85, wi_limit) - ref),
            "f85_minus_limit": wi_f85 - wi_limit,
            "vru": min(1.0, (0.55 if wi_landuse == "URBAN" else 0.2)
                       + {"secondary": 0.25, "primary": 0.18, "trunk": 0.08, "motorway": 0.0}[wi_class]),
            "traffic": min(1.0, max(0.0, wi_rp / 100)),
            "sample_conf": min(1.0, np.log1p(wi_ss) / max_log_ss),
            "length_m": wi_len * 1000,
        }])
        result = core.compute_scores(row, np.array([predicted]), st.session_state.scoring).iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Peer-predicted typical limit", f"{predicted:.0f} km/h",
                  f"posted {wi_limit} → gap {max(0, wi_limit - predicted):.0f} km/h", delta_color="off")
        m2.metric("Safe System reference", f"{ref} km/h", f"{wi_landuse.lower()} · {wi_class}", delta_color="off")
        m3.metric("Priority score", f"{result['score']:.1f}", f"safety score {100 - result['score']:.1f}",
                  delta_color="off")

        st.markdown(
            f"<span style='background:{BAND_COLORS[result['band']]};color:#fff;padding:4px 14px;"
            f"border-radius:99px;font-weight:600'>{result['band']}</span>  "
            f"*{RECOMMENDATIONS[result['band']]}*",
            unsafe_allow_html=True,
        )

        contrib = pd.DataFrame({
            "Component": ["Posted-limit excess", "Operating-speed excess", "VRU exposure",
                          "Traffic exposure", "Peer-model gap", "Extra component"],
            "Points": [result["c_posted"], result["c_operating"], result["c_vru"],
                       result["c_traffic"], result["c_peer"], result["c_extra"]],
        })
        contrib = contrib[contrib["Points"] > 0.001]
        fig = px.bar(contrib, x="Points", y="Component", orientation="h", title="Score contributions",
                     color="Component", color_discrete_sequence=["#d64545", "#ef8c33", "#8e44ad",
                                                                 "#2980b9", "#16a085", "#7f8c8d"])
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Segments table
# ---------------------------------------------------------------------------
if nav == "📋 Segments":
    st.subheader(f"Scored segments — {len(fdf):,} match the current filters")
    table = pd.DataFrame({
        "Priority": fscores["score"],
        "Band": fscores["band"],
        "Road": fdf["name"].fillna(fdf["id"]),
        "Country": fdf["country"],
        "Class": fdf["road_class"],
        "Land use": fdf["land_use"].str.lower(),
        "Posted (km/h)": fdf["speed_limit"],
        "Safe ref (km/h)": fdf["safe_ref"],
        "85th pct (km/h)": fdf["f85"].round(0),
        "Peer pred (km/h)": fscores["peer_predicted"].round(0),
        "Length (km)": (fdf["length_m"].fillna(0) / 1000).round(1),
    }).sort_values("Priority", ascending=False)

    st.dataframe(table, use_container_width=True, hide_index=True, height=560)

    export = table.assign(SegmentID=fdf["id"], SpeedSafetyScore=(100 - fscores["score"]).round(1),
                          Recommendation=fscores["band"].map(RECOMMENDATIONS))
    st.download_button(
        f"⬇️ Export CSV ({len(export):,} rows)",
        export.to_csv(index=False).encode("utf-8"),
        file_name="safer_roads_scored_segments.csv",
        mime="text/csv",
    )
