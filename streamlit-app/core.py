"""Data loading, scoring engine, and model training for the Safer Roads Streamlit app.

Mirrors the notebook pipeline (260625_Feature Engineers.ipynb) and the web
dashboard's scoring engine. Reuses the prepared JSONs from dashboard/public/data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
COUNTRIES = ["Maharashtra", "Thailand"]

PEER_FEATURES = [
    "road_class", "land_use", "median_speed", "f85", "weighted_sample",
    "sample_size", "ranked_pct", "length_m",
]
CATEGORICAL_FEATURES = ["road_class", "land_use"]
NUMERIC_FEATURES = [f for f in PEER_FEATURES if f not in CATEGORICAL_FEATURES]

BANDS = ["Critical review", "High priority", "Moderate priority", "Monitor"]
BAND_COLORS = {
    "Critical review": "#d64545",
    "High priority": "#ef8c33",
    "Moderate priority": "#e7c53a",
    "Monitor": "#3f9d58",
}
BAND_RGB = {
    "Critical review": [214, 69, 69],
    "High priority": [239, 140, 51],
    "Moderate priority": [231, 197, 58],
    "Monitor": [63, 157, 88],
}
RECOMMENDATIONS = {
    "Critical review": "Immediate engineering review and field validation",
    "High priority": "Schedule for speed-limit and corridor safety review",
    "Moderate priority": "Monitor and validate when new contextual data is available",
    "Monitor": "Monitor as part of normal network management",
}

REFERENCE_SPEEDS = {
    ("URBAN", "secondary"): 30, ("URBAN", "primary"): 50,
    ("URBAN", "trunk"): 50, ("URBAN", "motorway"): 80,
    ("RURAL", "secondary"): 60, ("RURAL", "primary"): 70,
    ("RURAL", "trunk"): 80, ("RURAL", "motorway"): 100,
}

MODEL_LABELS = {
    "rf": "Random Forest",
    "gbm": "Gradient Boosting",
    "knn": "K-Nearest Neighbors",
    "ridge": "Ridge Regression",
    "dt": "Decision Tree",
}

# Notebook sklearn reference (Section 8 output).
NOTEBOOK_BASELINE = {"Thailand": 5.63, "Maharashtra": 4.89}

BASELINE_CONFIG = {
    "weights": {"posted": 45, "operating": 20, "vru": 15, "traffic": 15, "peer": 5},
    "caps": {"posted": 30, "operating": 30, "peer": 20},
    "thresholds": {"critical": 70, "high": 50, "moderate": 30},
    "extra": {"signal": "none", "weight": 0},
}

PRESETS = {
    "Baseline": {"posted": 45, "operating": 20, "vru": 15, "traffic": 15, "peer": 5},
    "Safety-heavy": {"posted": 50, "operating": 20, "vru": 20, "traffic": 5, "peer": 5},
    "Exposure-heavy": {"posted": 40, "operating": 20, "vru": 15, "traffic": 20, "peer": 5},
}

EXTRA_SIGNALS = {
    "none": "None",
    "sample_conf": "Sample confidence (probe data volume)",
    "f85_minus_limit": "F85 minus posted limit (speeding pressure, capped at 20 km/h)",
    "len_norm": "Segment length (log-normalized)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def data_dir() -> Path:
    here = Path(__file__).parent
    for candidate in [here / "data", here.parent / "dashboard" / "public" / "data"]:
        if (candidate / "segments.json").exists():
            return candidate
    raise FileNotFoundError(
        "segments.json not found. Run `npm run prepare-data` in the dashboard folder "
        "or place segments.json + model-defaults.json in streamlit-app/data/."
    )


def load_segments() -> tuple[pd.DataFrame, list, dict]:
    payload = json.loads((data_dir() / "segments.json").read_text(encoding="utf-8"))
    rows = payload["segments"]
    df = pd.DataFrame({
        "id": [s["id"] for s in rows],
        "country": ["Maharashtra" if s["c"] == "M" else "Thailand" for s in rows],
        "name": [s["n"] for s in rows],
        "road_class": [s["rc"] for s in rows],
        "land_use": [s["lu"] for s in rows],
        "speed_limit": [s["sl"] for s in rows],
        "median_speed": [s["ms"] for s in rows],
        "f85": [s["f85"] for s in rows],
        "weighted_sample": [s["ws"] for s in rows],
        "sample_size": [s["ss"] for s in rows],
        "ranked_pct": [s["rp"] for s in rows],
        "length_m": [s["len"] for s in rows],
        "safe_ref": [s["ref"] for s in rows],
        "posted_excess": [s["pe"] for s in rows],
        "operating_excess": [s["oe"] for s in rows],
        "f85_minus_limit": [s["fml"] for s in rows],
        "vru": [s["vru"] for s in rows],
        "traffic": [s["te"] for s in rows],
        "sample_conf": [s["sc"] for s in rows],
    })
    geometries = [s["g"] for s in rows]
    return df, geometries, payload["meta"]


def load_model_defaults() -> dict:
    return json.loads((data_dir() / "model-defaults.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------
def compute_scores(df: pd.DataFrame, peer_pred: np.ndarray | None, cfg: dict) -> pd.DataFrame:
    w, caps, thr, extra = cfg["weights"], cfg["caps"], cfg["thresholds"], cfg["extra"]
    extra_w = 0 if extra["signal"] == "none" else extra["weight"]
    total = sum(w.values()) + extra_w
    scale = 100.0 / total if total else 0.0

    posted = np.clip(df["posted_excess"] / caps["posted"], 0, 1)
    operating = np.clip(df["operating_excess"] / caps["operating"], 0, 1)
    if peer_pred is not None:
        peer_gap = np.clip(df["speed_limit"].to_numpy() - peer_pred, 0, None)
    else:
        peer_gap = np.zeros(len(df))
    peer = np.clip(peer_gap / caps["peer"], 0, 1)

    if extra["signal"] == "sample_conf":
        extra_vals = df["sample_conf"].to_numpy()
    elif extra["signal"] == "f85_minus_limit":
        extra_vals = np.clip(df["f85_minus_limit"] / 20, 0, 1)
    elif extra["signal"] == "len_norm":
        log_len = np.log1p(df["length_m"].fillna(0))
        extra_vals = np.clip(log_len / max(log_len.max(), 1e-9), 0, 1)
    else:
        extra_vals = np.zeros(len(df))

    out = pd.DataFrame(index=df.index)
    out["c_posted"] = scale * w["posted"] * posted
    out["c_operating"] = scale * w["operating"] * operating
    out["c_vru"] = scale * w["vru"] * df["vru"]
    out["c_traffic"] = scale * w["traffic"] * df["traffic"]
    out["c_peer"] = scale * w["peer"] * peer
    out["c_extra"] = scale * extra_w * extra_vals
    out["peer_predicted"] = peer_pred if peer_pred is not None else np.nan
    out["peer_gap"] = peer_gap
    out["score"] = out[["c_posted", "c_operating", "c_vru", "c_traffic", "c_peer", "c_extra"]].sum(axis=1).round(1)
    out["band"] = np.select(
        [out["score"] >= thr["critical"], out["score"] >= thr["high"], out["score"] >= thr["moderate"]],
        BANDS[:3],
        default="Monitor",
    )
    return out


def sensitivity_vs_baseline(df: pd.DataFrame, peer_pred: np.ndarray | None, cfg: dict) -> dict:
    current = compute_scores(df, peer_pred, cfg)["score"]
    baseline = compute_scores(df, peer_pred, BASELINE_CONFIG)["score"]
    rho = current.corr(baseline, method="spearman")
    top_cur = set(current.nlargest(300).index)
    top_base = set(baseline.nlargest(300).index)
    overlap = len(top_cur & top_base)
    return {"spearman": round(float(rho), 3), "overlap": overlap, "overlap_pct": round(overlap / 3, 1)}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
HYPERPARAM_SPECS = {
    "rf": [
        ("n_estimators", "Number of trees", "int", "More trees = more stable, slower."),
        ("max_depth", "Max depth", "int", "Maximum depth of each tree."),
        ("min_samples_leaf", "Min samples per leaf", "int", "Higher = smoother, less overfitting."),
        ("min_samples_split", "Min samples to split", "int", "Minimum node size eligible for splitting."),
        ("max_features", "Max features per split", ["sqrt", "log2", "0.5", "0.75", "1.0"], "Feature subsampling at each split."),
        ("max_samples", "Bootstrap sample fraction", ["0.7", "0.85", "1.0"], "Fraction of rows sampled per tree."),
    ],
    "gbm": [
        ("n_estimators", "Boosting rounds", "int", "Number of sequential trees."),
        ("learning_rate", "Learning rate", "float", "Contribution of each tree."),
        ("max_depth", "Max depth", "int", "Depth of each boosted tree (keep shallow)."),
        ("min_samples_leaf", "Min samples per leaf", "int", "Higher = smoother trees."),
        ("subsample", "Subsample fraction", "float", "Row sampling per round."),
    ],
    "knn": [
        ("n_neighbors", "Neighbors (k)", "int", "How many similar segments to average."),
        ("weights", "Neighbor weighting", ["uniform", "distance"], "Uniform average or inverse-distance weighted."),
    ],
    "ridge": [
        ("alpha", "Regularization (alpha)", "float", "Higher = stronger coefficient shrinkage."),
    ],
    "dt": [
        ("max_depth", "Max depth", "int", "Depth of the tree."),
        ("min_samples_leaf", "Min samples per leaf", "int", "Higher = simpler tree."),
        ("min_samples_split", "Min samples to split", "int", "Minimum node size eligible for splitting."),
    ],
}

# JS camelCase keys in model-defaults.json -> sklearn kwargs
_JS_TO_PY = {
    "nEstimators": "n_estimators", "maxDepth": "max_depth", "minSamplesLeaf": "min_samples_leaf",
    "minSamplesSplit": "min_samples_split", "maxFeatures": "max_features", "maxSamples": "max_samples",
    "learningRate": "learning_rate", "subsample": "subsample", "k": "n_neighbors",
    "weights": "weights", "alpha": "alpha",
}

BUILTIN_DEFAULTS = {
    "rf": {"n_estimators": 250, "max_depth": 16, "min_samples_leaf": 25, "min_samples_split": 2,
           "max_features": "sqrt", "max_samples": 1.0},
    "gbm": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20, "subsample": 0.8},
    "knn": {"n_neighbors": 15, "weights": "distance"},
    "ridge": {"alpha": 1.0},
    "dt": {"max_depth": 8, "min_samples_leaf": 25, "min_samples_split": 10},
}


def _coerce(key: str, value):
    if key in ("max_features", "max_samples", "subsample", "learning_rate", "alpha"):
        if isinstance(value, str) and value not in ("sqrt", "log2", "uniform", "distance"):
            value = float(value)
        if key == "max_features" and isinstance(value, float) and value >= 1.0:
            return None  # sklearn: None = all features
        if key == "max_samples" and isinstance(value, float) and value >= 1.0:
            return None  # sklearn: None = full bootstrap
        return value
    if key == "weights":
        return value
    if key in ("n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "n_neighbors"):
        return int(value)
    return float(value)


def tuned_defaults(kind: str, country: str, model_defaults: dict) -> dict:
    """Per-country tuned params from model-defaults.json, mapped to sklearn names."""
    raw = model_defaults.get(kind, {}).get(country, {}).get("params", {})
    out = dict(BUILTIN_DEFAULTS[kind])
    for js_key, val in raw.items():
        out[_JS_TO_PY.get(js_key, js_key)] = val
    return out


def effective_params(kind: str, country: str, overrides: dict, model_defaults: dict) -> dict:
    """Tuned defaults overridden by user-set values; coerced for sklearn."""
    params = tuned_defaults(kind, country, model_defaults)
    params.update(overrides)
    return {k: _coerce(k, v) for k, v in params.items()}


def build_estimator(kind: str, params: dict):
    if kind == "rf":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if kind == "gbm":
        return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    if kind == "knn":
        return KNeighborsRegressor(**params)
    if kind == "ridge":
        return Ridge(**params)
    if kind == "dt":
        return DecisionTreeRegressor(random_state=RANDOM_STATE, **params)
    raise ValueError(kind)


def build_pipeline(kind: str, params: dict) -> Pipeline:
    """Mirrors the notebook's preprocessing ColumnTransformer."""
    pre = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
    ])
    steps = [("preprocess", pre)]
    if kind in ("knn", "ridge"):
        steps.append(("scale", StandardScaler()))
    steps.append(("model", build_estimator(kind, params)))
    return Pipeline(steps)


def feature_importances(kind: str, pipeline: Pipeline) -> pd.Series | None:
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    pretty = [n.split("__")[-1].replace("road_class_", "class: ").replace("land_use_", "land: ") for n in names]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_)
        vals = vals / (vals.sum() or 1)
    else:
        return None
    return pd.Series(vals, index=pretty).sort_values(ascending=False)


def train_per_country(df: pd.DataFrame, kind: str, overrides: dict, model_defaults: dict,
                      progress=None) -> dict:
    """Country-specific training mirroring notebook Section 8: 75/25 split,
    train/test metrics, then predict all segments for PeerGap scoring."""
    predictions = np.full(len(df), np.nan)
    countries_out = []
    fitted = {}

    for step, country in enumerate(COUNTRIES):
        if progress:
            progress(step / len(COUNTRIES), f"Training {country}…")
        mask = (df["country"] == country) & (df["speed_limit"] > 0)
        sub = df.loc[mask]
        X, y = sub[PEER_FEATURES], sub["speed_limit"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

        params = effective_params(kind, country, overrides, model_defaults)
        pipe = build_pipeline(kind, params)
        pipe.fit(X_train, y_train)

        pred_train, pred_test = pipe.predict(X_train), pipe.predict(X_test)
        predictions[np.where(mask)[0]] = pipe.predict(X)

        countries_out.append({
            "country": country,
            "params": params,
            "n_train": len(X_train), "n_test": len(X_test),
            "train_mae": mean_absolute_error(y_train, pred_train),
            "test_mae": mean_absolute_error(y_test, pred_test),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "train_r2": r2_score(y_train, pred_train),
            "test_r2": r2_score(y_test, pred_test),
            "scatter": pd.DataFrame({"actual": y_test.to_numpy(), "predicted": pred_test, "country": country}),
            "residuals": y_test.to_numpy() - pred_test,
            "importances": feature_importances(kind, pipe),
        })
        fitted[country] = pipe

    if progress:
        progress(1.0, "Done")
    return {"kind": kind, "overrides": overrides, "countries": countries_out,
            "predictions": predictions, "fitted": fitted}
