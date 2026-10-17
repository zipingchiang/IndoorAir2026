# Reliability-aware supervisory Indoor Air analysis
# Main manuscript output is intentionally limited to exactly 3 tables and 3 figures.
# Full reproducibility/supporting outputs are retained under raw_outputs/.

import argparse
import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    random_state: int = 42
    interval_min: int = 30
    rows_per_day_expected: int = 21
    primary_horizon_h: int = 2
    horizons_h: Tuple[int, ...] = (1, 2, 3)

    # Chronological split used in the manuscript: 255 + 54 + 56 = 365 days.
    n_train_days: int = 255
    n_val_days: int = 54
    n_test_days: int = 56

    # Empirical exposure-state thresholds, estimated from TRAINING ONLY.
    warning_q: float = 0.95
    severe_q: float = 0.975
    critical_q: float = 0.99

    # Applicability-domain settings.
    range_low_q: float = 0.01
    range_high_q: float = 0.99
    knn_k: int = 5
    local_support_q: float = 0.95

    # Pre-specified applicability-domain robustness grid. The baseline
    # manuscript definition (1st/99th percentiles + 95th-percentile local
    # support) is included together with stricter and looser alternatives.
    ad_range_pairs: Tuple[Tuple[float, float], ...] = (
        (0.005, 0.995),
        (0.010, 0.990),
        (0.025, 0.975),
    )
    ad_local_support_qs: Tuple[float, ...] = (0.90, 0.95, 0.975)

    # Routine-to-severe holdout.
    holdout_q: float = 0.75

    # Ventilation sensitivity analysis.
    ach_base: float = 0.4
    ach_scenarios: Tuple[float, ...] = (0.4, 0.6, 0.8)

    # Reliability-aware selective prediction. Coverage levels are fixed a priori
    # and are not tuned on the chronological test set.
    selective_coverages: Tuple[float, ...] = (1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50)


CFG = Config()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(err)),
    }


def trapezoid_rolling_by_day(
    df: pd.DataFrame,
    value_col: str,
    horizon_h: int,
    direction: str,
    interval_h: float,
) -> pd.Series:
    """Rolling trapezoidal integral that never crosses day boundaries.

    direction='hist'   => [t-H, t]
    direction='future' => [t, t+H]
    """
    n_intervals = int(round(horizon_h / interval_h))
    if not np.isclose(n_intervals * interval_h, horizon_h):
        raise ValueError("Horizon is not an integer multiple of the sampling interval")
    n_points = n_intervals + 1

    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby("date", sort=True).groups.items():
        idx = list(idx)
        values = df.loc[idx, value_col].to_numpy(dtype=float)
        n = len(values)
        if n < n_points:
            continue

        if direction == "hist":
            for end in range(n_points - 1, n):
                start = end - n_points + 1
                v = values[start : end + 1]
                out.loc[idx[end]] = float(np.trapezoid(v, dx=interval_h))
        elif direction == "future":
            for start in range(0, n - n_points + 1):
                end = start + n_points
                v = values[start:end]
                out.loc[idx[start]] = float(np.trapezoid(v, dx=interval_h))
        else:
            raise ValueError("direction must be 'hist' or 'future'")
    return out


def make_models(cfg: Config = CFG) -> Dict[str, object]:
    return {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            random_state=cfg.random_state,
            loss="squared_error",
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=350,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=cfg.random_state,
            n_jobs=-1,
        ),
    }


def sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    sanitize_for_csv(df).to_csv(path, index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# 1. Load, validate, and recompute analysis variables
# -----------------------------------------------------------------------------

def load_and_prepare(path: Path, cfg: Config = CFG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    required = [
        "timestamp",
        "indoor_co2_ppm",
        "outdoor_co2_ppm",
        "daily_arrival_passengers",
        "daily_departure_passengers",
        "daily_total_passengers_arr_plus_dep",
        "daily_service_disruption_flag",
        "scheduled_flights_prev30",
        "scheduled_flights_next30",
        "scheduled_flights_prev60",
        "scheduled_flights_next60",
        "est_departure_passengers_prev30",
        "est_departure_passengers_next30",
        "est_departure_passengers_prev60",
        "est_departure_passengers_next60",
    ]
    ensure_columns(df, required)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.normalize()
    df["time"] = df["timestamp"].dt.strftime("%H:%M")
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dow"] = df["timestamp"].dt.dayofweek

    numeric_cols = [
        "indoor_co2_ppm", "outdoor_co2_ppm",
        "daily_arrival_passengers", "daily_departure_passengers",
        "daily_total_passengers_arr_plus_dep", "daily_service_disruption_flag",
        "scheduled_flights_prev30", "scheduled_flights_next30",
        "scheduled_flights_prev60", "scheduled_flights_next60",
        "est_departure_passengers_prev30", "est_departure_passengers_next30",
        "est_departure_passengers_prev60", "est_departure_passengers_next60",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Core physical variables. Always recompute from raw indoor/outdoor columns.
    df["indoor_minus_outdoor_ppm_recalc"] = df["indoor_co2_ppm"] - df["outdoor_co2_ppm"]
    df["excess_co2_ppm_recalc"] = df["indoor_minus_outdoor_ppm_recalc"].clip(lower=0.0)

    # Cyclical calendar terms.
    tod_h = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * tod_h / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * tod_h / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    # Within-day lags only: never let a lag jump from 18:00 to next-day 08:00.
    for base in ["indoor_co2_ppm", "outdoor_co2_ppm", "excess_co2_ppm_recalc"]:
        for lag in [1, 2]:
            df[f"{base}_lag{lag}"] = df.groupby("date")[base].shift(lag)

    # Recompute absolute and excess integrals for 1/2/3 h.
    interval_h = cfg.interval_min / 60.0
    for h in cfg.horizons_h:
        for prefix, col in [
            ("abs", "indoor_co2_ppm"),
            ("exc", "excess_co2_ppm_recalc"),
        ]:
            df[f"hist_{prefix}_{h}h_recalc"] = trapezoid_rolling_by_day(
                df, col, h, "hist", interval_h
            )
            df[f"future_{prefix}_{h}h_recalc"] = trapezoid_rolling_by_day(
                df, col, h, "future", interval_h
            )

    # QC checks comparing uploaded precomputed values with recomputed values.
    qc_rows = []
    qc_rows.append(("n_rows", len(df)))
    qc_rows.append(("n_unique_days", df["date"].nunique()))
    qc_rows.append(("duplicate_timestamps", int(df["timestamp"].duplicated().sum())))
    qc_rows.append(("indoor_lt_outdoor_count", int((df["indoor_co2_ppm"] < df["outdoor_co2_ppm"]).sum())))
    qc_rows.append(("negative_indoor_count", int((df["indoor_co2_ppm"] < 0).sum())))
    qc_rows.append(("negative_outdoor_count", int((df["outdoor_co2_ppm"] < 0).sum())))
    qc_rows.append(("negative_daily_departures", int((df["daily_departure_passengers"] < 0).sum())))

    rows_per_day = df.groupby("date").size()
    qc_rows.append(("days_not_expected_rows", int((rows_per_day != cfg.rows_per_day_expected).sum())))
    qc_rows.append(("min_rows_per_day", int(rows_per_day.min())))
    qc_rows.append(("max_rows_per_day", int(rows_per_day.max())))

    # Verify daily total = arrival + departure.
    passenger_balance = (
        df["daily_arrival_passengers"] + df["daily_departure_passengers"]
        - df["daily_total_passengers_arr_plus_dep"]
    ).abs()
    qc_rows.append(("max_daily_passenger_balance_error", float(passenger_balance.max())))

    # Verify common precomputed exposure columns if present.
    for h in cfg.horizons_h:
        for kind in ["hist", "future"]:
            for prefix in ["abs", "exc"]:
                original = f"{kind}_{prefix}_{h}h_ppm_h"
                recalced = f"{kind}_{prefix}_{h}h_recalc"
                if original in df.columns:
                    diff = (pd.to_numeric(df[original], errors="coerce") - df[recalced]).abs()
                    qc_rows.append((f"max_abs_diff_{original}_vs_recalc", float(diff.max(skipna=True))))

    qc = pd.DataFrame(qc_rows, columns=["check", "value"])
    return df, qc


# -----------------------------------------------------------------------------
# 2. Feature sets and chronological split
# -----------------------------------------------------------------------------

def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    co2_time = [
        "indoor_co2_ppm",
        "outdoor_co2_ppm",
        "excess_co2_ppm_recalc",
        "indoor_co2_ppm_lag1",
        "indoor_co2_ppm_lag2",
        "outdoor_co2_ppm_lag1",
        "outdoor_co2_ppm_lag2",
        "excess_co2_ppm_recalc_lag1",
        "excess_co2_ppm_recalc_lag2",
        "hist_exc_1h_recalc",
        "hist_exc_2h_recalc",
        "hist_abs_1h_recalc",
        "hist_abs_2h_recalc",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]

    schedule = co2_time + [
        "scheduled_flights_prev30",
        "scheduled_flights_next30",
        "scheduled_flights_prev60",
        "scheduled_flights_next60",
    ]

    retrospective = schedule + [
        "est_departure_passengers_prev30",
        "est_departure_passengers_next30",
        "est_departure_passengers_prev60",
        "est_departure_passengers_next60",
        "daily_arrival_passengers",
        "daily_departure_passengers",
        "daily_total_passengers_arr_plus_dep",
        "daily_service_disruption_flag",
    ]

    # Optional columns may not be present in all reconstructed datasets.
    if "template_allocation_coverage_ratio" in df.columns:
        retrospective.append("template_allocation_coverage_ratio")

    return {
        "co2_time": co2_time,
        "primary_schedule": schedule,
        "retrospective_passenger_enriched": retrospective,
    }


def chronological_day_split(df: pd.DataFrame, cfg: Config = CFG) -> Dict[str, np.ndarray]:
    dates = np.array(sorted(df["date"].dropna().unique()))
    if len(dates) != cfg.n_train_days + cfg.n_val_days + cfg.n_test_days:
        # Generic fallback that preserves whole-day chronological blocking.
        n = len(dates)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        n_test = n - n_train - n_val
        return {
            "train": dates[:n_train],
            "validation": dates[n_train:n_train+n_val],
            "test": dates[n_train+n_val:n_train+n_val+n_test],
        }

    a = cfg.n_train_days
    b = a + cfg.n_val_days
    return {
        "train": dates[:a],
        "validation": dates[a:b],
        "test": dates[b:],
    }


def supervised_subset(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    days: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    cols = list(dict.fromkeys(["timestamp", "date"] + list(features) + [target]))
    x = df.loc[df["date"].isin(days), cols].copy()
    x = x.dropna(subset=list(features) + [target])
    return x


# -----------------------------------------------------------------------------
# 3. Descriptive analyses: instantaneous vs cumulative and passenger decoupling
# -----------------------------------------------------------------------------

def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    first_cols = [
        "daily_arrival_passengers",
        "daily_departure_passengers",
        "daily_total_passengers_arr_plus_dep",
        "daily_service_disruption_flag",
    ]
    if "daily_service_note" in df.columns:
        first_cols.append("daily_service_note")
    if "template_allocation_coverage_ratio" in df.columns:
        first_cols.append("template_allocation_coverage_ratio")

    g = df.groupby("date", sort=True)
    daily = pd.DataFrame({
        "date": g.size().index,
        "n_observations": g.size().values,
        "mean_indoor_co2_ppm": g["indoor_co2_ppm"].mean().values,
        "max_indoor_co2_ppm": g["indoor_co2_ppm"].max().values,
        "mean_outdoor_co2_ppm": g["outdoor_co2_ppm"].mean().values,
        "mean_excess_co2_ppm": g["excess_co2_ppm_recalc"].mean().values,
        "max_excess_co2_ppm": g["excess_co2_ppm_recalc"].max().values,
        "max_hist_exc_2h_ppm_h": g["hist_exc_2h_recalc"].max().values,
        "max_future_exc_2h_ppm_h": g["future_exc_2h_recalc"].max().values,
        "max_hist_abs_2h_ppm_h": g["hist_abs_2h_recalc"].max().values,
        "max_future_abs_2h_ppm_h": g["future_abs_2h_recalc"].max().values,
    })
    for c in first_cols:
        daily[c] = g[c].first().values

    daily["zero_departure_day"] = (daily["daily_departure_passengers"] == 0).astype(int)
    return daily


def correlation_tables(df: pd.DataFrame, daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    point_vars = [
        "indoor_co2_ppm", "outdoor_co2_ppm", "excess_co2_ppm_recalc",
        "hist_exc_2h_recalc", "hist_abs_2h_recalc",
        "scheduled_flights_prev30", "scheduled_flights_next30",
        "est_departure_passengers_prev30", "est_departure_passengers_next30",
    ]
    point_vars = [c for c in point_vars if c in df.columns]
    corr_point = df[point_vars].corr(method="spearman")
    corr_point.index.name = "variable"
    corr_point = corr_point.reset_index()

    daily_vars = [
        "daily_arrival_passengers", "daily_departure_passengers",
        "daily_total_passengers_arr_plus_dep",
        "mean_indoor_co2_ppm", "max_indoor_co2_ppm",
        "max_hist_exc_2h_ppm_h", "max_hist_abs_2h_ppm_h",
    ]
    corr_daily = daily[daily_vars].corr(method="spearman")
    corr_daily.index.name = "variable"
    corr_daily = corr_daily.reset_index()
    return corr_point, corr_daily


def exposure_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in CFG.horizons_h:
        a = df[f"hist_abs_{h}h_recalc"]
        e = df[f"hist_exc_{h}h_recalc"]
        v = pd.concat([a, e], axis=1).dropna()
        rows.append({
            "horizon_h": h,
            "n": len(v),
            "corr_pearson_abs_vs_excess": float(v.iloc[:, 0].corr(v.iloc[:, 1], method="pearson")),
            "corr_spearman_abs_vs_excess": float(v.iloc[:, 0].corr(v.iloc[:, 1], method="spearman")),
            "mean_abs_ppm_h": float(v.iloc[:, 0].mean()),
            "mean_excess_ppm_h": float(v.iloc[:, 1].mean()),
            "max_abs_ppm_h": float(v.iloc[:, 0].max()),
            "max_excess_ppm_h": float(v.iloc[:, 1].max()),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 4. Model evaluation across horizons and feature sets
# -----------------------------------------------------------------------------

def evaluate_horizons(
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    primary_features: Sequence[str],
    cfg: Config = CFG,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], pd.DataFrame]]:
    rows = []
    prediction_tables: Dict[Tuple[int, str], pd.DataFrame] = {}

    for h in cfg.horizons_h:
        target = f"future_exc_{h}h_recalc"
        train = supervised_subset(df, primary_features, target, splits["train"])
        val = supervised_subset(df, primary_features, target, splits["validation"])
        test = supervised_subset(df, primary_features, target, splits["test"])

        Xtr, ytr = train[list(primary_features)], train[target]
        Xte, yte = test[list(primary_features)], test[target]

        # Physical stationary-excess baseline.
        base_pred = h * test["excess_co2_ppm_recalc"].to_numpy(float)
        m = metrics(yte, base_pred)
        rows.append({"horizon_h": h, "model": "StationaryExcess", "split": "test", **m})

        pred_frame = test[["timestamp", "date", target]].copy()
        pred_frame["StationaryExcess"] = base_pred

        for name, model in make_models(cfg).items():
            fitted = clone(model).fit(Xtr, ytr)
            pred = fitted.predict(Xte)
            m = metrics(yte, pred)
            rows.append({"horizon_h": h, "model": name, "split": "test", **m})
            pred_frame[name] = pred

            # Validation metrics are saved separately to show chronology-aware tuning evidence.
            if len(val):
                pv = fitted.predict(val[list(primary_features)])
                mv = metrics(val[target], pv)
                rows.append({"horizon_h": h, "model": name, "split": "validation", **mv})

        prediction_tables[(h, "excess")] = pred_frame

    return pd.DataFrame(rows), prediction_tables


def compare_feature_sets_2h(
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    feature_dict: Dict[str, List[str]],
    cfg: Config = CFG,
) -> pd.DataFrame:
    target = "future_exc_2h_recalc"
    rows = []

    for fs_name, features in feature_dict.items():
        train = supervised_subset(df, features, target, splits["train"])
        test = supervised_subset(df, features, target, splits["test"])
        for model_name, model in make_models(cfg).items():
            fitted = clone(model).fit(train[features], train[target])
            pred = fitted.predict(test[features])
            rows.append({
                "feature_set": fs_name,
                "model": model_name,
                **metrics(test[target], pred),
            })
    return pd.DataFrame(rows)


def absolute_endpoint_robustness(
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    features: Sequence[str],
    cfg: Config = CFG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = "future_abs_2h_recalc"
    train = supervised_subset(df, features, target, splits["train"])
    test = supervised_subset(df, features, target, splits["test"])

    rows = []
    pred_table = test[["timestamp", "date", target]].copy()

    # Persistence baseline for absolute indoor CO2.
    base = cfg.primary_horizon_h * test["indoor_co2_ppm"].to_numpy(float)
    rows.append({"model": "StationaryIndoorCO2", **metrics(test[target], base)})
    pred_table["StationaryIndoorCO2"] = base

    for name, model in make_models(cfg).items():
        fitted = clone(model).fit(train[list(features)], train[target])
        pred = fitted.predict(test[list(features)])
        rows.append({"model": name, **metrics(test[target], pred)})
        pred_table[name] = pred
    return pd.DataFrame(rows), pred_table


# -----------------------------------------------------------------------------
# 5. Exposure states, severity error, and applicability domain
# -----------------------------------------------------------------------------

def training_thresholds(
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    features: Sequence[str],
    cfg: Config = CFG,
) -> Dict[str, float]:
    # Use the supervised 2-h training set so thresholds correspond to records
    # actually eligible for the prediction analysis.
    target = "future_exc_2h_recalc"
    tr = supervised_subset(df, features, target, splits["train"])
    s = tr["hist_exc_2h_recalc"].dropna()
    return {
        "warning": float(s.quantile(cfg.warning_q)),
        "severe": float(s.quantile(cfg.severe_q)),
        "critical": float(s.quantile(cfg.critical_q)),
    }


def severity_label(values: pd.Series, thresholds: Dict[str, float]) -> pd.Series:
    v = values.to_numpy(float)
    labels = np.full(len(v), "normal", dtype=object)
    labels[v >= thresholds["warning"]] = "warning"
    labels[v >= thresholds["severe"]] = "severe"
    labels[v >= thresholds["critical"]] = "critical"
    return pd.Series(labels, index=values.index)


def severity_performance(
    prediction_table: pd.DataFrame,
    target_col: str,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    y = prediction_table[target_col]
    subsets = {
        "all": pd.Series(True, index=prediction_table.index),
        "warning_or_higher": y >= thresholds["warning"],
        "severe_or_higher": y >= thresholds["severe"],
        "critical_or_higher": y >= thresholds["critical"],
    }
    rows = []
    for subset_name, mask in subsets.items():
        for model in ["GradientBoosting", "RandomForest"]:
            if model not in prediction_table.columns or int(mask.sum()) == 0:
                continue
            m = metrics(y[mask], prediction_table.loc[mask, model])
            rows.append({"subset": subset_name, "model": model, **m})
    return pd.DataFrame(rows)


def applicability_domain(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    cfg: Config = CFG,
) -> pd.DataFrame:
    Xtr = train_X.astype(float).copy()
    Xte = test_X.astype(float).copy()

    low = Xtr.quantile(cfg.range_low_q)
    high = Xtr.quantile(cfg.range_high_q)
    width = high - low

    range_scores = []
    for _, row in Xte.iterrows():
        scores = []
        for c in Xtr.columns:
            w = float(width[c])
            if not np.isfinite(w) or abs(w) < 1e-12:
                scores.append(0.0)
                continue
            x = float(row[c])
            l = float(low[c])
            u = float(high[c])
            scores.append(max(0.0, (x-u)/w, (l-x)/w))
        range_scores.append(max(scores) if scores else 0.0)

    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    Zte = scaler.transform(Xte)

    # Leave-one-out training reference: k+1 neighbors then remove self.
    nn_train = NearestNeighbors(n_neighbors=cfg.knn_k + 1).fit(Ztr)
    d_train, _ = nn_train.kneighbors(Ztr)
    loo_mean = d_train[:, 1:cfg.knn_k+1].mean(axis=1)
    local_ref = float(np.quantile(loo_mean, cfg.local_support_q))
    if local_ref <= 0:
        local_ref = 1.0

    nn = NearestNeighbors(n_neighbors=cfg.knn_k).fit(Ztr)
    d_test, _ = nn.kneighbors(Zte)
    local_mean = d_test.mean(axis=1)
    local_score = local_mean / local_ref

    result = pd.DataFrame({
        "range_score": np.asarray(range_scores, dtype=float),
        "local_score": np.asarray(local_score, dtype=float),
    }, index=test_X.index)
    result["in_domain"] = (
        (result["range_score"] <= 1e-12) & (result["local_score"] <= 1.0)
    )
    return result


def domain_performance(
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    features: Sequence[str],
    cfg: Config = CFG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = "future_exc_2h_recalc"
    train = supervised_subset(df, features, target, splits["train"])
    test = supervised_subset(df, features, target, splits["test"])

    ad = applicability_domain(train[list(features)], test[list(features)], cfg)
    detail_cols = [
        "timestamp", "date", target,
        "hist_exc_2h_recalc", "excess_co2_ppm_recalc", "indoor_co2_ppm",
    ]
    detail = test[detail_cols].copy()
    detail = detail.join(ad)
    # A continuous support score is used only for ranking/abstention analysis.
    # In-domain records satisfy range_score == 0 and local_score <= 1.
    # Any marginal range exceedance is therefore mapped above 1.
    detail["support_score"] = np.maximum(
        detail["local_score"].to_numpy(float),
        1.0 + detail["range_score"].to_numpy(float),
    )

    rows = []
    for model_name, model in make_models(cfg).items():
        fitted = clone(model).fit(train[list(features)], train[target])
        pred = fitted.predict(test[list(features)])
        detail[f"pred_{model_name}"] = pred

        for label, mask in {
            "in_domain": detail["in_domain"],
            "out_of_domain": ~detail["in_domain"],
        }.items():
            if int(mask.sum()) == 0:
                continue
            rows.append({
                "domain": label,
                "model": model_name,
                **metrics(detail.loc[mask, target], detail.loc[mask, f"pred_{model_name}"]),
            })

    return pd.DataFrame(rows), detail


# -----------------------------------------------------------------------------
# 6. Routine-to-severe range-holdout stress tests
# -----------------------------------------------------------------------------

def range_holdout_test(
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    features: Sequence[str],
    holdout_variable: str,
    cfg: Config = CFG,
) -> pd.DataFrame:
    target = "future_exc_2h_recalc"
    train = supervised_subset(df, features, target, splits["train"])
    test = supervised_subset(df, features, target, splits["test"])

    # holdout variable must be present; supervised_subset keeps only features+target,
    # so add it from df by index if it isn't already in the feature set.
    if holdout_variable not in train.columns:
        train[holdout_variable] = df.loc[train.index, holdout_variable]
    if holdout_variable not in test.columns:
        test[holdout_variable] = df.loc[test.index, holdout_variable]

    cutoff = float(train[holdout_variable].quantile(cfg.holdout_q))
    train_routine = train[train[holdout_variable] <= cutoff].dropna(subset=[holdout_variable])
    test_in = test[test[holdout_variable] <= cutoff].dropna(subset=[holdout_variable])
    test_ext = test[test[holdout_variable] > cutoff].dropna(subset=[holdout_variable])

    rows = []
    for model_name, model in make_models(cfg).items():
        fitted = clone(model).fit(train_routine[list(features)], train_routine[target])
        pred_in = fitted.predict(test_in[list(features)]) if len(test_in) else np.array([])
        pred_ext = fitted.predict(test_ext[list(features)]) if len(test_ext) else np.array([])

        m_in = metrics(test_in[target], pred_in) if len(test_in) else {k: np.nan for k in ["n","r2","rmse","mae","bias"]}
        m_ext = metrics(test_ext[target], pred_ext) if len(test_ext) else {k: np.nan for k in ["n","r2","rmse","mae","bias"]}

        edr = float(m_ext["rmse"] / m_in["rmse"]) if np.isfinite(m_ext["rmse"]) and np.isfinite(m_in["rmse"]) and m_in["rmse"] > 0 else np.nan
        mean_ext = float(test_ext[target].mean()) if len(test_ext) else np.nan
        bias_pct = float(100 * m_ext["bias"] / mean_ext) if mean_ext and np.isfinite(mean_ext) else np.nan

        rows.append({
            "holdout_variable": holdout_variable,
            "cutoff_q": cfg.holdout_q,
            "cutoff": cutoff,
            "model": model_name,
            "n_train_routine": len(train_routine),
            "n_test_in_range": int(m_in["n"]) if np.isfinite(m_in["n"]) else 0,
            "n_test_extrapolation": int(m_ext["n"]) if np.isfinite(m_ext["n"]) else 0,
            "rmse_in_range": m_in["rmse"],
            "rmse_extrapolation": m_ext["rmse"],
            "mae_in_range": m_in["mae"],
            "mae_extrapolation": m_ext["mae"],
            "bias_extrapolation": m_ext["bias"],
            "mean_observed_extrapolation": mean_ext,
            "bias_pct_of_mean_extrapolation": bias_pct,
            "edr": edr,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 7. Reliability-aware supervisory decision support
# -----------------------------------------------------------------------------

def binary_event_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict[str, float]:
    """Binary event-detection metrics with severe future exposure as the event."""
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length")

    tp = int(np.sum(yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))

    def safe_div(a: float, b: float) -> float:
        return float(a / b) if b else np.nan

    return {
        "n": int(len(yt)),
        "event_n": int(np.sum(yt)),
        "non_event_n": int(np.sum(~yt)),
        "event_prevalence": safe_div(np.sum(yt), len(yt)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "precision": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "false_negative_rate": safe_div(fn, tp + fn),
        "false_positive_rate": safe_div(fp, tn + fp),
        "accuracy": safe_div(tp + tn, len(yt)),
    }


def future_severe_detection(
    detail: pd.DataFrame,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    """Evaluate severe-event detection overall and by applicability domain.

    The severe threshold is estimated from training data only. No decision
    threshold is tuned on the chronological test set.
    """
    severe = float(thresholds["severe"])
    y_event = detail["future_exc_2h_recalc"].to_numpy(float) >= severe
    rows = []

    subset_masks = {
        "all": np.ones(len(detail), dtype=bool),
        "in_domain": detail["in_domain"].to_numpy(bool),
        "out_of_domain": ~detail["in_domain"].to_numpy(bool),
    }

    # Current measured cumulative state is included as a non-forecast reference.
    predictors = {
        "MeasuredCurrentState": detail["hist_exc_2h_recalc"].to_numpy(float) >= severe,
        "GradientBoosting": detail["pred_GradientBoosting"].to_numpy(float) >= severe,
        "RandomForest": detail["pred_RandomForest"].to_numpy(float) >= severe,
    }

    for subset_name, mask in subset_masks.items():
        for model_name, pred_event in predictors.items():
            if int(np.sum(mask)) == 0:
                continue
            m = binary_event_metrics(y_event[mask], pred_event[mask])
            rows.append({
                "subset": subset_name,
                "model": model_name,
                "severe_threshold_ppm_h": severe,
                **m,
            })
    return pd.DataFrame(rows)


def selective_prediction_risk_coverage(
    detail: pd.DataFrame,
    thresholds: Dict[str, float],
    cfg: Config = CFG,
) -> pd.DataFrame:
    """Risk--coverage analysis using training-defined applicability support.

    Test records are ranked only by the feature-space support score. At lower
    coverage, the least-supported records are deferred. This is a selective
    prediction analysis, not post-hoc error filtering: observed test outcomes
    are never used to decide which records are retained.
    """
    severe = float(thresholds["severe"])
    y = detail["future_exc_2h_recalc"].to_numpy(float)
    future_severe = y >= severe
    score = detail["support_score"].to_numpy(float)
    order = np.argsort(score, kind="mergesort")
    n = len(detail)
    total_severe = int(np.sum(future_severe))
    rows = []

    for model_name in ["GradientBoosting", "RandomForest"]:
        pred = detail[f"pred_{model_name}"].to_numpy(float)

        for requested_coverage in cfg.selective_coverages:
            n_keep = max(1, min(n, int(math.ceil(requested_coverage * n))))
            accepted_idx = order[:n_keep]
            accepted = np.zeros(n, dtype=bool)
            accepted[accepted_idx] = True

            reg = metrics(y[accepted], pred[accepted])
            cls = binary_event_metrics(
                future_severe[accepted],
                pred[accepted] >= severe,
            )
            severe_retained = int(np.sum(future_severe & accepted))
            severe_deferred = total_severe - severe_retained

            rows.append({
                "model": model_name,
                "gate": f"support_rank_{requested_coverage:.2f}",
                "requested_coverage": requested_coverage,
                "accepted_n": int(np.sum(accepted)),
                "deferred_n": int(np.sum(~accepted)),
                "actual_coverage": float(np.mean(accepted)),
                "max_support_score_accepted": float(np.max(score[accepted])),
                "rmse": reg["rmse"],
                "mae": reg["mae"],
                "bias": reg["bias"],
                "r2": reg["r2"],
                "accepted_future_severe_n": severe_retained,
                "deferred_future_severe_n": severe_deferred,
                "future_severe_retained_fraction": (
                    float(severe_retained / total_severe) if total_severe else np.nan
                ),
                "accepted_severe_sensitivity": cls["sensitivity"],
                "accepted_severe_false_negative_rate": cls["false_negative_rate"],
                "accepted_severe_precision": cls["precision"],
            })

        # Also report the manuscript's fixed applicability-domain gate.
        accepted = detail["in_domain"].to_numpy(bool)
        if int(np.sum(accepted)):
            reg = metrics(y[accepted], pred[accepted])
            cls = binary_event_metrics(
                future_severe[accepted],
                pred[accepted] >= severe,
            )
            severe_retained = int(np.sum(future_severe & accepted))
            rows.append({
                "model": model_name,
                "gate": "fixed_in_domain",
                "requested_coverage": np.nan,
                "accepted_n": int(np.sum(accepted)),
                "deferred_n": int(np.sum(~accepted)),
                "actual_coverage": float(np.mean(accepted)),
                "max_support_score_accepted": float(np.max(score[accepted])),
                "rmse": reg["rmse"],
                "mae": reg["mae"],
                "bias": reg["bias"],
                "r2": reg["r2"],
                "accepted_future_severe_n": severe_retained,
                "deferred_future_severe_n": total_severe - severe_retained,
                "future_severe_retained_fraction": (
                    float(severe_retained / total_severe) if total_severe else np.nan
                ),
                "accepted_severe_sensitivity": cls["sensitivity"],
                "accepted_severe_false_negative_rate": cls["false_negative_rate"],
                "accepted_severe_precision": cls["precision"],
            })

    return pd.DataFrame(rows)


def supervisory_policy_analysis(
    detail: pd.DataFrame,
    thresholds: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare forecast-only and reliability-aware supervisory policies.

    Three policy concepts are evaluated without fitting any new threshold to the
    test set:

    1. forecast_only: escalate only when predicted future 2-h exposure is severe.
    2. measured_plus_forecast: escalate when the measured historical 2-h state
       is already severe OR the forecast is severe.
    3. reliability_aware: escalate for measured/predicted severe states; when a
       non-severe forecast is OOD, defer automatic authority to operator review
       rather than treating that forecast as evidence for normal operation.

    "Defer" is not counted as a successful prediction. It is a supervisory
    action indicating that the model should not autonomously authorize a normal
    ventilation state outside empirical support.
    """
    severe = float(thresholds["severe"])
    y_future_severe = detail["future_exc_2h_recalc"].to_numpy(float) >= severe
    measured_severe = detail["hist_exc_2h_recalc"].to_numpy(float) >= severe
    ood = ~detail["in_domain"].to_numpy(bool)

    rows = []
    action_detail = detail[[
        "timestamp", "date", "future_exc_2h_recalc", "hist_exc_2h_recalc",
        "in_domain", "range_score", "local_score", "support_score",
    ]].copy()
    action_detail["observed_future_severe"] = y_future_severe
    action_detail["measured_current_severe"] = measured_severe

    for model_name in ["GradientBoosting", "RandomForest"]:
        pred = detail[f"pred_{model_name}"].to_numpy(float)
        pred_severe = pred >= severe
        action_detail[f"pred_{model_name}"] = pred
        action_detail[f"predicted_severe_{model_name}"] = pred_severe

        policies = {}
        policies["forecast_only"] = np.where(pred_severe, "escalate", "normal")
        policies["measured_plus_forecast"] = np.where(
            measured_severe | pred_severe, "escalate", "normal"
        )

        reliability_action = np.full(len(detail), "normal", dtype=object)
        reliability_action[ood] = "defer"
        reliability_action[measured_severe | pred_severe] = "escalate"
        policies["reliability_aware"] = reliability_action

        for policy_name, action in policies.items():
            action = np.asarray(action, dtype=object)
            action_detail[f"action_{model_name}_{policy_name}"] = action

            escalate = action == "escalate"
            defer = action == "defer"
            normal = action == "normal"
            severe_n = int(np.sum(y_future_severe))
            nonsevere = ~y_future_severe
            nonsevere_n = int(np.sum(nonsevere))

            severe_escalate = int(np.sum(y_future_severe & escalate))
            severe_defer = int(np.sum(y_future_severe & defer))
            severe_normal = int(np.sum(y_future_severe & normal))
            nonsevere_escalate = int(np.sum(nonsevere & escalate))
            nonsevere_defer = int(np.sum(nonsevere & defer))

            rows.append({
                "model": model_name,
                "policy": policy_name,
                "severe_threshold_ppm_h": severe,
                "n": len(detail),
                "future_severe_n": severe_n,
                "escalate_n": int(np.sum(escalate)),
                "defer_n": int(np.sum(defer)),
                "normal_n": int(np.sum(normal)),
                "escalate_rate": float(np.mean(escalate)),
                "defer_rate": float(np.mean(defer)),
                "normal_rate": float(np.mean(normal)),
                "future_severe_escalate_n": severe_escalate,
                "future_severe_defer_n": severe_defer,
                "future_severe_normal_n": severe_normal,
                "future_severe_escalate_rate": (
                    float(severe_escalate / severe_n) if severe_n else np.nan
                ),
                "future_severe_defer_rate": (
                    float(severe_defer / severe_n) if severe_n else np.nan
                ),
                "future_severe_normal_rate": (
                    float(severe_normal / severe_n) if severe_n else np.nan
                ),
                "future_severe_escalate_or_defer_rate": (
                    float((severe_escalate + severe_defer) / severe_n) if severe_n else np.nan
                ),
                "nonsevere_escalate_n": nonsevere_escalate,
                "nonsevere_defer_n": nonsevere_defer,
                "nonsevere_escalate_rate": (
                    float(nonsevere_escalate / nonsevere_n) if nonsevere_n else np.nan
                ),
                "nonsevere_defer_rate": (
                    float(nonsevere_defer / nonsevere_n) if nonsevere_n else np.nan
                ),
            })

    return pd.DataFrame(rows), action_detail


def severe_day_event_robustness(
    action_detail: pd.DataFrame,
    cfg: Config = CFG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse repeated 30-min severe records to day- and episode-level units.

    The record-level test observations are serially correlated. To avoid
    treating every overlapping 2-h window as an independent severe event, this
    robustness analysis reports two coarser units:

    * severe day: a calendar day containing >=1 observed future-severe record;
    * severe episode: consecutive future-severe records separated by no more
      than one sampling interval (30 min by default).

    For each unit, the primary robustness outcome is the action at the FIRST
    observed future-severe record. A unit is "initially protected" when that
    first action is either ``escalate`` or ``defer``; ``defer`` means automatic
    normal authorization is withheld, not that the severe state was correctly
    forecast. ``Fully missed`` means every severe record in that day/episode
    was left ``normal``.
    """
    if action_detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = action_detail.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp")
    sev = d[d["observed_future_severe"].astype(bool)].copy()
    if sev.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Episode IDs are defined only across observed future-severe records.
    # A new episode starts at a new day or after a gap larger than one sampling
    # interval. This makes the unit definition explicit and reproducible.
    max_gap = pd.Timedelta(minutes=cfg.interval_min)
    new_episode = (
        sev["date"].astype(str).ne(sev["date"].astype(str).shift())
        | sev["timestamp"].diff().gt(max_gap)
    )
    sev["episode_id"] = new_episode.cumsum().astype(int)

    policies = ["forecast_only", "measured_plus_forecast", "reliability_aware"]
    models = ["GradientBoosting", "RandomForest"]
    detail_rows = []
    summary_rows = []

    for scope, group_col in [("day", "date"), ("episode", "episode_id")]:
        for model_name in models:
            for policy_name in policies:
                action_col = f"action_{model_name}_{policy_name}"
                if action_col not in sev.columns:
                    continue

                unit_rows = []
                for unit_id, g in sev.groupby(group_col, sort=True):
                    g = g.sort_values("timestamp")
                    actions = g[action_col].astype(str).str.lower()
                    first_action = actions.iloc[0]
                    initially_protected = first_action in {"escalate", "defer"}
                    fully_missed = bool((actions == "normal").all())
                    any_normal = bool((actions == "normal").any())
                    row = {
                        "scope": scope,
                        "unit_id": str(unit_id),
                        "date": str(g["date"].iloc[0]),
                        "model": model_name,
                        "policy": policy_name,
                        "n_severe_records": int(len(g)),
                        "first_severe_timestamp": g["timestamp"].iloc[0],
                        "last_severe_timestamp": g["timestamp"].iloc[-1],
                        "first_action": first_action,
                        "initially_protected": initially_protected,
                        "fully_missed": fully_missed,
                        "contains_any_normal_severe_record": any_normal,
                    }
                    detail_rows.append(row)
                    unit_rows.append(row)

                u = pd.DataFrame(unit_rows)
                n_units = len(u)
                summary_rows.append({
                    "scope": scope,
                    "model": model_name,
                    "policy": policy_name,
                    "n_units": n_units,
                    "initial_escalate_n": int((u["first_action"] == "escalate").sum()),
                    "initial_defer_n": int((u["first_action"] == "defer").sum()),
                    "initial_normal_n": int((u["first_action"] == "normal").sum()),
                    "initially_protected_n": int(u["initially_protected"].sum()),
                    "initially_protected_rate": float(u["initially_protected"].mean()),
                    "fully_missed_n": int(u["fully_missed"].sum()),
                    "units_with_any_normal_severe_record_n": int(
                        u["contains_any_normal_severe_record"].sum()
                    ),
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def applicability_domain_sensitivity_analysis(
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    features: Sequence[str],
    prediction_detail: pd.DataFrame,
    thresholds: Dict[str, float],
    cfg: Config = CFG,
) -> pd.DataFrame:
    """Sensitivity of OOD and supervisory conclusions to AD definitions.

    The grid is fixed in ``Config`` and varies only training-derived support
    definitions. Predictive models and the severe threshold are NOT re-tuned on
    the test set. For every AD definition, the function reports how many future
    severe records are classified OOD and whether reliability-aware supervision
    still prevents unsupported non-severe forecasts from authorizing ``normal``.
    Day-level protection is evaluated at the first observed severe record of
    each severe day to reduce dependence on overlapping 30-min windows.
    """
    target = "future_exc_2h_recalc"
    train = supervised_subset(df, features, target, splits["train"])
    test = supervised_subset(df, features, target, splits["test"])

    # Align by original dataframe index; domain_performance preserves this index.
    detail = prediction_detail.loc[test.index].copy()
    severe_thr = float(thresholds["severe"])
    observed_severe = detail[target].to_numpy(float) >= severe_thr
    measured_severe = detail["hist_exc_2h_recalc"].to_numpy(float) >= severe_thr
    severe_dates = pd.Series(detail.loc[observed_severe, "date"].astype(str)).drop_duplicates()
    n_severe_days = int(len(severe_dates))

    # Original-index labels for the first future-severe record on each severe day.
    severe_frame = detail.loc[observed_severe, ["date", "timestamp"]].copy()
    severe_frame["timestamp"] = pd.to_datetime(severe_frame["timestamp"])
    first_severe_indices = (
        severe_frame.sort_values("timestamp").groupby("date", sort=True).head(1).index
    )

    rows = []
    for range_low, range_high in cfg.ad_range_pairs:
        for local_q in cfg.ad_local_support_qs:
            cfg_ad = replace(
                cfg,
                range_low_q=float(range_low),
                range_high_q=float(range_high),
                local_support_q=float(local_q),
            )
            ad = applicability_domain(train[list(features)], test[list(features)], cfg_ad)
            in_domain = ad["in_domain"].reindex(detail.index).to_numpy(bool)
            ood = ~in_domain

            severe_n = int(observed_severe.sum())
            nonsevere = ~observed_severe
            nonsevere_n = int(nonsevere.sum())
            severe_ood_n = int(np.sum(observed_severe & ood))
            nonsevere_ood_n = int(np.sum(nonsevere & ood))
            baseline_definition = bool(
                np.isclose(range_low, cfg.range_low_q)
                and np.isclose(range_high, cfg.range_high_q)
                and np.isclose(local_q, cfg.local_support_q)
            )

            for model_name in ["GradientBoosting", "RandomForest"]:
                pred = detail[f"pred_{model_name}"].to_numpy(float)
                pred_severe = pred >= severe_thr
                escalate = measured_severe | pred_severe
                defer = (~escalate) & ood
                normal = (~escalate) & in_domain

                severe_escalate_n = int(np.sum(observed_severe & escalate))
                severe_defer_n = int(np.sum(observed_severe & defer))
                severe_normal_n = int(np.sum(observed_severe & normal))
                nonsevere_defer_n = int(np.sum(nonsevere & defer))

                # Day-level action at the first observed future-severe record.
                action = np.full(len(detail), "normal", dtype=object)
                action[defer] = "defer"
                action[escalate] = "escalate"
                action_s = pd.Series(action, index=detail.index)
                first_actions = action_s.loc[first_severe_indices].astype(str)
                day_protected_n = int(first_actions.isin(["escalate", "defer"]).sum())
                day_normal_n = int((first_actions == "normal").sum())

                rows.append({
                    "baseline_definition": baseline_definition,
                    "range_low_q": float(range_low),
                    "range_high_q": float(range_high),
                    "local_support_q": float(local_q),
                    "model": model_name,
                    "n_test": int(len(detail)),
                    "in_domain_n": int(in_domain.sum()),
                    "ood_n": int(ood.sum()),
                    "ood_rate": float(ood.mean()),
                    "future_severe_n": severe_n,
                    "future_severe_ood_n": severe_ood_n,
                    "future_severe_ood_rate": (
                        float(severe_ood_n / severe_n) if severe_n else np.nan
                    ),
                    "nonsevere_ood_n": nonsevere_ood_n,
                    "nonsevere_ood_rate": (
                        float(nonsevere_ood_n / nonsevere_n) if nonsevere_n else np.nan
                    ),
                    "severe_escalate_n": severe_escalate_n,
                    "severe_defer_n": severe_defer_n,
                    "severe_normal_n": severe_normal_n,
                    "severe_protected_rate": (
                        float((severe_escalate_n + severe_defer_n) / severe_n)
                        if severe_n else np.nan
                    ),
                    "nonsevere_defer_n": nonsevere_defer_n,
                    "nonsevere_defer_rate": (
                        float(nonsevere_defer_n / nonsevere_n)
                        if nonsevere_n else np.nan
                    ),
                    "severe_day_n": n_severe_days,
                    "severe_day_initially_protected_n": day_protected_n,
                    "severe_day_initially_normal_n": day_normal_n,
                    "severe_day_initially_protected_rate": (
                        float(day_protected_n / n_severe_days) if n_severe_days else np.nan
                    ),
                })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 8. Ventilation sensitivity analysis
# -----------------------------------------------------------------------------

def simulate_ventilation_event(
    event: pd.DataFrame,
    trigger_idx: int,
    ach: float,
    cfg: Config = CFG,
) -> np.ndarray:
    """Euler sensitivity simulation using effective source inferred under baseline ACH.

    Baseline ACH reproduces the observed trajectory interval by interval because
    the source term is defined from the observed finite difference.
    """
    dt = cfg.interval_min / 60.0
    cin = event["indoor_co2_ppm"].to_numpy(float)
    cout = event["outdoor_co2_ppm"].to_numpy(float)
    n = len(event)

    # Effective source inferred from observed trajectory under reference ACH.
    source = np.full(n - 1, np.nan)
    for i in range(n - 1):
        source[i] = (cin[i+1] - cin[i]) / dt - cfg.ach_base * (cout[i] - cin[i])

    sim = cin.copy()
    sim[0] = cin[0]
    for i in range(n - 1):
        ach_i = cfg.ach_base if i < trigger_idx else ach
        # Before trigger, anchor to observed trajectory to isolate post-trigger change.
        if i < trigger_idx:
            sim[i] = cin[i]
            sim[i+1] = cin[i+1]
        else:
            sim[i+1] = sim[i] + dt * (source[i] + ach_i * (cout[i] - sim[i]))
    return sim


def integrate_segment(values: np.ndarray, start: int, n_intervals: int, dt_h: float) -> float:
    end = start + n_intervals + 1
    if start < 0 or end > len(values):
        return np.nan
    return float(np.trapezoid(values[start:end], dx=dt_h))


def ventilation_sensitivity_analysis(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    cfg: Config = CFG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    profiles = []
    dt_h = cfg.interval_min / 60.0
    h_intervals = int(round(cfg.primary_horizon_h / dt_h))

    # Analyze all days with an observed severe historical exposure trigger,
    # and preserve disruption/zero-departure labels for interpretation.
    for date, day in df.groupby("date", sort=True):
        day = day.sort_values("timestamp").copy().reset_index(drop=True)
        eligible = day["hist_exc_2h_recalc"].notna() & (day["hist_exc_2h_recalc"] >= thresholds["severe"])
        if not eligible.any():
            continue

        trigger_idx = int(np.flatnonzero(eligible.to_numpy())[0])
        trigger_time = day.loc[trigger_idx, "timestamp"]
        obs = day["indoor_co2_ppm"].to_numpy(float)
        cout = day["outdoor_co2_ppm"].to_numpy(float)
        baseline_excess = np.maximum(obs - cout, 0.0)
        base_2h = integrate_segment(baseline_excess, trigger_idx, h_intervals, dt_h)

        for ach in cfg.ach_scenarios:
            sim = simulate_ventilation_event(day, trigger_idx, ach, cfg)
            sim_excess = np.maximum(sim - cout, 0.0)
            sim_2h = integrate_segment(sim_excess, trigger_idx, h_intervals, dt_h)
            reduction_pct = 100 * (base_2h - sim_2h) / base_2h if np.isfinite(base_2h) and base_2h > 0 and np.isfinite(sim_2h) else np.nan

            rows.append({
                "date": date,
                "trigger_time": trigger_time,
                "trigger_hist_exc_2h_ppm_h": float(day.loc[trigger_idx, "hist_exc_2h_recalc"]),
                "daily_departure_passengers": float(day.loc[0, "daily_departure_passengers"]),
                "daily_service_disruption_flag": int(day.loc[0, "daily_service_disruption_flag"]),
                "ach_h_minus1": ach,
                "following_2h_excess_ppm_h": sim_2h,
                "baseline_following_2h_excess_ppm_h": base_2h,
                "following_2h_reduction_pct": reduction_pct,
            })

            for i in range(len(day)):
                profiles.append({
                    "date": date,
                    "timestamp": day.loc[i, "timestamp"],
                    "ach_h_minus1": ach,
                    "simulated_indoor_co2_ppm": sim[i],
                    "observed_indoor_co2_ppm": obs[i],
                    "outdoor_co2_ppm": cout[i],
                    "trigger_time": trigger_time,
                })

    return pd.DataFrame(rows), pd.DataFrame(profiles)


# -----------------------------------------------------------------------------
# 9. Manuscript-ready compact outputs: exactly 3 tables + 3 figures
# -----------------------------------------------------------------------------

def build_manuscript_table_1(
    horizon_perf: pd.DataFrame,
    sev_perf: pd.DataFrame,
    domain_perf: pd.DataFrame,
    holdout: pd.DataFrame,
) -> pd.DataFrame:
    """Compact evidence table for prediction skill and reliability degradation.

    This table intentionally focuses on the primary 2-h endpoint. The complete
    1/2/3-h, feature-set, absolute-endpoint, and diagnostic outputs remain in
    raw_outputs/ for reproducibility and supplementary reporting.
    """
    rows = []

    overall = horizon_perf[
        (horizon_perf["horizon_h"] == 2)
        & (horizon_perf["split"] == "test")
        & (horizon_perf["model"].isin(["GradientBoosting", "RandomForest"]))
    ]
    for _, r in overall.iterrows():
        rows.append({
            "analysis_block": "chronological_test",
            "evaluation_state": "overall",
            "model": r["model"],
            "n": int(r["n"]),
            "r2": r["r2"],
            "rmse_ppm_h": r["rmse"],
            "mae_ppm_h": r["mae"],
            "bias_ppm_h": r["bias"],
            "edr": np.nan,
            "note": "Primary future 2-h cumulative excess CO2 endpoint",
        })

    sev_keep = sev_perf[
        sev_perf["subset"].isin(["warning_or_higher", "severe_or_higher"])
    ]
    labels = {
        "warning_or_higher": "warning_or_higher",
        "severe_or_higher": "severe_or_higher",
    }
    for _, r in sev_keep.iterrows():
        rows.append({
            "analysis_block": "severity",
            "evaluation_state": labels[r["subset"]],
            "model": r["model"],
            "n": int(r["n"]),
            "r2": r["r2"],
            "rmse_ppm_h": r["rmse"],
            "mae_ppm_h": r["mae"],
            "bias_ppm_h": r["bias"],
            "edr": np.nan,
            "note": "Severity threshold estimated from training data only",
        })

    for _, r in domain_perf.iterrows():
        rows.append({
            "analysis_block": "applicability_domain",
            "evaluation_state": r["domain"],
            "model": r["model"],
            "n": int(r["n"]),
            "r2": r["r2"],
            "rmse_ppm_h": r["rmse"],
            "mae_ppm_h": r["mae"],
            "bias_ppm_h": r["bias"],
            "edr": np.nan,
            "note": "Empirical training-support classification",
        })

    # Primary extrapolation stress test: hold out upper-range historical 2-h exposure.
    ext = holdout[holdout["holdout_variable"] == "hist_exc_2h_recalc"]
    for _, r in ext.iterrows():
        rows.append({
            "analysis_block": "routine_to_severe_holdout",
            "evaluation_state": "extrapolation_subset",
            "model": r["model"],
            "n": int(r["n_test_extrapolation"]),
            "r2": np.nan,
            "rmse_ppm_h": r["rmse_extrapolation"],
            "mae_ppm_h": r["mae_extrapolation"],
            "bias_ppm_h": r["bias_extrapolation"],
            "edr": r["edr"],
            "note": f"Training restricted to <=75th percentile; cutoff={r['cutoff']:.1f}",
        })

    return pd.DataFrame(rows)


def build_manuscript_table_2(
    policy: pd.DataFrame,
    day_event_robustness: pd.DataFrame,
    ad_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Compact supervisory table with record-, day-, and AD-robustness evidence."""
    if policy.empty:
        return pd.DataFrame()
    keep = policy.copy()
    keep["future_severe_escalate_pct"] = 100.0 * keep["future_severe_escalate_rate"]
    keep["future_severe_defer_pct"] = 100.0 * keep["future_severe_defer_rate"]
    keep["future_severe_normal_pct"] = 100.0 * keep["future_severe_normal_rate"]
    keep["nonsevere_defer_pct"] = 100.0 * keep["nonsevere_defer_rate"]
    keep["overall_defer_pct"] = 100.0 * keep["defer_rate"]

    # Add independent-unit robustness. These values are repeated per matching
    # model/policy row so the manuscript table can stay at one row per policy.
    if not day_event_robustness.empty:
        for scope in ["day", "episode"]:
            sub = day_event_robustness[day_event_robustness["scope"] == scope][[
                "model", "policy", "n_units", "initially_protected_n",
                "initially_protected_rate", "fully_missed_n",
            ]].copy()
            sub = sub.rename(columns={
                "n_units": f"severe_{scope}_n",
                "initially_protected_n": f"severe_{scope}_initially_protected_n",
                "initially_protected_rate": f"severe_{scope}_initially_protected_rate",
                "fully_missed_n": f"severe_{scope}_fully_missed_n",
            })
            keep = keep.merge(sub, on=["model", "policy"], how="left")

    # AD sensitivity applies specifically to the reliability-aware gate. We
    # report the worst result across the pre-specified 3x3 robustness grid.
    keep["ad_sensitivity_min_severe_protected_pct"] = np.nan
    keep["ad_sensitivity_min_day_protected_pct"] = np.nan
    keep["ad_sensitivity_max_nonsevere_defer_pct"] = np.nan
    if not ad_sensitivity.empty:
        for model_name in ["GradientBoosting", "RandomForest"]:
            z = ad_sensitivity[ad_sensitivity["model"] == model_name]
            mask = (keep["model"] == model_name) & (keep["policy"] == "reliability_aware")
            if len(z):
                keep.loc[mask, "ad_sensitivity_min_severe_protected_pct"] = (
                    100.0 * float(z["severe_protected_rate"].min())
                )
                keep.loc[mask, "ad_sensitivity_min_day_protected_pct"] = (
                    100.0 * float(z["severe_day_initially_protected_rate"].min())
                )
                keep.loc[mask, "ad_sensitivity_max_nonsevere_defer_pct"] = (
                    100.0 * float(z["nonsevere_defer_rate"].max())
                )

    # Percent form for independent-unit protection.
    for scope in ["day", "episode"]:
        col = f"severe_{scope}_initially_protected_rate"
        if col in keep.columns:
            keep[f"severe_{scope}_initially_protected_pct"] = 100.0 * keep[col]

    cols = [
        "model", "policy", "n", "future_severe_n",
        "future_severe_escalate_pct", "future_severe_defer_pct",
        "future_severe_normal_pct", "nonsevere_defer_pct", "overall_defer_pct",
        "severe_day_n", "severe_day_initially_protected_n",
        "severe_day_initially_protected_pct", "severe_day_fully_missed_n",
        "severe_episode_n", "severe_episode_initially_protected_n",
        "severe_episode_initially_protected_pct", "severe_episode_fully_missed_n",
        "ad_sensitivity_min_severe_protected_pct",
        "ad_sensitivity_min_day_protected_pct",
        "ad_sensitivity_max_nonsevere_defer_pct",
    ]
    cols = [c for c in cols if c in keep.columns]
    return keep[cols]


def build_manuscript_table_3(vent: pd.DataFrame) -> pd.DataFrame:
    """Ventilation sensitivity for the full-suspension severe events only."""
    if vent.empty:
        return vent.copy()
    z = vent[vent["daily_departure_passengers"] == 0].copy()
    cols = [
        "date", "trigger_time", "trigger_hist_exc_2h_ppm_h",
        "ach_h_minus1", "baseline_following_2h_excess_ppm_h",
        "following_2h_excess_ppm_h", "following_2h_reduction_pct",
    ]
    z = z[cols].sort_values(["date", "ach_h_minus1"]).reset_index(drop=True)
    return z


def plot_manuscript_reliability_degradation(
    horizon_perf: pd.DataFrame,
    sev_perf: pd.DataFrame,
    domain_perf: pd.DataFrame,
    outdir: Path,
) -> None:
    """Figure M1: grouped RMSE comparison across distinct evaluation states."""
    overall = horizon_perf[
        (horizon_perf["horizon_h"] == 2)
        & (horizon_perf["split"] == "test")
        & (horizon_perf["model"].isin(["GradientBoosting", "RandomForest"]))
    ][["model", "rmse"]].copy()
    overall["state"] = "Overall"

    sev = sev_perf[sev_perf["subset"].isin(["warning_or_higher", "severe_or_higher"])][
        ["model", "subset", "rmse"]
    ].copy()
    sev["state"] = sev["subset"].map({
        "warning_or_higher": "Warning+",
        "severe_or_higher": "Severe+",
    })

    dom = domain_perf[["model", "domain", "rmse"]].copy()
    dom["state"] = dom["domain"].map({
        "in_domain": "In-domain",
        "out_of_domain": "OOD",
    })

    plot_df = pd.concat([
        overall[["model", "state", "rmse"]],
        sev[["model", "state", "rmse"]],
        dom[["model", "state", "rmse"]],
    ], ignore_index=True)
    order = ["Overall", "Warning+", "Severe+", "In-domain", "OOD"]
    model_order = ["GradientBoosting", "RandomForest"]
    labels = {"GradientBoosting": "Gradient boosting", "RandomForest": "Random forest"}

    x = np.arange(len(order))
    width = 0.36
    plt.figure(figsize=(8.4, 5.0))
    for j, model_name in enumerate(model_order):
        sub = plot_df[plot_df["model"] == model_name].set_index("state").reindex(order)
        offset = (j - 0.5) * width
        plt.bar(x + offset, sub["rmse"].to_numpy(float), width=width, label=labels[model_name])
    plt.xticks(x, order)
    plt.xlabel("Evaluation state")
    plt.ylabel("RMSE of future 2-h cumulative excess CO$_2$ (ppm h)")
    plt.title("Prediction Reliability Across Severity and Support States")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "manuscript_fig_1_reliability_degradation.png", dpi=600)
    plt.close()


def plot_manuscript_risk_coverage(selective: pd.DataFrame, outdir: Path) -> None:
    """Figure M2: selective-prediction risk--coverage trade-off."""
    x = selective[selective["gate"].str.startswith("support_rank_")].copy()
    if x.empty:
        return
    plt.figure(figsize=(8.0, 5.0))
    for model_name, sub in x.groupby("model"):
        sub = sub.sort_values("actual_coverage")
        label = "Gradient boosting" if model_name == "GradientBoosting" else "Random forest"
        plt.plot(100.0 * sub["actual_coverage"], sub["rmse"], marker="o", linewidth=1.5, label=label)
    plt.xlabel("Prediction coverage retained (%)")
    plt.ylabel("RMSE of retained predictions (ppm h)")
    plt.title("Reliability-Coverage Trade-off Under Selective Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "manuscript_fig_2_risk_coverage.png", dpi=600)
    plt.close()


def plot_manuscript_supervisory_policy(policy: pd.DataFrame, outdir: Path) -> None:
    """Figure M3: combined GB/RF severe-state disposition under three policies."""
    if policy.empty:
        return
    policy_order = ["forecast_only", "measured_plus_forecast", "reliability_aware"]
    model_order = ["GradientBoosting", "RandomForest"]
    rows = []
    for model_name in model_order:
        sub = policy[policy["model"] == model_name].set_index("policy")
        for p in policy_order:
            if p not in sub.index:
                continue
            r = sub.loc[p]
            rows.append({
                "model": model_name,
                "policy": p,
                "escalate": 100.0 * float(r["future_severe_escalate_rate"]),
                "defer": 100.0 * float(r["future_severe_defer_rate"]),
                "normal": 100.0 * float(r["future_severe_normal_rate"]),
            })
    d = pd.DataFrame(rows)
    if d.empty:
        return
    labels = []
    for _, r in d.iterrows():
        m = "GB" if r["model"] == "GradientBoosting" else "RF"
        p = {
            "forecast_only": "Forecast",
            "measured_plus_forecast": "Measured+",
            "reliability_aware": "Reliability",
        }[r["policy"]]
        labels.append(f"{m}\n{p}")

    x = np.arange(len(d))
    esc = d["escalate"].to_numpy(float)
    dfr = d["defer"].to_numpy(float)
    nor = d["normal"].to_numpy(float)
    plt.figure(figsize=(9.0, 5.0))
    plt.bar(x, esc, label="Escalate")
    plt.bar(x, dfr, bottom=esc, label="Defer")
    plt.bar(x, nor, bottom=esc + dfr, label="Normal")
    plt.xticks(x, labels)
    plt.ylabel("Disposition of observed future severe states (%)")
    plt.ylim(0, 100)
    plt.title("Supervisory Policies for Future Severe Exposure")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(outdir / "manuscript_fig_3_supervisory_policy.png", dpi=600)
    plt.close()

# -----------------------------------------------------------------------------
# 10. Main analysis
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full Indoor Air analysis for Indoor_Air_2026.csv")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV")
    parser.add_argument("--outdir", type=Path, default=Path("Indoor_Air_2026_outputs"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rawdir = args.outdir / "raw_outputs"
    rawdir.mkdir(parents=True, exist_ok=True)

    df, qc = load_and_prepare(args.input, CFG)
    save_csv(qc, rawdir / "table_00_data_qc.csv")

    fsets = feature_sets(df)
    primary_features = fsets["primary_schedule"]
    splits = chronological_day_split(df, CFG)

    split_rows = []
    for name, dates in splits.items():
        split_rows.append({
            "split": name,
            "n_days": len(dates),
            "start_date": pd.Timestamp(dates[0]).date().isoformat(),
            "end_date": pd.Timestamp(dates[-1]).date().isoformat(),
        })
    save_csv(pd.DataFrame(split_rows), rawdir / "table_01_chronological_split.csv")

    # Descriptive analyses.
    daily = build_daily_summary(df)
    save_csv(daily, rawdir / "table_02_daily_summary.csv")

    point_corr, daily_corr = correlation_tables(df, daily)
    save_csv(point_corr, rawdir / "table_03_point_spearman_correlations.csv")
    save_csv(daily_corr, rawdir / "table_04_daily_spearman_correlations.csv")

    exp_summary = exposure_metric_summary(df)
    save_csv(exp_summary, rawdir / "table_05_absolute_vs_excess_summary.csv")

    zero_events = daily[daily["zero_departure_day"] == 1].copy()
    save_csv(zero_events, rawdir / "table_06_zero_departure_event_summary.csv")

    disruption = daily[daily["daily_service_disruption_flag"] == 1].copy()
    save_csv(disruption, rawdir / "table_07_disruption_day_summary.csv")

    # Exposure thresholds from training only.
    thresholds = training_thresholds(df, splits, primary_features, CFG)
    threshold_table = pd.DataFrame([
        {"state": "warning", "quantile": CFG.warning_q, "threshold_ppm_h": thresholds["warning"]},
        {"state": "severe", "quantile": CFG.severe_q, "threshold_ppm_h": thresholds["severe"]},
        {"state": "critical", "quantile": CFG.critical_q, "threshold_ppm_h": thresholds["critical"]},
    ])
    save_csv(threshold_table, rawdir / "table_08_training_only_exposure_thresholds.csv")

    # Primary prediction across horizons.
    horizon_perf, pred_tables = evaluate_horizons(df, splits, primary_features, CFG)
    save_csv(horizon_perf, rawdir / "table_09_model_performance_by_horizon.csv")
    for (h, endpoint), pred in pred_tables.items():
        save_csv(pred, rawdir / f"predictions_{endpoint}_{h}h_test.csv")

    pred2 = pred_tables[(2, "excess")]
    sev_perf = severity_performance(pred2, "future_exc_2h_recalc", thresholds)
    save_csv(sev_perf, rawdir / "table_10_severity_dependent_performance.csv")

    # Feature-set sensitivity to document value/limitations of passenger reconstruction.
    fs_comp = compare_feature_sets_2h(df, splits, fsets, CFG)
    save_csv(fs_comp, rawdir / "table_11_feature_set_comparison_2h.csv")

    # Secondary absolute endpoint.
    abs_perf, abs_pred = absolute_endpoint_robustness(df, splits, primary_features, CFG)
    save_csv(abs_perf, rawdir / "table_12_absolute_endpoint_robustness.csv")
    save_csv(abs_pred, rawdir / "predictions_absolute_2h_test.csv")

    # Applicability-domain diagnostics.
    domain_perf, domain_detail = domain_performance(df, splits, primary_features, CFG)
    save_csv(domain_perf, rawdir / "table_13_applicability_domain_performance.csv")
    save_csv(domain_detail, rawdir / "applicability_domain_test_detail.csv")

    # Reliability-aware severe-event detection and selective prediction.
    severe_detection = future_severe_detection(domain_detail, thresholds)
    save_csv(severe_detection, rawdir / "table_16_future_severe_detection.csv")

    selective = selective_prediction_risk_coverage(domain_detail, thresholds, CFG)
    save_csv(selective, rawdir / "table_17_selective_prediction_risk_coverage.csv")

    # Supervisory decision policies: normal / escalate / defer.
    policy, policy_detail = supervisory_policy_analysis(domain_detail, thresholds)
    save_csv(policy, rawdir / "table_18_supervisory_policy_comparison.csv")
    save_csv(policy_detail, rawdir / "supervisory_decision_test_detail.csv")

    # Robustness checks requested for the supervisory interpretation:
    # (i) collapse overlapping severe records to day/episode units; and
    # (ii) vary the applicability-domain definition over a pre-specified grid.
    day_event_robustness, day_event_detail = severe_day_event_robustness(policy_detail, CFG)
    save_csv(day_event_robustness, rawdir / "table_19_severe_day_event_robustness.csv")
    save_csv(day_event_detail, rawdir / "severe_day_event_robustness_detail.csv")

    ad_sensitivity = applicability_domain_sensitivity_analysis(
        df, splits, primary_features, domain_detail, thresholds, CFG
    )
    save_csv(ad_sensitivity, rawdir / "table_20_applicability_domain_sensitivity.csv")

    # Routine-to-severe holdout tests.
    holdout_1 = range_holdout_test(df, splits, primary_features, "hist_exc_2h_recalc", CFG)
    holdout_2 = range_holdout_test(df, splits, primary_features, "indoor_co2_ppm", CFG)
    holdout = pd.concat([holdout_1, holdout_2], ignore_index=True)
    save_csv(holdout, rawdir / "table_14_routine_to_severe_holdout.csv")

    # Ventilation sensitivity analysis.
    vent, vent_profiles = ventilation_sensitivity_analysis(df, thresholds, CFG)
    save_csv(vent, rawdir / "table_15_ventilation_sensitivity.csv")
    save_csv(vent_profiles, rawdir / "ventilation_sensitivity_profiles.csv")

    # Add analysis-state columns and save a clean analysis-ready table.
    df["hist_exc_2h_state"] = severity_label(df["hist_exc_2h_recalc"].fillna(-np.inf), thresholds)
    save_csv(df, rawdir / "analysis_ready_recomputed.csv")

    # Manuscript-ready compact outputs: exactly 3 tables + 3 figures.
    manuscript_table_1 = build_manuscript_table_1(horizon_perf, sev_perf, domain_perf, holdout)
    manuscript_table_2 = build_manuscript_table_2(policy, day_event_robustness, ad_sensitivity)
    manuscript_table_3 = build_manuscript_table_3(vent)
    save_csv(manuscript_table_1, args.outdir / "manuscript_table_1_prediction_reliability.csv")
    save_csv(manuscript_table_2, args.outdir / "manuscript_table_2_supervisory_policy.csv")
    save_csv(manuscript_table_3, args.outdir / "manuscript_table_3_ventilation_sensitivity.csv")

    plot_manuscript_reliability_degradation(horizon_perf, sev_perf, domain_perf, args.outdir)
    plot_manuscript_risk_coverage(selective, args.outdir)
    plot_manuscript_supervisory_policy(policy, args.outdir)

    # Compact machine-readable manuscript summary.
    test2 = horizon_perf[(horizon_perf["horizon_h"] == 2) & (horizon_perf["split"] == "test")]
    summary = {
        "input_file": str(args.input),
        "n_rows": int(len(df)),
        "n_days": int(df["date"].nunique()),
        "date_start": df["date"].min().date().isoformat(),
        "date_end": df["date"].max().date().isoformat(),
        "co2": {
            "indoor_mean_ppm": float(df["indoor_co2_ppm"].mean()),
            "indoor_max_ppm": float(df["indoor_co2_ppm"].max()),
            "outdoor_mean_ppm": float(df["outdoor_co2_ppm"].mean()),
            "excess_mean_ppm": float(df["excess_co2_ppm_recalc"].mean()),
        },
        "training_thresholds_ppm_h": thresholds,
        "zero_departure_dates": [pd.Timestamp(x).date().isoformat() for x in zero_events["date"]],
        "primary_2h_test_metrics": test2[["model", "r2", "rmse", "mae", "bias", "n"]].to_dict(orient="records"),
        "feature_set_note": (
            "Primary prediction excludes daily passenger totals and reconstructed 30/60-min passenger estimates; "
            "those are evaluated only as retrospective sensitivity features to avoid same-day information leakage."
        ),
        "reliability_aware_supervisory_policy": policy.to_dict(orient="records"),
        "future_severe_detection": severe_detection.to_dict(orient="records"),
        "severe_day_event_robustness": day_event_robustness.to_dict(orient="records"),
        "applicability_domain_sensitivity": ad_sensitivity.to_dict(orient="records"),
        "manuscript_outputs": {
            "tables": [
                "manuscript_table_1_prediction_reliability.csv",
                "manuscript_table_2_supervisory_policy.csv",
                "manuscript_table_3_ventilation_sensitivity.csv",
            ],
            "figures": [
                "manuscript_fig_1_reliability_degradation.png",
                "manuscript_fig_2_risk_coverage.png",
                "manuscript_fig_3_supervisory_policy.png",
            ],
            "raw_outputs_directory": "raw_outputs",
        },
    }
    with open(args.outdir / "manuscript_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print("Indoor Air full analysis completed")
    print("=" * 78)
    print(f"Input  : {args.input}")
    print(f"Output : {args.outdir.resolve()}")
    print(f"Raw    : {rawdir.resolve()}")
    print("Manuscript outputs: 3 tables + 3 figures (600 dpi)")
    print(f"Rows   : {len(df):,}")
    print(f"Days   : {df['date'].nunique()}")
    print("Training-only 2-h exposure thresholds:")
    for k, v in thresholds.items():
        print(f"  {k:8s}: {v:8.2f} ppm h")
    print("\nPrimary 2-h chronological test performance:")
    print(test2[["model", "r2", "rmse", "mae", "bias", "n"]].round(4).to_string(index=False))
    print("\nReliability-aware supervisory policy comparison:")
    cols = [
        "model", "policy", "future_severe_n", "future_severe_escalate_rate",
        "future_severe_defer_rate", "future_severe_normal_rate",
        "nonsevere_escalate_rate", "nonsevere_defer_rate",
    ]
    print(policy[cols].round(4).to_string(index=False))

    print("\nSevere day/episode robustness (first severe record):")
    rcols = [
        "scope", "model", "policy", "n_units", "initially_protected_n",
        "fully_missed_n",
    ]
    print(day_event_robustness[rcols].to_string(index=False))

    print("\nApplicability-domain sensitivity for reliability-aware supervision:")
    for model_name in ["GradientBoosting", "RandomForest"]:
        z = ad_sensitivity[ad_sensitivity["model"] == model_name]
        if len(z):
            print(
                f"  {model_name}: severe protected "
                f"{100*z['severe_protected_rate'].min():.1f}--{100*z['severe_protected_rate'].max():.1f}% across AD grid; "
                f"severe-day initial protection "
                f"{100*z['severe_day_initially_protected_rate'].min():.1f}--{100*z['severe_day_initially_protected_rate'].max():.1f}%"
            )

    print("\nIMPORTANT: update manuscript numeric results from these outputs; do not reuse older values blindly.")


if __name__ == "__main__":
    main()
