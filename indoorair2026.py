import argparse
import json
import math
import warnings
from dataclasses import dataclass
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

    # Routine-to-severe holdout.
    holdout_q: float = 0.75

    # Ventilation counterfactual.
    ach_base: float = 0.4
    ach_scenarios: Tuple[float, ...] = (0.4, 0.6, 0.8)


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
    detail = test[["timestamp", "date", target]].copy()
    detail = detail.join(ad)

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
# 7. Ventilation counterfactual
# -----------------------------------------------------------------------------

def simulate_ventilation_event(
    event: pd.DataFrame,
    trigger_idx: int,
    ach: float,
    cfg: Config = CFG,
) -> np.ndarray:
    """Euler counterfactual using effective source inferred under baseline ACH.

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


def ventilation_counterfactuals(
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
# 8. Figures
# -----------------------------------------------------------------------------

def plot_daily_exposure(daily: pd.DataFrame, thresholds: Dict[str, float], outdir: Path) -> None:
    plt.figure(figsize=(11, 4.8))
    plt.plot(pd.to_datetime(daily["date"]), daily["max_hist_exc_2h_ppm_h"], linewidth=1.2)
    for key in ["warning", "severe", "critical"]:
        plt.axhline(thresholds[key], linestyle="--", linewidth=1.0, label=f"{key}: {thresholds[key]:.1f}")
    plt.xlabel("Date")
    plt.ylabel("Daily maximum rolling 2-h excess CO$_2$ (ppm h)")
    plt.title("Daily Maximum 2-h Cumulative Excess CO$_2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig1_daily_2h_exposure.png", dpi=300)
    plt.close()


def plot_abs_vs_excess(df: pd.DataFrame, outdir: Path) -> None:
    x = df[["hist_abs_2h_recalc", "hist_exc_2h_recalc"]].dropna()
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(x["hist_abs_2h_recalc"], x["hist_exc_2h_recalc"], s=12, alpha=0.45)
    plt.xlabel("Historical 2-h absolute indoor CO$_2$ exposure (ppm h)")
    plt.ylabel("Historical 2-h excess CO$_2$ exposure (ppm h)")
    plt.title("Absolute vs Excess Cumulative CO$_2$ Exposure")
    plt.tight_layout()
    plt.savefig(outdir / "fig_abs_vs_excess_2h.png", dpi=300)
    plt.close()


def plot_prediction(pred: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(11, 5.0))
    plt.plot(pred["timestamp"], pred["future_exc_2h_recalc"], label="Observed", linewidth=1.2)
    if "GradientBoosting" in pred:
        plt.plot(pred["timestamp"], pred["GradientBoosting"], label="Gradient boosting", linewidth=1.0)
    if "RandomForest" in pred:
        plt.plot(pred["timestamp"], pred["RandomForest"], label="Random forest", linewidth=1.0)
    plt.xlabel("Timestamp")
    plt.ylabel("Future 2-h excess CO$_2$ exposure (ppm h)")
    plt.title("Observed and Predicted 2-h Cumulative Excess CO$_2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig2_test_observed_predicted.png", dpi=300)
    plt.close()


def plot_severity_rmse(sev: pd.DataFrame, outdir: Path) -> None:
    order = ["all", "warning_or_higher", "severe_or_higher", "critical_or_higher"]
    plt.figure(figsize=(7.5, 4.8))
    for model in sev["model"].drop_duplicates():
        z = sev[sev["model"] == model].set_index("subset").reindex(order)
        plt.plot(order, z["rmse"], marker="o", label=model)
    plt.xlabel("Exposure subset")
    plt.ylabel("RMSE (ppm h)")
    plt.title("Severity-Dependent Prediction Error")
    plt.xticks(rotation=18)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig3_severity_rmse.png", dpi=300)
    plt.close()


def plot_domain(detail: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(7.0, 5.5))
    mask = detail["in_domain"].astype(bool)
    plt.scatter(detail.loc[mask, "future_exc_2h_recalc"], detail.loc[mask, "pred_RandomForest"], s=16, alpha=0.5, label="In-domain")
    plt.scatter(detail.loc[~mask, "future_exc_2h_recalc"], detail.loc[~mask, "pred_RandomForest"], s=20, alpha=0.7, label="OOD")
    lo = float(min(detail["future_exc_2h_recalc"].min(), detail["pred_RandomForest"].min()))
    hi = float(max(detail["future_exc_2h_recalc"].max(), detail["pred_RandomForest"].max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    plt.xlabel("Observed future 2-h excess exposure (ppm h)")
    plt.ylabel("RF prediction (ppm h)")
    plt.title("Applicability Domain: In-Domain vs OOD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig4_domain_observed_predicted.png", dpi=300)
    plt.close()


def plot_event_profiles(df: pd.DataFrame, outdir: Path) -> None:
    zero_dates = (
        df.groupby("date")["daily_departure_passengers"].first()
        .loc[lambda s: s == 0]
        .index
        .tolist()
    )
    for d in zero_dates:
        x = df[df["date"] == d].sort_values("timestamp")
        plt.figure(figsize=(8.0, 4.8))
        plt.plot(x["timestamp"], x["indoor_co2_ppm"], marker="o", label="Indoor CO$_2$")
        plt.plot(x["timestamp"], x["outdoor_co2_ppm"], marker="o", label="Outdoor CO$_2$")
        plt.xlabel("Time")
        plt.ylabel("CO$_2$ (ppm)")
        plt.title(f"Zero-Departure Day CO$_2$ Profile: {pd.Timestamp(d).date()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"fig_event_{pd.Timestamp(d).date()}.png", dpi=300)
        plt.close()


def plot_ventilation_profiles(profiles: pd.DataFrame, outdir: Path) -> None:
    if profiles.empty:
        return
    for d, sub in profiles.groupby("date"):
        # Prioritize disruption or zero-departure days for manuscript-style figures.
        first = sub.iloc[0]
        date_str = pd.Timestamp(d).date().isoformat()
        plt.figure(figsize=(8.3, 4.8))
        obs = sub[sub["ach_h_minus1"] == CFG.ach_base].drop_duplicates("timestamp")
        plt.plot(obs["timestamp"], obs["observed_indoor_co2_ppm"], label="Observed / reference")
        for ach in sorted(sub["ach_h_minus1"].unique()):
            if np.isclose(ach, CFG.ach_base):
                continue
            z = sub[sub["ach_h_minus1"] == ach]
            plt.plot(z["timestamp"], z["simulated_indoor_co2_ppm"], label=f"ACH={ach:.1f} h$^{{-1}}$")
        trigger = pd.to_datetime(first["trigger_time"])
        plt.axvline(trigger, linestyle="--", linewidth=1.0, label="Severe-state trigger")
        plt.xlabel("Time")
        plt.ylabel("Indoor CO$_2$ (ppm)")
        plt.title(f"Ventilation Counterfactual: {date_str}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"fig_ventilation_{date_str}.png", dpi=300)
        plt.close()


# -----------------------------------------------------------------------------
# 9. Main analysis
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full Indoor Air analysis for Indoor_Air_2026.csv")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV")
    parser.add_argument("--outdir", type=Path, default=Path("Indoor_Air_2026_outputs"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df, qc = load_and_prepare(args.input, CFG)
    save_csv(qc, args.outdir / "table_00_data_qc.csv")

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
    save_csv(pd.DataFrame(split_rows), args.outdir / "table_01_chronological_split.csv")

    # Descriptive analyses.
    daily = build_daily_summary(df)
    save_csv(daily, args.outdir / "table_02_daily_summary.csv")

    point_corr, daily_corr = correlation_tables(df, daily)
    save_csv(point_corr, args.outdir / "table_03_point_spearman_correlations.csv")
    save_csv(daily_corr, args.outdir / "table_04_daily_spearman_correlations.csv")

    exp_summary = exposure_metric_summary(df)
    save_csv(exp_summary, args.outdir / "table_05_absolute_vs_excess_summary.csv")

    zero_events = daily[daily["zero_departure_day"] == 1].copy()
    save_csv(zero_events, args.outdir / "table_06_zero_departure_event_summary.csv")

    disruption = daily[daily["daily_service_disruption_flag"] == 1].copy()
    save_csv(disruption, args.outdir / "table_07_disruption_day_summary.csv")

    # Exposure thresholds from training only.
    thresholds = training_thresholds(df, splits, primary_features, CFG)
    threshold_table = pd.DataFrame([
        {"state": "warning", "quantile": CFG.warning_q, "threshold_ppm_h": thresholds["warning"]},
        {"state": "severe", "quantile": CFG.severe_q, "threshold_ppm_h": thresholds["severe"]},
        {"state": "critical", "quantile": CFG.critical_q, "threshold_ppm_h": thresholds["critical"]},
    ])
    save_csv(threshold_table, args.outdir / "table_08_training_only_exposure_thresholds.csv")

    # Primary prediction across horizons.
    horizon_perf, pred_tables = evaluate_horizons(df, splits, primary_features, CFG)
    save_csv(horizon_perf, args.outdir / "table_09_model_performance_by_horizon.csv")
    for (h, endpoint), pred in pred_tables.items():
        save_csv(pred, args.outdir / f"predictions_{endpoint}_{h}h_test.csv")

    pred2 = pred_tables[(2, "excess")]
    sev_perf = severity_performance(pred2, "future_exc_2h_recalc", thresholds)
    save_csv(sev_perf, args.outdir / "table_10_severity_dependent_performance.csv")

    # Feature-set sensitivity to document value/limitations of passenger reconstruction.
    fs_comp = compare_feature_sets_2h(df, splits, fsets, CFG)
    save_csv(fs_comp, args.outdir / "table_11_feature_set_comparison_2h.csv")

    # Secondary absolute endpoint.
    abs_perf, abs_pred = absolute_endpoint_robustness(df, splits, primary_features, CFG)
    save_csv(abs_perf, args.outdir / "table_12_absolute_endpoint_robustness.csv")
    save_csv(abs_pred, args.outdir / "predictions_absolute_2h_test.csv")

    # Applicability-domain diagnostics.
    domain_perf, domain_detail = domain_performance(df, splits, primary_features, CFG)
    save_csv(domain_perf, args.outdir / "table_13_applicability_domain_performance.csv")
    save_csv(domain_detail, args.outdir / "applicability_domain_test_detail.csv")

    # Routine-to-severe holdout tests.
    holdout_1 = range_holdout_test(df, splits, primary_features, "hist_exc_2h_recalc", CFG)
    holdout_2 = range_holdout_test(df, splits, primary_features, "indoor_co2_ppm", CFG)
    holdout = pd.concat([holdout_1, holdout_2], ignore_index=True)
    save_csv(holdout, args.outdir / "table_14_routine_to_severe_holdout.csv")

    # Ventilation counterfactual.
    vent, vent_profiles = ventilation_counterfactuals(df, thresholds, CFG)
    save_csv(vent, args.outdir / "table_15_ventilation_counterfactual.csv")
    save_csv(vent_profiles, args.outdir / "ventilation_counterfactual_profiles.csv")

    # Add analysis-state columns and save a clean analysis-ready table.
    df["hist_exc_2h_state"] = severity_label(df["hist_exc_2h_recalc"].fillna(-np.inf), thresholds)
    save_csv(df, args.outdir / "analysis_ready_recomputed.csv")

    # Figures.
    plot_daily_exposure(daily, thresholds, args.outdir)
    plot_abs_vs_excess(df, args.outdir)
    plot_prediction(pred2, args.outdir)
    plot_severity_rmse(sev_perf, args.outdir)
    plot_domain(domain_detail, args.outdir)
    plot_event_profiles(df, args.outdir)
    plot_ventilation_profiles(vent_profiles, args.outdir)

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
    }
    with open(args.outdir / "manuscript_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print("Indoor Air full analysis completed")
    print("=" * 78)
    print(f"Input  : {args.input}")
    print(f"Output : {args.outdir.resolve()}")
    print(f"Rows   : {len(df):,}")
    print(f"Days   : {df['date'].nunique()}")
    print("Training-only 2-h exposure thresholds:")
    for k, v in thresholds.items():
        print(f"  {k:8s}: {v:8.2f} ppm h")
    print("\nPrimary 2-h chronological test performance:")
    print(test2[["model", "r2", "rmse", "mae", "bias", "n"]].round(4).to_string(index=False))
    print("\nIMPORTANT: update manuscript numeric results from these outputs; do not reuse older values blindly.")


if __name__ == "__main__":
    main()
