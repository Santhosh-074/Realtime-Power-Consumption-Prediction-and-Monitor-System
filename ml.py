"""Machine Learning module for RTPCPMS.

Covers SRS F-006, F-008..F-013:
  validate -> clean (missing values) -> feature extraction -> normalization
  -> train/test split -> train -> evaluate (MAE, RMSE, R2) -> forecast.
"""
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

ALGORITHMS = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "ridge": "Ridge Regression (linear)",
}

NA_VALUES = ["?", "", "NA", "N/A", "null", "NULL", "nan", "NaN"]
TIME_HINTS = ("timestamp", "datetime", "date_time", "time", "date")
VALUE_HINTS = ("global_active_power", "active_power", "consumption", "power",
               "energy", "usage", "load", "kwh", "kw")

# lag steps and rolling window depend on how often readings are recorded
LAGS = {"h": [1, 2, 3, 24], "D": [1, 2, 7]}
ROLL = {"h": 24, "D": 7}
MIN_POINTS = {"h": 240, "D": 60}
DEFAULT_HORIZON = {"h": 24, "D": 7}
MAX_HORIZON = {"h": 168, "D": 60}


class DatasetError(Exception):
    """Raised when an uploaded dataset cannot be used."""


# --------------------------------------------------------------------------
# Validation and cleaning (F-006, F-008)
# --------------------------------------------------------------------------
def _sniff_sep(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    counts = {s: first.count(s) for s in [",", ";", "\t", "|"]}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        raise DatasetError("Could not detect columns. Please upload a CSV with a header row.")
    return sep


def _pick_columns(columns):
    """Return (time_spec, value_col). time_spec is a column name or ('Date','Time')."""
    lower = {c: c.strip().lower() for c in columns}
    by_lower = {v: k for k, v in lower.items()}

    if "date" in by_lower and "time" in by_lower:
        time_spec = (by_lower["date"], by_lower["time"])
        used = set(time_spec)
    else:
        time_col = None
        for hint in TIME_HINTS:
            for c, lc in lower.items():
                if hint in lc:
                    time_col = c
                    break
            if time_col:
                break
        if time_col is None:
            time_col = list(columns)[0]
        time_spec = time_col
        used = {time_col}

    value_col = None
    candidates = [c for c in columns if c not in used]
    for hint in VALUE_HINTS:
        for c in candidates:
            if hint in lower[c]:
                value_col = c
                break
        if value_col:
            break
    if value_col is None and candidates:
        value_col = candidates[0]
    if value_col is None:
        raise DatasetError("The file needs a time column and a power consumption column.")
    return time_spec, value_col


def _parse_time(series):
    """Parse timestamps, deciding between day-first (16/12/2006) and month-first (12/16/2006).

    Both readings can be valid for dates like 3/4/2007, so candidates are ranked by:
    most rows parsed, then fewest backwards jumps in file order, then shortest time span.
    """
    best, best_score = None, None
    for dayfirst in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
        valid = parsed.dropna()
        if valid.empty:
            continue
        backwards = int((valid.diff() < pd.Timedelta(0)).sum())
        span = (valid.max() - valid.min()).total_seconds()
        score = (-len(valid), backwards, span)
        if best_score is None or score < best_score:
            best, best_score = parsed, score
    if best is None:
        raise DatasetError("No readable dates or times were found in the time column.")
    return best


def load_and_validate(path, enforce_min=True):
    """Read a CSV and return (clean_series, report, freq).

    clean_series: pandas Series indexed by regular timestamps (hourly or daily).
    report: dict describing what was found and fixed (shown to the user).
    enforce_min: when False the training-size minimum is skipped, used when
    appending a small batch to a dataset that is already big enough.
    """
    sep = _sniff_sep(path)
    try:
        header = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig")
    except Exception as exc:
        raise DatasetError(f"The file could not be read as CSV: {exc}")
    columns = [str(c) for c in header.columns]
    if len(columns) < 2:
        raise DatasetError("The file needs at least two columns (time and power).")

    time_spec, value_col = _pick_columns(columns)
    time_cols = list(time_spec) if isinstance(time_spec, tuple) else [time_spec]
    try:
        raw = pd.read_csv(path, sep=sep, usecols=time_cols + [value_col], dtype=str,
                          na_values=NA_VALUES, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:
        raise DatasetError(f"The file could not be read: {exc}")

    total_rows = len(raw)
    if total_rows == 0:
        raise DatasetError("The file has no data rows.")

    if isinstance(time_spec, tuple):
        time_text = raw[time_spec[0]].astype(str) + " " + raw[time_spec[1]].astype(str)
        time_label = f"{time_spec[0]} + {time_spec[1]}"
    else:
        time_text = raw[time_spec]
        time_label = time_spec

    ts = _parse_time(time_text)
    val = pd.to_numeric(raw[value_col], errors="coerce")

    invalid_ts = int(ts.isna().sum())
    if invalid_ts > 0.5 * total_rows:
        raise DatasetError(f"Column '{time_label}' does not look like dates/times.")
    if val.notna().sum() < 0.5 * total_rows:
        raise DatasetError(f"Column '{value_col}' does not look like numeric power values.")

    missing = int(val.isna().sum())
    df = pd.DataFrame({"ts": ts, "value": val}).dropna(subset=["ts"])
    duplicates = int(df.duplicated(subset=["ts"]).sum())
    negatives = int((df["value"] < 0).sum())
    df.loc[df["value"] < 0, "value"] = np.nan  # negative power is invalid
    df = df.drop_duplicates(subset=["ts"], keep="first").sort_values("ts").set_index("ts")

    if len(df) < 20:
        raise DatasetError("Too few valid rows after validation.")

    step = df.index.to_series().diff().median()
    freq = "h" if step < pd.Timedelta(days=1) else "D"
    series = df["value"].resample(freq).mean()
    gaps_filled = int(series.isna().sum())
    series = series.interpolate(limit_direction="both").astype(float)
    series.name = "value"

    if enforce_min and len(series) < MIN_POINTS[freq]:
        unit = "hourly" if freq == "h" else "daily"
        raise DatasetError(
            f"Not enough history: {len(series)} {unit} points found, "
            f"at least {MIN_POINTS[freq]} are needed to train a model.")

    report = {
        "time_column": time_label,
        "value_column": value_col,
        "frequency": "Hourly" if freq == "h" else "Daily",
        "total_rows": total_rows,
        "invalid_timestamps": invalid_ts,
        "missing_or_non_numeric_values": missing,
        "duplicate_timestamps": duplicates,
        "negative_values": negatives,
        "gaps_filled_by_interpolation": gaps_filled,
        "rows_after_cleaning": int(len(series)),
        "start": str(series.index[0]),
        "end": str(series.index[-1]),
    }
    return series, report, freq


# --------------------------------------------------------------------------
# Feature extraction (F-009) and normalization (F-010, inside the pipeline)
# --------------------------------------------------------------------------
def _time_features(index):
    return pd.DataFrame({
        "hour": index.hour,
        "dayofweek": index.dayofweek,
        "day": index.day,
        "month": index.month,
    }, index=index)


def build_frame(series, freq):
    feats = _time_features(series.index)
    for lag in LAGS[freq]:
        feats[f"lag_{lag}"] = series.shift(lag)
    feats["roll_mean"] = series.shift(1).rolling(ROLL[freq]).mean()
    feats["y"] = series
    return feats.dropna()


def _make_estimator(name):
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42)
    if name == "ridge":
        return Ridge(alpha=1.0)
    raise DatasetError("Unknown algorithm.")


# --------------------------------------------------------------------------
# Training and evaluation (F-011, F-013)
# --------------------------------------------------------------------------
def train_model(series, freq, algorithm, model_path):
    frame = build_frame(series, freq)
    if len(frame) < 50:
        raise DatasetError("Not enough rows to train a model.")
    X = frame.drop(columns="y")
    y = frame["y"]

    split = int(len(frame) * 0.8)  # chronological split: test data is the most recent 20%
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipe = Pipeline([("scale", MinMaxScaler()), ("model", _make_estimator(algorithm))])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    baseline_mae = float(mean_absolute_error(y_test, X_test["lag_1"]))  # "same as last step"

    tail = min(len(y_test), 24 * 7 if freq == "h" else 60)
    eval_data = {
        "labels": [str(t) for t in y_test.index[-tail:]],
        "actual": [round(float(v), 4) for v in y_test.values[-tail:]],
        "predicted": [round(float(v), 4) for v in pred[-tail:]],
    }

    # Final model uses every available row so forecasts start from the freshest data
    pipe.fit(X, y)
    joblib.dump({"pipeline": pipe, "freq": freq, "columns": list(X.columns)}, model_path)

    return {
        "mae": mae, "rmse": rmse, "r2": r2, "baseline_mae": baseline_mae,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "eval": eval_data,
    }


# --------------------------------------------------------------------------
# Forecast (F-012)
# --------------------------------------------------------------------------
def forecast(model_path, history, steps):
    """Predict `steps` future points after the last timestamp of `history`.

    history: pandas Series with a regular index (same frequency the model was trained on).
    Returns a list of (timestamp, value).
    """
    bundle = joblib.load(model_path)
    pipe, freq, columns = bundle["pipeline"], bundle["freq"], bundle["columns"]
    lags, window = LAGS[freq], ROLL[freq]
    step = pd.Timedelta(hours=1) if freq == "h" else pd.Timedelta(days=1)

    buf = list(history.values[-(max(max(lags), window) + 5):])
    if len(buf) < max(max(lags), window):
        raise DatasetError("Not enough history to forecast.")

    ts = history.index[-1]
    out = []
    for _ in range(steps):
        ts = ts + step
        row = {"hour": ts.hour, "dayofweek": ts.dayofweek, "day": ts.day, "month": ts.month}
        for lag in lags:
            row[f"lag_{lag}"] = buf[-lag]
        row["roll_mean"] = float(np.mean(buf[-window:]))
        X = pd.DataFrame([row])[columns]
        value = max(0.0, float(pipe.predict(X)[0]))
        buf.append(value)
        out.append((ts.to_pydatetime(), value))
    return out
