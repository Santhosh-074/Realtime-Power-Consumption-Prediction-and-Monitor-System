"""Dataset and prediction services used by the Flask routes."""
import json
import os
import uuid
from datetime import datetime

import pandas as pd

import ml
from models import Alert, Dataset, ModelRecord, Prediction, Reading, db

def forecast_cost_summary(dataset, tariff_per_kwh=7.0):
    """Calculate estimated energy consumption and cost from stored forecasts."""

    rows = (
        db.session.query(Prediction.ts, Prediction.value)
        .filter_by(dataset_id=dataset.id)
        .order_by(Prediction.ts)
        .all()
    )

    if not rows:
        return None

    # Daily readings represent average power over 24 hours.
    # Hourly readings represent average power over 1 hour.
    interval_hours = 24 if dataset.freq == "d" else 1

    forecast_energy_kwh = sum(
        max(0.0, float(value)) * interval_hours
        for _, value in rows
    )

    forecast_cost = forecast_energy_kwh * tariff_per_kwh

    start_time = rows[0][0]
    end_time = rows[-1][0]

    # Group predicted energy by calendar month.
    monthly_costs = {}

    for ts, value in rows:
        month_key = ts.strftime("%B %Y")

        energy_kwh = max(0.0, float(value)) * interval_hours

        if month_key not in monthly_costs:
            monthly_costs[month_key] = {
                "energy_kwh": 0.0,
                "cost": 0.0,
            }

        monthly_costs[month_key]["energy_kwh"] += energy_kwh
        monthly_costs[month_key]["cost"] += energy_kwh * tariff_per_kwh

    return {
        "forecast_start": start_time,
        "forecast_end": end_time,
        "forecast_hours": len(rows) if dataset.freq == "h" else None,
        "forecast_steps": len(rows),
        "forecast_energy_kwh": forecast_energy_kwh,
        "forecast_cost": forecast_cost,
        "forecast_tariff": tariff_per_kwh,
        "monthly_forecast_costs": monthly_costs,
    }



def _clear_derived(dataset_id):
    """Remove models, predictions and alerts when the underlying data changes."""
    for m in ModelRecord.query.filter_by(dataset_id=dataset_id).all():
        if m.path and os.path.exists(m.path):
            os.remove(m.path)
    ModelRecord.query.filter_by(dataset_id=dataset_id).delete()
    Prediction.query.filter_by(dataset_id=dataset_id).delete()
    Alert.query.filter_by(dataset_id=dataset_id).delete()


def get_series(dataset):
    rows = (db.session.query(Reading.ts, Reading.value)
            .filter_by(dataset_id=dataset.id).order_by(Reading.ts).all())
    s = pd.Series([r[1] for r in rows], index=pd.DatetimeIndex([r[0] for r in rows]), name="value")
    return s


def store_series(dataset, series, report, freq, sigma):
    """Replace the dataset's readings with `series` and refresh its statistics."""
    Reading.query.filter_by(dataset_id=dataset.id).delete()
    _clear_derived(dataset.id)
    records = [{"dataset_id": dataset.id, "ts": t.to_pydatetime(), "value": float(v)}
               for t, v in series.items()]
    for i in range(0, len(records), 5000):
        db.session.bulk_insert_mappings(Reading, records[i:i + 5000])

    dataset.freq = freq
    dataset.rows = len(series)
    dataset.start_ts = series.index[0].to_pydatetime()
    dataset.end_ts = series.index[-1].to_pydatetime()
    dataset.mean_value = float(series.mean())
    dataset.std_value = float(series.std() or 0.0)
    dataset.report_json = json.dumps(report)
    db.session.flush()
    check_current(dataset, series, sigma)


def check_current(dataset, series, sigma):
    """F-019: raise an alert when the latest reading exceeds the threshold."""
    thr = dataset.effective_threshold(sigma)
    Alert.query.filter_by(dataset_id=dataset.id, kind="current").delete()
    if thr is not None and len(series) and series.iloc[-1] > thr:
        db.session.add(Alert(
            dataset_id=dataset.id, kind="current", ts=series.index[-1].to_pydatetime(),
            value=float(series.iloc[-1]), threshold=float(thr),
            message=f"Current consumption {series.iloc[-1]:.2f} kW is above the threshold {thr:.2f} kW"))


def create_dataset(owner, name, filename, stored_path, sigma):
    series, report, freq = ml.load_and_validate(stored_path)
    ds = Dataset(owner_id=owner.id, name=name, filename=filename, stored_path=stored_path)
    db.session.add(ds)
    db.session.flush()
    store_series(ds, series, report, freq, sigma)
    db.session.commit()
    return ds, report


def update_dataset(dataset, path, mode, sigma):
    """F-007: replace the data or append new records to an existing dataset."""
    new_series, report, freq = ml.load_and_validate(path, enforce_min=(mode != "append"))
    if mode == "append":
        old = get_series(dataset)
        if freq != dataset.freq:
            raise ml.DatasetError("The new file has a different time frequency than the existing data.")
        merged = new_series.combine_first(old).sort_index()
        report["rows_after_cleaning"] = int(len(merged))
        report["start"], report["end"] = str(merged.index[0]), str(merged.index[-1])
        new_series = merged
    store_series(dataset, new_series, report, freq, sigma)
    db.session.commit()
    return report


def delete_dataset(dataset):
    Reading.query.filter_by(dataset_id=dataset.id).delete()
    _clear_derived(dataset.id)
    if dataset.stored_path and os.path.exists(dataset.stored_path):
        os.remove(dataset.stored_path)
    db.session.delete(dataset)
    db.session.commit()


def latest_model(dataset_id):
    return (ModelRecord.query.filter_by(dataset_id=dataset_id)
            .order_by(ModelRecord.trained_at.desc(), ModelRecord.id.desc()).first())


def train(dataset, algorithm, model_folder):
    series = get_series(dataset)
    path = os.path.join(model_folder, f"model_{dataset.id}_{uuid.uuid4().hex[:8]}.joblib")
    metrics = ml.train_model(series, dataset.freq, algorithm, path)
    rec = ModelRecord(
        dataset_id=dataset.id, algorithm=algorithm, mae=metrics["mae"], rmse=metrics["rmse"],
        r2=metrics["r2"], baseline_mae=metrics["baseline_mae"],
        train_rows=metrics["train_rows"], test_rows=metrics["test_rows"], path=path,
        eval_json=json.dumps(metrics["eval"]))
    db.session.add(rec)
    db.session.commit()
    return rec


def predict(dataset, steps, sigma):
    """Use the latest trained model (N-017) and store forecast + alerts."""
    model = latest_model(dataset.id)
    if model is None:
        raise ml.DatasetError("Train a model first.")
    steps = max(1, min(steps, ml.MAX_HORIZON[dataset.freq]))
    series = get_series(dataset)
    result = ml.forecast(model.path, series, steps)

    Prediction.query.filter_by(dataset_id=dataset.id).delete()
    Alert.query.filter_by(dataset_id=dataset.id, kind="forecast").delete()
    thr = dataset.effective_threshold(sigma)
    for ts, value in result:
        db.session.add(Prediction(dataset_id=dataset.id, model_id=model.id, ts=ts, value=value))
        if thr is not None and value > thr:
            db.session.add(Alert(
                dataset_id=dataset.id, kind="forecast", ts=ts, value=value, threshold=float(thr),
                message=f"Predicted consumption {value:.2f} kW is above the threshold {thr:.2f} kW"))
    db.session.commit()
    return result


def dataset_stats(dataset, sigma):
    """Calculate consumption statistics and estimated electricity costs."""

    tariff_per_kwh = 7.0

    thr = dataset.effective_threshold(sigma)

    # Get timestamps and power readings from the database
    rows = (
        db.session.query(Reading.ts, Reading.value)
        .filter_by(dataset_id=dataset.id)
        .order_by(Reading.ts)
        .all()
    )

    if not rows:
        return {}

    timestamps = [row[0] for row in rows]
    values = [float(row[1]) for row in rows]

    latest_ts = timestamps[-1]
    latest_value = values[-1]

    # Determine the interval represented by each reading.
    # Hourly average power: kW × 1 hour = kWh.
    # Daily average power: kW × 24 hours = kWh.
    interval_hours = 24 if dataset.freq == "d" else 1

    # Use the date range represented by the latest dataset reading.
    latest_date = latest_ts.date()
    latest_month = (latest_ts.year, latest_ts.month)
    latest_year = latest_ts.year

    daily_energy = sum(
        value * interval_hours
        for ts, value in zip(timestamps, values)
        if ts.date() == latest_date
    )

    monthly_energy = sum(
        value * interval_hours
        for ts, value in zip(timestamps, values)
        if (ts.year, ts.month) == latest_month
    )

    yearly_energy = sum(
        value * interval_hours
        for ts, value in zip(timestamps, values)
        if ts.year == latest_year
    )

    return {
        "current": latest_value,
        "current_ts": latest_ts,
        "mean": sum(values) / len(values),
        "max": max(values),
        "min": min(values),
        "threshold": thr,
        "abnormal": sum(
            1 for value in values
            if thr is not None and value > thr
        ),
        "status": (
            "HIGH"
            if thr is not None and latest_value > thr
            else "NORMAL"
        ),

        # Electricity consumption
        "daily_energy": daily_energy,
        "monthly_energy": monthly_energy,
        "yearly_energy": yearly_energy,

        # Estimated electricity costs
        "daily_cost": daily_energy * tariff_per_kwh,
        "monthly_cost": monthly_energy * tariff_per_kwh,
        "yearly_cost": yearly_energy * tariff_per_kwh,

        "tariff": tariff_per_kwh,
    }
