# RTPCPMS – Real-Time Power Consumption Prediction and Monitoring System

Dataset-based prototype for the Software Engineering course (NITK Surathkal, Dept. of IT).
Instead of live smart-meter data, the user uploads a CSV of historical power consumption and
the system validates it, trains a Machine Learning model, predicts future consumption,
shows monitoring charts, raises alerts and produces reports.

Stack (as in the SRS): Python **Flask** backend, **scikit-learn / pandas / NumPy** for ML,
**HTML + CSS + JavaScript + Bootstrap + Chart.js** frontend, relational database via SQLAlchemy
(**SQLite by default, MySQL supported**).

---

## 1. Run it

```bash
cd rtpcpms
python -m venv venv
venv\Scripts\activate            # Windows      (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

python make_sample_data.py       # creates sample_data/household_power_hourly.csv
flask --app app seed             # creates demo accounts (see below)
python app.py                    # open http://127.0.0.1:5000
```

Demo accounts created by `seed`:

| Role  | Email                | Password   |
|-------|----------------------|------------|
| Admin | rtpcmps@gmail.com    | rtpcmps@123 |
| User  | user@rtpcpms.local   | User@1234  |

Anyone can also register at `/register` (new accounts are always normal users; an admin can promote them).

### Check that everything works
```bash
python smoke_test.py             # runs 49 checks across all features on a temporary database
```

---

## 2. How to use it (the demo flow)

1. Log in as the user → **Dashboard** → upload `sample_data/household_power_hourly.csv`.
2. Open **Validation report & settings** to see what was found and cleaned (missing values, duplicates, etc.).
3. On the dashboard click **Train** (choose Random Forest / Gradient Boosting / Ridge). MAE, RMSE and R² appear, with an actual-vs-predicted chart on unseen test data.
4. Click **Predict** (for example the next 48 hours). The forecast chart appears and any predicted value above the threshold becomes an alert.
5. **Alerts** shows the alert history. **Reports** gives a summary for a date range, CSV download, and a printable page (use *Save as PDF* in the browser print dialog).
6. Log in as admin → **Admin** to manage users and roles, see all datasets, models and the activity log.

### What CSV files are accepted
Any CSV with a date/time column and a power column. The columns are detected automatically:
- time column: a column named like `timestamp`, `datetime`, `date`, or a `Date` + `Time` pair
- power column: `Global_active_power`, or a column with `power`, `consumption`, `energy`, `kw`, `usage`, `load` in its name, otherwise the first remaining column
- separators `,` `;` tab `|`, missing values like `?` or blanks

The real UCI "Individual Household Electric Power Consumption" file works as it is (minute data is averaged to hourly).
Daily data also works (forecast steps are then days).

---

## 3. Using MySQL instead of SQLite (SRS C-003)

```sql
CREATE DATABASE rtpcpms CHARACTER SET utf8mb4;
```
```bash
pip install PyMySQL
set DATABASE_URL=mysql+pymysql://root:yourpassword@localhost/rtpcpms     # Linux/Mac: export
python app.py
```
Tables are created automatically on first start.

---

## 4. Where each SRS requirement lives

| SRS item | Where |
|---|---|
| F-001 Login, F-004 RBAC | `app.py` (`login`, `admin_required`, ownership checks) |
| F-002 Password reset | `app.py` (`reset_request`, `reset_password`) |
| F-003 20-minute session timeout | `config.py` (`PERMANENT_SESSION_LIFETIME`) |
| F-005 Upload, F-006 Validate, F-007 Manage | `app.py` + `ml.load_and_validate` + `services.py` |
| F-008 Missing values, F-009 Features, F-010 Normalization | `ml.py` (`load_and_validate`, `build_frame`, `MinMaxScaler` in the pipeline) |
| F-011 Train, F-012 Predict, F-013 MAE/RMSE/R² | `ml.py` (`train_model`, `forecast`) |
| F-014 – F-017 Monitoring and dashboard | `templates/dashboard.html`, `static/js/dashboard.js`, `/api/...` routes |
| F-018 Reports | `reports.html`, `report_print.html`, CSV route |
| F-019 – F-021 Alerts | `services.py` (`check_current`, `predict`), `alerts.html` |
| N-004 Modular architecture | `app.py` (web), `ml.py` (ML), `services.py` (logic), `models.py` (DB), `templates/` (UI) |
| N-005 / N-024 Logs | `instance/app.log` and the Activity log tab in Admin |
| N-021 Password hashing | Werkzeug `generate_password_hash` |
| N-023 Dataset validation | `ml.load_and_validate` (file type, columns, numeric checks) |

Mapping to your DFD:
0.1 Registration/Login → `register`, `login` · 0.2 Upload & Validation → `upload` · 0.3 Preprocessing & Normalization → `ml.py` ·
0.4 Model Training → `train` · 0.5 Prediction → `predict` · 0.6 Dashboard & Alerts → `dashboard`, `alerts` · 0.7 Administration → `/admin`.

---

## 5. Honest notes and limits

- **Prototype, not production.** Password reset shows the reset link on screen because there is no mail server (SRS lists email/SMS alerts as future work). Replace that with an email in production.
- **HTTPS (N-020)** is deployment configuration, not application code: put the app behind HTTPS (for example nginx or a hosting platform) and set `USE_HTTPS=1` so cookies are marked secure. Also set a real `SECRET_KEY`.
- The default alert threshold is mean + 2 standard deviations; users can set their own per dataset.
- Prediction is multi-step: each predicted hour feeds the next, so accuracy drops the further ahead you forecast.
- Charts load Bootstrap and Chart.js from a CDN, so the browser needs internet access.
- Units are shown as kW (the UCI dataset uses kW). Other units will still work but the label will say kW.
- Only a PDF *print view* is provided (browser Save as PDF), not a server-generated PDF.

## 6. Files

```
app.py              Flask routes, authentication, RBAC, APIs, admin
ml.py               validation, cleaning, features, normalization, training, forecasting
services.py         dataset storage, train/predict orchestration, alerts, statistics
models.py           database tables
config.py           settings (database, session timeout, thresholds)
templates/          HTML pages          static/   CSS and Chart.js code
make_sample_data.py sample CSV generator        smoke_test.py   automated feature check
```
