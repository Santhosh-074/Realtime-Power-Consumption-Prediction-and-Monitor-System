"""End-to-end check of every major feature, using a throwaway database.

Run:  python smoke_test.py
"""
import io
import os
import shutil
import tempfile

tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tmp, "test.db")

import app as appmod  # noqa: E402  (must come after DATABASE_URL is set)

app = appmod.app
app.config["WTF_CSRF_ENABLED"] = False
app.config["UPLOAD_FOLDER"] = os.path.join(tmp, "uploads")
app.config["MODEL_FOLDER"] = os.path.join(tmp, "models")
os.makedirs(app.config["UPLOAD_FOLDER"])
os.makedirs(app.config["MODEL_FOLDER"])

passed = failed = 0


def check(name, condition, extra=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name} {extra}")


runner = app.test_cli_runner()
runner.invoke(args=["seed"])

user = app.test_client()
admin = app.test_client()
other = app.test_client()

print("Authentication")
r = user.post("/register", data={"name": "Vinay", "email": "vinay@test.com", "password": "password1", "confirm": "password1"})
check("register redirects to login", r.status_code == 302)
r = user.post("/register", data={"name": "X", "email": "vinay@test.com", "password": "password1", "confirm": "password1"}, follow_redirects=True)
check("duplicate email rejected", b"already exists" in r.data)
r = user.post("/login", data={"email": "vinay@test.com", "password": "wrong"}, follow_redirects=True)
check("wrong password rejected", b"Invalid email or password" in r.data)
r = user.post("/login", data={"email": "vinay@test.com", "password": "password1"})
check("user login ok", r.status_code == 302)
r = admin.post("/login", data={"email": "rtpcmps@gmail.com", "password": "rtpcmps@123"})
check("admin login ok", r.status_code == 302)
check("admin lands on admin panel", admin.get("/", follow_redirects=False).headers["Location"].endswith("/admin"))
check("user blocked from /admin (RBAC)", user.get("/admin").status_code == 403)
check("anonymous redirected to login", app.test_client().get("/dashboard").status_code == 302)

print("Password reset")
r = other.post("/reset", data={"email": "vinay@test.com"}, follow_redirects=True)
check("reset link shown", b"/reset/" in r.data)
with app.app_context():
    token = appmod.User.query.filter_by(email="vinay@test.com").first().reset_token
r = other.post(f"/reset/{token}", data={"password": "newpass123", "confirm": "newpass123"})
check("password reset accepted", r.status_code == 302)
r = other.post("/login", data={"email": "vinay@test.com", "password": "newpass123"})
check("login with new password", r.status_code == 302)
user.post("/login", data={"email": "vinay@test.com", "password": "newpass123"})

print("Dataset upload and validation")
r = user.post("/upload", data={"file": (io.BytesIO(b"hello"), "notes.txt")}, content_type="multipart/form-data", follow_redirects=True)
check("non-CSV rejected", b"upload a .csv" in r.data)
r = user.post("/upload", data={"file": (io.BytesIO(b"a,b\n1,2\n3,4\n"), "tiny.csv")}, content_type="multipart/form-data", follow_redirects=True)
check("too-small CSV rejected", any(m in r.data for m in (b"Too few", b"Not enough", b"does not look", b"No readable dates")))

sample = open("sample_data/household_power_hourly.csv", "rb").read()
r = user.post("/upload", data={"name": "Home 2025", "file": (io.BytesIO(sample), "household.csv")},
              content_type="multipart/form-data")
check("sample CSV uploaded", r.status_code == 302 and "/dashboard?dataset=1" in r.headers["Location"])

r = user.get("/datasets/1")
check("validation report page", b"Duplicate timestamps" in r.data and b"8760" in r.data)

print("Train, predict, alerts")
r = user.post("/datasets/1/predict", data={"steps": 24}, follow_redirects=True)
check("predict before training refused", b"Train a model first" in r.data)
for algo in ("random_forest", "ridge", "gradient_boosting"):
    r = user.post("/datasets/1/train", data={"algorithm": algo}, follow_redirects=True)
    check(f"train {algo}", b"Model trained" in r.data)
with app.app_context():
    m = appmod.services.latest_model(1)
    print(f"        latest model: {m.algorithm}  MAE={m.mae:.3f}  RMSE={m.rmse:.3f}  R2={m.r2:.3f}  baseline MAE={m.baseline_mae:.3f}")
    check("model R2 is sensible (> 0.5)", m.r2 > 0.5)
user.post("/datasets/1/train", data={"algorithm": "random_forest"})
r = user.post("/datasets/1/predict", data={"steps": 48}, follow_redirects=True)
check("forecast generated", b"Forecast generated for the next 48 hours" in r.data)
with app.app_context():
    check("48 predictions stored", appmod.Prediction.query.filter_by(dataset_id=1).count() == 48)
    check("no negative predictions", all(p.value >= 0 for p in appmod.Prediction.query.all()))

print("Dashboard and APIs")
r = user.get("/dashboard")
check("dashboard renders", r.status_code == 200 and b"Current consumption" in r.data and b"predChart" in r.data)
j = user.get("/api/datasets/1/history?start=2025-03-01&end=2025-03-07").get_json()
check("history API (7 days, raw hourly)", len(j["values"]) == 7 * 24 and j["bucket"] == "raw")
j = user.get("/api/datasets/1/history").get_json()
check("history API downsampled for full year", 0 < len(j["values"]) <= 1500 and j["bucket"] != "raw")
j = user.get("/api/datasets/1/prediction").get_json()
check("prediction API", len(j["forecast"]["values"]) == 48 and len(j["history"]["values"]) == 72)
j = user.get("/api/datasets/1/evaluation").get_json()
check("evaluation API", len(j["actual"]) == len(j["predicted"]) > 0)

print("Access control between users")
other.get("/logout")
other.post("/register", data={"name": "Other", "email": "other@test.com", "password": "password1", "confirm": "password1"})
other.post("/login", data={"email": "other@test.com", "password": "password1"})
check("other user cannot open dataset", other.get("/api/datasets/1/history").status_code == 404)
check("other user cannot delete dataset", other.post("/datasets/1/delete").status_code == 404)
check("admin can open any dataset", admin.get("/api/datasets/1/history").status_code == 200)

print("Threshold and alerts")
r = user.post("/datasets/1/settings", data={"name": "Home 2025", "threshold": "1.0"}, follow_redirects=True)
check("custom threshold saved", b"Settings saved" in r.data)
r = user.get("/alerts")
check("alert history lists forecast alerts", b"forecast" in r.data)
r = user.post("/datasets/1/settings", data={"name": "Home 2025", "threshold": "-5"}, follow_redirects=True)
check("invalid threshold rejected", b"positive number" in r.data)
user.post("/datasets/1/settings", data={"name": "Home 2025", "threshold": ""})

print("Reports")
r = user.get("/reports?dataset=1&start=2025-06-01&end=2025-06-30")
check("report page", b"Summary" in r.data and b"720" in r.data)
r = user.get("/reports/1/csv?start=2025-06-01&end=2025-06-30")
check("CSV download", r.mimetype == "text/csv" and r.data.startswith(b"timestamp,power_kw,type"))
r = user.get("/reports/1/print")
check("printable report", b"Power Consumption Report" in r.data)

print("Dataset management")
extra = b"timestamp,global_active_power_kw\n" + b"".join(
    f"2026-01-{d:02d} {h:02d}:00:00,{1.0 + 0.1 * (h % 5):.2f}\n".encode() for d in range(1, 6) for h in range(24))
r = user.post("/datasets/1/update", data={"mode": "append", "file": (io.BytesIO(extra), "more.csv")},
              content_type="multipart/form-data", follow_redirects=True)
check("append new records", b"Dataset updated" in r.data)
with app.app_context():
    ds = appmod.db.session.get(appmod.Dataset, 1)
    check("rows grew after append (with gap interpolated not required)", ds.rows > 8760)
    check("models cleared after update", appmod.ModelRecord.query.filter_by(dataset_id=1).count() == 0)

print("UCI-style format (semicolon, Date + Time, '?' missing)")
lines = ["Date;Time;Global_active_power;Global_reactive_power;Voltage"]
import datetime as dt  # noqa: E402
t0 = dt.datetime(2007, 1, 1)
for i in range(60 * 24 * 12):  # 12 days of minute data
    t = t0 + dt.timedelta(minutes=i)
    v = "?" if i % 997 == 0 else f"{1.0 + 0.5 * ((t.hour % 24) / 24):.3f}"
    lines.append(f"{t.day}/{t.month}/{t.year};{t:%H:%M:%S};{v};0.1;240.0")
r = user.post("/upload", data={"name": "UCI style", "file": (io.BytesIO("\n".join(lines).encode()), "uci.csv")},
              content_type="multipart/form-data")
check("UCI-style file accepted", r.status_code == 302, r.data[:200])
with app.app_context():
    ds = appmod.Dataset.query.filter_by(name="UCI style").first()
    check("minute data resampled to hourly (288 rows)", ds is not None and ds.rows == 288 and ds.freq == "h",
          f"rows={getattr(ds, 'rows', None)}")

print("Admin")
r = admin.get("/admin")
check("admin panel", r.status_code == 200 and b"Activity log" in r.data and b"dataset_upload" in r.data)
r = admin.post("/admin/users/2/role", data={"role": "admin"}, follow_redirects=True)
check("role change", b"Role updated" in r.data or b"cannot change" in r.data)
with app.app_context():
    uid = appmod.User.query.filter_by(email="other@test.com").first().id
r = admin.post(f"/admin/users/{uid}/delete", follow_redirects=True)
check("delete user", b"User deleted" in r.data)

print("Delete dataset")
r = user.post("/datasets/1/delete", follow_redirects=True)
check("dataset deleted", b"Dataset deleted" in r.data)
with app.app_context():
    check("readings removed", appmod.Reading.query.filter_by(dataset_id=1).count() == 0)

print("Logout")
r = user.get("/logout")
check("logout", r.status_code == 302)
check("dashboard locked after logout", user.get("/dashboard").status_code == 302)

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
