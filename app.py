import io
import json
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import urlparse

import pandas as pd
from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template,
                   request, session, url_for)
from flask_login import (LoginManager, current_user, login_required, login_user,
                         logout_user)
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import ml
import services
# from services import (
#     dataset_stats,
#     train,
#     predict,
#     forecast_cost_summary,
# )
from config import INSTANCE, Config
from models import (ActivityLog, Alert, Dataset, ModelRecord, Prediction, Reading, User, db)

app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(INSTANCE, exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

db.init_app(app)
csrf = CSRFProtect(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in (sessions expire after 20 minutes of inactivity)."
login_manager.login_message_category = "warning"

# N-005: system and error logs
logging.basicConfig(
    filename=app.config["LOG_FILE"], level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")

SIGMA = app.config["ALERT_SIGMA"]


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def log_action(action, detail=""):
    """N-024: record important system activities."""
    email = current_user.email if current_user.is_authenticated else None
    uid = current_user.id if current_user.is_authenticated else None
    db.session.add(ActivityLog(user_id=uid, user_email=email, action=action, detail=detail[:300]))
    db.session.commit()
    logging.info("%s | %s | %s", email, action, detail)


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)  # F-004 RBAC
        return view(*args, **kwargs)
    return wrapper


def accessible_datasets():
    q = Dataset.query
    if not current_user.is_admin:
        q = q.filter_by(owner_id=current_user.id)
    return q.order_by(Dataset.uploaded_at.desc())


def get_dataset_or_404(dataset_id):
    ds = db.session.get(Dataset, dataset_id)
    if ds is None or (not current_user.is_admin and ds.owner_id != current_user.id):
        abort(404)
    return ds


def safe_next(target):
    if not target:
        return None
    parsed = urlparse(target)
    return target if not parsed.netloc and not parsed.scheme and target.startswith("/") else None


def parse_date(text, end=False):
    if not text:
        return None
    try:
        d = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    return d + timedelta(days=1) - timedelta(seconds=1) if end else d


def save_upload(file_storage):
    """Validate the file name/type and store it. Returns (original, stored_path)."""
    original = secure_filename(file_storage.filename or "")
    if not original.lower().endswith(".csv"):
        raise ml.DatasetError("Please upload a .csv file.")
    stored = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex[:10]}_{original}")
    file_storage.save(stored)
    return original, stored


# --------------------------------------------------------------------------
# Authentication (F-001..F-004)
# --------------------------------------------------------------------------
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("admin_home" if current_user.is_admin else "dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if not name or not email or "@" not in email:
            flash("Please enter your name and a valid email address.", "danger")
        elif len(password) < 8:
            flash("Password must be at least 8 characters long.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "danger")
        else:
            user = User(name=name, email=email, role="user",
                        password_hash=generate_password_hash(password))
            db.session.add(user)
            db.session.commit()
            logging.info("%s | register", email)
            flash("Account created. You can log in now.", "success")
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET"])
def login():
    """Show the page where visitors choose User Login or Admin Login."""
    if current_user.is_authenticated:
        return redirect(url_for("admin_home" if current_user.is_admin else "dashboard"))
    return render_template("login.html")


def authenticate_role(expected_role):
    """Authenticate an account only if it has the required role."""
    if current_user.is_authenticated:
        if current_user.is_admin and expected_role == "admin":
            return redirect(url_for("admin_home"))
        if not current_user.is_admin and expected_role == "user":
            return redirect(url_for("dashboard"))
        logout_user()
        session.clear()
        flash("Please sign in with an account for this portal.", "warning")

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            if user.role != expected_role:
                db.session.add(ActivityLog(
                    user_email=email,
                    action="login_role_mismatch",
                    detail=f"account role does not match {expected_role} portal"
                ))
                db.session.commit()
                flash(
                    "This account does not have permission to use this login portal.",
                    "danger"
                )
                return render_template("role_login.html", role=expected_role)

            login_user(user)
            session.permanent = True
            log_action("login", f"successful {expected_role} login")
            target = safe_next(request.args.get("next"))
            if target:
                return redirect(target)
            return redirect(url_for("admin_home" if user.is_admin else "dashboard"))

        db.session.add(ActivityLog(
            user_email=email,
            action="login_failed",
            detail="invalid credentials"
        ))
        db.session.commit()
        logging.warning("%s | login_failed | portal=%s", email, expected_role)
        flash("Invalid email or password.", "danger")

    return render_template("role_login.html", role=expected_role)


@app.route("/user-login", methods=["GET", "POST"])
def user_login():
    return authenticate_role("user")


@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    return authenticate_role("admin")


@app.route("/logout")
@login_required
def logout():
    log_action("logout")
    logout_user()
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/reset", methods=["GET", "POST"])
def reset_request():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(email=email).first()
        if user:
            user.reset_token = secrets.token_urlsafe(24)
            user.reset_expires = datetime.utcnow() + timedelta(minutes=30)
            db.session.commit()
            link = url_for("reset_password", token=user.reset_token, _external=True)
            try:
                services.send_password_reset_email(app, user.email, link)
                logging.info("%s | password_reset_email_sent", email)
            except Exception:
                # Do not leave a usable token behind when delivery failed.
                user.reset_token = None
                user.reset_expires = None
                db.session.commit()
                logging.exception("%s | password_reset_email_failed", email)
        flash("If that email is registered, a password reset link will be sent shortly.", "info")
        return redirect(url_for("reset_request"))
    return render_template("reset_request.html")


@app.route("/reset/<token>", methods=["GET", "POST"])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if user is None or user.reset_expires is None or user.reset_expires < datetime.utcnow():
        flash("This reset link is invalid or has expired.", "danger")
        return redirect(url_for("reset_request"))
    if request.method == "POST":
        password = request.form.get("password", "")
        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "danger")
        elif password != request.form.get("confirm", ""):
            flash("Passwords do not match.", "danger")
        else:
            user.password_hash = generate_password_hash(password)
            user.reset_token = None
            user.reset_expires = None
            db.session.commit()
            logging.info("%s | password_reset_done", user.email)
            flash("Password updated. Please log in.", "success")
            return redirect(url_for("login"))
    return render_template("reset_form.html", token=token)


# --------------------------------------------------------------------------
# Dashboard (F-014..F-017)
# --------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    datasets = accessible_datasets().all()
    selected = None

    ds_id = request.args.get("dataset", type=int)

    if ds_id:
        selected = get_dataset_or_404(ds_id)
    elif datasets:
        selected = datasets[0]

    ctx = {
        "datasets": datasets,
        "selected": selected,
        "algorithms": ml.ALGORITHMS,
    }

    if selected:
        ctx.update(
            stats=services.dataset_stats(selected, SIGMA),
            model=services.latest_model(selected.id),
            alerts=(
                Alert.query.filter_by(dataset_id=selected.id)
                .order_by(Alert.created_at.desc(), Alert.id.desc())
                .limit(8)
                .all()
            ),
            default_steps=ml.DEFAULT_HORIZON[selected.freq],
            max_steps=ml.MAX_HORIZON[selected.freq],
            has_predictions=(
                Prediction.query.filter_by(dataset_id=selected.id).count() > 0
            ),
            unit_label="hours" if selected.freq == "h" else "days",

            # Predicted electricity consumption and cost
            forecast_summary=services.forecast_cost_summary(selected),
        )

    return render_template("dashboard.html", **ctx)

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        flash("Please choose a CSV file.", "danger")
        return redirect(request.referrer or url_for("dashboard"))
    stored = None
    try:
        original, stored = save_upload(file)
        name = request.form.get("name", "").strip() or os.path.splitext(original)[0]
        ds, report = services.create_dataset(current_user, name, original, stored, SIGMA)
        log_action("dataset_upload", f"{ds.name} ({ds.rows} rows)")
        flash("Dataset uploaded and validated. Next: train a model.", "success")
        return redirect(url_for("dashboard", dataset=ds.id))
    except ml.DatasetError as exc:
        db.session.rollback()
        if stored and os.path.exists(stored):
            os.remove(stored)
        log_action("dataset_upload_failed", str(exc))
        flash(str(exc), "danger")
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/datasets/<int:dataset_id>/train", methods=["POST"])
@login_required
def train(dataset_id):
    ds = get_dataset_or_404(dataset_id)
    algorithm = request.form.get("algorithm", "random_forest")
    if algorithm not in ml.ALGORITHMS:
        abort(400)
    try:
        rec = services.train(ds, algorithm, app.config["MODEL_FOLDER"])
        log_action("model_trained", f"{ds.name}: {algorithm}, MAE={rec.mae:.3f}")
        flash(f"Model trained ({ml.ALGORITHMS[algorithm]}). MAE {rec.mae:.3f}, R² {rec.r2:.3f}.", "success")
    except ml.DatasetError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("dashboard", dataset=ds.id))


@app.route("/datasets/<int:dataset_id>/predict", methods=["POST"])
@login_required
def predict(dataset_id):
    ds = get_dataset_or_404(dataset_id)
    steps = request.form.get("steps", type=int) or ml.DEFAULT_HORIZON[ds.freq]
    try:
        result = services.predict(ds, steps, SIGMA)
        log_action("prediction_generated", f"{ds.name}: {len(result)} steps")
        flash(f"Forecast generated for the next {len(result)} {'hours' if ds.freq == 'h' else 'days'}.", "success")
    except ml.DatasetError as exc:
        flash(str(exc), "danger")
    return redirect(url_for("dashboard", dataset=ds.id))


# --------------------------------------------------------------------------
# JSON APIs used by the charts
# --------------------------------------------------------------------------
BUCKETS = [("1h", pd.Timedelta(hours=1)), ("3h", pd.Timedelta(hours=3)),
           ("6h", pd.Timedelta(hours=6)), ("12h", pd.Timedelta(hours=12)),
           ("D", pd.Timedelta(days=1)), ("W", pd.Timedelta(days=7)),
           ("MS", pd.Timedelta(days=30))]


@app.route("/api/datasets/<int:dataset_id>/history")
@login_required
def api_history(dataset_id):
    ds = get_dataset_or_404(dataset_id)
    start = parse_date(request.args.get("start"))
    end = parse_date(request.args.get("end"), end=True)
    q = db.session.query(Reading.ts, Reading.value).filter_by(dataset_id=ds.id)
    if start:
        q = q.filter(Reading.ts >= start)
    if end:
        q = q.filter(Reading.ts <= end)
    rows = q.order_by(Reading.ts).all()
    if not rows:
        return jsonify({"labels": [], "values": [], "threshold": ds.effective_threshold(SIGMA), "bucket": None})
    s = pd.Series([r[1] for r in rows], index=pd.DatetimeIndex([r[0] for r in rows]))
    bucket = "raw"
    max_points = 1500
    if len(s) > max_points:
        span = s.index[-1] - s.index[0]
        needed = span / max_points
        for rule, delta in BUCKETS:
            bucket = rule
            if delta >= needed:
                break
        s = s.resample(bucket).mean().dropna()
    fmt = "%Y-%m-%d %H:%M"
    return jsonify({
        "labels": [t.strftime(fmt) for t in s.index],
        "values": [round(float(v), 4) for v in s.values],
        "threshold": ds.effective_threshold(SIGMA),
        "bucket": bucket,
    })


@app.route("/api/datasets/<int:dataset_id>/prediction")
@login_required
def api_prediction(dataset_id):
    ds = get_dataset_or_404(dataset_id)
    preds = (Prediction.query.filter_by(dataset_id=ds.id).order_by(Prediction.ts).all())
    recent_n = 72 if ds.freq == "h" else 21
    recent = (db.session.query(Reading.ts, Reading.value).filter_by(dataset_id=ds.id)
              .order_by(Reading.ts.desc()).limit(recent_n).all())[::-1]
    fmt = "%Y-%m-%d %H:%M"
    return jsonify({
        "history": {"labels": [r[0].strftime(fmt) for r in recent],
                    "values": [round(r[1], 4) for r in recent]},
        "forecast": {"labels": [p.ts.strftime(fmt) for p in preds],
                     "values": [round(p.value, 4) for p in preds]},
        "threshold": ds.effective_threshold(SIGMA),
    })


@app.route("/api/datasets/<int:dataset_id>/evaluation")
@login_required
def api_evaluation(dataset_id):
    ds = get_dataset_or_404(dataset_id)
    model = services.latest_model(ds.id)
    if not model:
        return jsonify({"labels": [], "actual": [], "predicted": []})
    return jsonify(json.loads(model.eval_json))


# --------------------------------------------------------------------------
# Dataset management (F-005..F-007)
# --------------------------------------------------------------------------
@app.route("/datasets")
@login_required
def datasets():
    return render_template("datasets.html", datasets=accessible_datasets().all())


@app.route("/datasets/<int:dataset_id>")
@login_required
def dataset_detail(dataset_id):
    ds = get_dataset_or_404(dataset_id)
    report = json.loads(ds.report_json) if ds.report_json else {}
    return render_template("dataset_detail.html", ds=ds, report=report,
                           auto_threshold=(ds.mean_value or 0) + SIGMA * (ds.std_value or 0),
                           effective=ds.effective_threshold(SIGMA))


@app.route("/datasets/<int:dataset_id>/update", methods=["POST"])
@login_required
def dataset_update(dataset_id):
    ds = get_dataset_or_404(dataset_id)
    file = request.files.get("file")
    mode = request.form.get("mode", "replace")
    if mode not in ("replace", "append") or not file or not file.filename:
        flash("Choose a CSV file and a mode.", "danger")
        return redirect(url_for("dataset_detail", dataset_id=ds.id))
    stored = None
    try:
        original, stored = save_upload(file)
        services.update_dataset(ds, stored, mode, SIGMA)
        old = ds.stored_path
        ds.filename, ds.stored_path = original, stored
        db.session.commit()
        if mode == "replace" and old and os.path.exists(old):
            os.remove(old)
        log_action("dataset_updated", f"{ds.name}: {mode}")
        flash("Dataset updated. Previous models and forecasts were cleared; please train again.", "success")
    except ml.DatasetError as exc:
        db.session.rollback()
        if stored and os.path.exists(stored):
            os.remove(stored)
        flash(str(exc), "danger")
    return redirect(url_for("dataset_detail", dataset_id=ds.id))


@app.route("/datasets/<int:dataset_id>/settings", methods=["POST"])
@login_required
def dataset_settings(dataset_id):
    ds = get_dataset_or_404(dataset_id)
    name = request.form.get("name", "").strip()
    if name:
        ds.name = name[:150]
    raw = request.form.get("threshold", "").strip()
    if raw == "":
        ds.threshold = None
    else:
        try:
            value = float(raw)
            if value <= 0:
                raise ValueError
            ds.threshold = value
        except ValueError:
            flash("Threshold must be a positive number (or empty for automatic).", "danger")
            return redirect(url_for("dataset_detail", dataset_id=ds.id))
    db.session.flush()
    series = services.get_series(ds)
    services.check_current(ds, series, SIGMA)
    Alert.query.filter_by(dataset_id=ds.id, kind="forecast").delete()
    thr = ds.effective_threshold(SIGMA)
    for p in Prediction.query.filter_by(dataset_id=ds.id).all():
        if thr is not None and p.value > thr:
            db.session.add(Alert(dataset_id=ds.id, kind="forecast", ts=p.ts, value=p.value, threshold=thr,
                                 message=f"Predicted consumption {p.value:.2f} kW is above the threshold {thr:.2f} kW"))
    db.session.commit()
    log_action("threshold_updated", f"{ds.name}: {thr:.3f}")
    flash("Settings saved.", "success")
    return redirect(url_for("dataset_detail", dataset_id=ds.id))


@app.route("/datasets/<int:dataset_id>/delete", methods=["POST"])
@login_required
def dataset_delete(dataset_id):
    ds = get_dataset_or_404(dataset_id)
    name = ds.name
    services.delete_dataset(ds)
    log_action("dataset_deleted", name)
    flash("Dataset deleted.", "success")
    return redirect(request.form.get("back") if safe_next(request.form.get("back")) else url_for("datasets"))


# --------------------------------------------------------------------------
# Alerts (F-019..F-021) and reports (F-018)
# --------------------------------------------------------------------------
@app.route("/alerts")
@login_required
def alerts():
    ids = [d.id for d in accessible_datasets().all()]
    q = Alert.query.filter(Alert.dataset_id.in_(ids)) if ids else Alert.query.filter(False)
    ds_id = request.args.get("dataset", type=int)
    if ds_id:
        q = q.filter_by(dataset_id=ds_id)
    items = q.order_by(Alert.created_at.desc(), Alert.id.desc()).limit(300).all()
    return render_template("alerts.html", alerts=items, datasets=accessible_datasets().all(), selected_id=ds_id)


def build_report(ds, start, end):
    rq = Reading.query.filter_by(dataset_id=ds.id)
    pq = Prediction.query.filter_by(dataset_id=ds.id)
    aq = Alert.query.filter_by(dataset_id=ds.id)
    if start:
        rq, pq = rq.filter(Reading.ts >= start), pq.filter(Prediction.ts >= start)
        aq = aq.filter(Alert.ts >= start)
    if end:
        rq, pq = rq.filter(Reading.ts <= end), pq.filter(Prediction.ts <= end)
        aq = aq.filter(Alert.ts <= end)
    readings = rq.order_by(Reading.ts).all()
    preds = pq.order_by(Prediction.ts).all()
    alerts_ = aq.order_by(Alert.ts).all()
    values = [r.value for r in readings]
    stats = None
    if values:
        stats = {"count": len(values), "mean": sum(values) / len(values),
                 "max": max(values), "min": min(values),
                 "total": sum(values)}
    return {"readings": readings, "preds": preds, "alerts": alerts_, "stats": stats,
            "model": services.latest_model(ds.id), "threshold": ds.effective_threshold(SIGMA)}


@app.route("/reports")
@login_required
def reports():
    dss = accessible_datasets().all()
    ds_id = request.args.get("dataset", type=int)
    selected = get_dataset_or_404(ds_id) if ds_id else (dss[0] if dss else None)
    start_s, end_s = request.args.get("start", ""), request.args.get("end", "")
    report = None
    if selected:
        report = build_report(selected, parse_date(start_s), parse_date(end_s, end=True))
    return render_template("reports.html", datasets=dss, selected=selected, report=report,
                           start=start_s, end=end_s)
#-------------------------------------------------------------------------

# @app.route("/reports/<int:dataset_id>/csv")
# @login_required
# def report_csv(dataset_id):
#     ds = get_dataset_or_404(dataset_id)
#     rep = build_report(ds, parse_date(request.args.get("start")), parse_date(request.args.get("end"), end=True))
#     out = io.StringIO()
#     out.write("timestamp,power_kw,type\n")
#     for r in rep["readings"]:
#         out.write(f"{r.ts},{r.value:.4f},actual\n")
#     for p in rep["preds"]:
#         out.write(f"{p.ts},{p.value:.4f},predicted\n")
#     log_action("report_downloaded", ds.name)
#     return Response(out.getvalue(), mimetype="text/csv",
#                     headers={"Content-Disposition": f"attachment; filename=report_{ds.id}.csv"})

#-------------------------
@app.route("/reports/<int:dataset_id>/csv")
@login_required
def report_csv(dataset_id):
    ds = get_dataset_or_404(dataset_id)

    rep = build_report(
        ds,
        parse_date(request.args.get("start")),
        parse_date(request.args.get("end"), end=True)
    )

    # Electricity tariff in rupees per kWh
    tariff_per_kwh = 7.0

    # Duration represented by each reading
    interval_hours = 24 if ds.freq == "d" else 1

    out = io.StringIO()

    # Report headings
    out.write(
        "timestamp,power_kw,energy_kwh,"
        "estimated_cost_inr,type\n"
    )

    total_actual_energy = 0.0
    total_actual_cost = 0.0

    total_predicted_energy = 0.0
    total_predicted_cost = 0.0

    # Export actual readings with their estimated costs
    for r in rep["readings"]:
        energy_kwh = float(r.value) * interval_hours
        estimated_cost = energy_kwh * tariff_per_kwh

        total_actual_energy += energy_kwh
        total_actual_cost += estimated_cost

        out.write(
            f"{r.ts},{r.value:.4f},"
            f"{energy_kwh:.4f},"
            f"{estimated_cost:.2f},actual\n"
        )

    # Export predicted readings with their estimated costs
    for p in rep["preds"]:
        energy_kwh = float(p.value) * interval_hours
        estimated_cost = energy_kwh * tariff_per_kwh

        total_predicted_energy += energy_kwh
        total_predicted_cost += estimated_cost

        out.write(
            f"{p.ts},{p.value:.4f},"
            f"{energy_kwh:.4f},"
            f"{estimated_cost:.2f},predicted\n"
        )

    # Add a blank line and report summary
    out.write("\n")
    out.write("REPORT SUMMARY\n")
    out.write("category,energy_kwh,estimated_cost_inr\n")

    out.write(
        f"Actual consumption,"
        f"{total_actual_energy:.2f},"
        f"{total_actual_cost:.2f}\n"
    )

    out.write(
        f"Predicted consumption,"
        f"{total_predicted_energy:.2f},"
        f"{total_predicted_cost:.2f}\n"
    )

    out.write(f"Electricity tariff,₹{tariff_per_kwh}/kWh,\n")

    log_action("report_downloaded", ds.name)

    return Response(
        out.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                f"attachment; filename=power_cost_report_{ds.id}.csv"
        }
    )

@app.route("/reports/<int:dataset_id>/print")
@login_required
def report_print(dataset_id):
    ds = get_dataset_or_404(dataset_id)
    start_s, end_s = request.args.get("start", ""), request.args.get("end", "")
    rep = build_report(ds, parse_date(start_s), parse_date(end_s, end=True))
    return render_template("report_print.html", ds=ds, report=rep, start=start_s, end=end_s,
                           generated=datetime.now())


# --------------------------------------------------------------------------
# Administration (2.2.1)
# --------------------------------------------------------------------------
@app.route("/admin")
@admin_required
def admin_home():
    return render_template(
        "admin.html",
        users=User.query.order_by(User.id).all(),
        datasets=Dataset.query.order_by(Dataset.uploaded_at.desc()).all(),
        models=ModelRecord.query.order_by(ModelRecord.trained_at.desc()).limit(20).all(),
        logs=ActivityLog.query.order_by(ActivityLog.id.desc()).limit(100).all(),
        counts={"users": User.query.count(), "datasets": Dataset.query.count(),
                "models": ModelRecord.query.count(), "alerts": Alert.query.count()})


@app.route("/admin/users/<int:user_id>/role", methods=["POST"])
@admin_required
def admin_role(user_id):
    user = db.session.get(User, user_id) or abort(404)
    role = request.form.get("role")
    if role not in ("admin", "user"):
        abort(400)
    if user.id == current_user.id:
        flash("You cannot change your own role.", "warning")
    else:
        user.role = role
        db.session.commit()
        log_action("role_changed", f"{user.email} -> {role}")
        flash("Role updated.", "success")
    return redirect(url_for("admin_home"))


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    user = db.session.get(User, user_id) or abort(404)
    if user.id == current_user.id:
        flash("You cannot delete your own account.", "warning")
        return redirect(url_for("admin_home"))
    for ds in Dataset.query.filter_by(owner_id=user.id).all():
        services.delete_dataset(ds)
    email = user.email
    db.session.delete(user)
    db.session.commit()
    log_action("user_deleted", email)
    flash("User deleted.", "success")
    return redirect(url_for("admin_home"))


# --------------------------------------------------------------------------
# Errors and CLI
# --------------------------------------------------------------------------
@app.errorhandler(403)
def forbidden(_):
    return render_template("error.html", code=403, message="You do not have permission to view this page."), 403


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, message="That page or dataset was not found."), 404


@app.errorhandler(413)
def too_large(_):
    flash("That file is too large (limit 200 MB).", "danger")
    return redirect(url_for("dashboard") if current_user.is_authenticated else url_for("login"))


@app.errorhandler(500)
def server_error(exc):
    logging.exception("Unhandled error: %s", exc)
    return render_template("error.html", code=500, message="Something went wrong. It has been logged."), 500


@app.cli.command("seed")
def seed():
    """Create a demo admin and a demo user."""
    for name, email, pw, role in [("Administrator", "rtpcmps@gmail.com", "rtpcmps@123", "admin"),
                                  ("Demo User", "user@rtpcpms.local", "User@1234", "user")]:
        if not User.query.filter_by(email=email).first():
            db.session.add(User(name=name, email=email, role=role, password_hash=generate_password_hash(pw)))
    db.session.commit()
    print("Admin: rtpcmps@gmail.com  / rtpcmps@123")
    print("User : user@rtpcpms.local / User@1234")


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
