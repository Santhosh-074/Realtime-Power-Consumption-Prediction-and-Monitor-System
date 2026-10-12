from datetime import datetime

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")  # 'admin' or 'user'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reset_token = db.Column(db.String(100), index=True)
    reset_expires = db.Column(db.DateTime)

    @property
    def is_admin(self):
        return self.role == "admin"


class Dataset(db.Model):
    __tablename__ = "datasets"
    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(150), nullable=False)
    filename = db.Column(db.String(255))
    stored_path = db.Column(db.String(500))
    freq = db.Column(db.String(5), default="h")  # 'h' hourly or 'D' daily
    rows = db.Column(db.Integer, default=0)
    start_ts = db.Column(db.DateTime)
    end_ts = db.Column(db.DateTime)
    mean_value = db.Column(db.Float)
    std_value = db.Column(db.Float)
    threshold = db.Column(db.Float)  # manual override; None means automatic
    report_json = db.Column(db.Text)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    owner = db.relationship("User", backref="datasets")

    def effective_threshold(self, sigma=2.0):
        if self.threshold is not None:
            return self.threshold
        if self.mean_value is None:
            return None
        return self.mean_value + sigma * (self.std_value or 0.0)


class Reading(db.Model):
    __tablename__ = "readings"
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False, index=True)
    ts = db.Column(db.DateTime, nullable=False, index=True)
    value = db.Column(db.Float, nullable=False)


class ModelRecord(db.Model):
    __tablename__ = "models"
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False, index=True)
    algorithm = db.Column(db.String(40))
    mae = db.Column(db.Float)
    rmse = db.Column(db.Float)
    r2 = db.Column(db.Float)
    baseline_mae = db.Column(db.Float)
    train_rows = db.Column(db.Integer)
    test_rows = db.Column(db.Integer)
    path = db.Column(db.String(500))
    eval_json = db.Column(db.Text)  # actual vs predicted on the test set
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)

    dataset = db.relationship("Dataset", backref="models")


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False, index=True)
    model_id = db.Column(db.Integer, db.ForeignKey("models.id"))
    ts = db.Column(db.DateTime, nullable=False)
    value = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Alert(db.Model):
    __tablename__ = "alerts"
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), nullable=False, index=True)
    kind = db.Column(db.String(20))  # 'current' or 'forecast'
    ts = db.Column(db.DateTime)
    value = db.Column(db.Float)
    threshold = db.Column(db.Float)
    message = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    dataset = db.relationship("Dataset", backref="alerts")


class ActivityLog(db.Model):
    __tablename__ = "activity_logs"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    user_email = db.Column(db.String(150))
    action = db.Column(db.String(60))
    detail = db.Column(db.String(300))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
