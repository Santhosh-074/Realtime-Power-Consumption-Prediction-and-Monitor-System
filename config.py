import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE = os.path.join(BASE_DIR, "instance")


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-in-production")

    # SRS C-003 asks for MySQL. Set DATABASE_URL to use it, for example:
    #   mysql+pymysql://user:password@localhost/rtpcpms
    # With no DATABASE_URL we fall back to SQLite so the project runs with zero setup.
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + os.path.join(INSTANCE, "rtpcpms.db").replace("\\", "/"),
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(INSTANCE, "uploads")
    MODEL_FOLDER = os.path.join(INSTANCE, "models")
    LOG_FILE = os.path.join(INSTANCE, "app.log")
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB (UCI dataset is ~127 MB)

    # F-003: session timeout after 20 minutes of inactivity
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=20)
    SESSION_REFRESH_EACH_REQUEST = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    # Turn on when the app is served over HTTPS (N-020)
    SESSION_COOKIE_SECURE = os.environ.get("USE_HTTPS", "0") == "1"

    # Optional SMTP settings for password reset email.
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", "587"))
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", "1") == "1"
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "")

    # Default alert threshold = mean + ALERT_SIGMA * std of the dataset
    ALERT_SIGMA = 2.0
