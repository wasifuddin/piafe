import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "piafe.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS Properties (
            property_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            address       TEXT,
            listing_type  TEXT CHECK(listing_type IN ('sale','rent','unknown')) DEFAULT 'unknown',
            created_date  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS Images (
            image_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id   INTEGER REFERENCES Properties(property_id),
            file_path     TEXT NOT NULL,
            file_format   TEXT,
            width         INTEGER,
            height        INTEGER,
            upload_date   TEXT DEFAULT (datetime('now')),
            status        TEXT CHECK(status IN ('queued','processing','complete','failed'))
                          DEFAULT 'queued'
        );

        CREATE TABLE IF NOT EXISTS RoomClassifications (
            classification_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id           INTEGER NOT NULL REFERENCES Images(image_id),
            model_id           INTEGER REFERENCES ModelVersions(model_id),
            room_type          TEXT NOT NULL,
            confidence_score   REAL NOT NULL,
            all_scores         TEXT,
            classification_date TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS VisualFeatures (
            feature_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id           INTEGER NOT NULL REFERENCES Images(image_id),
            model_id           INTEGER REFERENCES ModelVersions(model_id),
            flooring_type      TEXT,
            flooring_conf      REAL,
            lighting_quality   TEXT,
            lighting_conf      REAL,
            window_present     TEXT CHECK(window_present IN ('Present','Not Present')),
            window_conf        REAL,
            property_condition TEXT,
            condition_conf     REAL,
            feature_date       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS StructuredOutputs (
            output_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id       INTEGER NOT NULL REFERENCES Images(image_id),
            output_format  TEXT CHECK(output_format IN ('JSON','CSV')) NOT NULL,
            file_path      TEXT,
            generated_date TEXT DEFAULT (datetime('now')),
            export_status  TEXT DEFAULT 'ready'
        );

        CREATE TABLE IF NOT EXISTS ModelVersions (
            model_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name     TEXT NOT NULL,
            version_number TEXT NOT NULL,
            framework      TEXT CHECK(framework IN ('TensorFlow','PyTorch')),
            accuracy_score REAL,
            deployed_date  TEXT DEFAULT (datetime('now')),
            is_active      INTEGER DEFAULT 0
        );
    """)

    cur.execute("SELECT COUNT(*) FROM ModelVersions")
    if cur.fetchone()[0] == 0:
        cur.execute("""
            INSERT INTO ModelVersions (model_name, version_number, framework, accuracy_score, is_active)
            VALUES ('ResNet-50 Room Classifier', 'v1.0.0', 'TensorFlow', NULL, 1)
        """)

    conn.commit()
    conn.close()
