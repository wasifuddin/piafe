import json
from .db import get_connection


def save_image(file_path: str, file_format: str, width: int, height: int,
               property_id: int = None) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO Images (property_id, file_path, file_format, width, height, status)
           VALUES (?, ?, ?, ?, ?, 'queued')""",
        (property_id, file_path, file_format, width, height)
    )
    image_id = cur.lastrowid
    conn.commit()
    conn.close()
    return image_id


def update_image_status(image_id: int, status: str):
    conn = get_connection()
    conn.execute("UPDATE Images SET status=? WHERE image_id=?", (status, image_id))
    conn.commit()
    conn.close()


def get_recent_images(limit: int = 20):
    conn = get_connection()
    rows = conn.execute(
        """SELECT i.*, rc.room_type, rc.confidence_score,
                  vf.flooring_type, vf.lighting_quality, vf.window_present, vf.property_condition
           FROM Images i
           LEFT JOIN RoomClassifications rc ON rc.image_id = i.image_id
           LEFT JOIN VisualFeatures vf ON vf.image_id = i.image_id
           ORDER BY i.upload_date DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_classification(image_id: int, room_type: str, confidence: float,
                        all_scores: dict, model_id: int = 1):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO RoomClassifications
           (image_id, model_id, room_type, confidence_score, all_scores)
           VALUES (?, ?, ?, ?, ?)""",
        (image_id, model_id, room_type, confidence, json.dumps(all_scores))
    )
    conn.commit()
    conn.close()


def save_features(image_id: int, features: dict, model_id: int = 1):
    conn = get_connection()
    conn.execute(
        """INSERT INTO VisualFeatures
           (image_id, model_id,
            flooring_type, flooring_conf,
            lighting_quality, lighting_conf,
            window_present, window_conf,
            property_condition, condition_conf)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            image_id, model_id,
            features["flooring"]["value"],   features["flooring"]["confidence"],
            features["lighting"]["value"],   features["lighting"]["confidence"],
            features["window"]["value"],     features["window"]["confidence"],
            features["condition"]["value"],  features["condition"]["confidence"],
        )
    )
    conn.commit()
    conn.close()


def save_output(image_id: int, output_format: str, file_path: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO StructuredOutputs (image_id, output_format, file_path) VALUES (?,?,?)",
        (image_id, output_format, file_path)
    )
    conn.commit()
    conn.close()


def get_stats():
    conn = get_connection()
    stats = {}
    stats["total_images"] = conn.execute(
        "SELECT COUNT(*) FROM Images WHERE status='complete'"
    ).fetchone()[0]
    stats["avg_confidence"] = conn.execute(
        "SELECT ROUND(AVG(confidence_score)*100,1) FROM RoomClassifications"
    ).fetchone()[0] or 0.0
    stats["total_outputs"] = conn.execute(
        "SELECT COUNT(*) FROM StructuredOutputs"
    ).fetchone()[0]
    dist = conn.execute(
        "SELECT room_type, COUNT(*) as cnt FROM RoomClassifications GROUP BY room_type"
    ).fetchall()
    stats["room_distribution"] = {r["room_type"]: r["cnt"] for r in dist}
    conn.close()
    return stats


def get_active_model():
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM ModelVersions WHERE is_active=1 ORDER BY model_id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else {}
