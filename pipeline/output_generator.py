import json
import csv
import os
import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def generate(image_id: int, file_name: str,
             classification: dict, features: dict) -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = {
        "image_id":    image_id,
        "file_name":   file_name,
        "analysed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "room_type":   classification["room_type"],
        "room_confidence": round(classification["confidence"] * 100, 1),
        "all_room_scores": {
            k: round(v * 100, 1) for k, v in classification["all_scores"].items()
        },
        "features": {
            "flooring_type":      features["flooring"]["value"],
            "flooring_conf":      round(features["flooring"]["confidence"] * 100, 1),
            "lighting_quality":   features["lighting"]["value"],
            "lighting_conf":      round(features["lighting"]["confidence"] * 100, 1),
            "window_present":     features["window"]["value"],
            "window_conf":        round(features["window"]["confidence"] * 100, 1),
            "property_condition": features["condition"]["value"],
            "condition_conf":     round(features["condition"]["confidence"] * 100, 1),
        },
    }
    return result


def to_json(result: dict, image_id: int) -> str:
    path = os.path.join(OUTPUT_DIR, f"piafe_{image_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def to_csv(result: dict, image_id: int) -> str:
    path = os.path.join(OUTPUT_DIR, f"piafe_{image_id}.csv")
    row = {
        "image_id":          result["image_id"],
        "file_name":         result["file_name"],
        "analysed_at":       result["analysed_at"],
        "room_type":         result["room_type"],
        "room_confidence":   result["room_confidence"],
        **{f"score_{k.lower().replace(' ', '_')}": v
           for k, v in result["all_room_scores"].items()},
        "flooring_type":     result["features"]["flooring_type"],
        "flooring_conf":     result["features"]["flooring_conf"],
        "lighting_quality":  result["features"]["lighting_quality"],
        "lighting_conf":     result["features"]["lighting_conf"],
        "window_present":    result["features"]["window_present"],
        "window_conf":       result["features"]["window_conf"],
        "property_condition":result["features"]["property_condition"],
        "condition_conf":    result["features"]["condition_conf"],
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path
