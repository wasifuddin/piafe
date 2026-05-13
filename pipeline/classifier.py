import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import time

ROOM_LABELS = ["Bathroom", "Bedroom", "Kitchen", "Living Room", "Outdoor Space"]
MAX_RETRIES = 3

ROOM_KEYWORDS = {
    "Bathroom": [
        "bathtub", "shower", "toilet", "washbasin", "medicine_cabinet",
        "soap_dispenser", "tub", "bath_towel", "shower_curtain",
        "plunger", "wash_basin",
    ],
    "Bedroom": [
        "four-poster", "bed", "pillow", "quilt", "studio_couch",
        "cradle", "crib", "bedroom", "wardrobe", "sleeping_bag",
        "lampshade", "table_lamp", "alarm_clock",
    ],
    "Kitchen": [
        "refrigerator", "microwave", "stove", "toaster", "dishwasher",
        "mixing_bowl", "frying_pan", "wok", "spatula", "pot",
        "coffeepot", "espresso_maker", "cup", "pitcher", "caldron",
        "crock_pot", "dining_table", "plate", "colander", "strainer",
        "measuring_cup", "ladle",
    ],
    "Living Room": [
        "television", "monitor", "screen", "entertainment_center",
        "couch", "sofa", "bookcase", "rocking_chair", "chair",
        "desk", "window_shade", "remote_control", "vase",
        "coffee_table", "cushion", "throw_pillow", "library",
        "home_theater",
    ],
    "Outdoor Space": [
        "patio", "garden", "lawn_mower", "fountain", "park_bench",
        "picket_fence", "swimming_pool", "greenhouse", "lakeside",
        "flower_pot", "lawn", "gazebo", "sundial", "hammock",
        "barbecue", "mailbox", "birdhouse", "chainlink_fence",
    ],
}

_model = None


def _get_model():
    global _model
    if _model is None:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                _model = ResNet50(weights="imagenet", include_top=True,
                                  input_shape=(224, 224, 3))
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to download ResNet-50 weights after {MAX_RETRIES} attempts. "
                        f"Check your internet connection.\nLast error: {e}"
                    )
                time.sleep(3)
    return _model


def classify(array: np.ndarray) -> dict:
    model = _get_model()
    array_255 = (array * 255.0).astype(np.float32)
    array_preprocessed = preprocess_input(array_255.copy())

    preds = model.predict(array_preprocessed, verbose=0)
    decoded = decode_predictions(preds, top=20)[0]

    room_scores = {label: 0.0 for label in ROOM_LABELS}

    for (_, class_name, score) in decoded:
        class_lower = class_name.lower().replace(" ", "_")
        for room, keywords in ROOM_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in class_lower or class_lower in kw.lower():
                    room_scores[room] += float(score)
                    break

    total = sum(room_scores.values())
    if total > 0:
        room_scores = {k: v / total for k, v in room_scores.items()}
    else:
        room_scores = {k: 1.0 / len(ROOM_LABELS) for k in ROOM_LABELS}

    best_room = max(room_scores, key=room_scores.get)
    best_conf = room_scores[best_room]

    if best_conf < 0.05:
        best_room = "Living Room"
        room_scores["Living Room"] = max(room_scores["Living Room"], 0.25)
        total = sum(room_scores.values())
        room_scores = {k: v / total for k, v in room_scores.items()}
        best_conf = room_scores[best_room]

    return {
        "room_type":  best_room,
        "confidence": round(best_conf, 4),
        "all_scores": {k: round(v, 4) for k, v in room_scores.items()},
    }
