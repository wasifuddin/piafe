import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, Model

ROOM_LABELS = ["Bathroom", "Bedroom", "Kitchen", "Living Room", "Outdoor Space"]
MAX_RETRIES = 3

_model: Model | None = None


def _build_model() -> Model:
    base = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            base = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                            input_shape=(224, 224, 3))
            break
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to download ResNet-50 weights after {MAX_RETRIES} attempts. "
                    f"Please check your internet connection and try again.\n"
                    f"Last error: {e}"
                )
            time.sleep(3)

    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(ROOM_LABELS), activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


def _get_model() -> Model:
    global _model
    if _model is None:
        _model = _build_model()
    return _model


def classify(array: np.ndarray) -> dict:
    model = _get_model()
    array_255 = (array * 255.0).astype(np.float32)
    preds = model.predict(array_255, verbose=0)[0]
    idx = int(np.argmax(preds))

    return {
        "room_type":  ROOM_LABELS[idx],
        "confidence": float(preds[idx]),
        "all_scores": {label: float(score) for label, score in zip(ROOM_LABELS, preds)},
    }
