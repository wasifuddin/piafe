import numpy as np
import cv2
from PIL import Image

TARGET_SIZE = (224, 224)
DARK_THRESHOLD = 50
BRIGHT_THRESHOLD = 210


def preprocess(file_path: str) -> tuple[np.ndarray, Image.Image]:
    img_bgr = cv2.imread(file_path)
    if img_bgr is None:
        raise ValueError(f"OpenCV could not load: {file_path}")

    img_bgr = _correct_brightness(img_bgr)
    img_bgr = cv2.resize(img_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    array = img_rgb.astype(np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return array, pil_image


def _correct_brightness(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mean_brightness = float(np.mean(l_channel))

    if mean_brightness < DARK_THRESHOLD or mean_brightness > BRIGHT_THRESHOLD:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img_bgr
