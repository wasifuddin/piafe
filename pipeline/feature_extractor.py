import numpy as np
import cv2
from PIL import Image


def extract(pil_image: Image.Image) -> dict:
    img_rgb = np.array(pil_image, dtype=np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return {
        "flooring":  _detect_flooring(img_gray, img_rgb),
        "lighting":  _detect_lighting(img_bgr),
        "window":    _detect_window(img_gray),
        "condition": _detect_condition(img_gray, img_bgr),
    }


def _detect_flooring(gray: np.ndarray, rgb: np.ndarray) -> dict:
    h = gray.shape[0]
    floor_region = gray[h * 2 // 3:, :]
    lap_var = float(cv2.Laplacian(floor_region, cv2.CV_64F).var())
    floor_hsv = cv2.cvtColor(rgb[h * 2 // 3:, :], cv2.COLOR_RGB2HSV)
    mean_sat = float(np.mean(floor_hsv[:, :, 1]))

    if lap_var > 400:
        value, conf = "Tile", min(0.65 + lap_var / 4000, 0.95)
    elif lap_var > 120:
        value, conf = "Hardwood", min(0.60 + lap_var / 2000, 0.92)
    elif mean_sat < 30:
        value, conf = "Concrete", 0.70
    else:
        value, conf = "Carpet", min(0.55 + mean_sat / 300, 0.88)

    return {"value": value, "confidence": round(conf, 2)}


def _detect_lighting(bgr: np.ndarray) -> dict:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(float)
    mean_l = float(np.mean(l_channel))
    std_l  = float(np.std(l_channel))

    if mean_l > 140 and std_l > 35:
        value, conf = "Natural", min(0.60 + std_l / 200, 0.95)
    elif mean_l > 100:
        value, conf = "Good", min(0.55 + mean_l / 500, 0.90)
    elif mean_l > 60:
        value, conf = "Artificial", min(0.55 + (100 - mean_l) / 200, 0.85)
    else:
        value, conf = "Poor", min(0.55 + (60 - mean_l) / 120, 0.88)

    return {"value": value, "confidence": round(conf, 2)}


def _detect_window(gray: np.ndarray) -> dict:
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    min_window_area = (h * w) * 0.01

    window_found = False
    best_conf = 0.60

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_window_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / (bh + 1e-6)
        if 0.3 < aspect < 3.5 and y < h * 0.75:
            fill_ratio = area / (bw * bh + 1e-6)
            if fill_ratio > 0.35:
                window_found = True
                conf = min(0.60 + fill_ratio * 0.5 + area / (h * w), 0.96)
                best_conf = max(best_conf, conf)

    if window_found:
        return {"value": "Present", "confidence": round(best_conf, 2)}
    else:
        return {"value": "Not Present", "confidence": round(min(best_conf, 0.82), 2)}


def _detect_condition(gray: np.ndarray, bgr: np.ndarray) -> dict:
    lap_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv      = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))

    score = (lap_var / 800 * 0.6) + (mean_sat / 128 * 0.4)
    score = min(score, 1.0)

    if score > 0.75:
        value, conf = "New",  min(0.65 + score * 0.3, 0.95)
    elif score > 0.50:
        value, conf = "Good", min(0.60 + score * 0.3, 0.92)
    elif score > 0.25:
        value, conf = "Fair", min(0.55 + score * 0.3, 0.88)
    else:
        value, conf = "Poor", min(0.50 + score * 0.4, 0.85)

    return {"value": value, "confidence": round(conf, 2)}
