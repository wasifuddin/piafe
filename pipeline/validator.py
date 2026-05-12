import os
from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "TIFF"}
MIN_SIZE = 50


def validate(file_path: str) -> tuple[bool, str, dict]:
    if not os.path.isfile(file_path):
        return False, f"File not found: {file_path}", {}

    try:
        img = Image.open(file_path)
        img.verify()
        img = Image.open(file_path)
        img.load()
    except (UnidentifiedImageError, Exception) as e:
        return False, f"File is corrupted or not a valid image: {e}", {}

    fmt = (img.format or "").upper()
    if fmt not in SUPPORTED_FORMATS:
        return (False,
                f"Unsupported format '{fmt}'. Accepted: {', '.join(SUPPORTED_FORMATS)}",
                {})

    w, h = img.size
    if w < MIN_SIZE or h < MIN_SIZE:
        return (False,
                f"Image too small ({w}x{h}px). Minimum is {MIN_SIZE}x{MIN_SIZE}px.",
                {})

    metadata = {
        "format":   fmt,
        "width":    w,
        "height":   h,
        "mode":     img.mode,
        "file_size_kb": round(os.path.getsize(file_path) / 1024, 1),
    }
    return True, "Validation passed.", metadata
