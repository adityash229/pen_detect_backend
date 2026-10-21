"""Production-ready pen detection pipeline using YOLO and OpenCV.

This module accepts an image source, prepares it for inference, optionally
enhances blurry images, runs tiled YOLO inference, remaps detections to the
original image, crops each detected pen, annotates the image, and returns a
strict JSON-compatible dictionary.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import threading
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError

if TYPE_CHECKING:
    from ultralytics import YOLO


warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)


MODEL_PATH = Path("best.pt")
CLASS_NAME = "pen"
BLUR_THRESHOLD = 80.0
TILE_SIZE = 640
TILE_OVERLAP = 0.20
TILE_STRIDE = int(TILE_SIZE * (1 - TILE_OVERLAP))
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.45
CROP_PADDING = 0.15
CROP_MAX_DIM = 224

_MODEL: "YOLO | None" = None
_MODEL_PATH: Path | None = None
_MODEL_LOCK = threading.Lock()

_ORIENTATION_TAG = next(
    tag for tag, name in ExifTags.TAGS.items() if name == "Orientation"
)
_ROTATION_BY_ORIENTATION = {3: 180, 6: 270, 8: 90}


Detection = dict[str, float]
RawDetection = dict[str, float]
Tile = tuple[np.ndarray, int, int, float, int, int]


def _ensure_cv2_ultralytics_compat() -> None:
    """Patch minimal cv2 GUI symbols required during Ultralytics import."""
    constants = {
        "IMREAD_COLOR": 1,
        "IMREAD_GRAYSCALE": 0,
        "IMREAD_UNCHANGED": -1,
        "IMWRITE_JPEG_QUALITY": 1,
        "COLOR_RGB2BGR": 4,
        "COLOR_RGBA2BGR": 10,
        "COLOR_GRAY2BGR": 11,
        "COLOR_BGR2GRAY": 6,
        "COLOR_BGR2LAB": 44,
        "COLOR_LAB2BGR": 56,
        "INTER_LINEAR": 1,
        "INTER_AREA": 3,
        "LINE_AA": 16,
        "FONT_HERSHEY_SIMPLEX": 0,
        "CV_64F": 6,
    }
    for name, value in constants.items():
        if not hasattr(cv2, name):
            setattr(cv2, name, value)

    if not hasattr(cv2, "imread"):
        def _imread(filename: str, flags: int = cv2.IMREAD_COLOR):
            try:
                with Image.open(filename) as pil_image:
                    if flags == cv2.IMREAD_GRAYSCALE:
                        return np.array(pil_image.convert("L"), dtype=np.uint8)
                    if flags == cv2.IMREAD_UNCHANGED and pil_image.mode == "RGBA":
                        return np.array(pil_image.convert("RGBA"), dtype=np.uint8)[:, :, [2, 1, 0, 3]]
                    return np.array(pil_image.convert("RGB"), dtype=np.uint8)[:, :, ::-1]
            except Exception:
                return None

        cv2.imread = _imread  # type: ignore[attr-defined]

    if not hasattr(cv2, "imwrite"):
        def _imwrite(filename: str, image: np.ndarray, params: list[int] | None = None) -> bool:
            try:
                ext = Path(filename).suffix.lower() or ".jpg"
                fmt = "JPEG" if ext in {".jpg", ".jpeg"} else ext.lstrip(".").upper()
                rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] >= 3 else image
                Image.fromarray(rgb).save(filename, format=fmt)
                return True
            except Exception:
                return False

        cv2.imwrite = _imwrite  # type: ignore[attr-defined]

    if not hasattr(cv2, "imshow"):
        cv2.imshow = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    if not hasattr(cv2, "waitKey"):
        cv2.waitKey = lambda delay=0: -1  # type: ignore[attr-defined]
    if not hasattr(cv2, "destroyAllWindows"):
        cv2.destroyAllWindows = lambda: None  # type: ignore[attr-defined]
    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda *args, **kwargs: None  # type: ignore[attr-defined]


def _import_yolo():
    """Import YOLO only after cv2 has the attributes Ultralytics expects."""
    _ensure_cv2_ultralytics_compat()
    from ultralytics import YOLO

    return YOLO


def load_model(model_path: str | Path = MODEL_PATH) -> "YOLO":
    """Load a YOLO model once using double-checked locking."""
    global _MODEL, _MODEL_PATH

    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Model file not found: {resolved_path}")

    if _MODEL is not None and _MODEL_PATH == resolved_path:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_PATH != resolved_path:
            YOLO = _import_yolo()
            _MODEL = YOLO(str(resolved_path))
            _MODEL_PATH = resolved_path
        return _MODEL


def warmup() -> None:
    """Run one dummy inference so model initialization cost is paid up front."""
    model = load_model(MODEL_PATH)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    print("[pen_detector] warmup complete")


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image to a uint8 BGR OpenCV array."""
    image = _fix_exif_orientation(image)

    if image.mode == "RGBA":
        rgba = np.array(image, dtype=np.uint8)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    if image.mode in {"L", "I;16", "I", "F"}:
        gray = np.array(image.convert("L"), dtype=np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _fix_exif_orientation(image: Image.Image) -> Image.Image:
    """Rotate a Pillow image according to the EXIF Orientation tag."""
    try:
        orientation = image.getexif().get(_ORIENTATION_TAG)
    except Exception:
        orientation = None

    angle = _ROTATION_BY_ORIENTATION.get(int(orientation or 0))
    if angle is None:
        return image

    return image.rotate(angle, expand=True)


def _decode_base64_payload(source: str) -> bytes | None:
    """Decode a plain base64 string or data URL, returning None on failure."""
    payload = source.strip()
    if not payload:
        return None
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception:
        return None


def preprocess_image(source: str | Path | bytes | np.ndarray) -> np.ndarray:
    """Decode input into a validated uint8 BGR image."""
    image: np.ndarray | None = None

    if isinstance(source, np.ndarray):
        image = source
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Unsupported numpy image shape: {image.shape}")

    elif isinstance(source, bytes):
        try:
            with Image.open(BytesIO(source)) as pil_image:
                image = _pil_to_bgr(pil_image)
        except (UnidentifiedImageError, OSError, ValueError):
            buffer = np.frombuffer(source, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
            if image is not None and image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image is not None and image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    elif isinstance(source, (str, Path)):
        path: Path | None = None
        if isinstance(source, str) and (source.strip().startswith("data:") or len(source) > 260):
            raw = _decode_base64_payload(source)
            if raw is not None:
                image = preprocess_image(raw)
            else:
                raise ValueError("Input string is neither a valid file path nor base64 image data.")
        else:
            path = Path(source).expanduser()

        if image is None and path is not None and path.is_file():
            with Image.open(path) as pil_image:
                image = _pil_to_bgr(pil_image)
        elif image is None and isinstance(source, str):
            raw = _decode_base64_payload(source)
            if raw is not None:
                image = preprocess_image(raw)
            else:
                raise ValueError("Input string is neither a valid file path nor base64 image data.")
        elif image is None:
            raise ValueError(f"Image path does not exist: {path}")

    else:
        raise TypeError(f"Unsupported image source type: {type(source).__name__}")

    if image is None or image.size == 0:
        raise ValueError("Decoded image is empty.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def compute_blur_score(image: np.ndarray) -> float:
    """Compute variance of Laplacian as a focus/blur score."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhance blurry images using unsharp mask, LAB CLAHE, and denoising."""
    blur_score = compute_blur_score(image)
    if blur_score >= BLUR_THRESHOLD:
        return image

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(
        cv2.merge([l_enhanced, a_channel, b_channel]),
        cv2.COLOR_LAB2BGR,
    )

    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return np.ascontiguousarray(denoised, dtype=np.uint8)


def _letterbox(img: np.ndarray, new_shape: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with unchanged aspect ratio and pad to a square canvas."""
    height, width = img.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Cannot letterbox an empty image.")

    scale = min(new_shape / height, new_shape / width)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (new_shape - resized_w) // 2
    pad_y = (new_shape - resized_h) // 2
    canvas = np.full((new_shape, new_shape, 3), (114, 114, 114), dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    return canvas, float(scale), (int(pad_x), int(pad_y))


def tile_image(image: np.ndarray) -> list[Tile]:
    """Split an image into overlapping letterboxed inference tiles."""
    height, width = image.shape[:2]
    if height <= TILE_SIZE and width <= TILE_SIZE:
        tile, scale, (pad_x, pad_y) = _letterbox(image, TILE_SIZE)
        return [(tile, 0, 0, scale, pad_x, pad_y)]

    x_starts = list(range(0, max(width - TILE_SIZE, 0) + 1, TILE_STRIDE))
    y_starts = list(range(0, max(height - TILE_SIZE, 0) + 1, TILE_STRIDE))
    if not x_starts or x_starts[-1] != max(width - TILE_SIZE, 0):
        x_starts.append(max(width - TILE_SIZE, 0))
    if not y_starts or y_starts[-1] != max(height - TILE_SIZE, 0):
        y_starts.append(max(height - TILE_SIZE, 0))

    tiles: list[Tile] = []
    for ty in y_starts:
        for tx in x_starts:
            patch = image[ty:min(ty + TILE_SIZE, height), tx:min(tx + TILE_SIZE, width)]
            tile, scale, (pad_x, pad_y) = _letterbox(patch, TILE_SIZE)
            tiles.append((tile, int(tx), int(ty), scale, pad_x, pad_y))
    return tiles


def run_inference(tiles: list[Tile]) -> list[RawDetection]:
    """Run YOLO on each tile and return raw tile-space pen detections."""
    model = load_model(MODEL_PATH)
    raw_detections: list[RawDetection] = []

    for tile, tx, ty, scale, pad_x, pad_y in tiles:
        results = model.predict(
            tile,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            device="cpu",
        )
        if not results or results[0].boxes is None:
            continue

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = str(model.names[int(cls_id)])
            if cls_name != CLASS_NAME:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            raw_detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": float(box.conf[0]),
                    "tx": float(tx),
                    "ty": float(ty),
                    "scale": float(scale),
                    "pad_x": float(pad_x),
                    "pad_y": float(pad_y),
                }
            )

    return raw_detections


def remap_coordinates(raw: list[RawDetection], orig_shape: tuple[int, ...]) -> list[Detection]:
    """Map tile-space detections back into original image pixel coordinates."""
    image_h, image_w = orig_shape[:2]
    detections: list[Detection] = []

    for item in raw:
        scale = float(item["scale"])
        if scale <= 0:
            continue

        x1 = float(item["tx"]) + (float(item["x1"]) - float(item["pad_x"])) / scale
        y1 = float(item["ty"]) + (float(item["y1"]) - float(item["pad_y"])) / scale
        x2 = float(item["tx"]) + (float(item["x2"]) - float(item["pad_x"])) / scale
        y2 = float(item["ty"]) + (float(item["y2"]) - float(item["pad_y"])) / scale

        x1 = max(0.0, min(float(image_w), x1))
        y1 = max(0.0, min(float(image_h), y1))
        x2 = max(0.0, min(float(image_w), x2))
        y2 = max(0.0, min(float(image_h), y2))

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": float(item["conf"]),
            }
        )

    return detections


def _iou(a: Detection, b: Detection) -> float:
    """Compute intersection-over-union for two detection dictionaries."""
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def nms(detections: list[Detection], threshold: float = IOU_THRESHOLD) -> list[Detection]:
    """Greedily suppress overlapping detections while preserving separate pens."""
    selected: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item["conf"], reverse=True):
        if all(_iou(detection, kept) <= threshold for kept in selected):
            selected.append(detection)
    return selected


def _encode_jpeg_b64(image: np.ndarray, quality: int = 92) -> str:
    """Encode a BGR image as base64 JPEG."""
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Could not encode image as JPEG.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def crop_pens(image: np.ndarray, detections: list[Detection]) -> list[dict[str, Any]]:
    """Crop, resize, encode, and describe each detected pen."""
    image_h, image_w = image.shape[:2]
    crops: list[dict[str, Any]] = []

    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        pad_x = (x2 - x1) * CROP_PADDING
        pad_y = (y2 - y1) * CROP_PADDING

        cx1 = int(max(0, np.floor(x1 - pad_x)))
        cy1 = int(max(0, np.floor(y1 - pad_y)))
        cx2 = int(min(image_w, np.ceil(x2 + pad_x)))
        cy2 = int(min(image_h, np.ceil(y2 + pad_y)))

        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        max_dim = max(crop_h, crop_w)
        scale_down = CROP_MAX_DIM / max_dim if max_dim > CROP_MAX_DIM else 1.0
        if scale_down < 1.0:
            new_w = max(1, int(round(crop_w * scale_down)))
            new_h = max(1, int(round(crop_h * scale_down)))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        crops.append(
            {
                "id": index,
                "confidence": round(float(detection["conf"]), 4),
                "box": [
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                ],
                "crop_image": _encode_jpeg_b64(crop, quality=92),
                "label": CLASS_NAME,
            }
        )

    return crops


def annotate_image(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw green boxes and confidence labels on a BGR image copy."""
    annotated = image.copy()
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for detection in detections:
        x1 = int(round(detection["x1"]))
        y1 = int(round(detection["y1"]))
        x2 = int(round(detection["x2"]))
        y2 = int(round(detection["y2"]))
        label = f"{CLASS_NAME} {int(round(float(detection['conf']) * 100))}%"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(label, font, 0.55, 1)
        label_y1 = max(0, y1 - text_h - baseline - 6)
        label_y2 = max(text_h + baseline + 6, y1)
        label_x2 = min(annotated.shape[1], x1 + text_w + 8)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y2 - baseline - 3),
            font,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def detect_pens(source: str | Path | bytes | np.ndarray, model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Run the complete pen detection pipeline and return strict JSON data."""
    try:
        global MODEL_PATH
        MODEL_PATH = Path(model_path)

        image = preprocess_image(source)
        blur_score = compute_blur_score(image)
        was_enhanced = blur_score < BLUR_THRESHOLD
        working_image = enhance_image(image) if was_enhanced else image

        tiles = tile_image(working_image)
        raw = run_inference(tiles)
        remapped = remap_coordinates(raw, working_image.shape)
        detections = nms(remapped, IOU_THRESHOLD)
        crops = crop_pens(image, detections)
        annotated = annotate_image(image, detections)

        top_confidence = max((float(item["conf"]) for item in detections), default=0.0)
        return {
            "pen_detected": len(detections) > 0,
            "pen_count": len(detections),
            "confidence": round(top_confidence, 4),
            "blur_score": round(float(blur_score), 4),
            "was_enhanced": was_enhanced,
            "annotated_image": _encode_jpeg_b64(annotated, quality=92),
            "pen_crops": crops,
        }
    except Exception as exc:
        return {
            "pen_detected": False,
            "pen_count": 0,
            "confidence": 0.0,
            "blur_score": 0.0,
            "was_enhanced": False,
            "annotated_image": "",
            "pen_crops": [],
            "error": str(exc),
        }


def _strip_base64_for_display(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a CLI-friendly copy of the result without large base64 strings."""
    display = dict(payload)
    if display.get("annotated_image"):
        display["annotated_image"] = "<base64 omitted>"

    display_crops: list[dict[str, Any]] = []
    for crop in display.get("pen_crops", []):
        crop_display = dict(crop)
        if crop_display.get("crop_image"):
            crop_display["crop_image"] = "<base64 omitted>"
        display_crops.append(crop_display)
    display["pen_crops"] = display_crops
    return display


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for local testing."""
    parser = argparse.ArgumentParser(description="Detect pens in an image.")
    parser.add_argument("image_path", help="Path to the image to analyze.")
    parser.add_argument("model_path", nargs="?", default=str(MODEL_PATH), help="Optional YOLO model path.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    result = detect_pens(args.image_path, args.model_path)
    print(json.dumps(_strip_base64_for_display(result), indent=2))
