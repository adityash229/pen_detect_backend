
from pathlib import Path
import datetime
import io
import json
import os
import random
import re
import shutil
import tempfile
import threading
import warnings

import cv2
import numpy as np
from opencv_compat import ensure_cv2_compat

ensure_cv2_compat()

from ultralytics import YOLO
import mediapipe as mp

if not hasattr(np, "trapz"):  # RETRAIN-FIX
    np.trapz = np.trapezoid  # RETRAIN-FIX

try:
    from PIL import Image as PILImage
except Exception:  # ACCURACY
    PILImage = None

from config import (
    AGNOSTIC_NMS,
    CONFIDENCE_THRESHOLD,
    HIGH_CONF,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
    MIN_ASPECT_RATIO,
    MIN_BOX_AREA_RATIO,
    MODEL_PATH,
    PEN_CLASS_ID,
    POSSIBLE_LOW,
    TILE_NMS_IOU,
    UNCERTAIN_LOW,
)
from tiling_utils import apply_nms, split_image, tile_to_image_coords


CONF_THRESHOLD        = 0.25  # DETECTION-FIX: catch lower-confidence real pens.
PEN_MIN_ASPECT_RATIO  = 1.1   # DATASET
PEN_MAX_ASPECT_RATIO  = 20.0  # DATASET
MERGE_IOU_THRESHOLD   = 0.55  # DETECTION-FIX: avoid suppressing nearby/overlapping pens.
YOLO_IOU_THRESHOLD    = 0.55  # DETECTION-FIX: keep more overlapping YOLO boxes before custom NMS.
MAX_PENS              = 100   # DETECTION-FIX: return all pens, not just the first few.

ZOOM_SCALE            = 3
ZOOM_CONF_THRESHOLD   = 0.25  # DETECTION-FIX: zoom fallback should match the global pen threshold.
ZOOM_PADDING          = 0.30

# ACCURACY
HAND_DETECTION_CONF   = 0.70
HAND_TRACKING_CONF    = 0.60
HAND_CROP_PADDING     = 0.30
FULL_PASS_SKIP_CONF   = HIGH_CONF
ZOOM_EARLY_EXIT_CONF  = 0.65
ANGLE_MIN             = 0    # DETECTION-FIX: full valid range; never used to reject pens.
ANGLE_MAX             = 90   # DETECTION-FIX: full valid range; never used to reject pens.
IDEAL_ANGLE_MIN       = 10   # DETECTION-FIX: optional small boost only.
IDEAL_ANGLE_MAX       = 80   # DETECTION-FIX: optional small boost only.
IDEAL_ANGLE_BOOST     = 0.02
MIN_HAND_AREA_FRAC    = 0.02
LETTERBOX_SIZE        = 640
LETTERBOX_PAD_VALUE   = 114

# SCENE
LAP_VAR_THRESHOLD     = 25  # BLUR-FIX
STD_DEV_THRESHOLD     = 4  # BUG-02 FIX: allow low-contrast wooden/dark surfaces.
TIER2_CONF_UPPER      = 0.28  # BUG-01 FIX: trust YOLO tier-2 pen boxes near 0.25+.
TIER2_CONF_DEFAULT    = 0.28  # BUG-01 FIX: avoid rejecting short/diagonal transparent pens.
OCCLUDE_EXTEND_FRAC   = 0.35
WRITING_BOOST         = 0.08
WRITING_PINCH_RATIO   = 0.35

# BLUR
BLUR_CONF_SEVERE      = 0.25
BLUR_CONF_MODERATE    = 0.25
BLUR_ZOOM_SEVERE      = 0.25
BLUR_ZOOM_MODERATE    = 0.25
BLUR_TIER1_THRESH     = 2.0
BLUR_TIER2_MIN        = 1.4
BLUR_EDGE_DENSITY     = 0.02

# DATASET
TIER0_CONF_THRESHOLD  = 0.75
BACKGROUND_SKIP_CONF  = 0.25  # DETECTION-FIX: trust YOLO at/above the active threshold.
DATASET_DIR           = "dataset/pen_yolo"
AUGMENT_VARIANTS      = ["flip", "rot90", "bright", "blur", "crop"]


_pen_model = None
_general_model = None
_pen_model_lock = threading.Lock()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH_FILE = BASE_DIR / "model_path.txt"
DEFAULT_PEN_MODEL_PATHS = [
    BASE_DIR / "runs" / "detect" / "pen_retrain" / "weights" / "best.pt",
]  # MODEL-FIX


def _resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        if MODEL_PATH_FILE.is_file():
            saved_path = Path(MODEL_PATH_FILE.read_text(encoding="utf-8").strip())
            if saved_path.exists():
                path = saved_path
        requested_name = Path(model_path).name.lower()
        for candidate in DEFAULT_PEN_MODEL_PATHS:
            if candidate.exists() and (
                requested_name in {"best.pt", "last.pt"}
                or "pen" in requested_name
                or "model" in requested_name
            ):
                path = candidate
                break
    return str(path)


def _write_active_model_path(best_pt: Path) -> None:
    MODEL_PATH_FILE.write_text(str(best_pt.resolve()), encoding="utf-8")


def reload_pen_model(best_pt: str | Path):
    """Replace the cached pen detector with freshly trained weights."""
    global _pen_model
    model_path = Path(best_pt)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained weights not found at: {model_path}")

    with _pen_model_lock:
        updated_model = YOLO(str(model_path))
        updated_model.overrides["conf"] = CONF_THRESHOLD
        updated_model.overrides["iou"] = YOLO_IOU_THRESHOLD
        _write_active_model_path(model_path)
        _pen_model = updated_model

    print(f"[model] Reloaded pen detector from {model_path}")
    return _pen_model


def load_model(model_path: str = MODEL_PATH):
    global _pen_model
    with _pen_model_lock:
        if _pen_model is None:
            resolved_model_path = _resolve_model_path(model_path)  # MODEL-FIX
            if not Path(resolved_model_path).exists():  # MODEL-FIX
                raise FileNotFoundError(f"Pen model not found at: {resolved_model_path}")  # MODEL-FIX
            _pen_model = YOLO(resolved_model_path)
            _pen_model.overrides["conf"] = CONF_THRESHOLD
            _pen_model.overrides["iou"]  = YOLO_IOU_THRESHOLD
    return _pen_model


def load_general_model(model_path: str = "yolov8n.pt"):
    global _general_model
    if _general_model is None:
        _general_model = YOLO(_resolve_model_path(model_path))
    return _general_model


def filter_false_positives(boxes, scores, image_shape):
    filtered = []
    image_h, image_w = image_shape[:2]
    image_area = image_h * image_w
    min_box_area = int(MIN_BOX_AREA_RATIO * image_area)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        width = max(0.0, float(x2) - float(x1))
        height = max(0.0, float(y2) - float(y1))
        area = width * height

        if area < min_box_area:
            continue
        if area > image_area * 0.60:
            continue
        if min(height, width) <= 0:
            continue

        aspect_ratio = max(height, width) / min(height, width)
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        filtered.append(([float(x1), float(y1), float(x2), float(y2)], float(score)))

    return filtered


def predict(model, image):
    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        agnostic_nms=AGNOSTIC_NMS,
        max_det=max(MAX_DETECTIONS, MAX_PENS),
        verbose=False,
    )

    boxes = []
    scores = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = str(model.names.get(cls_id, cls_id) if isinstance(model.names, dict) else model.names[cls_id])
            if cls_name.strip().lower() != "pen":
                continue
            boxes.append([float(value) for value in box.xyxy[0].tolist()])
            scores.append(float(box.conf[0]))

    return filter_false_positives(boxes, scores, image.shape)


def predict_tiled(model, image, tile_size=640, overlap=0.2):
    """
    Use this for large images (meeting hall, CCTV, drone).
    Use predict() for normal phone photos.
    """
    image_h, image_w = image.shape[:2]
    min_box_area = int(MIN_BOX_AREA_RATIO * image_h * image_w)
    tiles = split_image(image, tile_size, overlap)
    all_boxes = []
    all_scores = []

    for tile, x_offset, y_offset in tiles:
        results = model.predict(
            source=tile,
            conf=CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False,
        )

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = str(model.names.get(cls_id, cls_id) if isinstance(model.names, dict) else model.names[cls_id])
                if cls_name.strip().lower() != "pen":
                    continue

                score = float(box.conf[0])
                tile_box = [float(value) for value in box.xyxy[0].tolist()]
                image_box = tile_to_image_coords(tile_box, x_offset, y_offset, image.shape)
                x1, y1, x2, y2 = image_box
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                if min(width, height) <= 0:
                    continue

                if score < 0.60:
                    area = width * height
                    aspect_ratio = max(height, width) / (min(height, width) + 1e-6)
                    if area < min_box_area:
                        continue
                    if aspect_ratio < MIN_ASPECT_RATIO:
                        continue

                all_boxes.append(image_box)
                all_scores.append(score)

    final_boxes, final_scores = apply_nms(all_boxes, all_scores, iou_threshold=TILE_NMS_IOU)
    return list(zip(final_boxes, final_scores))


# ACCURACY
def _decode_image_input(image):
    """Decode bytes/arrays to BGR and apply EXIF orientation when available."""
    if image is None:
        return None

    if isinstance(image, (bytes, bytearray)):
        raw = bytes(image)
        if PILImage is not None:
            try:
                pil_img = PILImage.open(io.BytesIO(raw))
                orientation = pil_img.getexif().get(274)
                if orientation in {3: 180, 6: 270, 8: 90}:
                    pil_img = pil_img.rotate({3: 180, 6: 270, 8: 90}[orientation], expand=True)
                pil_img = pil_img.convert("RGB")
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                pass

        data = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    if not isinstance(image, np.ndarray):
        return None

    img = image
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) != 3 or img.shape[2] != 3:
        return None
    else:
        img = img.copy()

    if PILImage is not None:
        try:
            pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            orientation = pil_img.getexif().get(274)
            if orientation in {3: 180, 6: 270, 8: 90}:
                pil_img = pil_img.rotate({3: 180, 6: 270, 8: 90}[orientation], expand=True)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    return img


# ACCURACY
def _letterbox_image(image: np.ndarray, size: int = LETTERBOX_SIZE):
    img_h, img_w = image.shape[:2]
    scale = min(size / img_w, size / img_h)
    new_w = int(round(img_w * scale))
    new_h = int(round(img_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    boxed = np.full((size, size, 3), LETTERBOX_PAD_VALUE, dtype=resized.dtype)

    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    boxed[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return boxed, {
        "scale": float(scale),
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
    }


# BLUR
def preprocess_image(image: np.ndarray, blur_level="sharp") -> tuple:
    """Normalize, enhance, and letterbox image for robust pen detection."""
    img = _decode_image_input(image)
    if img is None:
        return None, {"scale": 1.0, "pad_top": 0, "pad_left": 0}

    if blur_level in ("severe", "moderate"):
        for _ in range(2 if blur_level == "severe" else 1):
            blur_pass = cv2.GaussianBlur(img, (0, 0), sigmaX=1)
            img = cv2.addWeighted(img, 1.6, blur_pass, -0.6, 0)
            img = np.clip(img, 0, 255).astype(np.uint8)

    sigma = 1.5 if blur_level in ("severe", "moderate") else 2.0
    strength = 1.6 if blur_level in ("severe", "moderate") else 1.4
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    img = cv2.addWeighted(img, strength, blur, -(strength - 1), 0)
    img = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clip = 3.0 if blur_level in ("severe", "moderate") else 1.5
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    img = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

    if blur_level in ("severe", "moderate"):
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    return _letterbox_image(img)


# ACCURACY
def _remap_bbox_from_letterbox(bbox, meta, original_shape):
    scale = max(float(meta.get("scale", 1.0)), 1e-6)
    pad_top = int(meta.get("pad_top", 0))
    pad_left = int(meta.get("pad_left", 0))
    img_h, img_w = original_shape[:2]

    x1, y1, x2, y2 = bbox
    ox1 = int(round((x1 - pad_left) / scale))
    oy1 = int(round((y1 - pad_top) / scale))
    ox2 = int(round((x2 - pad_left) / scale))
    oy2 = int(round((y2 - pad_top) / scale))

    return [
        max(0, min(img_w, ox1)),
        max(0, min(img_h, oy1)),
        max(0, min(img_w, ox2)),
        max(0, min(img_h, oy2)),
    ]


# BLUR
def check_image_quality(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_bright = float(np.mean(gray))
    std_dev = float(np.std(gray))

    if mean_bright < 35:
        return {"ok": False, "reason": "too_dark", "blur_level": "extreme", "score": mean_bright}
    if mean_bright > 225:
        return {"ok": False, "reason": "overexposed", "blur_level": "extreme", "score": mean_bright}
    if std_dev < STD_DEV_THRESHOLD:  # BUG-02 FIX: was hard-coded 8, too strict for dark tables.
        return {"ok": False, "reason": "low_contrast", "blur_level": "extreme", "score": std_dev}

    if lap_var < 12:  # BLUR-FIX
        blur_level = "severe"
    elif lap_var < 25:  # BLUR-FIX
        blur_level = "moderate"
    elif lap_var < 80:
        blur_level = "mild"
    else:
        blur_level = "sharp"

    return {"ok": True, "reason": None, "blur_level": blur_level, "score": lap_var}


# ACCURACY
def _compute_iou(boxA: list, boxB: list) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter_area + 1e-6

    return inter_area / union


# DATASET
def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# DATASET
def _parse_custom_label_json(label_json_path: str, img_w: int, img_h: int):
    with open(label_json_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    yolo_lines = []
    pixel_boxes = []
    scenarios = {"horizontal": 0, "vertical": 0, "low_ratio": 0}

    for ann in labels:
        rm = ann.get("rectMask", {})
        label_name = ann.get("labels", {}).get("labelName")
        if not rm or label_name != "Pen" or ann.get("contentType", "rect") != "rect":
            continue

        xmin = float(rm["xMin"])
        ymin = float(rm["yMin"])
        bw = float(rm["width"])
        bh = float(rm["height"])

        cx = _clamp01((xmin + bw / 2) / img_w)
        cy = _clamp01((ymin + bh / 2) / img_h)
        nw = _clamp01(bw / img_w)
        nh = _clamp01(bh / img_h)
        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        pixel_boxes.append([xmin, ymin, xmin + bw, ymin + bh])

        if bw > bh:
            scenarios["horizontal"] += 1
        elif bh > bw:
            scenarios["vertical"] += 1

        ratio = max(bw, bh) / max(min(bw, bh), 1.0)
        if ratio < 1.8:
            scenarios["low_ratio"] += 1

    return yolo_lines, pixel_boxes, scenarios


# DATASET
def _write_yolo_label(path: Path, lines: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# DATASET
def _read_existing_yolo_labels(label_path: Path):
    valid_lines = []  # LABELIMG-FIX
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():  # LABELIMG-FIX
        line = raw_line.strip()  # LABELIMG-FIX
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:  # LABELIMG-FIX
            continue
        try:
            cls_id = int(float(parts[0]))  # LABELIMG-FIX
            cx, cy, w, h = map(float, parts[1:])  # LABELIMG-FIX
        except ValueError:
            continue
        if cls_id < 0:  # LABELIMG-FIX
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):  # LABELIMG-FIX
            continue
        if w <= 0.0 or h <= 0.0:  # LABELIMG-FIX
            continue
        if cx - (w / 2.0) < 0.0 or cx + (w / 2.0) > 1.0:  # LABELIMG-FIX
            continue
        if cy - (h / 2.0) < 0.0 or cy + (h / 2.0) > 1.0:  # LABELIMG-FIX
            continue
        valid_lines.append(f"{cls_id} {cx} {cy} {w} {h}")  # LABELIMG-FIX
    return valid_lines  # LABELIMG-FIX


# DATASET
def _copy_image_to_dataset(image_path: Path, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(image_path), str(target_path))


# DATASET
def _parse_yolo_lines(lines: list):
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


# DATASET
def _format_yolo_lines(boxes: list):
    return [f"0 {_clamp01(cx):.6f} {_clamp01(cy):.6f} {_clamp01(w):.6f} {_clamp01(h):.6f}" for cx, cy, w, h in boxes]


# DATASET
def _augment_training_image(image_path: Path, label_lines: list, train_img_dir: Path, train_lbl_dir: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return 0

    img_h, img_w = img.shape[:2]
    stem = image_path.stem
    boxes = _parse_yolo_lines(label_lines)
    written = 0

    flipped_img = cv2.flip(img, 1)
    flipped_boxes = [[1.0 - cx, cy, w, h] for cx, cy, w, h in boxes]
    cv2.imwrite(str(train_img_dir / f"{stem}_flip.jpeg"), flipped_img)
    _write_yolo_label(train_lbl_dir / f"{stem}_flip.txt", _format_yolo_lines(flipped_boxes))
    written += 1

    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot_boxes = [[1.0 - cy, cx, h, w] for cx, cy, w, h in boxes]
    cv2.imwrite(str(train_img_dir / f"{stem}_rot90.jpeg"), rotated)
    _write_yolo_label(train_lbl_dir / f"{stem}_rot90.txt", _format_yolo_lines(rot_boxes))
    written += 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= random.uniform(0.6, 1.4)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    bright_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(train_img_dir / f"{stem}_bright.jpeg"), bright_img)
    _write_yolo_label(train_lbl_dir / f"{stem}_bright.txt", label_lines)
    written += 1

    sigma = random.uniform(1.0, 2.5)
    blur_img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    cv2.imwrite(str(train_img_dir / f"{stem}_blur.jpeg"), blur_img)
    _write_yolo_label(train_lbl_dir / f"{stem}_blur.txt", label_lines)
    written += 1

    if len(boxes) > 1:
        cx, cy, bw, bh = random.choice(boxes)
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        pad_x = (x2 - x1) * 0.40
        pad_y = (y2 - y1) * 0.40
        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(img_w, int(x2 + pad_x))
        crop_y2 = min(img_h, int(y2 + pad_y))

        if crop_x2 > crop_x1 and crop_y2 > crop_y1:
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_h, crop_w = crop.shape[:2]
            crop_boxes = []
            for bx_cx, bx_cy, bx_w, bx_h in boxes:
                bx1 = (bx_cx - bx_w / 2) * img_w
                by1 = (bx_cy - bx_h / 2) * img_h
                bx2 = (bx_cx + bx_w / 2) * img_w
                by2 = (bx_cy + bx_h / 2) * img_h
                fully_inside = bx1 >= crop_x1 and by1 >= crop_y1 and bx2 <= crop_x2 and by2 <= crop_y2
                if not fully_inside:
                    continue
                ncx = ((bx1 + bx2) / 2 - crop_x1) / crop_w
                ncy = ((by1 + by2) / 2 - crop_y1) / crop_h
                nw = (bx2 - bx1) / crop_w
                nh = (by2 - by1) / crop_h
                crop_boxes.append([ncx, ncy, nw, nh])

            if crop_boxes:
                cv2.imwrite(str(train_img_dir / f"{stem}_crop1.jpeg"), crop)
                _write_yolo_label(train_lbl_dir / f"{stem}_crop1.txt", _format_yolo_lines(crop_boxes))
                written += 1

    return written


# DATASET
def convert_dataset_to_yolo(source_dir: str, output_dir: str, val_split: float = 0.2, augment: bool = True) -> dict:
    source = Path(source_dir)
    output = Path(output_dir).resolve()
    images_train = output / "images" / "train"
    images_val = output / "images" / "val"
    labels_train = output / "labels" / "train"
    labels_val = output / "labels" / "val"

    for folder in (images_train, images_val, labels_train, labels_val):
        folder.mkdir(parents=True, exist_ok=True)

    pairs = []
    scenario_counts = {"horizontal": 0, "vertical": 0, "low_ratio": 0}
    pen_count = 0
    image_dir = source  # RETRAIN-FIX
    if (source / "image").is_dir():  # RETRAIN-FIX
        image_dir = source / "image"  # RETRAIN-FIX
    elif (source / "images").is_dir():  # RETRAIN-FIX
        image_dir = source / "images"  # RETRAIN-FIX

    label_dir = image_dir  # RETRAIN-FIX
    for candidate in ("lable", "label", "labels", "annotations"):  # RETRAIN-FIX
        if (source / candidate).is_dir():  # RETRAIN-FIX
            label_dir = source / candidate  # RETRAIN-FIX
            break  # RETRAIN-FIX

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:  # RETRAIN-FIX
            continue

        if PILImage is None:
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
        else:
            with PILImage.open(image_path) as pil_img:
                img_w, img_h = pil_img.size

        json_label_path = label_dir / f"{image_path.stem}L.json"  # RETRAIN-FIX
        txt_label_path = label_dir / f"{image_path.stem}.txt"  # LABELIMG-FIX
        if json_label_path.exists():
            yolo_lines, _pixel_boxes, scenarios = _parse_custom_label_json(str(json_label_path), img_w, img_h)
        elif txt_label_path.exists():  # LABELIMG-FIX
            yolo_lines = _read_existing_yolo_labels(txt_label_path)  # LABELIMG-FIX
            scenarios = {"horizontal": 0, "vertical": 0, "low_ratio": 0}  # LABELIMG-FIX
        else:
            continue

        if not yolo_lines:
            continue

        pen_count += len(yolo_lines)
        for key in scenario_counts:
            scenario_counts[key] += scenarios[key]
        pairs.append((image_path, yolo_lines))

    random.seed(42)
    random.shuffle(pairs)
    val_count_raw = int(len(pairs) * val_split)
    val_pairs = pairs[-val_count_raw:] if val_count_raw else []
    train_pairs = pairs[:-val_count_raw] if val_count_raw else pairs

    train_count = 0
    for image_path, yolo_lines in train_pairs:
        target_img = images_train / image_path.name
        target_lbl = labels_train / f"{image_path.stem}.txt"
        _copy_image_to_dataset(image_path, target_img)
        _write_yolo_label(target_lbl, yolo_lines)
        train_count += 1
        if augment:
            train_count += _augment_training_image(image_path, yolo_lines, images_train, labels_train)

    for image_path, yolo_lines in val_pairs:
        target_img = images_val / image_path.name
        target_lbl = labels_val / f"{image_path.stem}.txt"
        _copy_image_to_dataset(image_path, target_img)
        _write_yolo_label(target_lbl, yolo_lines)

    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['pen']\n",
        encoding="utf-8",
    )

    return {
        "train_count": train_count,
        "val_count": len(val_pairs),
        "yaml_path": str(yaml_path),
        "pen_count": pen_count,
        "scenarios": scenario_counts,
    }


# DATASET
def finetune_on_dataset(
    dataset_yaml: str,
    base_weights: str = None,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    output_name: str = "pen_retrain",
    freeze_backbone: bool = True,
    device: str = "cpu",
) -> dict:
    base = base_weights or _resolve_model_path(MODEL_PATH)
    model = YOLO(base)
    runs_project_dir = BASE_DIR / "runs" / "detect"

    if freeze_backbone:
        for name, param in model.model.named_parameters():
            if any(f"model.{j}." in name for j in range(10)):
                param.requires_grad = False

    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=output_name,
        project=str(runs_project_dir),
        device=device,
        workers=0,  # RETRAIN-FIX
        mosaic=1.0,
        mixup=0.2,
        flipud=0.0,
        fliplr=0.5,
        degrees=15.0,
        scale=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        patience=15,
        save=True,
        exist_ok=True,
    )

    run_dir = runs_project_dir / output_name
    best_pt = run_dir / "weights" / "best.pt"

    if best_pt.exists():
        reload_pen_model(best_pt)
        print(f"[finetune] Model updated to {best_pt}")

    log_path = BASE_DIR / "retrain_log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.datetime.now().isoformat()} | "
            f"epochs={epochs} | "
            f"best={best_pt} | "
            f"dataset={dataset_yaml}\n"
        )

    return {
        "best_weights": str(best_pt) if best_pt.exists() else None,
        "run_dir": str(run_dir),
        "epochs_trained": epochs,
    }


# DATASET
def train_from_zip(zip_path: str, output_dataset_dir: str = None, epochs: int = 50, device: str = "cpu") -> dict:
    """
    Extract zip -> convert labels -> augment -> finetune model.
    Call this once to bootstrap pen detection from labelled data.
    """
    import zipfile

    tmp = tempfile.mkdtemp(prefix="pen_dataset_")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        source_dir = tmp
        for root, _dirs, files in os.walk(tmp):
            if any(f.lower().endswith((".jpeg", ".jpg")) for f in files):
                source_dir = root
                break

        if output_dataset_dir is None:
            output_dataset_dir = str(BASE_DIR / DATASET_DIR)

        summary = convert_dataset_to_yolo(
            source_dir=source_dir,
            output_dir=output_dataset_dir,
            val_split=0.2,
            augment=True,
        )
        print(f"[train_from_zip] Dataset ready: {summary}")

        ft_result = finetune_on_dataset(
            dataset_yaml=summary["yaml_path"],
            epochs=epochs,
            device=device,
        )

        return {**summary, **ft_result}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# DATASET
def train_from_folder(source_dir: str = None, output_dataset_dir: str = None) -> dict:  # RETRAIN-FIX
    """
    Build a YOLO dataset from a labelled folder and retrain the model.
    """
    if source_dir is None:  # RETRAIN-FIX
        source_dir = str(BASE_DIR / "dataset")  # RETRAIN-FIX
    if output_dataset_dir is None:  # RETRAIN-FIX
        output_dataset_dir = str(BASE_DIR / DATASET_DIR)  # RETRAIN-FIX

    print(f"[train_from_folder] Reading source folder: {source_dir}")  # RETRAIN-FIX
    print(f"[train_from_folder] Building YOLO dataset at: {output_dataset_dir}")  # RETRAIN-FIX
    print(f"[train_from_folder] Applying augmentations: {', '.join(AUGMENT_VARIANTS)}")  # RETRAIN-FIX
    summary = convert_dataset_to_yolo(  # RETRAIN-FIX
        source_dir=source_dir,
        output_dir=output_dataset_dir,
        val_split=0.2,
        augment=True,
    )
    if summary["train_count"] == 0 or summary["val_count"] == 0:  # RETRAIN-FIX
        raise ValueError(  # RETRAIN-FIX
            f"No valid YOLO-labelled samples were found in {source_dir}. "
            "Each .txt label must contain normalized 'class cx cy w h' values "
            "within [0, 1] and boxes must stay inside image bounds."
        )
    print(f"[train_from_folder] Dataset ready: {summary}")  # RETRAIN-FIX
    print("[train_from_folder] Starting fine-tuning with epochs=80, freeze_backbone=True, device=cpu")  # RETRAIN-FIX
    ft_result = finetune_on_dataset(  # RETRAIN-FIX
        dataset_yaml=summary["yaml_path"],
        epochs=80,
        freeze_backbone=True,
        device="cpu",
    )
    print(f"[train_from_folder] Fine-tuning complete: {ft_result}")  # RETRAIN-FIX
    return {**summary, **ft_result}  # RETRAIN-FIX


# DATASET
def add_labelled_image(
    image_path: str,
    label_json_path: str,
    dataset_dir: str,
) -> dict:
    """
    Add one custom-labelled JPEG to an existing YOLO dataset.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]
    yolo_lines, _pixel_boxes, _scenarios = _parse_custom_label_json(label_json_path, img_w, img_h)

    dataset_path = Path(dataset_dir)
    train_img_dir = dataset_path / "images" / "train"
    train_lbl_dir = dataset_path / "labels" / "train"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    train_img = train_img_dir / f"{stem}.jpeg"
    train_lbl = train_lbl_dir / f"{stem}.txt"
    shutil.copy2(image_path, train_img)
    _write_yolo_label(train_lbl, yolo_lines)

    return {"added": str(train_img), "labels": len(yolo_lines)}


def detect_objects(image: np.ndarray):
    """Run a lightweight COCO pass for person/object regions used by fallback zoom."""
    try:
        model = load_general_model()
        results_list = model(image, verbose=False, conf=CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD)
    except Exception:
        return []

    detections = []
    for result in results_list:  # DETECTION-FIX: process every YOLO result object.
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(box.conf[0]),
                "class_id": int(box.cls[0]),
            })

    return sorted(detections, key=lambda item: item["conf"], reverse=True)


# SCENE
def _get_hand_region_coco(image: np.ndarray) -> list:
    """Return all likely hand/person crops from COCO person detections."""
    detections = detect_objects(image)
    crops = []

    for det in detections:
        if det.get("class_id") != 0:
            continue

        x1, y1, x2, y2 = det["bbox"]
        h = y2 - y1
        lower_y1 = y1 + int(h * 0.60)
        lower_crop = image[lower_y1:y2, x1:x2]
        if lower_crop.size:
            crops.append((lower_crop, [x1, lower_y1, x2, y2]))

        mid_x = x1 + int((x2 - x1) * 0.55)
        right_crop = image[y1:y2, mid_x:x2]
        if right_crop.size:
            crops.append((right_crop, [mid_x, y1, x2, y2]))

    return crops


# SCENE
def _get_table_region(image: np.ndarray, hand_bboxes: list) -> list:
    """
    Return image regions that are likely table surface below and around hands.
    """
    img_h, img_w = image.shape[:2]
    table_crops = []

    if hand_bboxes:
        for hbbox in hand_bboxes:
            hx1, hy1, hx2, hy2 = hbbox
            table_y1 = min(img_h, hy2)
            table_y2 = min(img_h, hy2 + int((hy2 - hy1) * 0.6))
            table_x1 = max(0, hx1 - int((hx2 - hx1) * 0.2))
            table_x2 = min(img_w, hx2 + int((hx2 - hx1) * 0.2))
            if table_y2 > table_y1 and table_x2 > table_x1:
                crop = image[table_y1:table_y2, table_x1:table_x2]
                if crop.size:
                    table_crops.append((crop, [table_x1, table_y1, table_x2, table_y2]))
    else:
        table_y1 = int(img_h * 0.60)
        crop = image[table_y1:, :]
        if crop.size:
            table_crops.append((crop, [0, table_y1, img_w, img_h]))

    return table_crops


def classify_confidence(conf: float) -> str:
    if conf >= HIGH_CONF:
        return "high_confidence"
    if conf >= POSSIBLE_LOW:
        return "possible"
    if conf >= max(UNCERTAIN_LOW, CONF_THRESHOLD):
        return "uncertain"
    if conf >= CONF_THRESHOLD:
        return "uncertain"
    return "not_detected"


def annotate_image(image: np.ndarray, boxes):
    annotated = image.copy()

    for item in boxes:
        x, y, w, h = map(int, item["bbox"])
        status = item.get("status", "not_detected")
        color = (0, 255, 0)
        if status == "possible":
            color = (0, 255, 255)
        elif status == "uncertain":
            color = (0, 165, 255)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"pen: {item.get('conf', 0.0) * 100:.1f}%"
        cv2.putText(
            annotated,
            label,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


# ================================
# HAND DETECTION (MEDIAPIPE)
# ================================
# ACCURACY
def _is_valid_hand(xs, ys, img_w, img_h) -> bool:
    if not xs or not ys:
        return False

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area < MIN_HAND_AREA_FRAC * img_w * img_h:
        return False

    touches_left = x1 <= 1
    touches_top = y1 <= 1
    touches_right = x2 >= img_w - 2
    touches_bottom = y2 >= img_h - 2
    if touches_left and touches_top and touches_right and touches_bottom:
        return False

    inside_count = 0
    for px, py in zip(xs, ys):
        if x1 <= px <= x2 and y1 <= py <= y2:
            inside_count += 1

    return inside_count >= 5


# SCENE
def _is_writing_posture(hand_landmarks, img_w, img_h) -> bool:
    """
    Returns True if hand landmarks suggest a pinch/three-finger writing grip.
    """
    def lm_px(lm):
        return int(lm.x * img_w), int(lm.y * img_h)

    def dist(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    try:
        thumb_tip = lm_px(hand_landmarks.landmark[4])
        index_tip = lm_px(hand_landmarks.landmark[8])
        middle_tip = lm_px(hand_landmarks.landmark[12])
        wrist = lm_px(hand_landmarks.landmark[0])
        middle_mcp = lm_px(hand_landmarks.landmark[9])
    except Exception:
        return False

    hand_span = max(dist(wrist, middle_mcp), 1.0)
    pinch_dist = dist(thumb_tip, index_tip)

    if pinch_dist < hand_span * WRITING_PINCH_RATIO:
        return True

    if dist(thumb_tip, middle_tip) < hand_span * 0.40:
        return True

    return False


def detect_hand_mediapipe(image: np.ndarray):
    """
    Returns:
        list of tuples: [(crop, bbox, is_writing), ...]
        where bbox = [x1, y1, x2, y2]
    """
    img_h, img_w = image.shape[:2]

    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        return []

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=HAND_DETECTION_CONF,  # ACCURACY
        min_tracking_confidence=HAND_TRACKING_CONF,    # ACCURACY
    )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)
    hands.close()

    if not results.multi_hand_landmarks:
        return []

    hand_outputs = []

    for hand_landmarks in results.multi_hand_landmarks:
        xs, ys = [], []

        for lm in hand_landmarks.landmark:
            px = int(lm.x * img_w)
            py = int(lm.y * img_h)
            xs.append(px)
            ys.append(py)

        if not _is_valid_hand(xs, ys, img_w, img_h):  # ACCURACY
            continue

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        pad = int(max(x2 - x1, y2 - y1) * HAND_CROP_PADDING)  # ACCURACY

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_w, x2 + pad)
        y2 = min(img_h, y2 + pad)

        crop = image[y1:y2, x1:x2]
        if crop.size:
            writing = _is_writing_posture(hand_landmarks, img_w, img_h)  # SCENE
            hand_outputs.append((crop, [x1, y1, x2, y2], writing))  # SCENE

    return hand_outputs


# ================================
# PEN SHAPE FILTER (ANGLE-AWARE)
# ================================
# BLUR
# DATASET
def _is_pen_shaped(
    bbox,
    conf: float,
    image_crop: np.ndarray,
    img_h=None,
    img_w=None,
    blur_level="sharp",
    portrait_mode=False,
):
    """
    bbox: [x1, y1, x2, y2]
    Returns:
        (bool, angle_deg)
    """
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    ratio = max(w, h) / min(w, h)

    if conf >= 0.80 and PEN_MIN_ASPECT_RATIO <= ratio < 1.6:  # BANGLE-FIX
        angle = _estimate_pen_angle(image_crop, blur_level=blur_level)
        if angle is not None:
            return True, angle
        return True, None

    tier1_thresh = BLUR_TIER1_THRESH if blur_level in ("severe", "moderate") else 2.5
    tier2_min = BLUR_TIER2_MIN if blur_level in ("severe", "moderate") else 1.6

    if ratio >= tier1_thresh:
        return True, None

    if PEN_MIN_ASPECT_RATIO <= ratio < tier1_thresh:
        min_conf_tier2 = TIER2_CONF_DEFAULT
        if img_h and img_w and (y1 + y2) / 2 < img_h * 0.70:
            min_conf_tier2 = TIER2_CONF_UPPER
        if portrait_mode and h > w:
            min_conf_tier2 = max(0.35, min_conf_tier2 - 0.07)
        if blur_level in ("severe", "moderate"):
            min_conf_tier2 = max(0.30, min_conf_tier2 - 0.08)

        if conf < min_conf_tier2:
            return False, None

        angle = _estimate_pen_angle(image_crop, blur_level=blur_level)
        return True, angle

    return False, None


# ================================
# ANGLE ESTIMATION (HOUGH)
# ================================
# BLUR
def _estimate_pen_angle(image_crop: np.ndarray, blur_level="sharp"):
    """
    Returns:
        angle in degrees (0 = horizontal, 90 = vertical)
        or None if not detectable
    """
    try:
        if image_crop is None or image_crop.size == 0:
            return None

        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        if blur_level in ("severe", "moderate"):
            gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1)
            gray_sharp = cv2.addWeighted(
                gray,
                2.0,
                cv2.GaussianBlur(gray, (0, 0), sigmaX=3),
                -1.0,+
                0,
            )
            gray = np.clip(gray_sharp, 0, 255).astype(np.uint8)

        median_val = float(np.median(gray))
        sigma = 0.33
        low = int(max(0, (1.0 - sigma) * median_val))
        high = int(min(255, (1.0 + sigma) * median_val))
        if blur_level in ("severe", "moderate"):
            low = max(0, low // 2)
            high = max(low + 10, high // 2)

        edges = cv2.Canny(gray, low, high)
        min_len = 15 if blur_level in ("severe", "moderate") else 30
        max_gap = 20 if blur_level in ("severe", "moderate") else 10
        thresh = 15 if blur_level in ("severe", "moderate") else 30

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=thresh,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )

        if lines is None:
            return None

        angles = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            angle = abs(angle)

            if angle > 90:
                angle = 180 - angle

            angles.append(angle)

        if not angles:
            return None

        return float(np.median(angles))

    except Exception:
        return None


# ================================
# MULTI-PEN SEPARATION LOGIC
# ================================
# ACCURACY
def _get_bbox_orientation(bbox, image_ref=None, blur_level="sharp") -> float:  # BLUR
    if image_ref is not None:
        try:
            img_h, img_w = image_ref.shape[:2]
            x1 = max(0, min(img_w, int(bbox[0])))
            y1 = max(0, min(img_h, int(bbox[1])))
            x2 = max(0, min(img_w, int(bbox[2])))
            y2 = max(0, min(img_h, int(bbox[3])))
            crop = image_ref[y1:y2, x1:x2]
            angle = _estimate_pen_angle(crop, blur_level=blur_level)  # BLUR
            if angle is not None:
                return angle
        except Exception:
            pass

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return 90.0 if h > w else 0.0


def _are_separate_pens(bbox_a, bbox_b, img_h, img_w, image_ref=None, blur_level="sharp"):  # BLUR
    """
    Decide whether two detections should be treated as separate pens.
    """
    iou = _compute_iou(bbox_a, bbox_b)  # ACCURACY

    cxA = (bbox_a[0] + bbox_a[2]) / 2
    cyA = (bbox_a[1] + bbox_a[3]) / 2

    cxB = (bbox_b[0] + bbox_b[2]) / 2
    cyB = (bbox_b[1] + bbox_b[3]) / 2

    dist = np.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)

    if iou < 0.10 and dist > 0.15 * max(img_h, img_w):
        return True

    # DATASET
    if iou < 0.05:
        wA = bbox_a[2] - bbox_a[0]
        hA = bbox_a[3] - bbox_a[1]
        wB = bbox_b[2] - bbox_b[0]
        hB = bbox_b[3] - bbox_b[1]
        both_horizontal = wA > hA and wB > hB
        if both_horizontal:
            gap_y = abs((bbox_a[1] + bbox_a[3]) / 2 - (bbox_b[1] + bbox_b[3]) / 2)
            if gap_y < 0.20 * max(img_h, 1):
                return True

        both_vertical = hA > wA and hB > wB
        if both_vertical:
            gap_x = abs(cxA - cxB)
            if gap_x < 0.20 * max(img_w, 1):
                return True

    if iou < 0.08:
        return True  # BUG-05 FIX: low-overlap YOLO boxes are separate pens.

    if iou == 0:
        angleA = _get_bbox_orientation(bbox_a, image_ref, blur_level=blur_level)  # BLUR
        angleB = _get_bbox_orientation(bbox_b, image_ref, blur_level=blur_level)  # BLUR

        if abs(angleA - angleB) < 25:
            return True

    return False


# ================================
# PEN INFERENCE (MULTI-PEN SUPPORT)
# ================================
# ACCURACY
def _score_detection(conf, bbox, hand_bboxes, img_h, img_w) -> float:
    x1, y1, x2, y2 = bbox
    boost = 0.0

    for hbox in hand_bboxes:
        hx1, hy1, hx2, hy2 = hbox

        iou = _compute_iou(bbox, hbox)
        if iou > 0.15:
            boost += 0.05

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
            boost += 0.02

    w, h = x2 - x1, y2 - y1
    ratio = max(w, h) / max(min(w, h), 1)
    if ratio >= 2.5:
        boost += 0.03

    return min(conf + boost, 0.99)


# SCENE
def _expand_occluded_bbox(bbox, image, direction="auto") -> list:
    """
    Extend a detected pen bbox along its dominant axis to recover occluded length.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    img_h, img_w = image.shape[:2]

    if w >= h:
        extend = int(w * OCCLUDE_EXTEND_FRAC)
        x1 = max(0, x1 - extend)
        x2 = min(img_w, x2 + extend)
    else:
        extend = int(h * OCCLUDE_EXTEND_FRAC)
        y1 = max(0, y1 - extend)
        y2 = min(img_h, y2 + extend)

    return [x1, y1, x2, y2]


# BLUR
def _is_false_positive_background(crop: np.ndarray, blur_level="sharp") -> bool:
    """
    Returns True if a crop looks like flat colored paper instead of a pen.
    """
    # DETECTION-FIX:
    # Do not reject YOLO pen candidates based on handcrafted background texture.
    # Pens on cloth, bedsheets, dark tables, and smooth surfaces often have low
    # edge density/saturation variance, so YOLO confidence >= 0.25 is trusted.
    return False

    if crop is None or crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_density = np.sum(edges > 0) / max(1, edges.size)
    min_edge_density = BLUR_EDGE_DENSITY if blur_level in ("severe", "moderate") else 0.05
    if edge_density < min_edge_density:
        return True

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat_std = float(np.std(hsv[:, :, 1]))
    val_std = float(np.std(hsv[:, :, 2]))
    if sat_std < 20 and val_std < 25:
        return True

    mean_hue = float(np.mean(hsv[:, :, 0]))
    mean_sat = float(np.mean(hsv[:, :, 1]))
    mean_val = float(np.mean(hsv[:, :, 2]))
    is_bright_color = mean_sat > 100 and mean_val > 150
    is_sticky_hue = (15 <= mean_hue <= 40) or (140 <= mean_hue <= 170) or (35 <= mean_hue <= 85)
    if is_bright_color and is_sticky_hue:
        return True

    return False


def _enhance_transparent_pen(crop: np.ndarray) -> float:  # TRANSPARENT-PEN-FIX
    if crop is None or crop.size == 0:  # TRANSPARENT-PEN-FIX
        return 0.0  # TRANSPARENT-PEN-FIX

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)  # TRANSPARENT-PEN-FIX
    l_channel, a_channel, b_channel = cv2.split(lab)  # TRANSPARENT-PEN-FIX
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # TRANSPARENT-PEN-FIX
    l_enhanced = clahe.apply(l_channel)  # TRANSPARENT-PEN-FIX
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))  # TRANSPARENT-PEN-FIX
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)  # TRANSPARENT-PEN-FIX

    blur = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)  # TRANSPARENT-PEN-FIX
    sharpened = cv2.addWeighted(enhanced_bgr, 2.0, blur, -1.0, 0)  # TRANSPARENT-PEN-FIX

    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)  # TRANSPARENT-PEN-FIX
    edges = cv2.Canny(gray, 50, 150)  # TRANSPARENT-PEN-FIX
    lines = cv2.HoughLinesP(  # TRANSPARENT-PEN-FIX
        edges,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=max(10, min(crop.shape[:2]) // 2),
        maxLineGap=6,
    )
    if lines is None:  # TRANSPARENT-PEN-FIX
        return 0.0  # TRANSPARENT-PEN-FIX

    segments = []  # TRANSPARENT-PEN-FIX
    for line in lines:  # TRANSPARENT-PEN-FIX
        lx1, ly1, lx2, ly2 = line[0]  # TRANSPARENT-PEN-FIX
        angle = abs(np.degrees(np.arctan2((ly2 - ly1), (lx2 - lx1))))  # TRANSPARENT-PEN-FIX
        if angle > 90:  # TRANSPARENT-PEN-FIX
            angle = 180 - angle  # TRANSPARENT-PEN-FIX
        length = float(np.hypot(lx2 - lx1, ly2 - ly1))  # TRANSPARENT-PEN-FIX
        segments.append((angle, length, lx1, ly1, lx2, ly2))  # TRANSPARENT-PEN-FIX

    for idx, (angle_a, len_a, ax1, ay1, ax2, ay2) in enumerate(segments):  # TRANSPARENT-PEN-FIX
        if len_a < 12:  # TRANSPARENT-PEN-FIX
            continue  # TRANSPARENT-PEN-FIX
        mx_a = (ax1 + ax2) / 2.0  # TRANSPARENT-PEN-FIX
        my_a = (ay1 + ay2) / 2.0  # TRANSPARENT-PEN-FIX
        for angle_b, len_b, bx1, by1, bx2, by2 in segments[idx + 1:]:  # TRANSPARENT-PEN-FIX
            if len_b < 12 or abs(angle_a - angle_b) > 8:  # TRANSPARENT-PEN-FIX
                continue  # TRANSPARENT-PEN-FIX
            mx_b = (bx1 + bx2) / 2.0  # TRANSPARENT-PEN-FIX
            my_b = (by1 + by2) / 2.0  # TRANSPARENT-PEN-FIX
            separation = float(np.hypot(mx_a - mx_b, my_a - my_b))  # TRANSPARENT-PEN-FIX
            if 4 <= separation <= max(12, min(crop.shape[:2]) * 0.35):  # TRANSPARENT-PEN-FIX
                return 0.10  # TRANSPARENT-PEN-FIX

    return 0.0  # TRANSPARENT-PEN-FIX


# BLUR
# DATASET
def _run_pen_inference(
    image: np.ndarray,
    offset=(0, 0),
    zoom_priority=None,
    hand_bboxes=None,
    blur_level="sharp",
    portrait_mode=False,
):
    """
    Runs YOLO and returns multiple pen detections (no early return).
    """
    model = load_model()
    base_conf = float(model.overrides.get("conf", CONF_THRESHOLD))
    if blur_level == "severe":
        active_conf_threshold = max(BLUR_CONF_SEVERE, base_conf - 0.15)
    elif blur_level == "moderate":
        active_conf_threshold = max(BLUR_CONF_MODERATE, base_conf - 0.10)
    else:
        active_conf_threshold = base_conf
    if base_conf < CONF_THRESHOLD:
        active_conf_threshold = min(active_conf_threshold, base_conf)

    hand_bboxes = hand_bboxes or []  # ACCURACY
    img_h, img_w = image.shape[:2]

    detections = []

    def _warn_if_classification_result(results):
        if results.boxes is None and getattr(results, "probs", None) is not None:
            warnings.warn(
                "Model returned probs instead of boxes - "
                "classification model loaded instead of detection model",
                RuntimeWarning,
            )
        elif results.boxes is not None and len(results.boxes) == 0 and getattr(results, "probs", None) is not None:
            warnings.warn(
                "Model returned probs instead of boxes - "
                "classification model loaded instead of detection model",
                RuntimeWarning,
            )

    def _process_candidate(raw_conf, bbox):
        nonlocal detections
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(img_w, int(x1)))
        y1 = max(0, min(img_h, int(y1)))
        x2 = max(0, min(img_w, int(x2)))
        y2 = max(0, min(img_h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return

        ox, oy = offset
        x1g, y1g = x1 + ox, y1 + oy
        x2g, y2g = x2 + ox, y2 + oy

        conf = _score_detection(raw_conf, [x1g, y1g, x2g, y2g], hand_bboxes, img_h, img_w)  # ACCURACY
        crop = image[y1:y2, x1:x2]
        conf += _enhance_transparent_pen(crop)  # TRANSPARENT-PEN-FIX
        conf = min(conf, 0.99)  # TRANSPARENT-PEN-FIX
        if conf < active_conf_threshold:
            return

        is_pen, angle = _is_pen_shaped(
            [x1, y1, x2, y2],
            conf,
            crop,
            img_h=img_h,
            img_w=img_w,
            blur_level=blur_level,
            portrait_mode=portrait_mode,
        )  # BLUR

        if not is_pen:
            return

        if zoom_priority in ("hand_bbox", "coco_person"):  # SCENE
            expanded_bbox = _expand_occluded_bbox([x1, y1, x2, y2], image)
            ex1, ey1, ex2, ey2 = expanded_bbox
            expanded_crop = image[ey1:ey2, ex1:ex2]
            expanded_ok, expanded_angle = _is_pen_shaped(
                expanded_bbox,
                conf,
                expanded_crop,
                img_h=img_h,
                img_w=img_w,
                blur_level=blur_level,
                portrait_mode=portrait_mode,
            )
            if not expanded_ok:
                return
            x1, y1, x2, y2 = expanded_bbox
            x1g, y1g = x1 + ox, y1 + oy
            x2g, y2g = x2 + ox, y2 + oy
            crop = expanded_crop
            angle = expanded_angle if expanded_angle is not None else angle

        if angle is not None and IDEAL_ANGLE_MIN <= angle <= IDEAL_ANGLE_MAX:
            conf = min(conf + IDEAL_ANGLE_BOOST, 0.99)  # DETECTION-FIX: boost only, never reject by angle.

        if conf < BACKGROUND_SKIP_CONF:
            if _is_false_positive_background(crop, blur_level=blur_level):  # BLUR
                return

        detections.append({
            "bbox": [x1g, y1g, x2g, y2g],
            "conf": conf,
            "angle_deg": angle,
            "zoom_priority": zoom_priority,
            "writing_posture": None,  # SCENE
        })

    results_list = model(
        image,
        verbose=False,
        conf=CONF_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        max_det=MAX_PENS,
    )
    for results in results_list:
        _warn_if_classification_result(results)

        if results.boxes is None:
            continue
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = str(model.names.get(cls_id, cls_id) if isinstance(model.names, dict) else model.names[cls_id])
            if cls_name.strip().lower() != "pen":
                continue
            raw_conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            _process_candidate(raw_conf, [x1, y1, x2, y2])

    if blur_level in ("severe", "moderate"):
        image_large = cv2.resize(image, (1280, 1280), interpolation=cv2.INTER_LANCZOS4)
        results_large_list = model(
            image_large,
            imgsz=1280,
            verbose=False,
            conf=CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            max_det=MAX_PENS,
        )
        scale_x = img_w / 1280
        scale_y = img_h / 1280
        for results_large in results_large_list:
            _warn_if_classification_result(results_large)
            if results_large.boxes is None:
                continue
            for box in results_large.boxes:
                cls_id = int(box.cls[0])
                cls_name = str(model.names.get(cls_id, cls_id) if isinstance(model.names, dict) else model.names[cls_id])
                if cls_name.strip().lower() != "pen":
                    continue
                raw_conf = float(box.conf[0])
                x1l, y1l, x2l, y2l = map(int, box.xyxy[0])
                _process_candidate(
                    raw_conf,
                    [
                        int(x1l * scale_x),
                        int(y1l * scale_y),
                        int(x2l * scale_x),
                        int(y2l * scale_y),
                    ],
                )

    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)

    return detections[:MAX_PENS]


# ================================
# ZOOM + RE-DETECTION (PRIORITY BASED)
# ================================
# BLUR
# DATASET
def zoom_and_redetect(original_image, hand_bboxes, general_detections, blur_level="sharp", portrait_mode=False):
    """
    Smart zoom fallback with priority:
    1. Hand boxes
    2. COCO person
    3. Other objects
    """
    model = load_model()
    original_conf = model.overrides.get("conf", CONF_THRESHOLD)

    img_h, img_w = original_image.shape[:2]
    results_all = []
    priority_regions = []

    for bbox in hand_bboxes:
        priority_regions.append(("hand_bbox", bbox))

    for det in general_detections:
        if det.get("class_id") == 0:
            priority_regions.append(("coco_person", det["bbox"]))

    for det in general_detections:
        if det.get("class_id") != 0:
            priority_regions.append(("general_object", det["bbox"]))

    for priority, bbox in priority_regions:
        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1

        pad_w = int(w * ZOOM_PADDING)
        pad_h = int(h * ZOOM_PADDING)

        x1c = max(0, x1 - pad_w)
        y1c = max(0, y1 - pad_h)
        x2c = min(img_w, x2 + pad_w)
        y2c = min(img_h, y2 + pad_h)

        crop = original_image[y1c:y2c, x1c:x2c]

        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LANCZOS4)
        zoom_conf = ZOOM_CONF_THRESHOLD
        if blur_level == "severe":
            zoom_conf = BLUR_ZOOM_SEVERE
        elif blur_level == "moderate":
            zoom_conf = BLUR_ZOOM_MODERATE
        model.overrides["conf"] = zoom_conf

        zoom_hand_bboxes = [[0, 0, 640, 640]] if priority == "hand_bbox" else []  # ACCURACY
        detections = _run_pen_inference(
            crop_resized,
            offset=(0, 0),
            zoom_priority=priority,
            hand_bboxes=zoom_hand_bboxes,  # ACCURACY
            blur_level=blur_level,  # BLUR
            portrait_mode=portrait_mode,  # DATASET
        )

        model.overrides["conf"] = original_conf

        if not detections:
            continue

        scale_x = (x2c - x1c) / 640
        scale_y = (y2c - y1c) / 640

        for det in detections:
            px1, py1, px2, py2 = det["bbox"]

            ox1 = int(x1c + px1 * scale_x)
            oy1 = int(y1c + py1 * scale_y)
            ox2 = int(x1c + px2 * scale_x)
            oy2 = int(y1c + py2 * scale_y)

            det["bbox"] = [ox1, oy1, ox2, oy2]

        if priority == "hand_bbox":  # ACCURACY
            best = max(detections, key=lambda d: d["conf"])
            results_all.extend(detections)
            if best["conf"] >= ZOOM_EARLY_EXIT_CONF:
                break
            continue

        results_all.extend(detections)

        if priority == "coco_person":
            break

    model.overrides["conf"] = original_conf
    results_all = sorted(results_all, key=lambda x: x["conf"], reverse=True)

    return results_all[:MAX_PENS]


# ================================
# NMS (MULTI-PEN SAFE)
# ================================
def _nms_detections(detections, iou_thresh=MERGE_IOU_THRESHOLD, image_ref=None, blur_level="sharp"):  # BLUR
    """
    Custom NMS that preserves separate pens.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)

    final_dets = []
    img_h = max([d["bbox"][3] for d in detections])
    img_w = max([d["bbox"][2] for d in detections])

    for det in detections:
        keep = True

        for kept in final_dets:
            iou = _compute_iou(det["bbox"], kept["bbox"])  # ACCURACY

            if _are_separate_pens(det["bbox"], kept["bbox"], img_h, img_w, image_ref=image_ref, blur_level=blur_level):  # BLUR
                continue

            if iou > iou_thresh:
                keep = False
                break

        if keep:
            final_dets.append(det)

    return final_dets


# ================================
# MAIN PIPELINE
# ================================
def detect_pen(image: np.ndarray):
    """
    Full pipeline with:
    - MediaPipe hand detection
    - Multi-pen detection
    - Smart zoom fallback
    - Angle-aware filtering
    """
    original_image = _decode_image_input(image)  # ACCURACY

    if original_image is None or original_image.size == 0:
        return {
            "pen_detected": False,
            "pen_count": 0,
            "confidence": 0.0,
            "boxes": [],
            "method": [],
            "reason": "invalid_image",  # ACCURACY
            "blur_level": "unknown",  # BLUR
            "annotated": None,
        }

    # DATASET
    img_h, img_w = original_image.shape[:2]
    portrait_mode = img_h > img_w

    quality = check_image_quality(original_image)  # ACCURACY
    blur_level = quality.get("blur_level", "sharp")  # BLUR
    if not quality["ok"]:
        return {
            "pen_detected": False,
            "pen_count": 0,
            "confidence": 0.0,
            "boxes": [],
            "method": ["quality_rejected"],
            "reason": quality["reason"],  # ACCURACY
            "blur_level": blur_level,  # BLUR
            "annotated": None,
        }

    methods_used = []

    processed, preprocess_meta = preprocess_image(original_image, blur_level=blur_level)  # BLUR
    if processed is None or processed.size == 0:
        return {
            "pen_detected": False,
            "pen_count": 0,
            "confidence": 0.0,
            "boxes": [],
            "method": [],
            "reason": "invalid_image",  # ACCURACY
            "blur_level": blur_level,  # BLUR
            "annotated": None,
        }

    general_dets = detect_objects(processed)
    all_detections = []
    hand_data = detect_hand_mediapipe(processed)
    hand_bboxes = []

    if hand_data:
        methods_used.append("mediapipe")

        for crop, bbox, is_writing in hand_data:  # SCENE
            x1, y1, x2, y2 = bbox
            hand_bboxes.append(bbox)

            detections = _run_pen_inference(
                crop,
                offset=(x1, y1),
                zoom_priority="hand_bbox",
                hand_bboxes=hand_bboxes,  # ACCURACY
                blur_level=blur_level,  # BLUR
                portrait_mode=portrait_mode,  # DATASET
            )

            if is_writing:  # SCENE
                for det in detections:
                    det["conf"] = min(det["conf"] + WRITING_BOOST, 0.99)
                    det["writing_posture"] = True

            all_detections.extend(detections)

    else:
        methods_used.append("coco_fallback")
        coco_hand_crops = _get_hand_region_coco(processed)  # SCENE

        for crop, bbox in coco_hand_crops:  # SCENE
            x1, y1, x2, y2 = bbox
            hand_bboxes.append(bbox)

            detections = _run_pen_inference(
                crop,
                offset=(x1, y1),
                zoom_priority="coco_person",
                hand_bboxes=hand_bboxes,  # ACCURACY
                blur_level=blur_level,  # BLUR
                portrait_mode=portrait_mode,  # DATASET
            )

            all_detections.extend(detections)

    best_hand_conf = max([d["conf"] for d in all_detections], default=0.0)  # ACCURACY

    table_found = False  # SCENE
    table_crops = _get_table_region(processed, hand_bboxes)  # SCENE
    for t_crop, t_bbox in table_crops:  # SCENE
        tx1, ty1, tx2, ty2 = t_bbox
        table_dets = _run_pen_inference(
            t_crop,
            offset=(tx1, ty1),
            zoom_priority="table_region",
            hand_bboxes=[],
            blur_level=blur_level,  # BLUR
            portrait_mode=portrait_mode,  # DATASET
        )
        if table_dets:
            table_found = True
        all_detections.extend(table_dets)

    no_context = not hand_data and not coco_hand_crops
    if no_context:
        full_dets = _run_pen_inference(
            processed,
            offset=(0, 0),
            zoom_priority="full_image_fallback",
            hand_bboxes=[],
            blur_level=blur_level,
            portrait_mode=portrait_mode,
        )
        all_detections.extend(full_dets)
        if full_dets:
            methods_used.append("no_hand_full_scan")

    if table_found:  # SCENE
        methods_used.append("table_scan")

    if best_hand_conf < FULL_PASS_SKIP_CONF:
        full_dets = _run_pen_inference(
            processed,
            hand_bboxes=hand_bboxes,  # ACCURACY
            blur_level=blur_level,  # BLUR
            portrait_mode=portrait_mode,  # DATASET
        )
        all_detections.extend(full_dets)
    else:
        methods_used.append("skipped_full_pass")  # ACCURACY

    if not all_detections:
        zoom_dets = zoom_and_redetect(
            processed,
            hand_bboxes,
            general_dets,
            blur_level=blur_level,
            portrait_mode=portrait_mode,
        )  # DATASET

        if zoom_dets:
            methods_used.append("zoom_fallback")

        all_detections.extend(zoom_dets)

    final_dets = _nms_detections(all_detections, image_ref=processed, blur_level=blur_level)  # BLUR

    boxes_out = []

    for det in final_dets:
        x1, y1, x2, y2 = _remap_bbox_from_letterbox(
            det["bbox"],
            preprocess_meta,
            original_image.shape,
        )  # ACCURACY
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w == 0 or h == 0:
            continue

        boxes_out.append({
            "bbox": [x1, y1, w, h],
            "conf": det["conf"],
            "status": classify_confidence(det["conf"]),
            "angle_deg": det.get("angle_deg"),
            "zoom_priority": det.get("zoom_priority"),
            "writing_posture": det.get("writing_posture"),  # SCENE
        })

    pen_count = len(boxes_out)
    best_conf = max([b["conf"] for b in boxes_out], default=0.0)

    annotated = None
    if pen_count > 0:
        annotated = annotate_image(original_image, boxes_out)

    return {
        "pen_detected": pen_count > 0,
        "pen_count": pen_count,
        "confidence": best_conf,
        "boxes": boxes_out,
        "method": methods_used,
        "reason": None,  # ACCURACY
        "blur_level": blur_level,  # BLUR
        "annotated": annotated,
    }


# ACCURACY
def warmup_models():
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    try:
        load_model()(dummy, verbose=False)
        load_general_model()(dummy, verbose=False)
        print("[PenDetect] Models warmed up.")
    except Exception as e:
        print(f"[PenDetect] Warmup failed: {e}")


# ACCURACY
if __name__ != "__main__":
    warmup_models()
