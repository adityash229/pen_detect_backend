from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from opencv_compat import ensure_cv2_compat

ensure_cv2_compat()

from ultralytics import YOLO
from config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD as CONFIG_IOU_THRESHOLD

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH_FILE = BASE_DIR / "model_path.txt"
TRAINED_MODEL_PATH = BASE_DIR / "runs" / "detect" / "pen_retrain" / "weights" / "best.pt"
FALLBACK_MODEL_PATH = BASE_DIR / "yolo11s.pt"
DATASET_YAML_PATH = BASE_DIR / "pen-detection.v1i.yolov8" / "data_fixed.yaml"
BLOCKED_OBJECTS = ["wire", "rope", "cable", "cord", "string"]
IOU_THRESHOLD = 0.45
MIN_CONFIDENCE = 0.30
DETECTION_PASSES = [
    {"pass": "640", "imgsz": 640, "conf": 0.30},
    {"pass": "1280", "imgsz": 1280, "conf": 0.30},
    {"pass": "960", "imgsz": 960, "conf": 0.30},
]

app = FastAPI(title="Pen Detector API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: YOLO | None = None
loaded_model_path: Path | None = None
strict_pen_mode = False


def _startup_model_path() -> Path:
    if MODEL_PATH_FILE.is_file():
        saved_path = MODEL_PATH_FILE.read_text(encoding="utf-8").strip()
        if saved_path:
            path = Path(saved_path)
        else:
            path = Path("best.pt")
    else:
        path = Path("best.pt")

    return path if path.is_absolute() else BASE_DIR / path


def validate_startup_model() -> None:
    global model, loaded_model_path, strict_pen_mode

    candidate_path = _startup_model_path()
    if not candidate_path.is_file():
        print(f"ERROR: Model file not found: {candidate_path}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(candidate_path))
    loaded_model_path = candidate_path
    strict_pen_mode = candidate_path == TRAINED_MODEL_PATH
    class_names = normalize_model_names(model.names)

    print(f"Model loaded from: {candidate_path}")
    print(f"Model classes: {len(class_names)}")
    print(f"CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print(f"IOU_THRESHOLD: {CONFIG_IOU_THRESHOLD}")
    print("Server ready at http://localhost:8080")


# Decode uploaded bytes into an EXIF-corrected BGR image for inference.
def decode_upload_image(contents: bytes) -> np.ndarray:
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        with Image.open(BytesIO(contents)) as pil_image:
            corrected = ImageOps.exif_transpose(pil_image).convert("RGB")
            return cv2.cvtColor(np.array(corrected), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError):
        pass

    decoded = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return decoded


# Normalize YOLO class metadata into a stable integer-to-name dictionary.
def normalize_model_names(names: dict | list) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def validate_pen_detection_classes(active_model: YOLO, active_path: Path, require_pen_model: bool) -> dict[int, str]:
    names = normalize_model_names(active_model.names)
    print("YOLOv11 loaded. Classes:", names)
    valid_names = names.get(0, "").lower() == "pen" and all(
        label.lower() in {"pen", "non_pen"} for label in names.values()
    )
    if require_pen_model and not valid_names:
        raise RuntimeError(
            f"Expected trained pen detector weights with class 0='pen' and optional class 1='non_pen', "
            f"got {names} from {active_path}"
        )
    return names


# Load the trained YOLOv11 pen model, or fall back to the base model when weights are missing.
def load_model() -> YOLO:
    global model, loaded_model_path, strict_pen_mode

    if model is not None:
        return model

    candidate_path = TRAINED_MODEL_PATH if TRAINED_MODEL_PATH.is_file() else FALLBACK_MODEL_PATH
    if not candidate_path.is_file():
        raise RuntimeError(
            f"No YOLOv11 model file found. Checked trained weights at {TRAINED_MODEL_PATH} "
            f"and fallback at {FALLBACK_MODEL_PATH}."
        )

    strict_pen_mode = candidate_path == TRAINED_MODEL_PATH
    model = YOLO(str(candidate_path))
    loaded_model_path = candidate_path
    validate_pen_detection_classes(model, candidate_path, require_pen_model=strict_pen_mode)
    return model


# Return whether the currently loaded model is safe for pen inference.
def is_pen_model_ready() -> bool:
    active_model = load_model()
    names = normalize_model_names(active_model.names)
    return (
        strict_pen_mode
        and names.get(0, "").lower() == "pen"
        and all(label.lower() in {"pen", "non_pen"} for label in names.values())
    )


# Clamp a bounding box to image boundaries.
def clamp_bbox(bbox: list[int], image_shape: tuple[int, int, int]) -> list[int]:
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(img_w - 1, int(x1)))
    y1 = max(0, min(img_h - 1, int(y1)))
    x2 = max(0, min(img_w, int(x2)))
    y2 = max(0, min(img_h, int(y2)))
    return [x1, y1, x2, y2]


# Encode an image as a data URL for easy frontend previewing.
def encode_image(image: np.ndarray) -> str | None:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('utf-8')}"


# Change 3: Enhance contrast and sharpen the image before YOLO sees it.
def preprocess_for_pen_detection(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    image_u8 = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8, copy=False))

    alpha = 1.3
    beta = 10
    if hasattr(cv2, "convertScaleAbs"):
        try:
            enhanced = cv2.convertScaleAbs(image_u8, alpha=alpha, beta=beta)
        except Exception:
            enhanced = np.clip(image_u8.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    else:
        enhanced = np.clip(image_u8.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    if hasattr(cv2, "filter2D"):
        try:
            sharpened = cv2.filter2D(enhanced, -1, kernel)
        except Exception:
            sharpened = enhanced
    else:
        sharpened = enhanced

    return np.ascontiguousarray(sharpened, dtype=np.uint8)


# Change 2: Compute IoU so duplicate overlapping boxes can be removed manually.
def compute_iou(box1: list[int], box2: list[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# Change 2: Keep only unique pen boxes after sorting by highest confidence.
def apply_manual_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    sorted_detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    accepted: list[dict] = []

    for detection in sorted_detections:
        overlaps_existing = any(
            compute_iou(detection["bbox"], accepted_detection["bbox"]) > iou_threshold
            for accepted_detection in accepted
        )
        if not overlaps_existing:
            accepted.append(detection)

    return accepted


# Change 4: Add simple confidence labels for the frontend response.
def confidence_label(confidence: float) -> dict:
    if confidence >= 0.70:
        return {"label": "High confidence detection", "color": "green"}
    if confidence >= 0.50:
        return {"label": "Medium confidence", "color": "yellow"}
    if confidence >= 0.30:
        return {"label": "Low confidence - may be incorrect", "color": "orange"}
    return {"label": "Very low - unreliable", "color": "red"}


# Draw accepted pen detections only.
def draw_debug_boxes(image: np.ndarray, detections: list[dict], rejected: list[dict]) -> np.ndarray:
    annotated = image.copy()

    for item in detections:
        if item.get("class", "").strip().lower() != "pen":
            continue
        x1, y1, x2, y2 = item["bbox"]
        conf = float(item["confidence"])
        label = f"pen {conf:.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


# Apply the strict 5-rule filter so only real pens are accepted.
def is_pen(box, image: np.ndarray) -> tuple[bool, list[str]]:
    active_model = load_model()
    names = normalize_model_names(active_model.names)
    cls_name = names.get(int(box.cls[0]), str(int(box.cls[0])))

    failures: list[str] = []

    # Rule 1: class name must be exactly pen.
    if cls_name.lower().strip() != "pen":
        failures.append("not_pen")
        return False, failures

    # Rule 2: aspect ratio and box area filtering.
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1, y1, x2, y2 = clamp_bbox([x1, y1, x2, y2], image.shape)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        failures.append("invalid_bbox")
        return False, failures

    ratio = h / float(w)
    box_area = w * h
    image_area = image.shape[0] * image.shape[1]
    if 0.5 < ratio < 2.0 and box_area < image_area * 0.03:
        failures.append("aspect_ratio_small_box")
    elif not (ratio > 2.5 or ratio < 0.4 or box_area > image_area * 0.03):
        failures.append("aspect_ratio")

    # Rule 3: minimum confidence.
    conf = float(box.conf[0])
    if conf < MIN_CONFIDENCE:
        failures.append("low_confidence")

    crop = image[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        failures.append("empty_crop")
        return False, failures
    crop = np.ascontiguousarray(np.clip(crop, 0, 255).astype(np.uint8, copy=False))

    # Rule 4: dark desaturated wire/cable suppression.
    if hasattr(cv2, "cvtColor") and hasattr(cv2, "COLOR_BGR2HSV"):
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mean_saturation = float(hsv[:, :, 1].mean())
            mean_value = float(hsv[:, :, 2].mean())
            if mean_saturation < 30 and mean_value < 60:
                failures.append("dark_low_saturation")
        except Exception:
            pass
    else:
        hsv = crop.astype(np.uint8)

    # Rule 5: straightness check via Hough lines.
    if (
        hasattr(cv2, "cvtColor")
        and hasattr(cv2, "COLOR_BGR2GRAY")
        and hasattr(cv2, "Canny")
        and hasattr(cv2, "HoughLinesP")
    ):
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=30,
                minLineLength=20,
                maxLineGap=10,
            )
            if lines is None or len(lines) < 1:
                failures.append("no_straight_lines")
        except Exception:
            pass
    else:
        pass

    return len(failures) == 0, failures


# Convert a model prediction into the API response shape for accepted pens.
def format_confirmed_pen(box, bbox: list[int], pass_name: str) -> dict:
    conf = float(box.conf[0])
    label_info = confidence_label(conf)
    payload = {
        "class": "pen",
        "confidence": round(conf, 3),
        "confidence_pct": f"{conf * 100:.1f}%",
        "confidence_label": label_info["label"],
        "confidence_color": label_info["color"],
        "bbox": bbox,
        "reliable": conf >= 0.70,
        "pass": pass_name,
    }
    return payload


# Convert a rejected model prediction into a debug-friendly payload.
def format_rejected_detection(box, bbox: list[int], reason: list[str]) -> dict:
    active_model = load_model()
    names = normalize_model_names(active_model.names)
    cls_name = names.get(int(box.cls[0]), str(int(box.cls[0])))
    conf = float(box.conf[0])
    return {
        "class": cls_name,
        "confidence_pct": f"{conf * 100:.1f}%",
        "bbox": bbox,
        "reason": ",".join(reason) if reason else "failed_pen_filter",
    }


# Change 1: Run YOLO at three image sizes and collect every pen detection.
def collect_multi_resolution_pen_detections(active_model: YOLO, image: np.ndarray) -> tuple[list[dict], list[dict]]:
    names = normalize_model_names(active_model.names)
    confirmed_pens: list[dict] = []
    rejected: list[dict] = []

    for pass_config in DETECTION_PASSES:
        results = active_model.predict(
            image,
            imgsz=pass_config["imgsz"],
            conf=0.30,
            iou=0.45,
            verbose=False,
        )

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                bbox = clamp_bbox(list(map(int, box.xyxy[0])), image.shape)
                cls_name = names.get(int(box.cls[0]), str(int(box.cls[0]))).lower().strip()

                if cls_name == "pen":
                    # Call the 5-rule is_pen() filter before accepting
                    passed, reasons = is_pen(box, image)
                    if passed:
                        confirmed_pens.append(
                            format_confirmed_pen(box, bbox, pass_name=pass_config["pass"])
                        )
                    else:
                        # is_pen() rejected it — log reason for debugging
                        rejected.append(
                            format_rejected_detection(box, bbox, reasons)
                        )
                else:
                    continue

    return confirmed_pens, rejected


# Run the multi-resolution pen-only detection pipeline on one uploaded image.
@app.post("/detect")
async def detect_pen(file: UploadFile = File(...)):
    active_model = load_model()
    if not is_pen_model_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Trained YOLOv11 pen detector weights are not available. "
                f"Expected {TRAINED_MODEL_PATH} with class 0='pen' and optional class 1='non_pen'."
            ),
        )

    contents = await file.read()
    image = decode_upload_image(contents)
    processed_image = preprocess_for_pen_detection(image)
    all_pen_detections, rejected = collect_multi_resolution_pen_detections(active_model, processed_image)
    confirmed_pens = apply_manual_nms(all_pen_detections, iou_threshold=0.45)
    best_detection = confirmed_pens[0] if confirmed_pens else None

    annotated_image = encode_image(draw_debug_boxes(image, confirmed_pens, rejected))

    return JSONResponse(
        {
            "pen_detected": len(confirmed_pens) > 0,
            "detected": len(confirmed_pens) > 0,
            "count": len(confirmed_pens),
            "model_version": "YOLOv8",
            "model_path": str(loaded_model_path) if loaded_model_path else str(TRAINED_MODEL_PATH),
            "confidence": best_detection["confidence"] if best_detection else 0,
            "confidence_pct": best_detection["confidence_pct"] if best_detection else "0.0%",
            "confidence_label": best_detection["confidence_label"] if best_detection else "Very low - unreliable",
            "confidence_color": best_detection["confidence_color"] if best_detection else "red",
            "best_detection": best_detection,
            "detections": confirmed_pens,
            "rejected": rejected,
            "summary": "Pen found!" if confirmed_pens else "No pen detected.",
            "annotatedImage": annotated_image,
        }
    )


# Report backend liveness and the currently loaded YOLOv11 class metadata.
@app.get("/health")
def health() -> dict:
    active_model = load_model()
    return {
        "status": "running",
        "model": "YOLOv8",
        "classes": normalize_model_names(active_model.names),
        "version": "2.1",
    }


# Report which weights are loaded and which objects are explicitly blocked.
@app.get("/model-info")
def model_info() -> dict:
    active_model = load_model()
    return {
        "model_path": str(loaded_model_path) if loaded_model_path else str(TRAINED_MODEL_PATH),
        "class_names": normalize_model_names(active_model.names),
        "only_detects": ["pen"],
        "blocked_objects": BLOCKED_OBJECTS,
    }


# Load the YOLOv11 model as the service starts so startup failures happen early.
@app.on_event("startup")
def startup_event() -> None:
    validate_startup_model()
