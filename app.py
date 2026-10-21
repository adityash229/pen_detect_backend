from __future__ import annotations

import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from opencv_compat import ensure_cv2_compat

ensure_cv2_compat()

try:
    import mediapipe as mp
except Exception:
    mp = None

from ultralytics import YOLO

from feedback import get_feedback, init_db, insert_feedback
from config import DEFAULT_MODEL_PATH, HOST, LARGE_IMAGE_THRESHOLD, MODEL_PATH, PORT, TILE_OVERLAP, TILE_SIZE
from model import predict as predict_pen_boxes
from model import predict_tiled as predict_tiled_pen_boxes
from report_generator import REPORTS_DIR, UPLOADS_DIR, generate_report, get_all_reports, mark_audit_done
from yolo11_local import (
    ACCURATE_MODEL_NAME as YOLO11_ACCURATE_MODEL_NAME,
    FAST_MODEL_NAME as YOLO11_FAST_MODEL_NAME,
    detect_pen as detect_pen_yolo11,
    load_yolo11_model,
    verify_ultralytics_installation,
)
from routes.detect import router as detect_bp
from routes.label import router as label_bp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend" / "project-code" / "dist"
MODEL_PATH_FILE = BASE_DIR / "model_path.txt"
DEBUG_DIR = BASE_DIR / "debug_outputs"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_CACHE_DIR = DEBUG_DIR / "uploads"
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
LAST_DETECTION_IMAGE = STATIC_DIR / "last_detection.jpg"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_UPLOAD_IMAGE = UPLOADS_DIR / "image1.jpg"

DEFAULT_PASS1_CONF = 0.30
SECOND_PASS_CONF = 0.10
FALLBACK_FULL_CONF = 0.05
HIGH_RES_CONF = 0.10
DEFAULT_IOU = 0.45
HIGH_CONFIDENCE_RETURN = 0.40
RELIABLE_CONFIDENCE = 0.65
HAND_PADDING_RATIO = 0.20
BLOCKED_LABELS = {"wire", "cable", "rope", "cord", "string", "usb", "charger", "earphone", "headphone"}
MULTI_RESOLUTION_PASSES = [
    {"stage": "multi_res_640", "imgsz": 640, "conf": 0.15},
    {"stage": "multi_res_1280", "imgsz": 1280, "conf": 0.10},
    {"stage": "multi_res_960", "imgsz": 960, "conf": 0.12},
]

pen_model: YOLO | None = None
pen_class_id: int | None = None
pen_model_path: Path | None = None


class FeedbackSchema(BaseModel):
    confidence: float
    status: str
    user_label: str


def _iter_model_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(path)

    model_path_value = Path(MODEL_PATH)
    preferred_paths = [
        BASE_DIR / "runs" / "detect" / "pen_retrain" / "weights" / "best.pt",
        model_path_value,
    ]

    for path in preferred_paths:
        add(path)

    saved_model_path = _read_saved_model_path()
    if saved_model_path is not None:
        add(saved_model_path)

    for path in sorted((BASE_DIR / "runs").glob("**/weights/best.pt"), reverse=True):
        add(path)

    return candidates


def _normalize_model_names(names: dict | list) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def _score_model_names(names: dict[int, str]) -> tuple[int, int]:
    exact_pen = sum(1 for label in names.values() if label.lower() == "pen")
    contains_pen = sum(1 for label in names.values() if "pen" in label.lower())
    return exact_pen, contains_pen


def _candidate_priority(candidate: Path, saved_model_path: Path | None) -> tuple[int, int]:
    resolved_candidate = candidate.resolve(strict=False)
    matches_saved_path = (
        1 if saved_model_path is not None and resolved_candidate == saved_model_path.resolve(strict=False) else 0
    )
    return matches_saved_path, int(candidate.stat().st_mtime)


def _read_saved_model_path() -> Path | None:
    if not MODEL_PATH_FILE.is_file():
        return None

    saved_path = MODEL_PATH_FILE.read_text(encoding="utf-8").strip()
    return Path(saved_path) if saved_path else None


def _resolve_pen_class_id(names: dict[int, str]) -> int:
    for cls_id, label in names.items():
        if label.lower() == "pen":
            return cls_id
    for cls_id, label in names.items():
        if "pen" in label.lower():
            return cls_id
    raise RuntimeError(f"Loaded model does not contain an exact 'pen' class. model.names={names}")


def _load_model() -> YOLO:
    global pen_model, pen_class_id, pen_model_path
    if pen_model is not None:
        return pen_model

    saved_model_path = _read_saved_model_path()
    print("[startup] Pen backend model resolution")
    print(f"[startup] model_path.txt: {saved_model_path if saved_model_path else 'not set'}")
    print(f"[startup] env MODEL_PATH: {MODEL_PATH}")

    model_candidates = [path for path in _iter_model_candidates() if path.is_file()]
    if not model_candidates:
        raise FileNotFoundError("Model not found. Checked: " + ", ".join(str(path) for path in _iter_model_candidates()))

    best_score = (-1, -1)
    best_priority = (-1, -1)
    selected_model: YOLO | None = None
    selected_names: dict[int, str] | None = None
    selected_path: Path | None = None

    for candidate in model_candidates:
        try:
            model = YOLO(str(candidate))
            names = _normalize_model_names(model.names)
            score = _score_model_names(names)
            priority = _candidate_priority(candidate, saved_model_path)
            print(f"[startup] candidate model: {candidate} | names={names} | score={score} | priority={priority}")
            if score > best_score or (score == best_score and priority > best_priority):
                best_score = score
                best_priority = priority
                selected_model = model
                selected_names = names
                selected_path = candidate
        except Exception as exc:
            print(f"[startup] failed to load candidate {candidate}: {exc}")

    if selected_model is None or selected_names is None or selected_path is None:
        raise RuntimeError("Unable to load any candidate YOLO model.")

    pen_class_id = _resolve_pen_class_id(selected_names)
    pen_model = selected_model
    pen_model_path = selected_path

    print("[startup] Model loaded successfully")
    print(f"[startup] selected model path: {selected_path.resolve(strict=False)}")
    if saved_model_path is not None:
        matches_saved_path = selected_path.resolve(strict=False) == saved_model_path.resolve(strict=False)
        print(f"[startup] matches model_path.txt: {matches_saved_path}")
    print(f"[startup] model.names: {selected_names}")
    print(f"[startup] pen_class_id: {pen_class_id}")
    return pen_model


def _decode_upload_image(contents: bytes) -> np.ndarray:
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        with Image.open(BytesIO(contents)) as image:
            corrected = ImageOps.exif_transpose(image).convert("RGB")
            return cv2.cvtColor(np.array(corrected), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError):
        pass

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image


def _confidence_pct(confidence: float) -> str:
    return f"{confidence * 100:.1f}%"


# CHANGE 4: Convert confidence into clear frontend labels and colors.
def _confidence_label(confidence: float) -> dict[str, str]:
    if confidence >= 0.70:
        return {"label": "High confidence detection", "color": "green"}
    if confidence >= 0.50:
        return {"label": "Medium confidence", "color": "yellow"}
    if confidence >= 0.30:
        return {"label": "Low confidence - may be incorrect", "color": "orange"}
    return {"label": "Very low - unreliable", "color": "red"}


def _save_debug_image(image: np.ndarray, prefix: str) -> str | None:
    path = DEBUG_DIR / f"{prefix}_{uuid4().hex}.jpg"
    return str(path) if cv2.imwrite(str(path), image) else None


def _encode_image_to_base64(image: np.ndarray) -> str | None:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _ensure_sample_upload_image() -> None:
    """Create a tiny sample image so /get-image is immediately browser-testable."""
    if SAMPLE_UPLOAD_IMAGE.is_file():
        return
    sample = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.putText(
        sample,
        "image1.jpg",
        (38, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(SAMPLE_UPLOAD_IMAGE), sample)


def _upload_url(request: Request, filename: str) -> str:
    """Build an absolute URL for a file served by the /uploads mount."""
    return str(request.url_for("uploads", path=filename))


def _jpeg_response(image: np.ndarray) -> Response:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode image")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


def _clamp_bbox(bbox: list[float], image_shape: tuple[int, int, int]) -> list[int]:
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(img_w - 1, int(round(x1))))
    y1 = max(0, min(img_h - 1, int(round(y1))))
    x2 = max(0, min(img_w, int(round(x2))))
    y2 = max(0, min(img_h, int(round(y2))))
    return [x1, y1, x2, y2]


def _bbox_with_padding(bbox: list[int], image_shape: tuple[int, int, int], padding_ratio: float = 0.12) -> list[int]:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    return _clamp_bbox([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], image_shape)


def _format_detection(
    bbox: list[int],
    confidence: float,
    source: str,
    label: str = "pen",
    rejection_reason: list[str] | None = None,
) -> dict:
    label_info = _confidence_label(confidence)
    return {
        "class": label,
        "label": label,
        "confidence": round(confidence, 4),
        "confidence_pct": _confidence_pct(confidence),
        "confidence_label": label_info["label"],
        "confidence_color": label_info["color"],
        "bbox": bbox,
        "is_reliable": confidence >= 0.70,
        "source": source,
        "rejection_reason": rejection_reason or [],
    }


def _label_is_pen(label: str, cls_id: int | None = None) -> bool:
    normalized = label.strip().lower()
    if normalized == "pen":
        return True
    return pen_class_id is not None and cls_id == pen_class_id


def _draw_detections(image: np.ndarray, detections: list[tuple[list[float], float]]) -> np.ndarray:
    annotated = image.copy()
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box, score in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        label = f"pen {float(score):.0%}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(15, y1 - 5)),
            font,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    count_label = f"Pens: {len(detections)}"
    (count_w, count_h), count_baseline = cv2.getTextSize(count_label, font, 0.7, 2)
    cv2.rectangle(
        annotated,
        (10, 10),
        (count_w + 24, count_h + count_baseline + 22),
        (30, 30, 30),
        -1,
    )
    cv2.putText(
        annotated,
        count_label,
        (16, count_h + 16),
        font,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _draw_rejected(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    return image.copy()


def _run_model(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    iou: float,
    stage_name: str,
    imgsz: int | None = None,
    offset: tuple[int, int] = (0, 0),
) -> tuple[list[dict], list[str]]:
    kwargs = {"conf": 0.30, "iou": 0.45, "verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz

    results = model.predict(source=image, **kwargs)
    names = _normalize_model_names(model.names)
    detections: list[dict] = []
    logs: list[str] = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw_label = names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = _clamp_bbox(box.xyxy[0].tolist(), image.shape)
            x_off, y_off = offset
            bbox = [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off]

            logs.append(f"{raw_label}@{_confidence_pct(confidence)}")
            print(f"[{stage_name}] Detected: {raw_label} | Conf: {confidence:.4f}")

            lowered = raw_label.lower()
            if lowered in BLOCKED_LABELS:
                continue

            if not _label_is_pen(raw_label, cls_id):
                continue

            detections.append(_format_detection(bbox, confidence, stage_name))

    print(f"[{stage_name}] raw detection count: {len(logs)}")
    print(f"[{stage_name}] detections: {logs}")
    return detections, logs


# CHANGE 3: Increase contrast and gently sharpen the image before detection.
def _preprocess_for_detection(image: np.ndarray) -> np.ndarray:
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


def _analyze_pen_rules(box: list[int], image: np.ndarray) -> list[str]:
    x1, y1, x2, y2 = _clamp_bbox(box, image.shape)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return ["empty_crop", "invalid_bbox"]

    img_h, img_w = image.shape[:2]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    image_area = img_h * img_w
    box_area = w * h
    ratio = h / max(w, 1)

    failures: list[str] = []

    # Rule 1 + prompt square-box filter.
    if ((0.4 < ratio < 1.5) and box_area < image_area * 0.05) or (0.7 <= ratio <= 1.4):
        failures.append("aspect_ratio")

    # Rule 2.
    if box_area < image_area * 0.01:
        failures.append("min_size")

    # Rule 3.
    hsv = np.array(Image.fromarray(crop[:, :, ::-1]).convert("HSV"))
    hue_std = float(np.std(hsv[:, :, 0]))
    value_mean = float(np.mean(hsv[:, :, 2]))
    if hue_std < 15 and value_mean < 50:
        failures.append("dark_low_color_variance")

    # Rule 4.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gy, gx = np.gradient(gray.astype(np.float32))
    magnitude = np.hypot(gx, gy)
    edge_mask = magnitude > max(20.0, float(np.mean(magnitude) + np.std(magnitude)))
    edge_pixels = int(np.count_nonzero(edge_mask))
    straight_edge_ratio = 0.0
    if edge_pixels > 0:
        dominant_axis = np.maximum(np.abs(gx), np.abs(gy))
        aligned_edges = edge_mask & (dominant_axis >= (magnitude * 0.85))
        straight_edge_ratio = float(np.count_nonzero(aligned_edges)) / float(edge_pixels)
    if straight_edge_ratio < 0.3:
        failures.append("edge_straightness")

    return failures


def is_real_pen(box: list[int], image: np.ndarray) -> bool:
    failures = _analyze_pen_rules(box, image)
    return len(set(failures)) < 2


def _filter_pen_candidates(candidates: list[dict], image: np.ndarray) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []

    for candidate in candidates:
        if candidate["class"].lower() != "pen":
            continue

        if not is_real_pen(candidate["bbox"], image):
            candidate["rejection_reason"] = _analyze_pen_rules(candidate["bbox"], image)
            rejected.append(candidate)
            continue

        accepted.append(candidate)

    return accepted, rejected


def _best_detection(detections: list[dict]) -> dict | None:
    return max(detections, key=lambda item: item["confidence"], default=None)


# CHANGE 2: Calculate box overlap so duplicate detections can be removed.
def _compute_iou(box1: list[int], box2: list[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# CHANGE 2: Keep highest confidence boxes and skip duplicates with IoU over 0.5.
def _dedupe_detections(detections: list[dict], iou_threshold: float = 0.5, limit: int | None = None) -> list[dict]:
    kept: list[dict] = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if any(_compute_iou(detection["bbox"], existing["bbox"]) > iou_threshold for existing in kept):
            continue
        kept.append(detection)
        if limit is not None and len(kept) >= limit:
            break
    return kept


def _crop_for_second_pass(image: np.ndarray, bbox: list[int]) -> tuple[np.ndarray, tuple[int, int]]:
    x1, y1, x2, y2 = _bbox_with_padding(bbox, image.shape)
    crop = image[y1:y2, x1:x2]
    resized = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)
    return resized, (x1, y1)


def _remap_crop_detection(crop_detection: dict, origin: tuple[int, int], original_bbox: list[int]) -> dict:
    x_origin, y_origin = origin
    ox1, oy1, ox2, oy2 = original_bbox
    crop_w = max(1, ox2 - ox1)
    crop_h = max(1, oy2 - oy1)

    cx1, cy1, cx2, cy2 = crop_detection["bbox"]
    scale_x = crop_w / 640.0
    scale_y = crop_h / 640.0

    remapped = [
        int(round(x_origin + (cx1 * scale_x))),
        int(round(y_origin + (cy1 * scale_y))),
        int(round(x_origin + (cx2 * scale_x))),
        int(round(y_origin + (cy2 * scale_y))),
    ]

    return _format_detection(remapped, float(crop_detection["confidence"]), "crop_second_pass")


def _detect_hand_regions(image: np.ndarray) -> list[list[int]]:
    if mp is None or not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        return []

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
    finally:
        hands.close()

    if not results.multi_hand_landmarks:
        return []

    img_h, img_w = image.shape[:2]
    regions: list[list[int]] = []
    for landmarks in results.multi_hand_landmarks:
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        x1 = int(min(xs) * img_w)
        y1 = int(min(ys) * img_h)
        x2 = int(max(xs) * img_w)
        y2 = int(max(ys) * img_h)
        regions.append(_bbox_with_padding([x1, y1, x2, y2], image.shape, padding_ratio=HAND_PADDING_RATIO))
    return regions


def _run_hand_focus_fallback(
    model: YOLO,
    image: np.ndarray,
    min_conf: float,
    hand_regions: list[list[int]] | None = None,
) -> tuple[list[dict], int]:
    hand_regions = hand_regions if hand_regions is not None else _detect_hand_regions(image)
    passes_used = 0
    accepted: list[dict] = []

    for index, region in enumerate(hand_regions):
        x1, y1, x2, y2 = region
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        passes_used += 1
        stage = f"hand_focus_{index}"
        detections, _ = _run_model(model, crop, conf=min_conf, iou=DEFAULT_IOU, stage_name=stage, offset=(x1, y1))
        filtered, _ = _filter_pen_candidates(detections, image)
        accepted.extend(filtered)

    return accepted, passes_used


# CHANGE 1: Run the same image at 640, 1280, and 960, then combine pen detections.
def _run_multi_resolution_detection(model: YOLO, image: np.ndarray) -> tuple[list[dict], list[dict], list[str], int, list[str]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    stages: list[str] = []
    pass_summaries: list[str] = []

    for pass_config in MULTI_RESOLUTION_PASSES:
        stage = str(pass_config["stage"])
        stages.append(stage)
        all_detections, logs = _run_model(
            model,
            image,
            conf=float(pass_config["conf"]),
            iou=DEFAULT_IOU,
            stage_name=stage,
            imgsz=int(pass_config["imgsz"]),
        )

        pen_candidates = [
            item for item in all_detections
            if item["class"].strip().lower() == "pen" and not item.get("rejection_reason")
        ]
        non_pen_rejected = [
            item for item in all_detections
            if item["class"].strip().lower() != "pen" or item.get("rejection_reason")
        ]
        filtered_accepted, filtered_rejected = _filter_pen_candidates(pen_candidates, image)
        accepted.extend(filtered_accepted)
        rejected.extend(non_pen_rejected)
        rejected.extend(filtered_rejected)
        pass_summaries.append(f"{stage}: {logs[0] if logs else 'none'}")

    return accepted, rejected, stages, len(MULTI_RESOLUTION_PASSES), pass_summaries


# Decode uploaded bytes with OpenCV for the rewritten endpoint.
def _decode_upload_image_cv2(contents: bytes) -> np.ndarray:
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image


# Extract only exact pen detections from one YOLO prediction result.
def _collect_pen_boxes_from_results(
    model: YOLO,
    results,
    image_shape: tuple[int, int, int],
    source: str,
    offset: tuple[int, int] = (0, 0),
) -> list[dict]:
    names = _normalize_model_names(model.names)
    x_offset, y_offset = offset
    detections: list[dict] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, str(cls_id)).strip().lower()
            if class_name != "pen":
                continue

            x1, y1, x2, y2 = _clamp_bbox(box.xyxy[0].tolist(), image_shape)
            bbox = [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]
            confidence = float(box.conf[0])
            detections.append(_format_detection(bbox, confidence, source))

    return detections


# Run full-image detection using the exact requested resolutions and thresholds.
def _run_requested_full_image_passes(model: YOLO, processed_image: np.ndarray) -> list[dict]:
    passes = [
        ("FULL_640", 0.10, 640),
        ("FULL_960", 0.08, 960),
        ("FULL_1280", 0.08, 1280),
    ]
    detections: list[dict] = []

    for source, conf, imgsz in passes:
        results = model.predict(processed_image, conf=0.30, iou=0.45, imgsz=imgsz, max_det=50, verbose=False)
        detections.extend(_collect_pen_boxes_from_results(model, results, processed_image.shape, source))

    return detections


# Run tiled detection on a 2x2 grid and remap tile boxes to full-image coordinates.
def _run_requested_tile_scan(model: YOLO, processed_image: np.ndarray) -> list[dict]:
    height, width = processed_image.shape[:2]
    x_mid = width // 2
    y_mid = height // 2
    tiles = [
        (0, 0, x_mid, y_mid),
        (x_mid, 0, width, y_mid),
        (0, y_mid, x_mid, height),
        (x_mid, y_mid, width, height),
    ]
    detections: list[dict] = []

    for x1, y1, x2, y2 in tiles:
        tile = processed_image[y1:y2, x1:x2]
        if tile.size == 0:
            continue
        results = model.predict(tile, conf=0.30, iou=0.45, imgsz=640, max_det=50, verbose=False)
        detections.extend(_collect_pen_boxes_from_results(model, results, tile.shape, "TILE_SCAN", offset=(x1, y1)))

    return detections


# Sort detections visually from left-to-right within each row and assign pen IDs.
def _sort_and_number_pens(detections: list[dict]) -> list[dict]:
    def row_col_key(item: dict) -> tuple[int, float]:
        x1, y1, x2, y2 = item["bbox"]
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        return int(y_center // 80), x_center

    sorted_detections = sorted(detections, key=row_col_key)
    for index, detection in enumerate(sorted_detections, start=1):
        detection["pen_id"] = f"pen_{index}"
        detection["reliable"] = float(detection["confidence"]) >= 0.70
    return sorted_detections


# Reject wire-like false positives before final pen counting.
def _is_likely_real_pen(detection: dict, image_shape: tuple[int, int, int]) -> bool:
    confidence = float(detection["confidence"])
    x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
    img_h, img_w = image_shape[:2]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area_ratio = (width * height) / max(1, img_w * img_h)
    touches_edge = x1 <= 3 or y1 <= 3 or x2 >= img_w - 3 or y2 >= img_h - 3

    # Most wire/cable false positives in this project are below 30%.
    if confidence < 0.30:
        return False

    # Large low-confidence boxes are usually wires, laptop edges, or background regions.
    if confidence < 0.55 and area_ratio > 0.18:
        return False

    # Edge-touching low-confidence detections are commonly partial wires entering the photo.
    if confidence < 0.55 and touches_edge:
        return False

    # A box that is both very wide/tall and low confidence is more likely a cable sweep.
    long_side_ratio = max(width / max(height, 1), height / max(width, 1))
    if confidence < 0.45 and long_side_ratio > 6.0:
        return False

    return True


# Keep only high-precision pen candidates after all YOLO passes.
def _filter_likely_real_pens(detections: list[dict], image_shape: tuple[int, int, int]) -> list[dict]:
    return [detection for detection in detections if _is_likely_real_pen(detection, image_shape)]


# Draw each pen in a different color and save the latest debug image.
def _save_last_detection_image(image: np.ndarray, detections: list[dict]) -> None:
    detection_tuples = [(detection["bbox"], float(detection["confidence"])) for detection in detections]
    annotated = _draw_detections(image, detection_tuples)
    cv2.imwrite(str(LAST_DETECTION_IMAGE), annotated)


def _attach_detection_report(
    response: dict,
    image_bytes: bytes,
    user_email: str,
    organization: str,
    department: str,
    assigned_to: str,
) -> dict:
    """Attach a file-based detection report without breaking detection responses."""
    try:
        detection_result = {
            "pen_detected": bool(response.get("pen_detected", False)),
            "confidence": float(response.get("confidence_raw", response.get("confidence", 0.0)) or 0.0),
            "pen_count": int(response.get("count", response.get("pen_count", 0)) or 0),
        }
        response["report"] = generate_report(
            detection_result=detection_result,
            image_bytes=image_bytes,
            user_email=user_email,
            organization=organization,
            department=department,
            assigned_to=assigned_to,
        )
    except Exception as exc:
        print(f"[report_generator] failed to attach report: {exc}")
        response["report_error"] = str(exc)
    return response


def _response_from_detection_run(
    image: np.ndarray,
    file_name: str,
    accepted: list[dict],
    rejected: list[dict],
    stages: list[str],
    passes_used: int,
    hand_box: list[int] | None = None,
) -> dict:
    accepted = _dedupe_detections(accepted)
    best = _best_detection(accepted)
    debug_canvas = image.copy()
    if rejected:
        debug_canvas = _draw_rejected(debug_canvas, rejected)
    if accepted:
        debug_canvas = _draw_detections(
            debug_canvas,
            [(detection["bbox"], float(detection["confidence"])) for detection in accepted],
        )

    debug_path = _save_debug_image(debug_canvas, "pen_detect")
    annotated_base64 = _encode_image_to_base64(debug_canvas)

    if accepted:
        final_status = "hand_focus" if any(stage.startswith("hand_focus") for stage in stages) else "high_confidence"
        final_message = "Pen detected."
        final_conf = float(best["confidence"])
    else:
        final_status = "not_detected"
        final_message = "Pen not detected."
        final_conf = 0.0

    label_info = _confidence_label(final_conf)

    return {
        "detected": bool(accepted),
        "pen_detected": bool(accepted),
        "count": len(accepted),
        "confidence": _confidence_pct(final_conf),
        "confidence_pct": _confidence_pct(final_conf),
        "confidence_raw": round(final_conf, 4),
        "confidence_label": label_info["label"],
        "confidence_color": label_info["color"],
        "reliable": final_conf >= 0.70,
        "passes_used": passes_used,
        "detections": accepted,
        "rejected": rejected,
        "status": final_status,
        "message": final_message,
        "pen_box": {
            "x": best["bbox"][0],
            "y": best["bbox"][1],
            "width": best["bbox"][2] - best["bbox"][0],
            "height": best["bbox"][3] - best["bbox"][1],
            "label": "pen",
            "confidence": round(final_conf, 4),
        } if best else None,
        "hand_box": {
            "x": hand_box[0],
            "y": hand_box[1],
            "width": hand_box[2] - hand_box[0],
            "height": hand_box[3] - hand_box[1],
            "label": "hand_focus",
        } if hand_box else None,
        "object_box": None,
        "annotated_image": annotated_base64,
        "annotatedImage": f"data:image/jpeg;base64,{annotated_base64}" if annotated_base64 else None,
        "debug_image_path": debug_path,
        "stages": stages,
        "model_path": str(pen_model_path) if pen_model_path else None,
        "file_name": file_name,
    }


@app.on_event("startup")
def startup_event() -> None:
    _load_model()
    verify_ultralytics_installation()
    try:
        load_yolo11_model(YOLO11_FAST_MODEL_NAME)
    except Exception as exc:
        print(f"[YOLO11] startup warning: {exc}")


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    mode: str = Form("direct"),
    zoom_to_hand: bool = Form(False),
    detect_object_in_hand: bool = Form(False),
    min_conf: float = Form(DEFAULT_PASS1_CONF),
    user_email: str = Form("unknown@example.com"),
    organization: str = Form("Unknown Org"),
    department: str = Form("Unknown Dept"),
    assigned_to: str = Form("admin@example.com"),
):
    try:
        model = _load_model()
        contents = await file.read()
        image = _decode_upload_image_cv2(contents)
        detections = predict_pen_boxes(model, image)
        annotated = _draw_detections(image, detections)
        cv2.imwrite(str(LAST_DETECTION_IMAGE), annotated)

        best_detection = max(detections, key=lambda item: item[1], default=None)
        highest_confidence_raw = float(best_detection[1]) if best_detection else 0.0
        best_bbox = [int(round(value)) for value in best_detection[0]] if best_detection else None
        annotated_base64 = _encode_image_to_base64(annotated)

        response = {
            "detected": bool(detections),
            "pen_detected": bool(detections),
            "pen_count": len(detections),
            "count": len(detections),
            "total_pens_found": len(detections),
            "highest_confidence": _confidence_pct(highest_confidence_raw),
            "confidence": highest_confidence_raw,
            "confidence_raw": round(highest_confidence_raw, 4),
            "confidence_pct": _confidence_pct(highest_confidence_raw),
            "status": "detected" if detections else "not_detected",
            "message": f"{len(detections)} pen(s) detected." if detections else "Pen not detected in the full image.",
            "reliable": highest_confidence_raw >= 0.70,
            "detection_passes": ["predict"],
            "stages": ["predict"],
            "pen_box": {
                "x": best_bbox[0],
                "y": best_bbox[1],
                "width": best_bbox[2] - best_bbox[0],
                "height": best_bbox[3] - best_bbox[1],
                "label": "pen",
                "confidence": round(highest_confidence_raw, 4),
            } if best_bbox else None,
            "hand_box": None,
            "object_box": None,
            "annotated_image": annotated_base64,
            "annotatedImage": f"data:image/jpeg;base64,{annotated_base64}" if annotated_base64 else None,
            "detections": [
                {
                    "pen_id": f"pen_{index}",
                    "confidence": float(score),
                    "confidence_pct": _confidence_pct(float(score)),
                    "reliable": float(score) >= 0.70,
                    "bbox": [int(round(value)) for value in box],
                }
                for index, (box, score) in enumerate(detections, start=1)
            ],
        }
        return _attach_detection_report(
            response=response,
            image_bytes=contents,
            user_email=user_email,
            organization=organization,
            department=department,
            assigned_to=assigned_to,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/detect_large")
async def detect_large(file: UploadFile = File(...)):
    """Use this for meeting hall, CCTV, or drone images."""
    try:
        model = _load_model()
        contents = await file.read()
        image = _decode_upload_image_cv2(contents)
        detections = predict_tiled_pen_boxes(model, image, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
        result = _draw_detections(image, detections)
        return _jpeg_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/detect_auto")
async def detect_auto(file: UploadFile = File(...)):
    """Auto-select tiled or normal inference based on image size."""
    try:
        model = _load_model()
        contents = await file.read()
        image = _decode_upload_image_cv2(contents)
        image_h, image_w = image.shape[:2]
        if image_h * image_w > LARGE_IMAGE_THRESHOLD:
            detections = predict_tiled_pen_boxes(model, image, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
        else:
            detections = predict_pen_boxes(model, image)
        result = _draw_detections(image, detections)
        return _jpeg_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/last-detection-image")
def last_detection_image():
    if not LAST_DETECTION_IMAGE.is_file():
        raise HTTPException(status_code=404, detail="No detection image has been saved yet.")
    return FileResponse(str(LAST_DETECTION_IMAGE), media_type="image/jpeg")


@app.post("/detect-yolo11")
async def detect_yolo11(
    file: UploadFile = File(...),
    yolo11_model: str = Form(YOLO11_FAST_MODEL_NAME, alias="model_name"),
    conf: float = Form(0.25),
    iou: float = Form(0.5),
):
    try:
        contents = await file.read()
        image = _decode_upload_image(contents)
        upload_name = Path(file.filename or "upload.jpg").name
        cached_path = UPLOAD_CACHE_DIR / f"{uuid4().hex}_{upload_name}"
        cached_path.write_bytes(contents)

        # Validate that the selected YOLOv11 model can be loaded locally.
        load_yolo11_model(yolo11_model)
        result = detect_pen_yolo11(cached_path, model_path=yolo11_model, conf=conf, iou=iou)

        if result is None:
            return {
                "detected": False,
                "pen_detected": False,
                "message": "No 'pen' detection returned by YOLOv11. Use custom YOLOv11 pen weights for pen inference.",
                "model_name": yolo11_model,
                "result": None,
            }

        annotated = _draw_detections(image, [(result["bbox"], float(result["confidence"]))])
        debug_path = _save_debug_image(annotated, "yolo11_detect")
        annotated_base64 = _encode_image_to_base64(annotated)

        return {
            "detected": True,
            "pen_detected": True,
            "message": "YOLOv11 pen detection complete.",
            "model_name": yolo11_model,
            "result": result,
            "annotated_image": annotated_base64,
            "annotatedImage": f"data:image/jpeg;base64,{annotated_base64}" if annotated_base64 else None,
            "debug_image_path": debug_path,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback")
def feedback(data: FeedbackSchema) -> dict:
    insert_feedback(data.confidence, data.status, data.user_label)
    return {"message": "Feedback saved"}


@app.get("/feedback")
def fetch_feedback():
    return get_feedback()


@app.get("/reports")
def reports() -> list[dict]:
    return get_all_reports()


@app.post("/reports/{breach_id}/audit")
def audit_report(breach_id: str) -> dict:
    try:
        return mark_audit_done(breach_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/reports/{breach_id}/generate")
def generate_json_report(breach_id: str, body: dict | None = Body(default=None)):
    report_path = REPORTS_DIR / f"{breach_id}.json"

    if report_path.is_file():
        return JSONResponse(json.loads(report_path.read_text(encoding="utf-8")))

    now = datetime.now()
    scan_id = (body or {}).get("scan_id") or breach_id
    report = {
        "user_email": "unknown@example.com",
        "date": now.strftime("%Y-%m-%d"),
        "detected_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "breach_type": "pen_detected",
        "photo_url": "",
        "thumbnail_photo_url": "",
        "organization": "Unknown Org",
        "department": "Unknown Dept",
        "assigned_to": "admin@example.com",
        "image_save": True,
        "audit_done": False,
        "breach_id": scan_id,
    }

    return JSONResponse(report)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/get-image")
def get_image(request: Request) -> dict[str, str]:
    _ensure_sample_upload_image()
    return {"image_url": _upload_url(request, "image1.jpg")}


app.include_router(detect_bp, prefix="/api")
app.include_router(label_bp, prefix="/api")


app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR), html=False), name="uploads")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
