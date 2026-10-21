from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from opencv_compat import ensure_cv2_compat

ensure_cv2_compat()

import ultralytics
from ultralytics import YOLO


DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.5
FAST_MODEL_NAME = "yolo11n.pt"
ACCURATE_MODEL_NAME = "yolo11s.pt"
YOLO11_DATASET_YAML = Path("pen-detection.v1i.yolov8/data_fixed.yaml")
YOLO11_RUNS_DIR = Path("runs/detect")
YOLO11_DEFAULT_TRAIN_RUN = "pen_retrain"
YOLO11_MODEL_PATH_FILE = Path("yolo11_model_path.txt")


def verify_ultralytics_installation() -> str:
    version = ultralytics.__version__
    print(f"[YOLO11] ultralytics version: {version}")
    return version


def _as_path(path_or_name: str | Path) -> Path:
    return path_or_name if isinstance(path_or_name, Path) else Path(path_or_name)


def _ensure_existing_file(path_or_name: str | Path, kind: str) -> Path:
    path = _as_path(path_or_name)
    if not path.exists():
        raise FileNotFoundError(
            f"{kind} not found: {path}. "
            f"Place the file locally or provide a correct path."
        )
    return path


def load_yolo11_model(model_name: str = FAST_MODEL_NAME) -> YOLO:
    print(f"[YOLO11] Loading model: {model_name}")

    model = YOLO(model_name)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print(f"[YOLO11] model.names: {names}")
    if not any(str(label).lower() == "pen" for label in names.values()):
        print("[YOLO11] Warning: no 'pen' class found in model.names")
    return model


def resolve_yolo11_pen_model_path(model_path: str | Path | None = None) -> str:
    if model_path is not None:
        return str(model_path)

    candidates: list[Path] = []
    if YOLO11_MODEL_PATH_FILE.is_file():
        saved_path = YOLO11_MODEL_PATH_FILE.read_text(encoding="utf-8").strip()
        if saved_path:
            candidates.append(Path(saved_path))

    candidates.extend(
        [
            YOLO11_RUNS_DIR / YOLO11_DEFAULT_TRAIN_RUN / "weights" / "best.pt",
            Path(FAST_MODEL_NAME),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return FAST_MODEL_NAME


def _collect_detections(results: list[Any], model: YOLO) -> list[dict[str, Any]]:
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    detections: list[dict[str, Any]] = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = str(names.get(cls_id, cls_id))
            bbox = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )
            print(f"[YOLO11] Detected: {label} | Conf: {confidence:.4f}")

    if not detections:
        print("[YOLO11] Warning: no detections found")
    else:
        print(f"[YOLO11] Total detections: {len(detections)}")
        print(f"[YOLO11] Confidence scores: {[round(item['confidence'], 4) for item in detections]}")
    return detections


def run_image_inference(
    image_path: str | Path,
    model_name: str = FAST_MODEL_NAME,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    show: bool = True,
    save_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    image_file = _ensure_existing_file(image_path, "Image")
    model = load_yolo11_model(model_name)

    print(f"[YOLO11] Running inference on: {image_file}")
    results = model.predict(source=str(image_file), conf=conf, iou=iou, verbose=False)
    detections = _collect_detections(results, model)

    plotted = results[0].plot()
    target_path = Path(save_path) if save_path is not None else image_file.with_name(f"{image_file.stem}_yolo11_result.jpg")
    cv2.imwrite(str(target_path), plotted)
    print(f"[YOLO11] Saved plotted result to: {target_path}")

    if show:
        cv2.imshow("YOLOv11 Detection", plotted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


def train_yolo11_model(
    data_yaml_path: str | Path,
    model_name: str = FAST_MODEL_NAME,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    run_name: str = "pen_retrain",
) -> Path:
    dataset_file = _ensure_existing_file(data_yaml_path, "Dataset YAML")
    model = load_yolo11_model(model_name)

    print(f"[YOLO11] Training with dataset: {dataset_file}")
    results = model.train(
        data=str(dataset_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Training completed but best.pt was not found at: {best_path}")

    YOLO11_MODEL_PATH_FILE.write_text(str(best_path.resolve()), encoding="utf-8")
    print(f"[YOLO11] Training complete. best.pt: {best_path}")
    return best_path


def validate_yolo11_model(
    model_path: str | Path = FAST_MODEL_NAME,
    data_yaml_path: str | Path | None = None,
) -> dict[str, float]:
    if model_path not in {FAST_MODEL_NAME, ACCURATE_MODEL_NAME}:
        _ensure_existing_file(model_path, "Model")
    if data_yaml_path is not None:
        _ensure_existing_file(data_yaml_path, "Dataset YAML")

    model = load_yolo11_model(str(model_path))
    kwargs = {"verbose": False}
    if data_yaml_path is not None:
        kwargs["data"] = str(data_yaml_path)
    metrics = model.val(**kwargs)

    map50 = float(getattr(metrics.box, "map50", 0.0))
    map5095 = float(getattr(metrics.box, "map", 0.0))
    print(f"[YOLO11] mAP50: {map50:.4f}")
    print(f"[YOLO11] mAP50-95: {map5095:.4f}")
    return {"map50": map50, "map50_95": map5095}


def detect_pen(
    image_path: str | Path,
    model_path: str | Path | None = None,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
) -> dict[str, Any] | None:
    image_file = _ensure_existing_file(image_path, "Image")
    resolved_model_path = resolve_yolo11_pen_model_path(model_path)
    if resolved_model_path not in {FAST_MODEL_NAME, ACCURATE_MODEL_NAME}:
        _ensure_existing_file(resolved_model_path, "Model")

    model = load_yolo11_model(str(resolved_model_path))
    results = model.predict(source=str(image_file), conf=conf, iou=iou, verbose=False)
    detections = _collect_detections(results, model)

    pen_detections = [item for item in detections if item["label"].lower() == "pen"]
    if not pen_detections:
        print("[YOLO11] Warning: no 'pen' detections found")
        return None

    best = max(pen_detections, key=lambda item: item["confidence"])
    return {
        "label": "pen",
        "confidence": float(best["confidence"]),
        "bbox": [float(value) for value in best["bbox"]],
    }


__all__ = [
    "ACCURATE_MODEL_NAME",
    "FAST_MODEL_NAME",
    "DEFAULT_CONF",
    "DEFAULT_IOU",
    "YOLO11_DATASET_YAML",
    "YOLO11_DEFAULT_TRAIN_RUN",
    "detect_pen",
    "load_yolo11_model",
    "resolve_yolo11_pen_model_path",
    "run_image_inference",
    "train_yolo11_model",
    "validate_yolo11_model",
    "verify_ultralytics_installation",
]
