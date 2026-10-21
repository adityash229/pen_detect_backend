"""File-based JSON report generator for pen detection uploads."""

import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
THUMBNAILS_DIR = UPLOADS_DIR / "thumbnails"
REPORTS_DIR = BASE_DIR / "reports"
UPLOADS_URL = "http://127.0.0.1:8000/uploads"


UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)


def generate_breach_id() -> str:
    """Return a unique breach ID in BRH-YYYYMMDD-123456 format."""
    try:
        now = datetime.now()
        return f"BRH-{now.strftime('%Y%m%d')}-{random.randint(100000, 999999)}"
    except Exception as exc:
        print(f"[report_generator] generate_breach_id failed: {exc}")
        now = datetime.now()
        return f"BRH-{now.strftime('%Y%m%d')}-000000"


def save_uploaded_image(image_bytes, breach_id) -> tuple[str, str]:
    """Save uploaded bytes and a max-224px thumbnail to disk."""
    filename = f"{breach_id}.jpg"
    thumb_filename = f"{breach_id}_thumb.jpg"
    photo_path = UPLOADS_DIR / filename
    thumbnail_path = THUMBNAILS_DIR / thumb_filename
    photo_url = f"{UPLOADS_URL}/{filename}"
    thumbnail_photo_url = f"{UPLOADS_URL}/thumbnails/{thumb_filename}"

    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

        photo_path.write_bytes(image_bytes)

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("Could not decode uploaded image for thumbnail.")

        height, width = image.shape[:2]
        max_dim = max(height, width)
        scale = 224.0 / float(max_dim) if max_dim > 224 else 1.0
        thumb_width = max(1, int(round(width * scale)))
        thumb_height = max(1, int(round(height * scale)))
        thumbnail = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)

        if not cv2.imwrite(str(thumbnail_path), thumbnail):
            raise ValueError("Could not write thumbnail image.")

        return photo_url, thumbnail_photo_url
    except Exception as exc:
        print(f"[report_generator] save_uploaded_image failed: {exc}")
        return photo_url if photo_path.exists() else "", ""


def generate_report(
    detection_result,
    image_bytes,
    user_email,
    organization,
    department,
    assigned_to,
) -> dict:
    """Create, save, and return one detection report record."""
    breach_id = generate_breach_id()
    now = datetime.now()
    photo_path = ""
    thumbnail_path = ""
    image_save = False

    try:
        photo_path, thumbnail_path = save_uploaded_image(image_bytes, breach_id)
        image_save = bool(photo_path)

        pen_detected = bool(detection_result.get("pen_detected", False))
        confidence = float(detection_result.get("confidence", 0.0))

        if pen_detected and confidence >= 0.70:
            breach_type = "pen_detected"
        elif pen_detected and confidence < 0.70:
            breach_type = "low_confidence"
        else:
            breach_type = "pen_not_detected"

        record = {
            "user_email": user_email,
            "date": now.strftime("%Y-%m-%d"),
            "detected_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "breach_type": breach_type,
            "photo_url": photo_path,
            "thumbnail_photo_url": thumbnail_path,
            "organization": organization,
            "department": department,
            "assigned_to": assigned_to,
            "image_save": image_save,
            "audit_done": False,
            "breach_id": breach_id,
        }

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"{breach_id}.json"
        report_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
    except Exception as exc:
        print(f"[report_generator] generate_report failed: {exc}")
        return {
            "user_email": user_email,
            "date": now.strftime("%Y-%m-%d"),
            "detected_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "breach_type": "pen_not_detected",
            "photo_url": photo_path,
            "thumbnail_photo_url": thumbnail_path,
            "organization": organization,
            "department": department,
            "assigned_to": assigned_to,
            "image_save": image_save,
            "audit_done": False,
            "breach_id": breach_id,
        }


def get_all_reports() -> list[dict]:
    """Load all report JSON files sorted by detected_at descending."""
    try:
        if not REPORTS_DIR.exists():
            return []

        reports = []
        for report_path in REPORTS_DIR.glob("*.json"):
            try:
                reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            except Exception as exc:
                print(f"[report_generator] skipped invalid report {report_path}: {exc}")

        return sorted(reports, key=lambda item: item.get("detected_at", ""), reverse=True)
    except Exception as exc:
        print(f"[report_generator] get_all_reports failed: {exc}")
        return []


def mark_audit_done(breach_id) -> dict:
    """Mark a report as audited and save it back to disk."""
    report_path = REPORTS_DIR / f"{breach_id}.json"

    try:
        if not report_path.is_file():
            raise FileNotFoundError(f"Report not found: {report_path}")

        record = json.loads(report_path.read_text(encoding="utf-8"))
        record["audit_done"] = True
        report_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
    except FileNotFoundError:
        raise
    except Exception as exc:
        print(f"[report_generator] mark_audit_done failed: {exc}")
        return {}


if __name__ == "__main__":
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", dummy_image)
    if not ok:
        raise RuntimeError("Could not encode dummy test image.")

    fake_detection_result = {
        "pen_detected": True,
        "confidence": 0.82,
        "pen_count": 1,
    }

    report = generate_report(
        detection_result=fake_detection_result,
        image_bytes=encoded.tobytes(),
        user_email="john@company.com",
        organization="Acme Corp",
        department="Engineering",
        assigned_to="admin@company.com",
    )
    print(json.dumps(report, indent=2))
