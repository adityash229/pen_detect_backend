from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset" / "pen_yolo"


def _normalise_boxes(raw_boxes: str) -> list[dict]:
    try:
        boxes = json.loads(raw_boxes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="boxes must be valid JSON") from exc
    if not isinstance(boxes, list) or not boxes:
        raise HTTPException(status_code=400, detail="Draw at least one box first")

    normalised = []
    for box in boxes:
        rect_mask = box.get("rectMask") if isinstance(box, dict) else None
        source = rect_mask if isinstance(rect_mask, dict) else box

        try:
            x_min = float(source.get("xMin", source.get("x")))
            y_min = float(source.get("yMin", source.get("y")))
            width = float(source.get("width", source.get("w")))
            height = float(source.get("height", source.get("h")))
        except (AttributeError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="boxes must contain rectMask xMin, yMin, width, height",
            ) from exc

        if width <= 0 or height <= 0:
            continue
        normalised.append(
            {
                "contentType": "rect",
                "labels": {"labelName": "Pen"},
                "rectMask": {
                    "xMin": x_min,
                    "yMin": y_min,
                    "width": width,
                    "height": height,
                },
            }
        )

    if not normalised:
        raise HTTPException(status_code=400, detail="Draw at least one box first")
    return normalised


@router.post("/label")
async def save_labels(
    image: UploadFile = File(...),
    boxes: str = Form(...),
):
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="image and boxes required")

    normalised_boxes = _normalise_boxes(boxes)
    original_name = Path(image.filename or "upload.jpg")
    image_suffix = original_name.suffix.lower() or ".jpg"
    stem = f"{original_name.stem}_{uuid4().hex}"
    image_path = DATASET_DIR / "images" / "train" / f"{stem}{image_suffix}"
    json_path = DATASET_DIR / "labels" / "train" / f"{stem}.json"

    image_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    image_path.write_bytes(contents)
    json_path.write_text(json.dumps(normalised_boxes, indent=2), encoding="utf-8")

    return {"saved": True}
