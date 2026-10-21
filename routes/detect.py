from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from model import detect_pen

router = APIRouter()


def should_request_labels(result: dict, user_hint: int | None = None) -> bool:
    """Decide whether a detection result needs manual label review."""
    pen_count = int(result.get("pen_count", 0) or 0)
    confidence = float(result.get("confidence", 0.0) or 0.0)

    # No detections means the image should be manually labeled.
    if pen_count == 0:
        return True

    # If the user expected more pens than YOLO found, request correction labels.
    if user_hint is not None and pen_count < user_hint:
        return True

    # Low-confidence small-count results should be reviewed manually.
    if confidence < 0.45 and pen_count <= 2:
        return True

    # A single weak detection is ambiguous enough to ask for confirmation.
    if pen_count == 1 and confidence < 0.60:
        return True

    return False


@router.post("/detect")
async def detect_with_label_flag(
    image: UploadFile = File(...),
    user_hint: Optional[int] = Form(default=None),
):
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="image required")

    result = detect_pen(contents)
    needs_labels = should_request_labels(result, user_hint=user_hint)

    return {
        "pen_count": result.get("pen_count", 0),
        "confidence": result.get("confidence", 0.0),
        "boxes": result.get("boxes", []),
        "needs_labels": needs_labels,
    }
