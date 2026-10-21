from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision.ops as ops


def split_image(image: np.ndarray, tile_size: int = 640, overlap: float = 0.2) -> list[tuple[np.ndarray, int, int]]:
    """
    Split a large image into overlapping tiles.
    Returns a list of (tile, x_offset, y_offset).
    """
    h, w = image.shape[:2]
    step = max(1, int(tile_size * (1 - overlap)))
    tiles: list[tuple[np.ndarray, int, int]] = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = image[y:y + tile_size, x:x + tile_size]
            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                tile = cv2.copyMakeBorder(
                    tile,
                    0,
                    tile_size - th,
                    0,
                    tile_size - tw,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            tiles.append((tile, x, y))

    return tiles


def tile_to_image_coords(
    box: list[float],
    x_offset: int,
    y_offset: int,
    image_shape: tuple[int, int, int] | tuple[int, int] | None = None,
) -> list[float]:
    """Convert tile-local box coordinates to full-image coordinates."""
    x1, y1, x2, y2 = box
    x1 = x1 + x_offset
    y1 = y1 + y_offset
    x2 = x2 + x_offset
    y2 = y2 + y_offset

    if image_shape is not None:
        image_h, image_w = image_shape[:2]
        x1 = max(0.0, min(float(image_w), x1))
        y1 = max(0.0, min(float(image_h), y1))
        x2 = max(0.0, min(float(image_w), x2))
        y2 = max(0.0, min(float(image_h), y2))

    return [x1, y1, x2, y2]


def apply_nms(
    boxes: list[list[float]],
    scores: list[float],
    iou_threshold: float = 0.40,
) -> tuple[list[list[float]], list[float]]:
    """Remove duplicate boxes caused by tile overlap."""
    if not boxes:
        return [], []

    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = ops.nms(boxes_t, scores_t, iou_threshold)
    kept_indices = keep.tolist()
    kept_boxes = [boxes[index] for index in kept_indices]
    kept_scores = [scores[index] for index in kept_indices]
    return kept_boxes, kept_scores
