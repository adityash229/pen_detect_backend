"""
Utility functions for image preprocessing and annotation.
"""

import cv2
import base64
import numpy as np

def resize_image(image):
    """Resize image to 640x640 for YOLO inference."""
    return cv2.resize(image, (640, 640))

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=10):
    """Draw dashed rectangle for uncertain detections."""
    x1, y1 = pt1
    x2, y2 = pt2

    for i in range(x1, x2, dash_length * 2):
        cv2.line(img, (i, y1), (i + dash_length, y1), color, thickness)
        cv2.line(img, (i, y2), (i + dash_length, y2), color, thickness)

    for i in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, i), (x1, i + dash_length), color, thickness)
        cv2.line(img, (x2, i), (x2, i + dash_length), color, thickness)

def annotate_image(image, bbox, status):
    """Draw bounding box depending on confidence status."""
    x, y, w, h = bbox
    pt1 = (x, y)
    pt2 = (x + w, y + h)

    if status == "uncertain":
        draw_dashed_rectangle(image, pt1, pt2, (0, 255, 255))
    else:
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

    return image

def encode_base64(image):
    """Convert image to base64 PNG."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')