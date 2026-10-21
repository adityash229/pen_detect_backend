"""
Loads environment variables and configuration settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
RUN_MODEL_PATH = BASE_DIR / "runs" / "detect" / "pen_retrain" / "weights" / "best.pt"
DEFAULT_MODEL_PATH = RUN_MODEL_PATH
    
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))

HIGH_CONF = float(os.getenv("HIGH_CONF", 0.60))
POSSIBLE_LOW = float(os.getenv("POSSIBLE_LOW", 0.55))
UNCERTAIN_LOW = float(os.getenv("UNCERTAIN_LOW", 0.50))
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.40
NUM_CLASSES = 2
CLASS_NAMES = ["pen", "non_pen"]
PEN_CLASS_ID = 0
NON_PEN_CLASS_ID = 1
MAX_DETECTIONS = 30
AGNOSTIC_NMS = False
MIN_ASPECT_RATIO = 1.1
MIN_BOX_AREA_RATIO = 0.0005

# Tiling / sliding window settings
TILE_SIZE = 640
TILE_OVERLAP = 0.2
TILE_NMS_IOU = 0.40
LARGE_IMAGE_THRESHOLD = 1280 * 720

HOST = os.getenv("HOST", "192.168.29.170")
PORT = int(os.getenv("PORT", 8000))
