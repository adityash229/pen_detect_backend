from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np


def _to_rgb_array(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return image[:, :, ::-1]
    if image.shape[2] == 4:
        return image[:, :, [2, 1, 0, 3]]
    return image


def _to_bgr_array(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return image[:, :, ::-1]
    if image.shape[2] == 4:
        return image[:, :, [2, 1, 0, 3]]
    return image


def ensure_cv2_compat() -> None:
    import cv2

    # Common constants used by Ultralytics and local scripts.
    constants = {
        "IMREAD_COLOR": 1,
        "IMREAD_GRAYSCALE": 0,
        "COLOR_BGR2RGB": 4,
        "COLOR_RGB2BGR": 4,
        "COLOR_BGR2GRAY": 6,
        "COLOR_BGR2HSV": 40,
        "COLOR_RGB2GRAY": 7,
        "COLOR_BGRA2BGR": 8,
        "COLOR_BGR2BGRA": 9,
        "COLOR_RGBA2BGR": 10,
        "COLOR_GRAY2BGR": 11,
        "INTER_LINEAR": 1,
        "INTER_AREA": 3,
        "LINE_AA": 16,
        "FONT_HERSHEY_SIMPLEX": 0,
        "BORDER_CONSTANT": 0,
    }
    for name, value in constants.items():
        if not hasattr(cv2, name):
            setattr(cv2, name, value)

    if not hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads = lambda *args, **kwargs: None  # type: ignore[attr-defined]

    def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
        path = Path(filename)
        if not path.is_file():
            return None
        from PIL import Image

        try:
            image = Image.open(path)
            if flags == cv2.IMREAD_GRAYSCALE:
                image = image.convert("L")
                return np.array(image)
            image = image.convert("RGB")
            return _to_bgr_array(np.array(image))
        except Exception:
            return None

    def imdecode(buffer, flags: int = cv2.IMREAD_COLOR):
        from PIL import Image

        try:
            raw = np.asarray(buffer, dtype=np.uint8).tobytes()
            image = Image.open(BytesIO(raw))
            if flags == cv2.IMREAD_GRAYSCALE:
                image = image.convert("L")
                return np.array(image)
            image = image.convert("RGB")
            return _to_bgr_array(np.array(image))
        except Exception:
            return None

    def imencode(ext: str, img: np.ndarray, params=None):
        from PIL import Image

        try:
            image = Image.fromarray(_to_rgb_array(img))
            output = BytesIO()
            fmt = ext.lstrip(".").upper() or "PNG"
            if fmt == "JPG":
                fmt = "JPEG"
            image.save(output, format=fmt)
            return True, np.frombuffer(output.getvalue(), dtype=np.uint8)
        except Exception:
            return False, np.array([], dtype=np.uint8)

    def imwrite(filename: str, img: np.ndarray, params=None):
        success, encoded = imencode(Path(filename).suffix or ".png", img, params)
        if not success:
            return False
        try:
            Path(filename).write_bytes(encoded.tobytes())
            return True
        except Exception:
            return False

    def resize(img: np.ndarray, dsize, interpolation=None):
        from PIL import Image

        width, height = dsize
        pil = Image.fromarray(_to_rgb_array(img))
        pil = pil.resize((width, height), resample=Image.BILINEAR)
        return _to_bgr_array(np.array(pil))

    def cvtColor(img: np.ndarray, code: int):
        if code in (cv2.COLOR_BGR2RGB, cv2.COLOR_RGB2BGR):
            return img[:, :, ::-1]
        if code == cv2.COLOR_BGR2GRAY:
            return np.dot(img[:, :, :3], [0.114, 0.587, 0.299]).astype(np.uint8)
        if code == cv2.COLOR_BGR2HSV:
            bgr = img[:, :, :3].astype(np.float32) / 255.0
            b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
            maxc = np.max(bgr, axis=2)
            minc = np.min(bgr, axis=2)
            delta = maxc - minc

            hue = np.zeros_like(maxc)
            nonzero = delta > 1e-6
            red_max = (maxc == r) & nonzero
            green_max = (maxc == g) & nonzero
            blue_max = (maxc == b) & nonzero
            hue[red_max] = ((g[red_max] - b[red_max]) / delta[red_max]) % 6
            hue[green_max] = ((b[green_max] - r[green_max]) / delta[green_max]) + 2
            hue[blue_max] = ((r[blue_max] - g[blue_max]) / delta[blue_max]) + 4
            hue = (hue * 30).astype(np.uint8)

            saturation = np.zeros_like(maxc)
            saturation[maxc > 1e-6] = delta[maxc > 1e-6] / maxc[maxc > 1e-6]
            saturation = np.clip(saturation * 255, 0, 255).astype(np.uint8)
            value = np.clip(maxc * 255, 0, 255).astype(np.uint8)
            return np.stack([hue, saturation, value], axis=-1)
        if code == cv2.COLOR_RGB2GRAY:
            return np.dot(img[:, :, :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        if code == cv2.COLOR_GRAY2BGR:
            return np.stack([img, img, img], axis=-1)
        if code == cv2.COLOR_BGRA2BGR:
            return img[:, :, :3]
        if code == cv2.COLOR_BGR2BGRA:
            alpha = np.full((*img.shape[:2], 1), 255, dtype=img.dtype)
            return np.concatenate([img, alpha], axis=2)
        if code == cv2.COLOR_RGBA2BGR:
            return img[:, :, [2, 1, 0]]
        raise NotImplementedError(f"Unsupported color conversion code: {code}")

    def copyMakeBorder(img, top, bottom, left, right, borderType, value=0):
        if borderType != cv2.BORDER_CONSTANT:
            raise NotImplementedError(f"Unsupported border type: {borderType}")

        if img.ndim == 2:
            return np.pad(img, ((top, bottom), (left, right)), mode="constant", constant_values=value)

        fill = np.array(value, dtype=img.dtype)
        if fill.ndim == 0:
            fill = np.repeat(fill, img.shape[2])
        elif fill.size < img.shape[2]:
            fill = np.pad(fill, (0, img.shape[2] - fill.size), mode="edge")
        else:
            fill = fill[: img.shape[2]]

        out = np.empty((img.shape[0] + top + bottom, img.shape[1] + left + right, img.shape[2]), dtype=img.dtype)
        out[...] = fill.reshape((1, 1, img.shape[2]))
        out[top : top + img.shape[0], left : left + img.shape[1]] = img
        return out

    def convertScaleAbs(src: np.ndarray, alpha: float = 1.0, beta: float = 0.0):
        scaled = np.asarray(src, dtype=np.float32) * alpha + beta
        return np.clip(np.abs(scaled), 0, 255).astype(np.uint8)

    def filter2D(src: np.ndarray, ddepth: int, kernel: np.ndarray):
        if ddepth != -1:
            raise NotImplementedError("Only ddepth=-1 is supported by the cv2 compatibility filter2D.")

        kernel = np.asarray(kernel, dtype=np.float32)
        if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError("filter2D compatibility fallback requires an odd 2D kernel.")

        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2

        if src.ndim == 2:
            padded = np.pad(src, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge").astype(np.float32)
            output = np.zeros_like(src, dtype=np.float32)
            for y in range(kernel.shape[0]):
                for x in range(kernel.shape[1]):
                    output += padded[y : y + src.shape[0], x : x + src.shape[1]] * kernel[y, x]
            return np.clip(output, 0, 255).astype(src.dtype)

        padded = np.pad(src, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge").astype(np.float32)
        output = np.zeros_like(src, dtype=np.float32)
        for y in range(kernel.shape[0]):
            for x in range(kernel.shape[1]):
                output += padded[y : y + src.shape[0], x : x + src.shape[1], :] * kernel[y, x]
        return np.clip(output, 0, 255).astype(src.dtype)

    def _draw(image: np.ndarray):
        from PIL import Image, ImageDraw

        pil = Image.fromarray(_to_rgb_array(image))
        return pil, ImageDraw.Draw(pil)

    def rectangle(img, pt1, pt2, color, thickness=1, lineType=None):
        from PIL import Image

        pil, draw = _draw(img)
        draw.rectangle([pt1, pt2], outline=tuple(int(c) for c in color[:3]), width=max(1, int(thickness)))
        img[:] = _to_bgr_array(np.array(pil))
        return img

    def line(img, pt1, pt2, color, thickness=1, lineType=None):
        pil, draw = _draw(img)
        draw.line([pt1, pt2], fill=tuple(int(c) for c in color[:3]), width=max(1, int(thickness)))
        img[:] = _to_bgr_array(np.array(pil))
        return img

    def putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=None, bottomLeftOrigin=False):
        from PIL import ImageFont

        pil, draw = _draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text(org, text, fill=tuple(int(c) for c in color[:3]), font=font)
        img[:] = _to_bgr_array(np.array(pil))
        return img

    class VideoCapture:
        def __init__(self, index=0):
            self.index = index
            self._opened = False

        def isOpened(self):
            return self._opened

        def read(self):
            return False, None

        def release(self):
            return None

    class VideoWriter:
        def __init__(self, *args, **kwargs):
            self._opened = False

        def isOpened(self):
            return self._opened

        def write(self, frame):
            return None

        def release(self):
            return None

    if not hasattr(cv2, "imread"):
        cv2.imread = imread  # type: ignore[attr-defined]
    if not hasattr(cv2, "imdecode"):
        cv2.imdecode = imdecode  # type: ignore[attr-defined]
    if not hasattr(cv2, "imencode"):
        cv2.imencode = imencode  # type: ignore[attr-defined]
    if not hasattr(cv2, "imwrite"):
        cv2.imwrite = imwrite  # type: ignore[attr-defined]
    if not hasattr(cv2, "resize"):
        cv2.resize = resize  # type: ignore[attr-defined]
    if not hasattr(cv2, "cvtColor"):
        cv2.cvtColor = cvtColor  # type: ignore[attr-defined]
    if not hasattr(cv2, "copyMakeBorder"):
        cv2.copyMakeBorder = copyMakeBorder  # type: ignore[attr-defined]
    if not hasattr(cv2, "convertScaleAbs"):
        cv2.convertScaleAbs = convertScaleAbs  # type: ignore[attr-defined]
    if not hasattr(cv2, "filter2D"):
        cv2.filter2D = filter2D  # type: ignore[attr-defined]
    if not hasattr(cv2, "rectangle"):
        cv2.rectangle = rectangle  # type: ignore[attr-defined]
    if not hasattr(cv2, "line"):
        cv2.line = line  # type: ignore[attr-defined]
    if not hasattr(cv2, "putText"):
        cv2.putText = putText  # type: ignore[attr-defined]
    if not hasattr(cv2, "imshow"):
        cv2.imshow = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    if not hasattr(cv2, "waitKey"):
        cv2.waitKey = lambda delay=0: -1  # type: ignore[attr-defined]
    if not hasattr(cv2, "destroyAllWindows"):
        cv2.destroyAllWindows = lambda: None  # type: ignore[attr-defined]
    if not hasattr(cv2, "VideoCapture"):
        cv2.VideoCapture = VideoCapture  # type: ignore[attr-defined]
    if not hasattr(cv2, "VideoWriter"):
        cv2.VideoWriter = VideoWriter  # type: ignore[attr-defined]
