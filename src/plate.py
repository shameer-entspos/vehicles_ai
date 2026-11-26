"""Plate detector (optional YOLO) + EasyOCR reader."""
from dataclasses import dataclass
from pathlib import Path

import cv2

from .platefmt import fix_by_position, normalize

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@dataclass
class PlateRead:
    text: str
    conf: float
    valid: bool        # matches a configured plate pattern
    crop: object       # numpy image of the plate
    box: tuple         # (x1, y1, x2, y2) inside the vehicle crop


class PlateReader:
    def __init__(self, languages=("en",), gpu=False, plate_model="", min_conf=0.3,
                 patterns=(r"[A-Z]{2,4}\d{2,6}",), strict=False, min_plate_width=40):
        import easyocr
        self.reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
        self.min_conf, self.patterns, self.strict = min_conf, list(patterns), strict
        self.min_plate_width = min_plate_width
        self.plate_det = None
        if plate_model and Path(plate_model).exists():
            from ultralytics import YOLO
            self.plate_det = YOLO(plate_model)
            print(f"[plate] plate detector loaded: {plate_model}")
        else:
            print(f"[plate] WARNING: plate model '{plate_model}' not found -> OCR on lower part of "
                  "vehicle only (low accuracy). See README 'Plate detector model'.")

    def _locate(self, crop):
        h, w = crop.shape[:2]
        if self.plate_det is None:
            y0 = int(h * 0.45)
            return crop[y0:, :], (0, y0, w, h)
        res = self.plate_det.predict(crop, conf=0.3, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, None
        i = res.boxes.conf.argmax().item()
        x1, y1, x2, y2 = res.boxes.xyxy[i].int().tolist()
        pad = 4
        x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
        return crop[y1:y2, x1:x2], (x1, y1, x2, y2)

    @staticmethod
    def _prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scale = max(2.0, 160.0 / max(1, g.shape[1]))      # make plate ~160px+ wide
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cv2.createCLAHE(2.0, (8, 8)).apply(g)

    def read(self, vehicle_crop):
        """Return PlateRead or None."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None
        plate, box = self._locate(vehicle_crop)
        if plate is None or plate.size == 0 or plate.shape[1] < self.min_plate_width:
            return None
        prepped = self._prep(plate)
        out = self.reader.readtext(prepped, allowlist=ALLOW, detail=1)
        if not out:
            return None
        sh = prepped.shape[0]
        if plate.shape[0] / max(1, plate.shape[1]) > 0.45:   # two-row plate: top row first
            out.sort(key=lambda r: (1 if r[0][0][1] > sh * 0.45 else 0, r[0][0][0]))
        else:                                                # single row: left to right
            out.sort(key=lambda r: r[0][0][0])
        raw = normalize("".join(r[1] for r in out))
        total = sum(len(r[1]) for r in out) or 1
        conf = sum(r[2] * len(r[1]) for r in out) / total
        if len(raw) < 4 or conf < self.min_conf:
            return None
        text, valid = fix_by_position(raw, self.patterns)
        if self.strict and not valid:
            return None
        if not any(c.isdigit() for c in text):
            return None
        return PlateRead(text, float(conf), valid, plate, box)
