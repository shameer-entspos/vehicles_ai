"""Vehicle detection + multi-object tracking (YOLO + ByteTrack)."""
from dataclasses import dataclass

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Track:
    id: int
    box: tuple      # x1, y1, x2, y2 (ints)
    cls_name: str
    conf: float

    @property
    def bottom_center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, float(y2))


class VehicleTracker:
    def __init__(self, model_path, conf=0.4, imgsz=640, tracker="bytetrack.yaml", device="cpu"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf, self.imgsz, self.tracker, self.device = conf, imgsz, tracker, device

    def update(self, frame):
        r = self.model.track(
            frame, persist=True, classes=list(COCO_VEHICLES), conf=self.conf,
            imgsz=self.imgsz, tracker=self.tracker, device=self.device, verbose=False,
        )[0]
        tracks = []
        if r.boxes is None or r.boxes.id is None:
            return tracks
        ids = r.boxes.id.int().cpu().tolist()
        xyxy = r.boxes.xyxy.int().cpu().tolist()
        cls = r.boxes.cls.int().cpu().tolist()
        conf = r.boxes.conf.cpu().tolist()
        for i, b, c, cf in zip(ids, xyxy, cls, conf):
            tracks.append(Track(i, tuple(b), COCO_VEHICLES.get(c, "vehicle"), float(cf)))
        return tracks
