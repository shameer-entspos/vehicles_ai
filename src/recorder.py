"""Per-vehicle plate voting and record creation (de-duplicated)."""
import time
from datetime import datetime
from pathlib import Path

from .platefmt import similar


class TrackState:
    """Everything we learn about one tracked vehicle."""

    def __init__(self, track_id, vehicle_type, frame_no, video_sec):
        self.id, self.vehicle_type = track_id, vehicle_type
        self.first_frame = self.last_frame = frame_no
        self.first_time = datetime.now()
        self.video_sec = video_sec
        self.reads = []                 # (text, weight, conf)
        self.attempts = 0
        self.best_plate_crop, self.best_plate_conf = None, 0.0
        self.best_vehicle_img, self.best_vehicle_area = None, 0
        self.eligible = False           # was ever big enough to be readable
        self.committed, self.row_id = False, None
        self.direction, self.prev_pt = None, None
        self.plate_box = None           # last plate box in frame coords (for drawing)

    def add_read(self, read):
        w = read.conf * (1.0 if read.valid else 0.4)
        self.reads.append((read.text, w, read.conf))
        if read.conf > self.best_plate_conf:
            self.best_plate_conf, self.best_plate_crop = read.conf, read.crop

    def consensus(self):
        """Group similar reads (1-char OCR noise); return (plate, n_reads, avg_conf, weight)."""
        clusters = []
        for text, w, c in sorted(self.reads, key=lambda r: -r[1]):
            for cl in clusters:
                if similar(cl["text"], text):
                    cl["w"] += w; cl["n"] += 1; cl["c"] += c
                    break
            else:
                clusters.append({"text": text, "w": w, "n": 1, "c": c})
        if not clusters:
            return None
        b = max(clusters, key=lambda c: c["w"])
        return b["text"], b["n"], b["c"] / b["n"], b["w"]

    def is_confirmed(self, confirm_reads, confirm_conf):
        c = self.consensus()
        return c is not None and c[1] >= confirm_reads and c[2] >= confirm_conf


class Recorder:
    def __init__(self, db, cfg):
        self.db = db
        self.camera = cfg["camera_id"]
        self.snap = Path(cfg["snapshot_dir"])
        self.cooldown = cfg["cooldown_sec"]
        self.log_unreadable = cfg["log_unreadable"]
        self.min_reads = cfg["min_reads_accept"]
        self.high_conf = cfg["high_conf_accept"]
        self.min_track_frames = cfg["min_track_frames"]
        self.recent = {}                # plate -> (row_id, monotonic_time, status)
        self.history = []               # text lines for the on-screen panel
        self.count = 0

    @staticmethod
    def _save(img, path):
        import cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return str(path)

    def commit(self, tr):
        """Finalize a vehicle: write (or merge into) a DB record. Returns row id or None."""
        if tr.committed:
            return tr.row_id
        tr.committed = True
        cons = tr.consensus()
        long_enough = (tr.last_frame - tr.first_frame + 1) >= self.min_track_frames

        if cons is None:
            if not (self.log_unreadable and tr.eligible and long_enough):
                return None
            plate, n, conf, status = None, 0, None, "unreadable"
        else:
            plate, n, conf, _ = cons
            status = "confirmed" if (n >= self.min_reads or conf >= self.high_conf) else "low_conf"

        now_mono, now_iso = time.monotonic(), datetime.now().isoformat(timespec="seconds")
        stamp = tr.first_time.strftime("%Y%m%d_%H%M%S")
        plate_img = None

        if plate:
            # same plate (or 1-char OCR variant) seen recently -> same record
            for p, (rid, ts, st) in list(self.recent.items()):
                if now_mono - ts > self.cooldown:
                    del self.recent[p]
                elif similar(p, plate):
                    upgrade = status == "confirmed" and st != "confirmed"
                    if upgrade and tr.best_plate_crop is not None:
                        plate_img = self._save(tr.best_plate_crop, self.snap / f"{stamp}_{tr.id}_plate.jpg")
                    self.db.touch(rid, now_iso, n, plate if upgrade else None, conf if upgrade else None,
                                  status if upgrade else None, plate_img)
                    self.recent[p] = (rid, now_mono, "confirmed" if upgrade else st)
                    tr.row_id = rid
                    return rid
            if tr.best_plate_crop is not None:
                plate_img = self._save(tr.best_plate_crop, self.snap / f"{stamp}_{tr.id}_plate.jpg")

        veh_img = None
        if tr.best_vehicle_img is not None:
            veh_img = self._save(tr.best_vehicle_img, self.snap / f"{stamp}_{tr.id}_vehicle.jpg")

        rid = self.db.insert(
            plate=plate, plate_conf=conf, status=status, vehicle_type=tr.vehicle_type,
            camera_id=self.camera, direction=tr.direction,
            first_seen=tr.first_time.isoformat(timespec="seconds"), last_seen=now_iso, reads=n,
            track_id=tr.id, video_sec=round(tr.video_sec, 1), vehicle_image=veh_img, plate_image=plate_img)
        tr.row_id = rid
        if plate:
            self.recent[plate] = (rid, now_mono, status)
        self.count += 1
        line = f"{plate or 'UNREADABLE'}  {tr.vehicle_type}  [{status}]"
        self.history.append(line)
        print(f"[RECORD #{rid}] {line}  reads={n}" + (f" conf={conf:.2f}" if conf else ""))
        return rid
