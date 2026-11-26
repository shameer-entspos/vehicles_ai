"""CCTV vehicle scanner: detects vehicles, reads number plates, saves one record per vehicle.

  python main.py                                  # config.yaml
  python main.py --source videos/car-detection.mp4
  python main.py --source rtsp://user:pass@ip:554/stream --no-show
"""
import argparse
import time

import cv2
import yaml

from src.counter import LineCounter
from src.db import PlateDB
from src.detector import VehicleTracker
from src.recorder import Recorder, TrackState
from src.source import VideoSource


def crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2], (x1, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="video file, RTSP URL or webcam index (overrides config)")
    ap.add_argument("--no-show", action="store_true", help="headless, no preview window")
    ap.add_argument("--no-ocr", action="store_true", help="vehicles only, skip plate reading")
    ap.add_argument("--save-video", help="write annotated video to this path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    source = args.source or cfg["source"]
    src = VideoSource(source, loop=cfg.get("loop_video", False))
    W, H = src.size
    fps = src.fps
    print(f"[main] source={source}  {W}x{H} @ {fps:.1f}fps")

    counter, A, B = None, None, None
    if cfg.get("line"):
        (ax, ay), (bx, by) = cfg["line"]
        A, B = (ax * W, ay * H), (bx * W, by * H)
        counter = LineCounter(A, B, cfg.get("entry_direction", "positive"))

    tracker = VehicleTracker(cfg["vehicle_model"], cfg["vehicle_conf"], cfg["imgsz"],
                             cfg["tracker"], cfg["device"])
    db = PlateDB(cfg["db_path"])
    recorder = Recorder(db, cfg)

    reader = None
    if not args.no_ocr:
        from src.plate import PlateReader
        reader = PlateReader(cfg["ocr_languages"], cfg["ocr_gpu"], cfg.get("plate_model", ""),
                             cfg["plate_min_conf"], cfg["plate_patterns"], cfg["strict_format"],
                             cfg["min_plate_width_px"])

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    step = max(1, cfg["process_every_n_frames"])
    min_h = cfg["min_vehicle_height_px"]
    states = {}
    frame_no = 0
    t0 = time.time()

    try:
        while True:
            frame = src.read()
            if frame is None:
                break
            frame_no += 1
            if not src.is_stream and step > 1:
                src.skip(step - 1)
            video_sec = (time.time() - t0) if src.is_stream else (frame_no * step) / fps
            draw = (not args.no_show) or writer
            vis = frame.copy() if draw else None

            for t in tracker.update(frame):
                tr = states.get(t.id)
                if tr is None:
                    tr = states[t.id] = TrackState(t.id, t.cls_name, frame_no, video_sec)
                tr.last_frame = frame_no
                x1, y1, x2, y2 = t.box
                height, area = y2 - y1, (x2 - x1) * (y2 - y1)
                veh, (ox, oy) = crop(frame, t.box)

                if height >= min_h:
                    tr.eligible = True
                    if area > tr.best_vehicle_area * 1.2:      # keep the biggest/clearest view
                        tr.best_vehicle_area, tr.best_vehicle_img = area, veh.copy()

                # ---- read the number plate ----
                if (reader and not tr.committed and tr.attempts < cfg["ocr_max_attempts"]
                        and height >= min_h and frame_no % cfg["ocr_every_n_frames"] == 0):
                    tr.attempts += 1
                    read = reader.read(veh)
                    if read:
                        tr.add_read(read)
                        bx1, by1, bx2, by2 = read.box
                        tr.plate_box = (ox + bx1, oy + by1, ox + bx2, oy + by2)
                    if tr.is_confirmed(cfg["confirm_reads"], cfg["confirm_conf"]):
                        recorder.commit(tr)                    # plate confirmed -> save now

                # ---- optional entry/exit direction ----
                if counter:
                    ev = counter.check(tr.prev_pt, t.bottom_center)
                    tr.prev_pt = t.bottom_center
                    if ev and tr.direction is None:
                        tr.direction = ev
                        if tr.row_id:
                            db.update_direction(tr.row_id, ev)

                if vis is not None:
                    cons = tr.consensus()
                    plate = cons[0] if cons else None
                    color = (0, 200, 0) if tr.committed else (255, 200, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"#{t.id} {t.cls_name}" + (f"  {plate}" if plate else "")
                    cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if tr.plate_box:
                        px1, py1, px2, py2 = tr.plate_box
                        cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 255), 2)

            # vehicles that left the scene -> finalize their record
            for tid, tr in list(states.items()):
                if frame_no - tr.last_frame > cfg["lost_frames"]:
                    recorder.commit(tr)
                    del states[tid]

            if vis is not None:
                if counter:
                    cv2.line(vis, (int(A[0]), int(A[1])), (int(B[0]), int(B[1])), (0, 0, 255), 2)
                cv2.putText(vis, f"Records: {recorder.count}   {frame_no / max(1e-6, time.time() - t0):.1f} fps",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                for i, line in enumerate(recorder.history[-5:][::-1]):
                    cv2.putText(vis, line, (10, H - 14 - 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if writer:
                    writer.write(vis)
                if not args.no_show:
                    cv2.imshow("Vehicle plate scanner  (q = quit)", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        for tr in states.values():          # save vehicles still in view when we stop
            recorder.commit(tr)
        src.release()
        if writer:
            writer.release()
        db.close()
        cv2.destroyAllWindows()
        print(f"[main] done. {recorder.count} records -> {cfg['db_path']}")


if __name__ == "__main__":
    main()
