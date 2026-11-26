"""Video input: video file, webcam index, or RTSP/HTTP CCTV stream."""
import os
import time

# Must be set before OpenCV opens the stream. TCP is far more reliable than UDP for CCTV.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402


class VideoSource:
    def __init__(self, src, loop=False, reconnect_delay=3.0):
        s = str(src)
        self.src = int(s) if s.isdigit() else s
        self.is_stream = isinstance(self.src, int) or s.lower().startswith(("rtsp://", "http://", "https://"))
        self.loop = loop
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self._open()

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.src}")

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    @property
    def size(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """Returns a frame, or None when a video file ends."""
        ok, frame = self.cap.read()
        if ok:
            return frame
        if self.is_stream:
            # CCTV stream dropped: keep retrying forever
            while True:
                print(f"[source] stream lost, reconnecting in {self.reconnect_delay}s ...")
                time.sleep(self.reconnect_delay)
                try:
                    self._open()
                    ok, frame = self.cap.read()
                    if ok:
                        return frame
                except RuntimeError:
                    continue
        if self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            return frame if ok else None
        return None

    def skip(self, n):
        """Drop n frames cheaply (grab without decoding)."""
        for _ in range(n):
            self.cap.grab()

    def release(self):
        if self.cap is not None:
            self.cap.release()
