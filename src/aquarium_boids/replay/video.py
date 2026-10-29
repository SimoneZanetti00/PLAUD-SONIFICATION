"""Video view for replay (display only): the scene's original video behind the projected people.

Optional dependency: OpenCV (requirements-video.txt). It is imported only here, only for
--replay-view video; without it the app explains how to install it and uses the camera view.

Frame = round(25 x data time): the replay's data time is the video's frame number / 25 for the
ETH/UCY files (frame step 10 at 2.5 rows/s), checked against OpenTraj's video.avi (offset 0 for
zara01 and seq_hotel). Pause, R, loop and --replay-speed follow from the data time.

Decoding runs in a background thread: the main thread asks for (frame, size) and blits the latest
finished frame (OpenCV releases the GIL while decoding and resizing). Forward steps up to
MAX_GRAB frames are decoded sequentially (cheap grabs); anything else (restart, loop wrap, big
jumps) is a seek, which can take ~0.1-0.3 s on zara01: meanwhile the previous frame stays up.
"""
from __future__ import annotations

import threading
from pathlib import Path

MAX_GRAB = 150  # forward jumps up to this many frames are decoded, not seeked (6 s at 25 fps)
INSTALL_HINT = "pip install -r requirements-video.txt (OpenCV)"


def import_cv2():
    """(cv2, "") or (None, why)."""
    try:
        import cv2  # noqa: PLC0415  (optional dependency, imported on demand)
    except ImportError as exc:
        return None, f"the video view needs OpenCV, not installed ({exc}): {INSTALL_HINT}"
    return cv2, ""


class VideoFrames:
    """Decodes frames of one video on request, scaled, dimmed and converted to RGB bytes."""

    def __init__(self, path: str | Path, dim: int = 255, threaded: bool = True):
        cv2, why = import_cv2()
        if cv2 is None:
            raise RuntimeError(why)
        self._cv2 = cv2
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open {self.path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.dim = dim
        self._pos = 0  # index of the next frame the capture will decode
        self._want: tuple[int, tuple[int, int]] | None = None
        self._done: tuple[int, tuple[int, int]] | None = None
        self._latest: tuple[int, tuple[int, int], bytes] | None = None
        self.seeks = 0
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._closed = False
        self._thread = None
        if threaded:
            self._thread = threading.Thread(target=self._run, name="replay-video", daemon=True)
            self._thread.start()

    def frame_for(self, data_time: float) -> int:
        """round(fps x data time), clamped to the video."""
        return max(0, min(self.count - 1, round(self.fps * data_time)))

    def request(self, index: int, size: tuple[int, int]) -> None:
        """Ask for frame `index` scaled to `size`; returns at once (threaded) or after decoding."""
        want = (index, (int(size[0]), int(size[1])))
        if self._thread is None:
            if want != self._done:
                self._latest = (*want, self._decode(*want))
                self._done = want
            return
        with self._wake:
            if want != self._want:
                self._want = want
                self._wake.notify()

    def latest(self) -> tuple[int, tuple[int, int], bytes] | None:
        """(index, size, RGB bytes) of the newest finished frame, or None before the first one."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._wake.notify()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()

    def _run(self) -> None:
        while True:
            with self._wake:
                while not self._closed and (self._want is None or self._want == self._done):
                    self._wake.wait()
                if self._closed:
                    return
                want = self._want
            data = self._decode(*want)
            with self._lock:
                self._latest = (*want, data)
                self._done = want

    def _decode(self, index: int, size: tuple[int, int]) -> bytes:
        cv2, cap = self._cv2, self._cap
        step = index - self._pos
        if 0 <= step <= MAX_GRAB:
            for _ in range(step):
                cap.grab()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.seeks += 1
        ok, img = cap.read()
        self._pos = index + 1
        if not ok:
            img = self._blank()
        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        if self.dim < 255:
            img = cv2.convertScaleAbs(img, alpha=self.dim / 255.0)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes()

    def _blank(self):
        import numpy as np  # noqa: PLC0415  (comes with OpenCV)
        return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
